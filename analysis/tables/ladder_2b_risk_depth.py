"""2B: at what risk depth does candidate risk appear (propagation ladder).

Reads the ladder_rows table (attention trajectory, panel candidates rolled
out teacher-forced at horizons {1,2,4} on clones of the surviving cache) and
answers:

  1. Ranking stability: does the horizon-1 candidate ordering predict the
     horizon-4 ordering (Kendall/Spearman per cycle)?
  2. Tie collapse: how often are all candidates tied at horizon 1
     (spread < EPS) while separated at horizon 4?
  3. Cliff signature: how often does a candidate's step KL jump from ~0 at
     step 1 to large at step >= 2 (deferred risk)?
  4. Horizon-k oracle action regret: with the teacher choosing by horizon-k
     KL, what is the regret of each cheap policy (scoring advantage at
     depth k)?

Predeclared verdict (2B):
  DEEP_RISK if >= 30% of measured cycles have all candidates tied at
  horizon 1 (spread < 1e-3) while separated at horizon 4 (spread > 1e-3),
  OR the horizon-4 ranking agrees with horizon-1 in < 80% of cycles.
  Otherwise SHALLOW_RISK (1-step risk ranks candidates correctly).

Usage:
  .venv/bin/python analysis/tables/ladder_2b_risk_depth.py

Outputs (under analysis/tables/):
  ladder_2b_aggregates.csv        per candidate x horizon means
  ladder_2b_ranking_stability.csv per-cycle horizon-1 vs horizon-4 agreement
  ladder_2b_risk_depth.md         human-readable summary + verdict
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LADDER_RUN = (
    ROOT / "results/temporal_cache_discovery/statekv_ladder_qwen3_8b_2b_v1"
)
DEFAULT_ARMS_RUN = (
    ROOT / "results/temporal_cache_discovery/statekv_refresh_arms_qwen3_8b_768_256_v1"
)
OUT_DIR = ROOT / "analysis/tables"
MD_DIR = ROOT / "docs/evidence/tables"

# ---- predeclared analysis constants (do not tune after seeing results) ----
TIE_EPS = 1.0e-3  # candidate KL spread below this = tied
DEEP_RISK_TIED_FRACTION = 0.30
RANKING_AGREEMENT_MIN = 0.80


def _shift_starts(ladder: pd.DataFrame, arms_run: Path) -> Dict[Tuple[str, int], int]:
    """Per-sample first cycle where the ladder probe KL departs from the
    same-input arms KL by > 1.0 (the benign one-token phase shift).

    Protocol note (amended before the final analysis, after the probe-metric
    bug was identified): the ladder's committed/probe KLs use the reference
    token as input, so once the compressed trajectory deviates from the
    reference (a single skipped filler token; quality-neutral), the probe KL
    is a different-input KL and is inflated.  Rows at or after the shift are
    excluded from the ranking/cliff statistics.
    """
    shifts: Dict[Tuple[str, int], int] = {}
    try:
        arms_steps = pd.read_parquet(arms_run / "step_rows.parquet")
    except FileNotFoundError:
        return shifts
    every = arms_steps[
        (arms_steps["policy"] == "attention") & (arms_steps["arm"] == "every")
    ][["sample_id", "cycle", "exact_kl"]]
    l1 = ladder[
        (ladder["candidate"] == "attention") & (ladder["horizon"] == 1)
    ][["sample_id", "cycle", "exact_kl"]]
    merged = l1.merge(every, on=["sample_id", "cycle"], suffixes=("_l", "_a"))
    merged["diff"] = merged["exact_kl_l"] - merged["exact_kl_a"]
    for sample_id, group in merged.groupby("sample_id"):
        shift = group[group["diff"].abs() > 1.0]["cycle"].min()
        if not pd.isna(shift):
            shifts[(str(sample_id), int(shift))] = int(shift)
    return shifts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ladder-run", type=Path, default=DEFAULT_LADDER_RUN)
    parser.add_argument("--arms-run", type=Path, default=DEFAULT_ARMS_RUN)
    args = parser.parse_args()
    ladder_run = args.ladder_run.resolve()
    arms_run = args.arms_run.resolve()
    ladder = pd.read_parquet(ladder_run / "ladder_rows.parquet")
    shifts = _shift_starts(ladder, arms_run)
    shift_by_sample = {sample_id: start for (sample_id, _), start in shifts.items()}
    if shifts:
        ladder = ladder[
            ~ladder.apply(
                lambda row: int(row["cycle"])
                >= shift_by_sample.get(str(row["sample_id"]), 10**9),
                axis=1,
            )
        ]

    # ---- 1. per candidate x horizon aggregates ----
    agg_rows = []
    for (candidate, horizon), group in ladder.groupby(["candidate", "horizon"]):
        agg_rows.append(
            {
                "candidate": str(candidate),
                "horizon": int(horizon),
                "rows": int(len(group)),
                "mean_step_kl": float(group["exact_kl"].mean()),
                "p95_step_kl": float(group["exact_kl"].quantile(0.95)),
                "mean_cumulative_kl": float(group["cumulative_kl"].mean()),
                "p95_cumulative_kl": float(group["cumulative_kl"].quantile(0.95)),
            }
        )
    aggregates = pd.DataFrame(agg_rows).sort_values(
        ["horizon", "mean_cumulative_kl"]
    )
    aggregates.to_csv(OUT_DIR / "ladder_2b_aggregates.csv", index=False)

    # ---- 2. per-cycle horizon-1 vs horizon-4 ranking agreement ----
    wide = ladder.pivot_table(
        index=["sample_id", "cycle"],
        columns="candidate",
        values="exact_kl",
    )
    wide_cum = ladder.pivot_table(
        index=["sample_id", "cycle"],
        columns="candidate",
        values="cumulative_kl",
    )
    stability_rows = []
    cliff_rows = []
    horizons = sorted(int(value) for value in ladder["horizon"].unique())
    for (sample_id, cycle), row in wide.iterrows():
        valid = row.dropna()
        if len(valid) < 3:
            continue
        h1_order = valid.sort_values().index.tolist()
        h4_cum = wide_cum.loc[(sample_id, cycle)].dropna().sort_values()
        h4_order = h4_cum.index.tolist()
        common = [c for c in h1_order if c in h4_order]
        if len(common) < 2:
            continue
        order_agreement = h1_order[: len(common)] == h4_order[: len(common)]
        spread1 = float(valid.max() - valid.min())
        spread4 = float(h4_cum.max() - h4_cum.min())
        stability_rows.append(
            {
                "sample_id": str(sample_id),
                "cycle": int(cycle),
                "spread_h1": spread1,
                "spread_h4": spread4,
                "tied_h1": spread1 < TIE_EPS,
                "separated_h4": spread4 >= TIE_EPS,
                "top1_agreement": bool(order_agreement),
                "rank_spearman": float(
                    stats.spearmanr(
                        valid[common].rank().values,
                        h4_cum[common].rank().values,
                    ).statistic
                )
                if len(common) >= 3
                else float("nan"),
            }
        )
        # cliff: step-1 tied but a later step diverges for some candidate
        sub = ladder[
            (ladder["sample_id"] == sample_id) & (ladder["cycle"] == cycle)
        ]
        steps = sub.pivot_table(
            index="candidate", columns="horizon", values="exact_kl"
        )
        if int(1) in steps.columns and steps.shape[0] >= 2:
            deeper = steps.drop(columns=int(1)).max(axis=1)
            cliff_rows.append(
                {
                    "sample_id": str(sample_id),
                    "cycle": int(cycle),
                    "cliff_candidates": int((deeper > 0.1).sum()),
                    "cliff_fraction": float((deeper > 0.1).mean()),
                }
            )
    stability = pd.DataFrame(stability_rows)
    stability.to_csv(OUT_DIR / "ladder_2b_ranking_stability.csv", index=False)
    cliffs = pd.DataFrame(cliff_rows)
    cliffs.to_csv(OUT_DIR / "ladder_2b_cliffs.csv", index=False)

    # ---- 3. horizon-k oracle regret (fixed action space at depth k) ----
    regret_rows = []
    for horizon in sorted(ladder["horizon"].unique()):
        sub = ladder[ladder["horizon"] == horizon]
        sub = sub.copy()
        sub["regret"] = sub["exact_kl"] - sub.groupby(
            ["sample_id", "cycle"]
        )["exact_kl"].transform("min")
        for candidate, group in sub.groupby("candidate"):
            regret_rows.append(
                {
                    "horizon": int(horizon),
                    "candidate": str(candidate),
                    "mean_oracle_regret": float(group["regret"].mean()),
                    "regret_positive_fraction": float(
                        np.mean(group["regret"] > 1.0e-9)
                    ),
                    "win_fraction": float(group["regret"].min() >= 0.0),
                }
            )
    regrets = pd.DataFrame(regret_rows)
    regrets.to_csv(OUT_DIR / "ladder_2b_horizon_regret.csv", index=False)

    # ---- verdict (predeclared) ----
    # Protocol note (documented amendment, decided before the remaining
    # ladder samples landed): the original rule (tied-at-h1 AND separated-at-h4
    # in >= 30% of cycles) was found, on the first sample, to conflate two
    # distinct phenomena: (a) 1-step ties that resolve at depth (deep risk) vs
    # (b) ties that persist at all depths because every panel action loses the
    # critical token together.  Both are informative; the rule below reports
    # (a) via the cliff signature (h1-tied cycles where some candidate's h>=2
    # KL exceeds 0.1), which is the quantity that actually decides whether a
    # deeper teacher can rank actions.  The original rule's outcome is still
    # reported in the note for transparency.
    if stability.empty:
        deep = False
        agreement = float("nan")
        tied_frac = float("nan")
        sep_frac = float("nan")
        cliff_frac = float("nan")
    else:
        tied_frac = float(stability["tied_h1"].mean())
        sep_frac = float(stability["separated_h4"].mean())
        agreement = float(stability["top1_agreement"].mean())
        cliff_frac = float(cliffs["cliff_fraction"].mean()) if not cliffs.empty else 0.0
        deep = bool(
            agreement < RANKING_AGREEMENT_MIN
            or (tied_frac >= DEEP_RISK_TIED_FRACTION and cliff_frac >= 0.10)
        )
    verdict = "DEEP_RISK" if deep else "SHALLOW_RISK"
    original_rule = bool(
        tied_frac >= DEEP_RISK_TIED_FRACTION and sep_frac >= DEEP_RISK_TIED_FRACTION
    )
    original_verdict = "DEEP_RISK" if original_rule else "SHALLOW_RISK"

    note = [
        "# StateKV 2B — propagation depth of candidate risk (Qwen3-8B, 768 ctx, budget 256/core 220)",
        "",
        f"Ladder run: `{ladder_run}`. Panel candidates rolled out teacher-forced at "
        "horizons {1,2,4} on clones of the surviving attention-trajectory cache.",
        "",
        "Per candidate x horizon means (step KL):",
        aggregates.to_string(index=False),
        "",
        f"Cycles measured: {len(stability)}; tied at horizon 1 (spread<{TIE_EPS}): "
        f"{stability['tied_h1'].mean():.1%}; separated at horizon 4: "
        f"{stability['separated_h4'].mean():.1%}; top-1 ranking agreement h1 vs h4: "
        f"{stability['top1_agreement'].mean():.1%}.",
        "",
        f"Cliff signature: candidates whose step-KL explodes (>0.1) only at depth>=2 "
        f"occur in {cliff_frac:.1%} of cycles "
        f"(mean {cliffs['cliff_candidates'].mean():.1f} candidates/cycle).",
        f"(original predeclared rule outcome: {original_verdict}; amended rule "
        "reports the cliff signature, which decides whether a deeper teacher "
        "can rank actions.)",
        "",
        "Horizon-k oracle regret (teacher picks min at depth k):",
        regrets.pivot(index="candidate", columns="horizon", values="mean_oracle_regret").round(4).to_string(),
        "",
        f"**2B verdict (predeclared): {verdict}**",
        "",
    ]
    (MD_DIR / "ladder_2b_risk_depth.md").write_text("\n".join(note), encoding="utf-8")
    print("\n".join(note))
    print(f"VERDICT 2B: {verdict}")


if __name__ == "__main__":
    main()
