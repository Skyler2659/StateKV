"""Recoverable Gate R0: headroom analysis and preregistered verdict.

Question (analysis/statekv_recoverable_r0_protocol.md): under unified
recoverable backing-store semantics (all arms share the full-history pool,
budget 256, refresh every cycle), does the state-conditioned physical-risk
teacher retain oracle headroom over the strongest cheap recoverable
baseline (uniform / recency / attention / qk_pool / quest_like)?

Verdict rules are the preregistered G1-G5 from the protocol; they are
hard-coded here and must not be tuned after seeing results:

G1 headroom: teacher mean KL <= 0.70 * B* mean KL (B* = best cheap arm)
G2 stability: teacher paired wins >= 8/10 vs B* AND paired bootstrap 95% CI
   of (B* - teacher) per-sample mean KL excludes 0 above
G3 tail: teacher p95 step KL <= 1.05 * B* p95 step KL
G4 quality: substrate quality-valid (full_cache mean NIAH >= 0.8) AND
   teacher official score >= B* - 1.0 per task bucket AND
   teacher NIAH >= B* NIAH - 0.1
G5 fairness: run flags all_budgets_respected and execution_valid

GO iff G1-G5 all pass.  Otherwise NO_GO with subclass NO_HEADROOM
(ratio >= 1.0) / INSUFFICIENT_HEADROOM / INVALID_SUBSTRATE.

Also emits the decomposition ladder against the Gate 0 pure-eviction runs
(same samples): irreversible -> recoverable -> query-aware -> teacher.

Usage:
  .venv/bin/python analysis/tables/recoverable_r0_headroom.py

Outputs (under analysis/tables/):
  recoverable_r0_main.csv / .md     per-arm aggregates + verdict
  recoverable_r0_paired.csv         paired per-sample diffs vs teacher
  recoverable_r0_ladder.csv / .md   decomposition ladder
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_R0_RUN = (
    ROOT / "results/temporal_cache_discovery/statekv_recoverable_r0_qwen3_8b_v1"
)
DEFAULT_G0_RUN = (
    ROOT / "results/temporal_cache_discovery/statekv_teacher_gate_qwen3_8b_g0_v1"
)
DEFAULT_P35_RUN = (
    ROOT / "results/temporal_cache_discovery/statekv_pure_eviction_qwen3_8b_p35_v1"
)
OUT_DIR = ROOT / "analysis/tables"

# ---- preregistered verdict constants (protocol section 7; do not tune) ----
TEACHER_POLICY = "statekv_exact_mean"
CHEAP_ARMS = ("uniform", "recency", "attention", "qk_pool", "quest_like")
G1_MAX_RATIO = 0.70
G2_MIN_WINS = 8
G3_MAX_P95_RATIO = 1.05
G4_FULLCACHE_MIN_NIAH = 0.8
G4_MAX_OFFICIAL_DROP = 1.0
G4_MAX_NIAH_DROP = 0.1
BOOTSTRAP_SEED = 20260808
BOOTSTRAP_SAMPLES = 20000
MIN_SAMPLES = 10


def _paired_bootstrap_interval(values, seed, samples):
    array = np.asarray(list(values), dtype=np.float64)
    if array.size <= 1:
        value = float(array.mean()) if array.size else float("nan")
        return value, value
    rng = np.random.default_rng(int(seed))
    draws = rng.choice(
        array, size=(int(samples), int(array.size)), replace=True
    ).mean(axis=1)
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return float(lower), float(upper)


def _step_tail(steps: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for policy, group in steps.groupby("policy"):
        values = np.asarray(group["exact_kl"], dtype=np.float64)
        rows.append(
            {
                "policy": str(policy),
                "steps": int(len(values)),
                "mean_step_kl": float(values.mean()),
                "median_step_kl": float(np.median(values)),
                "p95_step_kl": float(np.quantile(values, 0.95)),
                "p99_step_kl": float(np.quantile(values, 0.99)),
                "max_step_kl": float(values.max()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r0-run", type=Path, default=DEFAULT_R0_RUN)
    parser.add_argument("--g0-run", type=Path, default=DEFAULT_G0_RUN)
    parser.add_argument("--p35-run", type=Path, default=DEFAULT_P35_RUN)
    args = parser.parse_args()
    r0_run = args.r0_run.resolve()
    g0_run = args.g0_run.resolve()
    p35_run = args.p35_run.resolve()

    samples = pd.read_csv(r0_run / "sample_results.csv")
    steps = pd.read_parquet(r0_run / "step_rows.parquet")
    cycles = pd.read_parquet(r0_run / "cycle_rows.parquet")
    run_summary = json.loads((r0_run / "summary.json").read_text())

    # ---- per-arm aggregates ----
    agg = (
        samples.groupby("policy")
        .agg(
            samples=("mean_trajectory_exact_kl", "size"),
            mean_trajectory_kl=("mean_trajectory_exact_kl", "mean"),
            median_trajectory_kl=("mean_trajectory_exact_kl", "median"),
            p95_trajectory_kl=(
                "mean_trajectory_exact_kl",
                lambda v: float(v.quantile(0.95)),
            ),
            mean_official_score=("official_score", "mean"),
            mean_govreport_rouge_l=("rouge_l", "mean"),
            mean_niah_retrieval=("needle_retrieval_accuracy", "mean"),
            mean_recovered_fraction=("mean_recovered_fraction", "mean"),
            mean_churn_layer_mean=("mean_churn_layer_mean", "mean"),
            recovery_events=("recovery_events", "mean"),
            mean_candidate_universe=("mean_candidate_universe_size", "mean"),
            pool_scoring_time_s=("pool_scoring_forward_time_total_s", "mean"),
        )
        .reset_index()
    )
    bucket_quality = (
        samples.groupby(["policy", "task_bucket"])["official_score"]
        .mean()
        .reset_index()
        .pivot(index="policy", columns="task_bucket", values="official_score")
        .rename(
            columns={
                "GovReport": "official_govreport",
                "NIAH": "official_niah",
            }
        )
        .reset_index()
    )
    agg = agg.merge(bucket_quality, on="policy", how="left")
    tail = _step_tail(steps)
    agg = agg.merge(
        tail[["policy", "p95_step_kl", "p99_step_kl", "max_step_kl"]],
        on="policy",
        how="left",
    )
    agg.to_csv(OUT_DIR / "recoverable_r0_main.csv", index=False)

    # ---- paired comparisons vs teacher ----
    teacher_by_sample = samples[samples["policy"] == TEACHER_POLICY].set_index(
        "sample_id"
    )["mean_trajectory_exact_kl"]
    paired_rows = []
    for policy in CHEAP_ARMS:
        cheap = samples[samples["policy"] == policy].set_index("sample_id")[
            "mean_trajectory_exact_kl"
        ]
        common = sorted(set(teacher_by_sample.index) & set(cheap.index))
        diffs = np.asarray(
            [cheap[s] - teacher_by_sample[s] for s in common],
            dtype=np.float64,
        )
        ci = _paired_bootstrap_interval(
            diffs, BOOTSTRAP_SEED, BOOTSTRAP_SAMPLES
        )
        paired_rows.append(
            {
                "baseline": str(policy),
                "paired_samples": len(common),
                "mean_baseline_minus_teacher_kl": float(diffs.mean()),
                "ci95_lower": ci[0],
                "ci95_upper": ci[1],
                "teacher_wins": int(np.sum(diffs > 1.0e-12)),
                "ties": int(np.sum(np.abs(diffs) <= 1.0e-12)),
                "teacher_losses": int(np.sum(diffs < -1.0e-12)),
                "min_diff": float(diffs.min()),
                "max_diff": float(diffs.max()),
            }
        )
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(OUT_DIR / "recoverable_r0_paired.csv", index=False)

    # ---- verdict ----
    cheap_agg = agg[agg["policy"].isin(CHEAP_ARMS)]
    best_cheap = cheap_agg.sort_values("mean_trajectory_kl").iloc[0]
    teacher_row = agg[agg["policy"] == TEACHER_POLICY].iloc[0]
    full_cache_row = agg[agg["policy"] == "full_cache"].iloc[0]
    ratio = float(
        teacher_row["mean_trajectory_kl"]
        / max(best_cheap["mean_trajectory_kl"], 1.0e-12)
    )
    best_paired = paired[paired["baseline"] == best_cheap["policy"]].iloc[0]

    g1 = bool(ratio <= G1_MAX_RATIO)
    g2 = bool(
        int(best_paired["teacher_wins"]) >= G2_MIN_WINS
        and float(best_paired["ci95_lower"]) > 0.0
    )
    g3 = bool(
        teacher_row["p95_step_kl"]
        <= G3_MAX_P95_RATIO * best_cheap["p95_step_kl"] + 1.0e-12
    )
    substrate_valid = bool(
        full_cache_row["mean_niah_retrieval"] >= G4_FULLCACHE_MIN_NIAH
    )
    quality_ok = bool(
        teacher_row["official_govreport"]
        >= best_cheap["official_govreport"] - G4_MAX_OFFICIAL_DROP
        and teacher_row["official_niah"]
        >= best_cheap["official_niah"] - G4_MAX_OFFICIAL_DROP
        and teacher_row["mean_niah_retrieval"]
        >= best_cheap["mean_niah_retrieval"] - G4_MAX_NIAH_DROP
    )
    g4 = bool(substrate_valid and quality_ok)
    g5 = bool(
        run_summary.get("all_budgets_respected")
        and run_summary.get("execution_valid")
        and len(samples[samples["policy"] == TEACHER_POLICY]) >= MIN_SAMPLES
    )
    go = bool(g1 and g2 and g3 and g4 and g5)
    if not substrate_valid:
        subclass = "INVALID_SUBSTRATE"
    elif ratio >= 1.0:
        subclass = "NO_HEADROOM"
    else:
        subclass = "INSUFFICIENT_HEADROOM"
    verdict = "GO" if go else "NO_GO"

    # ---- decomposition ladder vs pure-eviction runs (same samples) ----
    # Pure cheap trajectories live in the P35 run; the pure teacher arm lives
    # in the Gate 0 run.  Both are matched to the R0 samples and budget.
    p35_samples = pd.read_csv(p35_run / "sample_results.csv")
    p35_samples = p35_samples[p35_samples["total_budget"] == 256]
    g0_samples = pd.read_csv(g0_run / "sample_results.csv")
    g0_samples = g0_samples[g0_samples["total_budget"] == 256]
    pure = pd.concat([p35_samples, g0_samples], ignore_index=True)
    g0_agg = (
        pure.groupby("policy")["mean_trajectory_exact_kl"]
        .mean()
        .reset_index()
        .rename(columns={"mean_trajectory_exact_kl": "mean_trajectory_kl"})
    )
    ladder_rows = []
    for _, row in g0_agg.iterrows():
        ladder_rows.append(
            {
                "stage": "irreversible (Gate 0 pure eviction)",
                "arm": f"pure_{row['policy']}",
                "mean_trajectory_kl": float(row["mean_trajectory_kl"]),
            }
        )
    for _, row in agg.iterrows():
        stage = (
            "ceiling"
            if row["policy"] == "full_cache"
            else "recoverable teacher"
            if row["policy"] == TEACHER_POLICY
            else "recoverable query-aware"
            if row["policy"] in ("qk_pool", "quest_like")
            else "recoverable simple/control"
        )
        ladder_rows.append(
            {
                "stage": stage,
                "arm": f"rec_{row['policy']}",
                "mean_trajectory_kl": float(row["mean_trajectory_kl"]),
            }
        )
    ladder = pd.DataFrame(ladder_rows)

    g0_best_cheap = (
        g0_agg[g0_agg["policy"] != "teacher_panel"]
        .sort_values("mean_trajectory_kl")
        .iloc[0]
    )
    rec_simple = cheap_agg[cheap_agg["policy"].isin(("uniform", "recency"))]
    rec_query = cheap_agg[cheap_agg["policy"].isin(("qk_pool", "quest_like"))]
    d1_rule = float(
        g0_agg[g0_agg["policy"] == "attention"]["mean_trajectory_kl"].iloc[0]
        - agg[agg["policy"] == "attention"]["mean_trajectory_kl"].iloc[0]
    )
    d1_practical = float(
        g0_best_cheap["mean_trajectory_kl"] - best_cheap["mean_trajectory_kl"]
    )
    d1_teacher = float(
        g0_agg[g0_agg["policy"] == "teacher_panel"]["mean_trajectory_kl"].iloc[0]
        - teacher_row["mean_trajectory_kl"]
    )
    d2 = float(
        rec_simple["mean_trajectory_kl"].min()
        - rec_query["mean_trajectory_kl"].min()
    )
    d3 = float(
        best_cheap["mean_trajectory_kl"] - teacher_row["mean_trajectory_kl"]
    )
    decomposition = pd.DataFrame(
        [
            {
                "component": "D1 recoverability, same rule (attention)",
                "delta_kl": d1_rule,
            },
            {
                "component": (
                    "D1 recoverability, best cheap "
                    f"(pure {g0_best_cheap['policy']} -> rec {best_cheap['policy']})"
                ),
                "delta_kl": d1_practical,
            },
            {
                "component": "D1 recoverability, teacher (pure -> recoverable)",
                "delta_kl": d1_teacher,
            },
            {
                "component": "D2 query-aware retrieval (simple -> qk/quest)",
                "delta_kl": d2,
            },
            {
                "component": "D3 physical-risk scorer (B* -> teacher)",
                "delta_kl": d3,
            },
        ]
    )
    ladder_out = pd.concat([ladder, pd.DataFrame([{}]), decomposition.rename(
        columns={"component": "arm", "delta_kl": "mean_trajectory_kl"}
    ).assign(stage="decomposition")], ignore_index=True)
    ladder_out.to_csv(OUT_DIR / "recoverable_r0_ladder.csv", index=False)

    # ---- summary notes ----
    note = [
        "# StateKV Recoverable Gate R0 — unified recoverable-semantics headroom",
        "",
        f"Run: `{r0_run}`; ladder references: `{p35_run}` (pure cheap) and `{g0_run}` (pure teacher), same samples.",
        f"Protocol: analysis/statekv_recoverable_r0_protocol.md (G1-G5 preregistered).",
        "",
        "## Per-arm aggregates",
        "",
        agg.to_markdown(index=False),
        "",
        "## Paired vs teacher (baseline minus teacher, positive = teacher better)",
        "",
        paired.to_markdown(index=False),
        "",
        "## Verdict",
        "",
        f"B* (strongest cheap recoverable): {best_cheap['policy']} "
        f"mean KL {best_cheap['mean_trajectory_kl']:.4f}; teacher "
        f"{teacher_row['mean_trajectory_kl']:.4f} (ratio {ratio:.3f}).",
        f"G1 headroom ratio <= {G1_MAX_RATIO}: {g1}",
        f"G2 wins {int(best_paired['teacher_wins'])}/10 >= {G2_MIN_WINS} and "
        f"CI95 [{best_paired['ci95_lower']:.4f}, {best_paired['ci95_upper']:.4f}] "
        f"excludes 0: {g2}",
        f"G3 p95 teacher {teacher_row['p95_step_kl']:.4f} <= {G3_MAX_P95_RATIO}x "
        f"B* {best_cheap['p95_step_kl']:.4f}: {g3}",
        f"G4 quality-valid (full_cache NIAH "
        f"{full_cache_row['mean_niah_retrieval']:.2f} >= "
        f"{G4_FULLCACHE_MIN_NIAH}) and quality non-worse: {g4}",
        f"G5 fairness flags: {g5}",
        "",
        f"**R0 verdict (preregistered): {verdict}**"
        + ("" if go else f" ({subclass})"),
        "",
        "## Decomposition",
        "",
        decomposition.to_markdown(index=False),
        "",
    ]
    (OUT_DIR / "recoverable_r0_main.md").write_text(
        "\n".join(note), encoding="utf-8"
    )
    ladder_note = [
        "# Recoverable Gate R0 — decomposition ladder (same 10 samples)",
        "",
        ladder.to_markdown(index=False),
        "",
        decomposition.to_markdown(index=False),
        "",
    ]
    (OUT_DIR / "recoverable_r0_ladder.md").write_text(
        "\n".join(ladder_note), encoding="utf-8"
    )
    print("\n".join(note))
    print(f"\nVERDICT R0: {verdict}" + ("" if go else f" ({subclass})"))


if __name__ == "__main__":
    main()
