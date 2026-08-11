"""Gate 0/1: strict-pure-eviction teacher headroom and fixed-action-space regret.

Question (Gate 0): on the P35 substrate (Qwen3-8B 4bit, 768 ctx, budget
256/core 220, 10 samples, 64 gen tokens, strict pure eviction), does the
expensive state-conditioned physical teacher still beat the best cheap
selector (attention / b2_uniform / a2 / snapkv / dynamic_b3)?

Teacher arm: every cycle commits the minimum exact-KL action from a fixed
panel of cheap legal actions, each evaluated as a counterfactual clone of
the *surviving* cache (no deleted history read, no persistent full-KV
backing; the reference logits stream is the same Full-KV evaluator the
repo always uses).

Question (Gate 1): with the candidate action set fixed to the panel, how
large is the oracle action regret of each cheap policy under the teacher's
roll-in?  This separates "the teacher scores better" (fixed-action-space
regret) from "the teacher proposes better actions" (action-space).

Predeclared verdicts (hard-coded, not tuned post hoc):

Gate 0 -- HEADROOM if ALL of:
  * teacher mean trajectory KL < best-cheap mean KL by >= 20% relative,
  * teacher wins the paired per-sample comparison vs best cheap in >= 7/10,
  * teacher mean NIAH retrieval >= best-cheap mean NIAH (task quality not worse),
  * teacher p95 step KL <= best-cheap p95 step KL (tail not worse).
  Otherwise NO_HEADROOM (method track stops; closure documented).

Gate 1 -- SCORING_EDGE if ALL of:
  * mean fixed-action-space oracle regret of the best cheap policy
    >= 20% of that policy's own mean step KL (under teacher roll-in),
  * regret > 0 in >= 60% of cycles.
  Otherwise ACTION_SPACE_DOMINANT (teacher's edge comes from proposing
  actions cheap selectors never generate).

Usage:
  .venv/bin/python analysis/tables/gate0_teacher_headroom.py

Outputs (under analysis/tables/):
  gate0_teacher_headroom.csv          per-policy trajectory aggregates
  gate0_paired_comparisons.csv        per-sample teacher-minus-policy diffs
  gate0_step_tail.csv                 per-policy step-level tail stats
  gate1_fixed_action_space.csv        per-candidate panel aggregates
  gate1_action_choice.csv             teacher action-choice distribution
  gate0_teacher_headroom.md           human-readable summary + verdicts
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEACHER_RUN = (
    ROOT
    / "results/temporal_cache_discovery/statekv_teacher_gate_qwen3_8b_g0_v1"
)
DEFAULT_P35_RUN = (
    ROOT
    / "results/temporal_cache_discovery/statekv_pure_eviction_qwen3_8b_p35_v1"
)
OUT_DIR = ROOT / "analysis/tables"
MD_DIR = ROOT / "docs/evidence/tables"

# ---- predeclared analysis constants (do not tune after seeing results) ----
TEACHER_POLICY = "teacher_panel"
BUDGET = 256
HEADROOM_MIN_RELATIVE_KL_GAIN = 0.20
HEADROOM_MIN_PAIRED_WINS = 7
SCORING_MIN_RELATIVE_REGRET = 0.20
SCORING_MIN_POSITIVE_REGRET_FRAC = 0.60
MIN_SAMPLES_FOR_VERDICT = 10


def _step_tail(step_rows: pd.DataFrame) -> pd.DataFrame:
    grouped = step_rows.groupby("policy")["exact_kl"]
    rows = []
    for policy, group in grouped:
        values = np.asarray(group, dtype=np.float64)
        sorted_values = np.sort(values)
        cvar = float(sorted_values[: max(1, int(0.05 * len(values)))].mean())
        rows.append(
            {
                "policy": str(policy),
                "steps": int(len(values)),
                "mean_step_kl": float(values.mean()),
                "p95_step_kl": float(np.quantile(values, 0.95)),
                "p99_step_kl": float(np.quantile(values, 0.99)),
                "cvar95_step_kl": cvar,
                "max_step_kl": float(values.max()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-run", type=Path, default=DEFAULT_TEACHER_RUN)
    parser.add_argument("--p35-run", type=Path, default=DEFAULT_P35_RUN)
    args = parser.parse_args()
    teacher_run = args.teacher_run.resolve()
    p35_run = args.p35_run.resolve()

    teacher_samples = pd.read_csv(teacher_run / "sample_results.csv")
    teacher_steps = pd.read_parquet(teacher_run / "step_rows.parquet")
    teacher_panel = pd.read_parquet(teacher_run / "panel_rows.parquet")
    p35_samples = pd.read_csv(p35_run / "sample_results.csv")
    p35_steps = pd.read_parquet(p35_run / "step_rows.parquet")

    teacher_samples = teacher_samples[teacher_samples["total_budget"] == BUDGET]
    p35_samples = p35_samples[p35_samples["total_budget"] == BUDGET]
    teacher_steps = teacher_steps[teacher_steps["total_budget"] == BUDGET]
    p35_steps = p35_steps[p35_steps["total_budget"] == BUDGET]

    # ---- Gate 0: trajectory comparison (matched samples, same budget) ----
    teacher_agg = (
        teacher_samples.groupby("policy")
        .agg(
            samples=("mean_trajectory_exact_kl", "size"),
            mean_trajectory_kl=("mean_trajectory_exact_kl", "mean"),
            median_trajectory_kl=("mean_trajectory_exact_kl", "median"),
            p95_trajectory_kl=("mean_trajectory_exact_kl", lambda v: v.quantile(0.95)),
            mean_official_score=("official_score", "mean"),
            mean_niah=("needle_retrieval_accuracy", "mean"),
        )
        .reset_index()
    )
    cheap_agg = (
        p35_samples.groupby("policy")
        .agg(
            samples=("mean_trajectory_exact_kl", "size"),
            mean_trajectory_kl=("mean_trajectory_exact_kl", "mean"),
            median_trajectory_kl=("mean_trajectory_exact_kl", "median"),
            p95_trajectory_kl=("mean_trajectory_exact_kl", lambda v: v.quantile(0.95)),
            mean_official_score=("official_score", "mean"),
            mean_niah=("needle_retrieval_accuracy", "mean"),
        )
        .reset_index()
    )
    trajectory = pd.concat([teacher_agg, cheap_agg], ignore_index=True)
    trajectory["source"] = np.where(
        trajectory["policy"] == TEACHER_POLICY, "teacher", "cheap-p35"
    )
    trajectory.to_csv(OUT_DIR / "gate0_teacher_headroom.csv", index=False)

    # Paired per-sample comparison: teacher vs each cheap policy.
    teacher_by_sample = teacher_samples.set_index("sample_id")[
        "mean_trajectory_exact_kl"
    ]
    paired_rows = []
    for policy in sorted(p35_samples["policy"].unique()):
        cheap_by_sample = p35_samples[p35_samples["policy"] == policy].set_index(
            "sample_id"
        )["mean_trajectory_exact_kl"]
        common = sorted(set(teacher_by_sample.index) & set(cheap_by_sample.index))
        diffs = np.asarray(
            [teacher_by_sample[s] - cheap_by_sample[s] for s in common],
            dtype=np.float64,
        )
        paired_rows.append(
            {
                "policy": str(policy),
                "paired_samples": len(common),
                "mean_teacher_minus_cheap_kl": float(diffs.mean()),
                "teacher_wins": int(np.sum(diffs < 0)),
                "ties": int(np.sum(np.abs(diffs) < 1.0e-12)),
                "teacher_losses": int(np.sum(diffs > 0)),
                "teacher_mean_niah": float(
                    teacher_samples["needle_retrieval_accuracy"].mean()
                ),
                "cheap_mean_niah": float(
                    p35_samples[p35_samples["policy"] == policy][
                        "needle_retrieval_accuracy"
                    ].mean()
                ),
            }
        )
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(OUT_DIR / "gate0_paired_comparisons.csv", index=False)

    # Step-level tail statistics (teacher steps vs cheap steps).
    all_steps = pd.concat(
        [
            teacher_steps[["policy", "cycle", "exact_kl"]],
            p35_steps[["policy", "cycle", "exact_kl"]],
        ],
        ignore_index=True,
    )
    step_tail = _step_tail(all_steps)
    step_tail.to_csv(OUT_DIR / "gate0_step_tail.csv", index=False)

    # ---- Gate 0 verdict (predeclared) ----
    best_cheap = trajectory[
        trajectory["policy"] != TEACHER_POLICY
    ].sort_values("mean_trajectory_kl").iloc[0]
    teacher_row = trajectory[trajectory["policy"] == TEACHER_POLICY].iloc[0]
    relative_gain = (
        (best_cheap["mean_trajectory_kl"] - teacher_row["mean_trajectory_kl"])
        / max(best_cheap["mean_trajectory_kl"], 1.0e-12)
    )
    best_cheap_paired = paired[
        paired["policy"] == best_cheap["policy"]
    ].iloc[0]
    teacher_step_tail = step_tail[
        step_tail["policy"] == TEACHER_POLICY
    ].iloc[0]
    best_cheap_step_tail = step_tail[
        step_tail["policy"] == best_cheap["policy"]
    ].iloc[0]
    tail_ok = teacher_step_tail["p95_step_kl"] <= best_cheap_step_tail["p95_step_kl"] + 1.0e-12
    quality_ok = teacher_row["mean_niah"] >= best_cheap["mean_niah"] - 1.0e-12
    headroom = bool(
        len(teacher_samples) >= MIN_SAMPLES_FOR_VERDICT
        and relative_gain >= HEADROOM_MIN_RELATIVE_KL_GAIN
        and best_cheap_paired["teacher_wins"] >= HEADROOM_MIN_PAIRED_WINS
        and tail_ok
        and quality_ok
    )
    headroom_verdict = "HEADROOM" if headroom else "NO_HEADROOM"

    # ---- Gate 1: fixed-action-space regret under teacher roll-in ----
    panel = teacher_panel.copy()
    panel["regret_vs_selected"] = panel["exact_kl"] - panel.groupby(
        ["sample_id", "cycle"]
    )["exact_kl"].transform("min")
    candidate_rows = []
    for candidate, group in panel.groupby("candidate"):
        regrets = np.asarray(group["regret_vs_selected"], dtype=np.float64)
        candidate_rows.append(
            {
                "candidate": str(candidate),
                "rows": int(len(group)),
                "mean_kl": float(group["exact_kl"].mean()),
                "median_kl": float(group["exact_kl"].median()),
                "p95_kl": float(group["exact_kl"].quantile(0.95)),
                "mean_oracle_regret": float(regrets.mean()),
                "regret_positive_fraction": float(np.mean(regrets > 1.0e-9)),
                "mean_risk_rank": float(group["risk_rank"].mean()),
                "selected_fraction": float(group["selected"].mean()),
                "min_kl_fraction": float(
                    np.mean(np.abs(group["exact_kl"] - group["exact_kl"].min()) < 1.0e-9)
                ),
            }
        )
    candidate_table = pd.DataFrame(candidate_rows).sort_values("mean_kl")
    candidate_table.to_csv(OUT_DIR / "gate1_fixed_action_space.csv", index=False)

    # Teacher action-choice distribution by task.
    choice = (
        panel[panel["selected"]]
        .groupby(["sample_id", "candidate"])
        .size()
        .reset_index(name="cycles")
    )
    choice.to_csv(OUT_DIR / "gate1_action_choice.csv", index=False)

    # ---- Gate 1 verdict (predeclared) ----
    best_cheap_panel = candidate_table[
        candidate_table["candidate"].isin(
            ["attention", "b2_uniform", "a2_temporal_volatility", "snapkv", "dynamic_b3"]
        )
    ].sort_values("mean_kl").iloc[0]
    scoring_relative = (
        best_cheap_panel["mean_oracle_regret"]
        / max(best_cheap_panel["mean_kl"], 1.0e-12)
    )
    scoring = bool(
        scoring_relative >= SCORING_MIN_RELATIVE_REGRET
        and best_cheap_panel["regret_positive_fraction"]
        >= SCORING_MIN_POSITIVE_REGRET_FRAC
    )
    gate1_verdict = "SCORING_EDGE" if scoring else "ACTION_SPACE_DOMINANT"

    # ---- summary note ----
    note = [
        "# StateKV Gate 0/1 — strict-pure-eviction teacher headroom (Qwen3-8B, 768 ctx, budget 256/core 220)",
        "",
        f"Teacher arm: `{teacher_run}`; cheap trajectories: `{p35_run}` (matched samples, same budget, same substrate).",
        f"Teacher trajectory KL: {teacher_row['mean_trajectory_kl']:.4f} vs best cheap "
        f"({best_cheap['policy']}): {best_cheap['mean_trajectory_kl']:.4f} "
        f"(relative gain {relative_gain:.1%}).",
        f"Paired per-sample: teacher wins {int(best_cheap_paired['teacher_wins'])}/"
        f"{int(best_cheap_paired['paired_samples'])} vs {best_cheap['policy']}.",
        f"Step tail p95: teacher {teacher_step_tail['p95_step_kl']:.4f} vs "
        f"{best_cheap['policy']} {best_cheap_step_tail['p95_step_kl']:.4f} "
        f"(tail_ok={tail_ok}); NIAH teacher {teacher_row['mean_niah']:.2f} vs "
        f"{best_cheap['policy']} {best_cheap['mean_niah']:.2f} (quality_ok={quality_ok}).",
        "",
        f"**Gate 0 verdict (predeclared): {headroom_verdict}**",
        "",
        "Gate 1 (fixed action space, teacher roll-in):",
        f"best cheap panel candidate: {best_cheap_panel['candidate']} mean KL "
        f"{best_cheap_panel['mean_kl']:.4f}, mean oracle regret "
        f"{best_cheap_panel['mean_oracle_regret']:.4f} "
        f"(relative {scoring_relative:.1%}), regret>0 in "
        f"{best_cheap_panel['regret_positive_fraction']:.1%} of cycles.",
        "",
        f"**Gate 1 verdict (predeclared): {gate1_verdict}**",
        "",
        "Teacher selected-candidate counts: "
        + "; ".join(
            f"{row['candidate']}={int(row['cycles'])}"
            for _, row in choice.groupby("candidate")["cycles"].sum().reset_index().iterrows()
        ),
        "",
    ]
    (MD_DIR / "gate0_teacher_headroom.md").write_text("\n".join(note), encoding="utf-8")
    print("\n".join(note))
    print(f"\nVERDICT Gate0: {headroom_verdict} | Gate1: {gate1_verdict}")


if __name__ == "__main__":
    main()
