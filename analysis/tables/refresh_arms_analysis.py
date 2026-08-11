"""Refresh-arm comparison on fresh samples 101-105 (768/256, qwen3-8b).

Motivation: the ladder run found that on these samples the fresh per-step
ranking drops the NIAH needle before it is queried (attention decay), causing
catastrophic irreversible divergence on 4/5 NIAH samples for the every-refresh
attention policy.  The cycle-0 prefill-based ranking still contains the needle,
so never-refresh (or rarely-refresh) may survive.  This compares every vs
never vs fixed_k16 under strict pure eviction.

Predeclared verdict (refresh arms):
  NEVER_BETTER if never-refresh mean trajectory KL < every-refresh mean KL by
  >= 30% relative on the NIAH split AND never wins >= 4/5 paired NIAH samples.
  Otherwise NO_CLEAR_REFRESH_ADVANTAGE.

Usage:
  .venv/bin/python analysis/tables/refresh_arms_analysis.py

Outputs (under analysis/tables/):
  refresh_arms_summary.csv        per policy x arm aggregates (task split)
  refresh_arms_paired.csv         per-sample never-minus-every diffs
  refresh_arms_summary.md         human-readable summary + verdict
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = (
    ROOT
    / "results/temporal_cache_discovery/statekv_refresh_arms_qwen3_8b_768_256_v1"
)
OUT_DIR = ROOT / "analysis/tables"
MD_DIR = ROOT / "docs/evidence/tables"

NEVER_MIN_RELATIVE_GAIN = 0.30
NEVER_MIN_NIAH_WINS = 4


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    run = args.run.resolve()
    samples = pd.read_csv(run / "sample_results.csv")
    samples["is_niah"] = samples["sample_id"].str.startswith("synthetic")

    summary_rows = []
    for (policy, arm), group in samples.groupby(["policy", "arm"]):
        niah = group[group["is_niah"]]
        gov = group[~group["is_niah"]]
        summary_rows.append(
            {
                "policy": str(policy),
                "arm": str(arm),
                "samples": int(len(group)),
                "mean_kl_all": float(group["mean_trajectory_exact_kl"].mean()),
                "mean_kl_niah": float(niah["mean_trajectory_exact_kl"].mean()),
                "mean_kl_gov": float(gov["mean_trajectory_exact_kl"].mean()),
                "mean_niah_retrieval": float(niah["needle_retrieval_accuracy"].mean()),
                "mean_official_score": float(group["official_score"].mean()),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values(["policy", "arm"])
    summary.to_csv(OUT_DIR / "refresh_arms_summary.csv", index=False)

    paired_rows = []
    for policy in samples["policy"].unique():
        sub = samples[samples["policy"] == policy]
        never = sub[sub["arm"] == "never"].set_index("sample_id")
        every = sub[sub["arm"] == "every"].set_index("sample_id")
        common = sorted(set(never.index) & set(every.index))
        for sample_id in common:
            diff = float(never.loc[sample_id, "mean_trajectory_exact_kl"] - every.loc[sample_id, "mean_trajectory_exact_kl"])
            paired_rows.append(
                {
                    "policy": str(policy),
                    "sample_id": str(sample_id),
                    "is_niah": str(sample_id).startswith("synthetic"),
                    "never_minus_every_kl": diff,
                    "never_better": diff < 0,
                }
            )
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(OUT_DIR / "refresh_arms_paired.csv", index=False)

    # ---- verdict (predeclared) ----
    niah_paired = paired[paired["is_niah"]]
    verdicts = {}
    for policy in samples["policy"].unique():
        sub = paired[paired["policy"] == policy]
        niah = sub[sub["is_niah"]]
        if niah.empty:
            verdicts[policy] = "NO_NIAH_DATA"
            continue
        never_mean = summary[
            (summary["policy"] == policy) & (summary["arm"] == "never")
        ]["mean_kl_niah"].iloc[0]
        every_mean = summary[
            (summary["policy"] == policy) & (summary["arm"] == "every")
        ]["mean_kl_niah"].iloc[0]
        rel_gain = (every_mean - never_mean) / max(every_mean, 1.0e-12)
        wins = int((niah["never_better"]).sum())
        verdicts[policy] = (
            "NEVER_BETTER"
            if rel_gain >= NEVER_MIN_RELATIVE_GAIN and wins >= NEVER_MIN_NIAH_WINS
            else "NO_CLEAR_REFRESH_ADVANTAGE"
        )

    note = [
        "# StateKV refresh arms — never vs every vs fixed_k16 (samples 101-105, 768/256)",
        "",
        f"Run: `{run}`. Strict pure eviction, matched samples, same budget.",
        "",
        summary.to_string(index=False),
        "",
        "Paired never-minus-every trajectory KL (negative = never better):",
        paired.to_string(index=False),
        "",
        f"**Verdicts (predeclared): " + "; ".join(f"{k}: {v}" for k, v in verdicts.items()) + "**",
        "",
    ]
    (MD_DIR / "refresh_arms_summary.md").write_text("\n".join(note), encoding="utf-8")
    print("\n".join(note))


if __name__ == "__main__":
    main()
