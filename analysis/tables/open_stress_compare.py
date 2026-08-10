"""Paired comparison for open-search stress runs vs the R0 qk_pool arm.

Reads sample_results.csv from each run directory and prints a compact
paired table: mean/median KL, step p95, NIAH, GovReport, paired wins vs
qk_pool within the same run.  Usage:

    .venv/bin/python analysis/tables/open_stress_compare.py RUN_DIR [RUN_DIR...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

R0 = "results/temporal_cache_discovery/statekv_recoverable_r0_qwen3_8b_v1"


def load(run: str) -> pd.DataFrame:
    df = pd.read_csv(Path(run) / "sample_results.csv")
    return df


def main() -> None:
    runs = [R0] + list(sys.argv[1:])
    frames = []
    for run in runs:
        df = load(run)
        df["run"] = Path(run).parent.name + "/" + Path(run).name
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)
    kl_col = "mean_trajectory_exact_kl"
    pivot = all_df.pivot_table(
        index="sample_id", columns=["run", "policy"], values=kl_col
    )
    print("=== per-sample mean trajectory KL ===")
    print(pivot.round(4).to_string())
    summ = (
        all_df.groupby(["run", "policy"])[kl_col]
        .agg(["mean", "median", "count"])
        .round(4)
    )
    print("\n=== aggregate ===")
    print(summ.to_string())
    for col in ("needle_retrieval_accuracy", "official_score"):
        if col in all_df.columns:
            print(f"\n=== {col} (mean) ===")
            print(
                all_df.groupby(["run", "policy"])[col].mean().round(4).to_string()
            )


if __name__ == "__main__":
    main()
