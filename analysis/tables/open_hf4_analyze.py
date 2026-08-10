"""Headwise-probe analysis: own-top-k vs shared-core captured attention mass.

Input: results/temporal_cache_discovery/statekv_headwise_probe_qwen3_8b_v1/headwise_rows.parquet
Output: analysis/tables/open_hf4_headwise_{overall,by_layer}.csv + stdout summary
"""
from __future__ import annotations

import pandas as pd

RUN = "results/temporal_cache_discovery/statekv_headwise_probe_qwen3_8b_v1"
OUT = "analysis/tables"


def main() -> None:
    df = pd.read_parquet(f"{RUN}/headwise_rows.parquet")
    df["gain"] = df["own_mass"] - df["shared_mass"]
    overall = df[["shared_mass", "own_mass", "gain", "own_shared_overlap"]].mean()
    p = df["gain"].quantile([0.05, 0.5, 0.95])
    print("=== HF4 headwise captured-mass (mean over sample x cycle x layer x head) ===")
    print(overall.round(4).to_string())
    print("\ngain quantiles (5/50/95):")
    print(p.round(4).to_string())
    by_layer = df.groupby("layer")[["shared_mass", "own_mass", "gain", "own_shared_overlap"]].mean()
    by_layer.to_csv(f"{OUT}/open_hf4_headwise_by_layer.csv")
    df.to_csv(f"{OUT}/open_hf4_headwise_rows.csv", index=False)
    print("\n=== by layer (selected) ===")
    print(by_layer.round(4).to_string())
    by_task = df.groupby("task")[["shared_mass", "own_mass", "gain"]].mean()
    print("\n=== by task ===")
    print(by_task.round(4).to_string())


if __name__ == "__main__":
    main()
