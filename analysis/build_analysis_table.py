#!/usr/bin/env python3
"""Build analysis-ready loss tables from the canonical experiment outputs."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import load_required_tables, parse_json, write_dual


PRIMARY_HORIZON = 64


def build(input_dir: Path, analysis_dir: Path) -> None:
    tables = load_required_tables(input_dir)
    out = Path(analysis_dir) / "tables"

    step = tables["step"].copy()
    step["global_generation_index"] = step["anchor"] + step["future_step"] - 1
    step["analysis_primary"] = step["target_horizon"].eq(PRIMARY_HORIZON)
    step["analysis_condition"] = np.where(
        step["strategy"].eq("future_attention_oracle"),
        "horizon_conditioned_oracle_H64",
        "deployable_fixed_core_longest_replay",
    )
    step["sample_cluster"] = step["task"].astype(str) + "::" + step["sample_id"].astype(str)
    step["delta_nll_abs"] = step["delta_nll"].abs()
    step["attention_output_error_range"] = (
        step["attention_output_error_max"] - step["attention_output_error_mean"]
    )
    write_dual(step, out / "per_step_metrics")

    attention_rows = []
    for row in step.loc[step["analysis_primary"]].itertuples(index=False):
        for item in parse_json(row.attention_output_errors, "attention_output_errors"):
            attention_rows.append(
                {
                    "sample_id": row.sample_id,
                    "sample_cluster": row.sample_cluster,
                    "task": row.task,
                    "anchor": int(row.anchor),
                    "future_step": int(row.future_step),
                    "global_generation_index": int(row.global_generation_index),
                    "strategy": row.strategy,
                    "target_horizon": int(row.target_horizon),
                    "layer": int(item["layer"]),
                    "head": int(item["head"]),
                    "attention_output_relative_error": float(item["relative_error"]),
                    "delta_nll": float(row.delta_nll),
                    "approx_kl": float(row.approx_kl),
                }
            )
    attention = pd.DataFrame(attention_rows)
    write_dual(attention, out / "attention_output_by_head")

    horizon = tables["horizon"].copy()
    horizon["sample_cluster"] = (
        horizon["task"].astype(str) + "::" + horizon["sample_id"].astype(str)
    )
    horizon["oracle_conditioned"] = horizon["strategy"].eq("future_attention_oracle")
    horizon["avg_delta_nll_abs"] = horizon["avg_delta_nll"].abs()
    horizon["rank_avg_delta_nll_within_sample_anchor_horizon"] = horizon.groupby(
        ["sample_id", "anchor", "horizon"], sort=False
    )["avg_delta_nll"].rank(method="average", ascending=True)
    horizon["rank_avg_kl_within_sample_anchor_horizon"] = horizon.groupby(
        ["sample_id", "anchor", "horizon"], sort=False
    )["avg_approx_kl"].rank(method="average", ascending=True)
    write_dual(horizon, out / "per_horizon_metrics")

    # One row per selector/horizon/task plus overall, preserving the sample as
    # the uncertainty unit rather than treating replay tokens as independent.
    summaries = []
    rng = np.random.default_rng(20260724)
    group_specs = [(["strategy", "horizon", "task"], "task"), (["strategy", "horizon"], "overall")]
    metrics = [
        "avg_delta_nll",
        "max_delta_nll",
        "avg_approx_kl",
        "attention_output_error_mean",
        "oracle_overlap",
    ]
    for keys, scope in group_specs:
        for group_key, group in horizon.groupby(keys, dropna=False, sort=True):
            if not isinstance(group_key, tuple):
                group_key = (group_key,)
            base = dict(zip(keys, group_key))
            for metric in metrics:
                values = group[metric].dropna().to_numpy(float)
                if not len(values):
                    continue
                clusters = list(group.loc[group[metric].notna(), "sample_cluster"].unique())
                bootstrap = []
                by_cluster = {
                    key: sub[metric].dropna().to_numpy(float)
                    for key, sub in group.groupby("sample_cluster")
                    if sub[metric].notna().any()
                }
                for _ in range(1000):
                    picked = rng.choice(clusters, len(clusters), replace=True)
                    bootstrap.append(
                        float(np.mean(np.concatenate([by_cluster[key] for key in picked])))
                    )
                summaries.append(
                    {
                        **base,
                        "scope": scope,
                        "metric": metric,
                        "mean": float(np.mean(values)),
                        "median": float(np.median(values)),
                        "q10": float(np.quantile(values, 0.10)),
                        "q25": float(np.quantile(values, 0.25)),
                        "q75": float(np.quantile(values, 0.75)),
                        "q90": float(np.quantile(values, 0.90)),
                        "cluster_bootstrap_mean_ci_low": float(np.quantile(bootstrap, 0.025)),
                        "cluster_bootstrap_mean_ci_high": float(np.quantile(bootstrap, 0.975)),
                        "n_rows": int(len(values)),
                        "n_samples": int(len(clusters)),
                    }
                )
    write_dual(pd.DataFrame(summaries), out / "descriptive_loss_summaries")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, default=Path("analysis"))
    args = parser.parse_args()
    build(args.input_dir, args.analysis_dir)


if __name__ == "__main__":
    main()
