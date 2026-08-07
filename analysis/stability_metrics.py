#!/usr/bin/env python3
"""Extract score/set stability and horizon-dependence metrics."""
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from common import load_required_tables, parse_json, write_dual


def _oracle_pair_rows(candidate: pd.DataFrame) -> pd.DataFrame:
    rows = []
    oracle = candidate[candidate["strategy"].eq("future_attention_oracle")].copy()
    for (sample, task, anchor), group in oracle.groupby(["sample_id", "task", "anchor"]):
        for left, right in combinations(
            sorted(group["horizon_condition"].dropna().astype(int).unique()), 2
        ):
            lrow = group[group["horizon_condition"].eq(left)].iloc[0]
            overlaps = parse_json(lrow["overlaps"], "candidate overlaps")
            key = f"future_attention_oracle@{right}"
            record = overlaps.get(key, {})
            rows.append(
                {
                    "sample_id": sample,
                    "sample_cluster": f"{task}::{sample}",
                    "task": task,
                    "anchor": int(anchor),
                    "horizon_left": left,
                    "horizon_right": right,
                    "mean_jaccard": record.get("mean_jaccard", np.nan),
                    "mean_retention": record.get("mean_left_retention_ratio", np.nan),
                    "availability": "available",
                    "scope": "selected_core_all_28_layers",
                }
            )
    return pd.DataFrame(rows)


def build(input_dir: Path, analysis_dir: Path) -> None:
    tables = load_required_tables(input_dir)
    out = Path(analysis_dir) / "tables"
    temporal = tables["temporal"]
    score = temporal[temporal["signal_kind"].eq("score_drift")].copy()
    score["layer"] = score["layer"].astype(int)
    score["sample_cluster"] = score["task"].astype(str) + "::" + score["sample_id"].astype(str)
    score["pearson_score_correlation"] = score["score_autocorrelation"]
    score["score_relative_l2_change"] = score["normalized_l2_drift"]
    score["selection_margin_change"] = (
        score["selection_boundary_margin_future"]
        - score["selection_boundary_margin_anchor"]
    )
    score["kendall_rank_correlation"] = np.nan
    score["mean_rank_displacement"] = np.nan
    score["token_scope"] = "old_tokens_present_at_anchor_only"
    score["recent_excluded_from_eligible_core"] = True
    score["unavailable_metrics"] = (
        "kendall;per-token-rank-displacement;new-token-score-drift"
    )
    write_dual(score, out / "score_stability")

    selected = score[
        [
            "sample_id",
            "sample_cluster",
            "task",
            "anchor",
            "strategy",
            "layer",
            "lag",
            "top_core_jaccard",
            "selection_boundary_margin_anchor",
            "selection_boundary_margin_future",
        ]
    ].copy()
    # Both old-token top sets use the same selected-core budget. Thus retention
    # follows exactly from Jaccard: I/B = 2J/(1+J).
    selected["selected_core_retention"] = (
        2.0 * selected["top_core_jaccard"] / (1.0 + selected["top_core_jaccard"])
    )
    selected["selected_core_turnover"] = 1.0 - selected["selected_core_retention"]
    selected["selected_core_replacements_estimate"] = (
        220.0 * selected["selected_core_turnover"]
    )
    selected["sink_jaccard"] = 1.0
    selected["recent_jaccard"] = np.nan
    selected["whole_cache_jaccard"] = np.nan
    selected["set_scope"] = "selected_core_old_token_universe"
    selected["unavailable_metrics"] = (
        "token-identities-entering-or-leaving;recent-jaccard;whole-cache-jaccard;"
        "new-token-entry;rank-reversal"
    )
    write_dual(selected, out / "set_stability")

    oracle_pairs = _oracle_pair_rows(tables["candidate"])
    write_dual(oracle_pairs, out / "future_oracle_horizon_overlap")

    horizon = tables["horizon"].copy()
    rankings = (
        horizon.groupby(["task", "horizon", "strategy"], as_index=False)
        .agg(
            mean_avg_delta_nll=("avg_delta_nll", "mean"),
            median_avg_delta_nll=("avg_delta_nll", "median"),
            mean_avg_approx_kl=("avg_approx_kl", "mean"),
            mean_oracle_overlap=("oracle_overlap", "mean"),
            n_samples=("sample_id", "nunique"),
            n_rows=("sample_id", "size"),
        )
    )
    overall = (
        horizon.groupby(["horizon", "strategy"], as_index=False)
        .agg(
            mean_avg_delta_nll=("avg_delta_nll", "mean"),
            median_avg_delta_nll=("avg_delta_nll", "median"),
            mean_avg_approx_kl=("avg_approx_kl", "mean"),
            mean_oracle_overlap=("oracle_overlap", "mean"),
            n_samples=("sample_id", "nunique"),
            n_rows=("sample_id", "size"),
        )
        .assign(task="ALL")
    )
    rankings = pd.concat([rankings, overall], ignore_index=True)
    rankings["loss_rank"] = rankings.groupby(["task", "horizon"])[
        "mean_avg_delta_nll"
    ].rank(method="average")
    rankings["kl_rank"] = rankings.groupby(["task", "horizon"])[
        "mean_avg_approx_kl"
    ].rank(method="average")
    write_dual(rankings, out / "selector_horizon_rankings")

    # Explicit per-sample ranking reversal records.
    per = horizon[
        ["sample_id", "task", "anchor", "horizon", "strategy", "avg_delta_nll"]
    ].copy()
    per["loss_rank"] = per.groupby(["sample_id", "anchor", "horizon"])[
        "avg_delta_nll"
    ].rank(method="average")
    wide = per.pivot_table(
        index=["sample_id", "task", "anchor", "strategy"],
        columns="horizon",
        values="loss_rank",
    ).reset_index()
    for left, right in combinations([1, 4, 16, 64], 2):
        if left in wide and right in wide:
            wide[f"rank_change_{left}_to_{right}"] = wide[right] - wide[left]
            wide[f"rank_reversal_{left}_to_{right}"] = (
                wide[f"rank_change_{left}_to_{right}"].abs() >= 1.0
            )
    write_dual(wide, out / "per_sample_horizon_rank_reversals")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, default=Path("analysis"))
    args = parser.parse_args()
    build(args.input_dir, args.analysis_dir)


if __name__ == "__main__":
    main()
