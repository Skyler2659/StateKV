#!/usr/bin/env python3
"""Reconstruct the only valid same-token refresh counterfactuals in the run."""
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from common import write_dual


PRIMARY_HORIZON = 64


def _refresh_pairs(step: pd.DataFrame) -> pd.DataFrame:
    rows = []
    primary = step[step["analysis_primary"]].copy()
    anchors = sorted(primary["anchor"].unique())
    for stale_anchor, refresh_anchor in combinations(anchors, 2):
        refresh_lag = int(refresh_anchor - stale_anchor)
        stale_step = refresh_lag + 1
        stale = primary[
            primary["anchor"].eq(stale_anchor)
            & primary["future_step"].eq(stale_step)
        ]
        fresh = primary[
            primary["anchor"].eq(refresh_anchor)
            & primary["future_step"].eq(1)
        ]
        keys = ["sample_id", "task", "strategy"]
        merged = stale.merge(fresh, on=keys, suffixes=("_stale", "_fresh"))
        for row in merged.itertuples(index=False):
            same_token = (
                row.reference_token_id_stale == row.reference_token_id_fresh
                and row.reference_token_position_stale == row.reference_token_position_fresh
            )
            rows.append(
                {
                    "record_scope": "global_output",
                    "sample_id": row.sample_id,
                    "sample_cluster": f"{row.task}::{row.sample_id}",
                    "task": row.task,
                    "strategy": row.strategy,
                    "stale_anchor": stale_anchor,
                    "refresh_anchor": refresh_anchor,
                    "refresh_lag": refresh_lag,
                    "stale_future_step": stale_step,
                    "global_generation_index": refresh_anchor,
                    "target_horizon_condition": PRIMARY_HORIZON,
                    "same_reference_token_verified": bool(same_token),
                    "reference_token_id": int(row.reference_token_id_stale),
                    "reference_token_position": int(row.reference_token_position_stale),
                    "stale_loss": float(row.delta_nll_stale),
                    "refreshed_loss": float(row.delta_nll_fresh),
                    "refresh_benefit": float(row.delta_nll_stale - row.delta_nll_fresh),
                    "stale_approx_kl": float(row.approx_kl_stale),
                    "refreshed_approx_kl": float(row.approx_kl_fresh),
                    "refresh_benefit_approx_kl": float(
                        row.approx_kl_stale - row.approx_kl_fresh
                    ),
                    "stale_attention_output_error": float(
                        row.attention_output_error_mean_stale
                    ),
                    "refreshed_attention_output_error": float(
                        row.attention_output_error_mean_fresh
                    ),
                    "refresh_benefit_attention_output_error": float(
                        row.attention_output_error_mean_stale
                        - row.attention_output_error_mean_fresh
                    ),
                    "layer": np.nan,
                    "head": np.nan,
                    "metric": "delta_nll_and_approx_kl",
                    "counterfactual_scope": (
                        "sparse_saved_cross_anchor_same_token;"
                        "oracle_is_H64_conditioned"
                    ),
                }
            )
    return pd.DataFrame(rows)


def _head_pairs(attention: pd.DataFrame) -> pd.DataFrame:
    rows = []
    anchors = sorted(attention["anchor"].unique())
    for stale_anchor, refresh_anchor in combinations(anchors, 2):
        refresh_lag = int(refresh_anchor - stale_anchor)
        stale_step = refresh_lag + 1
        stale = attention[
            attention["anchor"].eq(stale_anchor)
            & attention["future_step"].eq(stale_step)
        ]
        fresh = attention[
            attention["anchor"].eq(refresh_anchor)
            & attention["future_step"].eq(1)
        ]
        keys = ["sample_id", "task", "strategy", "layer", "head"]
        merged = stale.merge(fresh, on=keys, suffixes=("_stale", "_fresh"))
        for row in merged.itertuples(index=False):
            rows.append(
                {
                    "record_scope": "diagnostic_attention_head",
                    "sample_id": row.sample_id,
                    "sample_cluster": f"{row.task}::{row.sample_id}",
                    "task": row.task,
                    "strategy": row.strategy,
                    "stale_anchor": stale_anchor,
                    "refresh_anchor": refresh_anchor,
                    "refresh_lag": refresh_lag,
                    "stale_future_step": stale_step,
                    "global_generation_index": refresh_anchor,
                    "target_horizon_condition": PRIMARY_HORIZON,
                    "same_reference_token_verified": True,
                    "reference_token_id": np.nan,
                    "reference_token_position": np.nan,
                    "stale_loss": np.nan,
                    "refreshed_loss": np.nan,
                    "refresh_benefit": np.nan,
                    "stale_approx_kl": np.nan,
                    "refreshed_approx_kl": np.nan,
                    "refresh_benefit_approx_kl": np.nan,
                    "stale_attention_output_error": float(
                        row.attention_output_relative_error_stale
                    ),
                    "refreshed_attention_output_error": float(
                        row.attention_output_relative_error_fresh
                    ),
                    "refresh_benefit_attention_output_error": float(
                        row.attention_output_relative_error_stale
                        - row.attention_output_relative_error_fresh
                    ),
                    "layer": int(row.layer),
                    "head": int(row.head),
                    "metric": "attention_output_relative_error",
                    "counterfactual_scope": "sparse_saved_cross_anchor_same_token",
                }
            )
    return pd.DataFrame(rows)


def _validity(step: pd.DataFrame, refresh: pd.DataFrame) -> pd.DataFrame:
    rows = []
    primary = step[step["analysis_primary"]].copy()
    groups = ["sample_id", "task", "anchor", "strategy"]
    threshold_specs = [
        ("absolute_average_delta_nll", "average_delta_nll", [0.01, 0.05, 0.10, 0.25]),
        ("absolute_running_max_delta_nll", "running_max_delta_nll", [0.05, 0.10, 0.25, 0.50]),
    ]
    for key, group in primary.groupby(groups):
        group = group.sort_values("future_step")
        for definition, metric, thresholds in threshold_specs:
            for threshold in thresholds:
                okay = group[group[metric].le(threshold)]
                observed = int(okay["future_step"].max()) if len(okay) else 0
                rows.append(
                    {
                        **dict(zip(groups, key)),
                        "definition": definition,
                        "metric": metric,
                        "threshold": float(threshold),
                        "threshold_label": str(threshold),
                        "observed_horizon": observed,
                        "measurement_limit": int(group["future_step"].max()),
                        "is_right_censored": bool(observed == group["future_step"].max()),
                        "availability": "available",
                        "interpretation_note": "max observed H satisfying threshold",
                    }
                )
                if definition == "absolute_average_delta_nll":
                    rows.append(
                        {
                            **dict(zip(groups, key)),
                            "definition": "relative_to_full_cache_delta_nll",
                            "metric": metric,
                            "threshold": float(threshold),
                            "threshold_label": str(threshold),
                            "observed_horizon": observed,
                            "measurement_limit": int(group["future_step"].max()),
                            "is_right_censored": bool(
                                observed == group["future_step"].max()
                            ),
                            "availability": "available_but_equivalent",
                            "interpretation_note": (
                                "delta_nll is already NLL(compressed)-NLL(full); "
                                "not an independent definition"
                            ),
                        }
                    )

    # Per-sample empirical thresholds prevent high-loss samples from dominating.
    deployable = primary[~primary["strategy"].eq("future_attention_oracle")]
    for (sample, task), sample_rows in deployable.groupby(["sample_id", "task"]):
        for percentile in [0.50, 0.75, 0.90]:
            threshold = float(sample_rows["average_delta_nll"].quantile(percentile))
            for (anchor, strategy), group in primary[
                primary["sample_id"].eq(sample)
            ].groupby(["anchor", "strategy"]):
                group = group.sort_values("future_step")
                okay = group[group["average_delta_nll"].le(threshold)]
                observed = int(okay["future_step"].max()) if len(okay) else 0
                rows.append(
                    {
                        "sample_id": sample,
                        "task": task,
                        "anchor": int(anchor),
                        "strategy": strategy,
                        "definition": "sample_normalized_percentile",
                        "metric": "average_delta_nll",
                        "threshold": threshold,
                        "threshold_label": f"sample_q{int(percentile * 100)}",
                        "observed_horizon": observed,
                        "measurement_limit": int(group["future_step"].max()),
                        "is_right_censored": bool(
                            observed == group["future_step"].max()
                        ),
                        "availability": "available_exploratory",
                        "interpretation_note": (
                            "threshold is per-sample percentile pooled over "
                            "deployable strategies/anchors/steps"
                        ),
                    }
                )

    # Relative-to-refreshed observations exist at only three saved boundaries;
    # they are point checks, not a reconstructed lifetime.
    global_refresh = refresh[refresh["record_scope"].eq("global_output")]
    for row in global_refresh.itertuples(index=False):
        for threshold in [0.01, 0.05, 0.10, 0.25]:
            rows.append(
                {
                    "sample_id": row.sample_id,
                    "task": row.task,
                    "anchor": int(row.stale_anchor),
                    "strategy": row.strategy,
                    "definition": "relative_to_refreshed_sparse_point",
                    "metric": "refresh_benefit",
                    "threshold": threshold,
                    "threshold_label": str(threshold),
                    "observed_horizon": int(row.refresh_lag),
                    "measurement_limit": int(row.refresh_lag),
                    "is_right_censored": np.nan,
                    "availability": "sparse_point_only",
                    "interpretation_note": (
                        f"benefit={row.refresh_benefit:.6g}; "
                        f"within_threshold={bool(row.refresh_benefit <= threshold)}; "
                        "cannot infer maximum validity horizon"
                    ),
                }
            )
    return pd.DataFrame(rows)


def build(analysis_dir: Path) -> None:
    out = Path(analysis_dir) / "tables"
    step = pd.read_parquet(out / "per_step_metrics.parquet")
    attention = pd.read_parquet(out / "attention_output_by_head.parquet")
    global_rows = _refresh_pairs(step)
    if not global_rows["same_reference_token_verified"].all():
        raise ValueError("cross-anchor refresh comparison did not align reference tokens")
    head_rows = _head_pairs(attention)
    refresh = pd.concat([global_rows, head_rows], ignore_index=True, sort=False)
    write_dual(refresh, out / "refresh_benefit_analysis")

    summaries = (
        global_rows.groupby(["task", "strategy", "refresh_lag"], as_index=False)
        .agg(
            mean_refresh_benefit=("refresh_benefit", "mean"),
            median_refresh_benefit=("refresh_benefit", "median"),
            q10_refresh_benefit=("refresh_benefit", lambda x: x.quantile(0.10)),
            q90_refresh_benefit=("refresh_benefit", lambda x: x.quantile(0.90)),
            fraction_positive=("refresh_benefit", lambda x: float((x > 0).mean())),
            n_samples=("sample_id", "nunique"),
        )
    )
    write_dual(summaries, out / "refresh_benefit_summaries")

    validity = _validity(step, refresh)
    validity["sample_cluster"] = validity["task"].astype(str) + "::" + validity[
        "sample_id"
    ].astype(str)
    write_dual(validity, out / "validity_horizon_sensitivity")

    # Attach sparse refresh outcomes to structural events measured at the
    # refreshed state (lag equals refresh_anchor-stale_anchor).
    event_path = out / "direction_shift_events.parquet"
    if event_path.exists():
        events = pd.read_parquet(event_path)
        attach = global_rows[
            [
                "sample_id",
                "strategy",
                "stale_anchor",
                "refresh_lag",
                "stale_loss",
                "refreshed_loss",
                "refresh_benefit",
            ]
        ].rename(columns={"stale_anchor": "anchor", "refresh_lag": "lag"})
        events = events.drop(
            columns=["stale_loss", "refreshed_loss", "refresh_benefit"], errors="ignore"
        ).merge(attach, on=["sample_id", "strategy", "anchor", "lag"], how="left")
        write_dual(events, out / "direction_shift_events")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, default=Path("analysis"))
    args = parser.parse_args()
    build(args.analysis_dir)


if __name__ == "__main__":
    main()
