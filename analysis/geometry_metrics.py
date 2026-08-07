#!/usr/bin/env python3
"""Materialize the geometry metrics that are genuinely present in the run."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import load_required_tables, robust_zscore, write_dual


UNAVAILABLE_GEOMETRY = (
    "full-history-new-direction-residual;online-ridge-leverage;"
    "expanding-local-ew-covariance-drift;principal-angles;"
    "projection-distance;canonical-correlation;time-resolved-effective-rank;"
    "whitened-coordinates;mahalanobis;skewness;kurtosis"
)


def _eventize(
    frame: pd.DataFrame,
    value: str,
    event_signal: str,
    high_is_event: bool,
    group_columns: list[str],
) -> pd.DataFrame:
    rows = frame.copy()
    signal = rows[value].astype(float) * (1.0 if high_is_event else -1.0)
    rows["_event_oriented_value"] = signal
    rows["robust_z"] = rows.groupby(group_columns, dropna=False)["_event_oriented_value"].transform(
        lambda values: robust_zscore(values.to_numpy(float))
    )
    rows["percentile_rank"] = rows.groupby(
        ["strategy", "layer", "head"], dropna=False
    )["_event_oriented_value"].rank(method="average", pct=True)
    rows["event_by_mad"] = rows["robust_z"].ge(3.0)
    rows["event_by_top_quantile"] = rows["percentile_rank"].ge(0.95)
    rows["is_candidate_event"] = rows["event_by_mad"] | rows["event_by_top_quantile"]
    rows["event_signal"] = event_signal
    rows["signal_magnitude"] = rows[value].astype(float)
    return rows.loc[rows["is_candidate_event"]].copy()


def build(input_dir: Path, analysis_dir: Path) -> None:
    tables = load_required_tables(input_dir)
    out = Path(analysis_dir) / "tables"
    temporal = tables["temporal"]
    geometry = temporal[temporal["signal_kind"].eq("value_geometry")].copy()
    geometry["layer"] = geometry["layer"].astype(int)
    geometry["head"] = geometry["head"].astype(int)
    geometry["sample_cluster"] = (
        geometry["task"].astype(str) + "::" + geometry["sample_id"].astype(str)
    )
    geometry["geometry_reference"] = "selected_core_at_anchor"
    geometry["geometry_time_scope"] = "anchor_only"
    geometry["unavailable_geometry_metrics"] = UNAVAILABLE_GEOMETRY

    residual = temporal[
        temporal["signal_kind"].eq("future_new_token_value_residual")
    ].copy()
    residual["layer"] = residual["layer"].astype(int)
    residual["head"] = residual["head"].astype(int)
    residual["sample_cluster"] = (
        residual["task"].astype(str) + "::" + residual["sample_id"].astype(str)
    )
    residual["global_generation_index"] = residual["anchor"] + residual["lag"] - 1
    residual["residual_reference"] = "full_rank_span_of_selected_core"
    residual["residual_is_full_history"] = False
    residual["unavailable_geometry_metrics"] = UNAVAILABLE_GEOMETRY
    write_dual(
        geometry,
        out / "geometry_anchor_summaries",
    )
    write_dual(residual, out / "future_selected_core_residuals")

    # Required geometry table contains both record kinds with explicit semantics.
    geometry_common = geometry.copy()
    geometry_common["geometry_record_kind"] = "anchor_selected_core_summary"
    residual_common = residual.copy()
    residual_common["geometry_record_kind"] = "future_token_selected_core_residual"
    combined = pd.concat([geometry_common, residual_common], ignore_index=True, sort=False)
    write_dual(combined, out / "geometry_metrics")

    spectra = []
    for row in geometry.itertuples(index=False):
        values = row.leading_singular_values
        if isinstance(values, str):
            import json

            values = json.loads(values)
        for rank, value in enumerate(values, start=1):
            spectra.append(
                {
                    "sample_id": row.sample_id,
                    "sample_cluster": row.sample_cluster,
                    "task": row.task,
                    "anchor": int(row.anchor),
                    "strategy": row.strategy,
                    "layer": int(row.layer),
                    "head": int(row.head),
                    "singular_value_rank": rank,
                    "singular_value": float(value),
                    "spectrum_scope": "selected_core_at_anchor_first_8_values",
                }
            )
    write_dual(pd.DataFrame(spectra), out / "singular_spectrum")

    score = pd.read_parquet(out / "score_stability.parquet")
    sets = pd.read_parquet(out / "set_stability.parquet")
    attention = temporal[temporal["signal_kind"].eq("query_attention_drift")].copy()
    attention["layer"] = attention["layer"].astype(int)
    attention["head"] = attention["head"].astype(int)
    attention["sample_cluster"] = (
        attention["task"].astype(str) + "::" + attention["sample_id"].astype(str)
    )
    attention["attention_distribution_shift"] = 1.0 - attention[
        "attention_distribution_cosine"
    ]
    attention["query_direction_shift"] = 1.0 - attention["query_cosine_to_anchor"]

    group = ["sample_id", "anchor", "strategy", "layer", "head"]
    events = [
        _eventize(
            residual,
            "future_new_token_residual",
            "selected_core_new_token_residual_spike",
            True,
            group,
        ),
        _eventize(
            attention,
            "attention_distribution_shift",
            "attention_distribution_shift",
            True,
            group,
        ),
        _eventize(
            attention,
            "query_direction_shift",
            "query_direction_shift",
            True,
            group,
        ),
    ]
    # Score/set diagnostics are KV-head aggregated, so head is intentionally NA.
    score_for_event = score.copy()
    score_for_event["head"] = np.nan
    score_for_event["future_margin_oriented"] = score_for_event[
        "selection_boundary_margin_future"
    ]
    events.append(
        _eventize(
            score_for_event,
            "future_margin_oriented",
            "selection_margin_collapse",
            False,
            ["sample_id", "anchor", "strategy", "layer"],
        )
    )
    set_for_event = sets.copy()
    set_for_event["head"] = np.nan
    events.append(
        _eventize(
            set_for_event,
            "selected_core_turnover",
            "selected_core_turnover",
            True,
            ["sample_id", "anchor", "strategy", "layer"],
        )
    )
    event = pd.concat(events, ignore_index=True, sort=False)

    step = pd.read_parquet(out / "per_step_metrics.parquet")
    step = step[step["analysis_primary"]][
        [
            "sample_id",
            "task",
            "anchor",
            "strategy",
            "future_step",
            "global_generation_index",
            "delta_nll",
            "approx_kl",
            "attention_output_error_mean",
        ]
    ].rename(columns={"future_step": "lag", "delta_nll": "stale_loss"})
    event = event.merge(
        step,
        on=["sample_id", "task", "anchor", "strategy", "lag"],
        how="left",
        suffixes=("", "_step"),
    )
    turnover = sets[
        ["sample_id", "anchor", "strategy", "layer", "lag", "selected_core_turnover"]
    ]
    event = event.merge(
        turnover,
        on=["sample_id", "anchor", "strategy", "layer", "lag"],
        how="left",
        suffixes=("", "_joined"),
    )
    if "selected_core_turnover_joined" in event:
        event["selected_set_turnover"] = event["selected_core_turnover_joined"].combine_first(
            event.get("selected_core_turnover")
        )
    else:
        event["selected_set_turnover"] = event.get("selected_core_turnover")
    oracle_overlap = tables["horizon"]
    oracle_overlap = oracle_overlap[oracle_overlap["horizon"].eq(64)][
        ["sample_id", "anchor", "strategy", "oracle_overlap"]
    ]
    event = event.merge(
        oracle_overlap, on=["sample_id", "anchor", "strategy"], how="left"
    )
    event["refresh_benefit"] = np.nan
    event["refreshed_loss"] = np.nan
    event["token_text"] = pd.NA
    event["token_text_availability"] = "unavailable_not_persisted"
    event["event_thresholds"] = "robust_z>=3 OR within-signal percentile>=95%"
    columns = [
        "sample_id",
        "task",
        "anchor",
        "lag",
        "global_generation_index",
        "layer",
        "head",
        "strategy",
        "event_signal",
        "signal_magnitude",
        "robust_z",
        "percentile_rank",
        "event_by_mad",
        "event_by_top_quantile",
        "stale_loss",
        "refreshed_loss",
        "refresh_benefit",
        "selected_set_turnover",
        "oracle_overlap",
        "token_text",
        "token_text_availability",
        "event_thresholds",
    ]
    write_dual(event.reindex(columns=columns), out / "direction_shift_events")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, default=Path("analysis"))
    args = parser.parse_args()
    build(args.input_dir, args.analysis_dir)


if __name__ == "__main__":
    main()
