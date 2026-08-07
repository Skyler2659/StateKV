"""Descriptive-only summaries, residual matrices, and correlations."""
from __future__ import annotations

import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def _bootstrap_mean_ci(
    values: Sequence[float], rng: np.random.Generator, draws: int
) -> List[Optional[float]]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not array.size:
        return [None, None]
    samples = rng.choice(array, size=(int(draws), array.size), replace=True).mean(axis=1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def _bootstrap_correlation_ci(
    left: np.ndarray,
    right: np.ndarray,
    method: str,
    rng: np.random.Generator,
    draws: int,
) -> List[Optional[float]]:
    if left.size < 3:
        return [None, None]
    values = []
    for _ in range(int(draws)):
        index = rng.integers(0, left.size, size=left.size)
        x, y = left[index], right[index]
        if np.std(x) <= 0 or np.std(y) <= 0:
            continue
        value = (
            spearmanr(x, y).statistic
            if method == "spearman"
            else pearsonr(x, y).statistic
        )
        if math.isfinite(float(value)):
            values.append(float(value))
    if not values:
        return [None, None]
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def _selection_sets(row: pd.Series) -> Dict[int, set]:
    layers = json.loads(row["layers"])
    return {
        int(layer["layer"]): set(int(value) for value in layer["selected_positions"])
        for layer in layers
    }


def _mean_set_jaccard(left: Dict[int, set], right: Dict[int, set]) -> float:
    values = []
    for layer in sorted(set(left) & set(right)):
        union = left[layer] | right[layer]
        values.append(
            len(left[layer] & right[layer]) / len(union) if union else 1.0
        )
    return float(np.mean(values)) if values else float("nan")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(
        "Object of type %s is not JSON serializable"
        % value.__class__.__name__
    )


def _correlation_row(
    name: str,
    frame: pd.DataFrame,
    x_column: str,
    y_column: str,
    rng: np.random.Generator,
    draws: int,
) -> Dict[str, Any]:
    clean = frame[[x_column, y_column]].replace([np.inf, -np.inf], np.nan).dropna()
    x = clean[x_column].to_numpy(dtype=np.float64)
    y = clean[y_column].to_numpy(dtype=np.float64)
    if x.size >= 2 and np.std(x) > 0 and np.std(y) > 0:
        pearson = float(pearsonr(x, y).statistic)
        spearman = float(spearmanr(x, y).statistic)
    else:
        pearson = None
        spearman = None
    pearson_ci = _bootstrap_correlation_ci(
        x, y, "pearson", rng, draws
    )
    spearman_ci = _bootstrap_correlation_ci(
        x, y, "spearman", rng, draws
    )
    return {
        "correlation": name,
        "x": x_column,
        "y": y_column,
        "valid_sample_count": int(x.size),
        "pearson": pearson,
        "pearson_ci_low": pearson_ci[0],
        "pearson_ci_high": pearson_ci[1],
        "spearman": spearman,
        "spearman_ci_low": spearman_ci[0],
        "spearman_ci_high": spearman_ci[1],
    }


def compute_descriptive_statistics(
    run_dir: Path,
    seed: int = 42,
    bootstrap_samples: int = 1000,
) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    horizon = pd.read_parquet(run_dir / "horizon_losses.parquet")
    step = pd.read_parquet(run_dir / "step_losses.parquet")
    candidate = pd.read_parquet(run_dir / "candidate_sets.parquet")
    temporal = pd.read_parquet(run_dir / "temporal_signals.parquet")
    valid_horizon = (
        horizon[horizon["valid"].eq(True)].copy()
        if "valid" in horizon.columns
        else horizon.iloc[0:0].copy()
    )
    valid_step = (
        step[step["valid"].eq(True)].copy()
        if "valid" in step.columns
        else step.iloc[0:0].copy()
    )
    rng = np.random.default_rng(int(seed))

    ranking_rows = []
    if {"horizon", "strategy", "avg_delta_nll"}.issubset(
        valid_horizon.columns
    ):
        for horizon_value, group in valid_horizon.groupby("horizon"):
            for strategy, values in group.groupby("strategy"):
                loss = values["avg_delta_nll"].dropna().to_numpy(
                    dtype=np.float64
                )
                ci = _bootstrap_mean_ci(loss, rng, bootstrap_samples)
                ranking_rows.append(
                    {
                        "horizon": int(horizon_value),
                        "strategy": strategy,
                        "mean_loss": (
                            float(np.mean(loss)) if loss.size else None
                        ),
                        "median_loss": (
                            float(np.median(loss)) if loss.size else None
                        ),
                        "bootstrap_95_ci": ci,
                        "n": int(loss.size),
                    }
                )

    per_unit_ranks = []
    index_columns = ["task", "sample_id", "anchor", "horizon"]
    if set(index_columns + ["strategy", "avg_delta_nll"]).issubset(
        valid_horizon.columns
    ):
        for key, group in valid_horizon.groupby(index_columns):
            ordered = group.sort_values(
                ["avg_delta_nll", "strategy"], kind="stable"
            )
            for rank, (_, row) in enumerate(ordered.iterrows(), start=1):
                per_unit_ranks.append(
                    {
                        "task": str(key[0]),
                        "sample_id": str(key[1]),
                        "anchor": int(key[2]),
                        "horizon": int(key[3]),
                        "strategy": row["strategy"],
                        "rank": rank,
                    }
                )
    rank_frame = pd.DataFrame(per_unit_ranks)
    reversals = []
    if not rank_frame.empty:
        for key, group in rank_frame.groupby(["task", "sample_id", "anchor"]):
            horizons = sorted(group["horizon"].unique())
            for left_h, right_h in combinations(horizons, 2):
                left = group[group["horizon"] == left_h].set_index("strategy")["rank"]
                right = group[group["horizon"] == right_h].set_index("strategy")["rank"]
                common = sorted(set(left.index) & set(right.index))
                count = 0
                for a, b in combinations(common, 2):
                    if (left[a] - left[b]) * (right[a] - right[b]) < 0:
                        count += 1
                reversals.append(
                    {
                        "task": key[0],
                        "sample_id": key[1],
                        "anchor": int(key[2]),
                        "left_horizon": int(left_h),
                        "right_horizon": int(right_h),
                        "pairwise_rank_reversals": int(count),
                    }
                )

    oracle = (
        candidate[
            candidate["strategy"].eq("future_attention_oracle")
            & candidate["valid"].eq(True)
        ].copy()
        if {"strategy", "valid"}.issubset(candidate.columns)
        else candidate.iloc[0:0].copy()
    )
    oracle_pairs = []
    for key, group in oracle.groupby(["task", "sample_id", "anchor"]):
        records = {
            int(row["horizon_condition"]): _selection_sets(row)
            for _, row in group.iterrows()
        }
        for left_h in sorted(records):
            for right_h in sorted(records):
                oracle_pairs.append(
                    {
                        "task": key[0],
                        "sample_id": key[1],
                        "anchor": int(key[2]),
                        "left_horizon": left_h,
                        "right_horizon": right_h,
                        "jaccard": _mean_set_jaccard(
                            records[left_h], records[right_h]
                        ),
                    }
                )
    oracle_frame = pd.DataFrame(oracle_pairs)
    oracle_matrix = []
    if not oracle_frame.empty:
        for key, group in oracle_frame.groupby(["left_horizon", "right_horizon"]):
            values = group["jaccard"].dropna().to_numpy(dtype=np.float64)
            ci = _bootstrap_mean_ci(values, rng, bootstrap_samples)
            oracle_matrix.append(
                {
                    "left_horizon": int(key[0]),
                    "right_horizon": int(key[1]),
                    "mean": float(values.mean()) if values.size else None,
                    "q25": float(np.quantile(values, 0.25)) if values.size else None,
                    "median": float(np.median(values)) if values.size else None,
                    "q75": float(np.quantile(values, 0.75)) if values.size else None,
                    "ci_low": ci[0],
                    "ci_high": ci[1],
                    "n": int(values.size),
                }
            )

    additive_rows = []
    additive_columns = {
        "task",
        "sample_id",
        "anchor",
        "strategy",
        "horizon",
        "avg_delta_nll",
    }
    if additive_columns.issubset(valid_horizon.columns):
        for key, group in valid_horizon.groupby(
            ["task", "sample_id", "anchor"]
        ):
            matrix = group.pivot_table(
                index="strategy",
                columns="horizon",
                values="avg_delta_nll",
                aggfunc="mean",
            )
            if matrix.empty or matrix.isna().any().any():
                continue
            values = matrix.to_numpy(dtype=np.float64)
            residual = (
                values
                - values.mean(axis=1, keepdims=True)
                - values.mean(axis=0, keepdims=True)
                + values.mean()
            )
            normalized = float(
                np.linalg.norm(residual) / max(np.linalg.norm(values), 1e-12)
            )
            for row_index, strategy in enumerate(matrix.index):
                for column_index, horizon_value in enumerate(matrix.columns):
                    additive_rows.append(
                        {
                            "task": key[0],
                            "sample_id": key[1],
                            "anchor": int(key[2]),
                            "strategy": strategy,
                            "horizon": int(horizon_value),
                            "loss": float(values[row_index, column_index]),
                            "additive_residual": float(
                                residual[row_index, column_index]
                            ),
                            "normalized_residual_frobenius_norm": normalized,
                        }
                    )
    additive_frame = pd.DataFrame(additive_rows)
    additive_frame.to_parquet(run_dir / "additive_residuals.parquet", index=False)

    join_keys = ["run_id", "task", "sample_id", "anchor", "strategy"]
    correlations = []
    query = (
        temporal[
            temporal["signal_kind"].eq("query_attention_drift")
        ].copy()
        if "signal_kind" in temporal.columns
        else temporal.iloc[0:0].copy()
    )
    if not query.empty:
        query_loss = query.merge(
            valid_step,
            left_on=join_keys + ["lag"],
            right_on=join_keys + ["future_step"],
            suffixes=("", "_loss"),
        )
        correlations.append(
            _correlation_row(
                "query_drift_vs_future_loss",
                query_loss,
                "query_cosine_to_anchor",
                "delta_nll",
                rng,
                bootstrap_samples,
            )
        )

    score = (
        temporal[temporal["signal_kind"].eq("score_drift")].copy()
        if "signal_kind" in temporal.columns
        else temporal.iloc[0:0].copy()
    )
    score_names = {
        "snapkv": "attention_score_drift_vs_future_loss",
        "v_ridge_leverage": "leverage_score_drift_vs_future_loss",
        "attention_weighted_v_ridge_leverage": "hybrid_score_drift_vs_future_loss",
    }
    for strategy, name in score_names.items():
        subset = score[score["strategy"].eq(strategy)]
        if subset.empty:
            continue
        joined = subset.merge(
            valid_step,
            left_on=join_keys + ["lag"],
            right_on=join_keys + ["future_step"],
            suffixes=("", "_loss"),
        )
        correlations.append(
            _correlation_row(
                name,
                joined,
                "normalized_l2_drift",
                "delta_nll",
                rng,
                bootstrap_samples,
            )
        )

    leverage_horizon = (
        valid_horizon[
            valid_horizon["strategy"].eq("v_ridge_leverage")
        ].copy()
        if {
            "strategy",
            "horizon",
            "validity_horizons",
            "selection_boundary_margin_mean",
        }.issubset(valid_horizon.columns)
        else valid_horizon.iloc[0:0].copy()
    )
    if not leverage_horizon.empty:
        candidate_keys = [
            "run_id",
            "task",
            "sample_id",
            "anchor",
            "strategy",
        ]
        longest = (
            leverage_horizon.sort_values("horizon", kind="stable")
            .groupby(candidate_keys, as_index=False)
            .tail(1)
        )
        margin_validity_rows = []
        for _, row in longest.iterrows():
            try:
                observations = json.loads(row["validity_horizons"])
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            for observation in observations:
                margin_validity_rows.append(
                    {
                        "selection_boundary_margin_mean": row[
                            "selection_boundary_margin_mean"
                        ],
                        "observed_validity_horizon": observation.get(
                            "observed_horizon"
                        ),
                        "metric": observation.get("metric"),
                        "threshold": observation.get("threshold"),
                        "is_right_censored": observation.get(
                            "is_right_censored"
                        ),
                    }
                )
        margin_validity = pd.DataFrame(margin_validity_rows)
        if not margin_validity.empty:
            for (metric, threshold), group in margin_validity.groupby(
                ["metric", "threshold"]
            ):
                correlation = _correlation_row(
                    "leverage_margin_vs_validity_horizon:%s@%g"
                    % (metric, float(threshold)),
                    group,
                    "selection_boundary_margin_mean",
                    "observed_validity_horizon",
                    rng,
                    bootstrap_samples,
                )
                correlation["right_censored_count"] = int(
                    group["is_right_censored"].eq(True).sum()
                )
                correlations.append(correlation)

    residual = (
        temporal[
            temporal["signal_kind"].eq("future_new_token_value_residual")
        ].copy()
        if "signal_kind" in temporal.columns
        else temporal.iloc[0:0].copy()
    )
    if not residual.empty:
        residual_grouped = (
            residual.groupby(join_keys + ["lag"], as_index=False)[
                "future_new_token_residual"
            ]
            .mean()
        )
        joined = residual_grouped.merge(
            valid_step,
            left_on=join_keys + ["lag"],
            right_on=join_keys + ["future_step"],
        )
        correlations.append(
            _correlation_row(
                "new_token_residual_vs_future_loss",
                joined,
                "future_new_token_residual",
                "delta_nll",
                rng,
                bootstrap_samples,
            )
        )

    geometry = (
        temporal[temporal["signal_kind"].eq("value_geometry")].copy()
        if "signal_kind" in temporal.columns
        else temporal.iloc[0:0].copy()
    )
    if not geometry.empty:
        geometry_grouped = (
            geometry.groupby(join_keys, as_index=False)["effective_rank"].mean()
        )
        joined = geometry_grouped.merge(valid_horizon, on=join_keys)
        correlations.append(
            _correlation_row(
                "effective_rank_vs_future_loss",
                joined,
                "effective_rank",
                "avg_delta_nll",
                rng,
                bootstrap_samples,
            )
        )

    if "oracle_overlap" in valid_horizon:
        correlations.append(
            _correlation_row(
                "oracle_overlap_vs_future_loss",
                valid_horizon,
                "oracle_overlap",
                "avg_delta_nll",
                rng,
                bootstrap_samples,
            )
        )

    correlations_frame = pd.DataFrame(correlations)
    correlations_frame.to_parquet(run_dir / "correlations.parquet", index=False)
    summary = {
        "strategy_ranking_by_horizon": ranking_rows,
        "per_sample_anchor_ranks": per_unit_ranks,
        "rank_reversals": reversals,
        "rank_reversal_total": int(
            sum(row["pairwise_rank_reversals"] for row in reversals)
        ),
        "oracle_set_horizon_variation": {
            "matrix": oracle_matrix,
            "raw_pair_count": len(oracle_pairs),
        },
        "additive_residual_row_count": len(additive_rows),
        "correlations": correlations,
        "notes": [
            "All summaries are descriptive.",
            "Bootstrap intervals resample observed rows with replacement.",
            "No theoretical or method-selection conclusion is generated.",
        ],
    }
    with open(
        run_dir / "descriptive_statistics.json", "w", encoding="utf-8"
    ) as handle:
        json.dump(
            summary,
            handle,
            indent=2,
            sort_keys=True,
            default=_json_default,
        )
    if per_unit_ranks:
        pd.DataFrame(per_unit_ranks).to_parquet(
            run_dir / "per_sample_anchor_ranks.parquet", index=False
        )
    if oracle_pairs:
        oracle_frame.to_parquet(
            run_dir / "oracle_horizon_overlap_pairs.parquet", index=False
        )
    return summary
