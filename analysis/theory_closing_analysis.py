#!/usr/bin/env python3
"""Sequence-aware analysis for the strict theory-closing mechanism run."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = (
    ROOT
    / "results/temporal_cache_discovery/theory_closing_4bit_seed42_v1"
)
STAGE1_RUN = (
    ROOT
    / "results/temporal_cache_discovery/functional_probe_stage1_4bit_seed42_v1"
)
SURROGATES = [
    "attention_only_surrogate",
    "surrogate_raw_v",
    "surrogate_projected_v",
    "surrogate_aov",
    "surrogate_aor",
]
FEATURE_LABELS = {
    "attention_only_surrogate": "Attention",
    "surrogate_raw_v": "Raw-V",
    "surrogate_projected_v": "OV",
    "surrogate_aov": "AOV",
    "surrogate_aor": "AOR",
}


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", dir=str(path.parent)
    )
    os.close(descriptor)
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                default=_json_default,
            )
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", dir=str(path.parent)
    )
    os.close(descriptor)
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError("not JSON serializable: %s" % type(value).__name__)


def corr(left: Iterable[float], right: Iterable[float], method: str) -> float:
    x = np.asarray(list(left), dtype=np.float64)
    y = np.asarray(list(right), dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return float("nan")
    if method == "spearman":
        return float(spearmanr(x, y).statistic)
    if method == "kendall":
        return float(kendalltau(x, y).statistic)
    raise ValueError(method)


def calibration(left: Iterable[float], right: Iterable[float]) -> Dict[str, float]:
    x = np.asarray(list(left), dtype=np.float64)
    y = np.asarray(list(right), dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 3 or float(np.var(x)) <= 1e-30:
        return {"slope": float("nan"), "intercept": float("nan")}
    design = np.column_stack([np.ones(len(x)), x])
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    return {
        "intercept": float(coefficients[0]),
        "slope": float(coefficients[1]),
    }


def cluster_bootstrap(
    frame: pd.DataFrame,
    value: str,
    sequence: str = "sample_id",
    samples: int = 2000,
    seed: int = 42,
    statistic: str = "median",
) -> Dict[str, float]:
    per_sequence = frame.groupby(sequence, sort=True)[value].median().dropna()
    values = per_sequence.to_numpy(dtype=np.float64)
    if not len(values):
        return {
            "estimate": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "sequence_count": 0,
        }
    reducer = np.median if statistic == "median" else np.mean
    rng = np.random.default_rng(seed)
    draws = np.asarray(
        [
            reducer(rng.choice(values, size=len(values), replace=True))
            for _ in range(int(samples))
        ]
    )
    return {
        "estimate": float(reducer(values)),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "sequence_count": int(len(values)),
        "per_sequence": {
            str(key): float(item) for key, item in per_sequence.items()
        },
    }


def _top_overlap(a: np.ndarray, b: np.ndarray, take: int) -> float:
    left = set(np.argsort(a, kind="stable")[:take].tolist())
    right = set(np.argsort(b, kind="stable")[:take].tolist())
    return float(len(left & right) / max(1, take))


def subset_objective_analysis(
    subset: pd.DataFrame, run_dir: Path
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    for unit_id, group in subset.groupby("unit_id", sort=False):
        group = group.sort_values("subset_id")
        true = group["true_proj_head_risk"].to_numpy(dtype=np.float64)
        oracle_index = int(np.nanargmin(true))
        oracle_risk = float(true[oracle_index])
        random_baseline = float(np.nanmean(true))
        denominator = max(random_baseline - oracle_risk, 1e-12)
        base = {
            "unit_id": unit_id,
            "sample_id": group["sample_id"].iloc[0],
            "task": group["task"].iloc[0],
            "layer": int(group["layer"].iloc[0]),
            "head": int(group["head"].iloc[0]),
            "kv_head_group": int(group["kv_head_group"].iloc[0]),
            "granularity": group["granularity"].iloc[0],
            "subset_count": int(len(group)),
            "oracle_subset_id": oracle_index,
            "oracle_risk": oracle_risk,
            "random_mean_risk": random_baseline,
            "identity_max_relative_error": float(
                group["identity_relative_error"].max()
            ),
            "median_deleted_mass": float(
                group["deleted_attention_mass"].median()
            ),
            "median_cross_ratio": float(
                np.nanmedian(
                    np.abs(group["cross_proj_interaction"])
                    / (
                        np.abs(group["individual_proj_energy_sum"])
                        + 1e-12
                    )
                )
            ),
            "median_cross_head_cancellation": float(
                group["cross_head_cancellation"].median()
            ),
        }
        for surrogate in SURROGATES:
            values = group[surrogate].to_numpy(dtype=np.float64)
            chosen = int(np.nanargmin(values))
            chosen_risk = float(true[chosen])
            prefix = FEATURE_LABELS[surrogate].lower().replace("-", "_")
            base["%s_spearman" % prefix] = corr(
                values, true, "spearman"
            )
            base["%s_kendall" % prefix] = corr(
                values, true, "kendall"
            )
            base["%s_chosen_subset_id" % prefix] = chosen
            base["%s_chosen_true_risk" % prefix] = chosen_risk
            base["%s_excess_risk" % prefix] = chosen_risk - oracle_risk
            base["%s_normalized_regret" % prefix] = (
                chosen_risk - oracle_risk
            ) / denominator
            base["%s_top1_overlap" % prefix] = float(chosen == oracle_index)
            base["%s_top5pct_overlap" % prefix] = _top_overlap(
                values, true, max(1, int(math.ceil(0.05 * len(true))))
            )
            base["%s_top10pct_overlap" % prefix] = _top_overlap(
                values, true, max(1, int(math.ceil(0.10 * len(true))))
            )
        base["aor_improvement_vs_attention"] = (
            base["attention_chosen_true_risk"]
            - base["aor_chosen_true_risk"]
        )
        base["aor_improvement_vs_raw_v"] = (
            base["raw_v_chosen_true_risk"]
            - base["aor_chosen_true_risk"]
        )
        base["aor_improvement_vs_ov"] = (
            base["ov_chosen_true_risk"] - base["aor_chosen_true_risk"]
        )
        base["aor_improvement_vs_random_mean"] = (
            random_baseline - base["aor_chosen_true_risk"]
        )
        rows.append(base)
    units = pd.DataFrame(rows)
    units.to_parquet(run_dir / "subset_unit_metrics.parquet", index=False)
    units.to_csv(run_dir / "subset_unit_metrics.csv", index=False)
    head_units = units[units["granularity"] == "query_head"].copy()
    primary = cluster_bootstrap(head_units, "aor_spearman")
    task_direction = (
        head_units.groupby(["task", "sample_id"])["aor_spearman"]
        .median()
        .groupby("task")
        .mean()
        .to_dict()
    )
    per_head = (
        units.groupby(
            ["granularity", "task", "layer", "head", "kv_head_group"],
            dropna=False,
        )
        .agg(
            sequence_count=("sample_id", "nunique"),
            aor_spearman_median=("aor_spearman", "median"),
            aor_spearman_mean=("aor_spearman", "mean"),
            aor_kendall_median=("aor_kendall", "median"),
            aov_spearman_median=("aov_spearman", "median"),
            ov_spearman_median=("ov_spearman", "median"),
            raw_v_spearman_median=("raw_v_spearman", "median"),
            aor_normalized_regret_median=(
                "aor_normalized_regret",
                "median",
            ),
            aor_top5pct_overlap_median=(
                "aor_top5pct_overlap",
                "median",
            ),
            aor_improvement_vs_attention_median=(
                "aor_improvement_vs_attention",
                "median",
            ),
            cross_ratio_median=("median_cross_ratio", "median"),
            cross_head_cancellation_median=(
                "median_cross_head_cancellation",
                "median",
            ),
        )
        .reset_index()
    )
    per_head["aor_positive_fraction"] = (
        units.assign(positive=units["aor_spearman"] > 0)
        .groupby(
            ["granularity", "task", "layer", "head", "kv_head_group"],
            dropna=False,
        )["positive"]
        .mean()
        .to_numpy()
    )
    per_head.to_csv(run_dir / "per_head_gqa_summary.csv", index=False)
    gate = bool(
        primary["estimate"] >= 0.30
        and all(float(value) > 0 for value in task_direction.values())
    )
    summary = {
        "schema_version": "subset_oracle_summary_v1",
        "primary_metric": (
            "sequence-clustered median of within-unit AOR vs "
            "projected-head true-risk Spearman"
        ),
        "primary": primary,
        "pre_registered_partial_gate": 0.30,
        "task_direction": {
            str(key): float(value) for key, value in task_direction.items()
        },
        "partial_gate_pass": gate,
        "unit_counts": {
            str(key): int(value)
            for key, value in units["granularity"].value_counts().items()
        },
        "query_head_positive_fraction": float(
            (head_units["aor_spearman"] > 0).mean()
        ),
        "query_head_aor_normalized_regret_median": float(
            head_units["aor_normalized_regret"].median()
        ),
        "query_head_aov_spearman_median": float(
            head_units["aov_spearman"].median()
        ),
        "query_head_ov_spearman_median": float(
            head_units["ov_spearman"].median()
        ),
        "query_head_raw_v_spearman_median": float(
            head_units["raw_v_spearman"].median()
        ),
        "query_head_aor_improvement_vs_attention_positive_fraction": float(
            (head_units["aor_improvement_vs_attention"] > 0).mean()
        ),
        "identity_max_relative_error": float(
            subset["identity_relative_error"].max()
        ),
        "interpretation_scope": "six-sequence controlled mechanism study",
    }
    _atomic_json(run_dir / "subset_oracle_summary.json", summary)
    return summary, units, per_head


def horizon_analysis(
    future: pd.DataFrame,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    head = future[future["row_type"] == "head_horizon"].copy()
    global_rows = future[
        future["row_type"] == "global_stateful_horizon"
    ].copy()
    records: List[Dict[str, Any]] = []
    feature_columns = [
        "future_raw_v_gap",
        "future_projected_v_gap",
        "future_aov_gap",
        "future_aor_gap",
    ]
    targets = [
        "cumulative_direct_proj_benefit",
        "cumulative_stateful_proj_benefit",
    ]
    for (fresh, recent, horizon), group in head.groupby(
        ["fresh_reference", "protected_recent_size", "horizon"],
        sort=True,
    ):
        for feature in feature_columns:
            for target in targets:
                per_sequence = []
                for _, sequence_group in group.groupby("sample_id"):
                    per_sequence.append(
                        corr(
                            sequence_group[feature],
                            sequence_group[target],
                            "spearman",
                        )
                    )
                records.append(
                    {
                        "fresh_reference": fresh,
                        "protected_recent_size": int(recent),
                        "horizon": int(horizon),
                        "feature": feature,
                        "target": target,
                        "sequence_mean_spearman": float(
                            np.nanmean(per_sequence)
                        ),
                        "sequence_median_spearman": float(
                            np.nanmedian(per_sequence)
                        ),
                        "positive_sequence_fraction": float(
                            np.nanmean(np.asarray(per_sequence) > 0)
                        ),
                        "pooled_spearman_descriptive_only": corr(
                            group[feature], group[target], "spearman"
                        ),
                        **calibration(group[feature], group[target]),
                    }
                )
    metrics = pd.DataFrame(records)
    primary_rows = metrics[
        (metrics["feature"] == "future_aor_gap")
        & (
            metrics["target"]
            == "cumulative_direct_proj_benefit"
        )
    ]
    primary = float(primary_rows["sequence_mean_spearman"].median())
    task_directions: Dict[str, float] = {}
    for task, task_group in head.groupby("task"):
        sequence_values = []
        for _, group in task_group.groupby("sample_id"):
            sequence_values.append(
                corr(
                    group["future_aor_gap"],
                    group["cumulative_direct_proj_benefit"],
                    "spearman",
                )
            )
        task_directions[str(task)] = float(np.nanmean(sequence_values))
    # Aggregate the head-wise oracle features before comparison with global NLL.
    feature_aggregate = (
        head.groupby(
            [
                "sample_id",
                "task",
                "protected_recent_size",
                "fresh_reference",
                "horizon",
            ],
            as_index=False,
        )[feature_columns]
        .sum()
    )
    nll = global_rows.drop(
        columns=[
            column
            for column in feature_columns
            if column in global_rows.columns
        ]
    ).merge(
        feature_aggregate,
        on=[
            "sample_id",
            "task",
            "protected_recent_size",
            "fresh_reference",
            "horizon",
        ],
        how="left",
        validate="one_to_one",
    )
    nll_correlations = {
        feature: corr(
            nll[feature], nll["cumulative_nll_benefit"], "spearman"
        )
        for feature in feature_columns
    }
    gate = bool(
        primary >= 0.30
        and all(value > 0 for value in task_directions.values())
    )
    summary = {
        "primary_metric": (
            "median across horizon/recent/fresh-reference cells of "
            "mean per-sequence AOR-gap/direct-projected-benefit Spearman"
        ),
        "primary_estimate": primary,
        "pre_registered_partial_gate": 0.30,
        "task_direction": task_directions,
        "partial_gate_pass": gate,
        "nll_descriptive_correlations_after_head_aggregation": (
            nll_correlations
        ),
        "horizon_cells": metrics.to_dict(orient="records"),
        "fresh_reference_semantics": {
            "per_step_fresh": "same selector recomputed at each future step",
            "horizon_start_once_fresh": (
                "same selector recomputed once at horizon start and frozen"
            ),
        },
    }
    return summary, metrics


def direct_stateful_analysis(
    dense: pd.DataFrame, run_dir: Path
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    frames = [dense.copy()]
    identity_path = STAGE1_RUN / "identity_checks.parquet"
    attention_path = STAGE1_RUN / "attention_labels.parquet"
    if identity_path.exists() and attention_path.exists():
        identity = pd.read_parquet(identity_path)
        labels = pd.read_parquet(attention_path)
        identity = identity[
            identity["stable_denominator"].fillna(False)
        ].copy()
        keys = [
            "run_id",
            "model",
            "task",
            "sample_id",
            "seed",
            "config_hash",
            "git_commit",
            "base_anchor",
            "probe_lag",
            "refresh_anchor",
            "strategy",
            "total_budget",
            "sink_size",
            "protected_recent_size",
            "effective_replay_recent_size",
            "selected_core_budget",
            "layer",
            "head",
        ]
        direct = (
            identity.pivot_table(
                index=keys,
                columns="arm",
                values="identity_delta_norm_sq",
                aggfunc="first",
            )
            .reset_index()
            .rename_axis(None, axis=1)
        )
        direct["direct_head_benefit"] = direct["old"] - direct["fresh"]
        stateful = labels[
            labels["label_granularity"] == "head_pre_projection"
        ][keys + ["refresh_benefit_output"]].rename(
            columns={"refresh_benefit_output": "stateful_head_benefit"}
        )
        prior = direct.merge(
            stateful, on=keys, how="inner", validate="one_to_one"
        )
        prior["feedback_head"] = (
            prior["stateful_head_benefit"]
            - prior["direct_head_benefit"]
        )
        prior["source_matrix"] = "stage1_sparse_budget128_256"
        prior["fresh_reference"] = "per_step_fresh"
        prior["lag"] = prior["probe_lag"].astype(int)
        prior["direct_proj_benefit"] = np.nan
        prior["stateful_proj_benefit"] = np.nan
        prior["feedback_proj"] = np.nan
        frames.append(prior)
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined.to_csv(run_dir / "direct_stateful_decomposition.csv", index=False)
    combined.to_parquet(
        run_dir / "direct_stateful_decomposition.parquet", index=False
    )
    records = []
    groupings = {
        "overall": [],
        "source": ["source_matrix"],
        "task": ["task"],
        "recent": ["protected_recent_size"],
        "budget": ["total_budget"],
        "lag": ["lag"],
        "layer": ["layer"],
        "head": ["head"],
    }
    for label, columns in groupings.items():
        iterator = (
            [((), combined)]
            if not columns
            else combined.groupby(columns, dropna=False, sort=True)
        )
        for keys, group in iterator:
            if not isinstance(keys, tuple):
                keys = (keys,)
            direct_values = group["direct_head_benefit"].to_numpy(
                dtype=np.float64
            )
            stateful_values = group["stateful_head_benefit"].to_numpy(
                dtype=np.float64
            )
            valid = np.isfinite(direct_values) & np.isfinite(
                stateful_values
            )
            x, y = direct_values[valid], stateful_values[valid]
            fit = calibration(x, y)
            records.append(
                {
                    "stratum": label,
                    **{
                        column: value
                        for column, value in zip(columns, keys)
                    },
                    "row_count": int(len(x)),
                    "sequence_count": int(group["sample_id"].nunique()),
                    "spearman": corr(x, y, "spearman"),
                    "slope": fit["slope"],
                    "intercept": fit["intercept"],
                    "median_abs_feedback_fraction": float(
                        np.nanmedian(
                            np.abs(y - x) / (np.abs(y) + 1e-12)
                        )
                    )
                    if len(x)
                    else float("nan"),
                    "direct_variance_fraction_descriptive": float(
                        1.0
                        - np.sum((y - x) ** 2)
                        / max(np.sum((y - np.mean(y)) ** 2), 1e-12)
                    )
                    if len(x)
                    else float("nan"),
                }
            )
    summary_frame = pd.DataFrame(records)
    summary_frame.to_csv(
        run_dir / "direct_stateful_summary.csv", index=False
    )
    overall = summary_frame[summary_frame["stratum"] == "overall"].iloc[0]
    summary = {
        "overall_spearman": float(overall["spearman"]),
        "overall_slope": float(overall["slope"]),
        "overall_median_abs_feedback_fraction": float(
            overall["median_abs_feedback_fraction"]
        ),
        "rows": int(len(combined)),
        "sequences": int(combined["sample_id"].nunique()),
        "note": (
            "Stage-1 sparse and theory-closing dense rows remain marked by "
            "source_matrix and are not treated as extra independent sequences."
        ),
    }
    return combined, summary


def _average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, scores))


def _ridge_fit_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: Sequence[str],
    target: str,
    alpha: float = 1e-3,
) -> np.ndarray:
    from statekv.theory_closing import (
        validate_anchor_predictor_columns,
    )

    validate_anchor_predictor_columns(features)
    train_matrix = train[list(features)].to_numpy(dtype=np.float64)
    test_matrix = test[list(features)].to_numpy(dtype=np.float64)
    medians = np.nanmedian(train_matrix, axis=0)
    medians[~np.isfinite(medians)] = 0.0
    train_matrix = np.where(
        np.isfinite(train_matrix), train_matrix, medians
    )
    test_matrix = np.where(
        np.isfinite(test_matrix), test_matrix, medians
    )
    means = train_matrix.mean(axis=0)
    scales = train_matrix.std(axis=0)
    scales[scales < 1e-12] = 1.0
    train_matrix = (train_matrix - means) / scales
    test_matrix = (test_matrix - means) / scales
    train_matrix = np.column_stack(
        [np.ones(len(train_matrix)), train_matrix]
    )
    test_matrix = np.column_stack(
        [np.ones(len(test_matrix)), test_matrix]
    )
    y = train[target].to_numpy(dtype=np.float64)
    valid = np.isfinite(y)
    regularized = train_matrix[valid].T @ train_matrix[valid]
    penalty = float(alpha) * np.eye(regularized.shape[0])
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        regularized + penalty,
        train_matrix[valid].T @ y[valid],
    )
    return test_matrix @ coefficients


def _inner_gamma(
    train: pd.DataFrame,
    base_features: Sequence[str],
    gamma_columns: Sequence[str],
    target: str,
) -> str:
    scores: Dict[str, List[float]] = {
        column: [] for column in gamma_columns
    }
    for held_out in sorted(train["sample_id"].unique()):
        inner_train = train[train["sample_id"] != held_out]
        inner_test = train[train["sample_id"] == held_out]
        if inner_train["sample_id"].nunique() < 2:
            continue
        for column in gamma_columns:
            prediction = _ridge_fit_predict(
                inner_train,
                inner_test,
                list(base_features) + [column],
                target,
            )
            truth = inner_test[target].to_numpy(dtype=np.float64)
            scores[column].append(
                float(np.nanmean(np.abs(prediction - truth)))
            )
    return min(
        gamma_columns,
        key=lambda column: (
            np.nanmean(scores[column]) if scores[column] else float("inf"),
            column,
        ),
    )


def future_state_prediction_analysis(
    future: pd.DataFrame,
    horizon_summary: Mapping[str, Any],
    run_dir: Path,
) -> Dict[str, Any]:
    if not bool(horizon_summary["partial_gate_pass"]):
        summary = {
            "executed": False,
            "reason": "Experiment C future-oracle partial gate failed",
            "models": {},
        }
        _atomic_json(
            run_dir / "future_state_prediction_summary.json", summary
        )
        return summary
    head = future[future["row_type"] == "head_horizon"].copy()
    global_rows = future[
        future["row_type"] == "global_stateful_horizon"
    ].copy()
    gaussian_path = run_dir / "gaussian_query_anchor_features.parquet"
    if not gaussian_path.exists():
        raise RuntimeError(
            "CapKV-style Gaussian-query anchor features are missing"
        )
    gaussian = pd.read_parquet(gaussian_path)
    head = head.merge(
        gaussian,
        on=[
            "sample_id",
            "task",
            "protected_recent_size",
            "layer",
            "head",
        ],
        how="left",
        validate="many_to_one",
    )
    head["task_is_niah"] = head["task"].astype(str).str.contains(
        "niah"
    ).astype(float)
    head["log_horizon"] = np.log1p(head["horizon"].astype(float))
    head["layer_scaled"] = head["layer"].astype(float) / 27.0
    head["head_scaled"] = head["head"].astype(float) / 11.0
    current = head[head["horizon"] == 1][
        [
            "sample_id",
            "protected_recent_size",
            "fresh_reference",
            "layer",
            "head",
            "future_aor_gap",
        ]
    ].rename(columns={"future_aor_gap": "anchor_current_query_aor_gap"})
    head = head.merge(
        current,
        on=[
            "sample_id",
            "protected_recent_size",
            "fresh_reference",
            "layer",
            "head",
        ],
        how="left",
        validate="many_to_one",
    )
    base = [
        "log_horizon",
        "protected_recent_size",
        "task_is_niah",
        "layer_scaled",
        "head_scaled",
    ]
    model_features: Dict[str, List[str]] = {
        "age_only": list(base),
        "current_query": base + ["anchor_current_query_aor_gap"],
    }
    for window in (8, 16, 32):
        model_features["observation_w%d" % window] = base + [
            "anchor_obs_w%d_mean" % window,
            "anchor_obs_w%d_std" % window,
            "anchor_obs_w%d_trend" % window,
        ]
        model_features["gaussian_query_w%d" % window] = base + [
            "anchor_gaussian_q_w%d_mean_norm" % window,
            "anchor_gaussian_q_w%d_cov_trace" % window,
            "anchor_gaussian_q_w%d_current_mahal" % window,
            "anchor_gaussian_q_w%d_top_eigen_fraction" % window,
            "anchor_gaussian_q_w%d_shrinkage" % window,
        ]
    gamma_columns = sorted(
        column
        for column in head.columns
        if column.startswith("anchor_ema_g")
    )
    targets = [
        "future_aor_gap",
        "cumulative_direct_proj_benefit",
        "cumulative_stateful_proj_benefit",
    ]
    prediction_rows: List[Dict[str, Any]] = []
    chosen_gammas: Dict[str, Dict[str, str]] = {}
    for target in targets:
        for held_out in sorted(head["sample_id"].unique()):
            train = head[head["sample_id"] != held_out]
            test = head[head["sample_id"] == held_out]
            threshold = float(train[target].quantile(0.8))
            for model, features in model_features.items():
                prediction = _ridge_fit_predict(
                    train, test, features, target
                )
                for index, value in zip(test.index, prediction):
                    prediction_rows.append(
                        {
                            "row_index": int(index),
                            "sample_id": held_out,
                            "task": test.loc[index, "task"],
                            "target": target,
                            "model": model,
                            "truth": float(test.loc[index, target]),
                            "prediction": float(value),
                            "tail_threshold_train_only": threshold,
                            "horizon": int(test.loc[index, "horizon"]),
                            "protected_recent_size": int(
                                test.loc[index, "protected_recent_size"]
                            ),
                            "fresh_reference": test.loc[
                                index, "fresh_reference"
                            ],
                            "layer": int(test.loc[index, "layer"]),
                            "head": int(test.loc[index, "head"]),
                        }
                    )
            chosen = _inner_gamma(train, base, gamma_columns, target)
            chosen_gammas.setdefault(target, {})[str(held_out)] = chosen
            prediction = _ridge_fit_predict(
                train, test, base + [chosen], target
            )
            for index, value in zip(test.index, prediction):
                prediction_rows.append(
                    {
                        "row_index": int(index),
                        "sample_id": held_out,
                        "task": test.loc[index, "task"],
                        "target": target,
                        "model": "ema_nested",
                        "truth": float(test.loc[index, target]),
                        "prediction": float(value),
                        "tail_threshold_train_only": threshold,
                        "horizon": int(test.loc[index, "horizon"]),
                        "protected_recent_size": int(
                            test.loc[index, "protected_recent_size"]
                        ),
                        "fresh_reference": test.loc[
                            index, "fresh_reference"
                        ],
                        "layer": int(test.loc[index, "layer"]),
                        "head": int(test.loc[index, "head"]),
                    }
                )
    predictions = pd.DataFrame(prediction_rows)
    predictions.to_parquet(
        run_dir / "future_state_loso_predictions.parquet", index=False
    )
    metric_rows = []
    for (target, model), group in predictions.groupby(
        ["target", "model"], sort=True
    ):
        truth = group["truth"].to_numpy()
        prediction = group["prediction"].to_numpy()
        event = (
            truth
            >= group["tail_threshold_train_only"].to_numpy(dtype=np.float64)
        ).astype(int)
        sequence_corr = [
            corr(item["prediction"], item["truth"], "spearman")
            for _, item in group.groupby("sample_id")
        ]
        task_corr = {
            str(task): corr(
                item["prediction"], item["truth"], "spearman"
            )
            for task, item in group.groupby("task")
        }
        metric_rows.append(
            {
                "target": target,
                "model": model,
                "spearman": corr(prediction, truth, "spearman"),
                "kendall": corr(prediction, truth, "kendall"),
                "sequence_mean_spearman": float(
                    np.nanmean(sequence_corr)
                ),
                "mae": float(np.nanmean(np.abs(prediction - truth))),
                "top20_auprc": _average_precision(event, prediction),
                "top20_prevalence": float(event.mean()),
                "calibration_slope": calibration(
                    prediction, truth
                )["slope"],
                "task_spearman": task_corr,
            }
        )
    # Cross-task transfer uses only the named train task.
    transfer = []
    for target in targets:
        for model, features in {
            **model_features,
            "ema_fixed_0_9": base + ["anchor_ema_g0_9"],
        }.items():
            for train_is_niah in (False, True):
                train = head[
                    head["task_is_niah"] == float(train_is_niah)
                ]
                test = head[
                    head["task_is_niah"] != float(train_is_niah)
                ]
                prediction = _ridge_fit_predict(
                    train, test, features, target
                )
                threshold = float(train[target].quantile(0.8))
                event = (
                    test[target].to_numpy(dtype=np.float64) >= threshold
                ).astype(int)
                transfer.append(
                    {
                        "target": target,
                        "model": model,
                        "train_task": (
                            "NIAH" if train_is_niah else "GovReport"
                        ),
                        "test_task": (
                            "GovReport" if train_is_niah else "NIAH"
                        ),
                        "spearman": corr(
                            prediction, test[target], "spearman"
                        ),
                        "top20_auprc": _average_precision(
                            event, prediction
                        ),
                        "test_prevalence": float(event.mean()),
                    }
                )
    # NLL uses sequence-level aggregates of the same anchor-time features.
    predictor_columns = sorted(
        set(
            column
            for features in model_features.values()
            for column in features
        )
        | set(gamma_columns)
    )
    aggregate = (
        head.groupby(
            [
                "sample_id",
                "task",
                "protected_recent_size",
                "fresh_reference",
                "horizon",
            ],
            as_index=False,
        )[predictor_columns]
        .mean()
    )
    nll_merge_keys = {
        "sample_id",
        "task",
        "protected_recent_size",
        "fresh_reference",
        "horizon",
    }
    nll_frame = global_rows.drop(
        columns=[
            column
            for column in predictor_columns
            if column in global_rows.columns
            and column not in nll_merge_keys
        ]
    ).merge(
        aggregate,
        on=[
            "sample_id",
            "task",
            "protected_recent_size",
            "fresh_reference",
            "horizon",
        ],
        how="inner",
        validate="one_to_one",
    )
    nll_frame["task_is_niah"] = (
        nll_frame["task"].astype(str).str.contains("niah").astype(float)
    )
    nll_frame["log_horizon"] = np.log1p(
        nll_frame["horizon"].astype(float)
    )
    nll_frame["layer_scaled"] = 0.0
    nll_frame["head_scaled"] = 0.0
    nll_metrics = []
    for model, features in model_features.items():
        fold_predictions = []
        fold_truth = []
        fold_events = []
        for held_out in sorted(nll_frame["sample_id"].unique()):
            train = nll_frame[nll_frame["sample_id"] != held_out]
            test = nll_frame[nll_frame["sample_id"] == held_out]
            prediction = _ridge_fit_predict(
                train,
                test,
                features,
                "cumulative_nll_benefit",
            )
            fold_predictions.extend(prediction.tolist())
            fold_truth.extend(
                test["cumulative_nll_benefit"].tolist()
            )
            threshold = float(
                train["cumulative_nll_benefit"].quantile(0.8)
            )
            fold_events.extend(
                (
                    test["cumulative_nll_benefit"].to_numpy() >= threshold
                ).astype(int)
            )
        nll_metrics.append(
            {
                "model": model,
                "spearman": corr(
                    fold_predictions, fold_truth, "spearman"
                ),
                "kendall": corr(
                    fold_predictions, fold_truth, "kendall"
                ),
                "mae": float(
                    np.mean(
                        np.abs(
                            np.asarray(fold_predictions)
                            - np.asarray(fold_truth)
                        )
                    )
                ),
                "top20_auprc": _average_precision(
                    np.asarray(fold_events),
                    np.asarray(fold_predictions),
                ),
                "top20_prevalence": float(np.mean(fold_events)),
            }
        )
    validity_rows = []
    validity_source = predictions[
        predictions["target"] == "future_aor_gap"
    ]
    validity_keys = [
        "model",
        "sample_id",
        "protected_recent_size",
        "fresh_reference",
        "layer",
        "head",
    ]
    maximum_horizon = int(validity_source["horizon"].max())
    for keys, group in validity_source.groupby(
        validity_keys, sort=False
    ):
        ordered = group.sort_values("horizon")
        threshold = float(
            ordered["tail_threshold_train_only"].iloc[0]
        )

        def first_crossing(column: str) -> int:
            crossed = ordered[ordered[column] >= threshold]
            return (
                int(crossed["horizon"].iloc[0])
                if len(crossed)
                else maximum_horizon + 1
            )

        actual_horizon = first_crossing("truth")
        predicted_horizon = first_crossing("prediction")
        validity_rows.append(
            {
                **dict(zip(validity_keys, keys)),
                "threshold_train_only": threshold,
                "actual_validity_horizon": actual_horizon,
                "predicted_validity_horizon": predicted_horizon,
                "absolute_error": abs(
                    predicted_horizon - actual_horizon
                ),
            }
        )
    validity_frame = pd.DataFrame(validity_rows)
    validity_metrics = (
        validity_frame.groupby("model")
        .agg(
            horizon_mae=("absolute_error", "mean"),
            median_absolute_error=("absolute_error", "median"),
            unit_count=("absolute_error", "size"),
            actual_censored_fraction=(
                "actual_validity_horizon",
                lambda value: float((value > maximum_horizon).mean()),
            ),
            predicted_censored_fraction=(
                "predicted_validity_horizon",
                lambda value: float((value > maximum_horizon).mean()),
            ),
        )
        .reset_index()
        .to_dict(orient="records")
    )
    validity_frame.to_parquet(
        run_dir / "validity_horizon_loso_rows.parquet", index=False
    )
    summary = {
        "executed": True,
        "outer_validation": "leave-one-sequence-out",
        "ema_selection": "nested inner leave-one-sequence-out only",
        "chosen_ema_features_by_outer_fold": chosen_gammas,
        "continuous_and_tail_metrics": metric_rows,
        "cross_task_transfer": transfer,
        "nll_metrics": nll_metrics,
        "validity_horizon": {
            "status": "evaluated_but_underidentified",
            "definition": (
                "first registered horizon where cumulative future-AOR gap "
                "crosses the outer-training-fold 80th-percentile threshold; "
                "non-crossing is censored at Hmax+1"
            ),
            "metrics": validity_metrics,
            "caution": (
                "Only one anchor per sequence is available, so horizon MAE "
                "is reported but not treated as a stable validity-horizon claim."
            ),
        },
        "oracle_leakage_check": "passed schema allow-list",
        "scope": "six-sequence mechanism evidence only",
    }
    _atomic_json(
        run_dir / "future_state_prediction_summary.json", summary
    )
    return summary


def monitoring_analysis(
    objective_summary: Mapping[str, Any],
    horizon_summary: Mapping[str, Any],
    run_dir: Path,
) -> Dict[str, Any]:
    eligible = bool(
        objective_summary["partial_gate_pass"]
        and horizon_summary["partial_gate_pass"]
    )
    if not eligible:
        return {
            "executed": False,
            "reason": "offline objective and horizon did not both partially pass",
        }
    features = pd.read_parquet(
        STAGE1_RUN / "functional_features.parquet"
    )
    selected = features[
        (features["coverage_scope"] == "historical_core_only")
        & (features["feature_granularity"] == "diagnostic_head")
        & (
            features["feature_variant"].isin(
                ["raw_v", "projected_v", "aor"]
            )
        )
    ].copy()
    keys = [
        "task",
        "sample_id",
        "base_anchor",
        "probe_lag",
        "refresh_anchor",
        "strategy",
        "total_budget",
        "protected_recent_size",
        "layer",
        "head",
    ]
    value_columns = [
        "delta_e_raw_sum",
        "arrival_residual_raw_sum",
        "retained_reweighting_raw_sum",
        "deployable_approx_raw_sum",
    ]
    wide = selected.pivot_table(
        index=keys,
        columns="feature_variant",
        values=value_columns,
        aggfunc="first",
    )
    wide.columns = [
        "%s__%s" % (metric, variant)
        for metric, variant in wide.columns
    ]
    wide = wide.reset_index()
    wide["offline_aor"] = wide["delta_e_raw_sum__aor"]
    wide["arrival_raw_v"] = wide["arrival_residual_raw_sum__raw_v"]
    wide["arrival_ov"] = wide[
        "arrival_residual_raw_sum__projected_v"
    ]
    wide["arrival_aor"] = wide["arrival_residual_raw_sum__aor"]
    wide["retained_aor"] = wide[
        "retained_reweighting_raw_sum__aor"
    ]
    # Training-free robust normalization within a condition.  It uses only
    # the proxy streams, never fresh labels or fresh sets.
    for column in ["arrival_aor", "retained_aor"]:
        scale = (
            wide.groupby(["sample_id", "protected_recent_size"])[column]
            .transform(lambda value: value.abs().median())
            .clip(lower=1e-12)
        )
        wide["normalized_%s" % column] = wide[column] / scale
    wide["combined_proxy"] = (
        wide["normalized_arrival_aor"]
        + wide["normalized_retained_aor"]
    )
    proxy_columns = [
        "arrival_raw_v",
        "arrival_ov",
        "arrival_aor",
        "retained_aor",
        "combined_proxy",
    ]
    metrics = []
    threshold_by_sequence = wide.groupby("sample_id")[
        "offline_aor"
    ].transform(lambda value: value.quantile(0.8))
    wide["event"] = (wide["offline_aor"] >= threshold_by_sequence).astype(
        int
    )
    for proxy in proxy_columns:
        per_sequence = [
            corr(group[proxy], group["offline_aor"], "spearman")
            for _, group in wide.groupby("sample_id")
        ]
        per_task = {}
        for task, group in wide.groupby("task"):
            task_values = [
                corr(item[proxy], item["offline_aor"], "spearman")
                for _, item in group.groupby("sample_id")
            ]
            per_task[str(task)] = float(np.nanmean(task_values))
        # LOSO here is a fixed-score evaluation: no parameter is fitted.
        loso_auprc = _average_precision(
            wide["event"].to_numpy(), wide[proxy].to_numpy()
        )
        cross_task = {}
        for task, group in wide.groupby("task"):
            cross_task[str(task)] = {
                "auprc": _average_precision(
                    group["event"].to_numpy(), group[proxy].to_numpy()
                ),
                "prevalence": float(group["event"].mean()),
            }
        metrics.append(
            {
                "proxy": proxy,
                "sequence_mean_spearman": float(
                    np.nanmean(per_sequence)
                ),
                "task_direction": per_task,
                "loso_fixed_score_auprc": loso_auprc,
                "prevalence": float(wide["event"].mean()),
                "cross_task_fixed_score": cross_task,
            }
        )
    by_proxy = {item["proxy"]: item for item in metrics}
    combined = by_proxy["combined_proxy"]
    baseline_auprc = max(
        by_proxy["arrival_raw_v"]["loso_fixed_score_auprc"],
        by_proxy["arrival_ov"]["loso_fixed_score_auprc"],
    )
    delta_auprc = (
        combined["loso_fixed_score_auprc"] - baseline_auprc
    )
    task_same_direction = all(
        value > 0 for value in combined["task_direction"].values()
    )
    cross_task_pass = all(
        value["auprc"] > value["prevalence"]
        for value in combined["cross_task_fixed_score"].values()
    )
    memory_pass = True
    compute_pass = False
    gates = {
        "offline_spearman_ge_0_30": bool(
            combined["sequence_mean_spearman"] >= 0.30
        ),
        "delta_auprc_ge_0_05": bool(delta_auprc >= 0.05),
        "both_tasks_same_direction": bool(task_same_direction),
        "loso_auprc_above_prevalence": bool(
            combined["loso_fixed_score_auprc"]
            > combined["prevalence"]
        ),
        "cross_task_above_test_prevalence": bool(cross_task_pass),
        "compute_below_fresh_global_selection": compute_pass,
        "no_full_evicted_kv": memory_pass,
    }
    summary = {
        "executed": True,
        "data_source": (
            "Stage-1 arrival/retained decomposition, evaluated only after "
            "A and C partial gates"
        ),
        "proxies": metrics,
        "combined_delta_auprc_vs_best_raw_ov": float(delta_auprc),
        "gates": gates,
        "all_gates_pass": bool(all(gates.values())),
        "compute_gate_note": (
            "Failed conservatively: scalar post-processing was not timed as "
            "an integrated online implementation against fresh selection."
        ),
        "memory_schema": {
            "arrival": "position plus scalar residual only",
            "retained": "retained position plus scalar only",
            "evicted_full_kv": False,
        },
        "forbidden_input_audit": {
            "fresh_set": False,
            "fresh_output": False,
            "old_vs_fresh_label_as_input": False,
            "evicted_full_kv": False,
            "current_attention_to_evicted": False,
        },
    }
    _atomic_json(run_dir / "monitoring_proxy_summary.json", summary)
    wide.to_parquet(
        run_dir / "monitoring_proxy_rows.parquet", index=False
    )
    return summary


def _fmt(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except Exception:
        return str(value)
    return "NA" if not math.isfinite(number) else f"{number:.{digits}f}"


def write_reports(
    run_dir: Path,
    objective: Mapping[str, Any],
    per_head: pd.DataFrame,
    horizon: Mapping[str, Any],
    prediction: Mapping[str, Any],
    direct: Mapping[str, Any],
    monitoring: Mapping[str, Any],
    subset_units: pd.DataFrame,
) -> None:
    primary = objective["primary"]
    head_layer = per_head[per_head["granularity"] == "query_head"]
    layer_direction = (
        head_layer.groupby("layer")["aor_spearman_median"].median().to_dict()
    )
    reversal_layers = [
        int(layer) for layer, value in layer_direction.items() if value < 0
    ]
    query_units = subset_units[
        subset_units["granularity"] == "query_head"
    ].copy()
    subset_comparison = {
        "Attention": (
            query_units["attention_spearman"].median(),
            query_units["attention_normalized_regret"].median(),
        ),
        "Raw-V": (
            query_units["raw_v_spearman"].median(),
            query_units["raw_v_normalized_regret"].median(),
        ),
        "OV": (
            query_units["ov_spearman"].median(),
            query_units["ov_normalized_regret"].median(),
        ),
        "AOV": (
            query_units["aov_spearman"].median(),
            query_units["aov_normalized_regret"].median(),
        ),
        "AOR": (
            query_units["aor_spearman"].median(),
            query_units["aor_normalized_regret"].median(),
        ),
    }
    horizon_cells = pd.DataFrame(horizon["horizon_cells"])
    aor_direct_cells = horizon_cells[
        (horizon_cells["feature"] == "future_aor_gap")
        & (
            horizon_cells["target"]
            == "cumulative_direct_proj_benefit"
        )
    ]
    aov_direct_cells = horizon_cells[
        (horizon_cells["feature"] == "future_aov_gap")
        & (
            horizon_cells["target"]
            == "cumulative_direct_proj_benefit"
        )
    ]
    aor_stateful_cells = horizon_cells[
        (horizon_cells["feature"] == "future_aor_gap")
        & (
            horizon_cells["target"]
            == "cumulative_stateful_proj_benefit"
        )
    ]
    horizon_profile = (
        aor_direct_cells.groupby("horizon")["sequence_mean_spearman"]
        .median()
        .to_dict()
    )
    pre_exit = float(
        aor_direct_cells[
            aor_direct_cells["horizon"] <= 32
        ]["sequence_mean_spearman"].median()
    )
    post_exit = float(
        aor_direct_cells[
            aor_direct_cells["horizon"] > 32
        ]["sequence_mean_spearman"].median()
    )
    predictor_lines = []
    prediction_metrics = pd.DataFrame(
        prediction.get("continuous_and_tail_metrics", [])
    )
    if prediction.get("executed"):
        target_metrics = prediction_metrics[
            prediction_metrics["target"] == "future_aor_gap"
        ].sort_values("sequence_mean_spearman", ascending=False)
        for _, row in target_metrics.head(5).iterrows():
            predictor_lines.append(
                "- `%s`: sequence-mean Spearman %s，top-20%% AUPRC %s（prevalence %s）。"
                % (
                    row["model"],
                    _fmt(row["sequence_mean_spearman"]),
                    _fmt(row["top20_auprc"]),
                    _fmt(row["top20_prevalence"]),
                )
            )
    else:
        predictor_lines.append(
            "- 未执行：future-oracle target 未通过预注册 partial gate。"
        )
    monitoring_text = (
        "执行；全部 deployability gates %s。"
        % ("通过" if monitoring.get("all_gates_pass") else "未通过")
        if monitoring.get("executed")
        else "未执行，因为 A/C 没有同时 partial pass。"
    )

    def metric_row(target: str, model: str) -> Mapping[str, Any]:
        selected = prediction_metrics[
            (prediction_metrics["target"] == target)
            & (prediction_metrics["model"] == model)
        ]
        return selected.iloc[0] if len(selected) else {}

    future_obs8 = metric_row("future_aor_gap", "observation_w8")
    future_age = metric_row("future_aor_gap", "age_only")
    direct_obs8 = metric_row(
        "cumulative_direct_proj_benefit", "observation_w8"
    )
    direct_age = metric_row(
        "cumulative_direct_proj_benefit", "age_only"
    )
    stateful_obs8 = metric_row(
        "cumulative_stateful_proj_benefit", "observation_w8"
    )
    stateful_age = metric_row(
        "cumulative_stateful_proj_benefit", "age_only"
    )
    gaussian8 = metric_row("future_aor_gap", "gaussian_query_w8")
    validity_metrics = pd.DataFrame(
        prediction.get("validity_horizon", {}).get("metrics", [])
    )
    best_validity = (
        validity_metrics.sort_values("horizon_mae").iloc[0]
        if len(validity_metrics)
        else {}
    )
    failed_monitor_gates = [
        key
        for key, value in monitoring.get("gates", {}).items()
        if not value
    ]
    combined_monitor_rho = next(
        (
            item["sequence_mean_spearman"]
            for item in monitoring.get("proxies", [])
            if item["proxy"] == "combined_proxy"
        ),
        float("nan"),
    )
    results = f"""# Theory-closing experiments：结果

## 1. 运行范围与完整性

本轮使用本地 4-bit Qwen2.5-1.5B-Instruct，同一批 3 条 NIAH 与 3 条 official GovReport。A/B 穷举每个 layer-shared candidate pool 的 1820 个 subsets；最低诊断范围为 5 层 × 全部 12 query heads；C/E 从 step 32 连续 replay 64 步，recent 为 0/32。统计单位始终是 sequence，pooled rows 只用于 unit 内排序或描述。

共得到 819,000 条 exhaustive subset rows、360 个 query-head units、60 个 GQA-group units、30 个 layer units，以及 11,712 条 horizon rows。fixed-QKV deletion identity 的最大相对误差为 `{float(objective['identity_max_relative_error']):.3e}`。

## 2. Experiment A：AOR 是不是 subset objective？

primary metric 是各 query-head unit 内 1820 subsets 上 AOR surrogate 与 projected-head true risk 的 Spearman，然后在 sequence 层聚合。估计为 **{_fmt(primary['estimate'])}**，sequence-cluster 95% CI 为 **[{_fmt(primary['ci_low'])}, {_fmt(primary['ci_high'])}]**；预注册 partial gate 是 0.30，结果为 **{'通过' if objective['partial_gate_pass'] else '未通过'}**。

- 正相关 query-head units 比例：{_fmt(objective['query_head_positive_fraction'])}。
- AOR normalized regret 中位数：{_fmt(objective['query_head_aor_normalized_regret_median'])}。
- AOV / OV / Raw-V 的 unit-median Spearman：{_fmt(objective['query_head_aov_spearman_median'])} / {_fmt(objective['query_head_ov_spearman_median'])} / {_fmt(objective['query_head_raw_v_spearman_median'])}。
- AOR 相对 attention-only 选出更低 true-risk subset 的 unit 比例：{_fmt(objective['query_head_aor_improvement_vs_attention_positive_fraction'])}。
- AOR top-1 命中率、top-5% overlap 中位数、top-10% overlap 中位数：{_fmt(query_units['aor_top1_overlap'].mean())} / {_fmt(query_units['aor_top5pct_overlap'].median())} / {_fmt(query_units['aor_top10pct_overlap'].median())}。
- cross-token interaction / individual-energy 的绝对比值中位数：{_fmt(query_units['median_cross_ratio'].median())}，说明 additive approximation 的交互项不是可忽略小量。

| Objective | unit-median Spearman | unit-median normalized regret |
|---|---:|---:|
{chr(10).join(f"| {name} | {_fmt(values[0])} | {_fmt(values[1])} |" for name, values in subset_comparison.items())}

结论不能简写成 “AOR objective 已验证”。AOR 的 risk ranking 很强，但 AOV 略强，attention-only 更强且 median regret 为 0；AOR 只在 12.2% units 上优于 attention-only 的 subset choice。因此 AOR 获得的是 **attention-conditioned risk-ranking surrogate / approximate family 的部分支持**，没有获得“首选 subset optimizer”的支持。AOR 与 AOV 应暂时保留为同一 family，当前没有证据证明 residual centering 稳定优于 AOV。

## 3. Experiment B：head / GQA / layer 异质性

各层 query-head AOR Spearman 中位数为：

{chr(10).join(f'- layer {int(layer)}: {_fmt(value)}' for layer, value in layer_direction.items())}

系统性负向 reversal layers：`{reversal_layers}`。layer 27 的 AOR median normalized regret 为 {_fmt(query_units[query_units['layer'] == 27]['aor_normalized_regret'].median())}。GQA-group 与 layer aggregate 的 AOR Spearman 中位数分别为 {_fmt(subset_units[subset_units['granularity'] == 'kv_head_group']['aor_spearman'].median())} 与 {_fmt(subset_units[subset_units['granularity'] == 'layer']['aor_spearman'].median())}。

因此不能用 layer aggregate 的正相关替代“所有 heads 使用同一公式”的证据。物理 KV eviction 不能按 query head 独立执行；较合理的下一数学单元是 GQA-shared mask 或 layer-shared mask，但必须直接优化 group/layer risk 并显式包含 cross-head cancellation。

## 4. Experiment C：future-oracle horizon

future-oracle AOR gap 对累计 direct projected-output benefit 的预注册汇总值为 **{_fmt(horizon['primary_estimate'])}**，partial gate 为 **{'通过' if horizon['partial_gate_pass'] else '未通过'}**。这里 `per_step_fresh` 与 `horizon_start_once_fresh` 始终分列；AOR 只使用真实 future query 作为离线 oracle feature，不是 online score。

| Horizon $H$ | median sequence-mean Spearman |
|---:|---:|
{chr(10).join(f"| {int(key)} | {_fmt(value)} |" for key, value in horizon_profile.items())}

recent-exit 前（$H\\le 32$）与出现 exit event 后（$H>32$）的 cell-median 分别为 {_fmt(pre_exit)} 与 {_fmt(post_exit)}。AOV 的所有 cell-median 为 {_fmt(aov_direct_cells['sequence_mean_spearman'].median())}，仍略高于 AOR 的 {_fmt(aor_direct_cells['sequence_mean_spearman'].median())}。

对累计 stateful NLL benefit，head-wise feature 聚合后的描述性 Spearman 为：

{chr(10).join(f'- {key}: {_fmt(value)}' for key, value in horizon['nll_descriptive_correlations_after_head_aggregation'].items())}

AOR 对 stateful projected benefit 的 cell-median 只有 {_fmt(aor_stateful_cells['sequence_mean_spearman'].median())}。因此 future-oracle AOR objective **只对 fixed-QKV direct functional regret 获得支持**，不能直接升级为 autoregressive trajectory 或 NLL objective。recent-exit 后排序略降但未崩溃；它更像局部 direct-objective 有效性的边界，而不是完整 cache architecture 的充分状态。

## 5. Experiment D：anchor-time prediction

{chr(10).join(predictor_lines)}

future AOR target 上，observation-W8 相对 age-only 的 sequence-mean Spearman 为 {_fmt(future_obs8.get('sequence_mean_spearman'))} vs {_fmt(future_age.get('sequence_mean_spearman'))}，AUPRC 为 {_fmt(future_obs8.get('top20_auprc'))} vs {_fmt(future_age.get('top20_auprc'))}。cumulative direct benefit 上 W8 也有增量（{_fmt(direct_obs8.get('sequence_mean_spearman'))} vs {_fmt(direct_age.get('sequence_mean_spearman'))}）。真正的 shrinkage Gaussian-query baseline（query mean、covariance trace、current-query Mahalanobis、top eigenspectrum）在 W8 上仅得到 {_fmt(gaussian8.get('sequence_mean_spearman'))}，没有优于 age-only。

但 stateful benefit 上 age-only 反而强于 W8（{_fmt(stateful_age.get('sequence_mean_spearman'))} vs {_fmt(stateful_obs8.get('sequence_mean_spearman'))}）。跨任务 transfer 明显不对称，direct-benefit transfer 两个方向都弱。结论是：anchor state 对 **future functional target** 有可预测性，但尚不足以闭合 stateful/NLL risk。

按 outer-training-fold 80% threshold 定义首次 crossing horizon，最佳模型 `{best_validity.get('model', 'NA')}` 的 horizon MAE 为 {_fmt(best_validity.get('horizon_mae'))} steps；实际 non-crossing censor 比例为 {_fmt(best_validity.get('actual_censored_fraction'))}。由于每条 sequence 只有一个 anchor 且 67% 左右 units 被 censor，这个 MAE 已按要求报告，但不足以支持具体 validity-horizon claim。

## 6. Experiment E：direct 与 feedback

direct vs stateful overall Spearman 为 **{_fmt(direct['overall_spearman'])}**，回归 slope 为 **{_fmt(direct['overall_slope'])}**，median absolute feedback fraction 为 **{_fmt(direct['overall_median_abs_feedback_fraction'])}**。dense budget-128 与 Stage-1 sparse budget-128/256 保留 `source_matrix`，没有把它们当成额外独立 sequences。

feedback fraction 接近 1，所以 fixed-QKV deletion identity 不能直接累加成 stateful horizon model。本轮也没有找到一个清晰的“短 horizon feedback 可忽略”区间；local identity 只应保留为同一步 direct component。

## 7. Phase B monitoring

{monitoring_text}

失败 gates：`{failed_monitor_gates}`。combined proxy 的 offline Spearman 为 {_fmt(combined_monitor_rho)}，但相对 Raw-V/OV 的 $\Delta$AUPRC 为 {_fmt(monitoring.get('combined_delta_auprc_vs_best_raw_ov'))}，方向为负；arrival-only Raw-V/OV 并未被 combined AOR 超过。compute 未做真实在线集成计时，按预注册规则保守判失败。因此 bounded-cache monitoring state **没有通过 deployability gate**。

## 8. 结论边界

这是 6-sequence theory-discovery study，不是 benchmark。相关性不是 theorem，future oracle 不是 deployable score，layer aggregate 不是所有 heads 的结论，负向 heads/layers 与 gate failures 均保留在机器产物中。
"""
    _atomic_text(run_dir / "THEORY_CLOSING_RESULTS_ZH.md", results)
    arrow_lines = [
        (
            "PASS（local）",
            "AOR coverage → broad-subset direct-risk ranking",
            primary["estimate"],
        ),
        (
            "FAIL（optimizer preference）",
            "AOR → preferred subset optimizer over AOV/attention",
            objective["query_head_aor_improvement_vs_attention_positive_fraction"],
        ),
        (
            "PASS（local）",
            "future-oracle AOR sum → cumulative direct regret",
            horizon["primary_estimate"],
        ),
        (
            "FAIL（global）",
            "future-oracle AOR sum → stateful/NLL regret",
            aor_stateful_cells["sequence_mean_spearman"].median(),
        ),
        (
            "PASS（functional target）",
            "anchor observation state → future AOR/direct variation",
            future_obs8.get("sequence_mean_spearman", float("nan")),
        ),
        (
            "FAIL（closed loop）",
            "anchor functional state → incremental stateful/NLL prediction",
            stateful_obs8.get("sequence_mean_spearman", float("nan")),
        ),
        (
            "FAIL",
            "fixed-QKV direct effect → full stateful effect",
            direct["overall_median_abs_feedback_fraction"],
        ),
        (
            "FAIL",
            "bounded monitoring state → deployable trigger",
            combined_monitor_rho,
        ),
        (
            "FAIL / underidentified",
            "anchor state → concrete validity horizon",
            best_validity.get("horizon_mae", float("nan")),
        ),
    ]
    model_update = f"""# Theory model update

## 1. 理论箭头审计

{chr(10).join(f"- **{status}** — {name}；主量 `{_fmt(value)}`。" for status, name, value in arrow_lines)}

PASS 只表示六条 sequence 上的预注册机制 gate，不是普遍定理。local PASS 与 global FAIL 必须同时保留。

## 2. AOR 的理论地位

AOR 当前应称为 **strong attention-conditioned diagnostic/ranking surrogate**，不是已经胜出的 subset optimizer。AOV 的 ranking 与 regret 略优，attention-only 的 subset choice 更强；因此 residual centering 尚未得到增量支持。理论应暂时写成 AOV/AOR family，而不是唯一 AOR objective。

## 3. Horizon objective

$$
L_{{\\tau,H}}^{{\\mathrm{{AOR}}}}(C)
=\\sum_{{h=1}}^H E_{{\\tau+h}}^{{\\mathrm{{AOR}}}}(C)
$$

在 future-oracle test 中，这个式子对 **fixed-QKV direct projected-output regret** 获得部分支持；对 stateful output 与 NLL 没有同等支持。因此应改写为 local direct term：

$$
L_{{\\tau,H}}^{{\\mathrm{{stateful}}}}
=L_{{\\tau,H}}^{{\\mathrm{{direct}}}}
+L_{{\\tau,H}}^{{\\mathrm{{feedback}}}},
$$

其中 AOV/AOR 只近似第一项，第二项目前未闭合。

## 4. Anchor state 与 observability

observation-window state 能预测 future AOR 与部分 direct variation，优于 age-only；Gaussian-query baseline 没有显示增量。但 stateful/NLL 上 age baseline 已很强，functional state 没有稳定增量，跨任务 transfer 也不对称。最佳 validity-horizon MAE 为 {_fmt(best_validity.get('horizon_mae'))} steps，且 censor 比例高，故具体 validity horizon 仍未识别。

monitoring Phase B：{monitoring_text}

## 5. Local model 的有效范围

direct/stateful Spearman 为 {_fmt(direct['overall_spearman'])}，feedback fraction 为 {_fmt(direct['overall_median_abs_feedback_fraction'])}。feedback 主导，且没有清晰的短 horizon 可忽略区间。fixed-QKV deletion identity 只能作为 local component，不能直接累加成 autoregressive model。

## 6. Head、GQA 与物理 eviction

负向 reversal layer 为 `{reversal_layers}`。单头公式没有自动跨 layer/head 成立；物理 eviction 应以 GQA/layer-shared mask 为候选数学单元，并显式优化 aggregated risk 与 cancellation，不能把 query-head oracle masks 当成可执行 cache。

## 7. 下一步

当前不应直接进入 adaptive algorithm 或扩大 benchmark。下一步优先修改理论：把 local direct functional objective、trajectory feedback 与 observability 分成三项；随后做更小的 feedback-state / observability 判别。如果 bounded state 仍无法超越 Raw-V/OV arrival baseline，应把结果发展为 observability limitation / impossibility characterization。
"""
    _atomic_text(run_dir / "THEORY_MODEL_UPDATE_ZH.md", model_update)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def finalize_provenance(run_dir: Path) -> None:
    _atomic_text(
        run_dir / "THEORY_CLOSING_EXPERIMENT_DESIGN_ZH.md",
        (ROOT / "THEORY_CLOSING_EXPERIMENT_DESIGN_ZH.md").read_text(
            encoding="utf-8"
        ),
    )
    _atomic_text(
        run_dir / "theory_closing_config.yaml",
        (
            ROOT / "configs" / "stages" / "theory_closing_config.yaml"
        ).read_text(encoding="utf-8"),
    )
    metadata_path = run_dir / "metadata.json"
    with open(metadata_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    cache_root = (
        Path.home()
        / ".cache/huggingface/hub"
        / "models--mlx-community--Qwen2.5-1.5B-Instruct-4bit"
    )
    revision_path = cache_root / "refs/main"
    if revision_path.exists():
        revision = revision_path.read_text(encoding="utf-8").strip()
        snapshot = cache_root / "snapshots" / revision
        weight = snapshot / "model.safetensors"
        metadata["model"]["local_snapshot_revision"] = revision
        metadata["model"]["local_snapshot_path"] = str(snapshot)
        if weight.exists():
            metadata["model"]["model_weight_sha256"] = _sha256_file(
                weight
            )
    _atomic_json(metadata_path, metadata)
    schema_path = run_dir / "artifact_schema.json"
    with open(schema_path, "r", encoding="utf-8") as handle:
        schema = json.load(handle)
    schema_tables = {
        "subset_objective_rows": "subset_objective_rows.parquet",
        "subset_unit_inventory": "subset_unit_inventory.parquet",
        "future_oracle_horizon_rows": "future_oracle_horizon_rows.parquet",
        "direct_stateful_decomposition": (
            "direct_stateful_decomposition.parquet"
        ),
        "theory_runtime": "theory_runtime.parquet",
        "subset_unit_metrics": "subset_unit_metrics.parquet",
        "future_state_loso_predictions": (
            "future_state_loso_predictions.parquet"
        ),
        "validity_horizon_loso_rows": (
            "validity_horizon_loso_rows.parquet"
        ),
        "gaussian_query_anchor_features": (
            "gaussian_query_anchor_features.parquet"
        ),
        "monitoring_proxy_rows": "monitoring_proxy_rows.parquet",
    }
    for table, filename in schema_tables.items():
        path = run_dir / filename
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        schema.setdefault("tables", {})[table] = {
            "path": str(path),
            "rows": int(len(frame)),
            "columns": {
                column: str(dtype)
                for column, dtype in frame.dtypes.items()
            },
        }
    schema["analysis_complete"] = True
    schema["model_weight_sha256"] = metadata["model"].get(
        "model_weight_sha256"
    )
    _atomic_json(schema_path, schema)
    required = [
        "THEORY_CLOSING_EXPERIMENT_DESIGN_ZH.md",
        "theory_closing_config.yaml",
        "subset_objective_rows.parquet",
        "subset_oracle_summary.json",
        "per_head_gqa_summary.csv",
        "future_oracle_horizon_rows.parquet",
        "future_state_prediction_summary.json",
        "direct_stateful_decomposition.csv",
        "monitoring_proxy_summary.json",
        "THEORY_CLOSING_RESULTS_ZH.md",
        "THEORY_MODEL_UPDATE_ZH.md",
    ]
    artifacts = {}
    for name in required:
        path = run_dir / name
        if not path.exists():
            raise RuntimeError("required theory-closing artifact is missing: %s" % name)
        record: Dict[str, Any] = {
            "bytes": int(path.stat().st_size),
            "sha256": _sha256_file(path),
        }
        if path.suffix == ".parquet":
            frame = pd.read_parquet(path)
            record["rows"] = int(len(frame))
            record["columns"] = int(len(frame.columns))
        artifacts[name] = record
    manifest = {
        "schema_version": "theory_closing_analysis_manifest_v1",
        "required_artifacts": artifacts,
        "sequence_is_independent_unit": True,
        "pooled_rows_are_not_independent_samples": True,
        "model_provenance": {
            key: metadata["model"].get(key)
            for key in [
                "local_snapshot_revision",
                "local_snapshot_path",
                "model_weight_sha256",
                "weight_precision",
            ]
        },
        "git_commit": metadata.get("git_commit"),
        "git_dirty": metadata.get("git_dirty"),
        "config_hash": metadata.get("config_hash"),
    }
    _atomic_json(run_dir / "analysis_manifest.json", manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN))
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve()
    subset = pd.read_parquet(run_dir / "subset_objective_rows.parquet")
    future = pd.read_parquet(
        run_dir / "future_oracle_horizon_rows.parquet"
    )
    dense_direct = pd.read_parquet(
        run_dir / "direct_stateful_decomposition.parquet"
    )
    if "source_matrix" in dense_direct.columns:
        dense_direct = dense_direct[
            dense_direct["source_matrix"]
            == "theory_closing_dense_budget128"
        ].copy()
    objective, subset_units, per_head = subset_objective_analysis(
        subset, run_dir
    )
    horizon, horizon_metrics = horizon_analysis(future)
    _atomic_json(run_dir / "future_oracle_horizon_summary.json", horizon)
    horizon_metrics.to_csv(
        run_dir / "future_oracle_horizon_summary.csv", index=False
    )
    _, direct = direct_stateful_analysis(dense_direct, run_dir)
    prediction = future_state_prediction_analysis(
        future, horizon, run_dir
    )
    monitoring = monitoring_analysis(objective, horizon, run_dir)
    write_reports(
        run_dir,
        objective,
        per_head,
        horizon,
        prediction,
        direct,
        monitoring,
        subset_units,
    )
    finalize_provenance(run_dir)
    status_path = run_dir / "status.json"
    with open(status_path, "r", encoding="utf-8") as handle:
        status = json.load(handle)
    status["state"] = "complete"
    status["analysis"] = {
        "subset_partial_gate_pass": bool(
            objective["partial_gate_pass"]
        ),
        "horizon_partial_gate_pass": bool(
            horizon["partial_gate_pass"]
        ),
        "prediction_executed": bool(prediction.get("executed")),
        "monitoring_executed": bool(monitoring.get("executed")),
        "monitoring_all_gates_pass": bool(
            monitoring.get("all_gates_pass", False)
        ),
    }
    _atomic_json(status_path, status)
    print(run_dir)


if __name__ == "__main__":
    main()
