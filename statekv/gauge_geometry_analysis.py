"""Sequence-grouped Stage-A analysis and strict gate for gauge geometry."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from statekv.config import DiscoveryConfig
from statekv.output_sensitivity_analysis import (
    _sequence_cluster_bootstrap,
    top1_regret,
    topk_overlap,
)
from statekv.theory_closing import _atomic_frame
from statekv.trajectory_analysis import atomic_json


FIXED_GEOMETRIES: Dict[str, str] = {
    "G0_RAW_GLOBAL": "g0_global_bound",
    "G1_UNIFORM_CENTERED": "g1_centered_global_bound",
    "G1P_PROBABILITY_VARIANCE": "g1p_probability_variance",
    "G2_BASE_FISHER": "g2_base_fisher",
    "G3_MIDPOINT_FISHER": "g3_midpoint_fisher",
    "G4_GL2_ORACLE": "g4_gl2",
    "G4_GL3_ORACLE": "g4_gl3",
    "G4_GL5_ORACLE": "g4_gl5",
    "G4_SIMPSON9_ORACLE": "g4_simpson9",
    "G4B_RANGE_BOUND": "g4b_range_bound",
    "G7_ORDER2": "g7_order2",
    "G7_ORDER3": "g7_order3",
    "G7_ORDER4": "g7_order4",
}


def task_bucket(task: str) -> str:
    return "GovReport" if "gov" in str(task).lower() else "NIAH"


def finite_correlation(
    x: Sequence[float], y: Sequence[float], kind: str
) -> float:
    left = np.asarray(x, dtype=np.float64)
    right = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(left) & np.isfinite(right)
    left, right = left[finite], right[finite]
    if (
        len(left) < 2
        or np.allclose(left, left[0])
        or np.allclose(right, right[0])
    ):
        return 0.0
    if kind == "spearman":
        return float(stats.spearmanr(left, right).statistic)
    if kind == "kendall":
        return float(stats.kendalltau(left, right).statistic)
    return float(stats.pearsonr(left, right).statistic)


def nonnegative_calibration(
    prediction: Sequence[float], target: Sequence[float]
) -> Tuple[float, float]:
    x = np.asarray(prediction, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if len(x) < 2 or np.allclose(x, x[0]):
        return 0.0, max(float(np.mean(y)) if len(y) else 0.0, 0.0)
    centered_x = x - float(x.mean())
    slope = max(
        float(
            np.dot(centered_x, y - float(y.mean()))
            / max(np.dot(centered_x, centered_x), 1.0e-30)
        ),
        0.0,
    )
    intercept = float(y.mean() - slope * x.mean())
    return slope, intercept


def geometry_columns(rows: pd.DataFrame) -> Dict[str, str]:
    result = dict(FIXED_GEOMETRIES)
    for column in rows.columns:
        if column.startswith(("g5a_k", "g5b_k", "g5c_k")):
            result[column.upper()] = column
        elif column.startswith(("g6_two_", "g6_collapse_")):
            result[column.upper()] = column
        elif column.startswith("g4b_truncated_"):
            result[column.upper()] = column
    return result


def _r2(target: np.ndarray, prediction: np.ndarray) -> float:
    residual = float(np.square(target - prediction).sum())
    total = float(np.square(target - float(target.mean())).sum())
    return float(1.0 - residual / max(total, 1.0e-30))


def _pointwise_record(
    current: pd.DataFrame,
    family: str,
    prediction_column: str,
    prediction: np.ndarray,
    calibrated: bool,
) -> Dict[str, Any]:
    target = current["exact_kl"].to_numpy(dtype=np.float64)
    predicted = np.maximum(np.asarray(prediction, dtype=np.float64), 0.0)
    absolute = np.abs(predicted - target)
    relative = absolute / np.maximum(target, 1.0e-12)
    ratio = np.maximum(
        (predicted + 1.0e-12) / (target + 1.0e-12),
        (target + 1.0e-12) / (predicted + 1.0e-12),
    )
    return {
        "family": family,
        "prediction_column": prediction_column,
        "task_bucket": str(current["task_bucket"].iloc[0]),
        "calibrated_oof": bool(calibrated),
        "rows": int(len(current)),
        "spearman": finite_correlation(predicted, target, "spearman"),
        "pearson": finite_correlation(predicted, target, "pearson"),
        "r2": _r2(target, predicted),
        "median_relative_error": float(np.median(relative)),
        "p90_relative_error": float(np.quantile(relative, 0.90)),
        "maximum_relative_error": float(np.max(relative)),
        "median_symmetric_ratio": float(np.median(ratio)),
        "p90_symmetric_ratio": float(np.quantile(ratio, 0.90)),
        "negative_prediction_fraction": float(
            (
                np.asarray(prediction, dtype=np.float64) < 0.0
            ).mean()
        ),
    }


def pointwise_summary(
    rows: pd.DataFrame, families: Mapping[str, str]
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    sequences = sorted(rows["sample_id"].astype(str).unique())
    calibration_rows: List[Dict[str, Any]] = []
    records: List[Dict[str, Any]] = []
    oof_summary_rows: List[Dict[str, Any]] = []
    sequence_values = rows["sample_id"].astype(str).to_numpy()
    for family, column in families.items():
        oof_prediction = np.full(len(rows), np.nan, dtype=np.float64)
        for held_out in sequences:
            training_mask = sequence_values != held_out
            test_mask = ~training_mask
            slope, intercept = nonnegative_calibration(
                rows.loc[training_mask, column],
                rows.loc[training_mask, "exact_kl"],
            )
            oof_prediction[test_mask] = np.maximum(
                slope
                * rows.loc[test_mask, column].to_numpy(dtype=np.float64)
                + intercept,
                0.0,
            )
            calibration_rows.append(
                {
                    "held_out_sequence": held_out,
                    "family": family,
                    "slope": float(slope),
                    "intercept": float(intercept),
                    "training_sequence_count": int(
                        rows.loc[training_mask, "sample_id"].nunique()
                    ),
                }
            )
        for task in ("GovReport", "NIAH"):
            task_mask = rows["task_bucket"].to_numpy() == task
            current = rows.loc[task_mask]
            records.append(
                _pointwise_record(
                    current,
                    family,
                    column,
                    current[column].to_numpy(dtype=np.float64),
                    False,
                )
            )
            records.append(
                _pointwise_record(
                    current,
                    family,
                    column,
                    oof_prediction[task_mask],
                    True,
                )
            )
        oof_summary_rows.append(
            {
                "family": family,
                "finite_oof_rows": int(np.isfinite(oof_prediction).sum()),
                "total_rows": int(len(oof_prediction)),
            }
        )
    detail_splits: List[Dict[str, Any]] = []
    primary = {
        key: value
        for key, value in families.items()
        if key
        in {
            "G0_RAW_GLOBAL",
            "G1_UNIFORM_CENTERED",
            "G2_BASE_FISHER",
            "G3_MIDPOINT_FISHER",
            "G4_GL3_ORACLE",
            "G4_GL5_ORACLE",
            "G4_SIMPSON9_ORACLE",
            "G7_ORDER3",
            "G7_ORDER4",
        }
    }
    for family, column in primary.items():
        for split_name, split_column in (
            ("anchor", "anchor"),
            ("horizon_offset", "horizon_offset"),
        ):
            for keys, current in rows.groupby(
                ["task_bucket", split_column], sort=True
            ):
                detail_splits.append(
                    {
                        "family": family,
                        "split": split_name,
                        "task_bucket": str(keys[0]),
                        "split_value": int(keys[1]),
                        "rows": int(len(current)),
                        "spearman": finite_correlation(
                            current[column],
                            current["exact_kl"],
                            "spearman",
                        ),
                        "median_symmetric_ratio": float(
                            np.median(
                                np.maximum(
                                    (
                                        current[column].to_numpy()
                                        + 1.0e-12
                                    )
                                    / (
                                        current["exact_kl"].to_numpy()
                                        + 1.0e-12
                                    ),
                                    (
                                        current["exact_kl"].to_numpy()
                                        + 1.0e-12
                                    )
                                    / (
                                        np.maximum(
                                            current[column].to_numpy(), 0.0
                                        )
                                        + 1.0e-12
                                    ),
                                )
                            )
                        ),
                    }
                )
    summary = {
        "task_summary": records,
        "anchor_horizon_split": detail_splits,
        "calibration_coefficients": calibration_rows,
        "calibration_protocol": {
            "outer_unit": "sequence",
            "training_sequences_per_fold": 23,
            "held_out_sequences_per_fold": 1,
            "task_id_used": False,
        },
    }
    return summary, pd.DataFrame(oof_summary_rows)


def build_segment_rows(
    rows: pd.DataFrame,
    families: Mapping[str, str],
    horizons: Sequence[int],
) -> pd.DataFrame:
    keys = [
        "sample_id",
        "task",
        "task_bucket",
        "anchor",
        "candidate_id",
        "candidate_index",
        "candidate_source",
    ]
    output: List[pd.DataFrame] = []
    value_columns = ["exact_kl"] + list(families.values())
    for horizon in horizons:
        current = rows[rows["horizon_offset"] <= int(horizon)]
        segment = (
            current.groupby(keys, as_index=False)[value_columns]
            .sum()
            .assign(horizon=int(horizon))
        )
        output.append(segment)
    return pd.concat(output, ignore_index=True)


def action_unit_metrics(
    segment: pd.DataFrame,
    family_columns: Mapping[str, str],
) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    group_columns = [
        "sample_id",
        "task",
        "task_bucket",
        "anchor",
        "horizon",
    ]
    for keys, current in segment.groupby(group_columns, sort=True):
        truth = current["exact_kl"].to_numpy(dtype=np.float64)
        span = max(float(truth.max() - truth.min()), 1.0e-12)
        for family, column in family_columns.items():
            prediction = current[column].to_numpy(dtype=np.float64)
            pair_truth: List[float] = []
            pair_prediction: List[float] = []
            for left in range(len(truth)):
                for right in range(left + 1, len(truth)):
                    true_delta = float(truth[left] - truth[right])
                    predicted_delta = float(
                        prediction[left] - prediction[right]
                    )
                    if abs(true_delta) > 1.0e-15:
                        pair_truth.append(true_delta)
                        pair_prediction.append(predicted_delta)
            sign_accuracy = float(
                np.mean(
                    np.sign(pair_truth) == np.sign(pair_prediction)
                )
            )
            selected = int(np.argmin(prediction))
            records.append(
                {
                    "sample_id": str(keys[0]),
                    "task": str(keys[1]),
                    "task_bucket": str(keys[2]),
                    "anchor": int(keys[3]),
                    "horizon": int(keys[4]),
                    "family": family,
                    "candidate_spearman": finite_correlation(
                        prediction, truth, "spearman"
                    ),
                    "candidate_kendall": finite_correlation(
                        prediction, truth, "kendall"
                    ),
                    "pairwise_sign_accuracy": sign_accuracy,
                    "top1_regret": top1_regret(truth, prediction),
                    "normalized_regret": float(
                        (truth[selected] - truth.min()) / span
                    ),
                    "top3_overlap": topk_overlap(truth, prediction, 3),
                    "selected_candidate_id": str(
                        current.iloc[selected]["candidate_id"]
                    ),
                    "oracle_candidate_id": str(
                        current.iloc[int(np.argmin(truth))]["candidate_id"]
                    ),
                    "candidate_count": int(len(current)),
                }
            )
    return pd.DataFrame(records)


def _best_training_variant(
    action: pd.DataFrame,
    held_out: str,
    candidates: Sequence[str],
) -> str:
    training = action[
        (action["sample_id"] != held_out)
        & action["family"].isin(candidates)
    ]
    scored = []
    for family in candidates:
        current = training[training["family"] == family]
        task_medians = current.groupby("task_bucket")[
            "candidate_spearman"
        ].median()
        score = (
            float(task_medians.min())
            if len(task_medians) == 2
            else -float("inf")
        )
        scored.append((score, family))
    return max(scored, key=lambda item: (item[0], item[1]))[1]


def append_oof_selected_variants(
    segment: pd.DataFrame,
    action: pd.DataFrame,
    families: Dict[str, str],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    groups: Dict[str, List[str]] = {
        "G5A_OOF_SELECTED": sorted(
            key for key in families if key.startswith("G5A_K")
        ),
        "G5B_OOF_SELECTED": sorted(
            key for key in families if key.startswith("G5B_K")
        ),
        "G5C_OOF_SELECTED": sorted(
            key for key in families if key.startswith("G5C_K")
        ),
        "G6_OOF_SELECTED": sorted(
            key for key in families if key.startswith("G6_")
        ),
    }
    selections: Dict[str, Dict[str, str]] = {}
    selected_frames: List[pd.DataFrame] = []
    sequences = sorted(segment["sample_id"].unique())
    for selected_name, candidates in groups.items():
        selections[selected_name] = {}
        for held_out in sequences:
            chosen = _best_training_variant(action, held_out, candidates)
            selections[selected_name][str(held_out)] = chosen
            column = families[chosen]
            current = segment[segment["sample_id"] == held_out].copy()
            current[selected_name] = current[column]
            selected_frames.append(
                current[
                    [
                        "sample_id",
                        "task",
                        "task_bucket",
                        "anchor",
                        "candidate_id",
                        "candidate_index",
                        "candidate_source",
                        "horizon",
                        "exact_kl",
                        selected_name,
                    ]
                ]
            )
    for selected_name in groups:
        combined = pd.concat(
            [
                frame
                for frame in selected_frames
                if selected_name in frame.columns
            ],
            ignore_index=True,
        )
        selected_action = action_unit_metrics(
            combined, {selected_name: selected_name}
        )
        action = pd.concat([action, selected_action], ignore_index=True)
    return segment, action, selections


def append_pointwise_oof_selected(
    pointwise: Dict[str, Any],
    rows: pd.DataFrame,
    selections: Mapping[str, Mapping[str, str]],
    families: Mapping[str, str],
) -> None:
    for selected_name, by_sequence in selections.items():
        selected_parts = []
        for sequence, chosen in by_sequence.items():
            current = rows[rows["sample_id"].astype(str) == str(sequence)].copy()
            current["_selected_prediction"] = current[families[chosen]]
            selected_parts.append(current)
        selected = pd.concat(selected_parts, ignore_index=True)
        for task in ("GovReport", "NIAH"):
            current = selected[selected["task_bucket"] == task]
            pointwise["task_summary"].append(
                _pointwise_record(
                    current,
                    selected_name,
                    "outer-fold-training-selected",
                    current["_selected_prediction"].to_numpy(),
                    False,
                )
            )


def summarized_action(
    action: pd.DataFrame, cfg: DiscoveryConfig
) -> Dict[str, Any]:
    metric_columns = [
        "candidate_spearman",
        "candidate_kendall",
        "pairwise_sign_accuracy",
        "top1_regret",
        "normalized_regret",
        "top3_overlap",
    ]
    task_records = (
        action.groupby(["family", "task_bucket"], as_index=False)[
            metric_columns
        ]
        .median()
        .to_dict("records")
    )
    pooled = (
        action.groupby("family", as_index=False)[metric_columns]
        .median()
        .to_dict("records")
    )
    anchor = (
        action.groupby(
            ["family", "task_bucket", "anchor"], as_index=False
        )[metric_columns]
        .median()
        .to_dict("records")
    )
    horizon = (
        action.groupby(
            ["family", "task_bucket", "horizon"], as_index=False
        )[metric_columns]
        .median()
        .to_dict("records")
    )
    bootstrap: Dict[str, Any] = {}
    primary = {
        "G0_RAW_GLOBAL",
        "G1_UNIFORM_CENTERED",
        "G2_BASE_FISHER",
        "G3_MIDPOINT_FISHER",
        "G4_GL5_ORACLE",
        "G5C_OOF_SELECTED",
        "G6_OOF_SELECTED",
    }
    for family in sorted(primary & set(action["family"])):
        for task in ("GovReport", "NIAH"):
            current = action[
                (action["family"] == family)
                & (action["task_bucket"] == task)
            ]
            bootstrap["%s:%s" % (family, task)] = {
                metric: _sequence_cluster_bootstrap(
                    current,
                    metric,
                    "sample_id",
                    int(cfg.runtime.bootstrap_samples),
                    int(cfg.runtime.seed),
                )
                for metric in (
                    "candidate_spearman",
                    "normalized_regret",
                    "top1_regret",
                )
            }
    return {
        "task_split": json.loads(pd.DataFrame(task_records).to_json(orient="records")),
        "pooled": json.loads(pd.DataFrame(pooled).to_json(orient="records")),
        "anchor_split": json.loads(pd.DataFrame(anchor).to_json(orient="records")),
        "horizon_split": json.loads(pd.DataFrame(horizon).to_json(orient="records")),
        "sequence_cluster_bootstrap_95ci": bootstrap,
        "bootstrap_unit": "sequence",
    }


def quadrature_summary(rows: pd.DataFrame) -> Dict[str, Any]:
    records = []
    for family, column in (
        ("G4_GL2", "g4_gl2"),
        ("G4_GL3", "g4_gl3"),
        ("G4_GL5", "g4_gl5"),
        ("G4_SIMPSON9", "g4_simpson9"),
    ):
        for task, current in rows.groupby("task_bucket"):
            relative = np.abs(
                current[column].to_numpy()
                - current["exact_kl"].to_numpy()
            ) / np.maximum(current["exact_kl"].to_numpy(), 1.0e-12)
            records.append(
                {
                    "family": family,
                    "task_bucket": task,
                    "rows": int(len(current)),
                    "median_relative_error": float(np.median(relative)),
                    "p90_relative_error": float(np.quantile(relative, 0.90)),
                    "maximum_relative_error": float(np.max(relative)),
                    "spearman": finite_correlation(
                        current[column], current["exact_kl"], "spearman"
                    ),
                }
            )
    return {
        "quadrature": records,
        "exact_identity_max_abs_error": float(
            rows["kl_cumulant_identity_abs_error"].max()
        ),
        "fisher_identity_max_abs_error": float(
            rows["fisher_variance_identity_abs_error"].max()
        ),
        "range_bound": {
            "coverage": float(rows["g4b_range_covered"].mean()),
            "violations": int((~rows["g4b_range_covered"]).sum()),
            "overflow_fraction": float(
                rows["g4b_range_overflow"].mean()
            ),
            "median_log_bound_minus_log_kl": float(
                np.median(
                    rows["g4b_range_log_bound"].to_numpy()
                    - np.log(
                        np.maximum(
                            rows["exact_kl"].to_numpy(), 1.0e-300
                        )
                    )
                )
            ),
            "claimed_global_bound_only_if_zero_violations": True,
        },
    }


def decomposition_summary(rows: pd.DataFrame) -> Dict[str, Any]:
    columns = [
        "common_shift_energy_fraction",
        "top256_centered_energy_fraction",
        "tail_centered_energy_fraction",
        "fisher_near_null_euclidean_fraction",
        "fisher_near_null_vocab_fraction",
        "output_entropy",
        "top1_margin",
        "fisher_effective_rank",
        "exact_kl",
        "layer27_actual_residual_norm",
        "layer27_direct_norm",
    ]
    task = (
        rows.groupby("task_bucket")[columns]
        .median()
        .reset_index()
        .to_dict("records")
    )
    split = (
        rows.groupby(["task_bucket", "anchor"])[columns]
        .median()
        .reset_index()
        .to_dict("records")
    )
    return {
        "task_medians": task,
        "task_anchor_medians": split,
        "definitions": {
            "near_null_probability_threshold": 1.0e-6,
            "top_centered_vocabulary": 256,
            "energy_fraction_denominator": "uniform-centered Euclidean energy",
        },
    }


def topk_summary(
    pointwise: Mapping[str, Any],
    action_summary: Mapping[str, Any],
    selections: Mapping[str, Any],
) -> Dict[str, Any]:
    point = [
        row
        for row in pointwise["task_summary"]
        if str(row["family"]).startswith(("G5A_", "G5B_", "G5C_"))
        and not bool(row["calibrated_oof"])
    ]
    action = [
        row
        for row in action_summary["task_split"]
        if str(row["family"]).startswith(("G5A_", "G5B_", "G5C_"))
    ]
    return {
        "pointwise_task_split": point,
        "action_task_split": action,
        "outer_fold_training_only_selection": selections,
    }


def cumulant_summary(
    rows: pd.DataFrame,
    pointwise: Mapping[str, Any],
    action_summary: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "pointwise_task_split": [
            row
            for row in pointwise["task_summary"]
            if str(row["family"]).startswith("G7_")
        ],
        "action_task_split": [
            row
            for row in action_summary["task_split"]
            if str(row["family"]).startswith("G7_")
        ],
        "negative_prediction_fraction": {
            "G7_ORDER3": float(rows["g7_order3_negative"].mean()),
            "G7_ORDER4": float(rows["g7_order4_negative"].mean()),
        },
    }


def _task_map(
    records: Iterable[Mapping[str, Any]], family: str
) -> Dict[str, Mapping[str, Any]]:
    return {
        str(row["task_bucket"]): row
        for row in records
        if str(row["family"]) == family
    }


def stage_a_gate(
    cfg: DiscoveryConfig,
    pointwise: Mapping[str, Any],
    action: pd.DataFrame,
    action_summary: Mapping[str, Any],
    quadrature: Mapping[str, Any],
) -> Dict[str, Any]:
    raw_point = _task_map(
        [
            row
            for row in pointwise["task_summary"]
            if not bool(row["calibrated_oof"])
        ],
        "G0_RAW_GLOBAL",
    )
    raw_action = _task_map(
        action_summary["task_split"], "G0_RAW_GLOBAL"
    )
    quadrature_records = {
        str(row["family"]): row
        for row in quadrature["quadrature"]
        if str(row["task_bucket"]) in {"GovReport", "NIAH"}
    }
    by_quad: Dict[str, List[Mapping[str, Any]]] = {}
    for row in quadrature["quadrature"]:
        by_quad.setdefault(str(row["family"]), []).append(row)
    quadrature_pass = any(
        max(float(row["p90_relative_error"]) for row in rows)
        <= float(cfg.gauge_geometry.quadrature_relative_error_gate)
        and max(float(row["maximum_relative_error"]) for row in rows)
        <= 10.0
        * float(cfg.gauge_geometry.quadrature_relative_error_gate)
        for family, rows in by_quad.items()
        if family in {"G4_GL3", "G4_GL5"}
    )
    candidates = [
        "G2_BASE_FISHER",
        "G3_MIDPOINT_FISHER",
        "G5A_OOF_SELECTED",
        "G5B_OOF_SELECTED",
        "G5C_OOF_SELECTED",
        "G6_OOF_SELECTED",
    ]
    family_checks: Dict[str, Any] = {}
    raw_ratio = min(
        float(raw_point[task]["median_symmetric_ratio"])
        for task in ("GovReport", "NIAH")
    )
    for family in candidates:
        point = _task_map(
            [
                row
                for row in pointwise["task_summary"]
                if not bool(row["calibrated_oof"])
            ],
            family,
        )
        action_task = _task_map(action_summary["task_split"], family)
        if set(point) != {"GovReport", "NIAH"} or set(action_task) != {
            "GovReport",
            "NIAH",
        }:
            family_checks[family] = {"passed": False, "missing": True}
            continue
        kl_spearman = all(
            float(point[task]["spearman"])
            > float(raw_point[task]["spearman"])
            for task in ("GovReport", "NIAH")
        )
        action_spearman = all(
            float(action_task[task]["candidate_spearman"])
            >= float(raw_action[task]["candidate_spearman"]) - 1.0e-12
            for task in ("GovReport", "NIAH")
        )
        ratio_improvement = all(
            raw_ratio
            / max(
                float(point[task]["median_symmetric_ratio"]), 1.0e-30
            )
            >= float(cfg.gauge_geometry.geometry_ratio_improvement_gate)
            for task in ("GovReport", "NIAH")
        )
        regret = all(
            float(action_task[task]["normalized_regret"])
            <= float(raw_action[task]["normalized_regret"]) + 1.0e-12
            for task in ("GovReport", "NIAH")
        )
        current = action[action["family"].isin([family, "G0_RAW_GLOBAL"])]
        wide = current.pivot_table(
            index=[
                "sample_id",
                "task_bucket",
                "anchor",
                "horizon",
            ],
            columns="family",
            values="candidate_spearman",
        ).dropna()
        wide["increment"] = (
            wide[family] - wide["G0_RAW_GLOBAL"]
        )
        sequence = (
            wide.groupby(["sample_id", "task_bucket"])["increment"]
            .median()
            .reset_index()
        )
        task_positive_fraction = (
            sequence.groupby("task_bucket")["increment"]
            .apply(lambda values: float((values >= 0.0).mean()))
            .to_dict()
        )
        anchor = (
            wide.groupby(["task_bucket", "anchor"])["increment"]
            .median()
            .reset_index()
        )
        anchor_nonnegative = (
            anchor.groupby("task_bucket")["increment"]
            .apply(lambda values: int((values >= 0.0).sum()))
            .to_dict()
        )
        robust = all(
            task_positive_fraction.get(task, 0.0) > 0.5
            and anchor_nonnegative.get(task, 0) >= 2
            for task in ("GovReport", "NIAH")
        )
        passed = bool(
            kl_spearman
            and action_spearman
            and ratio_improvement
            and regret
            and robust
            and quadrature_pass
        )
        family_checks[family] = {
            "passed": passed,
            "kl_spearman_better_both_tasks": kl_spearman,
            "action_spearman_nonnegative_increment_both_tasks": action_spearman,
            "ratio_improvement_at_least_100x_both_tasks": ratio_improvement,
            "normalized_regret_not_worse_both_tasks": regret,
            "not_single_anchor_or_sequence_driven": robust,
            "quadrature_gate_shared": quadrature_pass,
            "positive_sequence_fraction": task_positive_fraction,
            "nonnegative_anchor_count": anchor_nonnegative,
        }
    passing = [
        family
        for family, check in family_checks.items()
        if bool(check.get("passed", False))
    ]
    return {
        "stage_a_passed": bool(passing),
        "stage_a_passing_families": passing,
        "g4_low_order_quadrature_passed": bool(quadrature_pass),
        "family_checks": family_checks,
        "stage_b_authorized": bool(passing),
        "stage_c_authorized": False,
        "stage_d_authorized": False,
        "gate_frozen_before_formal_run": True,
    }


def inherited_baselines(run_dir: Path, cfg: DiscoveryConfig) -> Dict[str, Any]:
    source = (
        run_dir.parent
        / str(cfg.gauge_geometry.source_run_id)
        / "output_bridge_ranking_summary.json"
    )
    if not source.exists():
        return {"available": False}
    with open(source, "r", encoding="utf-8") as handle:
        summary = json.load(handle)
    keep = {
        "ATTENTION_MASS",
        "DIRECT_ONLY",
        "O0",
        "O1",
    }
    return {
        "available": True,
        "source": str(source),
        "pooled": [
            row for row in summary["pooled"] if row["bridge_family"] in keep
        ],
        "task_split": [
            row
            for row in summary["task_split"]
            if row["bridge_family"] in keep
        ],
    }


def analyze_stage_a(cfg: DiscoveryConfig, run_dir: Path) -> Dict[str, Any]:
    rows_path = run_dir / "oracle_geometry_rows.parquet"
    if not rows_path.exists():
        raise FileNotFoundError(rows_path)
    rows = pd.read_parquet(rows_path).reset_index(drop=True)
    rows["task_bucket"] = rows["task"].map(task_bucket)
    families = geometry_columns(rows)
    pointwise, _ = pointwise_summary(rows, families)
    segment = build_segment_rows(
        rows, families, cfg.gauge_geometry.evaluation_horizons
    )
    action = action_unit_metrics(segment, families)
    _, action, selections = append_oof_selected_variants(
        segment, action, families
    )
    append_pointwise_oof_selected(
        pointwise, rows, selections, families
    )
    action_summary = summarized_action(action, cfg)
    action_summary["outer_fold_topk_margin_selection"] = selections
    action_summary["inherited_baselines"] = inherited_baselines(
        run_dir, cfg
    )
    quadrature = quadrature_summary(rows)
    decomposition = decomposition_summary(rows)
    topk = topk_summary(pointwise, action_summary, selections)
    cumulant = cumulant_summary(rows, pointwise, action_summary)
    gate = stage_a_gate(
        cfg, pointwise, action, action_summary, quadrature
    )

    _atomic_frame(
        action, run_dir / "oracle_geometry_action_rows.parquet"
    )
    atomic_json(
        run_dir / "oracle_geometry_kl_summary.json", pointwise
    )
    atomic_json(
        run_dir / "oracle_geometry_action_summary.json", action_summary
    )
    atomic_json(
        run_dir / "oracle_geometry_decomposition_summary.json",
        decomposition,
    )
    atomic_json(
        run_dir / "path_fisher_quadrature_summary.json", quadrature
    )
    atomic_json(run_dir / "topk_gap_geometry_summary.json", topk)
    atomic_json(run_dir / "cumulant_geometry_summary.json", cumulant)
    atomic_json(run_dir / "gauge_geometry_gate_decision.json", gate)
    status_path = run_dir / "status.json"
    status = (
        json.loads(status_path.read_text())
        if status_path.exists()
        else {}
    )
    status["stage_a_analysis_complete"] = True
    status["stage_a_passed"] = bool(gate["stage_a_passed"])
    status["state"] = (
        "stage_a_passed_stage_b_pending"
        if gate["stage_a_passed"]
        else "stage_a_failed_later_stages_skipped"
    )
    atomic_json(status_path, status)
    return gate
