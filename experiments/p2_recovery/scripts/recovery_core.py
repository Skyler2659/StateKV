"""Shared mathematical primitives for the P2-Recovery program."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
P2_DIR = ROOT / "experiments/p2_state_local_risk/scripts"
if str(P2_DIR) not in sys.path:
    sys.path.insert(0, str(P2_DIR))

from p2_core import (  # noqa: E402
    fisher_variance,
    vector_metrics,
)


def finite_action_metrics(
    predicted: Any,
    truth: Any,
    state_probability: Any,
    *,
    norm_floor: float = 1.0e-12,
    low_norm_threshold: float = 1.0e-8,
) -> Dict[str, Any]:
    """Decompose Euclidean and Fisher error for one finite action."""
    pred = np.asarray(predicted, dtype=np.float64).reshape(-1)
    target = np.asarray(truth, dtype=np.float64).reshape(-1)
    error = pred - target
    target_square = float(np.dot(target, target))
    target_norm = target_square**0.5
    pred_square = float(np.dot(pred, pred))
    dot = float(np.dot(pred, target))
    if target_square > norm_floor**2:
        projection_coefficient = dot / target_square
        parallel_error_norm = abs(dot / target_norm - target_norm)
        orthogonal_error_norm = max(
            pred_square - dot * dot / target_square, 0.0
        ) ** 0.5
    else:
        projection_coefficient = 0.0
        parallel_error_norm = float(np.linalg.norm(error))
        orthogonal_error_norm = 0.0
    truth_fisher_norm = max(
        fisher_variance(state_probability, target), 0.0
    ) ** 0.5
    error_fisher_norm = max(
        fisher_variance(state_probability, error), 0.0
    ) ** 0.5
    return {
        **vector_metrics(
            pred,
            target,
            norm_floor=norm_floor,
            low_norm_threshold=low_norm_threshold,
        ),
        "projection_coefficient": float(projection_coefficient),
        "parallel_error_norm": float(parallel_error_norm),
        "parallel_relative_error": float(
            parallel_error_norm / max(target_norm, norm_floor)
        ),
        "orthogonal_error_norm": float(orthogonal_error_norm),
        "orthogonal_relative_error": float(
            orthogonal_error_norm / max(target_norm, norm_floor)
        ),
        "fisher_truth_norm": float(truth_fisher_norm),
        "fisher_error_norm": float(error_fisher_norm),
        "fisher_relative_error": float(
            error_fisher_norm
            / max(truth_fisher_norm, norm_floor)
        ),
    }


def midpoint_nodes(segment_count: int) -> np.ndarray:
    count = int(segment_count)
    if count < 1:
        raise ValueError("segment_count must be positive")
    return (np.arange(count, dtype=np.float64) + 0.5) / count


def midpoint_integral(
    jvp_values: Sequence[Any],
) -> np.ndarray:
    values = [
        np.asarray(value, dtype=np.float64)
        for value in jvp_values
    ]
    if not values:
        raise ValueError("at least one JVP is required")
    return np.mean(np.stack(values, axis=0), axis=0)


def trapezoidal_integral(
    start_jvp: Any, end_jvp: Any
) -> np.ndarray:
    return 0.5 * (
        np.asarray(start_jvp, dtype=np.float64)
        + np.asarray(end_jvp, dtype=np.float64)
    )


def adaptive_segment_count(
    *,
    midpoint_drift: float,
    secant_mismatch: float,
    drift_threshold: float,
    mismatch_threshold: float,
    maximum_segments: int = 4,
) -> int:
    """Choose cost from internal probes only; no exact risk is accepted."""
    if (
        float(midpoint_drift) <= float(drift_threshold)
        and float(secant_mismatch) <= float(mismatch_threshold)
    ):
        return 1
    return min(2, int(maximum_segments))


def state_local_quadratic_risk(
    gradient: Any,
    displacement: Any,
    state_probability: Any,
) -> float:
    """Scalar risk predictor; exact endpoint KL is absent by design."""
    g = np.asarray(gradient, dtype=np.float64).reshape(-1)
    d = np.asarray(displacement, dtype=np.float64).reshape(-1)
    return float(
        np.dot(g, d)
        + 0.5 * fisher_variance(state_probability, d)
    )


def log_log_slope(
    gamma: Sequence[float],
    residual_norm: Sequence[float],
    floor: float = 1.0e-30,
) -> float:
    x = np.asarray(gamma, dtype=np.float64)
    y = np.asarray(residual_norm, dtype=np.float64)
    valid = (
        np.isfinite(x)
        & np.isfinite(y)
        & (x > 0.0)
        & (y > float(floor))
    )
    if int(valid.sum()) < 2:
        return float("nan")
    return float(
        np.polyfit(np.log(x[valid]), np.log(y[valid]), 1)[0]
    )


def sequence_gate(
    sequence_rows: Any,
    row_rows: Any,
    rule: Mapping[str, Any],
) -> Dict[str, Any]:
    import pandas as pd

    sequence = pd.DataFrame(sequence_rows)
    rows = pd.DataFrame(row_rows)
    task_cosine = (
        sequence.groupby("task")["cosine"].median().to_dict()
    )
    metrics = {
        "overall_sequence_first_cosine": float(
            sequence["cosine"].median()
        ),
        "overall_sequence_first_relative_l2": float(
            sequence["relative_l2"].median()
        ),
        "task_cosine": {
            str(key): float(value)
            for key, value in task_cosine.items()
        },
        "row_pass_fraction": float(
            rows["cosine"]
            .ge(float(rule["row_cosine_min"]))
            .mean()
        ),
        "all_finite": bool(
            sequence["finite"].all() and rows["finite"].all()
        ),
    }
    checks = {
        "overall_cosine": metrics[
            "overall_sequence_first_cosine"
        ]
        >= float(rule["overall_sequence_first_cosine_min"]),
        "each_task_cosine": all(
            value >= float(rule["each_task_cosine_min"])
            for value in metrics["task_cosine"].values()
        ),
        "overall_relative_l2": metrics[
            "overall_sequence_first_relative_l2"
        ]
        <= float(
            rule["overall_sequence_first_relative_l2_max"]
        ),
        "row_pass_fraction": metrics["row_pass_fraction"]
        >= float(rule["row_pass_fraction_min"]),
        "all_finite": metrics["all_finite"]
        == bool(rule["all_finite"]),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": metrics,
    }


__all__ = [
    "adaptive_segment_count",
    "finite_action_metrics",
    "log_log_slope",
    "midpoint_integral",
    "midpoint_nodes",
    "sequence_gate",
    "state_local_quadratic_risk",
    "trapezoidal_integral",
]
