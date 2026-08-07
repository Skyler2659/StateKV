"""Pure numerical utilities for a gated Fisher-pullback experiment."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from statekv.theory_closing import _atomic_frame
from statekv.trajectory_analysis import atomic_json


def fisher_matrix(probability: np.ndarray) -> np.ndarray:
    p = np.asarray(probability, dtype=np.float64)
    return np.diag(p) - np.outer(p, p)


def pullback_quadratic(
    probability: np.ndarray, jvp_direction: np.ndarray
) -> float:
    p = np.asarray(probability, dtype=np.float64)
    value = np.asarray(jvp_direction, dtype=np.float64)
    mean = float(np.dot(p, value))
    return max(float(np.dot(p, value * value) - mean * mean), 0.0)


def explicit_pullback(
    jacobian: np.ndarray, probability: np.ndarray
) -> np.ndarray:
    jac = np.asarray(jacobian, dtype=np.float64)
    return jac.T @ fisher_matrix(probability) @ jac


def symmetric_finite_difference(
    function: Callable[[np.ndarray], np.ndarray],
    point: np.ndarray,
    direction: np.ndarray,
    radius: float,
) -> np.ndarray:
    x = np.asarray(point, dtype=np.float64)
    v = np.asarray(direction, dtype=np.float64)
    epsilon = float(radius)
    return (
        np.asarray(function(x + epsilon * v), dtype=np.float64)
        - np.asarray(function(x - epsilon * v), dtype=np.float64)
    ) / (2.0 * epsilon)


def fisher_output_random_direction(
    probability: np.ndarray, random_vector: np.ndarray
) -> np.ndarray:
    """Apply Diag(sqrt(p))(I-ss^T) to a standard random vector."""

    p = np.asarray(probability, dtype=np.float64)
    g = np.asarray(random_vector, dtype=np.float64)
    root = np.sqrt(p)
    return root * (g - root * float(np.dot(root, g)))


def fisher_vjp_sketch(
    jacobian: np.ndarray,
    probability: np.ndarray,
    random_vectors: np.ndarray,
) -> np.ndarray:
    jac = np.asarray(jacobian, dtype=np.float64)
    directions = [
        jac.T @ fisher_output_random_direction(probability, row)
        for row in np.asarray(random_vectors, dtype=np.float64)
    ]
    return np.stack(directions, axis=1)


def low_rank_from_sketch(
    sketch: np.ndarray, rank: int
) -> Tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(sketch, dtype=np.float64)
    left, singular, _ = np.linalg.svd(matrix, full_matrices=False)
    take = min(int(rank), int(left.shape[1]))
    eigenvalues = np.square(singular[:take]) / max(
        int(matrix.shape[1]), 1
    )
    return left[:, :take], eigenvalues


def spectral_band_energy(
    direction: np.ndarray,
    eigenvectors: np.ndarray,
    eigenvalues: np.ndarray,
    start: int,
    stop: int,
) -> float:
    value = np.asarray(direction, dtype=np.float64)
    vectors = np.asarray(eigenvectors, dtype=np.float64)[:, int(start) : int(stop)]
    weights = np.asarray(eigenvalues, dtype=np.float64)[int(start) : int(stop)]
    coordinates = vectors.T @ value
    return float(np.sqrt(max(np.dot(weights, coordinates**2), 0.0)))


def q_refresh_source(step: int, interval: int) -> int:
    if int(step) < 0 or int(interval) <= 0:
        raise ValueError("step must be nonnegative and interval positive")
    return int(step) - int(step) % int(interval)


def anchor_frozen_sources(
    horizon: int, anchor_source: int = 0
) -> np.ndarray:
    return np.full(int(horizon), int(anchor_source), dtype=np.int64)


def periodic_q_sources(horizon: int, interval: int) -> np.ndarray:
    return np.asarray(
        [q_refresh_source(step, interval) for step in range(int(horizon))],
        dtype=np.int64,
    )


def recursive_q_envelope(
    direct: Sequence[float],
    rho: float,
    coefficient: float,
    intercept: float,
    initial: float = 0.0,
) -> np.ndarray:
    """Roll out from the predicted state only; no realized future state input."""

    state = max(float(initial), 0.0)
    output = []
    for value in direct:
        state = (
            max(float(rho), 0.0) * state
            + max(float(coefficient), 0.0) * max(float(value), 0.0)
            + max(float(intercept), 0.0)
        )
        output.append(state)
    return np.asarray(output, dtype=np.float64)


def refresh_continues_envelope(
    state: float,
    new_direct: Sequence[float],
    rho: float,
    coefficient: float,
    intercept: float,
) -> np.ndarray:
    return recursive_q_envelope(
        new_direct, rho, coefficient, intercept, initial=float(state)
    )


def pairwise_difference(left: float, right: float) -> float:
    return float(left) - float(right)


def normalized_conformal_radius(
    true_delta: Sequence[float],
    predicted_delta: Sequence[float],
    tau: float,
) -> np.ndarray:
    truth = np.asarray(true_delta, dtype=np.float64)
    prediction = np.asarray(predicted_delta, dtype=np.float64)
    return np.abs(truth - prediction) / (
        np.abs(prediction) + max(float(tau), 1.0e-12)
    )


def normalized_conformal_interval(
    predicted_delta: float,
    normalized_margin: float,
    tau: float,
) -> Tuple[float, float]:
    scale = abs(float(predicted_delta)) + max(float(tau), 1.0e-12)
    radius = max(float(normalized_margin), 0.0) * scale
    return float(predicted_delta - radius), float(predicted_delta + radius)


def horizon_stratified_scores(
    frame: pd.DataFrame,
    horizon_column: str,
    score_column: str,
) -> Dict[int, np.ndarray]:
    return {
        int(horizon): current[score_column].to_numpy(dtype=np.float64)
        for horizon, current in frame.groupby(horizon_column)
    }


def predicted_top_candidate_pairs(
    candidate_ids: Sequence[str],
    predicted_risk: Sequence[float],
    top_k: int,
) -> Sequence[Tuple[str, str]]:
    ids = np.asarray(candidate_ids, dtype=object)
    risk = np.asarray(predicted_risk, dtype=np.float64)
    selected = np.argsort(risk, kind="stable")[: min(int(top_k), len(ids))]
    return [
        (str(ids[selected[left]]), str(ids[selected[right]]))
        for left in range(len(selected))
        for right in range(left + 1, len(selected))
    ]


def matched_refresh_count(
    left_events: Sequence[bool], right_events: Sequence[bool]
) -> bool:
    return int(np.asarray(left_events, dtype=bool).sum()) == int(
        np.asarray(right_events, dtype=bool).sum()
    )


PULLBACK_ROW_SCHEMA = {
    "sample_id": "string",
    "task": "string",
    "anchor": "int64",
    "horizon_offset": "int64",
    "candidate_id": "string",
    "pullback_mode": "string",
    "actual_q_energy": "float64",
    "direct_q_energy": "float64",
}

Q_STATE_ROW_SCHEMA = {
    "sample_id": "string",
    "task": "string",
    "anchor": "int64",
    "horizon_offset": "int64",
    "candidate_id": "string",
    "q_family": "string",
    "realized": "float64",
    "bound": "float64",
}


def empty_frame(schema: Mapping[str, str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            column: pd.Series(dtype=dtype)
            for column, dtype in schema.items()
        }
    )


def write_gated_skips(
    run_dir: Path,
    blocking_stage: str,
    reason: str,
) -> Dict[str, Any]:
    """Create every required later-stage artifact with explicit skip metadata."""

    root = Path(run_dir)
    _atomic_frame(
        empty_frame(PULLBACK_ROW_SCHEMA),
        root / "pullback_jvp_rows.parquet",
    )
    _atomic_frame(
        empty_frame(Q_STATE_ROW_SCHEMA),
        root / "q_state_envelope_rows.parquet",
    )
    skipped = {
        "status": "not_run_by_preregistered_gate",
        "blocking_stage": str(blocking_stage),
        "reason": str(reason),
        "rows": 0,
        "post_hoc_gate_relaxation": False,
    }
    for name in (
        "pullback_linearization_summary.json",
        "pullback_low_rank_summary.json",
        "pullback_subspace_drift_summary.json",
        "q_state_envelope_coverage_summary.json",
        "q_state_envelope_tightness_summary.json",
        "q_state_action_ranking_summary.json",
        "spectral_band_envelope_summary.json",
        "pairwise_q_calibration_summary.json",
        "q_refresh_policy_summary.json",
        "q_free_generation_results.json",
    ):
        atomic_json(root / name, dict(skipped, artifact=name))
    atomic_json(root / "gauge_later_stages_skip.json", skipped)
    return skipped


def load_gate_and_apply_skips(run_dir: Path) -> Dict[str, Any]:
    root = Path(run_dir)
    gate_path = root / "gauge_geometry_gate_decision.json"
    gate = json.loads(gate_path.read_text())
    if bool(gate.get("stage_a_passed", False)):
        return {
            "status": "stage_b_authorized",
            "stage_a_passed": True,
        }
    return write_gated_skips(
        root,
        "Stage A",
        "No non-oracle gauge family passed every frozen Stage-A gate.",
    )

