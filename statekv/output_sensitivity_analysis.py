"""Nested, sequence-grouped analysis for output-sensitivity closure.

The module intentionally keeps the validated E2 dynamics family fixed.  It
only fits nonnegative residual-to-logit readouts and sequence-clustered
calibration margins on top of recursive E2 coordinates.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import lsq_linear, nnls

from statekv.config import DiscoveryConfig
from statekv.robust_envelope_analysis import (
    EnvelopeModel,
    fit_nonnegative_envelope,
    recursive_envelope,
)
from statekv.theory_closing import _atomic_frame
from statekv.trajectory_analysis import atomic_json


LAYERS = [0, 7, 14, 15, 21, 27]
OPERATING_FEATURES = [
    "output_entropy",
    "inverse_logit_margin",
    "attention_entropy",
    "attention_concentration",
    "attention_top8_mass",
    "prefix_length",
    "current_hidden_norm",
    "current_projected_output_norm",
]
BRIDGE_FAMILIES = [
    "O0",
    "O1",
    "O2",
    "O2_NO_L27",
    "O3",
    "O4_CONT",
    "O4_REGIME",
    "JAC_FD",
]
FORBIDDEN_OUTPUT_FEATURES = {
    "task",
    "task_id",
    "sequence_id",
    "sample_id",
    "future_token",
    "future_kl",
    "future_nll",
    "candidate_realized_error",
    "answer_correctness",
}


def _task_bucket(task: str) -> str:
    return "GovReport" if "gov" in str(task).lower() else "NIAH"


def deployable_output_features(columns: Sequence[str]) -> bool:
    return not bool(
        {str(value) for value in columns} & FORBIDDEN_OUTPUT_FEATURES
    )


def _stable_key(*values: Any) -> str:
    return hashlib.sha256(
        ":".join(str(value) for value in values).encode("utf-8")
    ).hexdigest()


def nested_sequence_partition(
    sequences: Sequence[str],
    held_out: str,
    task_by_sequence: Mapping[str, str],
) -> Tuple[List[str], List[str], List[str]]:
    """Return 10 fit, 5 state-margin, and 8 output-calibration sequences.

    The allocation is balanced by task and depends only on stable identifiers.
    No outcome is consulted.
    """

    grouped: Dict[str, List[str]] = {"NIAH": [], "GovReport": []}
    for sequence in sequences:
        if sequence == held_out:
            continue
        grouped[_task_bucket(task_by_sequence[sequence])].append(sequence)
    for task in grouped:
        grouped[task] = sorted(
            grouped[task], key=lambda value: _stable_key(held_out, task, value)
        )
        if len(grouped[task]) < 11:
            raise ValueError("24-sequence balanced design is required")
    calibration = grouped["NIAH"][:4] + grouped["GovReport"][:4]
    remainder = {
        task: [value for value in grouped[task] if value not in calibration]
        for task in grouped
    }
    held_task = _task_bucket(task_by_sequence[held_out])
    other_task = "GovReport" if held_task == "NIAH" else "NIAH"
    state_margin = remainder[held_task][:2] + remainder[other_task][:3]
    fit = [
        value
        for task in ("NIAH", "GovReport")
        for value in remainder[task]
        if value not in state_margin
    ]
    if not (len(fit) == 10 and len(state_margin) == 5 and len(calibration) == 8):
        raise RuntimeError("nested split sizes differ from pre-registration")
    if set(fit) & set(state_margin) or set(fit) & set(calibration):
        raise RuntimeError("nested split leakage")
    if set(state_margin) & set(calibration) or held_out in set(
        fit + state_margin + calibration
    ):
        raise RuntimeError("held-out sequence leakage")
    return sorted(fit), sorted(state_margin), sorted(calibration)


def conformal_order_statistic(
    scores: Sequence[float], level: float
) -> Tuple[float, int, bool]:
    """Finite-sample split-conformal order statistic over sequence scores."""

    values = np.sort(np.asarray(scores, dtype=np.float64))
    if not len(values):
        raise ValueError("at least one calibration sequence is required")
    if not 0.0 < float(level) < 1.0:
        raise ValueError("coverage level must lie in (0,1)")
    rank = min(len(values), int(math.ceil((len(values) + 1) * level)))
    return float(values[rank - 1]), int(rank), bool(rank == len(values))


def clustered_additive_margin(
    residual: Sequence[float],
    sequence_ids: Sequence[str],
    level: float,
) -> Tuple[float, int, bool]:
    values = np.maximum(np.asarray(residual, dtype=np.float64), 0.0)
    ids = np.asarray(sequence_ids)
    if len(values) != len(ids):
        raise ValueError("residual and sequence ids are not aligned")
    sequence_scores = [
        float(values[ids == sequence].max())
        for sequence in sorted(set(ids.tolist()))
    ]
    return conformal_order_statistic(sequence_scores, level)


def clustered_coordinate_margin(
    residual: np.ndarray,
    sequence_ids: Sequence[str],
    level: float,
) -> Tuple[np.ndarray, int, bool]:
    values = np.maximum(np.asarray(residual, dtype=np.float64), 0.0)
    ids = np.asarray(sequence_ids)
    if values.ndim != 2 or len(values) != len(ids):
        raise ValueError("coordinate calibration arrays are not aligned")
    sequence_scores = np.stack(
        [
            values[ids == sequence].max(axis=0)
            for sequence in sorted(set(ids.tolist()))
        ]
    )
    n = len(sequence_scores)
    rank = min(n, int(math.ceil((n + 1) * level)))
    return (
        np.sort(sequence_scores, axis=0)[rank - 1],
        int(rank),
        bool(rank == n),
    )


def pairwise_feature_difference(
    left: np.ndarray, right: np.ndarray
) -> np.ndarray:
    return np.asarray(left, dtype=np.float64) - np.asarray(
        right, dtype=np.float64
    )


def pairwise_prediction(left: float, right: float) -> float:
    return float(left) - float(right)


def swapped_pairwise_interval(
    lower: float, upper: float
) -> Tuple[float, float]:
    return -float(upper), -float(lower)


def dominance_decision(
    predicted_delta: float, margin: float
) -> Tuple[bool, int]:
    """Return (abstain, sign); -1 means the left action dominates."""

    lower = float(predicted_delta) - float(margin)
    upper = float(predicted_delta) + float(margin)
    if upper < 0.0:
        return False, -1
    if lower > 0.0:
        return False, 1
    return True, 0


def refresh_lcb_trigger(
    lower_confidence_bound: float,
    refresh_cost: float,
    maximum_refresh_count: int,
    current_refresh_count: int,
) -> bool:
    return bool(
        int(current_refresh_count) < int(maximum_refresh_count)
        and float(lower_confidence_bound) > float(refresh_cost)
    )


def validate_finite_difference_grid(
    probes: pd.DataFrame,
    minimum_directions: int = 8,
    minimum_radii: int = 3,
) -> bool:
    counts = probes.groupby(["sample_id", "anchor", "layer"]).agg(
        directions=("direction_index", "nunique"),
        radii=("relative_radius", "nunique"),
    )
    return bool(
        (counts["directions"] >= int(minimum_directions)).all()
        and (counts["radii"] >= int(minimum_radii)).all()
        and probes["finite_difference_symmetric"].all()
    )


def direction_estimates_agree(
    jvp: Sequence[float],
    finite_difference: Sequence[float],
    minimum_cosine: float = 0.9,
) -> bool:
    left = np.asarray(jvp, dtype=np.float64)
    right = np.asarray(finite_difference, dtype=np.float64)
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    if denominator <= 0.0:
        return bool(np.allclose(left, right))
    return bool(float(left @ right / denominator) >= minimum_cosine)


def json_numbers_consistent(
    frame: pd.DataFrame,
    summary: Mapping[str, Any],
    column: str,
    summary_key: str,
    tolerance: float = 1e-12,
) -> bool:
    return bool(
        abs(float(frame[column].mean()) - float(summary[summary_key]))
        <= float(tolerance)
    )


def softmax_kl_inequality_holds(
    exact_kl: np.ndarray, logit_l2_sq: np.ndarray, tolerance: float = 1e-7
) -> bool:
    return bool(
        np.all(
            np.asarray(exact_kl, dtype=np.float64)
            <= 0.25 * np.asarray(logit_l2_sq, dtype=np.float64)
            + float(tolerance)
        )
    )


def pairwise_auc(true_delta: Sequence[float], predicted_delta: Sequence[float]) -> float:
    """AUC for predicting that the left action is worse (positive delta)."""

    truth = np.asarray(true_delta, dtype=np.float64) > 0.0
    score = np.asarray(predicted_delta, dtype=np.float64)
    positive = int(truth.sum())
    negative = int((~truth).sum())
    if positive == 0 or negative == 0:
        return float("nan")
    ranks = stats.rankdata(score, method="average")
    return float(
        (ranks[truth].sum() - positive * (positive + 1) / 2.0)
        / (positive * negative)
    )


def top1_regret(
    true_utility: Sequence[float], predicted_utility: Sequence[float]
) -> float:
    truth = np.asarray(true_utility, dtype=np.float64)
    prediction = np.asarray(predicted_utility, dtype=np.float64)
    if len(truth) != len(prediction) or not len(truth):
        raise ValueError("top-1 arrays must be nonempty and aligned")
    return float(truth[int(np.argmin(prediction))] - truth.min())


def topk_overlap(
    true_utility: Sequence[float],
    predicted_utility: Sequence[float],
    k: int,
) -> float:
    truth = np.asarray(true_utility, dtype=np.float64)
    prediction = np.asarray(predicted_utility, dtype=np.float64)
    count = min(int(k), len(truth))
    if count <= 0:
        raise ValueError("k must be positive")
    left = set(np.argsort(truth)[:count].tolist())
    right = set(np.argsort(prediction)[:count].tolist())
    return float(len(left & right) / count)


def _wide_rows(rows: pd.DataFrame) -> Tuple[pd.DataFrame, List[int]]:
    keys = [
        "sample_id",
        "task",
        "trajectory_id",
        "trajectory_kind",
        "candidate_id",
        "candidate_index",
        "candidate_source",
        "anchor",
        "horizon_offset",
        "target_index",
    ]
    layers = sorted(int(value) for value in rows["layer"].unique())
    error = rows.pivot_table(
        index=keys, columns="layer", values="residual_error", aggfunc="first"
    )
    error.columns = ["e_l%d" % int(value) for value in error.columns]
    direct = rows.pivot_table(
        index=keys, columns="layer", values="direct_coordinate", aggfunc="first"
    )
    direct.columns = ["d_l%d" % int(value) for value in direct.columns]
    output = rows.pivot_table(
        index=keys,
        columns="layer",
        values="projected_output_error",
        aggfunc="first",
    )
    output.columns = ["po_l%d" % int(value) for value in output.columns]
    metric_columns = [
        "exact_kl",
        "js",
        "full_nll",
        "perturbed_nll",
        "delta_nll",
        "logit_l2_sq",
        "fisher_quadratic",
        "active_cache_tokens",
        "total_budget",
        "protected_recent",
        "uses_future_compressed_truth",
        "token_position_aligned",
    ] + OPERATING_FEATURES
    metrics = rows.groupby(keys, as_index=True)[metric_columns].first()
    direct_diagnostics = rows.groupby(keys, as_index=True).agg(
        deleted_attention_mass_total=("deleted_attention_mass", "sum")
    )
    wide = (
        error.join(direct, how="inner")
        .join(output, how="inner")
        .join(metrics, how="inner")
        .join(direct_diagnostics, how="inner")
        .reset_index()
    )
    wide["projected_output_error"] = wide[
        ["po_l%d" % layer for layer in layers]
    ].sum(axis=1)
    wide["logit_l2"] = np.sqrt(
        np.maximum(wide["logit_l2_sq"].to_numpy(dtype=np.float64), 0.0)
    )
    return wide, layers


def _transition_arrays(
    wide: pd.DataFrame,
    layers: Sequence[int],
    sequences: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    e_columns = ["e_l%d" % layer for layer in layers]
    d_columns = ["d_l%d" % layer for layer in layers]
    previous_parts: List[np.ndarray] = []
    direct_parts: List[np.ndarray] = []
    target_parts: List[np.ndarray] = []
    ids: List[str] = []
    selected = wide[wide["sample_id"].isin(sequences)]
    for (_, trajectory), group in selected.groupby(
        ["sample_id", "trajectory_id"], sort=False
    ):
        ordered = group.sort_values("horizon_offset")
        actual = ordered[e_columns].to_numpy(dtype=np.float64)
        direct = ordered[d_columns].to_numpy(dtype=np.float64)
        previous_parts.append(
            np.vstack([np.zeros((1, len(layers))), actual[:-1]])
        )
        direct_parts.append(direct)
        target_parts.append(actual)
        ids.extend(ordered["sample_id"].astype(str).tolist())
    return (
        np.concatenate(previous_parts),
        np.concatenate(direct_parts),
        np.concatenate(target_parts),
        np.asarray(ids),
    )


def _attach_recursive_e2(
    wide: pd.DataFrame,
    layers: Sequence[int],
    model: EnvelopeModel,
    margin: np.ndarray,
) -> pd.DataFrame:
    output: List[pd.DataFrame] = []
    d_columns = ["d_l%d" % layer for layer in layers]
    for _, group in wide.groupby("trajectory_id", sort=False):
        ordered = group.sort_values("horizon_offset").copy()
        bound, exploded = recursive_envelope(
            model,
            ordered[d_columns].to_numpy(dtype=np.float64),
            margin,
        )
        for index, layer in enumerate(layers):
            ordered["b_l%d" % layer] = bound[:, index]
        ordered["e2_exploded"] = exploded
        output.append(ordered)
    return pd.concat(output, ignore_index=True, sort=False)


@dataclass
class Bridge:
    family: str
    coefficient: np.ndarray
    intercept: float
    metadata: Dict[str, Any]

    def predict(self, rows: pd.DataFrame, layers: Sequence[int]) -> np.ndarray:
        e = rows[["b_l%d" % layer for layer in layers]].to_numpy(
            dtype=np.float64
        )
        d = rows[["d_l%d" % layer for layer in layers]].to_numpy(
            dtype=np.float64
        )
        if self.family == "O0":
            return np.linalg.norm(e, axis=1)
        if self.family == "O1":
            x = np.column_stack([e[:, -1], d[:, -1]])
            return np.maximum(0.0, x @ self.coefficient + self.intercept)
        if self.family in {"O2", "O2_NO_L27"}:
            x = np.column_stack([e, d])
            if self.family == "O2_NO_L27":
                x = np.column_stack([e[:, :-1], d[:, :-1]])
            return np.maximum(0.0, x @ self.coefficient + self.intercept)
        if self.family == "O3":
            bins = np.asarray(self.metadata["bins"], dtype=np.int64)
            offset = rows["horizon_offset"].to_numpy(dtype=np.int64)
            bin_index = np.searchsorted(bins, offset, side="left")
            amplitude = np.asarray(
                self.metadata["amplitude"], dtype=np.float64
            )[bin_index]
            intercept = np.asarray(
                self.metadata["intercept_by_bin"], dtype=np.float64
            )[bin_index]
            x = np.column_stack([e, d])
            return np.maximum(
                0.0, amplitude * (x @ self.coefficient) + intercept
            )
        if self.family == "O4_CONT":
            x = np.column_stack([e, d])
            observables = _normalized_observables(
                rows,
                self.metadata["observable_min"],
                self.metadata["observable_span"],
            )
            design = np.column_stack(
                [x] + [x * observables[:, [index]] for index in range(
                    observables.shape[1]
                )]
            )
            return np.maximum(
                0.0, design @ self.coefficient + self.intercept
            )
        if self.family == "O4_REGIME":
            x = np.column_stack([e, d])
            entropy = rows["output_entropy"].to_numpy(dtype=np.float64)
            high = entropy > float(self.metadata["entropy_threshold"])
            width = x.shape[1]
            result = np.zeros(len(rows), dtype=np.float64)
            result[~high] = (
                x[~high] @ self.coefficient[:width]
                + float(self.metadata["low_intercept"])
            )
            result[high] = (
                x[high] @ self.coefficient[width:]
                + float(self.metadata["high_intercept"])
            )
            return np.maximum(0.0, result)
        raise ValueError("unsupported empirical bridge: %s" % self.family)


def bridge_coefficients_nonnegative(bridge: Bridge) -> bool:
    return bool(
        np.all(np.asarray(bridge.coefficient) >= -1e-12)
        and float(bridge.intercept) >= -1e-12
    )


def _fit_nnls(design: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, float]:
    augmented = np.column_stack(
        [np.asarray(design, dtype=np.float64), np.ones(len(design))]
    )
    current_target = np.asarray(target, dtype=np.float64)
    if not len(current_target):
        raise ValueError("nonnegative bridge fit received no rows")
    try:
        coefficient, _ = nnls(
            augmented,
            current_target,
            maxiter=max(3 * augmented.shape[1], 1000),
        )
    except RuntimeError:
        coefficient = lsq_linear(
            augmented,
            current_target,
            bounds=(0.0, np.inf),
            method="bvls",
            max_iter=max(10 * augmented.shape[1], 1000),
        ).x
    return coefficient[:-1], float(coefficient[-1])


def _normalized_observables(
    rows: pd.DataFrame,
    minimum: Sequence[float],
    span: Sequence[float],
) -> np.ndarray:
    values = rows[OPERATING_FEATURES].to_numpy(dtype=np.float64)
    return np.clip(
        (values - np.asarray(minimum)) / np.asarray(span), 0.0, 1.0
    )


def fit_bridges(
    training: pd.DataFrame, layers: Sequence[int]
) -> Dict[str, Bridge]:
    e = training[["b_l%d" % layer for layer in layers]].to_numpy(
        dtype=np.float64
    )
    d = training[["d_l%d" % layer for layer in layers]].to_numpy(
        dtype=np.float64
    )
    x = np.column_stack([e, d])
    target = training["logit_l2"].to_numpy(dtype=np.float64)
    result: Dict[str, Bridge] = {
        "O0": Bridge("O0", np.ones(len(layers)), 0.0, {}),
    }
    coefficient, intercept = _fit_nnls(
        np.column_stack([e[:, -1], d[:, -1]]), target
    )
    result["O1"] = Bridge("O1", coefficient, intercept, {})
    coefficient, intercept = _fit_nnls(x, target)
    result["O2"] = Bridge("O2", coefficient, intercept, {})
    coefficient, intercept = _fit_nnls(
        np.column_stack([e[:, :-1], d[:, :-1]]), target
    )
    result["O2_NO_L27"] = Bridge(
        "O2_NO_L27",
        coefficient,
        intercept,
        {"ablation": "layer_27_excluded"},
    )

    maximum_offset = int(training["horizon_offset"].max())
    bins = np.asarray(
        sorted(set(min(value, maximum_offset) for value in [4, 8, 16, 32])),
        dtype=np.int64,
    )
    matrix = []
    intercepts = []
    offsets = training["horizon_offset"].to_numpy(dtype=np.int64)
    lower = 0
    for upper in bins:
        mask = (offsets > lower) & (offsets <= upper)
        current, current_intercept = _fit_nnls(x[mask], target[mask])
        matrix.append(current)
        intercepts.append(current_intercept)
        lower = int(upper)
    coefficient_matrix = np.stack(matrix)
    left, singular, right = np.linalg.svd(
        coefficient_matrix, full_matrices=False
    )
    amplitude = np.abs(left[:, 0]) * math.sqrt(max(singular[0], 0.0))
    shared = np.abs(right[0]) * math.sqrt(max(singular[0], 0.0))
    result["O3"] = Bridge(
        "O3",
        shared,
        0.0,
        {
            "bins": bins.tolist(),
            "amplitude": amplitude.tolist(),
            "intercept_by_bin": intercepts,
            "rank": 1,
        },
    )

    observable = training[OPERATING_FEATURES].to_numpy(dtype=np.float64)
    minimum = np.nanmin(observable, axis=0)
    maximum = np.nanmax(observable, axis=0)
    span = np.maximum(maximum - minimum, 1e-12)
    normalized = np.clip((observable - minimum) / span, 0.0, 1.0)
    continuous_design = np.column_stack(
        [x] + [x * normalized[:, [index]] for index in range(len(
            OPERATING_FEATURES
        ))]
    )
    coefficient, intercept = _fit_nnls(continuous_design, target)
    result["O4_CONT"] = Bridge(
        "O4_CONT",
        coefficient,
        intercept,
        {
            "observable_min": minimum.tolist(),
            "observable_span": span.tolist(),
            "features": list(OPERATING_FEATURES),
            "task_feature_used": False,
        },
    )
    threshold = float(training["output_entropy"].median())
    high = training["output_entropy"].to_numpy(dtype=np.float64) > threshold
    low_coefficient, low_intercept = _fit_nnls(x[~high], target[~high])
    high_coefficient, high_intercept = _fit_nnls(x[high], target[high])
    result["O4_REGIME"] = Bridge(
        "O4_REGIME",
        np.concatenate([low_coefficient, high_coefficient]),
        min(low_intercept, high_intercept),
        {
            "entropy_threshold": threshold,
            "low_intercept": low_intercept,
            "high_intercept": high_intercept,
            "gate_feature": "output_entropy",
            "task_feature_used": False,
        },
    )
    if not all(bridge_coefficients_nonnegative(value) for value in result.values()):
        raise RuntimeError("output bridge contains a negative coefficient")
    return result


def _jacobian_gain_table(probes: pd.DataFrame) -> pd.DataFrame:
    middle_radius = sorted(probes["relative_radius"].unique())[
        len(probes["relative_radius"].unique()) // 2
    ]
    primary = probes[
        np.isclose(probes["relative_radius"], middle_radius)
    ]
    gain = (
        primary.groupby(["sample_id", "anchor", "layer"], as_index=False)[
            "symmetric_directional_gain"
        ]
        .max()
        .rename(columns={"symmetric_directional_gain": "gain"})
    )
    return gain


def _jacobian_prediction(
    rows: pd.DataFrame,
    gain_table: pd.DataFrame,
    layers: Sequence[int],
) -> np.ndarray:
    lookup = {
        (str(row.sample_id), int(row.anchor), int(row.layer)): float(row.gain)
        for row in gain_table.itertuples()
    }
    result = np.zeros(len(rows), dtype=np.float64)
    for index, row in enumerate(rows.itertuples()):
        result[index] = sum(
            lookup[(str(row.sample_id), int(row.anchor), int(layer))]
            * float(getattr(row, "b_l%d" % layer))
            for layer in layers
        )
    return result


def _correlation(left: Sequence[float], right: Sequence[float], kind: str) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if len(a) < 2 or np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return 0.0
    if kind == "spearman":
        return float(stats.spearmanr(a, b).statistic)
    return float(stats.kendalltau(a, b).statistic)


def _json_records(frame: pd.DataFrame) -> List[Dict[str, Any]]:
    return json.loads(frame.to_json(orient="records"))


def _sequence_cluster_bootstrap(
    frame: pd.DataFrame,
    value: str,
    cluster: str,
    samples: int,
    seed: int,
    reducer: str = "median",
) -> Dict[str, float]:
    per_sequence = frame.groupby(cluster)[value].mean()
    values = per_sequence.to_numpy(dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"estimate": float("nan"), "low": float("nan"), "high": float("nan")}
    rng = np.random.default_rng(int(seed))
    draws = []
    for _ in range(int(samples)):
        sample = rng.choice(values, size=len(values), replace=True)
        draws.append(
            float(np.median(sample) if reducer == "median" else np.mean(sample))
        )
    estimate = float(
        np.median(values) if reducer == "median" else np.mean(values)
    )
    return {
        "estimate": estimate,
        "low": float(np.quantile(draws, 0.025)),
        "high": float(np.quantile(draws, 0.975)),
        "cluster_count": int(len(values)),
    }


def _summarize_bridge_rows(
    output_rows: pd.DataFrame,
    state_rows: pd.DataFrame,
    probes: pd.DataFrame,
    cfg: DiscoveryConfig,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    coverage_table = (
        output_rows.groupby(["bridge_family", "task_bucket"], as_index=False)
        .agg(
            logit_coverage=("logit_covered", "mean"),
            induced_kl_coverage=("kl_covered", "mean"),
            rows=("logit_covered", "size"),
            sequences=("held_out_sequence", "nunique"),
        )
    )
    simultaneous_output = (
        output_rows.groupby(
            ["held_out_sequence", "bridge_family", "task_bucket"],
            as_index=False,
        )
        .agg(
            simultaneous_logit_covered=("logit_covered", "all"),
            simultaneous_kl_covered=("kl_covered", "all"),
        )
        .groupby(["bridge_family", "task_bucket"], as_index=False)
        .agg(
            simultaneous_logit_coverage=(
                "simultaneous_logit_covered",
                "mean",
            ),
            simultaneous_kl_coverage=(
                "simultaneous_kl_covered",
                "mean",
            ),
        )
    )
    tightness_table = (
        output_rows.groupby(["bridge_family", "task_bucket"], as_index=False)
        .agg(
            median_logit_slack=(
                "logit_slack",
                "median",
            ),
            median_logit_looseness=("logit_looseness", "median"),
            p90_logit_looseness=(
                "logit_looseness",
                lambda value: float(np.quantile(value, 0.9)),
            ),
            median_kl_slack=("kl_slack", "median"),
            median_kl_looseness=("kl_looseness", "median"),
            p90_kl_looseness=(
                "kl_looseness",
                lambda value: float(np.quantile(value, 0.9)),
            ),
        )
    )
    canonical_logits = output_rows[
        output_rows["bridge_family"] == "O2"
    ].copy()
    canonical_logits["exact_softmax_bound"] = (
        0.25 * canonical_logits["logit_l2_sq"]
    )
    canonical_logits["exact_softmax_kl_looseness"] = (
        canonical_logits["exact_softmax_bound"] + 1e-12
    ) / (canonical_logits["exact_kl"] + 1e-12)
    exact_softmax_tightness = (
        canonical_logits.groupby("task_bucket", as_index=False)
        .agg(
            median_exact_kl=("exact_kl", "median"),
            median_realized_logit_l2=("realized_logit_l2", "median"),
            median_exact_softmax_bound=("exact_softmax_bound", "median"),
            median_exact_softmax_kl_looseness=(
                "exact_softmax_kl_looseness",
                "median",
            ),
            p90_exact_softmax_kl_looseness=(
                "exact_softmax_kl_looseness",
                lambda value: float(np.quantile(value, 0.9)),
            ),
        )
    )
    state_summary = (
        state_rows.groupby(["task_bucket", "horizon_offset"], as_index=False)
        .agg(
            pointwise_coverage=("pointwise_covered", "mean"),
            trajectory_coverage=("trajectory_covered", "mean"),
            median_looseness=("looseness", "median"),
            exploded=("exploded", "mean"),
        )
    )
    by_layer = (
        state_rows.groupby(["task_bucket", "layer"], as_index=False)
        .agg(
            pointwise_coverage=("pointwise_covered", "mean"),
            median_looseness=("looseness", "median"),
        )
    )
    calibration = (
        output_rows.groupby("held_out_sequence", as_index=False)
        .agg(
            calibration_sequences=("calibration_sequences", "first"),
            conformal_rank=("conformal_rank", "first"),
            conformal_is_maximum=("conformal_is_maximum", "first"),
        )
    )
    radius_table = (
        probes.assign(task_bucket=probes["task"].map(_task_bucket))
        .groupby(["task_bucket", "layer", "relative_radius"], as_index=False)
        .agg(
            median_directional_gain=("symmetric_directional_gain", "median"),
            p90_directional_gain=(
                "symmetric_directional_gain",
                lambda value: float(np.quantile(value, 0.9)),
            ),
            median_symmetry_error=("radius_symmetry_error", "median"),
            directions=("direction_index", "nunique"),
            sequences=("sample_id", "nunique"),
        )
    )
    coverage = {
        "independent_unit": "sequence",
        "sequence_count": int(output_rows["held_out_sequence"].nunique()),
        "task_sequence_counts": output_rows.groupby("task_bucket")[
            "held_out_sequence"
        ].nunique().astype(int).to_dict(),
        "coverage_level": 0.95,
        "calibration": _json_records(calibration),
        "finite_sample_warning": (
            "With eight calibration sequences, both 90% and 95% split-"
            "conformal order statistics equal the calibration maximum. "
            "Observed coverage is not a deployment-distribution theorem."
        ),
        "output_bridge": _json_records(coverage_table),
        "output_bridge_simultaneous_sequence": _json_records(
            simultaneous_output
        ),
        "state_e2": _json_records(state_summary),
        "state_e2_by_layer": _json_records(by_layer),
        "layer_27": _json_records(by_layer[by_layer["layer"] == 27]),
        "state_e2_gate": {
            "pointwise_coverage_at_least_0_90_both_tasks": bool(
                (
                    state_rows.groupby("task_bucket")[
                        "pointwise_covered"
                    ].mean()
                    >= 0.90
                ).all()
            ),
            "no_numerical_explosion": bool(
                not state_rows["exploded"].any()
            ),
            "trajectory_coverage_by_task": state_rows.groupby(
                "task_bucket"
            )["trajectory_covered"].mean().to_dict(),
        },
        "softmax_kl_inequality_all_rows": softmax_kl_inequality_holds(
            output_rows["exact_kl"], output_rows["logit_l2_sq"]
        ),
        "jacobian_arm": {
            "estimator": "symmetric finite directional differences",
            "claimed_operator_norm": bool(
                probes["claimed_operator_norm"].any()
            ),
            "directions_per_operating_point": int(
                probes.groupby(["sample_id", "anchor", "layer"])[
                    "direction_index"
                ].nunique().min()
            ),
            "radii": sorted(
                float(value)
                for value in probes["relative_radius"].unique()
            ),
            "radius_sensitivity": _json_records(radius_table),
        },
    }
    tightness = {
        "definition": "bound/(realized+1e-9)",
        "table": _json_records(tightness_table),
        "exact_softmax_inequality_tightness": _json_records(
            exact_softmax_tightness
        ),
        "nonvacuous_gate": {
            "median_logit_looseness_at_most": float(
                cfg.output_sensitivity.output_looseness_gate
            )
        },
    }
    return coverage, tightness


def _candidate_segment_rows(
    output_rows: pd.DataFrame,
    horizons: Sequence[int],
) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    key_columns = [
        "held_out_sequence",
        "task",
        "task_bucket",
        "anchor",
        "candidate_id",
        "candidate_index",
        "candidate_source",
        "bridge_family",
    ]
    for keys, group in output_rows.groupby(key_columns, sort=False):
        ordered = group.sort_values("horizon_offset")
        base = dict(zip(key_columns, keys))
        for horizon in horizons:
            current = ordered[ordered["horizon_offset"] <= int(horizon)]
            if len(current) != int(horizon):
                continue
            records.append(
                {
                    **base,
                    "horizon": int(horizon),
                    "predicted_objective": float(
                        current["induced_kl_bound"].sum()
                    ),
                    "raw_predicted_objective": float(
                        0.25
                        * np.square(current["raw_logit_prediction"]).sum()
                    ),
                    "true_exact_kl": float(current["exact_kl"].sum()),
                    "projected_output_error": float(
                        current["projected_output_error"].sum()
                    ),
                    "js": float(current["js"].sum()),
                    "delta_nll": float(current["delta_nll"].sum()),
                }
            )
    return pd.DataFrame(records)


def _analytical_baseline_segment_rows(
    output_rows: pd.DataFrame, horizons: Sequence[int]
) -> pd.DataFrame:
    canonical = output_rows[output_rows["bridge_family"] == "O2"]
    records: List[Dict[str, Any]] = []
    keys = [
        "held_out_sequence",
        "task",
        "task_bucket",
        "anchor",
        "candidate_id",
        "candidate_index",
        "candidate_source",
    ]
    for values, group in canonical.groupby(keys, sort=False):
        for horizon in horizons:
            current = group[group["horizon_offset"] <= int(horizon)]
            if len(current) != int(horizon):
                continue
            common = {
                **dict(zip(keys, values)),
                "horizon": int(horizon),
                "true_exact_kl": float(current["exact_kl"].sum()),
                "projected_output_error": float(
                    current["projected_output_error"].sum()
                ),
                "js": float(current["js"].sum()),
                "delta_nll": float(current["delta_nll"].sum()),
            }
            for family, column in (
                ("DIRECT_ONLY", "dynamic_direct_energy"),
                ("ATTENTION_MASS", "deleted_attention_mass_total"),
            ):
                objective = float(current[column].sum())
                records.append(
                    {
                        **common,
                        "bridge_family": family,
                        "predicted_objective": objective,
                        "raw_predicted_objective": objective,
                    }
                )
    return pd.DataFrame(records)


def _calibrate_pairwise_margins(
    calibration: pd.DataFrame,
    logit_prediction: np.ndarray,
    logit_margin: float,
    horizons: Sequence[int],
) -> Dict[int, Tuple[float, int, bool]]:
    current = calibration.copy()
    current["predicted_objective_step"] = 0.25 * np.square(
        np.asarray(logit_prediction, dtype=np.float64) + float(logit_margin)
    )
    sequence_scores: Dict[int, Dict[str, float]] = {
        int(horizon): {} for horizon in horizons
    }
    for sequence, sequence_rows in current.groupby("sample_id", sort=False):
        for horizon in horizons:
            actions: List[Tuple[float, float]] = []
            for _, group in sequence_rows.groupby(
                ["anchor", "candidate_id"], sort=False
            ):
                selected = group[
                    group["horizon_offset"] <= int(horizon)
                ]
                if len(selected) != int(horizon):
                    continue
                actions.append(
                    (
                        float(selected["predicted_objective_step"].sum()),
                        float(selected["exact_kl"].sum()),
                    )
                )
            maximum = 0.0
            for left in range(len(actions)):
                for right in range(left + 1, len(actions)):
                    residual = abs(
                        (actions[left][0] - actions[right][0])
                        - (actions[left][1] - actions[right][1])
                    )
                    maximum = max(maximum, float(residual))
            sequence_scores[int(horizon)][str(sequence)] = maximum
    result: Dict[int, Tuple[float, int, bool]] = {}
    for horizon, scores in sequence_scores.items():
        result[int(horizon)] = conformal_order_statistic(
            list(scores.values()), 0.95
        )
    return result


def _action_feature_rows(
    rows: pd.DataFrame,
    predictions: Mapping[str, np.ndarray],
    margins: Mapping[str, float],
    horizons: Sequence[int],
) -> pd.DataFrame:
    current = rows.copy()
    for family, values in predictions.items():
        current["risk_%s" % family] = 0.25 * np.square(
            np.asarray(values, dtype=np.float64) + float(margins[family])
        )
    records: List[Dict[str, Any]] = []
    keys = [
        "sample_id",
        "task",
        "anchor",
        "candidate_id",
        "candidate_index",
        "candidate_source",
    ]
    for values, group in current.groupby(keys, sort=False):
        for horizon in horizons:
            selected = group[group["horizon_offset"] <= int(horizon)]
            if len(selected) != int(horizon):
                continue
            records.append(
                {
                    **dict(zip(keys, values)),
                    "horizon": int(horizon),
                    "true_exact_kl": float(selected["exact_kl"].sum()),
                    "projected_output_error": float(
                        selected["projected_output_error"].sum()
                    ),
                    **{
                        "risk_%s" % family: float(
                            selected["risk_%s" % family].sum()
                        )
                        for family in predictions
                    },
                }
            )
    return pd.DataFrame(records)


def _calibrate_action_pair_margin(
    actions: pd.DataFrame,
    predicted_column: str,
    level: float = 0.95,
) -> Tuple[float, int, bool]:
    sequence_scores = []
    for _, sequence_rows in actions.groupby("sample_id", sort=False):
        maximum = 0.0
        for _, context in sequence_rows.groupby(
            ["anchor", "horizon"], sort=False
        ):
            predicted = context[predicted_column].to_numpy(dtype=np.float64)
            truth = context["true_exact_kl"].to_numpy(dtype=np.float64)
            residual = predicted - truth
            if len(residual):
                maximum = max(
                    maximum,
                    float(residual.max() - residual.min()),
                )
        sequence_scores.append(maximum)
    return conformal_order_statistic(sequence_scores, level)


def _ranking_summary(segment: pd.DataFrame, cfg: DiscoveryConfig) -> Dict[str, Any]:
    per_context: List[Dict[str, Any]] = []
    group_columns = [
        "held_out_sequence",
        "task_bucket",
        "anchor",
        "horizon",
        "bridge_family",
    ]
    for keys, group in segment.groupby(group_columns, sort=False):
        true = group["true_exact_kl"].to_numpy(dtype=np.float64)
        predicted = group["predicted_objective"].to_numpy(dtype=np.float64)
        projected = group["projected_output_error"].to_numpy(dtype=np.float64)
        scale = max(float(true.max() - true.min()), 1e-12)
        chosen = int(np.argmin(predicted))
        per_context.append(
            {
                **dict(zip(group_columns, keys)),
                "spearman_exact_kl": _correlation(true, predicted, "spearman"),
                "kendall_exact_kl": _correlation(true, predicted, "kendall"),
                "spearman_projected_output": _correlation(
                    projected, predicted, "spearman"
                ),
                "top1_regret": float(true[chosen] - true.min()),
                "normalized_regret": float(
                    (true[chosen] - true.min()) / scale
                ),
                "top3_overlap": topk_overlap(true, predicted, 3),
                "selected_candidate": str(group.iloc[chosen]["candidate_source"]),
            }
        )
    context = pd.DataFrame(per_context)
    pooled = (
        context.groupby("bridge_family", as_index=False)
        .agg(
            median_spearman=("spearman_exact_kl", "median"),
            median_kendall=("kendall_exact_kl", "median"),
            median_projected_spearman=(
                "spearman_projected_output",
                "median",
            ),
            median_normalized_regret=("normalized_regret", "median"),
            median_top1_regret=("top1_regret", "median"),
            median_top3_overlap=("top3_overlap", "median"),
        )
    )
    task = (
        context.groupby(["bridge_family", "task_bucket"], as_index=False)
        .agg(
            median_spearman=("spearman_exact_kl", "median"),
            median_normalized_regret=("normalized_regret", "median"),
            median_top1_regret=("top1_regret", "median"),
            median_top3_overlap=("top3_overlap", "median"),
        )
    )
    by_horizon = (
        context.groupby(["bridge_family", "task_bucket", "horizon"], as_index=False)
        .agg(
            median_spearman=("spearman_exact_kl", "median"),
            median_normalized_regret=("normalized_regret", "median"),
        )
    )
    by_anchor = (
        context.groupby(["bridge_family", "task_bucket", "anchor"], as_index=False)
        .agg(
            median_spearman=("spearman_exact_kl", "median"),
            median_normalized_regret=("normalized_regret", "median"),
        )
    )
    lookup = pooled.set_index("bridge_family")["median_spearman"].to_dict()
    old = float(lookup.get("O0", float("nan")))
    increments = {
        family: float(value - old) for family, value in lookup.items()
    }
    task_lookup = task.set_index(
        ["bridge_family", "task_bucket"]
    )["median_spearman"].to_dict()
    task_increments = {
        family: {
            current_task: float(
                task_lookup.get((family, current_task), float("nan"))
                - task_lookup.get(("O0", current_task), float("nan"))
            )
            for current_task in ("NIAH", "GovReport")
        }
        for family in lookup
    }
    bootstrap = {}
    for (family, current_task), group in context.groupby(
        ["bridge_family", "task_bucket"], sort=False
    ):
        bootstrap["%s:%s" % (family, current_task)] = {
            metric: _sequence_cluster_bootstrap(
                group,
                metric,
                "held_out_sequence",
                int(cfg.runtime.bootstrap_samples),
                int(cfg.runtime.seed),
            )
            for metric in (
                "spearman_exact_kl",
                "normalized_regret",
                "top1_regret",
                "top3_overlap",
            )
        }
    return {
        "per_sequence_anchor_horizon": _json_records(context),
        "pooled": _json_records(pooled),
        "task_split": _json_records(task),
        "horizon_split": _json_records(by_horizon),
        "anchor_split": _json_records(by_anchor),
        "spearman_increment_over_old_e2": increments,
        "task_spearman_increment_over_old_e2": task_increments,
        "sequence_cluster_bootstrap_95ci": bootstrap,
        "strict_action_gate_threshold": float(
            cfg.output_sensitivity.action_spearman_increment_gate
        ),
    }


def _pairwise_rows(
    segment: pd.DataFrame,
    cfg: DiscoveryConfig,
    margin_lookup: Mapping[Tuple[str, str, int], Tuple[float, int, bool]],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    context_columns = [
        "held_out_sequence",
        "task",
        "task_bucket",
        "anchor",
        "horizon",
        "bridge_family",
    ]
    for keys, group in segment.groupby(context_columns, sort=False):
        lookup_key = (str(keys[0]), str(keys[5]), int(keys[4]))
        if lookup_key not in margin_lookup:
            continue
        ordered = group.sort_values("candidate_index").reset_index(drop=True)
        predicted = ordered["predicted_objective"].to_numpy(dtype=np.float64)
        truth = ordered["true_exact_kl"].to_numpy(dtype=np.float64)
        margin, rank, is_maximum = margin_lookup[lookup_key]
        for left in range(len(ordered)):
            for right in range(left + 1, len(ordered)):
                true_delta = float(truth[left] - truth[right])
                predicted_delta = pairwise_prediction(
                    predicted[left], predicted[right]
                )
                lower = predicted_delta - margin
                upper = predicted_delta + margin
                abstain, decision = dominance_decision(predicted_delta, margin)
                true_sign = int(np.sign(true_delta))
                records.append(
                    {
                        **dict(zip(context_columns, keys)),
                        "left_candidate_id": ordered.iloc[left]["candidate_id"],
                        "right_candidate_id": ordered.iloc[right]["candidate_id"],
                        "left_candidate_source": ordered.iloc[left][
                            "candidate_source"
                        ],
                        "right_candidate_source": ordered.iloc[right][
                            "candidate_source"
                        ],
                        "true_delta": true_delta,
                        "predicted_delta": predicted_delta,
                        "interval_lower": lower,
                        "interval_upper": upper,
                        "calibration_margin": float(margin),
                        "calibration_sequence_count": 8,
                        "conformal_rank": int(rank),
                        "conformal_is_maximum": bool(is_maximum),
                        "interval_covered": bool(
                            lower - 1e-12 <= true_delta <= upper + 1e-12
                        ),
                        "abstain": bool(abstain),
                        "dominance_sign": int(decision),
                        "true_sign": true_sign,
                        "sign_correct": bool(
                            int(np.sign(predicted_delta)) == true_sign
                        ),
                        "dominance_correct": bool(
                            (not abstain) and decision == true_sign
                        ),
                        "dominance_regret": float(
                            abs(true_delta)
                            if (not abstain) and decision != true_sign
                            else 0.0
                        ),
                        "same_sequence_cluster": True,
                        "bootstrap_unit": "sequence",
                    }
                )
    rows = pd.DataFrame(records)
    summary_rows: List[Dict[str, Any]] = []
    for keys, group in rows.groupby(
        ["bridge_family", "task_bucket"], sort=False
    ):
        decided = group[~group["abstain"]]
        true_positive = (
            (decided["dominance_sign"] != 0)
            & (decided["dominance_sign"] == decided["true_sign"])
        ).sum()
        possible = (group["true_sign"] != 0).sum()
        summary_rows.append(
            {
                "bridge_family": keys[0],
                "task_bucket": keys[1],
                "pairwise_sign_accuracy": float(group["sign_correct"].mean()),
                "pairwise_auc": pairwise_auc(
                    group["true_delta"], group["predicted_delta"]
                ),
                "interval_coverage": float(group["interval_covered"].mean()),
                "dominance_precision": float(
                    decided["dominance_correct"].mean()
                    if len(decided)
                    else float("nan")
                ),
                "dominance_recall": float(true_positive / max(possible, 1)),
                "abstention_rate": float(group["abstain"].mean()),
                "conditional_dominance_regret": float(
                    decided["dominance_regret"].mean()
                    if len(decided)
                    else float("nan")
                ),
                "decided_pairs": int(len(decided)),
                "pairs": int(len(group)),
            }
        )
    table = pd.DataFrame(summary_rows)
    best = (
        table[table["bridge_family"] != "O0"]
        .groupby("bridge_family")["pairwise_sign_accuracy"]
        .min()
        .sort_values(ascending=False)
    )
    best_family = str(best.index[0])
    strict = bool(
        best.iloc[0] >= cfg.output_sensitivity.pairwise_sign_accuracy_gate
    )
    partial = bool(
        best.iloc[0]
        >= cfg.output_sensitivity.partial_pairwise_sign_accuracy_gate
    )
    summary = {
        "independent_calibration_unit": "sequence",
        "candidate_pairs_are_independent": False,
        "table": _json_records(table),
        "best_worst_task_family": best_family,
        "best_worst_task_sign_accuracy": float(best.iloc[0]),
        "strict_pairwise_gate_passed": strict,
        "partial_pairwise_gate_passed": partial,
        "strict_threshold": float(
            cfg.output_sensitivity.pairwise_sign_accuracy_gate
        ),
        "partial_threshold": float(
            cfg.output_sensitivity.partial_pairwise_sign_accuracy_gate
        ),
        "sequence_cluster_bootstrap_95ci": {
            "%s:%s" % (family, task): {
                metric: _sequence_cluster_bootstrap(
                    rows[
                        (rows["bridge_family"] == family)
                        & (rows["task_bucket"] == task)
                    ],
                    metric,
                    "held_out_sequence",
                    int(cfg.runtime.bootstrap_samples),
                    int(cfg.runtime.seed),
                    reducer="mean",
                )
                for metric in (
                    "sign_correct",
                    "interval_covered",
                    "dominance_correct",
                    "abstain",
                )
            }
            for family, task in table[
                ["bridge_family", "task_bucket"]
            ].itertuples(index=False, name=None)
        },
    }
    return rows, summary


def _selection_rows(segment: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    context_columns = [
        "held_out_sequence",
        "task",
        "task_bucket",
        "anchor",
        "horizon",
        "bridge_family",
    ]
    for keys, group in segment.groupby(context_columns, sort=False):
        true = group["true_exact_kl"].to_numpy(dtype=np.float64)
        prediction = group["predicted_objective"].to_numpy(dtype=np.float64)
        chosen = int(np.argmin(prediction))
        oracle = int(np.argmin(true))
        scale = max(float(true.max() - true.min()), 1e-12)
        records.append(
            {
                **dict(zip(context_columns, keys)),
                "selected_candidate_id": group.iloc[chosen]["candidate_id"],
                "selected_candidate_source": group.iloc[chosen][
                    "candidate_source"
                ],
                "oracle_candidate_id": group.iloc[oracle]["candidate_id"],
                "oracle_candidate_source": group.iloc[oracle][
                    "candidate_source"
                ],
                "selected_exact_kl": float(true[chosen]),
                "oracle_exact_kl": float(true[oracle]),
                "top1_regret": float(true[chosen] - true[oracle]),
                "normalized_regret": float(
                    (true[chosen] - true[oracle]) / scale
                ),
                "top3_overlap": topk_overlap(true, prediction, 3),
                "candidate_count": int(len(group)),
            }
        )
    rows = pd.DataFrame(records)
    canonical = segment[segment["bridge_family"] == "O2"]
    fixed_sources = {
        "FIXED_attention": "attention",
        "FIXED_AOV": "aov",
        "FIXED_AOR": "aor",
        "FIXED_direct_greedy": "direct_energy_greedy",
        "FIXED_v_ridge": "v_ridge",
        "FIXED_old_core": "preceding_anchor_old_core",
    }
    fixed_records: List[Dict[str, Any]] = []
    for keys, group in canonical.groupby(
        [
            "held_out_sequence",
            "task",
            "task_bucket",
            "anchor",
            "horizon",
        ],
        sort=False,
    ):
        truth = group["true_exact_kl"].to_numpy(dtype=np.float64)
        oracle = float(truth.min())
        scale = max(float(truth.max() - truth.min()), 1e-12)
        for family, source in fixed_sources.items():
            selected = group[group["candidate_source"] == source].iloc[0]
            value = float(selected["true_exact_kl"])
            fixed_records.append(
                {
                    **dict(
                        zip(
                            [
                                "held_out_sequence",
                                "task",
                                "task_bucket",
                                "anchor",
                                "horizon",
                            ],
                            keys,
                        )
                    ),
                    "bridge_family": family,
                    "selected_candidate_id": selected["candidate_id"],
                    "selected_candidate_source": source,
                    "oracle_candidate_id": str(
                        group.loc[
                            group["true_exact_kl"].idxmin(),
                            "candidate_id",
                        ]
                    ),
                    "oracle_candidate_source": str(
                        group.loc[
                            group["true_exact_kl"].idxmin(),
                            "candidate_source",
                        ]
                    ),
                    "selected_exact_kl": value,
                    "oracle_exact_kl": oracle,
                    "top1_regret": value - oracle,
                    "normalized_regret": (value - oracle) / scale,
                    "top3_overlap": float(
                        source
                        in set(
                            group.nsmallest(3, "true_exact_kl")[
                                "candidate_source"
                            ]
                        )
                    )
                    / 3.0,
                    "candidate_count": int(len(group)),
                }
            )
    rows = pd.concat(
        [rows, pd.DataFrame(fixed_records)], ignore_index=True, sort=False
    )
    summary = (
        rows.groupby(["bridge_family", "task_bucket"], as_index=False)
        .agg(
            median_selected_kl=("selected_exact_kl", "median"),
            median_top1_regret=("top1_regret", "median"),
            median_normalized_regret=("normalized_regret", "median"),
            median_top3_overlap=("top3_overlap", "median"),
        )
    )
    return rows, {"table": _json_records(summary)}


def _strict_action_gate(
    ranking: Mapping[str, Any],
    pairwise: Mapping[str, Any],
) -> Dict[str, Any]:
    rank_task = pd.DataFrame(ranking["task_split"])
    rank_pooled = pd.DataFrame(ranking["pooled"])
    candidates = [
        family
        for family in (
            "O1",
            "O2",
            "O2_NO_L27",
            "O3",
            "O4_CONT",
            "O4_REGIME",
            "JAC_FD",
            "PAIR_CORR",
        )
        if family in set(rank_task["bridge_family"])
    ]
    worst = (
        rank_task[rank_task["bridge_family"].isin(candidates)]
        .groupby("bridge_family")["median_spearman"]
        .min()
        .sort_values(ascending=False)
    )
    best = str(worst.index[0])
    old_task = rank_task[rank_task["bridge_family"] == "O0"].set_index(
        "task_bucket"
    )
    best_task = rank_task[rank_task["bridge_family"] == best].set_index(
        "task_bucket"
    )
    pooled = rank_pooled.set_index("bridge_family")
    increments = {
        task: float(
            best_task.loc[task, "median_spearman"]
            - old_task.loc[task, "median_spearman"]
        )
        for task in ("NIAH", "GovReport")
    }
    pair_table = pd.DataFrame(pairwise["table"])
    pair_best = pair_table[pair_table["bridge_family"] == best]
    pair_both = bool(
        len(pair_best) == 2
        and (
            pair_best["pairwise_sign_accuracy"] >= 0.65
        ).all()
    )
    no_layer27 = rank_task[
        rank_task["bridge_family"] == "O2_NO_L27"
    ].set_index("task_bucket")
    no_layer27_increment = {
        task: float(
            no_layer27.loc[task, "median_spearman"]
            - old_task.loc[task, "median_spearman"]
        )
        for task in ("NIAH", "GovReport")
    }
    checks = {
        "pooled_spearman_increment_at_least_0_05": bool(
            float(
                pooled.loc[best, "median_spearman"]
                - pooled.loc["O0", "median_spearman"]
            )
            >= 0.05
        ),
        "both_task_spearman_increments_nonnegative": bool(
            all(value >= 0.0 for value in increments.values())
        ),
        "both_task_normalized_regret_lower": bool(
            all(
                best_task.loc[task, "median_normalized_regret"]
                < old_task.loc[task, "median_normalized_regret"]
                for task in ("NIAH", "GovReport")
            )
        ),
        "not_layer27_only": bool(
            all(value >= 0.0 for value in no_layer27_increment.values())
        ),
        "pairwise_sign_accuracy_both_at_least_0_65": pair_both,
        "top1_regret_better_both_tasks": bool(
            all(
                best_task.loc[task, "median_top1_regret"]
                < old_task.loc[task, "median_top1_regret"]
                for task in ("NIAH", "GovReport")
            )
        ),
    }
    return {
        "best_new_family_by_worst_task_spearman": best,
        "task_spearman_increment_over_O0": increments,
        "O2_without_layer27_task_increment_over_O0": no_layer27_increment,
        "checks": checks,
        "passed": bool(all(checks.values())),
    }


def _empty_stage_d(
    run_dir: Path,
    reason: str,
    stage_b_partial: bool,
    stage_c_partial: bool,
) -> None:
    refresh_columns = [
        "sample_id",
        "task",
        "policy",
        "horizon",
        "maximum_refresh_count",
        "actual_refresh_count",
        "cumulative_exact_kl",
    ]
    _atomic_frame(
        pd.DataFrame(columns=refresh_columns),
        run_dir / "refresh_lcb_policy_rows.parquet",
    )
    atomic_json(
        run_dir / "refresh_lcb_policy_summary.json",
        {
            "status": "not_run_by_preregistered_gate",
            "reason": reason,
            "stage_b_partial_passed": bool(stage_b_partial),
            "stage_c_partial_passed": bool(stage_c_partial),
            "matched_refresh_count_evaluated": False,
        },
    )
    atomic_json(
        run_dir / "free_generation_results.json",
        {
            "status": "not_run_by_preregistered_gate",
            "reason": reason,
            "teacher_forced_and_free_generation_separated": True,
        },
    )


def analyze_output_sensitivity(cfg: DiscoveryConfig, run_dir: Path) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    raw = pd.read_parquet(run_dir / "output_candidate_rows.parquet")
    inventory = pd.read_parquet(run_dir / "output_candidate_inventory.parquet")
    probes = pd.read_parquet(run_dir / "output_jacobian_probe_rows.parquet")
    if raw["sample_id"].nunique() < 24:
        raise RuntimeError("at least 24 independent sequences are required")
    if inventory.groupby(["sample_id", "anchor"])["candidate_id"].nunique().min() != 24:
        raise RuntimeError("each sequence-anchor must have 24 candidates")
    if inventory.groupby(["sample_id", "anchor"])["mask_hash"].nunique().min() != 24:
        raise RuntimeError(
            "each sequence-anchor must have 24 distinct physical masks"
        )
    if not bool(raw["token_position_aligned"].all()):
        raise RuntimeError("token positions are not aligned")
    if bool(raw["uses_future_compressed_truth"].any()):
        raise RuntimeError("E2 inputs read compressed future truth")
    if bool(inventory["uses_task_feature"].any()):
        raise RuntimeError("task id entered a candidate feature")
    if inventory.groupby(["sample_id", "anchor"])["total_budget"].nunique().max() != 1:
        raise RuntimeError("candidate budgets differ")
    wide, layers = _wide_rows(raw)
    if layers != list(cfg.output_sensitivity.diagnostic_layers):
        raise RuntimeError("diagnostic layers differ from pre-registration")
    sequences = sorted(wide["sample_id"].unique())
    task_by_sequence = wide.groupby("sample_id")["task"].first().to_dict()
    gain_table = _jacobian_gain_table(probes)
    output_records: List[Dict[str, Any]] = []
    state_records: List[Dict[str, Any]] = []
    correction_segment_records: List[Dict[str, Any]] = []
    pairwise_margin_lookup: Dict[
        Tuple[str, str, int], Tuple[float, int, bool]
    ] = {}
    fold_models: Dict[str, Any] = {}
    for held_out in sequences:
        fit_sequences, state_sequences, calibration_sequences = (
            nested_sequence_partition(sequences, held_out, task_by_sequence)
        )
        fit_previous, fit_direct, fit_target, _ = _transition_arrays(
            wide, layers, fit_sequences
        )
        state_previous, state_direct, state_target, state_ids = _transition_arrays(
            wide, layers, state_sequences
        )
        e2 = fit_nonnegative_envelope(
            fit_previous, fit_direct, fit_target, layers, "E2"
        )
        state_prediction = np.stack(
            [
                e2.step(previous, direct)
                for previous, direct in zip(state_previous, state_direct)
            ]
        )
        state_margin, state_rank, state_is_max = clustered_coordinate_margin(
            state_target - state_prediction, state_ids, 0.95
        )
        fold_bound = _attach_recursive_e2(wide, layers, e2, state_margin)
        training = fold_bound[
            fold_bound["sample_id"].isin(fit_sequences)
            & (fold_bound["trajectory_kind"] == "candidate")
        ]
        calibration = fold_bound[
            fold_bound["sample_id"].isin(calibration_sequences)
            & (fold_bound["trajectory_kind"] == "candidate")
        ]
        test = fold_bound[
            (fold_bound["sample_id"] == held_out)
            & (fold_bound["trajectory_kind"] == "candidate")
        ].copy()
        bridges = fit_bridges(training, layers)
        fold_models[held_out] = {
            "fit_sequences": fit_sequences,
            "state_margin_sequences": state_sequences,
            "output_calibration_sequences": calibration_sequences,
            "state_conformal_rank": state_rank,
            "state_conformal_is_maximum": state_is_max,
            "state_margin": state_margin.tolist(),
            "e2": {
                "a": e2.a.tolist(),
                "b": e2.b.tolist(),
                "nonnegative": bool(
                    np.all(e2.a >= 0.0) and np.all(e2.b >= 0.0)
                ),
            },
            "bridges": {},
        }
        training_predictions: Dict[str, np.ndarray] = {}
        calibration_predictions: Dict[str, np.ndarray] = {}
        test_predictions: Dict[str, np.ndarray] = {}
        output_margins: Dict[str, float] = {}
        for family in BRIDGE_FAMILIES:
            if family == "JAC_FD":
                training_prediction = _jacobian_prediction(
                    training, gain_table, layers
                )
                calibration_prediction = _jacobian_prediction(
                    calibration, gain_table, layers
                )
                test_prediction = _jacobian_prediction(test, gain_table, layers)
                bridge_metadata = {
                    "source": "symmetric_finite_direction_maximum",
                    "claimed_operator_norm": False,
                    "primary_relative_radius": float(
                        sorted(probes["relative_radius"].unique())[
                            len(probes["relative_radius"].unique()) // 2
                        ]
                    ),
                }
            else:
                bridge = bridges[family]
                training_prediction = bridge.predict(training, layers)
                calibration_prediction = bridge.predict(calibration, layers)
                test_prediction = bridge.predict(test, layers)
                bridge_metadata = {
                    "coefficient": bridge.coefficient.tolist(),
                    "intercept": float(bridge.intercept),
                    "metadata": bridge.metadata,
                    "nonnegative": bridge_coefficients_nonnegative(bridge),
                }
            training_predictions[family] = training_prediction
            calibration_predictions[family] = calibration_prediction
            test_predictions[family] = test_prediction
            margin, rank, is_maximum = clustered_additive_margin(
                calibration["logit_l2"].to_numpy() - calibration_prediction,
                calibration["sample_id"].astype(str).to_numpy(),
                0.95,
            )
            output_margins[family] = float(margin)
            pairwise_margins = _calibrate_pairwise_margins(
                calibration,
                calibration_prediction,
                margin,
                cfg.output_sensitivity.evaluation_horizons,
            )
            for horizon, values in pairwise_margins.items():
                pairwise_margin_lookup[
                    (held_out, family, int(horizon))
                ] = values
            fold_models[held_out]["bridges"][family] = {
                **bridge_metadata,
                "additive_margin_95": margin,
                "conformal_rank": rank,
                "conformal_is_maximum": is_maximum,
                "pairwise_margin_95": {
                    str(horizon): {
                        "margin": float(values[0]),
                        "rank": int(values[1]),
                        "is_maximum": bool(values[2]),
                    }
                    for horizon, values in pairwise_margins.items()
                },
            }
            bound = test_prediction + margin
            for row_index, (_, row) in enumerate(test.iterrows()):
                logit = float(row["logit_l2"])
                exact_kl = float(row["exact_kl"])
                kl_bound = 0.25 * float(bound[row_index] ** 2)
                output_records.append(
                    {
                        "held_out_sequence": held_out,
                        "task": row["task"],
                        "task_bucket": _task_bucket(row["task"]),
                        "trajectory_id": row["trajectory_id"],
                        "candidate_id": row["candidate_id"],
                        "candidate_index": int(row["candidate_index"]),
                        "candidate_source": row["candidate_source"],
                        "anchor": int(row["anchor"]),
                        "horizon_offset": int(row["horizon_offset"]),
                        "bridge_family": family,
                        "raw_logit_prediction": float(
                            test_prediction[row_index]
                        ),
                        "additive_margin": float(margin),
                        "logit_bound": float(bound[row_index]),
                        "realized_logit_l2": logit,
                        "logit_l2_sq": float(row["logit_l2_sq"]),
                        "logit_covered": bool(
                            logit <= float(bound[row_index]) + 1e-12
                        ),
                        "logit_looseness": float(
                            (float(bound[row_index]) + 1e-9)
                            / (logit + 1e-9)
                        ),
                        "logit_slack": float(
                            float(bound[row_index]) - logit
                        ),
                        "induced_kl_bound": kl_bound,
                        "exact_kl": exact_kl,
                        "kl_covered": bool(exact_kl <= kl_bound + 1e-12),
                        "kl_slack": float(kl_bound - exact_kl),
                        "kl_looseness": float(
                            (kl_bound + 1e-12) / (exact_kl + 1e-12)
                        ),
                        "js": float(row["js"]),
                        "delta_nll": float(row["delta_nll"]),
                        "projected_output_error": float(
                            row["projected_output_error"]
                        ),
                        "dynamic_direct_energy": float(
                            sum(
                                float(row["d_l%d" % layer]) ** 2
                                for layer in layers
                            )
                        ),
                        "deleted_attention_mass_total": float(
                            row["deleted_attention_mass_total"]
                        ),
                        "e2_exploded": bool(row["e2_exploded"]),
                        "calibration_sequences": len(calibration_sequences),
                        "conformal_rank": int(rank),
                        "conformal_is_maximum": bool(is_maximum),
                        "task_feature_used": False,
                        "future_label_used": False,
                    }
                )
        zero_margins = {family: 0.0 for family in BRIDGE_FAMILIES}
        fit_actions = _action_feature_rows(
            training,
            training_predictions,
            zero_margins,
            cfg.output_sensitivity.evaluation_horizons,
        )
        calibration_actions = _action_feature_rows(
            calibration,
            calibration_predictions,
            zero_margins,
            cfg.output_sensitivity.evaluation_horizons,
        )
        test_actions = _action_feature_rows(
            test,
            test_predictions,
            zero_margins,
            cfg.output_sensitivity.evaluation_horizons,
        )
        risk_columns = ["risk_%s" % family for family in BRIDGE_FAMILIES]
        correction_coefficient, correction_intercept = _fit_nnls(
            fit_actions[risk_columns].to_numpy(dtype=np.float64),
            fit_actions["true_exact_kl"].to_numpy(dtype=np.float64),
        )
        calibration_actions["pair_corr_objective"] = np.maximum(
            0.0,
            calibration_actions[risk_columns].to_numpy(dtype=np.float64)
            @ correction_coefficient
            + correction_intercept,
        )
        test_actions["pair_corr_objective"] = np.maximum(
            0.0,
            test_actions[risk_columns].to_numpy(dtype=np.float64)
            @ correction_coefficient
            + correction_intercept,
        )
        correction_margin, correction_rank, correction_is_max = (
            _calibrate_action_pair_margin(
                calibration_actions, "pair_corr_objective"
            )
        )
        for horizon in cfg.output_sensitivity.evaluation_horizons:
            pairwise_margin_lookup[
                (held_out, "PAIR_CORR", int(horizon))
            ] = (
                float(correction_margin),
                int(correction_rank),
                bool(correction_is_max),
            )
        fold_models[held_out]["pairwise_monotone_correction"] = {
            "input_families": list(BRIDGE_FAMILIES),
            "coefficient": correction_coefficient.tolist(),
            "intercept": float(correction_intercept),
            "nonnegative": bool(
                np.all(correction_coefficient >= -1e-12)
                and correction_intercept >= -1e-12
            ),
            "task_feature_used": False,
            "calibration_margin_95": float(correction_margin),
            "conformal_rank": int(correction_rank),
            "conformal_is_maximum": bool(correction_is_max),
        }
        for row in test_actions.itertuples():
            correction_segment_records.append(
                {
                    "held_out_sequence": held_out,
                    "task": row.task,
                    "task_bucket": _task_bucket(row.task),
                    "anchor": int(row.anchor),
                    "candidate_id": row.candidate_id,
                    "candidate_index": int(row.candidate_index),
                    "candidate_source": row.candidate_source,
                    "bridge_family": "PAIR_CORR",
                    "horizon": int(row.horizon),
                    "predicted_objective": float(row.pair_corr_objective),
                    "raw_predicted_objective": float(
                        row.pair_corr_objective
                    ),
                    "true_exact_kl": float(row.true_exact_kl),
                    "projected_output_error": float(
                        row.projected_output_error
                    ),
                    "js": float("nan"),
                    "delta_nll": float("nan"),
                }
            )
        state_test = fold_bound[
            (fold_bound["sample_id"] == held_out)
            & (fold_bound["trajectory_kind"] == "state_reference")
        ].sort_values("horizon_offset")
        trajectory_all = True
        temporary: List[Dict[str, Any]] = []
        for _, row in state_test.iterrows():
            for layer in layers:
                realized = float(row["e_l%d" % layer])
                bound_value = float(row["b_l%d" % layer])
                covered = realized <= bound_value + 1e-12
                trajectory_all = bool(trajectory_all and covered)
                temporary.append(
                    {
                        "held_out_sequence": held_out,
                        "task": row["task"],
                        "task_bucket": _task_bucket(row["task"]),
                        "horizon_offset": int(row["horizon_offset"]),
                        "layer": int(layer),
                        "realized": realized,
                        "bound": bound_value,
                        "pointwise_covered": covered,
                        "looseness": float(
                            (bound_value + 1e-9) / (realized + 1e-9)
                        ),
                        "exploded": bool(row["e2_exploded"]),
                        "recursion_used_compressed_future_truth": False,
                    }
                )
        for value in temporary:
            value["trajectory_covered"] = trajectory_all
        state_records.extend(temporary)
    output_rows = pd.DataFrame(output_records)
    state_rows = pd.DataFrame(state_records)
    _atomic_frame(output_rows, run_dir / "output_bridge_rows.parquet")
    _atomic_frame(state_rows, run_dir / "output_state_envelope_rows.parquet")
    atomic_json(run_dir / "output_bridge_fold_models.json", fold_models)
    coverage, tightness = _summarize_bridge_rows(
        output_rows, state_rows, probes, cfg
    )
    atomic_json(run_dir / "output_bridge_coverage_summary.json", coverage)
    atomic_json(run_dir / "output_bridge_tightness_summary.json", tightness)
    segment = _candidate_segment_rows(
        output_rows, cfg.output_sensitivity.evaluation_horizons
    )
    segment = pd.concat(
        [
            segment,
            _analytical_baseline_segment_rows(
                output_rows,
                cfg.output_sensitivity.evaluation_horizons,
            ),
            pd.DataFrame(correction_segment_records),
        ],
        ignore_index=True,
        sort=False,
    )
    ranking = _ranking_summary(segment, cfg)
    canonical_actions = segment[segment["bridge_family"] == "O2"]
    action_ranges = (
        canonical_actions.groupby(
            ["held_out_sequence", "anchor", "horizon"], as_index=False
        )
        .agg(
            exact_kl_range=(
                "true_exact_kl",
                lambda value: float(value.max() - value.min()),
            ),
            projected_output_range=(
                "projected_output_error",
                lambda value: float(value.max() - value.min()),
            ),
        )
    )
    ranking["candidate_pool_diagnostics"] = {
        "registered_candidates_per_sequence_anchor": int(
            inventory.groupby(["sample_id", "anchor"])[
                "candidate_id"
            ].nunique().min()
        ),
        "distinct_physical_masks_per_sequence_anchor": int(
            inventory.groupby(["sample_id", "anchor"])[
                "mask_hash"
            ].nunique().min()
        ),
        "candidate_sources": sorted(
            str(value) for value in inventory["candidate_source"].unique()
        ),
        "median_exact_kl_action_range": float(
            action_ranges["exact_kl_range"].median()
        ),
        "median_projected_output_action_range": float(
            action_ranges["projected_output_range"].median()
        ),
    }
    atomic_json(run_dir / "output_bridge_ranking_summary.json", ranking)
    pairwise, pairwise_summary = _pairwise_rows(
        segment, cfg, pairwise_margin_lookup
    )
    _atomic_frame(pairwise, run_dir / "pairwise_action_rows.parquet")
    atomic_json(
        run_dir / "pairwise_action_calibration_summary.json",
        pairwise_summary,
    )
    selection_rows, selection_summary = _selection_rows(segment)
    selection_summary["strict_action_gate"] = _strict_action_gate(
        ranking, pairwise_summary
    )
    _atomic_frame(selection_rows, run_dir / "selection_policy_rows.parquet")
    atomic_json(
        run_dir / "selection_policy_summary.json", selection_summary
    )
    bridge_table = pd.DataFrame(coverage["output_bridge"])
    coverage_min = (
        bridge_table.groupby("bridge_family")["logit_coverage"].min()
    )
    tight_table = pd.DataFrame(tightness["table"])
    looseness_max = (
        tight_table.groupby("bridge_family")[
            "median_logit_looseness"
        ].max()
    )
    rank_table = pd.DataFrame(ranking["task_split"])
    rank_min = rank_table.groupby("bridge_family")[
        "median_spearman"
    ].min()
    stage_b_candidates = [
        family
        for family in BRIDGE_FAMILIES
        if family != "O0"
        and coverage_min.get(family, 0.0)
        >= cfg.output_sensitivity.output_pointwise_coverage_gate
        and looseness_max.get(family, float("inf"))
        <= cfg.output_sensitivity.output_looseness_gate
        and rank_min.get(family, -1.0) >= 0.0
    ]
    stage_b_partial = bool(stage_b_candidates)
    stage_c_partial = bool(pairwise_summary["partial_pairwise_gate_passed"])
    diversity_rows = []
    for _, group in segment[segment["bridge_family"] == "O0"].groupby(
        ["held_out_sequence", "anchor", "horizon"], sort=False
    ):
        # Diagnose state-coordinate collisions from raw O0 state energy.  A
        # calibrated output margin is common to candidates and can otherwise
        # spuriously suppress action variation.
        predicted = group["raw_predicted_objective"].to_numpy(
            dtype=np.float64
        )
        truth = group["true_exact_kl"].to_numpy(dtype=np.float64)
        predicted_cv = float(
            np.std(predicted) / max(abs(np.mean(predicted)), 1e-12)
        )
        truth_cv = float(np.std(truth) / max(abs(np.mean(truth)), 1e-12))
        diversity_rows.append(
            {
                "predicted_cv": predicted_cv,
                "truth_cv": truth_cv,
                "state_feature_collision": bool(
                    predicted_cv < 0.05 and truth_cv > 0.10
                ),
            }
        )
    collision_rate = float(
        pd.DataFrame(diversity_rows)["state_feature_collision"].mean()
    )
    rank_task_frame = pd.DataFrame(ranking["task_split"])
    pair_table_frame = pd.DataFrame(pairwise_summary["table"])
    best_abstention = float(
        pair_table_frame[
            pair_table_frame["bridge_family"]
            == pairwise_summary["best_worst_task_family"]
        ]["abstention_rate"].max()
    )
    o2_worst = float(
        rank_task_frame[
            rank_task_frame["bridge_family"] == "O2"
        ]["median_spearman"].min()
    )
    o3_worst = float(
        rank_task_frame[
            rank_task_frame["bridge_family"] == "O3"
        ]["median_spearman"].min()
    )
    o4_worst = float(
        rank_task_frame[
            rank_task_frame["bridge_family"].isin(
                ["O4_CONT", "O4_REGIME"]
            )
        ]
        .groupby("bridge_family")["median_spearman"]
        .min()
        .max()
    )
    if not coverage["state_e2_gate"]["pointwise_coverage_at_least_0_90_both_tasks"]:
        primary_failure = "F1_state_envelope_information_or_validity"
    elif not stage_b_partial:
        primary_failure = "F2_output_bridge"
    elif o4_worst > o2_worst + 0.05:
        primary_failure = "F4_context_regime"
    elif o3_worst > o2_worst + 0.05:
        primary_failure = "F3_scalarization"
    elif not stage_c_partial or best_abstention > 0.80:
        primary_failure = "F5_pairwise_uncertainty"
    elif collision_rate > 0.50:
        primary_failure = "F1_state_information"
    else:
        primary_failure = "none_before_free_generation"
    # Stage D requires real stateful policy and generation, not a replay label.
    # The runner is deliberately gated here; a failed B/C must yield explicit
    # skipped artifacts rather than post-hoc policy claims.
    _empty_stage_d(
        run_dir,
        (
            "Stage B and Stage C partial gates must both pass before stateful "
            "refresh/free-generation execution; conditional runner has not "
            "been invoked."
            if stage_b_partial and stage_c_partial
            else "Stage B/C preregistered partial gate failed."
        ),
        stage_b_partial,
        stage_c_partial,
    )
    decision = {
        "stage_b_partial_passed": stage_b_partial,
        "stage_b_passing_families": stage_b_candidates,
        "stage_c_partial_passed": stage_c_partial,
        "stage_d_executed": False,
        "stage_d_requires_conditional_stateful_runner": bool(
            stage_b_partial and stage_c_partial
        ),
        "preliminary_failure_diagnosis": {
            "primary": primary_failure,
            "state_feature_collision_rate": collision_rate,
            "best_pairwise_abstention_rate": best_abstention,
            "O2_worst_task_spearman": o2_worst,
            "O3_worst_task_spearman": o3_worst,
            "O4_best_worst_task_spearman": o4_worst,
            "F6_requires_free_generation": True,
        },
    }
    atomic_json(run_dir / "output_sensitivity_gate_decision.json", decision)
    return decision
