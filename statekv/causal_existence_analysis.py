"""Metrics and dataset views for causal future-utility prediction."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


def future_attention_labels(
    attention: np.ndarray,
    position_ids: np.ndarray,
    position_lengths: np.ndarray,
    horizons: Sequence[int],
) -> Dict[int, np.ndarray]:
    """Sum future attention for tokens that exist at the decision boundary.

    The output for horizon ``h`` has the same shape as ``attention`` and is NaN
    where a complete future window is unavailable or where a token has not yet
    appeared. Position identities are checked rather than assumed from columns.
    """

    values = np.asarray(attention, dtype=np.float64)
    if values.ndim != 4:
        raise ValueError("attention must be [cycle, layer, head, token]")
    cycles, layers, heads, width = values.shape
    positions = np.asarray(position_ids)
    lengths = np.asarray(position_lengths)
    if positions.shape != (cycles, width) or lengths.shape != (cycles,):
        raise ValueError("position metadata is misaligned")
    output: Dict[int, np.ndarray] = {}
    for horizon in horizons:
        h = int(horizon)
        if h <= 0:
            raise ValueError("future horizon must be positive")
        labels = np.full(values.shape, np.nan, dtype=np.float32)
        for cycle in range(cycles - h):
            current_count = int(lengths[cycle])
            current_positions = positions[cycle, :current_count]
            accumulator = np.zeros((layers, heads, current_count), dtype=np.float64)
            for future_cycle in range(cycle + 1, cycle + h + 1):
                future_count = int(lengths[future_cycle])
                row_by_position = {
                    int(position): row
                    for row, position in enumerate(
                        positions[future_cycle, :future_count].tolist()
                    )
                }
                try:
                    columns = [row_by_position[int(position)] for position in current_positions]
                except KeyError as exc:
                    raise ValueError("a causal candidate disappeared from full-pool labels") from exc
                accumulator += np.take(
                    values[future_cycle], columns, axis=-1
                )
            labels[cycle, :, :, :current_count] = accumulator.astype(np.float32)
        output[h] = labels
    return output


def topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    take = min(max(0, int(k)), int(scores.size))
    if take == 0:
        return np.asarray([], dtype=np.int64)
    # Stable position order is the deterministic tie breaker.
    return np.lexsort((np.arange(scores.size), -scores))[:take]


def ndcg_at_k(truth: np.ndarray, prediction: np.ndarray, k: int) -> float:
    truth = np.asarray(truth, dtype=np.float64)
    chosen = topk_indices(prediction, k)
    ideal = topk_indices(truth, k)
    if ideal.size == 0:
        return float("nan")
    discounts = 1.0 / np.log2(np.arange(2, ideal.size + 2))
    dcg = float(np.sum(truth[chosen] * discounts[: chosen.size]))
    ideal_dcg = float(np.sum(truth[ideal] * discounts))
    return dcg / ideal_dcg if ideal_dcg > 0.0 else 1.0


def deterministic_pairwise_accuracy(
    truth: np.ndarray, prediction: np.ndarray, maximum_pairs: int = 4096
) -> float:
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    count = int(truth.size)
    if count < 2:
        return float("nan")
    total_pairs = count * (count - 1) // 2
    if total_pairs <= int(maximum_pairs):
        left, right = np.triu_indices(count, k=1)
    else:
        # Deterministic coprime strides cover the pair space without allocating
        # its quadratic upper triangle.
        index = np.arange(int(maximum_pairs), dtype=np.int64)
        left = (index * 104729 + 17) % count
        right = (index * 130363 + count // 2 + 1) % count
        same = left == right
        right[same] = (right[same] + 1) % count
    informative = truth[left] != truth[right]
    left, right = left[informative], right[informative]
    if left.size == 0:
        return 1.0
    true_order = truth[left] > truth[right]
    predicted_order = prediction[left] > prediction[right]
    predicted_tie = prediction[left] == prediction[right]
    return float(np.mean(np.where(predicted_tie, 0.5, predicted_order == true_order)))


def boundary_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    baseline: np.ndarray,
    k: int,
) -> Dict[str, float]:
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    baseline = np.asarray(baseline, dtype=np.float64)
    if not (truth.shape == prediction.shape == baseline.shape):
        raise ValueError("truth, prediction, and baseline must align")
    if not (
        np.isfinite(truth).all()
        and np.isfinite(prediction).all()
        and np.isfinite(baseline).all()
    ):
        raise ValueError("boundary metrics require finite arrays")
    oracle = topk_indices(truth, k)
    selected = topk_indices(prediction, k)
    fixed = topk_indices(baseline, k)
    oracle_value = float(truth[oracle].sum())
    selected_value = float(truth[selected].sum())
    baseline_value = float(truth[fixed].sum())
    denominator = oracle_value - baseline_value
    recovery = (
        (selected_value - baseline_value) / denominator
        if denominator > 1.0e-12
        else 0.0
    )
    rho = spearmanr(truth, prediction).statistic
    return {
        "future_topk_recall": float(len(set(oracle) & set(selected)) / max(1, len(oracle))),
        "spearman": float(0.0 if not np.isfinite(rho) else rho),
        "pairwise_accuracy": deterministic_pairwise_accuracy(truth, prediction),
        "ndcg": ndcg_at_k(truth, prediction, k),
        "oracle_value": oracle_value,
        "selected_value": selected_value,
        "baseline_value": baseline_value,
        "oracle_gap_recovery": float(recovery),
        "beats_baseline": float(selected_value > baseline_value),
    }


def causal_scalar_features(
    attention: np.ndarray,
    cycle: int,
    count: int,
) -> np.ndarray:
    """Runtime-legal history features for one layer/head/token set."""

    values = np.asarray(attention, dtype=np.float64)
    current = values[int(cycle), :count]
    history = values[: int(cycle) + 1, :count]
    lag1 = values[max(0, int(cycle) - 1), :count]
    lag4 = values[max(0, int(cycle) - 4), :count]
    mean = history.mean(axis=0)
    maximum = history.max(axis=0)
    slope = current - lag4
    ema = history[0].copy()
    for row in history[1:]:
        ema = 0.9 * ema + 0.1 * row
    age = np.linspace(1.0, 0.0, num=count, endpoint=True)
    rank = rankdata(-current, method="average") / max(1, count)
    return np.stack(
        [current, lag1, lag4, mean, maximum, slope, ema, age, rank], axis=1
    ).astype(np.float32)


def aggregate_sequence_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate boundary evidence before forming the oracle-gap ratio."""

    keys = ["sample_id", "task", "split", "method", "future_horizon"]
    mean_columns = [
        "future_topk_recall",
        "spearman",
        "pairwise_accuracy",
        "ndcg",
    ]
    available_means = [column for column in mean_columns if column in frame]
    means = frame.groupby(keys, as_index=False)[available_means].mean()
    totals = frame.groupby(keys, as_index=False)[
        ["oracle_value", "selected_value", "baseline_value"]
    ].sum()
    output = means.merge(totals, on=keys, validate="one_to_one")
    denominator = output["oracle_value"] - output["baseline_value"]
    output["oracle_gap_recovery"] = np.where(
        denominator > 1.0e-12,
        (output["selected_value"] - output["baseline_value"]) / denominator,
        0.0,
    )
    output["beats_baseline"] = (
        output["selected_value"] > output["baseline_value"]
    ).astype(float)
    return output


__all__ = [
    "boundary_metrics",
    "aggregate_sequence_metrics",
    "causal_scalar_features",
    "deterministic_pairwise_accuracy",
    "future_attention_labels",
    "ndcg_at_k",
    "topk_indices",
]
