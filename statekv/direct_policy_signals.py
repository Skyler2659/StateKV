"""Training-free direct-policy signals from access, geometry, and position."""
from __future__ import annotations

import numpy as np


def attention_head_peak_score(attention_bank: np.ndarray) -> np.ndarray:
    """Retain tokens used strongly by any recent query head."""

    bank = np.asarray(attention_bank, dtype=np.float64)
    if bank.ndim != 3 or bank.shape[0] == 0:
        raise ValueError("attention_bank must have shape [queries, heads, tokens]")
    return np.max(bank, axis=(0, 1))


def attention_temporal_volatility_score(attention_bank: np.ndarray) -> np.ndarray:
    """Measure changes in head-averaged access across recent queries."""

    bank = np.asarray(attention_bank, dtype=np.float64)
    if bank.ndim != 3 or bank.shape[0] == 0:
        raise ValueError("attention_bank must have shape [queries, heads, tokens]")
    return np.std(np.mean(bank, axis=1), axis=0)


def diagonal_leverage_score(
    features: np.ndarray, eligible_rows: np.ndarray
) -> np.ndarray:
    """Return diagonal-whitened feature leverage without fitted parameters."""

    array = np.asarray(features, dtype=np.float64)
    rows = np.asarray(eligible_rows, dtype=np.int64)
    if array.ndim != 3:
        raise ValueError("features must have shape [heads, tokens, dimension]")
    if rows.size == 0:
        raise ValueError("eligible_rows must be nonempty")
    if rows.min() < 0 or rows.max() >= array.shape[1]:
        raise ValueError("eligible row is outside the feature tensor")
    eligible = array[:, rows, :]
    center = np.mean(eligible, axis=1, keepdims=True)
    variance = np.mean(np.square(eligible - center), axis=1, keepdims=True)
    floor = max(float(np.mean(variance)) * 1.0e-6, 1.0e-12)
    whitened = np.square(array - center) / np.maximum(variance, floor)
    return np.mean(whitened, axis=(0, 2))


def adjacent_value_change_score(values: np.ndarray) -> np.ndarray:
    """Score representation boundaries by adjacent value-vector change."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 3:
        raise ValueError("values must have shape [heads, tokens, dimension]")
    score = np.zeros(int(array.shape[1]), dtype=np.float64)
    if array.shape[1] > 1:
        score[1:] = np.mean(np.square(np.diff(array, axis=1)), axis=(0, 2))
    return score


def uniform_position_coverage_score(
    token_count: int, eligible_rows: np.ndarray, count: int
) -> np.ndarray:
    """Encode an evenly spaced position coreset as a top-k priority vector."""

    rows = np.asarray(eligible_rows, dtype=np.int64)
    take = min(max(int(count), 0), int(rows.size))
    score = np.zeros(int(token_count), dtype=np.float64)
    if take == 0:
        return score
    indices = np.floor(
        (np.arange(take, dtype=np.float64) + 0.5) * rows.size / take
    ).astype(np.int64)
    selected = rows[indices]
    if np.unique(selected).size != take:
        raise RuntimeError("uniform position coverage produced duplicate rows")
    score[selected] = 1.0
    return score


__all__ = [
    "adjacent_value_change_score",
    "attention_head_peak_score",
    "attention_temporal_volatility_score",
    "diagonal_leverage_score",
    "uniform_position_coverage_score",
]
