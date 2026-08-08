"""Low-overhead rolling runtimes for direct shared cache-set policies.

Each policy accumulates one score state and emits one shared top-k set. Runtime
selection never evaluates a bank of cache algorithms.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Mapping, Sequence

import numpy as np


def normalized_eligible_score(
    score: np.ndarray, eligible_rows: np.ndarray
) -> np.ndarray:
    """Normalize a nonnegative score only over rows eligible for eviction."""

    values = np.maximum(np.asarray(score), 0.0)
    rows = np.asarray(eligible_rows, dtype=np.int64)
    output = np.zeros_like(values)
    denominator = float(values[rows].sum())
    if denominator > 0.0:
        output[rows] = values[rows] / denominator
    return output


def contribution_token_score(
    attention: np.ndarray,
    values: np.ndarray,
    *,
    dtype: np.dtype = np.dtype(np.float32),
) -> np.ndarray:
    r"""Compute the per-token value-aware attention score for one query.

    For query head :math:`h`, attention :math:`\alpha_{h,t}`, value
    :math:`v_t`, and attention output :math:`o_h`, the score is

    .. math::

       s_t = \operatorname{mean}_h
             \alpha_{h,t}\lVert v_t-o_h\rVert_2.

    Grouped-query heads are reshaped around their KV head, avoiding a repeated
    ``[query_heads, tokens, dimension]`` value tensor.
    """

    target_dtype = np.dtype(dtype)
    weights = np.asarray(attention, dtype=target_dtype)
    array = np.asarray(values, dtype=target_dtype)
    if weights.ndim != 2:
        raise ValueError("attention must have shape [query_heads, tokens]")
    if array.ndim != 3:
        raise ValueError("values must have shape [kv_heads, tokens, dimension]")
    query_heads, tokens = weights.shape
    kv_heads = int(array.shape[0])
    if int(array.shape[1]) != tokens:
        raise ValueError("attention and value token dimensions differ")
    if query_heads % kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    groups = query_heads // kv_heads
    grouped = weights.reshape(kv_heads, groups, tokens)
    output = np.einsum("kgt,ktd->kgd", grouped, array, optimize=True)
    squared_distance = (
        np.square(array).sum(axis=-1)[:, None, :]
        + np.square(output).sum(axis=-1)[:, :, None]
        - 2.0 * np.einsum("ktd,kgd->kgt", array, output, optimize=True)
    )
    distance = np.sqrt(np.maximum(squared_distance, 0.0))
    return np.mean(grouped * distance, axis=(0, 1), dtype=target_dtype)


def stable_topk_rows(
    score: np.ndarray, eligible_rows: np.ndarray, count: int
) -> np.ndarray:
    """Return deterministic top-k eligible rows using stable tie handling."""

    rows = np.asarray(eligible_rows, dtype=np.int64)
    take = min(max(int(count), 0), int(rows.size))
    if take == 0:
        return np.empty(0, dtype=np.int64)
    return rows[np.argsort(-np.asarray(score)[rows], kind="stable")[:take]]


def protected_rescue_score(
    attention_score: np.ndarray,
    contribution_score: np.ndarray,
    eligible_rows: np.ndarray,
    core_budget: int,
    rescue_slots: int,
) -> np.ndarray:
    """Encode a protected attention core plus contribution rescue slots.

    The returned priority vector retains the top ``core_budget-rescue_slots``
    attention rows unconditionally.  Contribution may fill only the remaining
    rescue slots, which bounds the number of deviations from the attention
    action without evaluating multiple model trajectories.
    """

    attention = np.asarray(attention_score, dtype=np.float64)
    contribution = np.asarray(contribution_score, dtype=np.float64)
    rows = np.asarray(eligible_rows, dtype=np.int64)
    if attention.shape != contribution.shape:
        raise ValueError("attention and contribution scores must align")
    take = min(max(int(core_budget), 0), int(rows.size))
    rescue = min(max(int(rescue_slots), 0), take)
    protected_count = take - rescue
    protected = stable_topk_rows(attention, rows, protected_count)
    protected_set = set(protected.tolist())
    remaining = np.asarray(
        [row for row in rows.tolist() if row not in protected_set],
        dtype=np.int64,
    )
    rescued = stable_topk_rows(contribution, remaining, rescue)
    priority = np.zeros_like(attention, dtype=np.float64)
    priority[protected] = 2.0
    priority[rescued] = 1.0
    if int((priority[rows] > 0.0).sum()) != take:
        raise RuntimeError("protected-rescue policy did not fill the core budget")
    return priority


@dataclass
class RollingDirectPolicy:
    """Windowed value-aware score accumulator for one direct policy."""

    layers: Sequence[int]
    window: int = 4
    contribution_weight: float = 0.25
    dtype: np.dtype = np.dtype(np.float32)
    _contribution: Dict[int, Deque[np.ndarray]] = field(init=False)
    _latest_attention: Dict[int, np.ndarray] = field(init=False)

    def __post_init__(self) -> None:
        if int(self.window) <= 0:
            raise ValueError("window must be positive")
        if not 0.0 <= float(self.contribution_weight) <= 1.0:
            raise ValueError("contribution_weight must be in [0, 1]")
        self.layers = tuple(int(layer) for layer in self.layers)
        if not self.layers:
            raise ValueError("at least one diagnostic layer is required")
        self.dtype = np.dtype(self.dtype)
        self._contribution = {
            layer: deque(maxlen=int(self.window)) for layer in self.layers
        }
        self._latest_attention = {}

    def _pad_layer(self, layer: int, tokens: int) -> None:
        current = self._latest_attention.get(layer)
        if current is not None:
            if current.size > tokens:
                raise ValueError("token count cannot shrink")
            if current.size < tokens:
                self._latest_attention[layer] = np.pad(
                    current, (0, tokens - current.size)
                )
        bank = self._contribution[layer]
        for index, score in enumerate(bank):
            if score.size > tokens:
                raise ValueError("token count cannot shrink")
            if score.size < tokens:
                bank[index] = np.pad(score, (0, tokens - score.size))

    def update_layer(
        self, layer: int, attention: np.ndarray, values: np.ndarray
    ) -> None:
        """Consume one layer's attention and values for the current query."""

        layer = int(layer)
        if layer not in self._contribution:
            raise ValueError("layer is not configured")
        weights = np.asarray(attention, dtype=self.dtype)
        tokens = int(weights.shape[1])
        self._pad_layer(layer, tokens)
        self._latest_attention[layer] = np.mean(
            weights, axis=0, dtype=self.dtype
        )
        self._contribution[layer].append(
            contribution_token_score(weights, values, dtype=self.dtype)
        )

    def score(self, eligible_rows: np.ndarray) -> np.ndarray:
        """Return the normalized, layer-averaged direct-policy score."""

        layer_scores = []
        rows = np.asarray(eligible_rows, dtype=np.int64)
        for layer in self.layers:
            if layer not in self._latest_attention:
                raise RuntimeError("every diagnostic layer must be updated")
            bank = self._contribution[layer]
            if not bank:
                raise RuntimeError("contribution window is empty")
            attention = normalized_eligible_score(
                self._latest_attention[layer], rows
            )
            contribution = normalized_eligible_score(
                np.mean(np.stack(tuple(bank), axis=0), axis=0), rows
            )
            weight = float(self.contribution_weight)
            layer_scores.append(
                (1.0 - weight) * attention + weight * contribution
            )
        return np.mean(np.stack(layer_scores, axis=0), axis=0)

    def select(self, eligible_rows: np.ndarray, count: int) -> np.ndarray:
        """Emit one shared top-k set for every model layer."""

        return stable_topk_rows(self.score(eligible_rows), eligible_rows, count)

    @property
    def working_set_bytes(self) -> int:
        """Exact bytes held by the rolling score state (excluding the KV cache)."""

        attention = sum(value.nbytes for value in self._latest_attention.values())
        contribution = sum(
            value.nbytes
            for bank in self._contribution.values()
            for value in bank
        )
        return int(attention + contribution)


@dataclass
class RollingTemporalVolatilityPolicy:
    """Keep tokens whose recent head-averaged attention is changing."""

    layers: Sequence[int]
    window: int = 4
    dtype: np.dtype = np.dtype(np.float32)
    _attention: Dict[int, Deque[np.ndarray]] = field(init=False)

    def __post_init__(self) -> None:
        if int(self.window) <= 1:
            raise ValueError("window must contain at least two queries")
        self.layers = tuple(int(layer) for layer in self.layers)
        if not self.layers:
            raise ValueError("at least one diagnostic layer is required")
        self.dtype = np.dtype(self.dtype)
        self._attention = {
            layer: deque(maxlen=int(self.window)) for layer in self.layers
        }

    def update_layer(self, layer: int, attention: np.ndarray) -> None:
        """Consume one query's attention for a diagnostic layer."""

        layer = int(layer)
        if layer not in self._attention:
            raise ValueError("layer is not configured")
        weights = np.asarray(attention, dtype=self.dtype)
        if weights.ndim != 2:
            raise ValueError("attention must have shape [heads, tokens]")
        tokens = int(weights.shape[1])
        bank = self._attention[layer]
        for index, score in enumerate(bank):
            if score.size > tokens:
                raise ValueError("token count cannot shrink")
            if score.size < tokens:
                bank[index] = np.pad(score, (0, tokens - score.size))
        bank.append(np.mean(weights, axis=0, dtype=self.dtype))

    def score(self, eligible_rows: np.ndarray) -> np.ndarray:
        """Return normalized volatility averaged across diagnostic layers."""

        rows = np.asarray(eligible_rows, dtype=np.int64)
        layer_scores = []
        for layer in self.layers:
            bank = self._attention[layer]
            if len(bank) != int(self.window):
                raise RuntimeError("every layer needs a complete attention window")
            volatility = np.std(np.stack(tuple(bank), axis=0), axis=0)
            layer_scores.append(normalized_eligible_score(volatility, rows))
        return np.mean(np.stack(layer_scores, axis=0), axis=0)

    def select(self, eligible_rows: np.ndarray, count: int) -> np.ndarray:
        """Emit the single shared top-k volatility set."""

        return stable_topk_rows(self.score(eligible_rows), eligible_rows, count)

    @property
    def working_set_bytes(self) -> int:
        return int(
            sum(
                value.nbytes
                for bank in self._attention.values()
                for value in bank
            )
        )


def batch_blend_score(
    attention_bank: Mapping[int, np.ndarray],
    values: Mapping[int, np.ndarray],
    eligible_rows: np.ndarray,
    *,
    contribution_weight: float = 0.25,
    dtype: np.dtype = np.dtype(np.float32),
) -> np.ndarray:
    """Reference batch computation for checking the rolling accumulator."""

    layer_scores = []
    for layer, bank_value in attention_bank.items():
        bank = np.asarray(bank_value, dtype=dtype)
        contribution = np.mean(
            np.stack(
                [
                    contribution_token_score(query, values[layer], dtype=dtype)
                    for query in bank
                ],
                axis=0,
            ),
            axis=0,
        )
        attention = normalized_eligible_score(
            np.mean(bank[-1], axis=0, dtype=np.dtype(dtype)), eligible_rows
        )
        contribution = normalized_eligible_score(contribution, eligible_rows)
        layer_scores.append(
            (1.0 - float(contribution_weight)) * attention
            + float(contribution_weight) * contribution
        )
    return np.mean(np.stack(layer_scores, axis=0), axis=0)


def batch_temporal_volatility_score(
    attention_bank: Mapping[int, np.ndarray],
    eligible_rows: np.ndarray,
    *,
    dtype: np.dtype = np.dtype(np.float32),
) -> np.ndarray:
    """Reference batch score for the rolling temporal-volatility policy."""

    rows = np.asarray(eligible_rows, dtype=np.int64)
    layer_scores = []
    for bank_value in attention_bank.values():
        bank = np.asarray(bank_value, dtype=dtype)
        if bank.ndim != 3:
            raise ValueError("attention bank must have shape [queries, heads, tokens]")
        volatility = np.std(np.mean(bank, axis=1, dtype=dtype), axis=0)
        layer_scores.append(normalized_eligible_score(volatility, rows))
    return np.mean(np.stack(layer_scores, axis=0), axis=0)


__all__ = [
    "RollingDirectPolicy",
    "RollingTemporalVolatilityPolicy",
    "batch_blend_score",
    "batch_temporal_volatility_score",
    "contribution_token_score",
    "normalized_eligible_score",
    "protected_rescue_score",
    "stable_topk_rows",
]
