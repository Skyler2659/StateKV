"""Training-free scoring, coreset, merge, and quantization primitives.

The functions in this module are deliberately model-agnostic.  Model-backed
pilots supply logits, VJPs, attention distributions, and value tensors; these
utilities keep the estimators small enough to test independently.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    weights = np.exp(values - float(np.max(values)))
    return weights / max(float(weights.sum()), 1.0e-300)


def vjp_action_scores(
    actions: np.ndarray,
    gradients: np.ndarray,
    state: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Score actions with a shared output-side VJP sketch.

    ``gradients`` has shape ``[hidden, probes]``.  With Fisher-root random
    cotangents the action-only estimate is unbiased for one half of the local
    Fisher quadratic.  When ``state`` is supplied, the returned increment is
    ``<state, action>_Q + 0.5 ||action||_Q^2`` in the sketched metric.
    """

    action = np.asarray(actions, dtype=np.float64)
    sketch = np.asarray(gradients, dtype=np.float64)
    if action.ndim != 2 or sketch.ndim != 2:
        raise ValueError("actions and gradients must both be matrices")
    if action.shape[1] != sketch.shape[0] or sketch.shape[1] == 0:
        raise ValueError("hidden dimensions must align and probes must be nonempty")
    projected = action @ sketch
    score = 0.5 * np.mean(np.square(projected), axis=1)
    if state is not None:
        prior = np.asarray(state, dtype=np.float64)
        if prior.shape != action.shape:
            raise ValueError("state must have the same shape as actions")
        score = score + np.mean((prior @ sketch) * projected, axis=1)
    return np.asarray(score, dtype=np.float64)


def margin_cotangents(logits: np.ndarray, count: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return unit cotangents for top-1 versus its closest competitors."""

    values = np.asarray(logits, dtype=np.float64).reshape(-1)
    take = min(max(int(count), 0), max(int(values.size) - 1, 0))
    if take == 0:
        return np.empty((0, values.size), dtype=np.float64), np.empty(0, dtype=np.int64)
    order = np.argsort(-values, kind="stable")
    top = int(order[0])
    competitors = order[1 : take + 1].astype(np.int64)
    output = np.zeros((take, values.size), dtype=np.float64)
    output[:, top] = 1.0 / np.sqrt(2.0)
    output[np.arange(take), competitors] = -1.0 / np.sqrt(2.0)
    return output, competitors


def entropy_cotangent(logits: np.ndarray) -> np.ndarray:
    """Gradient of categorical entropy with respect to logits."""

    probability = softmax(logits)
    log_probability = np.log(np.maximum(probability, 1.0e-300))
    entropy = -float(np.dot(probability, log_probability))
    return -probability * (log_probability + entropy)


def _repeat_values(values: np.ndarray, query_heads: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 3:
        raise ValueError("values must have shape [kv_heads, tokens, dimension]")
    kv_heads = int(array.shape[0])
    if query_heads % kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    return np.repeat(array, query_heads // kv_heads, axis=0)


def attention_output(attention: np.ndarray, values: np.ndarray) -> np.ndarray:
    weights = np.asarray(attention, dtype=np.float64)
    if weights.ndim != 2:
        raise ValueError("attention must have shape [query_heads, tokens]")
    repeated = _repeat_values(values, int(weights.shape[0]))
    if weights.shape[1] != repeated.shape[1]:
        raise ValueError("attention and value token dimensions differ")
    return np.einsum("ht,htd->hd", weights, repeated)


def deletion_output(
    attention: np.ndarray, values: np.ndarray, retained: Sequence[int]
) -> np.ndarray:
    """Attention output after hard deletion and softmax renormalization."""

    weights = np.asarray(attention, dtype=np.float64)
    repeated = _repeat_values(values, int(weights.shape[0]))
    rows = np.asarray(sorted(set(int(value) for value in retained)), dtype=np.int64)
    if rows.size == 0:
        raise ValueError("at least one token must be retained")
    kept = weights[:, rows]
    normalizer = np.maximum(kept.sum(axis=1, keepdims=True), 1.0e-300)
    return np.einsum("ht,htd->hd", kept / normalizer, repeated[:, rows, :])


def nearest_value_merge(
    attention: np.ndarray, values: np.ndarray, retained: Sequence[int]
) -> Dict[str, np.ndarray]:
    """Transfer deleted attention mass to the nearest retained value.

    This is an attention-space diagnostic for a mergeable cache.  It preserves
    total attention mass and chooses, independently per query head, the retained
    representative minimizing the standard triangle-inequality error bound.
    """

    assignments = nearest_value_assignments(values, retained)
    merged = merge_output_with_assignments(attention, values, assignments)
    return {**merged, "assignments": assignments}


def nearest_value_assignments(
    values: np.ndarray, retained: Sequence[int]
) -> np.ndarray:
    """Map every value token to its nearest retained representative per KV head."""

    array = np.asarray(values, dtype=np.float64)
    rows = np.asarray(sorted(set(int(value) for value in retained)), dtype=np.int64)
    if rows.size == 0:
        raise ValueError("at least one token must be retained")
    if rows.min() < 0 or rows.max() >= array.shape[1]:
        raise ValueError("retained index is outside the value tensor")
    assignments = np.empty((int(array.shape[0]), int(array.shape[1])), dtype=np.int64)
    for head in range(int(array.shape[0])):
        source = array[head]
        representatives = source[rows]
        squared = (
            np.square(source).sum(axis=1, keepdims=True)
            + np.square(representatives).sum(axis=1)[None, :]
            - 2.0 * (source @ representatives.T)
        )
        assignments[head] = rows[np.argmin(squared, axis=1)]
        assignments[head, rows] = rows
    return assignments


def merge_output_with_assignments(
    attention: np.ndarray, values: np.ndarray, assignments: np.ndarray
) -> Dict[str, np.ndarray]:
    """Evaluate a fixed KV-head assignment under a new query attention vector."""

    weights = np.asarray(attention, dtype=np.float64)
    array = np.asarray(values, dtype=np.float64)
    mapping = np.asarray(assignments, dtype=np.int64)
    if mapping.shape != array.shape[:2]:
        raise ValueError("assignments must have shape [kv_heads, tokens]")
    query_heads = int(weights.shape[0])
    if query_heads % array.shape[0]:
        raise ValueError("query heads must be divisible by KV heads")
    group = query_heads // int(array.shape[0])
    output = np.empty((query_heads, int(array.shape[2])), dtype=np.float64)
    bound = np.empty(query_heads, dtype=np.float64)
    for head in range(query_heads):
        kv_head = head // group
        represented = array[kv_head, mapping[kv_head], :]
        output[head] = np.einsum("t,td->d", weights[head], represented)
        distance = np.linalg.norm(array[kv_head] - represented, axis=1)
        bound[head] = float(np.dot(weights[head], distance))
    return {"output": output, "bound": bound}


def scenario_token_scores(
    attentions: np.ndarray,
    values: np.ndarray,
    reduction: str,
    contribution_weighted: bool = True,
) -> np.ndarray:
    """Aggregate token importance across a bank of past query scenarios."""

    scenarios = np.asarray(attentions, dtype=np.float64)
    if scenarios.ndim != 3:
        raise ValueError("attentions must have shape [scenarios, heads, tokens]")
    repeated = _repeat_values(values, int(scenarios.shape[1]))
    if repeated.shape[1] != scenarios.shape[2]:
        raise ValueError("attention and value token dimensions differ")
    if contribution_weighted:
        full = np.einsum("sht,htd->shd", scenarios, repeated)
        squared_distance = (
            np.square(repeated).sum(axis=-1)[None, :, :]
            + np.square(full).sum(axis=-1)[:, :, None]
            - 2.0 * np.einsum("htd,shd->sht", repeated, full)
        )
        distance = np.sqrt(np.maximum(squared_distance, 0.0))
        per_scenario = np.mean(scenarios * distance, axis=1)
    else:
        per_scenario = np.mean(scenarios, axis=1)
    if reduction == "mean":
        return np.mean(per_scenario, axis=0)
    if reduction == "max":
        return np.max(per_scenario, axis=0)
    if reduction == "mean_plus_std":
        return np.mean(per_scenario, axis=0) + np.std(per_scenario, axis=0)
    if reduction == "ema":
        ages = np.arange(per_scenario.shape[0] - 1, -1, -1, dtype=np.float64)
        weights = np.power(0.5, ages)
        weights /= weights.sum()
        return np.einsum("s,st->t", weights, per_scenario)
    if reduction == "q75":
        return np.quantile(per_scenario, 0.75, axis=0)
    raise ValueError("unknown scenario reduction=%s" % reduction)


def select_top_with_mandatory(
    scores: np.ndarray, budget: int, mandatory: Iterable[int]
) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    forced = sorted(set(int(index) for index in mandatory))
    if int(budget) < len(forced):
        raise ValueError("budget is smaller than the mandatory set")
    if any(index < 0 or index >= values.size for index in forced):
        raise ValueError("mandatory index is outside the score vector")
    eligible = np.asarray(
        [index for index in range(values.size) if index not in set(forced)],
        dtype=np.int64,
    )
    take = min(int(budget) - len(forced), int(eligible.size))
    ranked = eligible[np.argsort(-values[eligible], kind="stable")[:take]]
    return np.asarray(sorted(forced + ranked.tolist()), dtype=np.int64)


def symmetric_quantize(values: np.ndarray, bits: int) -> np.ndarray:
    """Per-vector symmetric quantization along the last dimension."""

    array = np.asarray(values, dtype=np.float64)
    if int(bits) < 2 or int(bits) > 16:
        raise ValueError("bits must be in [2, 16]")
    maximum_code = float(2 ** (int(bits) - 1) - 1)
    peak = np.max(np.abs(array), axis=-1, keepdims=True)
    scale = np.where(peak > 0.0, peak / maximum_code, 1.0)
    code = np.clip(np.rint(array / scale), -maximum_code, maximum_code)
    return code * scale


__all__ = [
    "attention_output",
    "deletion_output",
    "entropy_cotangent",
    "margin_cotangents",
    "merge_output_with_assignments",
    "nearest_value_merge",
    "nearest_value_assignments",
    "scenario_token_scores",
    "select_top_with_mandatory",
    "softmax",
    "symmetric_quantize",
    "vjp_action_scores",
]
