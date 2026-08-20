"""MLX/MLX-LM 4-bit benchmark runner."""
from __future__ import annotations

import math
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.benchmarks.factory import load_benchmark
from src.artifacts import (
    ScoreArtifact,
    ScoreUnit,
    SelectionArtifact,
    SelectionUnit,
    SnapshotRef,
    save_artifact,
)
from src.config import ExperimentConfig
from src.eviction.registry import (
    get_method_spec,
    method_requires_attention,
    method_supports_backend,
    unsupported_reason,
    canonicalize_method as registry_canonicalize_method,
)
from src.eviction.score_normalization import list_stats
from src.evaluation.official_metrics import evaluate_official
from src.model_adapters import apply_prompt_format, build_model_adapter
from src.runners.base import BaseRunner, text_hash
from src.utils.io import save_results


SUPPORTED_MLX_METHODS = {
    "full",
    "basic",
    "basic_generate",
    "recency",
    "sink_recent",
    "sink_recency",
    "streamingllm",
    "random",
    "sink_recent_random",
    "attention",
    "accumulated_attention",
    "windowed_attention",
    "latest_attention_shared",
    "temporal_volatility_shared",
    "token_rarity_shared",
    "query_overlap_shared",
    "position_coverage_shared",
    "attention_decay",
    "h2o",
    "tova",
    "vatp",
    "snap",
    "snapkv",
    "snapkv_style",
    "approximate_snapkv",
    "pyramidkv",
    "pyramidkv_style",
    "layer_budget_attention",
    "l1",
    "l1_leverage",
    "l1_prefill_only",
    "l1_decode_only",
    "l2",
    "l2_leverage",
    "key_l2_leverage",
    "value_l2_leverage",
    "kv_l2_leverage",
    "l2_prefill_only",
    "l2_key_prefill_only",
    "l2_decode_only",
    "conditional_v_leverage",
    "conditional_k_leverage",
    "attention_residual_v_leverage",
    "window_residual_v_leverage",
    "attention_weighted_v_leverage",
    "window_weighted_v_leverage",
    "joint_kv_leverage",
    "ridge_v_allocation",
    "ridge_v_fixed",
    "ridge_v_shared",
    "diversity_v_leverage",
    "compactor",
    "compactor_style",
    "compactor_l2_attention",
    "adakv",
    "knorm",
    "keydiff",
    "curdkv",
    "vnorml1",
    "vnorml2",
    "key_l2_norm",
    "value_l2_norm",
    "key_l1_norm",
    "value_l1_norm",
    "key_norm",
    "value_norm",
    "sink_recent_l1",
    "sink_recent_l2",
    "attention+l1",
    "attention_l1",
    "attn_l1",
    "hybrid",
    "attention+l2",
    "attention_l2",
    "attn_l2",
    "attention_l1_compactor",
    "attention_l2_compactor",
    "attention_norm",
    "attention_recency",
    "attention_sink_recency",
    "budget_split_hybrid",
    "compactor",
    "sink_recent_attention_l1",
    "oracle_evidence",
    "oracle_answer_region",
}

ATTENTION_SCORE_METHODS = {
    "attention",
    "windowed_attention",
    "latest_attention_shared",
    "temporal_volatility_shared",
    "attention_decay",
    "h2o",
    "tova",
    "vatp",
}
SNAP_METHODS = {"snap", "snapkv"}
INNOVATION_PREFILL_METHODS = {
    "conditional_v_leverage",
    "conditional_k_leverage",
    "attention_residual_v_leverage",
    "window_residual_v_leverage",
    "attention_weighted_v_leverage",
    "window_weighted_v_leverage",
    "joint_kv_leverage",
    "ridge_v_allocation",
    "ridge_v_fixed",
    "ridge_v_shared",
    "diversity_v_leverage",
}
PREFILL_COMPRESS_METHODS = {
    "snapkv", "pyramidkv", "compactor", "adakv",
} | INNOVATION_PREFILL_METHODS
VARIABLE_HEAD_PREFILL_METHODS = {
    "compactor", "adakv", "ridge_v_allocation", "ridge_v_fixed",
}
COMPACTORLIKE_HYBRID_METHODS = {"attention_l1_compactor", "attention_l2_compactor"}
HYBRID_METHODS = {
    "attention+l1",
    "attention_l1",
    "attn_l1",
    "hybrid",
    "attention+l2",
    "attention_l2",
    "attn_l2",
    "attention_l1_compactor",
    "attention_l2_compactor",
    "attention_l1",
    "attention_l2",
    "attention_norm",
    "attention_recency",
    "attention_sink_recency",
    "budget_split_hybrid",
    "attention_residual_v_leverage",
    "attention_weighted_v_leverage",
}
MANUAL_COMPACT_METHODS = {
    "attention",
    "windowed_attention",
    "latest_attention_shared",
    "temporal_volatility_shared",
    "token_rarity_shared",
    "query_overlap_shared",
    "position_coverage_shared",
    "attention_decay",
    "h2o",
    "tova",
    "vatp",
    "snapkv",
    "pyramidkv",
    "l1_leverage",
    "l1_prefill_only",
    "l1_decode_only",
    "l2_leverage",
    "key_l2_leverage",
    "value_l2_leverage",
    "kv_l2_leverage",
    "l2_prefill_only",
    "l2_key_prefill_only",
    "l2_decode_only",
    "conditional_v_leverage",
    "conditional_k_leverage",
    "attention_residual_v_leverage",
    "window_residual_v_leverage",
    "attention_weighted_v_leverage",
    "window_weighted_v_leverage",
    "joint_kv_leverage",
    "ridge_v_allocation",
    "ridge_v_fixed",
    "ridge_v_shared",
    "diversity_v_leverage",
    "compactor",
    "adakv",
    "knorm",
    "keydiff",
    "curdkv",
    "vnorml1",
    "vnorml2",
    "key_l2_norm",
    "value_l2_norm",
    "key_l1_norm",
    "value_l1_norm",
    "random",
    "sink_recent_random",
    "sink_recent_l1",
    "sink_recent_l2",
    "attention_l1",
    "attention_l2",
    "attention_l1_compactor",
    "attention_l2_compactor",
    "attention_norm",
    "attention_recency",
    "attention_sink_recency",
    "budget_split_hybrid",
    "oracle_evidence",
    "oracle_answer_region",
}
METHODS_NEED_ATTENTION = ATTENTION_SCORE_METHODS | SNAP_METHODS | HYBRID_METHODS
METHODS_NEED_ATTENTION = METHODS_NEED_ATTENTION | PREFILL_COMPRESS_METHODS
SHARED_DIRECT_ATTENTION_METHODS = {
    "latest_attention_shared",
    "temporal_volatility_shared",
}
SHARED_DIRECT_STATIC_METHODS = {
    "token_rarity_shared",
    "query_overlap_shared",
    "position_coverage_shared",
}
SHARED_DIRECT_METHODS = (
    SHARED_DIRECT_ATTENTION_METHODS | SHARED_DIRECT_STATIC_METHODS
)


class ScoreUnavailableError(RuntimeError):
    """Raised instead of silently replacing a missing scientific signal."""


def snapkv_pool_scores_numpy(
    scores: Any,
    kernel: int,
    method: str = "max",
) -> np.ndarray:
    """Repository SnapKV pooling shared by benchmark and discovery paths."""
    values = np.asarray(scores, dtype=np.float32).reshape(-1)
    if not np.isfinite(values).all():
        raise FloatingPointError("SnapKV pooling input contains NaN/Inf")
    kernel = max(1, int(kernel))
    if kernel <= 1 or values.size <= 1:
        return values.copy()
    pad = max(0, kernel // 2)
    padded = np.pad(values, (pad, pad), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(
        padded, kernel
    )[: values.shape[0]]
    normalized_method = str(method or "max").lower()
    if normalized_method in {"avg", "mean"}:
        pooled = windows.mean(axis=-1)
    elif normalized_method == "max":
        pooled = windows.max(axis=-1)
    else:
        raise ValueError(
            "unsupported SnapKV pooling method=%r" % method
        )
    return pooled.astype(np.float32, copy=False)


def canonical_method(method: str) -> str:
    try:
        return registry_canonicalize_method(method)[0]
    except Exception:
        method = method.lower().replace("-", "_")
        if method == "snap":
            return "snapkv"
        return method


def normalize_text(text: Optional[str]) -> str:
    return " ".join((text or "").strip().lower().split())


def answer_f1(prediction: str, ground_truth: str) -> float:
    pred = normalize_text(prediction).split()
    gold = normalize_text(ground_truth).split()
    if not pred or not gold:
        return 0.0
    common = Counter(pred) & Counter(gold)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(gold)
    return 2 * precision * recall / (precision + recall)


def tensor_to_list(value: Any) -> List[int]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        return [int(x) for x in value.detach().cpu().flatten().tolist()]
    if hasattr(value, "tolist"):
        raw = value.tolist()
        if raw and isinstance(raw[0], list):
            raw = raw[0]
        return [int(x) for x in raw]
    return [int(x) for x in value]


def token_type(token_text: str) -> str:
    stripped = token_text.strip()
    if not stripped:
        return "whitespace"
    if stripped.isdigit():
        return "number"
    if any(ch.isdigit() for ch in stripped):
        return "alphanumeric"
    if stripped.isalpha():
        return "word"
    if all(not ch.isalnum() for ch in stripped):
        return "punctuation"
    return "mixed"


def _minmax_mx(values):
    import mlx.core as mx

    if values is None:
        return None
    vals = values.astype(mx.float32)
    if vals.shape[0] == 0:
        return vals
    lo = mx.min(vals)
    hi = mx.max(vals)
    return mx.where((hi - lo) > 1e-8, (vals - lo) / (hi - lo), mx.zeros_like(vals))


def _normalize_mx(values, mode: str):
    import mlx.core as mx

    vals = values.astype(mx.float32)
    mode = str(mode or "none").lower()
    if vals.shape[0] == 0 or mode == "none":
        return vals
    if mode == "minmax":
        return _minmax_mx(vals)
    if mode == "zscore":
        mean = mx.mean(vals)
        std = mx.maximum(mx.std(vals), 1e-8)
        return (vals - mean) / std
    if mode == "softmax":
        return mx.softmax(vals, axis=0)
    if mode == "rank":
        order = [int(x) for x in mx.argsort(vals).tolist()]
        denom = max(1, int(vals.shape[0]) - 1)
        ranks = np.zeros(int(vals.shape[0]), dtype=np.float32)
        for rank, idx in enumerate(order):
            ranks[idx] = float(rank) / float(denom)
        return mx.array(ranks)
    return vals


def _merge_score_vectors(attn, geom, lambda_attn: float, normalization: str = "rank"):
    import mlx.core as mx

    if attn is None and geom is None:
        return None
    if attn is None:
        return _normalize_mx(geom, normalization)
    if geom is None:
        return _normalize_mx(attn, normalization)
    n = min(int(attn.shape[0]), int(geom.shape[0]))
    if n <= 0:
        return None
    a = _normalize_mx(attn[:n], normalization)
    g = _normalize_mx(geom[:n], normalization)
    return float(lambda_attn) * a + (1.0 - float(lambda_attn)) * g


def _zscore_mx(values):
    import mlx.core as mx

    vals = values.astype(mx.float32)
    mean = mx.mean(vals)
    std = mx.maximum(mx.std(vals), 1e-8)
    return (vals - mean) / std


def _record_compactor_prefill_tensors(attn_module: Any, q_pre: Any, k_pre: Any, q_post: Any, k_post: Any) -> None:
    """Store full-prompt Q/K chunks needed by faithful Compactor scoring."""
    state = getattr(attn_module, "_l1kv_attention_state", None)
    layer_idx = getattr(attn_module, "_l1kv_layer_idx", None)
    if state is None or layer_idx is None:
        return
    if not state.get("enabled", False) or state.get("current_method") != "compactor":
        return
    if state.get("phase") != "prefill":
        return
    try:
        lid = int(layer_idx)
        state.setdefault("prefill_q_post", {}).setdefault(lid, []).append(q_post.astype(q_post.dtype))
        state.setdefault("prefill_k_post", {}).setdefault(lid, []).append(k_post.astype(k_post.dtype))
        state.setdefault("prefill_k_pre", {}).setdefault(lid, []).append(k_pre.astype(k_pre.dtype))
    except Exception:
        state.setdefault("hook_errors", 0)
        state["hook_errors"] += 1


def _record_attention_from_hook(
    attn_module: Any,
    queries: Any,
    keys: Any,
    query_len: Optional[int] = None,
) -> None:
    """Record explicitly defined causal attention signals.

    For accumulated attention, every query in the current forward call is
    processed in bounded chunks. Observation-window methods retain only their
    configured final query rows. Scores are stored both per KV head and pooled
    across KV heads; no substitute signal is produced on failure.
    """
    import mlx.core as mx

    state = getattr(attn_module, "_l1kv_attention_state", None)
    layer_idx = getattr(attn_module, "_l1kv_layer_idx", None)
    if state is None or layer_idx is None:
        return
    if not state.get("enabled", False):
        return
    if keys is None or queries is None or keys.shape[-2] == 0:
        return
    try:
        q_total = int(queries.shape[-2])
        k_total = int(keys.shape[-2])
        if q_total <= 0 or k_total <= 0:
            return
        max_observe = max(1, int(state.get("max_observe", 32)))
        record_all = bool(state.get("record_all_queries", False))
        q_first = 0 if record_all else max(0, q_total - max_observe)
        attention_chunk = max(1, int(state.get("attention_chunk_size", 128)))
        k = keys.astype(mx.float32)
        n_q_heads = int(queries.shape[1])
        n_kv_heads = int(k.shape[1])
        if n_kv_heads <= 0 or n_q_heads % n_kv_heads != 0:
            raise RuntimeError(
                f"cannot map {n_q_heads} query heads to {n_kv_heads} KV heads"
            )
        repeats = n_q_heads // n_kv_heads
        head_chunks = []
        record_query_head_window = bool(
            state.get("temporal_record_query_head_window", False)
        )
        query_head_chunks = []
        for q_start in range(q_first, q_total, attention_chunk):
            q_end = min(q_total, q_start + attention_chunk)
            q_slice = queries[:, :, q_start:q_end, :].astype(mx.float32)
            take = q_end - q_start
            q = q_slice.reshape(
                q_slice.shape[0], n_kv_heads, repeats, take, q_slice.shape[-1]
            )
            logits = mx.sum(
                q[:, :, :, :, None, :] * k[:, :, None, None, :, :], axis=-1
            ) * float(getattr(attn_module, "scale", 1.0))
            if query_len is not None:
                q_abs = mx.arange(q_start, q_end)
                allowed = k_total - int(query_len) + q_abs + 1
                key_pos = mx.arange(k_total)
                causal = key_pos.reshape(1, -1) < allowed.reshape(-1, 1)
                logits = mx.where(
                    causal.reshape(1, 1, 1, take, k_total), logits, -mx.inf
                )
            attn = mx.softmax(logits, axis=-1, precise=True)
            if record_query_head_window:
                query_head_chunks.append(
                    mx.mean(attn, axis=0)
                    .reshape(n_q_heads, take, k_total)
                    .astype(mx.float32)
                )
            # Average batch and the query-head group mapped to each KV head.
            head_chunks.append(mx.mean(attn, axis=(0, 2)).astype(mx.float32))
        if not head_chunks:
            return
        all_head_rows = mx.concatenate(head_chunks, axis=1)
        all_query_head_rows = (
            mx.concatenate(query_head_chunks, axis=1)
            if query_head_chunks
            else None
        )
        all_pooled_rows = mx.mean(all_head_rows, axis=0).astype(mx.float32)
        q_recorded = int(all_pooled_rows.shape[0])
        recent_take = min(max_observe, q_recorded)
        head_rows = all_head_rows[:, -recent_take:, :]
        pooled_rows = all_pooled_rows[-recent_take:, :]
        pooled = pooled_rows[-1]
        pooled_heads = head_rows[:, -1, :]
        window_sum = mx.sum(all_pooled_rows, axis=0)
        window_sum_heads = mx.sum(all_head_rows, axis=1)
        seq_len = int(pooled.shape[0])
        lid = int(layer_idx)
        state.setdefault("last", {})[lid] = pooled
        state.setdefault("last_heads", {})[lid] = pooled_heads

        accumulated = state.setdefault("accumulated", {})
        prev = accumulated.get(lid)
        if prev is None or int(prev.shape[0]) < seq_len:
            new_prev = mx.zeros((seq_len,), dtype=pooled.dtype)
            if prev is not None and int(prev.shape[0]) > 0:
                new_prev = mx.concatenate([prev, new_prev[int(prev.shape[0]) :]], axis=0)
            prev = new_prev
        prev = prev + mx.pad(window_sum, [(0, max(0, int(prev.shape[0]) - seq_len))])[: int(prev.shape[0])]
        accumulated[lid] = prev

        accumulated_heads = state.setdefault("accumulated_heads", {})
        prev_heads = accumulated_heads.get(lid)
        if prev_heads is None:
            prev_heads = mx.zeros((n_kv_heads, seq_len), dtype=pooled.dtype)
        elif int(prev_heads.shape[1]) < seq_len:
            prev_heads = mx.concatenate(
                [prev_heads, mx.zeros((n_kv_heads, seq_len - int(prev_heads.shape[1])), dtype=pooled.dtype)],
                axis=1,
            )
        prev_heads = prev_heads + window_sum_heads
        accumulated_heads[lid] = prev_heads

        decayed = state.setdefault("decayed", {})
        gamma = float(state.get("decay_gamma", 0.95))
        prev_decay = decayed.get(lid)
        if prev_decay is None or int(prev_decay.shape[0]) < seq_len:
            new_prev = mx.zeros((seq_len,), dtype=pooled.dtype)
            if prev_decay is not None and int(prev_decay.shape[0]) > 0:
                new_prev = mx.concatenate(
                    [prev_decay, new_prev[int(prev_decay.shape[0]) :]],
                    axis=0,
                )
            prev_decay = new_prev
        decay_weights = mx.power(
            mx.array(gamma, dtype=pooled.dtype),
            mx.arange(q_recorded - 1, -1, -1).astype(pooled.dtype),
        )
        weighted = mx.sum(all_pooled_rows * decay_weights.reshape(-1, 1), axis=0)
        prev_decay = prev_decay * (gamma ** q_recorded) + weighted
        decayed[lid] = prev_decay

        decayed_heads = state.setdefault("decayed_heads", {})
        prev_decay_heads = decayed_heads.get(lid)
        if prev_decay_heads is None:
            prev_decay_heads = mx.zeros((n_kv_heads, seq_len), dtype=pooled.dtype)
        elif int(prev_decay_heads.shape[1]) < seq_len:
            prev_decay_heads = mx.concatenate(
                [prev_decay_heads, mx.zeros((n_kv_heads, seq_len - int(prev_decay_heads.shape[1])), dtype=pooled.dtype)],
                axis=1,
            )
        weighted_heads = mx.sum(
            all_head_rows * decay_weights.reshape(1, -1, 1), axis=1
        )
        decayed_heads[lid] = prev_decay_heads * (gamma ** q_recorded) + weighted_heads

        observe = state.setdefault("observe", {}).setdefault(lid, [])
        observe.extend([pooled_rows[i] for i in range(int(pooled_rows.shape[0]))])
        if len(observe) > max_observe:
            del observe[:-max_observe]
        observe_heads = state.setdefault("observe_heads", {}).setdefault(lid, [])
        observe_heads.extend([head_rows[:, i, :] for i in range(int(head_rows.shape[1]))])
        if len(observe_heads) > max_observe:
            del observe_heads[:-max_observe]
        if all_query_head_rows is not None:
            query_head_rows = all_query_head_rows[:, -recent_take:, :]
            observe_query_heads = state.setdefault(
                "observe_query_heads", {}
            ).setdefault(lid, [])
            observe_query_heads.extend(
                [
                    query_head_rows[:, i, :]
                    for i in range(int(query_head_rows.shape[1]))
                ]
            )
            if len(observe_query_heads) > max_observe:
                del observe_query_heads[:-max_observe]
        counts = state.setdefault("query_counts", {}).setdefault(lid, {"prefill": 0, "decode": 0})
        phase = "prefill" if state.get("phase") == "prefill" else "decode"
        counts[phase] = int(counts.get(phase, 0)) + q_recorded
    except Exception as exc:
        state.setdefault("hook_errors", 0)
        state["hook_errors"] += 1
        state.setdefault("hook_error_events", []).append(
            {
                "layer": None if layer_idx is None else int(layer_idx),
                "phase": state.get("phase"),
                "method": state.get("current_method"),
                "reason": f"{type(exc).__name__}: {exc}",
            }
        )


def _cache_head_valid_attention_mask(cache: Any, n_query_heads: int, k_len: int):
    import mlx.core as mx

    valid = getattr(cache, "head_valid_mask", None)
    if valid is None:
        return None
    try:
        h_kv = int(valid.shape[0])
        old_len = int(valid.shape[1])
        if old_len < int(k_len):
            pad = mx.ones((h_kv, int(k_len) - old_len), dtype=mx.bool_)
            valid = mx.concatenate([valid, pad], axis=1)
            cache.head_valid_mask = valid
        elif old_len > int(k_len):
            valid = valid[:, : int(k_len)]
            cache.head_valid_mask = valid
        if h_kv <= 0:
            return None
        if int(n_query_heads) == h_kv:
            valid_q = valid
        elif int(n_query_heads) % h_kv == 0:
            valid_q = mx.repeat(valid, int(n_query_heads) // h_kv, axis=0)
        else:
            raise RuntimeError(
                f"cannot map {n_query_heads} query heads to {h_kv} KV-head masks"
            )
        return mx.where(
            valid_q[:, None, :],
            mx.zeros((int(n_query_heads), 1, int(k_len)), dtype=mx.float32),
            mx.full((int(n_query_heads), 1, int(k_len)), -mx.inf, dtype=mx.float32),
        )
    except Exception as exc:
        raise RuntimeError(f"failed to construct per-head cache mask: {exc}") from exc


class MLXL1Estimator:
    """Woodruff-style L1 leverage estimator implemented with MLX arrays.

    The exponential reweighting used by L1 sketches is intentionally heavy
    tailed. For KV eviction, a single near-zero random weight can dominate the
    QR factor and produce unstable token rankings, so the MLX runner uses a
    deterministic sketch with a modest weight floor and a rank-aware
    Moore--Penrose basis. Estimation failures are explicit, never norm fallbacks.
    """

    def __init__(
        self,
        sketch_dim: int = 1024,
        seed: int = 0,
        weight_floor: float = 1e-3,
        condition_limit: float = 1e6,
    ):
        self.sketch_dim = int(sketch_dim)
        self.seed = int(seed)
        self.weight_floor = float(weight_floor)
        self.condition_limit = float(condition_limit)
        self.r_inv = None
        self.last_dim = None
        self.fit_count = 0
        self.last_diagnostics: Dict[str, Any] = {
            "calculation": "approximate_l1_woodruff",
            "fit_count": 0,
            "fallback": False,
            "fallback_reason": None,
        }

    def scores(self, rows, force_refit: bool = False):
        import mlx.core as mx

        n, d = rows.shape
        if n <= 1:
            values = mx.sum(mx.abs(rows.astype(mx.float32)), axis=1)
            scores = mx.where(values > 0, mx.ones_like(values), mx.zeros_like(values))
            self.last_diagnostics = {
                "calculation": "degenerate_single_row_l1_leverage",
                "n_rows": int(n),
                "n_features": int(d),
                "effective_rank": int(n > 0),
                "condition_number": 1.0 if n > 0 else None,
                "fit_count": self.fit_count,
                "refit": False,
                "fallback": False,
                "fallback_reason": None,
                "sketch_dim": self.sketch_dim,
            }
            return scores
        rows_f = rows.astype(mx.float32)
        should_fit = self.r_inv is None or self.last_dim != d or bool(force_refit)
        if should_fit:
            rng = np.random.default_rng(
                self.seed + self.fit_count * 1_000_003 + int(n) * 9176 + int(d)
            )
            self.fit_count += 1
            if n < self.sketch_dim:
                u = mx.array(rng.uniform(1e-8, 1 - 1e-8, size=(n, 1)).astype(np.float32))
                weights = mx.maximum(-mx.log(1.0 - u), self.weight_floor)
                weighted = rows_f / weights
            else:
                buckets = mx.array(
                    rng.integers(0, self.sketch_dim, size=n, dtype=np.int32)
                )
                signs = mx.array(
                    rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=n)
                )
                u = mx.array(rng.uniform(1e-8, 1 - 1e-8, size=(n, 1)).astype(np.float32))
                weights = mx.maximum(-mx.log(1.0 - u), self.weight_floor)
                weighted_rows = rows_f / weights * signs.reshape(-1, 1)
                idx = mx.arange(self.sketch_dim).reshape(-1, 1)
                mask = (buckets.reshape(1, -1) == idx).astype(mx.float32)
                weighted = mask @ weighted_rows
            try:
                with mx.stream(mx.cpu):
                    _, r = mx.linalg.qr(weighted)
                    singular_values = mx.linalg.svd(r, compute_uv=False)
                    singular_np = np.asarray(singular_values.tolist(), dtype=np.float64)
                    s_max = float(singular_np.max()) if singular_np.size else 0.0
                    tolerance = max(weighted.shape) * np.finfo(np.float32).eps * s_max
                    kept = singular_np[singular_np > tolerance]
                    if not math.isfinite(s_max) or s_max <= 0 or kept.size == 0:
                        raise RuntimeError("non_finite_or_zero_l1_basis_singular_values")
                    condition = s_max / max(float(kept.min()), 1e-12)
                    if condition > self.condition_limit:
                        raise RuntimeError(
                            f"l1_basis_condition_exceeds_limit:{condition:.6g}>{self.condition_limit:.6g}"
                        )
                    self.r_inv = mx.linalg.pinv(r)
                self.last_dim = d
                self.last_diagnostics = {
                    "calculation": "approximate_l1_woodruff",
                    "n_rows": int(n),
                    "n_features": int(d),
                    "effective_rank": int(kept.size),
                    "condition_number": condition,
                    "rank_tolerance": float(tolerance),
                    "fit_count": self.fit_count,
                    "refit": True,
                    "fallback": False,
                    "fallback_reason": None,
                    "sketch_dim": self.sketch_dim,
                    "used_count_sketch": bool(n >= self.sketch_dim),
                }
            except Exception as exc:
                self.r_inv = None
                self.last_dim = d
                self.last_diagnostics = {
                    "calculation": "approximate_l1_woodruff",
                    "n_rows": int(n),
                    "n_features": int(d),
                    "effective_rank": None,
                    "condition_number": None,
                    "fit_count": self.fit_count,
                    "refit": True,
                    "fallback": False,
                    "fallback_reason": str(exc),
                    "sketch_dim": self.sketch_dim,
                    "failed": True,
                }
                raise RuntimeError(f"L1 leverage estimator failed: {exc}") from exc
        else:
            self.last_diagnostics = {
                **self.last_diagnostics,
                "n_rows": int(n),
                "n_features": int(d),
                "fit_count": self.fit_count,
                "refit": False,
                "fallback": False,
                "fallback_reason": None,
            }
        proj = rows_f @ self.r_inv
        scores = mx.sum(mx.abs(proj), axis=1)
        if not bool(mx.all(mx.isfinite(scores)).item()):
            self.last_diagnostics = {
                **self.last_diagnostics,
                "failed": True,
                "fallback_reason": "non_finite_l1_scores",
            }
            raise RuntimeError("L1 leverage estimator produced non-finite scores")
        return scores


class MLXL2Estimator:
    """Rank-aware L2 leverage with an explicitly reusable whitening fit."""

    def __init__(self):
        self.transform = None
        self.last_dim = None
        self.fit_count = 0
        self.last_diagnostics: Dict[str, Any] = {
            "calculation": "exact_gram_eigh",
            "fit_count": 0,
            "fallback": False,
            "fallback_reason": None,
        }

    def scores(self, rows, force_refit: bool = False):
        import mlx.core as mx
        from src.scoring.leverage.l2 import l2_whitener_numpy

        n, d = rows.shape
        rows_f = rows.astype(mx.float32)
        try:
            should_fit = self.transform is None or self.last_dim != int(d) or bool(force_refit)
            if should_fit:
                transform, diagnostics = l2_whitener_numpy(np.asarray(rows_f))
                self.transform = mx.array(transform)
                self.last_dim = int(d)
                self.fit_count += 1
                self.last_diagnostics = {
                    **diagnostics.to_dict(),
                    "fit_count": self.fit_count,
                    "refit": True,
                    "fallback": False,
                    "fallback_reason": None,
                }
            else:
                self.last_diagnostics = {
                    **self.last_diagnostics,
                    "n_rows": int(n),
                    "n_features": int(d),
                    "fit_count": self.fit_count,
                    "refit": False,
                    "calculation": "cached_whitener_l2_leverage",
                    "fallback": False,
                    "fallback_reason": None,
                }
            if int(self.transform.shape[1]) == 0:
                return mx.zeros((int(n),), dtype=mx.float32)
            projected = rows_f @ self.transform
            scores = mx.sum(projected * projected, axis=1)
            if not bool(mx.all(mx.isfinite(scores)).item()):
                raise RuntimeError("non_finite_l2_scores")
            return mx.maximum(scores, 0.0)
        except Exception as exc:
            self.last_diagnostics = {
                **self.last_diagnostics,
                "n_rows": int(n),
                "n_features": int(d),
                "fit_count": self.fit_count,
                "fallback": False,
                "fallback_reason": str(exc),
                "failed": True,
            }
            raise RuntimeError(f"L2 leverage estimator failed: {exc}") from exc


class MLXCacheEvictor:
    """Manual KVCache evictor for non-rotating MLX cache strategies."""

    def __init__(
        self,
        method: str,
        budget: int,
        cfg: ExperimentConfig,
        num_layers: int,
        attention_state: Optional[Dict[str, Any]] = None,
        oracle_positions: Optional[List[int]] = None,
        stream_token_ids: Optional[List[int]] = None,
    ):
        self.method = canonical_method(method)
        self.budget = int(budget)
        self.cfg = cfg
        self.num_layers = int(num_layers)
        self.attention_state = attention_state or {
            "last": {},
            "accumulated": {},
            "decayed": {},
            "observe": {},
            "observe_heads": {},
            "prefill_q_post": {},
            "prefill_k_post": {},
            "prefill_k_pre": {},
            "hook_errors": 0,
        }
        self.oracle_positions = sorted({int(x) for x in (oracle_positions or [])})
        self.stream_token_ids = [int(value) for value in (stream_token_ids or [])]
        self.stream_token_counts = Counter(self.stream_token_ids)
        self.query_signature_counts = Counter(self.stream_token_ids[-64:])
        self.position_maps: Dict[int, Any] = {}
        self.next_positions: Dict[int, int] = {}
        self.last_selected: Dict[int, List[int]] = {}
        self.last_scores: Dict[int, List[float]] = {}
        self.last_scores_by_head: Dict[int, Dict[str, List[float]]] = {}
        self.last_component_sources: Dict[int, Dict[str, List[str]]] = {}
        self.last_selected_by_head: Dict[int, Dict[str, List[int]]] = {}
        self.head_position_maps: Dict[int, Any] = {}
        self._component_sources_current: Dict[int, Dict[int, List[str]]] = {}
        self._last_attn_scores: Dict[int, Any] = {}
        self._last_geom_scores: Dict[int, Any] = {}
        self._current_scores_by_head: Dict[int, Any] = {}
        self._static_score_cache: Dict[int, Any] = {}
        self._static_score_cache_by_head: Dict[int, Any] = {}
        self.l1_estimators: Dict[Tuple[int, Optional[int]], MLXL1Estimator] = {}
        self.l2_estimators: Dict[Tuple[int, Optional[int]], MLXL2Estimator] = {}
        self.profile_times = {
            "score_time_s": 0.0,
            "topk_time_s": 0.0,
            "cache_rebuild_time_s": 0.0,
        }
        self.eviction_count = 0
        self.score_update_count = 0
        self.phase = "prefill"
        self.score_phase_counts = {"prefill": 0, "decode": 0}
        self.score_refit_count = 0
        self.score_refit_phase_counts = {"prefill": 0, "decode": 0}
        self.estimator_events: List[Dict[str, Any]] = []
        self.score_failures: List[Dict[str, Any]] = []
        self.eviction_step = 0
        self.prefill_decision: Optional[Dict[str, Any]] = None

    def set_phase(self, phase: str) -> None:
        self.phase = str(phase or "decode").lower()

    def append_stream_token(self, token_id: int) -> None:
        token = int(token_id)
        self.stream_token_ids.append(token)
        self.stream_token_counts[token] += 1

    def _record_score_refit(self) -> None:
        phase = self.phase if self.phase in self.score_refit_phase_counts else "decode"
        self.score_refit_count += 1
        self.score_refit_phase_counts[phase] += 1

    def _get_estimator(self, kind: str, layer_idx: int, head_idx: Optional[int] = None):
        key = (int(layer_idx), None if head_idx is None else int(head_idx))
        if kind == "l1":
            if key not in self.l1_estimators:
                head_seed = 0 if head_idx is None else (int(head_idx) + 1) * 65_537
                self.l1_estimators[key] = MLXL1Estimator(
                    self.cfg.eviction.sketch_dim,
                    seed=int(getattr(self.cfg, "seed", 0)) + int(layer_idx) * 1009 + head_seed,
                )
            return self.l1_estimators[key]
        if kind == "l2":
            if key not in self.l2_estimators:
                self.l2_estimators[key] = MLXL2Estimator()
            return self.l2_estimators[key]
        raise ValueError(f"unknown estimator kind={kind!r}")

    def _should_refit_estimator(self, estimator: Any) -> bool:
        if int(getattr(estimator, "fit_count", 0)) == 0:
            return True
        policy = str(getattr(self.cfg.eviction, "update_policy", "every_n_steps") or "every_n_steps").lower()
        if self.method in {
            "l1_prefill_only",
            "l2_prefill_only",
            "l2_key_prefill_only",
            "compactor",
            "attention_l1_compactor",
            "attention_l2_compactor",
        }:
            policy = "prefill_only"
        if policy in {"prefill_only", "never_after_prefill"}:
            return False
        if policy == "decode_only" and self.phase == "prefill":
            return False
        if policy != "every_n_steps":
            raise ValueError(f"unsupported update_policy={policy!r}")
        interval = int(getattr(self.cfg.eviction, "update_interval", 1))
        if interval <= 0:
            return False
        return self.eviction_step > 0 and self.eviction_step % interval == 0

    def _estimator_scores(
        self,
        estimator: Any,
        rows: Any,
        layer_idx: int,
        score_type: str,
        head_idx: Optional[int] = None,
        force_refit: Optional[bool] = None,
    ):
        before = int(getattr(estimator, "fit_count", 0))
        should_refit = self._should_refit_estimator(estimator) if force_refit is None else bool(force_refit)
        try:
            scores = estimator.scores(rows, force_refit=should_refit)
        except Exception as exc:
            diagnostics = dict(getattr(estimator, "last_diagnostics", {}) or {})
            event = {
                "layer": int(layer_idx),
                "head": None if head_idx is None else int(head_idx),
                "phase": self.phase,
                "eviction_step": self.eviction_step,
                "score_type": score_type,
                "failed": True,
                "reason": str(exc),
                **diagnostics,
            }
            self.score_failures.append(event)
            self.estimator_events.append(event)
            raise
        after = int(getattr(estimator, "fit_count", before))
        refit_delta = max(0, after - before)
        for _ in range(refit_delta):
            self._record_score_refit()
        event = {
            "layer": int(layer_idx),
            "head": None if head_idx is None else int(head_idx),
            "phase": self.phase,
            "eviction_step": self.eviction_step,
            "score_type": score_type,
            "failed": False,
            **dict(getattr(estimator, "last_diagnostics", {}) or {}),
        }
        self.estimator_events.append(event)
        return scores

    def sync_maps(self, cache: List[Any]) -> None:
        import mlx.core as mx

        for layer_idx, c in enumerate(cache):
            seq_len = int(c.offset)
            head_map = getattr(c, "head_position_map", None)
            if head_map is not None:
                h_count = int(head_map.shape[0])
                old_len = int(head_map.shape[1])
                appended_positions: List[int] = []
                if old_len < seq_len:
                    start = self.next_positions.get(layer_idx, int(getattr(c, "logical_offset", seq_len)) - (seq_len - old_len))
                    extra = mx.arange(start, start + (seq_len - old_len))
                    appended_positions = list(range(start, start + (seq_len - old_len)))
                    extra = mx.broadcast_to(extra.reshape(1, -1), (h_count, int(extra.shape[0])))
                    head_map = mx.concatenate([head_map, extra], axis=1)
                    c.head_position_map = head_map
                    self.next_positions[layer_idx] = start + int(extra.shape[1])
                    c.valid_token_head_slots = int(
                        getattr(c, "valid_token_head_slots", 0)
                    ) + h_count * (seq_len - old_len)
                elif old_len > seq_len:
                    head_map = head_map[:, :seq_len]
                    c.head_position_map = head_map
                valid = getattr(c, "head_valid_mask", None)
                if valid is None:
                    valid = mx.ones((h_count, seq_len), dtype=mx.bool_)
                    c.head_valid_mask = valid
                elif int(valid.shape[1]) < seq_len:
                    pad = mx.ones((h_count, seq_len - int(valid.shape[1])), dtype=mx.bool_)
                    valid = mx.concatenate([valid, pad], axis=1)
                    c.head_valid_mask = valid
                elif int(valid.shape[1]) > seq_len:
                    valid = valid[:, :seq_len]
                    c.head_valid_mask = valid
                self.head_position_maps[layer_idx] = head_map
                existing = self.last_selected_by_head.get(layer_idx)
                if existing is not None and old_len <= seq_len:
                    if appended_positions:
                        for h in range(h_count):
                            existing.setdefault(str(h), []).extend(appended_positions)
                        union = {
                            int(value)
                            for values in existing.values()
                            for value in values
                        }
                        self.last_selected_by_head[layer_idx] = existing
                        self.last_selected[layer_idx] = sorted(union)
                        self.position_maps[layer_idx] = mx.array(
                            sorted(union), dtype=mx.int32
                        )
                    continue
                by_head: Dict[str, List[int]] = {}
                union = set()
                head_rows = head_map.tolist()
                valid_rows = valid.tolist()
                for h, (positions, flags) in enumerate(zip(head_rows, valid_rows)):
                    vals = [
                        int(position)
                        for position, is_valid in zip(positions, flags)
                        if bool(is_valid) and int(position) >= 0
                    ]
                    by_head[str(h)] = vals
                    union.update(vals)
                self.last_selected_by_head[layer_idx] = by_head
                self.last_selected[layer_idx] = sorted(union)
                self.position_maps[layer_idx] = mx.array(sorted(union), dtype=mx.int32)
                continue
            current = self.position_maps.get(layer_idx)
            if current is None or len(current) > seq_len:
                self.position_maps[layer_idx] = mx.arange(seq_len)
                self.next_positions[layer_idx] = seq_len
            elif len(current) < seq_len:
                start = self.next_positions.get(layer_idx, len(current))
                extra = mx.arange(start, start + (seq_len - len(current)))
                self.position_maps[layer_idx] = mx.concatenate([current, extra])
                self.next_positions[layer_idx] = start + len(extra)

    def evict_for_space(self, cache: List[Any], num_coming: int = 1) -> None:
        budget = max(1, self.budget - int(num_coming))
        if cache and int(cache[0].offset) > budget:
            self.evict(cache, budget)

    def evict(self, cache: List[Any], budget: Optional[int] = None) -> None:
        import mlx.core as mx

        budget = int(budget or self.budget)
        self.sync_maps(cache)
        if self.method in SHARED_DIRECT_METHODS:
            self._evict_shared_direct(cache, budget)
            return
        decision_units: List[Dict[str, Any]] = []
        made_eviction_decision = False
        for layer_idx, c in enumerate(cache):
            seq_len = int(c.offset)
            layer_budget = self._layer_budget(layer_idx, len(cache), budget)
            if seq_len <= layer_budget:
                self.last_selected[layer_idx] = self._to_int_list(
                    self.position_maps[layer_idx]
                )
                universe = self._to_int_list(self.position_maps[layer_idx])
                for head_idx in range(int(c.keys.shape[1])):
                    decision_units.append(
                        {
                            "layer": layer_idx,
                            "head": head_idx,
                            "universe_positions": universe,
                            "score_positions": [],
                            "scores": [],
                            "selected_positions": universe,
                            "requested_budget": layer_budget,
                        }
                    )
                continue
            made_eviction_decision = True
            score_start = time.perf_counter()
            scores = self._compute_scores(c, layer_idx, seq_len)
            self.profile_times["score_time_s"] += time.perf_counter() - score_start
            if scores is not None:
                self.score_update_count += 1
                phase = self.phase if self.phase in self.score_phase_counts else "decode"
                self.score_phase_counts[phase] += 1
                self.last_scores[layer_idx] = self._to_float_list(scores)

            topk_start = time.perf_counter()
            keep = self._select_indices(scores, seq_len, layer_budget, layer_idx)
            self.profile_times["topk_time_s"] += time.perf_counter() - topk_start

            rebuild_start = time.perf_counter()
            old_position_map = self.position_maps[layer_idx]
            selected_positions = mx.take(old_position_map, keep, axis=0)
            universe = self._to_int_list(old_position_map)
            selected_original = self._to_int_list(selected_positions)
            by_head = self._current_scores_by_head.get(layer_idx)
            if by_head is not None and int(by_head.shape[-1]) >= seq_len:
                for head_idx in range(int(by_head.shape[0])):
                    decision_units.append(
                        {
                            "layer": layer_idx,
                            "head": head_idx,
                            "universe_positions": universe,
                            "score_positions": universe,
                            "scores": self._to_float_list(by_head[head_idx, :seq_len]),
                            "selected_positions": selected_original,
                            "requested_budget": layer_budget,
                        }
                    )
            else:
                # The physical shared-token selection applies to every KV
                # head. A genuinely shared score (for example recency or a
                # pooled hybrid) is emitted as a separate head=None score unit
                # rather than being mislabeled as head-wise.
                for head_idx in range(int(c.keys.shape[1])):
                    decision_units.append(
                        {
                            "layer": layer_idx,
                            "head": head_idx,
                            "universe_positions": universe,
                            "score_positions": [],
                            "scores": [],
                            "selected_positions": selected_original,
                            "requested_budget": layer_budget,
                            "selection_only": True,
                        }
                    )
                if scores is not None:
                    decision_units.append(
                        {
                            "layer": layer_idx,
                            "head": None,
                            "universe_positions": universe,
                            "score_positions": universe,
                            "scores": self._to_float_list(scores),
                            "selected_positions": [],
                            "requested_budget": layer_budget,
                            "score_only": True,
                        }
                    )
            c.keys = mx.take(c.keys[:, :, :seq_len, :], keep, axis=2)
            c.values = mx.take(c.values[:, :, :seq_len, :], keep, axis=2)
            c.offset = int(keep.shape[0])
            self._prune_attention_state(layer_idx, keep, seq_len)
            current_sources = self._component_sources_current.get(layer_idx, {})
            if current_sources:
                self.last_component_sources[layer_idx] = {
                    str(int(orig)): current_sources.get(int(cur), ["unknown"])
                    for cur, orig in zip(keep.tolist(), selected_positions.tolist())
                }
            self.position_maps[layer_idx] = selected_positions
            self.last_selected[layer_idx] = self._to_int_list(
                self.position_maps[layer_idx]
            )
            self.profile_times["cache_rebuild_time_s"] += (
                time.perf_counter() - rebuild_start
            )
            self.eviction_count += 1
        if self.phase == "prefill" and decision_units:
            self.prefill_decision = {
                "phase": "pre_answer",
                "budget_scope": "total_kv",
                "budget_unit": "token_slots_per_kv_head",
                "requested_budget": budget,
                "units": decision_units,
            }
        # ``update_interval`` is defined in actual eviction decisions (one
        # opportunity per generated token), not in calls to ``evict``.  The
        # decode loop also calls ``evict`` after appending a token, when the
        # cache is already within budget.  Counting that no-op call makes the
        # decision steps 1, 3, 5, ... and can prevent even intervals such as
        # 16/32/64 from ever triggering an estimator refit.
        if made_eviction_decision:
            self.eviction_step += 1

    def prefill_compress(self, cache: List[Any], budget: Optional[int] = None) -> None:
        import mlx.core as mx

        budget = int(budget or self.budget)
        if self.method not in PREFILL_COMPRESS_METHODS:
            self.evict(cache, budget)
            return
        self.sync_maps(cache)
        decision_units: List[Dict[str, Any]] = []
        for layer_idx, c in enumerate(cache):
            seq_len = int(c.offset)
            if seq_len <= 0:
                continue
            score_start = time.perf_counter()
            if self.method == "compactor":
                keep_by_head, scores_by_head = self._compactor_headwise_keep(c, layer_idx, seq_len, budget)
                layer_budget = max((len(v) for v in keep_by_head), default=0)
            elif self.method == "adakv":
                keep_by_head, scores_by_head = self._adakv_headwise_keep(
                    layer_idx, seq_len, budget
                )
                layer_budget = max((len(v) for v in keep_by_head), default=0)
            elif self.method in INNOVATION_PREFILL_METHODS:
                layer_budget = min(int(seq_len), int(budget))
                if seq_len <= layer_budget:
                    keep_by_head = [
                        list(range(seq_len)) for _ in range(int(c.keys.shape[1]))
                    ]
                    scores_by_head = None
                else:
                    keep_by_head, scores_by_head = self._innovation_headwise_keep(
                        c, layer_idx, seq_len, budget
                    )
                    layer_budget = max((len(v) for v in keep_by_head), default=0)
            else:
                layer_budget = self._prefill_layer_budget(layer_idx, len(cache), budget, seq_len)
                if seq_len <= layer_budget:
                    keep_by_head = [list(range(seq_len)) for _ in range(int(c.keys.shape[1]))]
                    scores_by_head = None
                else:
                    keep_by_head, scores_by_head = self._snap_pyramid_headwise_keep(
                        layer_idx,
                        seq_len,
                        layer_budget,
                    )
            self.profile_times["score_time_s"] += time.perf_counter() - score_start
            if scores_by_head is not None:
                self.score_update_count += 1
                phase = self.phase if self.phase in self.score_phase_counts else "prefill"
                self.score_phase_counts[phase] += 1
                self._record_head_scores(layer_idx, scores_by_head)

            if seq_len <= layer_budget and all(len(v) == seq_len for v in keep_by_head):
                self.last_selected[layer_idx] = list(range(seq_len))
                self.last_selected_by_head[layer_idx] = {
                    str(h): list(range(seq_len)) for h in range(int(c.keys.shape[1]))
                }
                self.position_maps[layer_idx] = mx.arange(seq_len)
                self.next_positions[layer_idx] = int(getattr(c, "logical_offset", seq_len))
                continue

            topk_start = time.perf_counter()
            keep_by_head = [sorted({int(x) for x in values if 0 <= int(x) < seq_len}) for values in keep_by_head]
            self.profile_times["topk_time_s"] += time.perf_counter() - topk_start

            universe = self._to_int_list(self.position_maps[layer_idx])
            score_array = None
            if scores_by_head is not None:
                score_array = np.asarray(scores_by_head.tolist(), dtype=np.float64)
            for head_idx, keep_values in enumerate(keep_by_head):
                score_len = int(score_array.shape[1]) if score_array is not None else 0
                decision_units.append(
                    {
                        "layer": layer_idx,
                        "head": head_idx,
                        "universe_positions": universe,
                        "score_positions": universe[:score_len],
                        "scores": (
                            [float(x) for x in score_array[head_idx].tolist()]
                            if score_array is not None
                            else []
                        ),
                        "selected_positions": [universe[idx] for idx in keep_values],
                        "requested_budget": layer_budget,
                    }
                )

            rebuild_start = time.perf_counter()
            if (
                self.method in INNOVATION_PREFILL_METHODS
                and self.method not in {"ridge_v_allocation", "ridge_v_fixed"}
            ):
                self._apply_shared_prefill_keep(
                    c, layer_idx, keep_by_head[0], seq_len
                )
            else:
                self._apply_headwise_keep(c, layer_idx, keep_by_head, seq_len)
            self.profile_times["cache_rebuild_time_s"] += time.perf_counter() - rebuild_start
            self.eviction_count += 1
        if decision_units:
            if self.method in VARIABLE_HEAD_PREFILL_METHODS:
                for unit in decision_units:
                    unit["requested_budget"] = len(unit["selected_positions"])
                pair_budget = int(budget) * max(
                    1, int(cache[0].keys.shape[1]) if cache else 1
                )
                budget_unit = "token_head_pairs"
            else:
                pair_budget = budget
                budget_unit = "token_slots_per_kv_head"
            self.prefill_decision = {
                "phase": "pre_answer",
                "budget_scope": "prompt_prefill",
                "budget_unit": budget_unit,
                "requested_budget": pair_budget,
                "units": decision_units,
            }
        self.eviction_step += 1

    def _record_head_scores(self, layer_idx: int, scores_by_head: Any) -> None:
        import mlx.core as mx

        arr = np.asarray(scores_by_head.tolist(), dtype=np.float32)
        finite = np.isfinite(arr)
        fill = float(arr[finite].max() + 1.0) if finite.any() else 0.0
        arr = np.where(finite, arr, fill).astype(np.float32)
        scores = mx.array(arr)
        self.last_scores_by_head[layer_idx] = {
            str(head): [float(value) for value in arr[head].tolist()]
            for head in range(int(arr.shape[0]))
        }
        if len(scores.shape) == 2:
            agg = mx.mean(scores, axis=0)
        else:
            agg = scores
        self.last_scores[layer_idx] = self._to_float_list(agg)

    def _prefill_layer_budget(self, layer_idx: int, num_layers: int, base_budget: int, seq_len: int) -> int:
        if self.method != "pyramidkv" or num_layers <= 1:
            return int(base_budget)
        window = min(max(1, int(getattr(self.cfg.eviction, "window_size", 64))), int(seq_len))
        if seq_len <= base_budget:
            return int(seq_len)
        base = max(1, int(base_budget) - window)
        if seq_len < base * 2:
            return min(seq_len, base + window)
        beta = max(1, int(getattr(self.cfg.eviction, "pyramid_beta", 20)))
        min_num = base // beta
        max_num = base * 2 - min_num
        hist_len = max(0, seq_len - window)
        if max_num >= hist_len:
            max_num = hist_len
            min_num = base * 2 - max_num
        steps = (max_num - min_num) // max(1, int(num_layers) - 1)
        prefix_keep = max(1, max_num - int(layer_idx) * steps)
        return max(1, min(seq_len, prefix_keep + window))

    def _snap_pyramid_headwise_keep(self, layer_idx: int, seq_len: int, budget: int):
        import mlx.core as mx

        obs = min(max(1, int(getattr(self.cfg.eviction, "window_size", 64))), seq_len)
        hist_len = max(0, seq_len - obs)
        rows = self.attention_state.get("observe_heads", {}).get(layer_idx, [])
        usable = [row[:, :seq_len] for row in rows if int(row.shape[-1]) >= seq_len]
        if usable:
            stacked = mx.stack(usable[-obs:], axis=0)
            prefix_scores = mx.sum(stacked[:, :, :hist_len], axis=0).astype(mx.float32)
        else:
            raise ScoreUnavailableError(
                f"observation-window attention unavailable for layer={layer_idx}; "
                "refusing recency fallback for SnapKV/PyramidKV"
            )
        pooled = self._pool_scores_by_head(prefix_scores)
        h_count = int(pooled.shape[0]) if len(pooled.shape) == 2 else int(self._cache_num_heads(layer_idx))
        hist_budget = max(0, int(budget) - obs)
        keep_by_head: List[List[int]] = []
        for h in range(h_count):
            parts: List[int] = []
            if hist_budget > 0 and hist_len > 0:
                score = pooled[h, :hist_len]
                take = min(hist_budget, hist_len)
                if take >= hist_len:
                    idx = mx.arange(hist_len)
                else:
                    idx = mx.argpartition(-score, max(0, take - 1))[:take]
                parts.extend(int(x) for x in idx.tolist())
            parts.extend(range(max(0, seq_len - obs), seq_len))
            keep_by_head.append(sorted(set(parts)))
        recent_fill = mx.array(1.0, dtype=mx.float32)
        if hist_len > 0:
            recent_fill = mx.max(pooled[:, :hist_len]) + 1.0
            prefix_part = pooled[:, :hist_len]
        else:
            prefix_part = mx.zeros((h_count, 0), dtype=mx.float32)
        if obs > 0:
            recent_part = mx.ones((h_count, seq_len - hist_len), dtype=mx.float32) * recent_fill
            full_scores = mx.concatenate([prefix_part, recent_part], axis=1)
        else:
            full_scores = prefix_part
        return keep_by_head, full_scores

    def _adakv_headwise_keep(self, layer_idx: int, seq_len: int, budget: int):
        """Ada-SnapKV from Algorithms 1-2 of Feng et al."""
        import mlx.core as mx

        obs = min(
            max(1, int(getattr(self.cfg.eviction, "window_size", 32))),
            int(seq_len),
            max(1, int(budget)),
        )
        hist_len = max(0, int(seq_len) - obs)
        rows = self.attention_state.get("observe_heads", {}).get(layer_idx, [])
        usable = [row[:, :seq_len] for row in rows if int(row.shape[-1]) >= seq_len]
        if not usable:
            raise ScoreUnavailableError(
                f"observation-window attention unavailable for layer={layer_idx}; "
                "refusing fallback for Ada-SnapKV"
            )
        stacked = mx.stack(usable[-obs:], axis=0)
        prefix_scores = mx.sum(stacked[:, :, :hist_len], axis=0).astype(mx.float32)
        pooled = self._pool_scores_by_head(prefix_scores)
        h_count = int(pooled.shape[0])
        total_hist_budget = min(
            h_count * hist_len,
            max(0, int(budget) - obs) * h_count,
        )

        adaptive = np.zeros(h_count, dtype=np.float64)
        if total_hist_budget > 0 and hist_len > 0:
            flat = np.asarray(pooled[:, :hist_len].tolist(), dtype=np.float64).reshape(-1)
            if total_hist_budget >= flat.size:
                chosen = np.arange(flat.size)
            else:
                chosen = np.argpartition(-flat, total_hist_budget - 1)[:total_hist_budget]
            adaptive = np.bincount(chosen // hist_len, minlength=h_count).astype(np.float64)

        uniform = float(total_hist_budget) / max(1, h_count)
        alpha = max(
            0.0,
            min(1.0, float(getattr(self.cfg.eviction, "adakv_safeguard_alpha", 0.2))),
        )
        target = alpha * adaptive + (1.0 - alpha) * uniform
        allocation = np.minimum(np.floor(target).astype(np.int64), hist_len)
        residual = target - np.floor(target)
        while int(allocation.sum()) < total_hist_budget:
            eligible = np.flatnonzero(allocation < hist_len)
            if eligible.size == 0:
                break
            best = int(eligible[np.argmax(residual[eligible])])
            allocation[best] += 1
            residual[best] = -1.0
        while int(allocation.sum()) > total_hist_budget:
            eligible = np.flatnonzero(allocation > 0)
            if eligible.size == 0:
                break
            worst = int(eligible[np.argmin(residual[eligible])])
            allocation[worst] -= 1

        keep_by_head: List[List[int]] = []
        for head_idx in range(h_count):
            take = int(allocation[head_idx])
            selected: List[int] = []
            if take > 0 and hist_len > 0:
                score = pooled[head_idx, :hist_len]
                if take >= hist_len:
                    idx = mx.arange(hist_len)
                else:
                    idx = mx.argpartition(-score, max(0, take - 1))[:take]
                selected.extend(int(x) for x in idx.tolist())
            selected.extend(range(hist_len, seq_len))
            keep_by_head.append(sorted(set(selected)))

        if hist_len > 0:
            recent_fill = mx.max(pooled[:, :hist_len]) + 1.0
            prefix_part = pooled[:, :hist_len]
        else:
            recent_fill = mx.array(1.0, dtype=mx.float32)
            prefix_part = mx.zeros((h_count, 0), dtype=mx.float32)
        recent_part = mx.ones((h_count, obs), dtype=mx.float32) * recent_fill
        full_scores = mx.concatenate([prefix_part, recent_part], axis=1)
        return keep_by_head, full_scores

    @staticmethod
    def _top_score_indices_numpy(
        scores: np.ndarray,
        take: int,
        excluded: Optional[set] = None,
    ) -> List[int]:
        """Deterministic descending score order with optional exclusions."""
        values = np.asarray(scores, dtype=np.float64).reshape(-1)
        excluded = excluded or set()
        candidates = np.asarray(
            [idx for idx in range(values.size) if idx not in excluded],
            dtype=np.int64,
        )
        take = min(max(0, int(take)), int(candidates.size))
        if take <= 0:
            return []
        candidate_scores = np.nan_to_num(
            values[candidates], nan=-np.inf, neginf=-np.inf, posinf=np.inf
        )
        # Stable sorting makes tied synthetic tests and repeated runs identical.
        order = np.argsort(-candidate_scores, kind="stable")[:take]
        return [int(x) for x in candidates[order].tolist()]

    @staticmethod
    def _relative_ridge_lambda(rows: Any, coefficient: float) -> float:
        """Convert a dimensionless ridge coefficient to the row matrix scale."""
        import mlx.core as mx

        rows_f = rows.astype(mx.float32)
        dim = max(1, int(rows_f.shape[-1]))
        gram_trace = float(mx.sum(rows_f * rows_f).item())
        mean_gram_diagonal = gram_trace / float(dim)
        return max(1e-8, float(coefficient) * max(mean_gram_diagonal, 1e-8))

    def _ridge_leverage_rows(
        self,
        rows: Any,
        relative_lambda: float,
        *,
        absolute_lambda: Optional[float] = None,
    ):
        """Return ridge scores, whitened rows, absolute lambda and d_eff."""
        import mlx.core as mx

        rows_f = rows.astype(mx.float32)
        dim = int(rows_f.shape[-1])
        gram = mx.matmul(rows_f.transpose(1, 0), rows_f)
        eigenvalues, eigenvectors = mx.linalg.eigh(gram, stream=mx.cpu)
        eigenvalues = mx.maximum(eigenvalues.astype(mx.float32), 0.0)
        lam = (
            float(absolute_lambda)
            if absolute_lambda is not None
            else self._relative_ridge_lambda(rows_f, relative_lambda)
        )
        lam = max(float(lam), 1e-8)
        inv_sqrt = 1.0 / mx.sqrt(eigenvalues + lam)
        transform = eigenvectors.astype(mx.float32) * inv_sqrt.reshape(1, dim)
        whitened = mx.matmul(rows_f, transform)
        scores = mx.sum(whitened * whitened, axis=1).astype(mx.float32)
        if not bool(mx.all(mx.isfinite(scores)).item()):
            raise RuntimeError("experimental ridge leverage produced non-finite scores")
        return mx.maximum(scores, 0.0), whitened, lam, float(mx.sum(scores).item())

    def _conditional_residual_rows(
        self,
        target: Any,
        condition: Any,
        ridge_coefficient: float,
    ):
        """Ridge residual target - condition @ argmin_B ||target-condition B||."""
        import mlx.core as mx

        target_f = target.astype(mx.float32)
        condition_f = condition.astype(mx.float32)
        dim = int(condition_f.shape[-1])
        gram = mx.matmul(condition_f.transpose(1, 0), condition_f)
        lam = self._relative_ridge_lambda(condition_f, ridge_coefficient)
        rhs = mx.matmul(condition_f.transpose(1, 0), target_f)
        coefficients = mx.linalg.solve(
            gram + mx.eye(dim, dtype=mx.float32) * lam,
            rhs,
            stream=mx.cpu,
        )
        return target_f - mx.matmul(condition_f, coefficients), lam

    def _prefill_accumulated_attention_by_head(
        self,
        layer_idx: int,
        seq_len: int,
        head_count: int,
    ):
        import mlx.core as mx

        attention = self.attention_state.get("accumulated_heads", {}).get(layer_idx)
        if (
            attention is None
            or int(attention.shape[0]) != int(head_count)
            or int(attention.shape[-1]) < int(seq_len)
        ):
            raise ScoreUnavailableError(
                "per-KV-head accumulated prefill attention is unavailable for "
                f"layer={layer_idx}, expected=({head_count}, >= {seq_len})"
            )
        return mx.maximum(attention[:, :seq_len].astype(mx.float32), 0.0)

    def _prefill_window_attention_by_head(
        self,
        layer_idx: int,
        seq_len: int,
        head_count: int,
    ):
        """Return final-query-window attention without all-query materialization."""
        import mlx.core as mx

        window = min(
            max(1, int(getattr(self.cfg.eviction, "observation_window", 32))),
            int(seq_len),
        )
        observed = self.attention_state.get("observe_heads", {}).get(layer_idx, [])
        usable = [
            row[:, :seq_len]
            for row in observed[-window:]
            if int(row.shape[-1]) >= int(seq_len)
        ]
        if not usable:
            raise ScoreUnavailableError(
                "per-KV-head observation-window prefill attention is unavailable for "
                f"layer={layer_idx}, expected=({head_count}, >= {seq_len})"
            )
        attention = mx.sum(mx.stack(usable, axis=0), axis=0).astype(mx.float32)
        if int(attention.shape[0]) != int(head_count):
            raise ScoreUnavailableError(
                "observation-window attention head mismatch for "
                f"layer={layer_idx}, expected={head_count}, got={attention.shape[0]}"
            )
        # Match SnapKV's configured local smoothing before selecting the core.
        attention = self._pool_scores_by_head(attention)
        return mx.maximum(attention[:, :seq_len], 0.0)

    @staticmethod
    def _proportional_head_allocation(
        weights: np.ndarray,
        total_budget: int,
        per_head_cap: int,
    ) -> np.ndarray:
        """Integer proportional allocation with one slot per head when possible."""
        weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        head_count = int(weights.size)
        cap = max(0, int(per_head_cap))
        total = min(max(0, int(total_budget)), head_count * cap)
        allocation = np.zeros(head_count, dtype=np.int64)
        if head_count == 0 or total == 0 or cap == 0:
            return allocation
        if total >= head_count:
            allocation[:] = 1
            total -= head_count
        safe = np.where(np.isfinite(weights) & (weights > 0), weights, 0.0)
        if float(safe.sum()) <= 0:
            safe[:] = 1.0
        if total > 0:
            target = total * safe / float(safe.sum())
            room = cap - allocation
            extra = np.minimum(np.floor(target).astype(np.int64), room)
            allocation += extra
            residual = target - np.floor(target)
            while int(allocation.sum()) < min(int(total_budget), head_count * cap):
                eligible = np.flatnonzero(allocation < cap)
                if eligible.size == 0:
                    break
                best = int(eligible[np.argmax(residual[eligible])])
                allocation[best] += 1
                residual[best] -= 1.0
        return allocation

    def _innovation_headwise_keep(
        self,
        c: Any,
        layer_idx: int,
        seq_len: int,
        budget: int,
    ):
        """One-shot per-head implementations of the proposed geometry methods."""
        import mlx.core as mx

        method = self.method
        keys = self._score_rows_by_head(c, seq_len, source="k")
        values = self._score_rows_by_head(c, seq_len, source="v")
        head_count = int(values.shape[0])
        take = min(int(seq_len), max(1, int(budget)))
        attention = None
        shared_attention_core: Optional[List[int]] = None
        if method in {
            "attention_residual_v_leverage",
            "window_residual_v_leverage",
            "attention_weighted_v_leverage",
            "window_weighted_v_leverage",
        }:
            if method in {
                "window_residual_v_leverage",
                "window_weighted_v_leverage",
            }:
                attention = self._prefill_window_attention_by_head(
                    layer_idx, seq_len, head_count
                )
            else:
                attention = self._prefill_accumulated_attention_by_head(
                    layer_idx, seq_len, head_count
                )
        if method in {
            "attention_residual_v_leverage",
            "window_residual_v_leverage",
        }:
            ratio = max(0.0, min(1.0, float(getattr(
                self.cfg.eviction, "attention_residual_budget_ratio", 0.5
            ))))
            attention_budget = min(take, max(1, int(round(take * ratio))))
            shared_attention_core = self._top_score_indices_numpy(
                np.asarray(mx.mean(attention, axis=0).tolist()), attention_budget
            )

        scores_by_head: List[Any] = []
        keep_by_head: List[List[int]] = []
        effective_dimensions: List[float] = []
        whitened_by_head: List[Any] = []

        for head_idx in range(head_count):
            k_rows = keys[head_idx]
            v_rows = values[head_idx]

            if method in {"conditional_v_leverage", "conditional_k_leverage"}:
                if method == "conditional_v_leverage":
                    target, condition = v_rows, k_rows
                else:
                    target, condition = k_rows, v_rows
                residual, regression_lambda = self._conditional_residual_rows(
                    target,
                    condition,
                    float(getattr(self.cfg.eviction, "conditional_ridge_lambda", 1e-3)),
                )
                score, _, leverage_lambda, d_eff = self._ridge_leverage_rows(
                    residual,
                    float(getattr(self.cfg.eviction, "conditional_leverage_lambda", 1e-3)),
                )
                keep = self._top_score_indices_numpy(np.asarray(score.tolist()), take)
                self.estimator_events.append({
                    "layer": int(layer_idx), "head": int(head_idx), "phase": "prefill",
                    "score_type": method, "calculation": "conditional_ridge_residual_leverage",
                    "regression_lambda": regression_lambda,
                    "leverage_lambda": leverage_lambda, "effective_dimension": d_eff,
                    "failed": False,
                })

            elif method in {
                "attention_residual_v_leverage",
                "window_residual_v_leverage",
            }:
                attn = attention[head_idx]
                core = list(shared_attention_core or [])
                selected_values = mx.take(
                    v_rows, mx.array(core, dtype=mx.int32), axis=0
                )
                dim = int(v_rows.shape[-1])
                gram = mx.matmul(selected_values.transpose(1, 0), selected_values)
                projection_lambda = self._relative_ridge_lambda(
                    selected_values,
                    float(getattr(
                        self.cfg.eviction, "attention_residual_ridge_lambda", 1e-3
                    )),
                )
                projection = mx.linalg.solve(
                    gram + mx.eye(dim, dtype=mx.float32) * projection_lambda,
                    gram,
                    stream=mx.cpu,
                )
                residual = v_rows - mx.matmul(v_rows, projection)
                score, _, leverage_lambda, d_eff = self._ridge_leverage_rows(
                    residual,
                    float(getattr(
                        self.cfg.eviction, "attention_residual_ridge_lambda", 1e-3
                    )),
                )
                geometry = self._top_score_indices_numpy(
                    np.asarray(score.tolist()), take - len(core), excluded=set(core)
                )
                keep = core + geometry
                if len(keep) < take:
                    keep += self._top_score_indices_numpy(
                        np.asarray(attn.tolist()), take - len(keep), excluded=set(keep)
                    )
                self.estimator_events.append({
                    "layer": int(layer_idx), "head": int(head_idx), "phase": "prefill",
                    "score_type": method,
                    "calculation": (
                        "window_attention_core_residual_v_leverage"
                        if method == "window_residual_v_leverage"
                        else "attention_core_residual_v_leverage"
                    ),
                    "attention_query_window": (
                        int(getattr(self.cfg.eviction, "observation_window", 32))
                        if method == "window_residual_v_leverage"
                        else None
                    ),
                    "attention_budget": len(core), "geometry_budget": len(geometry),
                    "projection_lambda": projection_lambda,
                    "leverage_lambda": leverage_lambda, "effective_dimension": d_eff,
                    "failed": False,
                })

            elif method in {
                "attention_weighted_v_leverage",
                "window_weighted_v_leverage",
            }:
                attn = attention[head_idx]
                epsilon = max(1e-8, float(getattr(
                    self.cfg.eviction, "attention_weight_epsilon", 1e-4
                )))
                # Normalize only the global scale.  Relative token weights—and
                # therefore the query-conditioned geometry—remain unchanged.
                attn = attn / mx.maximum(mx.mean(attn), epsilon)
                weighted = v_rows * mx.sqrt(attn + epsilon).reshape(seq_len, 1)
                score, _, leverage_lambda, d_eff = self._ridge_leverage_rows(
                    weighted,
                    float(getattr(
                        self.cfg.eviction, "attention_weighted_ridge_lambda", 1e-3
                    )),
                )
                keep = self._top_score_indices_numpy(np.asarray(score.tolist()), take)
                self.estimator_events.append({
                    "layer": int(layer_idx), "head": int(head_idx), "phase": "prefill",
                    "score_type": method,
                    "calculation": (
                        "window_attention_weighted_v_ridge_leverage"
                        if method == "window_weighted_v_leverage"
                        else "attention_weighted_v_ridge_leverage"
                    ),
                    "attention_query_window": (
                        int(getattr(self.cfg.eviction, "observation_window", 32))
                        if method == "window_weighted_v_leverage"
                        else None
                    ),
                    "leverage_lambda": leverage_lambda, "effective_dimension": d_eff,
                    "failed": False,
                })

            elif method == "joint_kv_leverage":
                gamma = max(0.0, min(1.0, float(getattr(
                    self.cfg.eviction, "joint_kv_gamma", 0.5
                ))))
                k_scale = mx.sqrt(mx.maximum(mx.mean(k_rows * k_rows), 1e-12))
                v_scale = mx.sqrt(mx.maximum(mx.mean(v_rows * v_rows), 1e-12))
                joint = mx.concatenate([
                    k_rows / k_scale * math.sqrt(gamma),
                    v_rows / v_scale * math.sqrt(1.0 - gamma),
                ], axis=1)
                score, _, leverage_lambda, d_eff = self._ridge_leverage_rows(
                    joint,
                    float(getattr(self.cfg.eviction, "joint_kv_ridge_lambda", 1e-6)),
                )
                keep = self._top_score_indices_numpy(np.asarray(score.tolist()), take)
                self.estimator_events.append({
                    "layer": int(layer_idx), "head": int(head_idx), "phase": "prefill",
                    "score_type": method, "calculation": "scale_normalized_joint_kv_leverage",
                    "gamma": gamma, "leverage_lambda": leverage_lambda,
                    "effective_dimension": d_eff, "failed": False,
                })

            elif method in {
                "ridge_v_allocation",
                "ridge_v_fixed",
                "ridge_v_shared",
                "diversity_v_leverage",
            }:
                if method in {
                    "ridge_v_allocation",
                    "ridge_v_fixed",
                    "ridge_v_shared",
                }:
                    dim = int(v_rows.shape[-1])
                    gram = mx.matmul(v_rows.transpose(1, 0), v_rows)
                    eigenvalues, _ = mx.linalg.eigh(gram, stream=mx.cpu)
                    eigen_np = np.maximum(
                        np.asarray(eigenvalues.tolist(), dtype=np.float64), 0.0
                    )
                    target_rank = min(
                        dim,
                        max(1, int(round(dim * float(take) / float(seq_len)))),
                    )
                    tail_count = max(0, dim - target_rank)
                    tail_energy = float(eigen_np[:tail_count].sum()) if tail_count else 0.0
                    mean_eigenvalue = float(eigen_np.mean()) if eigen_np.size else 0.0
                    floor = float(getattr(
                        self.cfg.eviction, "ridge_budget_lambda_floor", 1e-4
                    )) * max(mean_eigenvalue, 1e-8)
                    adaptive_lambda = max(floor, tail_energy / float(target_rank), 1e-8)
                    score, _, leverage_lambda, d_eff = self._ridge_leverage_rows(
                        v_rows, 0.0, absolute_lambda=adaptive_lambda
                    )
                    keep = (
                        []
                        if method == "ridge_v_allocation"
                        else self._top_score_indices_numpy(
                            np.asarray(score.tolist()), take
                        )
                    )
                    self.estimator_events.append({
                        "layer": int(layer_idx), "head": int(head_idx), "phase": "prefill",
                        "score_type": method, "calculation": "budget_adaptive_ridge_v_leverage",
                        "target_rank": target_rank, "tail_energy": tail_energy,
                        "leverage_lambda": leverage_lambda,
                        "effective_dimension": d_eff, "failed": False,
                    })
                else:
                    score, whitened, leverage_lambda, d_eff = self._ridge_leverage_rows(
                        v_rows,
                        float(getattr(self.cfg.eviction, "diversity_ridge_lambda", 1e-3)),
                    )
                    whitened_by_head.append(whitened)
                    keep = self._top_score_indices_numpy(
                        np.asarray(score.tolist()), take
                    )
                    self.estimator_events.append({
                        "layer": int(layer_idx), "head": int(head_idx), "phase": "prefill",
                        "score_type": method, "calculation": "ridge_candidates_pending_shared_pivoted_qr",
                        "leverage_lambda": leverage_lambda,
                        "effective_dimension": d_eff, "failed": False,
                    })
                effective_dimensions.append(d_eff)

            else:
                raise ValueError(f"unsupported innovation method={method!r}")

            scores_by_head.append(score.astype(mx.float32))
            keep_by_head.append(sorted(set(int(x) for x in keep)))

        score_matrix = mx.stack(scores_by_head, axis=0)
        if method == "ridge_v_allocation":
            allocation = self._proportional_head_allocation(
                np.asarray(effective_dimensions, dtype=np.float64),
                total_budget=min(seq_len * head_count, take * head_count),
                per_head_cap=seq_len,
            )
            keep_by_head = [
                sorted(self._top_score_indices_numpy(
                    np.asarray(scores_by_head[head].tolist()), int(allocation[head])
                ))
                for head in range(head_count)
            ]
            for event in self.estimator_events[-head_count:]:
                if event.get("score_type") == method:
                    event["allocated_budget"] = int(
                        allocation[int(event.get("head", 0))]
                    )
        elif method != "ridge_v_fixed":
            aggregate = np.asarray(
                mx.mean(score_matrix, axis=0).tolist(), dtype=np.float64
            )
            if method in {
                "attention_residual_v_leverage",
                "window_residual_v_leverage",
            }:
                shared = list(shared_attention_core or [])
                shared += self._top_score_indices_numpy(
                    aggregate, take - len(shared), excluded=set(shared)
                )
            elif method == "diversity_v_leverage":
                multiplier = max(1, int(getattr(
                    self.cfg.eviction, "diversity_candidate_multiplier", 4
                )))
                candidate_count = min(seq_len, max(take, multiplier * take))
                candidates = self._top_score_indices_numpy(
                    aggregate, candidate_count
                )
                joint_whitened = mx.concatenate(whitened_by_head, axis=1)
                candidate_rows = np.asarray(
                    mx.take(
                        joint_whitened,
                        mx.array(candidates, dtype=mx.int32),
                        axis=0,
                    ).tolist(),
                    dtype=np.float64,
                )
                from scipy.linalg import qr

                _, r_matrix, pivots = qr(
                    candidate_rows.T,
                    mode="economic",
                    pivoting=True,
                    check_finite=False,
                )
                diagonal = np.abs(np.diag(r_matrix))
                tolerance = (
                    max(candidate_rows.shape)
                    * np.finfo(np.float64).eps
                    * (float(diagonal.max()) if diagonal.size else 0.0)
                )
                numerical_rank = int(np.sum(diagonal > tolerance))
                diversity_take = min(take, numerical_rank, len(candidates))
                shared = [int(candidates[int(p)]) for p in pivots[:diversity_take]]
                if len(shared) < take:
                    shared += [
                        idx for idx in candidates if idx not in set(shared)
                    ][: take - len(shared)]
                for event in self.estimator_events[-head_count:]:
                    if event.get("score_type") == method:
                        event.update({
                            "calculation": "shared_ridge_candidates_pivoted_qr",
                            "candidate_count": candidate_count,
                            "pivoted_qr_rank": numerical_rank,
                        })
            else:
                shared = self._top_score_indices_numpy(aggregate, take)
            shared = sorted(set(int(x) for x in shared))
            keep_by_head = [list(shared) for _ in range(head_count)]

        return keep_by_head, score_matrix

    def _pool_scores_by_head(self, scores: Any):
        import mlx.core as mx

        if scores is None:
            return None
        kernel = int(
            getattr(
                self.cfg.eviction,
                "pooling_kernel",
                getattr(self.cfg.eviction, "kernel_size", 1),
            )
            or 1
        )
        if kernel <= 1 or int(scores.shape[-1]) <= 1:
            return scores.astype(mx.float32)
        method = str(getattr(self.cfg.eviction, "pooling_method", "avgpool") or "avgpool").lower()
        values = np.array(scores.tolist(), dtype=np.float32)
        pad = max(0, kernel // 2)
        padded = np.pad(values, ((0, 0), (pad, pad)), mode="edge")
        windows = np.lib.stride_tricks.sliding_window_view(padded, kernel, axis=1)[:, : values.shape[1], :]
        if method in {"avg", "mean", "avgpool"}:
            pooled = windows.mean(axis=-1)
        elif method in {"max", "maxpool"}:
            pooled = windows.max(axis=-1)
        else:
            raise ValueError(f"Unsupported pooling_method={method!r}; expected avgpool or maxpool")
        return mx.array(pooled.astype(np.float32))

    def _compactor_headwise_keep(self, c: Any, layer_idx: int, seq_len: int, budget: int):
        import mlx.core as mx

        q_post, k_post, k_pre = self._compactor_prefill_qk(layer_idx, c, seq_len)
        scores = self._compactor_scores(q_post, k_post, k_pre, seq_len)
        h_count = int(scores.shape[1])
        protected_first, protected_last = self._compactor_protected_counts(seq_len, budget)
        scores_np = np.asarray(scores.tolist(), dtype=np.float32)
        finite_max = float(scores_np[np.isfinite(scores_np)].max()) if np.isfinite(scores_np).any() else 0.0
        protected_score = finite_max + max(1.0, abs(finite_max) * 0.01)
        if protected_first > 0:
            scores_np[:protected_first, :] = protected_score
        if protected_last > 0:
            scores_np[max(0, seq_len - protected_last) : seq_len, :] = protected_score
        scores = mx.array(scores_np)
        top_k = min(seq_len * h_count, max(1, int(budget) * h_count))
        flat = scores.reshape(-1)
        if top_k >= int(flat.shape[0]):
            chosen = mx.arange(int(flat.shape[0]))
        else:
            chosen = mx.argpartition(-flat, max(0, top_k - 1))[:top_k]
        keep_by_head: List[List[int]] = [[] for _ in range(h_count)]
        for raw in chosen.tolist():
            token = int(raw) // h_count
            head = int(raw) - token * h_count
            if 0 <= token < seq_len:
                keep_by_head[head].append(token)
        return [sorted(set(v)) for v in keep_by_head], mx.transpose(scores, (1, 0))

    def _compactor_prefill_qk(self, layer_idx: int, c: Any, seq_len: int):
        import mlx.core as mx

        def concat_chunks(name: str):
            chunks = self.attention_state.get(name, {}).get(layer_idx, [])
            if not chunks:
                return None
            return mx.concatenate(chunks, axis=2)[:, :, :seq_len, :].astype(mx.float32)

        q_post = concat_chunks("prefill_q_post")
        k_post = concat_chunks("prefill_k_post")
        k_pre = concat_chunks("prefill_k_pre")
        if q_post is None or k_post is None or k_pre is None:
            missing = [
                name
                for name, value in (("q_post", q_post), ("k_post", k_post), ("k_pre", k_pre))
                if value is None
            ]
            raise ScoreUnavailableError(
                f"Compactor prefill tensors unavailable for layer={layer_idx}: {missing}"
            )
        q_post = q_post[0].transpose(1, 0, 2)
        k_post = k_post[0].transpose(1, 0, 2)
        k_pre = k_pre[0].transpose(1, 0, 2)
        return q_post, k_post, k_pre

    def _compactor_scores(self, q_post: Any, k_post: Any, k_pre: Any, seq_len: int):
        import mlx.core as mx

        leverage = self._compactor_leverage_scores(k_pre)
        attn = self._compactor_noncausal_attention_scores(q_post, k_post)
        blend = float(getattr(self.cfg.eviction, "compactor_accum_blending", 0.5))
        return attn + leverage * blend

    def _compactor_leverage_scores(self, k_pre: Any):
        import mlx.core as mx

        n, h_count, dim = [int(x) for x in k_pre.shape]
        sketch_dim = max(1, min(int(getattr(self.cfg.eviction, "compactor_sketch_dim", 48)), dim))
        seed = int(getattr(self.cfg, "seed", 0)) + 1777
        rng = np.random.default_rng(seed)
        phi_np = rng.standard_normal((dim, sketch_dim)).astype(np.float32) / math.sqrt(float(sketch_dim))
        phi = mx.array(phi_np)
        x = mx.matmul(k_pre.transpose(1, 0, 2), phi)
        chunk_size = int(getattr(self.cfg.eviction, "compactor_chunk_size", 512))
        chunk_size = n if chunk_size <= 0 else max(1, chunk_size)
        scores_np = np.zeros((n, h_count), dtype=np.float32)
        reg = 5e-3
        for h in range(h_count):
            for start in range(0, n, chunk_size):
                end = min(n, start + chunk_size)
                chunk = x[h, start:end, :].astype(mx.float32)
                chunk = chunk - mx.mean(chunk, axis=0, keepdims=True)
                gram = mx.matmul(chunk.transpose(1, 0), chunk)
                gram = gram + mx.eye(int(gram.shape[0]), dtype=mx.float32) * reg
                u, s, _ = mx.linalg.svd(gram, stream=mx.cpu)
                s = mx.maximum(s, 1e-8)
                sv = u * (1.0 / mx.sqrt(s)).reshape(1, -1)
                proj = mx.matmul(chunk, sv)
                vals = mx.sum(proj * proj, axis=1)
                scores_np[start:end, h] = np.asarray(vals.tolist(), dtype=np.float32)
        return _zscore_mx(mx.array(scores_np))

    def _compactor_noncausal_attention_scores(self, q_post: Any, k_post: Any):
        import mlx.core as mx

        n, hq, dim = [int(x) for x in q_post.shape]
        hkv = int(k_post.shape[1])
        group = max(1, hq // max(1, hkv))
        chunk_size = max(1, int(getattr(self.cfg.eviction, "compactor_attention_chunk_size", 128)))
        out_np = np.zeros((n, hkv), dtype=np.float32)
        for start in range(0, n, chunk_size):
            end = min(n, start + chunk_size)
            for h in range(hkv):
                qh = q_post[start:end, h * group : (h + 1) * group, :].reshape(-1, dim).astype(mx.float32)
                kh = k_post[start:end, h, :].astype(mx.float32)
                logits = mx.matmul(qh, kh.transpose(1, 0))
                probs = mx.softmax(logits, axis=-1, precise=True)
                out_np[start:end, h] = np.asarray(mx.sum(probs, axis=0).tolist(), dtype=np.float32)
        return _zscore_mx(mx.array(out_np))

    def _compactor_protected_counts(self, seq_len: int, budget: int) -> Tuple[int, int]:
        first = getattr(self.cfg.eviction, "compactor_protected_first_tokens", None)
        last = getattr(self.cfg.eviction, "compactor_protected_last_tokens", None)
        # Compactor itself does not prescribe fixed boundary protection. Keep
        # it opt-in so the named method is not silently mixed with a heuristic.
        first = 0 if first is None else int(first)
        last = 0 if last is None else int(last)
        if first + last >= seq_len:
            return 0, 0
        first = min(max(0, first), max(0, int(budget)))
        last = min(max(0, last), max(0, int(budget) - first))
        return first, last

    def _apply_headwise_keep(self, c: Any, layer_idx: int, keep_by_head: List[List[int]], seq_len: int) -> None:
        import mlx.core as mx

        old_position_map = self.position_maps.get(layer_idx)
        if old_position_map is None or int(old_position_map.shape[0]) < seq_len:
            old_position_map = mx.arange(seq_len)
        h_count = int(c.keys.shape[1])
        max_len = max(1, max((len(v) for v in keep_by_head), default=0))
        key_parts = []
        value_parts = []
        map_rows = []
        valid_rows = []
        selected_by_head: Dict[str, List[int]] = {}
        selected_union = set()
        for h in range(h_count):
            values = keep_by_head[h] if h < len(keep_by_head) else []
            idx = mx.array(values, dtype=mx.int32) if values else mx.array([], dtype=mx.int32)
            n = int(idx.shape[0])
            if n > 0:
                kh = mx.take(c.keys[:, h : h + 1, :seq_len, :], idx, axis=2)
                vh = mx.take(c.values[:, h : h + 1, :seq_len, :], idx, axis=2)
                pos = mx.take(old_position_map[:seq_len], idx, axis=0)
                pos_vals = [int(x) for x in pos.tolist()]
            else:
                kh = mx.zeros((c.keys.shape[0], 1, 0, c.keys.shape[-1]), dtype=c.keys.dtype)
                vh = mx.zeros((c.values.shape[0], 1, 0, c.values.shape[-1]), dtype=c.values.dtype)
                pos = mx.array([], dtype=mx.int32)
                pos_vals = []
            if n < max_len:
                pad_n = max_len - n
                kh = mx.concatenate([kh, mx.zeros((c.keys.shape[0], 1, pad_n, c.keys.shape[-1]), dtype=c.keys.dtype)], axis=2)
                vh = mx.concatenate([vh, mx.zeros((c.values.shape[0], 1, pad_n, c.values.shape[-1]), dtype=c.values.dtype)], axis=2)
                pos = mx.concatenate([pos, mx.full((pad_n,), -1, dtype=mx.int32)], axis=0)
            key_parts.append(kh)
            value_parts.append(vh)
            map_rows.append(pos.reshape(1, max_len))
            valid_row = [True] * n + [False] * (max_len - n)
            valid_rows.append(mx.array(valid_row, dtype=mx.bool_).reshape(1, max_len))
            selected_by_head[str(h)] = pos_vals
            selected_union.update(pos_vals)
        c.keys = mx.concatenate(key_parts, axis=1)
        c.values = mx.concatenate(value_parts, axis=1)
        c.offset = int(max_len)
        c.logical_offset = int(getattr(c, "logical_offset", seq_len))
        c.head_position_map = mx.concatenate(map_rows, axis=0)
        c.head_valid_mask = mx.concatenate(valid_rows, axis=0)
        c.valid_token_head_slots = int(sum(len(v) for v in keep_by_head))
        self.head_position_maps[layer_idx] = c.head_position_map
        self.position_maps[layer_idx] = mx.array(sorted(selected_union), dtype=mx.int32)
        self.next_positions[layer_idx] = int(c.logical_offset)
        self.last_selected[layer_idx] = sorted(selected_union)
        self.last_selected_by_head[layer_idx] = selected_by_head

    def _apply_shared_prefill_keep(
        self,
        c: Any,
        layer_idx: int,
        keep_values: List[int],
        seq_len: int,
    ) -> None:
        """Apply one common token set without enabling the sparse-head kernel."""
        import mlx.core as mx

        keep_values = sorted({
            int(x) for x in keep_values if 0 <= int(x) < int(seq_len)
        })
        keep = mx.array(keep_values, dtype=mx.int32)
        old_position_map = self.position_maps.get(layer_idx)
        if old_position_map is None or int(old_position_map.shape[0]) < seq_len:
            old_position_map = mx.arange(seq_len)
        selected_positions = mx.take(old_position_map[:seq_len], keep, axis=0)
        c.keys = mx.take(c.keys[:, :, :seq_len, :], keep, axis=2)
        c.values = mx.take(c.values[:, :, :seq_len, :], keep, axis=2)
        c.offset = int(keep.shape[0])
        c.logical_offset = int(getattr(c, "logical_offset", seq_len))
        # A fresh experiment cache should not have these attributes, but clear
        # them defensively so the optimized dense attention path is explicit.
        for attr in ("head_position_map", "head_valid_mask"):
            if hasattr(c, attr):
                delattr(c, attr)
        selected = [int(x) for x in selected_positions.tolist()]
        self.position_maps[layer_idx] = selected_positions
        self.next_positions[layer_idx] = int(c.logical_offset)
        self.last_selected[layer_idx] = selected
        self.last_selected_by_head[layer_idx] = {
            str(head): list(selected) for head in range(int(c.keys.shape[1]))
        }

    def _cache_num_heads(self, layer_idx: int) -> int:
        observed = self.attention_state.get("observe_heads", {}).get(layer_idx, [])
        if observed:
            return int(observed[-1].shape[0])
        return int(self.model_info_num_kv_heads())

    def model_info_num_kv_heads(self) -> int:
        return int(getattr(self.cfg.model, "num_key_value_heads", 1) or 1)

    def _score_rows_by_head(self, c: Any, seq_len: int, source: Optional[str] = None):
        import mlx.core as mx

        source = (source or self.cfg.eviction.score_source).lower()
        source = {"value": "v", "key": "k", "key_value_concat": "kv"}.get(source, source)
        if source not in {"v", "k", "kv"}:
            raise ValueError(f"unsupported geometry score_source={source!r}")
        # [batch, kv_head, token, feature] -> [kv_head, token, feature].
        # Leverage is computed independently for every KV head. Averaging rows
        # before fitting changes the represented subspace and is not a valid
        # head-wise leverage estimate.
        v_rows = c.values[:, :, :seq_len, :][0].astype(mx.float32)
        if source == "v":
            return v_rows
        k_rows = c.keys[:, :, :seq_len, :][0].astype(mx.float32)
        if source == "k":
            return k_rows
        return mx.concatenate([k_rows, v_rows], axis=-1)

    def _score_rows(self, c: Any, seq_len: int, source: Optional[str] = None):
        """Return the explicit legacy head-mean row representation.

        This helper remains only for Compactor parity code. Paper-path
        geometry methods call :meth:`_geometry_scores_by_head` instead.
        """
        import mlx.core as mx

        return mx.mean(self._score_rows_by_head(c, seq_len, source=source), axis=0)

    def _geometry_scores_by_head(
        self,
        c: Any,
        layer_idx: int,
        seq_len: int,
        *,
        source: str,
        score_kind: str,
        force_refit: Optional[bool] = None,
    ):
        import mlx.core as mx

        rows_by_head = self._score_rows_by_head(c, seq_len, source=source)
        scores = []
        for head_idx in range(int(rows_by_head.shape[0])):
            rows = rows_by_head[head_idx]
            if score_kind == "l1_leverage":
                score = self._estimator_scores(
                    self._get_estimator("l1", layer_idx, head_idx),
                    rows,
                    layer_idx,
                    f"{source}_l1_leverage",
                    head_idx=head_idx,
                    force_refit=force_refit,
                )
            elif score_kind == "l2_leverage":
                score = self._estimator_scores(
                    self._get_estimator("l2", layer_idx, head_idx),
                    rows,
                    layer_idx,
                    f"{source}_l2_leverage",
                    head_idx=head_idx,
                    force_refit=force_refit,
                )
            elif score_kind == "l1_norm":
                score = mx.sum(mx.abs(rows), axis=1)
            elif score_kind == "l2_norm":
                score = mx.sqrt(mx.sum(rows * rows, axis=1))
            else:
                raise ValueError(f"unsupported geometry score_kind={score_kind!r}")
            scores.append(score.astype(mx.float32))
        by_head = mx.stack(scores, axis=0)
        self._current_scores_by_head[layer_idx] = by_head
        self._record_head_scores(layer_idx, by_head)
        return by_head

    def _aggregate_head_scores(self, scores_by_head: Any):
        import mlx.core as mx

        strategy = str(getattr(self.cfg.eviction, "head_strategy", "shared") or "shared").lower()
        if strategy not in {"shared", "mean", "shared_mean"}:
            raise ValueError(
                "live per-head cache mutation is not implemented for head_strategy="
                f"{strategy!r}; use shared (mean of independently computed head scores)"
            )
        return mx.mean(scores_by_head, axis=0).astype(mx.float32)

    def _keydiff_scores_by_head(self, c: Any, layer_idx: int, seq_len: int):
        """Efficient KeyDiff score from Eq. (8): -cos(key, mean(keys))."""
        import mlx.core as mx

        keys = self._score_rows_by_head(c, seq_len, source="k")
        anchor = mx.mean(keys, axis=1, keepdims=True)
        numerator = mx.sum(keys * anchor, axis=-1)
        key_norm = mx.sqrt(mx.sum(keys * keys, axis=-1))
        anchor_norm = mx.sqrt(mx.sum(anchor * anchor, axis=-1))
        cosine = numerator / mx.maximum(key_norm * anchor_norm, 1e-8)
        scores = -cosine.astype(mx.float32)
        self._current_scores_by_head[layer_idx] = scores
        self._record_head_scores(layer_idx, scores)
        return scores

    def _curdkv_scores_by_head(self, c: Any, layer_idx: int, seq_len: int):
        """CurDKV Algorithm 1 Gaussian projected K/V row-norm product."""
        import mlx.core as mx

        keys = self._score_rows_by_head(c, seq_len, source="k")
        values = self._score_rows_by_head(c, seq_len, source="v")
        dim = min(int(keys.shape[-1]), int(values.shape[-1]))
        rank = max(
            1,
            min(int(getattr(self.cfg.eviction, "curdkv_projection_dim", 20)), dim),
        )
        rng = np.random.default_rng(
            int(self.cfg.seed) + int(layer_idx) * 1009 + int(seq_len) * 9173
        )
        projection = mx.array(
            rng.normal(0.0, 1.0 / math.sqrt(float(rank)), size=(dim, rank)).astype(np.float32)
        )
        key_projected = mx.matmul(keys[..., :dim], projection)
        value_projected = mx.matmul(values[..., :dim], projection)
        key_score = mx.sum(key_projected * key_projected, axis=-1)
        value_score = mx.sum(value_projected * value_projected, axis=-1)
        scores = key_score * value_score
        scores = scores / mx.maximum(mx.sum(scores, axis=1, keepdims=True), 1e-12)
        scores = scores.astype(mx.float32)
        self._current_scores_by_head[layer_idx] = scores
        self._record_head_scores(layer_idx, scores)
        return scores

    def _dynamic_geometry_scores_by_head(
        self,
        c: Any,
        layer_idx: int,
        seq_len: int,
        *,
        source: str,
        score_kind: str,
    ):
        """Refresh leverage scores only at the configured decode interval.

        Between refreshes, cached per-head scores are kept aligned by cache
        pruning.  Newly appended decode tokens receive zero geometric score
        until the next refresh, while the normal recent-token protection keeps
        them available.  This makes ``update_interval=N`` a real score-refresh
        interval and exposes the intended quality/latency tradeoff; merely
        reusing an old whitener while projecting every cache row on every token
        remains nearly as expensive as a full dynamic score computation.
        """
        import mlx.core as mx

        cached = self._current_scores_by_head.get(layer_idx)
        refresh = cached is None
        policy = str(
            getattr(self.cfg.eviction, "update_policy", "every_n_steps")
            or "every_n_steps"
        ).lower()
        if not refresh:
            if policy in {"prefill_only", "never_after_prefill"}:
                refresh = False
            elif policy == "decode_only" and self.phase == "prefill":
                refresh = False
            elif policy in {"every_n_steps", "decode_only"}:
                interval = int(getattr(self.cfg.eviction, "update_interval", 1))
                refresh = (
                    self.phase == "decode"
                    and interval > 0
                    and self.eviction_step > 0
                    and self.eviction_step % interval == 0
                )
            else:
                raise ValueError(f"unsupported update_policy={policy!r}")

        if refresh:
            return self._geometry_scores_by_head(
                c,
                layer_idx,
                seq_len,
                source=source,
                score_kind=score_kind,
                force_refit=True,
            )

        if cached is None:
            raise ScoreUnavailableError(
                f"no cached {score_kind} scores for layer={layer_idx} under policy={policy}"
            )
        cached_len = int(cached.shape[-1])
        if cached_len < seq_len:
            cached = mx.concatenate(
                [
                    cached,
                    mx.zeros(
                        (int(cached.shape[0]), seq_len - cached_len),
                        dtype=cached.dtype,
                    ),
                ],
                axis=1,
            )
        elif cached_len > seq_len:
            cached = cached[:, :seq_len]
        self._current_scores_by_head[layer_idx] = cached
        self._record_head_scores(layer_idx, cached)
        return cached

    def _compute_scores(self, c: Any, layer_idx: int, seq_len: int):
        import mlx.core as mx

        method = self.method
        if method in ("full", "basic", "basic_generate"):
            return None
        if method in ("random", "sink_recent_random", "oracle_evidence", "oracle_answer_region"):
            return None
        if method in ("recency", "sink_recent", "streamingllm"):
            return mx.arange(seq_len).astype(mx.float32)
        if method in ATTENTION_SCORE_METHODS:
            mode = "accumulated"
            if method == "windowed_attention":
                mode = "windowed"
            elif method == "attention_decay":
                mode = "decayed"
            elif method == "tova":
                mode = "last_query"
            scores = self._attention_scores(layer_idx, seq_len, mode=mode)
            if method == "vatp":
                attention_by_head = self._current_scores_by_head.get(layer_idx)
                value_l1_by_head = self._geometry_scores_by_head(
                    c,
                    layer_idx,
                    seq_len,
                    source="v",
                    score_kind="l1_norm",
                )
                if (
                    attention_by_head is not None
                    and tuple(attention_by_head.shape) == tuple(value_l1_by_head.shape)
                ):
                    # VATP is defined per attention head.  Query heads have already
                    # been reduced onto their corresponding physical KV head by the
                    # attention hook, so multiply before the shared-head reduction.
                    vatp_by_head = (
                        attention_by_head.astype(mx.float32)
                        * value_l1_by_head.astype(mx.float32)
                    )
                    self._current_scores_by_head[layer_idx] = vatp_by_head
                    self._record_head_scores(layer_idx, vatp_by_head)
                    scores = self._aggregate_head_scores(vatp_by_head)
                else:
                    # Defensive fallback for architectures whose exposed attention
                    # head layout cannot be mapped one-to-one to the KV cache.
                    scores = scores * self._aggregate_head_scores(value_l1_by_head)
            self._last_attn_scores[layer_idx] = scores
            return scores
        if method in SNAP_METHODS:
            scores = self._attention_scores(layer_idx, seq_len, mode="snapkv")
            self._last_attn_scores[layer_idx] = scores
            return scores
        if method in {
            "key_l2_norm",
            "value_l2_norm",
            "key_l1_norm",
            "value_l1_norm",
            "knorm",
            "vnorml1",
            "vnorml2",
        }:
            if method == "knorm":
                source, kind, sign = "k", "l2_norm", -1.0
            elif method == "vnorml1":
                source, kind, sign = "v", "l1_norm", 1.0
            elif method == "vnorml2":
                source, kind, sign = "v", "l2_norm", 1.0
            else:
                source = "k" if method.startswith("key") else "v"
                kind = "l1_norm" if "_l1_" in method else "l2_norm"
                sign = 1.0
            by_head = self._geometry_scores_by_head(
                c, layer_idx, seq_len, source=source, score_kind=kind
            )
            if sign < 0:
                by_head = -by_head
                self._current_scores_by_head[layer_idx] = by_head
                self._record_head_scores(layer_idx, by_head)
            return self._aggregate_head_scores(by_head)
        if method == "keydiff":
            return self._aggregate_head_scores(
                self._keydiff_scores_by_head(c, layer_idx, seq_len)
            )
        if method == "curdkv":
            return self._aggregate_head_scores(
                self._curdkv_scores_by_head(c, layer_idx, seq_len)
            )
        if method in ("l1_decode_only", "l2_decode_only") and self.phase == "prefill":
            return None
        score_source = str(self.cfg.eviction.score_source or "v").lower()
        score_source = {"value": "v", "key": "k", "key_value_concat": "kv"}.get(
            score_source, score_source
        )
        if method in (
            "l1_prefill_only",
            "l2_prefill_only",
            "l2_key_prefill_only",
            "compactor",
            "attention_l1_compactor",
            "attention_l2_compactor",
        ):
            cached = self._static_score_cache.get(layer_idx)
            if cached is None:
                if method == "l1_prefill_only":
                    by_head = self._geometry_scores_by_head(
                        c, layer_idx, seq_len, source=score_source,
                        score_kind="l1_leverage", force_refit=True,
                    )
                    cached = self._aggregate_head_scores(by_head)
                elif method == "l2_key_prefill_only":
                    by_head = self._geometry_scores_by_head(
                        c, layer_idx, seq_len, source="k",
                        score_kind="l2_leverage", force_refit=True,
                    )
                    cached = self._aggregate_head_scores(by_head)
                elif method in ("compactor", "attention_l1_compactor", "attention_l2_compactor"):
                    rows = self._score_rows(c, seq_len, source=score_source)
                    key_rows = self._score_rows(c, seq_len, source="k")
                    if method == "attention_l1_compactor":
                        geom = self._estimator_scores(
                            self._get_estimator("l1", layer_idx), rows, layer_idx, "value_l1_leverage", force_refit=True
                        )
                    else:
                        geom = self._estimator_scores(
                            self._get_estimator("l2", layer_idx), key_rows, layer_idx, "key_l2_leverage", force_refit=True
                        )
                    attn = self._attention_scores(layer_idx, seq_len, mode="accumulated")
                    self._last_attn_scores[layer_idx] = attn
                    self._last_geom_scores[layer_idx] = geom
                    cached = _merge_score_vectors(
                        attn,
                        geom,
                        self.cfg.eviction.lambda_attn,
                        self.cfg.eviction.score_normalization,
                    )
                    self._current_scores_by_head.pop(layer_idx, None)
                else:
                    by_head = self._geometry_scores_by_head(
                        c, layer_idx, seq_len, source=score_source,
                        score_kind="l2_leverage", force_refit=True,
                    )
                    cached = self._aggregate_head_scores(by_head)
                self._static_score_cache[layer_idx] = cached
            if int(cached.shape[0]) < seq_len:
                pad = mx.zeros((seq_len - int(cached.shape[0]),), dtype=cached.dtype)
                cached = mx.concatenate([cached, pad], axis=0)
                self._static_score_cache[layer_idx] = cached
            return cached[:seq_len]
        if method in ("l1", "l1_leverage", "l1_decode_only", "sink_recent_l1"):
            return self._aggregate_head_scores(
                self._dynamic_geometry_scores_by_head(
                    c, layer_idx, seq_len, source=score_source,
                    score_kind="l1_leverage",
                )
            )
        if method in ("l2", "l2_leverage", "l2_decode_only", "sink_recent_l2"):
            return self._aggregate_head_scores(
                self._dynamic_geometry_scores_by_head(
                    c, layer_idx, seq_len, source=score_source,
                    score_kind="l2_leverage",
                )
            )
        if method in {"key_l2_leverage", "value_l2_leverage", "kv_l2_leverage"}:
            explicit_source = {
                "key_l2_leverage": "k",
                "value_l2_leverage": "v",
                "kv_l2_leverage": "kv",
            }[method]
            return self._aggregate_head_scores(
                self._dynamic_geometry_scores_by_head(
                    c, layer_idx, seq_len, source=explicit_source,
                    score_kind="l2_leverage",
                )
            )
        if method in HYBRID_METHODS:
            attn = self._attention_scores(layer_idx, seq_len, mode="accumulated")
            rows = self._score_rows(c, seq_len, source=score_source)
            geom = self._geom_scores(rows, layer_idx)
            self._last_attn_scores[layer_idx] = attn
            self._last_geom_scores[layer_idx] = geom
            merged = _merge_score_vectors(
                attn,
                geom,
                self.cfg.eviction.lambda_attn,
                self.cfg.eviction.score_normalization,
            )
            self._current_scores_by_head.pop(layer_idx, None)
            return merged
        return None

    def _layer_budget(self, layer_idx: int, num_layers: int, base_budget: int) -> int:
        if self.method != "pyramidkv" or num_layers <= 1:
            return int(base_budget)
        total = int(base_budget) * int(num_layers)
        mode = str(getattr(self.cfg.eviction, "pyramid_mode", "funnel") or "funnel")
        if mode == "funnel":
            weights = np.arange(num_layers, 0, -1, dtype=np.float64)
        elif mode == "inverse_funnel":
            weights = np.arange(1, num_layers + 1, dtype=np.float64)
        else:
            return int(base_budget)
        raw = np.maximum((weights / weights.sum() * total).astype(int), 4)
        return max(1, min(int(base_budget), int(raw[int(layer_idx)])))

    def _select_indices(
        self,
        scores: Any,
        seq_len: int,
        budget: int,
        layer_idx: Optional[int] = None,
    ):
        import mlx.core as mx

        if seq_len <= budget:
            return mx.arange(seq_len)
        method = self.method
        if method in ("recency", "basic", "basic_generate"):
            return mx.arange(max(0, seq_len - budget), seq_len)
        if method in ("random", "sink_recent_random"):
            return self._select_random_indices(seq_len, budget, layer_idx)
        if method in ("oracle_evidence", "oracle_answer_region"):
            return self._select_oracle_indices(seq_len, budget, int(layer_idx or 0), scores)
        if method in ("l1_decode_only", "l2_decode_only") and self.phase == "prefill":
            return self._select_sink_recent_indices(seq_len, budget)
        if method in SNAP_METHODS:
            return self._select_snapkv_indices(scores, seq_len, budget)
        if method == "tova":
            return self._select_paper_topk_indices(scores, seq_len, budget)
        if method in {"h2o", "vatp"}:
            return self._select_h2o_indices(scores, seq_len, budget)
        if method in {"knorm", "vnorml1", "vnorml2"}:
            return self._select_paper_topk_indices(scores, seq_len, budget)
        if method == "keydiff":
            recent = int(
                max(0.0, min(1.0, float(getattr(self.cfg.eviction, "keydiff_recent_ratio", 0.0))))
                * int(budget)
            )
            return self._select_paper_topk_indices(
                scores, seq_len, budget, recent=recent
            )
        if method == "curdkv":
            return self._select_paper_topk_indices(
                scores,
                seq_len,
                budget,
                protected_first=int(getattr(self.cfg.eviction, "curdkv_num_sink", 4)),
            )
        if (
            method in HYBRID_METHODS
            and method not in COMPACTORLIKE_HYBRID_METHODS
            and self.cfg.eviction.hybrid_mode == "budget_split"
        ):
            return self._select_hybrid_indices(seq_len, budget, int(layer_idx or 0), scores)

        sink = min(self.cfg.eviction.sink_size, max(0, budget - 1))
        max_recent = max(0, budget - sink - 1)
        recent = min(self.cfg.eviction.recent_size, max_recent)
        mid_budget = max(0, budget - sink - recent - 1)
        parts = []
        if sink > 0:
            parts.append(mx.arange(sink))
        if recent > 0:
            parts.append(mx.arange(seq_len - 1 - recent, seq_len - 1))
        if mid_budget > 0 and scores is not None:
            start = sink
            end = seq_len - 1 - recent
            if end > start:
                cand = scores[start:end]
                take = min(mid_budget, cand.shape[0])
                if take >= cand.shape[0]:
                    idx = mx.arange(cand.shape[0])
                else:
                    idx = mx.argpartition(-cand, max(0, take - 1))[:take]
                parts.append(idx + start)
        parts.append(mx.array([seq_len - 1]))
        keep = mx.sort(mx.concatenate(parts)) if parts else mx.arange(seq_len)
        if keep.shape[0] > budget:
            keep = keep[:budget]
        return keep

    def _select_paper_topk_indices(
        self,
        scores: Any,
        seq_len: int,
        budget: int,
        *,
        protected_first: int = 0,
        recent: int = 0,
    ):
        """Top-score selection with only the protections specified by a paper."""
        import mlx.core as mx

        if seq_len <= budget:
            return mx.arange(seq_len)
        if scores is None:
            raise ScoreUnavailableError("paper-method selection requires token scores")
        protected_first = min(max(0, int(protected_first)), int(budget), int(seq_len))
        recent = min(
            max(0, int(recent)),
            max(0, int(budget) - protected_first),
            max(0, int(seq_len) - protected_first),
        )
        reserved = list(range(protected_first))
        if recent:
            reserved.extend(range(max(protected_first, seq_len - recent), seq_len))
        reserved = sorted(set(reserved))
        take = max(0, int(budget) - len(reserved))
        reserved_set = set(reserved)
        candidates = [idx for idx in range(seq_len) if idx not in reserved_set]
        if take and candidates:
            candidate_idx = mx.array(candidates, dtype=mx.int32)
            candidate_scores = mx.take(scores[:seq_len], candidate_idx, axis=0)
            take = min(take, len(candidates))
            if take >= len(candidates):
                chosen = candidate_idx
            else:
                local = mx.argpartition(-candidate_scores, max(0, take - 1))[:take]
                chosen = mx.take(candidate_idx, local, axis=0)
            reserved.extend(int(x) for x in chosen.tolist())
        return mx.array(sorted(set(reserved))[: int(budget)], dtype=mx.int32)

    def _select_h2o_indices(self, scores: Any, seq_len: int, budget: int):
        """H2O/VATP balance of accumulated-attention heavy hitters and recency."""
        configured = getattr(self.cfg.eviction, "h2o_recent_size", None)
        recent = int(configured) if configured is not None else max(1, int(budget) // 2)
        recent = min(recent, int(budget), int(seq_len))
        return self._select_paper_topk_indices(
            scores, seq_len, budget, recent=recent
        )

    def _select_sink_recent_indices(self, seq_len: int, budget: int):
        import mlx.core as mx

        if seq_len <= budget:
            return mx.arange(seq_len)
        sink = min(int(self.cfg.eviction.sink_size), max(0, int(budget)))
        recent_budget = max(0, int(budget) - sink)
        recent_start = max(sink, seq_len - recent_budget)
        keep = sorted(set(range(sink)) | set(range(recent_start, seq_len)))
        return mx.array(keep[:budget], dtype=mx.int32)

    def _select_random_indices(self, seq_len: int, budget: int, layer_idx: Optional[int]):
        import mlx.core as mx

        if seq_len <= budget:
            return mx.arange(seq_len)
        rng = np.random.default_rng(
            int(self.cfg.seed) + int(layer_idx or 0) * 1009 + self.eviction_count * 9173 + seq_len
        )
        reserved_parts = []
        if self.method == "sink_recent_random":
            sink = min(self.cfg.eviction.sink_size, budget)
            recent = min(self.cfg.eviction.recent_size, max(0, budget - sink))
            if sink > 0:
                reserved_parts.extend(range(sink))
            if recent > 0:
                reserved_parts.extend(range(seq_len - recent, seq_len))
        reserved = sorted(set(x for x in reserved_parts if 0 <= x < seq_len))
        remaining = max(0, budget - len(reserved))
        candidates = [i for i in range(seq_len) if i not in set(reserved)]
        if remaining > 0 and candidates:
            chosen = rng.choice(candidates, size=min(remaining, len(candidates)), replace=False)
            reserved.extend(int(x) for x in chosen.tolist())
        return mx.array(sorted(set(reserved))[:budget], dtype=mx.int32)

    def _select_oracle_indices(self, seq_len: int, budget: int, layer_idx: int, scores: Any):
        import mlx.core as mx

        if seq_len <= budget:
            return mx.arange(seq_len)
        pos_map = self.position_maps.get(layer_idx)
        current = []
        oracle_set = set(self.oracle_positions)
        if pos_map is not None and oracle_set:
            current = [
                i
                for i, original_pos in enumerate(pos_map.tolist())
                if int(original_pos) in oracle_set
            ]
        sink = min(self.cfg.eviction.sink_size, budget)
        recent = min(self.cfg.eviction.recent_size, max(0, budget - sink))
        reserved = list(range(sink))
        if recent > 0:
            reserved.extend(range(seq_len - recent, seq_len))
        keep = mx.array(sorted(set(reserved + current)), dtype=mx.int32)
        return self._ensure_keep_budget(keep, seq_len, budget, scores)

    def _direct_policy_layer_score(self, layer_idx: int, seq_len: int):
        """Return the frozen per-layer signal for a shared direct policy."""

        import mlx.core as mx

        if self.method == "latest_attention_shared":
            latest = self.attention_state.get("last", {}).get(layer_idx)
            if latest is None:
                raise ScoreUnavailableError(
                    f"latest attention is unavailable for layer={layer_idx}"
                )
            if int(latest.shape[0]) < int(seq_len):
                latest = mx.concatenate(
                    [
                        latest,
                        mx.zeros(
                            (int(seq_len) - int(latest.shape[0]),),
                            dtype=latest.dtype,
                        ),
                    ],
                    axis=0,
                )
            return latest[:seq_len].astype(mx.float32)

        if self.method != "temporal_volatility_shared":
            raise ValueError(f"not a shared direct policy: {self.method}")
        window = int(getattr(self.cfg.eviction, "attention_window", 4))
        observed = self.attention_state.get("observe", {}).get(layer_idx, [])
        if len(observed) < window:
            raise ScoreUnavailableError(
                f"temporal volatility needs {window} queries at layer={layer_idx}; "
                f"found={len(observed)}"
            )
        aligned = []
        for vector in observed[-window:]:
            current = vector[:seq_len]
            if int(current.shape[0]) < int(seq_len):
                current = mx.concatenate(
                    [
                        current,
                        mx.zeros(
                            (int(seq_len) - int(current.shape[0]),),
                            dtype=current.dtype,
                        ),
                    ],
                    axis=0,
                )
            aligned.append(current.astype(mx.float32))
        return mx.std(mx.stack(aligned, axis=0), axis=0).astype(mx.float32)

    def _shared_direct_score(self, seq_len: int):
        """Build the one-dimensional score used by a shared-set policy."""

        import mlx.core as mx

        if self.method in SHARED_DIRECT_STATIC_METHODS:
            if not self.stream_token_ids:
                raise ScoreUnavailableError(
                    f"{self.method} requires the observed stream token ids"
                )
            positions = self._to_int_list(self.position_maps[0])
            if len(positions) != int(seq_len):
                raise RuntimeError("static shared score position map is misaligned")
            if self.method == "position_coverage_shared":
                return mx.zeros((int(seq_len),), dtype=mx.float32)
            if self.method not in {"token_rarity_shared", "query_overlap_shared"}:
                raise ValueError(f"not a static shared policy: {self.method}")
            total = max(1, len(self.stream_token_ids))
            if self.method == "query_overlap_shared":
                base_scores = []
                for token in self.stream_token_ids:
                    query_count = int(self.query_signature_counts[int(token)])
                    stream_count = max(1, int(self.stream_token_counts[int(token)]))
                    base_scores.append(
                        (1.0 / float(stream_count)) if query_count > 0 else 0.0
                    )
                values = []
                for position in positions:
                    if position < 0 or position >= len(base_scores):
                        raise ScoreUnavailableError(
                            f"token id unavailable at logical position={position}"
                        )
                    start = max(0, position - 2)
                    end = min(len(base_scores), position + 3)
                    values.append(max(base_scores[start:end], default=0.0))
                return mx.array(np.asarray(values, dtype=np.float32))
            values = []
            for position in positions:
                if position < 0 or position >= len(self.stream_token_ids):
                    raise ScoreUnavailableError(
                        f"token id unavailable at logical position={position}"
                    )
                start = max(0, position - 2)
                end = min(len(self.stream_token_ids), position + 3)
                local = []
                for neighbor in self.stream_token_ids[start:end]:
                    count = max(1, int(self.stream_token_counts[int(neighbor)]))
                    local.append(math.log((total + 1.0) / (count + 1.0)))
                values.append(float(sum(local) / max(1, len(local))))
            return mx.array(np.asarray(values, dtype=np.float32))

        configured = getattr(self.cfg.eviction, "direct_policy_layers", None)
        layers = configured or [0, 7, 14, 15, 21, 27]
        layers = sorted({int(layer) for layer in layers})
        invalid = [layer for layer in layers if layer < 0 or layer >= self.num_layers]
        if invalid or not layers:
            raise ValueError(f"invalid direct-policy diagnostic layers: {invalid}")
        sink = min(int(self.cfg.eviction.sink_size), int(seq_len))
        recent = min(
            int(self.cfg.eviction.recent_size), max(0, int(seq_len) - sink)
        )
        eligible_end = int(seq_len) - recent
        if eligible_end <= sink:
            raise ScoreUnavailableError("shared direct policy has no eligible core")
        normalized = []
        for layer in layers:
            score = mx.maximum(
                self._direct_policy_layer_score(layer, seq_len), 0.0
            )
            denominator = float(mx.sum(score[sink:eligible_end]).item())
            if not math.isfinite(denominator) or denominator <= 0.0:
                raise ScoreUnavailableError(
                    f"non-positive eligible signal at layer={layer}"
                )
            current = mx.zeros((int(seq_len),), dtype=mx.float32)
            current[sink:eligible_end] = score[sink:eligible_end] / denominator
            normalized.append(current)
        return mx.mean(mx.stack(normalized, axis=0), axis=0).astype(mx.float32)

    def _select_shared_direct_indices(self, scores: Any, seq_len: int, budget: int):
        """Apply the frozen sink/recent/core partition to one shared score."""

        import mlx.core as mx

        target = min(int(seq_len), int(budget))
        if int(seq_len) <= target:
            return mx.arange(int(seq_len), dtype=mx.int32)
        sink = min(int(self.cfg.eviction.sink_size), target)
        recent = min(
            int(self.cfg.eviction.recent_size), max(0, target - sink)
        )
        core = max(0, target - sink - recent)
        eligible_end = int(seq_len) - recent
        parts = []
        if sink:
            parts.append(mx.arange(sink, dtype=mx.int32))
        if core:
            take = min(core, max(0, eligible_end - sink))
            if take and self.method == "position_coverage_shared":
                logical_positions = self._to_int_list(self.position_maps[0])
                candidate_indices = list(range(sink, eligible_end))
                if take >= len(candidate_indices):
                    chosen_values = candidate_indices
                else:
                    candidate_positions = {
                        index: logical_positions[index]
                        for index in candidate_indices
                    }
                    ideals = np.linspace(
                        min(candidate_positions.values()),
                        max(candidate_positions.values()),
                        num=take,
                    )
                    available = set(candidate_indices)
                    chosen_values = []
                    for ideal in ideals:
                        chosen = min(
                            available,
                            key=lambda index: (
                                abs(candidate_positions[index] - float(ideal)),
                                candidate_positions[index],
                            ),
                        )
                        chosen_values.append(chosen)
                        available.remove(chosen)
                parts.append(mx.array(chosen_values, dtype=mx.int32))
            elif take:
                candidates = scores[sink:eligible_end]
                chosen = mx.argpartition(-candidates, max(0, take - 1))[:take]
                parts.append((chosen + sink).astype(mx.int32))
        if recent:
            parts.append(mx.arange(eligible_end, int(seq_len), dtype=mx.int32))
        keep = mx.sort(mx.concatenate(parts))
        if int(keep.shape[0]) != target:
            raise RuntimeError(
                f"shared direct selection size={keep.shape[0]} expected={target}"
            )
        return keep

    def _evict_shared_direct(self, cache: List[Any], budget: int) -> None:
        """Prune every layer with one shared retained set."""

        import mlx.core as mx

        lengths = [int(current.offset) for current in cache]
        if not lengths or len(set(lengths)) != 1:
            raise RuntimeError("shared direct policy requires aligned layer lengths")
        seq_len = lengths[0]
        if seq_len <= int(budget):
            for layer_idx in range(len(cache)):
                self.last_selected[layer_idx] = self._to_int_list(
                    self.position_maps[layer_idx]
                )
            return
        reference_positions = self._to_int_list(self.position_maps[0])
        for layer_idx in range(1, len(cache)):
            if self._to_int_list(self.position_maps[layer_idx]) != reference_positions:
                raise RuntimeError("shared direct policy position maps diverged")

        score_start = time.perf_counter()
        scores = self._shared_direct_score(seq_len)
        self.profile_times["score_time_s"] += time.perf_counter() - score_start
        self.score_update_count += 1
        phase = self.phase if self.phase in self.score_phase_counts else "decode"
        self.score_phase_counts[phase] += 1

        topk_start = time.perf_counter()
        keep = self._select_shared_direct_indices(scores, seq_len, int(budget))
        self.profile_times["topk_time_s"] += time.perf_counter() - topk_start
        selected_positions = mx.take(self.position_maps[0], keep, axis=0)
        score_values = self._to_float_list(scores)
        selected_values = self._to_int_list(selected_positions)

        rebuild_start = time.perf_counter()
        decision_units = []
        for layer_idx, current in enumerate(cache):
            universe = self._to_int_list(self.position_maps[layer_idx])
            current.keys = mx.take(
                current.keys[:, :, :seq_len, :], keep, axis=2
            )
            current.values = mx.take(
                current.values[:, :, :seq_len, :], keep, axis=2
            )
            current.offset = int(keep.shape[0])
            self._prune_attention_state(layer_idx, keep, seq_len)
            self.position_maps[layer_idx] = selected_positions
            self.last_selected[layer_idx] = selected_values
            self.last_scores[layer_idx] = score_values
            self._current_scores_by_head.pop(layer_idx, None)
            decision_units.append(
                {
                    "layer": layer_idx,
                    "head": None,
                    "universe_positions": universe,
                    "score_positions": universe,
                    "scores": score_values,
                    "selected_positions": selected_values,
                    "requested_budget": int(budget),
                }
            )
        self.profile_times["cache_rebuild_time_s"] += (
            time.perf_counter() - rebuild_start
        )
        self.eviction_count += 1
        if self.phase == "prefill":
            self.prefill_decision = {
                "phase": "pre_answer",
                "budget_scope": "total_kv",
                "budget_unit": "token_slots_per_kv_head",
                "requested_budget": int(budget),
                "units": decision_units,
            }
        self.eviction_step += 1

    def _attention_scores(self, layer_idx: int, seq_len: int, mode: str):
        import mlx.core as mx

        if mode == "snapkv":
            window = max(1, int(getattr(self.cfg.eviction, "observation_window", 32)))
            observed = self.attention_state.get("observe", {}).get(layer_idx, [])
            usable = [vec[:seq_len] for vec in observed[-window:] if int(vec.shape[0]) >= seq_len]
            if usable:
                scores = mx.sum(mx.stack(usable, axis=0), axis=0).astype(mx.float32)
                observed_heads = self.attention_state.get("observe_heads", {}).get(layer_idx, [])
                head_usable = [
                    vec[:, :seq_len]
                    for vec in observed_heads[-window:]
                    if int(vec.shape[-1]) >= seq_len
                ]
                if head_usable:
                    self._current_scores_by_head[layer_idx] = mx.sum(
                        mx.stack(head_usable, axis=0), axis=0
                    ).astype(mx.float32)
                    self._record_head_scores(layer_idx, self._current_scores_by_head[layer_idx])
                return self._pool_scores_1d(scores)
        if mode == "windowed":
            window = max(1, int(getattr(self.cfg.eviction, "attention_window", 32)))
            observed = self.attention_state.get("observe", {}).get(layer_idx, [])
            usable = [vec[:seq_len] for vec in observed[-window:] if int(vec.shape[0]) >= seq_len]
            if usable:
                observed_heads = self.attention_state.get("observe_heads", {}).get(layer_idx, [])
                head_usable = [
                    vec[:, :seq_len]
                    for vec in observed_heads[-window:]
                    if int(vec.shape[-1]) >= seq_len
                ]
                if head_usable:
                    self._current_scores_by_head[layer_idx] = mx.sum(
                        mx.stack(head_usable, axis=0), axis=0
                    ).astype(mx.float32)
                    self._record_head_scores(layer_idx, self._current_scores_by_head[layer_idx])
                return mx.sum(mx.stack(usable, axis=0), axis=0).astype(mx.float32)
        if mode == "decayed":
            decayed = self.attention_state.get("decayed", {}).get(layer_idx)
            if decayed is not None and int(decayed.shape[0]) >= seq_len:
                heads = self.attention_state.get("decayed_heads", {}).get(layer_idx)
                if heads is not None and int(heads.shape[-1]) >= seq_len:
                    self._current_scores_by_head[layer_idx] = heads[:, :seq_len].astype(mx.float32)
                    self._record_head_scores(layer_idx, self._current_scores_by_head[layer_idx])
                return decayed[:seq_len].astype(mx.float32)
        if mode == "last_query":
            latest = self.attention_state.get("last", {}).get(layer_idx)
            if latest is not None and int(latest.shape[0]) >= seq_len:
                heads = self.attention_state.get("last_heads", {}).get(layer_idx)
                if heads is not None and int(heads.shape[-1]) >= seq_len:
                    self._current_scores_by_head[layer_idx] = heads[:, :seq_len].astype(mx.float32)
                    self._record_head_scores(layer_idx, self._current_scores_by_head[layer_idx])
                return latest[:seq_len].astype(mx.float32)
        if mode == "accumulated":
            accumulated = self.attention_state.get("accumulated", {}).get(layer_idx)
            if accumulated is not None and int(accumulated.shape[0]) >= seq_len:
                heads = self.attention_state.get("accumulated_heads", {}).get(layer_idx)
                if heads is not None and int(heads.shape[-1]) >= seq_len:
                    self._current_scores_by_head[layer_idx] = heads[:, :seq_len].astype(mx.float32)
                    self._record_head_scores(layer_idx, self._current_scores_by_head[layer_idx])
                return accumulated[:seq_len].astype(mx.float32)
        errors = self.attention_state.get("hook_error_events", [])
        detail = errors[-1]["reason"] if errors else "no attention rows were recorded"
        raise ScoreUnavailableError(
            f"attention score unavailable for layer={layer_idx}, mode={mode}, "
            f"seq_len={seq_len}: {detail}"
        )

    def _pool_scores_1d(self, scores: Any):
        import mlx.core as mx

        kernel = int(
            getattr(
                self.cfg.eviction,
                "pooling_kernel",
                getattr(self.cfg.eviction, "kernel_size", 1),
            )
            or 1
        )
        if kernel <= 1 or int(scores.shape[0]) <= 1:
            return scores.astype(mx.float32)
        method = str(getattr(self.cfg.eviction, "pooling_method", "max") or "max").lower()
        pooled = snapkv_pool_scores_numpy(
            scores.tolist(), kernel, method
        )
        return mx.array(pooled)

    def _geom_scores(self, rows: Any, layer_idx: int):
        import mlx.core as mx

        if self.method in ("attention_l2", "attn_l2", "attention_l2_compactor"):
            return self._estimator_scores(
                self._get_estimator("l2", layer_idx), rows, layer_idx, "value_l2_leverage"
            )
        if self.method == "attention_norm":
            return mx.sqrt(mx.sum(rows * rows, axis=1))
        if self.method in ("attention_recency", "attention_sink_recency"):
            return mx.arange(int(rows.shape[0])).astype(mx.float32)
        return self._estimator_scores(
            self._get_estimator("l1", layer_idx), rows, layer_idx, "value_l1_leverage"
        )

    def _select_snapkv_indices(self, scores: Any, seq_len: int, budget: int):
        import mlx.core as mx

        if seq_len <= budget:
            return mx.arange(seq_len)
        sink = min(self.cfg.eviction.sink_size, max(0, budget - 1))
        obs = min(max(1, int(getattr(self.cfg.eviction, "window_size", 32))), seq_len)
        hist_len = max(0, seq_len - obs)
        parts = []
        if sink > 0:
            parts.append(mx.arange(sink))
        if hist_len > sink and scores is not None:
            hist_budget = max(0, budget - sink - obs)
            if hist_budget > 0:
                cand = scores[sink:hist_len]
                take = min(hist_budget, int(cand.shape[0]))
                if take >= int(cand.shape[0]):
                    idx = mx.arange(int(cand.shape[0]))
                else:
                    idx = mx.argpartition(-cand, max(0, take - 1))[:take]
                parts.append(idx + sink)
        parts.append(mx.arange(max(0, seq_len - obs), seq_len))
        keep = self._unique_sorted_indices(mx.concatenate(parts)) if parts else mx.arange(seq_len)
        if keep.shape[0] > budget:
            recent = mx.arange(max(0, seq_len - budget), seq_len)
            keep = mx.sort(recent)
        return keep

    def _select_hybrid_indices(self, seq_len: int, budget: int, layer_idx: int, scores: Any):
        import mlx.core as mx

        sink_b = min(seq_len, max(0, int(budget * float(self.cfg.eviction.sink_budget_ratio))))
        configured_sink = min(seq_len, max(0, int(self.cfg.eviction.sink_size)))
        sink_b = min(seq_len, max(sink_b, configured_sink))
        recent_b = min(
            seq_len - sink_b,
            max(0, int(budget * float(self.cfg.eviction.recent_budget_ratio))),
        )
        if sink_b == 0 and self.cfg.eviction.sink_size > 0:
            sink_b = min(self.cfg.eviction.sink_size, max(0, budget - 1))
        if recent_b == 0 and self.cfg.eviction.recent_size > 0:
            recent_b = min(self.cfg.eviction.recent_size, max(0, budget - sink_b - 1))

        parts = []
        source_map: Dict[int, List[str]] = {}
        if sink_b > 0:
            sink_idx = mx.arange(sink_b)
            parts.append(sink_idx)
            for idx in sink_idx.tolist():
                source_map.setdefault(int(idx), []).append("sink")
        if recent_b > 0:
            recent_idx = mx.arange(max(0, seq_len - recent_b), seq_len)
            parts.append(recent_idx)
            for idx in recent_idx.tolist():
                source_map.setdefault(int(idx), []).append("recent")
        selected = self._unique_sorted_indices(mx.concatenate(parts)) if parts else mx.array([], dtype=mx.int32)

        def take_top(score_vec: Any, take: int, current: Any, source_name: str):
            if score_vec is None or take <= 0 or int(current.shape[0]) >= budget:
                return current
            vec = score_vec[:seq_len].astype(mx.float32)
            mask_np = np.ones(seq_len, dtype=bool)
            if int(current.shape[0]) > 0:
                mask_np[[int(x) for x in current.tolist() if 0 <= int(x) < seq_len]] = False
            mask = mx.array(mask_np)
            masked = mx.where(mask, vec, -mx.inf)
            valid_count = int(mx.sum(mask).item())
            if valid_count <= 0:
                return current
            take_n = min(int(take), valid_count, max(0, budget - int(current.shape[0])))
            if take_n <= 0:
                return current
            idx = mx.argpartition(-masked, max(0, take_n - 1))[:take_n]
            for token_idx in idx.tolist():
                source_map.setdefault(int(token_idx), []).append(source_name)
            return self._unique_sorted_indices(mx.concatenate([current, idx]))

        remaining = max(0, budget - int(selected.shape[0]))
        attn_take = min(int(budget * float(self.cfg.eviction.attn_budget_ratio)), remaining)
        selected = take_top(self._last_attn_scores.get(layer_idx), attn_take, selected, "attention")

        remaining = max(0, budget - int(selected.shape[0]))
        geom_ratio = float(getattr(self.cfg.eviction, "l1_budget_ratio", 0.3))
        if self.method in ("attention+l2", "attention_l2", "attn_l2"):
            geom_ratio = float(getattr(self.cfg.eviction, "l1_budget_ratio", 0.3))
        geom_take = min(int(budget * geom_ratio), remaining)
        selected = take_top(self._last_geom_scores.get(layer_idx), geom_take, selected, "geometry")

        if int(selected.shape[0]) < budget:
            selected = take_top(scores, budget - int(selected.shape[0]), selected, "combined")
        if int(selected.shape[0]) < budget:
            fill = mx.arange(max(0, seq_len - budget), seq_len)
            selected = self._unique_sorted_indices(mx.concatenate([selected, fill]))
        keep = mx.sort(selected)
        if int(keep.shape[0]) > budget:
            keep = self._trim_hybrid_keep(keep, budget, source_map, scores)
        for idx in keep.tolist():
            source_map.setdefault(int(idx), ["fill"])
        self._component_sources_current[layer_idx] = source_map
        return keep

    def _trim_hybrid_keep(self, keep: Any, budget: int, source_map: Dict[int, List[str]], scores: Any = None):
        import mlx.core as mx

        values = sorted({int(x) for x in keep.tolist()})
        chosen: List[int] = []

        def add(token_idx: int) -> None:
            if token_idx not in chosen and len(chosen) < budget:
                chosen.append(token_idx)

        sink_values = [x for x in values if "sink" in source_map.get(x, [])]
        recent_values = [x for x in values if "recent" in source_map.get(x, [])]
        for idx in sorted(sink_values):
            add(idx)
        for idx in sorted(recent_values, reverse=True):
            add(idx)

        remaining = [x for x in values if x not in set(chosen)]
        if scores is not None:
            score_vals = scores.tolist()
            remaining.sort(
                key=lambda x: float(score_vals[x]) if 0 <= x < len(score_vals) else float("-inf"),
                reverse=True,
            )
        else:
            remaining.sort(reverse=True)
        for idx in remaining:
            add(idx)
        return mx.array(sorted(chosen[:budget]), dtype=mx.int32)

    @staticmethod
    def _unique_sorted_indices(indices: Any):
        import mlx.core as mx

        values = sorted({int(x) for x in indices.tolist()})
        return mx.array(values, dtype=mx.int32)

    def _ensure_keep_budget(self, keep: Any, seq_len: int, budget: int, scores: Any = None):
        import mlx.core as mx

        values = sorted({int(x) for x in keep.tolist() if 0 <= int(x) < seq_len})
        target = min(seq_len, int(budget))
        if len(values) < target:
            selected = set(values)
            if scores is not None:
                vec = scores[:seq_len].astype(mx.float32)
                candidates = [i for i in range(seq_len) if i not in selected]
                if candidates:
                    cand_scores = [(float(vec[i].item()), i) for i in candidates]
                    cand_scores.sort(reverse=True)
                    values.extend(i for _, i in cand_scores[: target - len(values)])
            if len(values) < target:
                for i in range(max(0, seq_len - target), seq_len):
                    if i not in set(values):
                        values.append(i)
                    if len(set(values)) >= target:
                        break
        if len(values) > target:
            values = sorted(values)[-target:]
        return mx.array(sorted(set(values))[:target], dtype=mx.int32)

    def _prune_attention_state(self, layer_idx: int, keep: Any, seq_len: int) -> None:
        import mlx.core as mx

        for key in ("last", "accumulated", "decayed"):
            vec = self.attention_state.get(key, {}).get(layer_idx)
            if vec is not None and int(vec.shape[0]) >= seq_len:
                self.attention_state[key][layer_idx] = mx.take(vec[:seq_len], keep, axis=0)
        for key in ("last_heads", "accumulated_heads", "decayed_heads"):
            vec = self.attention_state.get(key, {}).get(layer_idx)
            if vec is not None and int(vec.shape[-1]) >= seq_len:
                self.attention_state[key][layer_idx] = mx.take(
                    vec[:, :seq_len], keep, axis=1
                )
        static_scores = self._static_score_cache.get(layer_idx)
        if static_scores is not None and int(static_scores.shape[0]) >= seq_len:
            self._static_score_cache[layer_idx] = mx.take(static_scores[:seq_len], keep, axis=0)
        head_scores = self._current_scores_by_head.get(layer_idx)
        if head_scores is not None and int(head_scores.shape[-1]) >= seq_len:
            pruned = mx.take(head_scores[:, :seq_len], keep, axis=1)
            self._current_scores_by_head[layer_idx] = pruned
            self._record_head_scores(layer_idx, pruned)
        observe = self.attention_state.get("observe", {}).get(layer_idx)
        if observe:
            aligned = []
            for vec in observe:
                current = vec[:seq_len]
                if int(current.shape[0]) < seq_len:
                    current = mx.concatenate(
                        [
                            current,
                            mx.zeros(
                                (seq_len - int(current.shape[0]),),
                                dtype=current.dtype,
                            ),
                        ],
                        axis=0,
                    )
                aligned.append(mx.take(current, keep, axis=0))
            self.attention_state["observe"][layer_idx] = aligned
        observe_heads = self.attention_state.get("observe_heads", {}).get(layer_idx)
        if observe_heads:
            aligned_heads = []
            for vec in observe_heads:
                current = vec[:, :seq_len]
                if int(current.shape[-1]) < seq_len:
                    current = mx.concatenate(
                        [
                            current,
                            mx.zeros(
                                (
                                    int(current.shape[0]),
                                    seq_len - int(current.shape[-1]),
                                ),
                                dtype=current.dtype,
                            ),
                        ],
                        axis=1,
                    )
                aligned_heads.append(mx.take(current, keep, axis=1))
            self.attention_state["observe_heads"][layer_idx] = aligned_heads

    @staticmethod
    def _to_int_list(arr: Any) -> List[int]:
        return [int(x) for x in arr.tolist()]

    @staticmethod
    def _to_float_list(arr: Any) -> List[float]:
        return [float(x) for x in arr.astype(arr.dtype).tolist()]


class MLXRunner(BaseRunner):
    """Formal MLX-LM backend for KV cache eviction experiments."""

    backend_name = "mlx"

    def __init__(self, cfg: ExperimentConfig):
        super().__init__(cfg)
        self.model = None
        self.tokenizer = None
        self.hf_tokenizer = None
        self.model_info: Dict[str, Any] = {}
        self.attention_state: Dict[str, Any] = {
            "last": {},
            "accumulated": {},
            "decayed": {},
            "observe": {},
            "observe_heads": {},
            "prefill_q_post": {},
            "prefill_k_post": {},
            "prefill_k_pre": {},
            "hook_errors": 0,
            "max_observe": 32,
            "decay_gamma": 0.95,
            "enabled": False,
            "phase": "idle",
            "current_method": None,
        }

    def run(
        self,
        methods: List[str],
        budgets: List[int],
        budget_ratios: Optional[List[float]] = None,
        skip_analysis: bool = False,
    ) -> Path:
        self.load_model()
        _, samples = load_benchmark(self.cfg, self.hf_tokenizer)
        out_dir = self.make_run_dir()
        self.save_run_metadata(out_dir, self.model_info, samples=samples)

        results: List[Dict[str, Any]] = []
        for budget in budgets:
            for method in methods:
                for sample_idx, sample in enumerate(samples):
                    actual_budget = self._actual_budget(sample, budget, budget_ratios or [])
                    try:
                        row = self.run_one(sample, sample_idx, method, actual_budget, out_dir)
                    except Exception as exc:
                        row = self.error_result(sample, sample_idx, method, actual_budget, exc)
                    results.append(row)
                    save_results(
                        row,
                        out_dir
                        / "samples"
                        / f"{method}_b{actual_budget}_s{sample_idx}.json",
                    )

        self.save_result_bundle(results, out_dir)
        if not skip_analysis:
            try:
                from scripts.run_analysis import run_analysis

                run_analysis(results, self.cfg, out_dir)
                from scripts.plot_results import (
                    plot_accuracy_by_budget,
                    plot_metric_by_method_budget,
                    plot_latency_by_method_budget,
                    plot_method_budget_heatmap,
                    plot_model_method_heatmap,
                    plot_evidence_recall_by_depth,
                    plot_latency,
                    plot_overlap,
                    plot_rank,
                    plot_selected_positions,
                    write_case_study_markdown,
                )

                fig_dir = out_dir / "figures"
                figure_outputs = {
                    "accuracy_by_budget": plot_accuracy_by_budget(out_dir, fig_dir),
                    "accuracy_by_method_budget": plot_metric_by_method_budget(
                        out_dir, fig_dir, "accuracy", "Accuracy by Cache Budget", "Accuracy", "accuracy_by_method_budget"
                    ),
                    "evidence_recall_by_method_budget": (
                        plot_metric_by_method_budget(
                            out_dir, fig_dir, "avg_evidence_recall", "Evidence Recall by Cache Budget", "Evidence Recall", "evidence_recall_by_method_budget"
                        )
                        if self.cfg.analysis.evidence_recall
                        else ""
                    ),
                    "official_score_by_method_budget": plot_metric_by_method_budget(
                        out_dir, fig_dir, "avg_official_score", "Official Score by Cache Budget", "Official Score", "official_score_by_method_budget"
                    ),
                    "latency_by_method_budget": plot_latency_by_method_budget(out_dir, fig_dir),
                    "method_budget_heatmap": plot_method_budget_heatmap(
                        out_dir, fig_dir, "accuracy", "Method x Budget Accuracy", "method_budget_heatmap"
                    ),
                    "official_score_heatmap": plot_method_budget_heatmap(
                        out_dir, fig_dir, "avg_official_score", "Method x Budget Official Score", "official_score_heatmap"
                    ),
                    "evidence_recall_heatmap": (
                        plot_evidence_recall_by_depth(out_dir, fig_dir)
                        if self.cfg.analysis.evidence_recall
                        else ""
                    ),
                    "method_overlap_heatmap": plot_overlap(out_dir, fig_dir),
                    "rank_correlation_heatmap": plot_rank(out_dir, fig_dir),
                    "latency_breakdown": plot_latency(out_dir, fig_dir),
                    "selected_token_position_distribution": plot_selected_positions(out_dir, fig_dir),
                    "model_method_accuracy_heatmap": plot_model_method_heatmap(
                        out_dir, fig_dir, "accuracy", "Model x Method Accuracy", "model_method_accuracy_heatmap"
                    ),
                    "model_method_official_score_heatmap": plot_model_method_heatmap(
                        out_dir, fig_dir, "avg_official_score", "Model x Method Official Score", "model_method_official_score_heatmap"
                    ),
                    "model_method_evidence_recall_heatmap": (
                        plot_model_method_heatmap(
                            out_dir, fig_dir, "avg_evidence_recall", "Model x Method Evidence Recall", "model_method_evidence_recall_heatmap"
                        )
                        if self.cfg.analysis.evidence_recall
                        else ""
                    ),
                    "case_study_markdown": (
                        write_case_study_markdown(out_dir)
                        if self.cfg.analysis.case_study
                        else ""
                    ),
                }
                save_results(figure_outputs, fig_dir / "figures_summary.json")
            except Exception as exc:
                save_results({"error": str(exc)}, out_dir / "analysis" / "analysis_error.json")
        return out_dir

    def load_model(self) -> None:
        import mlx.core as mx
        import mlx.nn as nn
        from mlx_lm import load

        cfg = self.cfg.model
        mx.random.seed(self.cfg.seed)
        t0 = time.perf_counter()
        loaded = load(
            cfg.name,
            return_config=True,
            revision=cfg.revision,
            tokenizer_config={"trust_remote_code": cfg.trust_remote_code},
        )
        self.model, self.tokenizer, raw_config = loaded
        load_time = time.perf_counter() - t0
        quant_time = 0.0
        quantized_layers = 0
        if cfg.quant_bits and cfg.mlx_weight_quantize:
            t0 = time.perf_counter()
            nn.quantize(
                self.model,
                group_size=int(cfg.quant_group_size),
                bits=int(cfg.quant_bits),
            )
            mx.eval(self.model.parameters())
            quant_time = time.perf_counter() - t0
            try:
                quantized_layers = sum(
                    1
                    for _, module in self.model.named_modules()
                    if "Quantized" in type(module).__name__
                )
            except Exception:
                quantized_layers = 0
        self.hf_tokenizer = getattr(self.tokenizer, "_tokenizer", self.tokenizer)
        self.model_info = {
            "model_name": cfg.name,
            "backend": "mlx",
            "quant_bits": cfg.quant_bits,
            "quant_group_size": cfg.quant_group_size,
            "mlx_weight_quantize": cfg.mlx_weight_quantize,
            "load_time_s": load_time,
            "quantize_time_s": quant_time,
            "quantized_layers": quantized_layers,
            "num_layers": len(self.model.model.layers),
            "model_type": raw_config.get("model_type"),
            "vocab_size": raw_config.get("vocab_size"),
            "hidden_size": raw_config.get("hidden_size"),
            "num_attention_heads": raw_config.get("num_attention_heads"),
            "num_key_value_heads": raw_config.get("num_key_value_heads"),
            "max_position_embeddings": raw_config.get("max_position_embeddings"),
            "tokenizer_class": type(self.hf_tokenizer).__name__,
        }
        self.install_attention_hooks()
        self.model_info["attention_hook_installed"] = bool(
            self.attention_state.get("hook_installed", False)
        )
        self.model_info["attention_hook_adapter"] = self.attention_state.get("hook_adapter")
        self.model_info["attention_hook_classes"] = self.attention_state.get("hook_classes", [])
        self.model_info["attention_hooked_layers"] = self.attention_state.get("hooked_layers", 0)
        self.model_info["attention_hook_expected_layers"] = self.attention_state.get("expected_hook_layers", 0)
        adapter = build_model_adapter(
            cfg,
            raw_config=raw_config,
            tokenizer=self.hf_tokenizer,
            cache_format="mlx_kv",
            attention_hook_installed=self.model_info["attention_hook_installed"],
        )
        self.model_info.update(adapter.to_dict())

    def reset_attention_state(self) -> None:
        self.attention_state["last"] = {}
        self.attention_state["last_heads"] = {}
        self.attention_state["accumulated"] = {}
        self.attention_state["accumulated_heads"] = {}
        self.attention_state["decayed"] = {}
        self.attention_state["decayed_heads"] = {}
        self.attention_state["observe"] = {}
        self.attention_state["observe_heads"] = {}
        self.attention_state["observe_query_heads"] = {}
        self.attention_state["prefill_q_post"] = {}
        self.attention_state["prefill_k_post"] = {}
        self.attention_state["prefill_k_pre"] = {}
        self.attention_state["hook_errors"] = 0
        self.attention_state["hook_error_events"] = []
        self.attention_state["query_counts"] = {}
        self.attention_state["temporal_queries"] = {}
        self.attention_state["temporal_queries_post_rope"] = {}
        self.attention_state["temporal_attention_outputs"] = {}
        self.attention_state["temporal_attention_distributions"] = {}
        self.attention_state["temporal_new_values"] = {}
        self.attention_state["temporal_new_keys"] = {}
        self.attention_state["temporal_attention_inputs"] = {}
        self.attention_state["temporal_residual_inputs"] = {}
        self.attention_state["temporal_post_attention_residuals"] = {}
        self.attention_state["temporal_layer_outputs"] = {}
        self.attention_state["temporal_attention_outputs_all_heads"] = {}
        self.attention_state[
            "temporal_attention_distributions_all_heads"
        ] = {}
        self.attention_state["direct_policy_attention"] = {}
        self.attention_state["temporal_projected_attention_outputs"] = {}
        self.attention_state["temporal_projected_injections"] = {}
        self.attention_state["temporal_query_overrides"] = {}
        self.attention_state["temporal_new_key_overrides"] = {}
        self.attention_state["temporal_new_value_overrides"] = {}
        self.attention_state["temporal_attention_input_overrides"] = {}
        self.attention_state["temporal_layer_input_overrides"] = {}
        self.attention_state["max_observe"] = max(
            1,
            int(getattr(self.cfg.eviction, "observation_window", 32)),
            int(getattr(self.cfg.eviction, "attention_window", 32)),
        )
        self.attention_state["attention_chunk_size"] = max(
            1, int(getattr(self.cfg.eviction, "compactor_attention_chunk_size", 128))
        )
        self.attention_state["record_all_queries"] = False
        self.attention_state["temporal_record_query_head_window"] = False
        self.attention_state["temporal_record_direct_policy"] = False
        self.attention_state["decay_gamma"] = float(getattr(self.cfg.eviction, "decay_gamma", 0.95))
        self.attention_state["enabled"] = False
        self.attention_state["phase"] = "idle"
        self.attention_state["current_method"] = None

    def configure_attention_recording(self, method: str) -> None:
        method = canonical_method(method)
        self.attention_state["record_all_queries"] = bool(
            method in {"attention", "attention_decay", "h2o", "vatp"}
            or method in HYBRID_METHODS
            or method in {"compactor"}
        )

    def install_attention_hooks(self) -> None:
        """Install a minimal Qwen-style MLX attention hook for runtime scores."""
        try:
            from mlx_lm.models.base import scaled_dot_product_attention
        except Exception as exc:
            self.attention_state["hook_error_message"] = str(exc)
            self.attention_state["hook_installed"] = False
            return

        installed = 0
        hook_classes = []
        required_attributes = {
            "q_proj", "k_proj", "v_proj", "o_proj", "rope",
            "n_heads", "n_kv_heads", "scale",
        }
        for layer_idx, layer in enumerate(getattr(self.model.model, "layers", [])):
            attn = getattr(layer, "self_attn", None)
            if attn is None:
                continue
            missing = sorted(name for name in required_attributes if not hasattr(attn, name))
            if missing:
                self.attention_state.setdefault("hook_error_events", []).append(
                    {
                        "layer": layer_idx,
                        "phase": "install",
                        "method": None,
                        "reason": f"unsupported attention class {type(attn).__name__}; missing={missing}",
                    }
                )
                continue
            attn._l1kv_layer_idx = layer_idx
            attn._l1kv_attention_state = self.attention_state
            layer._l1kv_layer_idx = layer_idx
            layer._l1kv_attention_state = self.attention_state
            block_cls = type(layer)
            if not getattr(block_cls, "_l1kv_temporal_block_patched", False):
                def patched_block_call(
                    self_block, x, mask=None, cache=None
                ):
                    import mlx.core as mx

                    block_state = getattr(
                        self_block, "_l1kv_attention_state", None
                    )
                    block_layer = int(
                        getattr(self_block, "_l1kv_layer_idx", -1)
                    )
                    block_input = x
                    block_override = (
                        block_state.get(
                            "temporal_layer_input_overrides", {}
                        ).get(block_layer)
                        if block_state
                        else None
                    )
                    if block_override is not None:
                        replacement = mx.array(block_override).reshape(
                            1, 1, int(x.shape[-1])
                        )
                        block_input = mx.concatenate(
                            [x[:, :-1, :], replacement], axis=1
                        )
                    record_block = bool(
                        block_state
                        and block_state.get(
                            "temporal_record_diagnostics", False
                        )
                        and block_layer
                        in set(
                            int(value)
                            for value in block_state.get(
                                "temporal_selected_layers", []
                            )
                        )
                    )
                    if record_block:
                        block_state.setdefault(
                            "temporal_residual_inputs", {}
                        )[block_layer] = block_input[0, -1, :]
                    residual = self_block.self_attn(
                        self_block.input_layernorm(block_input), mask, cache
                    )
                    hidden = block_input + residual
                    if record_block:
                        block_state.setdefault(
                            "temporal_post_attention_residuals", {}
                        )[block_layer] = hidden[0, -1, :]
                    residual = self_block.mlp(
                        self_block.post_attention_layernorm(hidden)
                    )
                    output = hidden + residual
                    if record_block:
                        block_state.setdefault(
                            "temporal_layer_outputs", {}
                        )[block_layer] = output[0, -1, :]
                    return output

                block_cls.__call__ = patched_block_call
                block_cls._l1kv_temporal_block_patched = True
            cls = type(attn)
            hook_classes.append(f"{cls.__module__}.{cls.__name__}")
            if getattr(cls, "_l1kv_patched", False):
                installed += 1
                continue

            def patched_call(self_attn, x, mask=None, cache=None, _sdpa=scaled_dot_product_attention):
                import mlx.core as mx

                B, L, _ = x.shape
                control_state = getattr(
                    self_attn, "_l1kv_attention_state", None
                )
                control_layer = int(
                    getattr(self_attn, "_l1kv_layer_idx", -1)
                )
                attention_input = x
                input_override = (
                    control_state.get(
                        "temporal_attention_input_overrides", {}
                    ).get(control_layer)
                    if control_state
                    else None
                )
                if input_override is not None:
                    replacement = mx.array(input_override).reshape(
                        1, 1, int(x.shape[-1])
                    )
                    attention_input = mx.concatenate(
                        [x[:, :-1, :], replacement], axis=1
                    )
                queries = self_attn.q_proj(attention_input)
                keys = self_attn.k_proj(attention_input)
                values = self_attn.v_proj(attention_input)

                queries = queries.reshape(B, L, self_attn.n_heads, -1)
                keys = keys.reshape(B, L, self_attn.n_kv_heads, -1)
                if hasattr(self_attn, "q_norm"):
                    queries = self_attn.q_norm(queries)
                if hasattr(self_attn, "k_norm"):
                    keys = self_attn.k_norm(keys)
                queries = queries.transpose(0, 2, 1, 3)
                keys = keys.transpose(0, 2, 1, 3)
                values = values.reshape(B, L, self_attn.n_kv_heads, -1).transpose(0, 2, 1, 3)
                query_override = (
                    control_state.get(
                        "temporal_query_overrides", {}
                    ).get(control_layer)
                    if control_state
                    else None
                )
                key_override = (
                    control_state.get(
                        "temporal_new_key_overrides", {}
                    ).get(control_layer)
                    if control_state
                    else None
                )
                value_override = (
                    control_state.get(
                        "temporal_new_value_overrides", {}
                    ).get(control_layer)
                    if control_state
                    else None
                )
                if query_override is not None:
                    replacement = mx.array(query_override).reshape(
                        1, int(queries.shape[1]), 1, int(queries.shape[-1])
                    )
                    queries = mx.concatenate(
                        [queries[:, :, :-1, :], replacement], axis=2
                    )
                if key_override is not None:
                    replacement = mx.array(key_override).reshape(
                        1, int(keys.shape[1]), 1, int(keys.shape[-1])
                    )
                    keys = mx.concatenate(
                        [keys[:, :, :-1, :], replacement], axis=2
                    )
                if value_override is not None:
                    replacement = mx.array(value_override).reshape(
                        1, int(values.shape[1]), 1, int(values.shape[-1])
                    )
                    values = mx.concatenate(
                        [values[:, :, :-1, :], replacement], axis=2
                    )
                queries_pre_rope = queries
                keys_pre_rope = keys

                if cache is not None:
                    rope_offset = int(getattr(cache, "logical_offset", cache.offset))
                    queries = self_attn.rope(queries, offset=rope_offset)
                    keys = self_attn.rope(keys, offset=rope_offset)
                    keys_post_rope = keys
                    keys, values = cache.update_and_fetch(keys, values)
                    cache.logical_offset = rope_offset + int(L)
                else:
                    queries = self_attn.rope(queries)
                    keys = self_attn.rope(keys)
                    keys_post_rope = keys

                _record_compactor_prefill_tensors(
                    self_attn,
                    queries_pre_rope,
                    keys_pre_rope,
                    queries,
                    keys_post_rope,
                )
                _record_attention_from_hook(self_attn, queries, keys, query_len=int(L))
                head_mask = _cache_head_valid_attention_mask(
                    cache,
                    int(queries.shape[1]),
                    int(keys.shape[-2]),
                )
                if head_mask is not None:
                    head_mask = head_mask.astype(queries.dtype)
                    mask = head_mask if mask is None else mask + head_mask
                temporal_state = getattr(
                    self_attn, "_l1kv_attention_state", None
                )
                temporal_layer = int(
                    getattr(self_attn, "_l1kv_layer_idx", -1)
                )
                selected_temporal_layer = bool(
                    temporal_state
                    and temporal_layer
                    in set(
                        int(value)
                        for value in temporal_state.get(
                            "temporal_selected_layers", []
                        )
                    )
                )
                record_temporal = bool(
                    selected_temporal_layer
                    and temporal_state.get("temporal_record_diagnostics", False)
                )
                record_direct_policy = bool(
                    selected_temporal_layer
                    and temporal_state.get("temporal_record_direct_policy", False)
                )
                if record_temporal or record_direct_policy:
                    selected_heads = [
                        int(value)
                        for value in temporal_state.get(
                            "temporal_selected_heads", {}
                        ).get(temporal_layer, [])
                    ]
                    repeats = int(queries.shape[1]) // int(keys.shape[1])
                    repeated_keys = mx.repeat(keys, repeats, axis=1)
                    last_logits = mx.sum(
                        queries[:, :, -1:, :].astype(mx.float32)
                        * repeated_keys.astype(mx.float32),
                        axis=-1,
                    ) * float(self_attn.scale)
                    if head_mask is not None:
                        last_logits = last_logits + head_mask.astype(mx.float32)
                    last_attention = mx.softmax(
                        last_logits, axis=-1, precise=True
                    )
                    if record_direct_policy:
                        temporal_state.setdefault("direct_policy_attention", {})[
                            temporal_layer
                        ] = last_attention[0, :, :]
                if record_temporal:
                    temporal_state.setdefault("temporal_queries", {})[
                        temporal_layer
                    ] = queries_pre_rope[0, selected_heads, -1, :]
                    temporal_state.setdefault(
                        "temporal_queries_post_rope", {}
                    )[temporal_layer] = queries[0, selected_heads, -1, :]
                    temporal_state.setdefault(
                        "temporal_attention_distributions", {}
                    )[temporal_layer] = last_attention[0, selected_heads, :]
                    temporal_state.setdefault(
                        "temporal_attention_distributions_all_heads", {}
                    )[temporal_layer] = last_attention[0, :, :]
                    temporal_state.setdefault("temporal_new_values", {})[
                        temporal_layer
                    ] = values[0, :, -1, :]
                    temporal_state.setdefault("temporal_new_keys", {})[
                        temporal_layer
                    ] = keys_pre_rope[0, :, -1, :]
                    temporal_state.setdefault(
                        "temporal_attention_inputs", {}
                    )[temporal_layer] = attention_input[0, -1, :]
                output = _sdpa(
                    queries,
                    keys,
                    values,
                    cache=cache,
                    scale=self_attn.scale,
                    mask=mask,
                )
                if record_temporal:
                    temporal_state.setdefault(
                        "temporal_attention_outputs", {}
                    )[temporal_layer] = output[0, selected_heads, -1, :]
                    temporal_state.setdefault(
                        "temporal_attention_outputs_all_heads", {}
                    )[temporal_layer] = output[0, :, -1, :]
                output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
                projected_output = self_attn.o_proj(output)
                projected_injection = (
                    temporal_state.get(
                        "temporal_projected_injections", {}
                    ).get(temporal_layer)
                    if temporal_state
                    else None
                )
                if projected_injection is not None:
                    replacement = (
                        projected_output[:, -1:, :]
                        + mx.array(projected_injection).reshape(
                            1, 1, int(projected_output.shape[-1])
                        )
                    )
                    projected_output = mx.concatenate(
                        [projected_output[:, :-1, :], replacement],
                        axis=1,
                    )
                if record_temporal:
                    temporal_state.setdefault(
                        "temporal_projected_attention_outputs", {}
                    )[temporal_layer] = projected_output[0, -1, :]
                return projected_output

            cls._l1kv_patched = True
            cls.__call__ = patched_call
            installed += 1
        expected = len(getattr(self.model.model, "layers", []))
        self.attention_state["hook_installed"] = installed == expected and expected > 0
        self.attention_state["hooked_layers"] = installed
        self.attention_state["expected_hook_layers"] = expected
        self.attention_state["hook_adapter"] = "qwen_projection_flow_v2"
        self.attention_state["hook_classes"] = sorted(set(hook_classes))
        if installed != expected:
            self.attention_state["hook_error_message"] = (
                f"attention hook coverage incomplete: installed={installed}, expected={expected}"
            )

    def run_one(
        self,
        sample: Dict[str, Any],
        sample_idx: int,
        method: str,
        budget: int,
        out_dir: Path,
    ) -> Dict[str, Any]:
        method_key = canonical_method(method)
        try:
            spec = get_method_spec(method)
        except Exception as exc:
            return self.skipped_result(sample, sample_idx, method, budget, str(exc))
        reason = unsupported_reason(method, "mlx")
        if reason or method_key not in SUPPORTED_MLX_METHODS:
            return self.skipped_result(
                sample,
                sample_idx,
                method,
                budget,
                reason or f"MLX backend does not yet support method={method!r}",
                spec=spec,
            )
        if spec.requires_attention and not self.model_info.get("supports_attention_output"):
            return self.skipped_result(
                sample,
                sample_idx,
                method,
                budget,
                "MLX backend does not expose attention weights for this model",
                spec=spec,
            )

        prompt_ids, full_ids, answer_positions, prompt_text = self.sample_tokens(sample)
        oracle_positions = self.oracle_positions_for_sample(sample, method_key)
        generation = self.generate_with_cache(prompt_ids, method_key, budget, oracle_positions)
        ppl_stats = self.teacher_forced_ppl(
            full_ids,
            answer_positions,
            method_key,
            budget,
            oracle_positions,
        )

        selected = generation["selected_tokens_by_layer"]
        evidence = [int(x) for x in sample.get("evidence_positions") or []]
        evidence_stats = self.evidence_stats(
            selected,
            evidence,
            generation.get("selected_tokens_by_head") or {},
        )
        path_stats = self.path_stats(
            selected,
            generation.get("selected_tokens_by_head") or {},
            sample.get("path_annotation") or {},
        )
        selected_artifacts = self.save_selected_and_scores(
            out_dir,
            method,
            budget,
            sample_idx,
            selected,
            generation.get("scores_by_layer") or {},
        )
        mechanism_artifacts = self.save_phase_locked_artifacts(
            out_dir,
            method,
            budget,
            sample_idx,
            sample,
            prompt_ids[:-1],
            generation.get("prefill_decision"),
            generation.get("estimator_events", []),
        )
        gt = sample.get("ground_truth") or sample.get("metadata", {}).get("answer")
        generated_text = generation["generated_text"]
        contains_gt = normalize_text(gt) in normalize_text(generated_text) if gt else False
        exact = normalize_text(generated_text) == normalize_text(gt) if gt else False
        f1 = answer_f1(generated_text, str(gt or ""))
        metadata = sample.get("metadata", {})
        official = evaluate_official(
            self.cfg.benchmark.name,
            metadata.get("task"),
            generated_text,
            sample,
            metadata,
        )
        eval_mode = (self.cfg.benchmark.evaluation or "ppl").lower()
        use_official_primary = (
            eval_mode in {"official", "both"}
            and official.get("official_score") is not None
        )
        if use_official_primary:
            primary_metric = official.get("official_metric_name") or "official_score"
            primary_score = official.get("official_score")
        else:
            primary_metric = "ppl"
            primary_score = ppl_stats.get("ppl")
        row_correct = bool(contains_gt or exact)
        if use_official_primary and official.get("official_correct") is not None:
            row_correct = bool(official.get("official_correct"))

        context_length = len(full_ids)
        effective_update_policy = self.cfg.eviction.update_policy
        effective_update_interval = self.cfg.eviction.update_interval
        effective_score_source = self.cfg.eviction.score_source
        effective_sink_size = int(self.cfg.eviction.sink_size)
        effective_recent_size = int(self.cfg.eviction.recent_size)
        if method_key in {"full", "random"}:
            effective_sink_size = 0
            effective_recent_size = 0
        elif method_key == "recency":
            effective_sink_size = 0
            effective_recent_size = int(budget)
        elif method_key in {"sink_recent", "streamingllm"}:
            effective_sink_size = min(int(self.cfg.eviction.sink_size), int(budget))
            effective_recent_size = max(0, int(budget) - effective_sink_size)
        if method_key in {
            "l1_prefill_only",
            "l2_prefill_only",
            "l2_key_prefill_only",
            "compactor",
            "attention_l1_compactor",
            "attention_l2_compactor",
            "adakv",
        } or method_key in INNOVATION_PREFILL_METHODS:
            effective_update_policy = "prefill_only"
            effective_update_interval = 0
        elif method_key in {"l1_decode_only", "l2_decode_only"}:
            effective_update_policy = "decode_only"
        if method_key == "l2_key_prefill_only":
            effective_score_source = "key"
        elif method_key == "compactor":
            effective_score_source = "pre_rope_key_approximate_leverage+non_causal_attention"
        elif method_key == "adakv":
            effective_score_source = "adaptive_head_budget+snapkv_observation_attention"
        elif method_key == "knorm":
            effective_score_source = "negative_key_l2_norm"
        elif method_key == "keydiff":
            effective_score_source = "negative_cosine_to_mean_key"
        elif method_key == "vatp":
            effective_score_source = "accumulated_attention_times_value_l1_norm"
        elif method_key == "curdkv":
            effective_score_source = "gaussian_projected_key_value_row_norm_product"
        elif method_key == "conditional_v_leverage":
            effective_score_source = "ridge_residual_v_given_k_leverage"
        elif method_key == "conditional_k_leverage":
            effective_score_source = "ridge_residual_k_given_v_leverage"
        elif method_key == "attention_residual_v_leverage":
            effective_score_source = "attention_core+residual_value_ridge_leverage"
        elif method_key == "window_residual_v_leverage":
            effective_score_source = "observation_window_attention_core+residual_value_ridge_leverage"
        elif method_key == "attention_weighted_v_leverage":
            effective_score_source = "attention_weighted_value_ridge_leverage"
        elif method_key == "joint_kv_leverage":
            effective_score_source = "scale_normalized_joint_key_value_leverage"
        elif method_key == "ridge_v_allocation":
            effective_score_source = "budget_adaptive_ridge_value_leverage+effective_dimension_budget"
        elif method_key == "ridge_v_fixed":
            effective_score_source = "budget_adaptive_ridge_value_leverage+uniform_head_budget"
        elif method_key == "diversity_v_leverage":
            effective_score_source = "ridge_value_leverage+pivoted_qr"
        elif method_key == "attention_l1_compactor":
            effective_score_source = "rank_accumulated_attention+rank_value_l1_leverage"
        elif method_key == "attention_l2_compactor":
            effective_score_source = "rank_accumulated_attention+rank_key_l2_leverage"
        row: Dict[str, Any] = {
            "label": f"{method}_b{budget}_s{sample_idx}",
            "experiment_name": self.cfg.experiment_name,
            "run_id": self.cfg.run_id,
            "code_commit_hash": self.run_provenance.get("git_commit"),
            "repository_state_id": self.run_provenance.get("repository_state_id"),
            "config_hash": self.run_provenance.get("config_hash"),
            "sample_manifest_hash": self.run_provenance.get("sample_manifest_hash"),
            "sample_id": sample_idx,
            "sample_idx": sample_idx,
            "official_dataset_index": metadata.get("official_dataset_index"),
            "method": method,
            "canonical_method": method_key,
            "method_family": spec.family,
            "budget": budget,
            "cache_budget": budget,
            "model": self.cfg.model.name,
            "model_name": self.cfg.model.name,
            "model_family": self.model_info.get("model_family"),
            "backend": "mlx",
            "quant_bits": self.cfg.model.quant_bits,
            "benchmark": self.cfg.benchmark.name,
            "dataset_split": self.cfg.benchmark.split,
            "context_length": context_length,
            "max_new_tokens": self.cfg.benchmark.max_new_tokens,
            "tokenizer": self.model_info.get("tokenizer_class"),
            "prompt": prompt_text if self.cfg.save_prompt_text else None,
            "prompt_hash": text_hash(prompt_text),
            "prediction": generated_text.strip(),
            "generated_text": generated_text,
            "generated_token_ids": generation["generated_token_ids"],
            "ground_truth": gt,
            "answers": sample.get("answers") or metadata.get("answers"),
            "all_classes": sample.get("all_classes") or metadata.get("all_classes"),
            "contains_ground_truth": contains_gt,
            "exact_match": exact,
            "answer_f1": f1,
            "correct": row_correct,
            "official_score": official.get("official_score"),
            "official_correct": official.get("official_correct"),
            "official_metric_name": official.get("official_metric_name"),
            "official_metric_implementation": official.get("official_metric_implementation"),
            "official_references": official.get("official_references"),
            "dataset_official": metadata.get("dataset_official"),
            "official_prompt": metadata.get("official_prompt"),
            "primary_metric": primary_metric,
            "primary_score": primary_score,
            "metric": {
                "ppl": ppl_stats.get("ppl"),
                "contains_ground_truth": contains_gt,
                "official_score": official.get("official_score"),
                "official_metric_name": official.get("official_metric_name"),
                "primary_metric": primary_metric,
                "primary_score": primary_score,
            },
            "ppl": ppl_stats.get("ppl"),
            "mean_nll": ppl_stats.get("mean_nll"),
            "n_eval_tokens": ppl_stats.get("n_eval_tokens"),
            "loss": ppl_stats.get("mean_nll"),
            "latency": generation["total_time_s"] + ppl_stats.get("ppl_time_s", 0.0),
            "generated_token_count": generation.get("generated_token_count", 0),
            "tokens_per_second": generation["tokens_per_second"],
            "avg_ms_per_token": generation["avg_ms_per_token"],
            "total_time_s": generation["total_time_s"] + ppl_stats.get("ppl_time_s", 0.0),
            "generation_time_s": generation.get("generation_time_s"),
            "prefill_time_s": generation["prefill_time_s"],
            "decode_time_s": generation["decode_time_s"],
            "eviction_time_s": generation["eviction_time_s"],
            "decode_loop_time_s": generation.get("decode_loop_time_s"),
            "end_to_end_decode_tokens_per_second": generation.get(
                "end_to_end_decode_tokens_per_second"
            ),
            "eviction_overhead_fraction": generation.get(
                "eviction_overhead_fraction"
            ),
            "score_time_s": generation["score_time_s"],
            "topk_time_s": generation["topk_time_s"],
            "cache_rebuild_time_s": generation["cache_rebuild_time_s"],
            "ppl_time_s": ppl_stats.get("ppl_time_s"),
            "ppl_prefill_time_s": ppl_stats.get("ppl_prefill_time_s"),
            "ppl_decode_time_s": ppl_stats.get("ppl_decode_time_s"),
            "ppl_eviction_time_s": ppl_stats.get("ppl_eviction_time_s"),
            "ppl_score_time_s": ppl_stats.get("ppl_score_time_s"),
            "ppl_end_to_end_decode_tokens_per_second": ppl_stats.get(
                "ppl_end_to_end_decode_tokens_per_second"
            ),
            "ppl_eviction_overhead_fraction": ppl_stats.get(
                "ppl_eviction_overhead_fraction"
            ),
            "ppl_score_refit_count": ppl_stats.get("ppl_score_refit_count"),
            "max_kv_len": generation["max_kv_len"],
            "final_kv_len": generation["final_kv_len"],
            "avg_kv_len": generation["avg_kv_len"],
            "peak_token_head_slots": generation.get("peak_token_head_slots"),
            "final_token_head_slots": generation.get("final_token_head_slots"),
            "avg_token_head_slots": generation.get("avg_token_head_slots"),
            "slot_time_integral": generation.get("slot_time_integral"),
            "max_kv_len_observed": generation["max_kv_len"],
            "cache_shape_summary": generation["cache_shape_summary"],
            "peak_memory_bytes": generation.get("peak_memory_bytes"),
            "active_memory_bytes": generation.get("active_memory_bytes"),
            "sink_size": effective_sink_size,
            "recent_size": effective_recent_size,
            "configured_sink_size": self.cfg.eviction.sink_size,
            "configured_recent_size": self.cfg.eviction.recent_size,
            "score_budget": max(
                0,
                budget
                - self.cfg.eviction.sink_size
                - self.cfg.eviction.recent_size
                - 1,
            ),
            "score_budget_excludes_current_token": True,
            "score_budget_is_candidate_capacity_upper_bound": True,
            "score_source": effective_score_source,
            "configured_score_source": self.cfg.eviction.score_source,
            "seed": self.cfg.seed,
            "score_normalization": self.cfg.eviction.score_normalization,
            "attention_score_source": self._attention_score_source(method_key),
            "attention_definition": self._attention_score_source(method_key),
            "attention_definition_params": {
                "prefill_query_scope": (
                    "all_queries"
                    if method_key in {"attention", "attention_decay", "h2o", "vatp"} or method_key in HYBRID_METHODS
                    else "observation_window"
                ),
                "observation_window": int(self.cfg.eviction.observation_window),
                "attention_window": int(self.cfg.eviction.attention_window),
                "direct_policy_layers": list(
                    self.cfg.eviction.direct_policy_layers or []
                ),
                "decay_gamma": float(self.attention_state.get("decay_gamma", 0.0)),
            },
            "geometry_score_source": self._geometry_score_source(method_key),
            "sketch_dim": self.cfg.eviction.sketch_dim,
            "update_interval": effective_update_interval,
            "update_policy": effective_update_policy,
            "layer_strategy": self.cfg.eviction.layer_strategy,
            "head_strategy": self.cfg.eviction.head_strategy,
            "hybrid_mode": self.cfg.eviction.hybrid_mode,
            "lambda_attn": self.cfg.eviction.lambda_attn,
            "attn_budget_ratio": self.cfg.eviction.attn_budget_ratio,
            "l1_budget_ratio": self.cfg.eviction.l1_budget_ratio,
            "l2_budget_ratio": None,
            "recent_budget_ratio": self.cfg.eviction.recent_budget_ratio,
            "sink_budget_ratio": self.cfg.eviction.sink_budget_ratio,
            "selected_tokens": selected,
            "selected_tokens_by_layer": selected,
            "selected_token_sources": generation.get("selected_token_sources") or {},
            "selected_tokens_by_head": generation.get("selected_tokens_by_head") or {},
            "selected_token_types": self.selected_token_types(
                selected, prompt_ids + generation["generated_token_ids"]
            ),
            "selected_token_texts": self.selected_token_texts(
                selected, prompt_ids + generation["generated_token_ids"]
            ),
            "selected_token_distances_to_query": self.distances_to_target(
                selected, metadata.get("answer_token_start")
            ),
            "selected_token_distances_to_evidence": self.distances_to_evidence(
                selected, evidence
            ),
            "evidence_positions": evidence,
            "evidence_recall": evidence_stats["evidence_recall"],
            "evidence_precision": evidence_stats["evidence_precision"],
            "evidence_overlap_count": evidence_stats["evidence_overlap_count"],
            "evidence_any_unit_recall": evidence_stats["evidence_any_unit_recall"],
            "evidence_recall_by_unit": evidence_stats["evidence_recall_by_unit"],
            **path_stats,
            "needle_depth": metadata.get("needle_depth", metadata.get("depth_bucket")),
            "needle_token_start": metadata.get("needle_token_start"),
            "needle_token_end": metadata.get("needle_token_end"),
            "answer_token_start": metadata.get("answer_token_start"),
            "answer_token_end": metadata.get("answer_token_end"),
            "score_update_count": generation["score_update_count"],
            "score_phase_counts": generation.get("score_phase_counts", {}),
            "score_refit_count": generation.get("score_refit_count", 0),
            "score_refit_phase_counts": generation.get("score_refit_phase_counts", {}),
            "estimator_events": generation.get("estimator_events", []),
            "estimator_failures": generation.get("estimator_failures", []),
            "estimator_fallback": bool(generation.get("estimator_fallback_count", 0)),
            "estimator_fallback_count": int(generation.get("estimator_fallback_count", 0)),
            "fallback_reason": None,
            "eviction_count": generation["eviction_count"],
            "score_stats": self.score_stats_from_layers(generation.get("scores_by_layer") or {}),
            "effective_rank": self.estimator_metric_by_unit(
                generation.get("estimator_events", []), "effective_rank"
            ),
            "condition_number": self.estimator_metric_by_unit(
                generation.get("estimator_events", []), "condition_number"
            ),
            "leverage_concentration": self.score_concentration_by_unit(
                generation.get("scores_by_head", {})
            ),
            "score_phase": "pre_answer",
            "raw_score_stats": self.score_stats_from_layers(generation.get("scores_by_layer") or {}),
            "normalized_score_stats": self.normalized_score_stats_from_layers(
                generation.get("scores_by_layer") or {},
                self.cfg.eviction.score_normalization,
            ),
            "top_score_values": self.score_stats_from_layers(generation.get("scores_by_layer") or {}).get("top_values", []),
            "score_update_interval": effective_update_interval,
            "decode_only_prefill_policy": (
                "sink_recent" if method_key in {"l1_decode_only", "l2_decode_only"} else None
            ),
            "cache_budget_scope": generation.get("cache_budget_scope", "total_kv"),
            "prefill_compression": bool(generation.get("prefill_compression", False)),
            "prefill_requested_budgets_by_unit": self.decision_budgets(
                generation.get("prefill_decision")
            ),
            "sparse_head_mask": bool(generation.get("sparse_head_mask", False)),
            "paper_method": spec.paper_method,
            "paper_title": spec.paper_title,
            "paper_url": spec.paper_url,
            "reference_implementation_url": spec.reference_implementation_url,
            "faithful_baseline": spec.implementation_fidelity in {"faithful", "faithful_core"},
            "baseline_fidelity": spec.implementation_fidelity,
            "fidelity_notes": spec.fidelity_notes,
            "protected_first_tokens": (
                (0 if getattr(self.cfg.eviction, "compactor_protected_first_tokens", None) is None else self.cfg.eviction.compactor_protected_first_tokens)
                if method_key == "compactor" else None
            ),
            "protected_last_tokens": (
                (0 if getattr(self.cfg.eviction, "compactor_protected_last_tokens", None) is None else self.cfg.eviction.compactor_protected_last_tokens)
                if method_key == "compactor" else None
            ),
            "vector_shape": self.vector_shape_summary(generation["cache_shape_summary"], method_key),
            "approximate": bool(spec.approximate),
            "experimental": bool(spec.experimental),
            "oracle": bool(spec.oracle),
            "skipped": False,
            "skipped_reason": None,
            "unsupported_reason": None,
            "attention_hook_errors": generation.get("attention_hook_errors", 0),
            "attention_hook_error_events": generation.get("attention_hook_error_events", []),
            "attention_query_counts": generation.get("attention_query_counts", {}),
            "scores_by_head": generation.get("scores_by_head", {}),
            **mechanism_artifacts,
            "selected_tokens_path": selected_artifacts.get("selected_tokens_path"),
            "scores_path": selected_artifacts.get("scores_path"),
            "metadata": metadata,
        }
        row["sanity_checks"] = self.sanity_checks(row, spec)
        row["sanity_check_failed"] = bool(row["sanity_checks"].get("violations"))
        if method_key in MANUAL_COMPACT_METHODS:
            row["unsupported_warning"] = (
                "MLX manual KVCache compaction keeps already-rotated keys and compacts "
                "the physical cache while preserving a logical RoPE offset for subsequent "
                "tokens. This is a functional research runner for controlled eviction "
                "comparisons; dedicated production cache adapters are still recommended "
                "before claiming kernel-level parity."
            )
        return row

    def sanity_checks(self, row: Dict[str, Any], spec: Any) -> Dict[str, Any]:
        violations: List[str] = []
        budget = int(row.get("budget") or 0)
        context_length = int(row.get("context_length") or 0)
        generated_count = len(row.get("generated_token_ids") or [])
        selected = row.get("selected_tokens_by_layer") or {}
        selected_by_head = row.get("selected_tokens_by_head") or {}
        final_kv_len = row.get("final_kv_len")
        method_key = row.get("canonical_method") or row.get("method")
        scope = row.get("cache_budget_scope") or "total_kv"
        sink_required = method_key not in {
            "recency",
            "random",
            "full",
            "streamingllm",
            "h2o",
            "tova",
            "knorm",
            "keydiff",
            "vnorml1",
            "vnorml2",
            "vatp",
        } and method_key not in PREFILL_COMPRESS_METHODS
        if (
            method_key != "full"
            and scope == "total_kv"
            and final_kv_len is not None
            and budget > 0
            and int(final_kv_len) > budget
        ):
            violations.append(f"final_kv_len={final_kv_len} exceeds budget={budget}")
        sink_size = int(row.get("sink_size") or 0)
        recent_size = int(row.get("recent_size") or 0)
        for layer, values in selected.items():
            vals = [int(x) for x in values]
            if len(vals) != len(set(vals)):
                violations.append(f"layer {layer}: duplicate selected tokens")
            if any(x < 0 or (context_length and x >= context_length + row.get("max_new_tokens", 0)) for x in vals):
                violations.append(f"layer {layer}: selected token out of original stream range")
            if method_key != "full" and method_key not in PREFILL_COMPRESS_METHODS and budget > 0 and len(vals) > budget:
                violations.append(f"layer {layer}: selected count {len(vals)} exceeds budget {budget}")
            if sink_required and sink_size > 0 and len(vals) >= sink_size:
                missing_sink = [i for i in range(sink_size) if i not in set(vals)]
                if missing_sink and not getattr(spec, "oracle", False):
                    violations.append(f"layer {layer}: missing sink tokens {missing_sink[:5]}")
            if recent_size > 0 and context_length > 0 and len(vals) >= recent_size:
                recent_start = max(0, int(row.get("final_kv_len") or context_length) - recent_size)
                # Original-position maps can include generated tokens, so only check count/budget here.
                _ = recent_start
        if method_key in PREFILL_COMPRESS_METHODS:
            unit_budgets = row.get("prefill_requested_budgets_by_unit") or {}
            for layer, head_map in selected_by_head.items():
                total_pairs = 0
                for head, values in (head_map or {}).items():
                    vals = [int(x) for x in values]
                    total_pairs += len(vals)
                    if len(vals) != len(set(vals)):
                        violations.append(f"layer {layer} head {head}: duplicate selected tokens")
                    if any(x < 0 or (context_length and x >= context_length + row.get("max_new_tokens", 0)) for x in vals):
                        violations.append(f"layer {layer} head {head}: selected token out of original stream range")
                    requested = int(
                        (unit_budgets.get(str(layer), {}) or {}).get(str(head), budget)
                    )
                    if requested > 0 and len(vals) > requested + generated_count:
                        violations.append(
                            f"layer {layer} head {head}: selected count {len(vals)} "
                            f"exceeds prompt_budget+generated={requested + generated_count}"
                        )
                if method_key in VARIABLE_HEAD_PREFILL_METHODS and budget > 0 and head_map:
                    h_count = max(1, len(head_map))
                    allowed_pairs = (budget + generated_count) * h_count
                    if total_pairs > allowed_pairs:
                        violations.append(f"layer {layer}: selected pairs {total_pairs} exceeds (budget+generated)*heads={allowed_pairs}")
        return {"passed": not violations, "violations": violations}

    @staticmethod
    def decision_budgets(decision: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
        result: Dict[str, Dict[str, int]] = {}
        for unit in (decision or {}).get("units", []):
            layer = str(unit.get("layer"))
            head = "shared" if unit.get("head") is None else str(unit.get("head"))
            result.setdefault(layer, {})[head] = int(unit.get("requested_budget", 0))
        return result

    @staticmethod
    def _attention_score_source(method_key: str) -> Optional[str]:
        if method_key == "latest_attention_shared":
            return "latest_query_attention_six_layer_normalized_shared_set"
        if method_key == "temporal_volatility_shared":
            return "four_query_attention_std_six_layer_normalized_shared_set"
        if method_key == "tova":
            return "latest_query_causal_attention_averaged_over_query_heads"
        if method_key in {"h2o", "vatp"}:
            return "all_prefill_queries_plus_decode_accumulated_causal_attention"
        if method_key == "windowed_attention":
            return "rolling_observation_window_causal_attention"
        if method_key == "attention_decay":
            return "all_prefill_queries_plus_decode_querywise_decayed_causal_attention"
        if method_key in ATTENTION_SCORE_METHODS:
            return "all_prefill_queries_plus_decode_accumulated_causal_attention"
        if method_key in SNAP_METHODS:
            return "observation_window_current_query_attention"
        if method_key == "pyramidkv":
            return "layer_budget_observation_window_attention"
        if method_key == "compactor":
            return "prefill_non_causal_attention"
        if method_key == "adakv":
            return "observation_window_attention_with_adaptive_head_budget"
        if method_key in {
            "attention_residual_v_leverage",
            "attention_weighted_v_leverage",
        }:
            return "all_prefill_queries_accumulated_causal_attention_by_kv_head"
        if method_key == "window_residual_v_leverage":
            return "final_observation_window_causal_attention_by_kv_head"
        if method_key == "window_weighted_v_leverage":
            return "final_observation_window_causal_attention_by_kv_head"
        if method_key in HYBRID_METHODS:
            return "all_prefill_queries_plus_decode_accumulated_causal_attention"
        return None

    @staticmethod
    def _geometry_score_source(method_key: str) -> Optional[str]:
        if method_key == "token_rarity_shared":
            return "observed_stream_local_span_token_rarity"
        if method_key == "query_overlap_shared":
            return "prompt_tail_overlap_inverse_stream_frequency"
        if method_key == "position_coverage_shared":
            return "deterministic_logical_position_coverage"
        if method_key == "knorm":
            return "negative_key_l2_norm"
        if method_key == "keydiff":
            return "negative_cosine_to_mean_key"
        if method_key == "vnorml1":
            return "value_l1_norm"
        if method_key == "vnorml2":
            return "value_l2_norm"
        if method_key == "vatp":
            return "value_l1_norm"
        if method_key == "curdkv":
            return "gaussian_projected_key_value_row_norm_product"
        if method_key == "conditional_v_leverage":
            return "ridge_residual_v_given_k_leverage"
        if method_key == "conditional_k_leverage":
            return "ridge_residual_k_given_v_leverage"
        if method_key == "attention_residual_v_leverage":
            return "attention_core_orthogonal_complement_value_ridge_leverage"
        if method_key == "window_residual_v_leverage":
            return "window_attention_core_orthogonal_complement_value_ridge_leverage"
        if method_key == "attention_weighted_v_leverage":
            return "attention_weighted_value_ridge_leverage"
        if method_key == "window_weighted_v_leverage":
            return "window_attention_weighted_value_ridge_leverage"
        if method_key == "joint_kv_leverage":
            return "scale_normalized_joint_key_value_ridge_leverage"
        if method_key == "ridge_v_allocation":
            return "budget_adaptive_value_ridge_leverage_and_effective_dimension"
        if method_key == "ridge_v_fixed":
            return "budget_adaptive_value_ridge_leverage_with_uniform_head_budget"
        if method_key == "ridge_v_shared":
            return "budget_adaptive_value_ridge_leverage_with_shared_token_selection"
        if method_key == "diversity_v_leverage":
            return "value_ridge_leverage_candidates_and_pivoted_qr"
        if method_key in ("key_l2_norm", "key_l1_norm"):
            return "key"
        if method_key in ("value_l2_norm", "value_l1_norm"):
            return "value"
        if method_key in ("l1", "l1_leverage", "l1_prefill_only", "l1_decode_only", "sink_recent_l1"):
            return "l1_leverage"
        if method_key in ("l2", "l2_leverage", "l2_prefill_only", "l2_decode_only", "sink_recent_l2"):
            return "l2_leverage"
        if method_key == "key_l2_leverage":
            return "key_l2_leverage"
        if method_key == "value_l2_leverage":
            return "value_l2_leverage"
        if method_key == "kv_l2_leverage":
            return "key_value_concat_l2_leverage"
        if method_key == "l2_key_prefill_only":
            return "key_l2_leverage"
        if method_key == "compactor":
            return "pre_rope_key_approximate_leverage+non_causal_attention"
        if method_key in ("attention_l2", "attn_l2", "attention_l2_compactor"):
            return "l2_leverage"
        if method_key == "attention_l1_compactor":
            return "l1_leverage"
        if method_key == "attention_norm":
            return "value_l2_norm"
        if method_key == "attention_recency":
            return "recency"
        if method_key in HYBRID_METHODS:
            return "l1_leverage"
        return None

    def generate_with_cache(
        self,
        prompt_ids: List[int],
        method: str,
        budget: int,
        oracle_positions: Optional[List[int]] = None,
    ):
        import mlx.core as mx

        mx.reset_peak_memory()
        self.reset_attention_state()
        self.attention_state["enabled"] = method in METHODS_NEED_ATTENTION
        self.attention_state["current_method"] = method
        self.configure_attention_recording(method)
        cache = self.make_cache(method, budget)
        evictor = None
        if method in MANUAL_COMPACT_METHODS:
            evictor = MLXCacheEvictor(
                method,
                budget,
                self.cfg,
                len(cache),
                attention_state=self.attention_state,
                oracle_positions=oracle_positions,
                stream_token_ids=prompt_ids,
            )

        t_start = time.perf_counter()
        prefill_time = self.prefill(prompt_ids[:-1], cache, evictor, budget)
        prefill_decision = (
            dict(evictor.prefill_decision)
            if evictor and evictor.prefill_decision
            else self.baseline_prefill_decision(method, budget, len(prompt_ids) - 1)
        )
        if evictor:
            evictor.set_phase("decode")
        self.attention_state["phase"] = "decode"
        if method in PREFILL_COMPRESS_METHODS:
            # These methods make exactly one prompt-compression decision.
            # Continuing to accumulate attention on their heterogeneous
            # head-wise cache has no consumer and would mix a new universe
            # into the prefill statistic.
            self.attention_state["enabled"] = False
        current = int(prompt_ids[-1])
        generated: List[int] = []
        kv_lens: List[int] = []
        kv_slot_pairs: List[int] = []
        decode_time = 0.0
        eviction_time = 0.0
        for _ in range(max(0, self.cfg.benchmark.max_new_tokens)):
            if evictor and method not in PREFILL_COMPRESS_METHODS:
                t0 = time.perf_counter()
                evictor.evict_for_space(cache, 1)
                eviction_time += time.perf_counter() - t0
            t0 = time.perf_counter()
            logits = self.model(mx.array([[current]]), cache=cache)
            next_token = mx.argmax(logits[:, -1, :], axis=-1)
            mx.eval(next_token)
            decode_time += time.perf_counter() - t0
            if evictor:
                evictor.sync_maps(cache)
                if method not in PREFILL_COMPRESS_METHODS:
                    t0 = time.perf_counter()
                    evictor.evict(cache, budget)
                    eviction_time += time.perf_counter() - t0
            token = int(next_token.item())
            generated.append(token)
            if evictor:
                evictor.append_stream_token(token)
            kv_lens.append(self.cache_len(cache))
            kv_slot_pairs.append(self.cache_slot_pairs(cache))
            current = token
            if token in getattr(self.tokenizer, "eos_token_ids", set()):
                break

        if evictor:
            evictor.sync_maps(cache)
            selected = evictor.last_selected
            scores = evictor.last_scores
            component_sources = evictor.last_component_sources
            selected_by_head = evictor.last_selected_by_head
            profile = evictor.profile_times
            eviction_count = evictor.eviction_count
            score_update_count = evictor.score_update_count
        else:
            # The prompt's final token is the first decode input. The sampled
            # token produced after the last forward has not itself entered KV.
            total_consumed = max(0, len(prompt_ids) - 1 + len(generated))
            if method in ("full", "basic", "basic_generate"):
                keep = list(range(total_consumed))
            elif method in {"sink_recent", "streamingllm"}:
                if total_consumed <= budget:
                    keep = list(range(total_consumed))
                else:
                    sink = min(int(self.cfg.eviction.sink_size), max(0, budget))
                    recent_budget = max(0, int(budget) - sink)
                    recent_start = max(sink, total_consumed - recent_budget)
                    keep = sorted(
                        set(range(sink)) | set(range(recent_start, total_consumed))
                    )
            else:
                keep = list(range(max(0, total_consumed - budget), total_consumed))
            selected = {i: keep for i in range(len(cache))}
            scores = {}
            component_sources = {}
            selected_by_head = {}
            profile = {"score_time_s": 0.0, "topk_time_s": 0.0, "cache_rebuild_time_s": 0.0}
            eviction_count = max(0, total_consumed - budget)
            score_update_count = 0

        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        total = time.perf_counter() - t_start
        decode_tokens = max(1, len(generated))
        decode_loop_time = decode_time + eviction_time
        return {
            "generated_token_ids": generated,
            "generated_token_count": len(generated),
            "generated_text": text,
            "selected_tokens_by_layer": {str(k): v for k, v in selected.items()},
            "scores_by_layer": {str(k): v for k, v in scores.items()},
            "selected_token_sources": {str(k): v for k, v in component_sources.items()},
            "selected_tokens_by_head": {str(k): v for k, v in selected_by_head.items()},
            "cache_budget_scope": "prompt_prefill" if method in PREFILL_COMPRESS_METHODS else "total_kv",
            "prefill_compression": bool(method in PREFILL_COMPRESS_METHODS),
            "sparse_head_mask": bool(method in VARIABLE_HEAD_PREFILL_METHODS),
            "prefill_time_s": prefill_time,
            "decode_time_s": decode_time,
            "eviction_time_s": eviction_time,
            "generation_time_s": total,
            "decode_loop_time_s": decode_loop_time,
            "end_to_end_decode_tokens_per_second": (
                len(generated) / decode_loop_time if decode_loop_time > 0 else 0.0
            ),
            "eviction_overhead_fraction": (
                eviction_time / decode_loop_time if decode_loop_time > 0 else 0.0
            ),
            "score_time_s": profile.get("score_time_s", 0.0),
            "topk_time_s": profile.get("topk_time_s", 0.0),
            "cache_rebuild_time_s": profile.get("cache_rebuild_time_s", 0.0),
            "total_time_s": total,
            "tokens_per_second": len(generated) / decode_time if decode_time > 0 else 0.0,
            "avg_ms_per_token": (decode_time / decode_tokens) * 1000.0,
            "max_kv_len": max(kv_lens) if kv_lens else self.cache_len(cache),
            "final_kv_len": self.cache_len(cache),
            "avg_kv_len": sum(kv_lens) / len(kv_lens) if kv_lens else self.cache_len(cache),
            "peak_token_head_slots": max(kv_slot_pairs) if kv_slot_pairs else self.cache_slot_pairs(cache),
            "final_token_head_slots": self.cache_slot_pairs(cache),
            "avg_token_head_slots": (
                sum(kv_slot_pairs) / len(kv_slot_pairs)
                if kv_slot_pairs else self.cache_slot_pairs(cache)
            ),
            "slot_time_integral": sum(kv_slot_pairs),
            "peak_memory_bytes": int(mx.get_peak_memory()),
            "active_memory_bytes": int(mx.get_active_memory()),
            "cache_shape_summary": self.cache_shape_summary(cache),
            "eviction_count": eviction_count,
            "score_update_count": score_update_count,
            "score_phase_counts": (
                dict(evictor.score_phase_counts) if evictor else {"prefill": 0, "decode": 0}
            ),
            "score_refit_count": (
                int(evictor.score_refit_count) if evictor else 0
            ),
            "score_refit_phase_counts": (
                dict(evictor.score_refit_phase_counts) if evictor else {"prefill": 0, "decode": 0}
            ),
            "estimator_events": list(evictor.estimator_events) if evictor else [],
            "estimator_failures": list(evictor.score_failures) if evictor else [],
            "estimator_fallback_count": 0,
            "attention_hook_errors": int(self.attention_state.get("hook_errors", 0)),
            "attention_hook_error_events": list(
                self.attention_state.get("hook_error_events", [])
            ),
            "attention_query_counts": dict(
                self.attention_state.get("query_counts", {})
            ),
            "prefill_decision": prefill_decision,
            "scores_by_head": (
                {str(k): v for k, v in evictor.last_scores_by_head.items()}
                if evictor else {}
            ),
        }

    def teacher_forced_ppl(
        self,
        full_ids: List[int],
        answer_positions: List[int],
        method: str,
        budget: int,
        oracle_positions: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        import mlx.core as mx

        if not answer_positions:
            return {"ppl": None, "mean_nll": None, "ppl_time_s": 0.0}
        self.reset_attention_state()
        self.attention_state["enabled"] = method in METHODS_NEED_ATTENTION
        self.attention_state["current_method"] = method
        self.configure_attention_recording(method)
        answer_positions = answer_positions[: max(1, self.cfg.benchmark.max_new_tokens)]
        answer_start = min(answer_positions)
        if answer_start <= 0:
            return {"ppl": None, "mean_nll": None, "ppl_time_s": 0.0}
        prompt_prefix = full_ids[: answer_start]
        cache = self.make_cache(method, budget)
        evictor = None
        if method in MANUAL_COMPACT_METHODS:
            evictor = MLXCacheEvictor(
                method,
                budget,
                self.cfg,
                len(cache),
                attention_state=self.attention_state,
                oracle_positions=oracle_positions,
                stream_token_ids=prompt_prefix,
            )
        t_start = time.perf_counter()
        prefill_time = self.prefill(prompt_prefix[:-1], cache, evictor, budget)
        if evictor:
            evictor.set_phase("decode")
        self.attention_state["phase"] = "decode"
        if method in PREFILL_COMPRESS_METHODS:
            self.attention_state["enabled"] = False
        current = int(prompt_prefix[-1])
        nlls: List[float] = []
        decode_time = 0.0
        eviction_time = 0.0
        for pos in answer_positions:
            target = int(full_ids[pos])
            if evictor and method not in PREFILL_COMPRESS_METHODS:
                t0 = time.perf_counter()
                evictor.evict_for_space(cache, 1)
                eviction_time += time.perf_counter() - t0
            t0 = time.perf_counter()
            logits = self.model(mx.array([[current]]), cache=cache)
            log_probs = logits[:, -1, :] - mx.logsumexp(logits[:, -1, :], axis=-1, keepdims=True)
            mx.eval(log_probs)
            decode_time += time.perf_counter() - t0
            nlls.append(-float(log_probs[0, target].item()))
            if evictor:
                evictor.sync_maps(cache)
                if method not in PREFILL_COMPRESS_METHODS:
                    t0 = time.perf_counter()
                    evictor.evict(cache, budget)
                    eviction_time += time.perf_counter() - t0
            current = target
            if evictor:
                evictor.append_stream_token(target)
        mean_nll = sum(nlls) / len(nlls) if nlls else None
        total_time = time.perf_counter() - t_start
        decode_loop_time = decode_time + eviction_time
        score_time = (
            float(evictor.profile_times.get("score_time_s", 0.0)) if evictor else 0.0
        )
        return {
            "ppl": math.exp(mean_nll) if mean_nll is not None else None,
            "mean_nll": mean_nll,
            "n_eval_tokens": len(nlls),
            "ppl_time_s": total_time,
            "ppl_prefill_time_s": prefill_time,
            "ppl_decode_time_s": decode_time,
            "ppl_eviction_time_s": eviction_time,
            "ppl_score_time_s": score_time,
            "ppl_end_to_end_decode_tokens_per_second": (
                len(nlls) / decode_loop_time if decode_loop_time > 0 else 0.0
            ),
            "ppl_eviction_overhead_fraction": (
                eviction_time / decode_loop_time if decode_loop_time > 0 else 0.0
            ),
            "ppl_score_refit_count": (
                int(evictor.score_refit_count) if evictor else 0
            ),
        }

    def prefill(
        self,
        token_ids: List[int],
        cache: List[Any],
        evictor: Optional[MLXCacheEvictor],
        budget: int,
    ) -> float:
        import mlx.core as mx

        if not token_ids:
            return 0.0
        elapsed = 0.0
        step = max(1, int(self.cfg.model.prefill_step_size))
        if evictor:
            evictor.set_phase("prefill")
        self.attention_state["phase"] = "prefill"
        for start in range(0, len(token_ids), step):
            chunk = token_ids[start : start + step]
            t0 = time.perf_counter()
            logits = self.model(mx.array([chunk]), cache=cache)
            mx.eval(logits)
            elapsed += time.perf_counter() - t0
            if evictor:
                evictor.sync_maps(cache)
                # Do not mutate the cache between prefill chunks. The paper
                # artifact is defined on the complete pre-answer prompt
                # snapshot, and all methods must see that same universe.
        if evictor and evictor.method in PREFILL_COMPRESS_METHODS:
            t0 = time.perf_counter()
            evictor.prefill_compress(cache, budget)
            elapsed += time.perf_counter() - t0
        elif evictor:
            t0 = time.perf_counter()
            evictor.evict(cache, budget)
            elapsed += time.perf_counter() - t0
        return elapsed

    def make_cache(self, method: str, budget: int) -> List[Any]:
        from mlx_lm.models.cache import KVCache, RotatingKVCache

        num_layers = len(self.model.model.layers)
        if method == "recency":
            return [RotatingKVCache(max_size=int(budget), keep=0) for _ in range(num_layers)]
        if method in {"sink_recent", "streamingllm"}:
            keep = min(self.cfg.eviction.sink_size, max(0, int(budget) - 1))
            return [RotatingKVCache(max_size=int(budget), keep=keep) for _ in range(num_layers)]
        return [KVCache() for _ in range(num_layers)]

    def baseline_prefill_decision(
        self, method: str, budget: int, prompt_cache_length: int
    ) -> Optional[Dict[str, Any]]:
        method = canonical_method(method)
        if method not in {"full", "recency", "sink_recent", "streamingllm"}:
            return None
        universe = list(range(max(0, int(prompt_cache_length))))
        if method == "full" or len(universe) <= budget:
            selected = universe
            requested = len(universe) if method == "full" else int(budget)
        elif method in {"sink_recent", "streamingllm"}:
            sink = min(int(self.cfg.eviction.sink_size), int(budget))
            selected = sorted(
                set(range(sink))
                | set(range(max(sink, len(universe) - (int(budget) - sink)), len(universe)))
            )
            requested = int(budget)
        else:
            selected = universe[-int(budget):]
            requested = int(budget)
        heads = int(self.model_info.get("num_key_value_heads") or 1)
        layers = int(self.model_info.get("num_layers") or 1)
        return {
            "phase": "pre_answer",
            "budget_scope": "total_kv",
            "budget_unit": "token_slots_per_kv_head",
            "requested_budget": requested,
            "units": [
                {
                    "layer": layer,
                    "head": head,
                    "universe_positions": universe,
                    "score_positions": [],
                    "scores": [],
                    "selected_positions": selected,
                    "requested_budget": requested,
                }
                for layer in range(layers)
                for head in range(heads)
            ],
        }

    @staticmethod
    def cache_len(cache: List[Any]) -> int:
        if not cache:
            return 0
        return int(max(len(c) for c in cache))

    @staticmethod
    def cache_shape_summary(cache: List[Any]) -> Dict[str, Any]:
        layers = []
        total_valid_slots = 0
        total_physical_slots = 0
        for idx, c in enumerate(cache):
            if getattr(c, "keys", None) is None:
                continue
            kv_heads = int(c.keys.shape[1])
            physical_len = int(c.keys.shape[2])
            valid = getattr(c, "head_valid_mask", None)
            if valid is None:
                valid_by_head = [physical_len] * kv_heads
            else:
                valid_by_head = [
                    int(np.asarray(valid[head].tolist(), dtype=bool).sum())
                    for head in range(kv_heads)
                ]
            valid_slots = sum(valid_by_head)
            physical_slots = kv_heads * physical_len
            total_valid_slots += valid_slots
            total_physical_slots += physical_slots
            layers.append(
                {
                    "layer": idx,
                    "offset": int(getattr(c, "offset", 0)),
                    "logical_offset": int(
                        getattr(c, "logical_offset", getattr(c, "offset", 0))
                    ),
                    "len": int(len(c)),
                    "keys_shape": [int(x) for x in c.keys.shape],
                    "values_shape": [int(x) for x in c.values.shape],
                    "kv_heads": kv_heads,
                    "physical_slots_per_head": physical_len,
                    "valid_slots_by_head": valid_by_head,
                    "valid_token_head_slots": valid_slots,
                    "physical_token_head_slots": physical_slots,
                }
            )
        return {
            "num_layers": len(layers),
            "layers": layers,
            "total_valid_token_head_slots": total_valid_slots,
            "total_physical_token_head_slots": total_physical_slots,
        }

    @staticmethod
    def cache_slot_pairs(cache: List[Any]) -> int:
        total = 0
        for c in cache:
            keys = getattr(c, "keys", None)
            if keys is None:
                continue
            valid = getattr(c, "head_valid_mask", None)
            if valid is None:
                total += int(keys.shape[1]) * int(keys.shape[2])
            elif hasattr(c, "valid_token_head_slots"):
                total += int(c.valid_token_head_slots)
            else:
                total += int(np.asarray(valid.tolist(), dtype=bool).sum())
        return int(total)

    def sample_tokens(self, sample: Dict[str, Any]) -> Tuple[List[int], List[int], List[int], str]:
        full_ids = tensor_to_list(sample.get("input_ids"))
        answer_positions = [int(x) for x in sample.get("answer_positions") or sample.get("eval_positions") or []]
        prompt_text = sample.get("prompt")
        if prompt_text:
            prompt_text = apply_prompt_format(self.hf_tokenizer, prompt_text, self.cfg.model)
            prompt_ids = [int(x) for x in self.hf_tokenizer.encode(prompt_text)]
        elif answer_positions:
            prompt_ids = full_ids[: min(answer_positions)]
            prompt_text = self.hf_tokenizer.decode(prompt_ids)
        else:
            prompt_ids = full_ids
            prompt_text = self.hf_tokenizer.decode(prompt_ids)
        if not full_ids and "full_text" in sample:
            full_ids = [int(x) for x in self.hf_tokenizer.encode(sample["full_text"])]
        return prompt_ids, full_ids, answer_positions, prompt_text

    def _actual_budget(
        self,
        sample: Dict[str, Any],
        budget: int,
        budget_ratios: List[float],
    ) -> int:
        if not budget_ratios:
            return int(budget)
        full_ids = tensor_to_list(sample.get("input_ids"))
        actual = int(budget)
        for ratio in budget_ratios:
            actual = max(1, int(len(full_ids) * float(ratio)))
        return actual

    def save_selected_and_scores(
        self,
        out_dir: Path,
        method: str,
        budget: int,
        sample_idx: int,
        selected: Dict[str, List[int]],
        scores: Dict[str, List[float]],
    ) -> Dict[str, Optional[str]]:
        paths: Dict[str, Optional[str]] = {"selected_tokens_path": None, "scores_path": None}
        if self.cfg.save_selected_tokens:
            path = out_dir / "selected_tokens" / f"{method}_b{budget}_s{sample_idx}.json"
            save_results(selected, path)
            paths["selected_tokens_path"] = str(path)
        if self.cfg.save_scores and scores:
            path = out_dir / "scores" / f"{method}_b{budget}_s{sample_idx}.json"
            save_results(scores, path)
            paths["scores_path"] = str(path)
        return paths

    def save_phase_locked_artifacts(
        self,
        out_dir: Path,
        method: str,
        budget: int,
        sample_idx: int,
        sample: Dict[str, Any],
        prompt_cache_token_ids: List[int],
        decision: Optional[Dict[str, Any]],
        estimator_events: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Write schema-v2 score/selection files from one pre-answer decision."""
        result = {
            "score_artifact_path": None,
            "selection_artifact_path": None,
            "artifact_schema_version": None,
            "mechanism_artifacts_alignment_safe": False,
            "mechanism_artifacts_stale_reason": "no pre-answer eviction decision was captured",
        }
        if not decision or not decision.get("units"):
            return result
        metadata = sample.get("metadata", {}) or {}
        official_index = metadata.get("official_dataset_index")
        stable_sample_id = (
            f"official:{official_index}"
            if official_index is not None
            else f"sample:{sample_idx}:{text_hash(sample.get('prompt') or '')}"
        )
        token_payload = json.dumps(
            [int(token) for token in prompt_cache_token_ids], separators=(",", ":")
        )
        token_ids_hash = hashlib.sha256(token_payload.encode("utf-8")).hexdigest()
        snapshot_id = hashlib.sha256(
            f"{stable_sample_id}|pre_answer|{token_ids_hash}".encode("utf-8")
        ).hexdigest()
        snapshot = SnapshotRef(
            snapshot_id=snapshot_id,
            sample_id=stable_sample_id,
            phase="pre_answer",
            context_length=len(prompt_cache_token_ids),
            prompt_length=len(prompt_cache_token_ids),
            token_ids_hash=token_ids_hash,
        )
        selection_units = []
        score_units = []
        for unit in decision["units"]:
            universe = tuple(int(x) for x in unit["universe_positions"])
            selected_positions = tuple(int(x) for x in unit["selected_positions"])
            if not unit.get("score_only"):
                selection_units.append(
                    SelectionUnit(
                        layer=int(unit["layer"]),
                        head=(None if unit.get("head") is None else int(unit["head"])),
                        selected_positions=selected_positions,
                        universe_positions=universe,
                        requested_budget=int(unit.get("requested_budget", budget)),
                        effective_budget=len(selected_positions),
                    )
                )
            if unit.get("scores") and not unit.get("selection_only"):
                score_units.append(
                    ScoreUnit(
                        layer=int(unit["layer"]),
                        head=(None if unit.get("head") is None else int(unit["head"])),
                        original_positions=tuple(int(x) for x in unit["score_positions"]),
                        universe_positions=universe,
                        scores=tuple(float(x) for x in unit["scores"]),
                    )
                )
        if str(decision["budget_unit"]) == "token_head_pairs":
            by_layer: Dict[int, int] = {}
            for unit in selection_units:
                by_layer[unit.layer] = by_layer.get(unit.layer, 0) + int(
                    unit.effective_budget or 0
                )
            effective_budget = max(by_layer.values(), default=0)
        else:
            effective_budget = max(
                (int(unit.effective_budget or 0) for unit in selection_units), default=0
            )
        artifact_slug = f"{canonical_method(method)}_b{budget}_s{sample_idx}"
        selection = SelectionArtifact(
            artifact_id=f"selection:{artifact_slug}:{snapshot_id[:16]}",
            snapshot=snapshot,
            method=canonical_method(method),
            requested_budget=int(decision.get("requested_budget", budget)),
            effective_budget=effective_budget,
            budget_scope=str(decision["budget_scope"]),
            budget_unit=str(decision["budget_unit"]),
            units=tuple(selection_units),
            metadata={
                "physical_token_head_pairs": sum(
                    int(unit.effective_budget or 0) for unit in selection_units
                ),
                "implementation": "mlx_cache_evictor_v2",
            },
        )
        selection_path = out_dir / "artifacts" / f"selection_{artifact_slug}.json"
        save_artifact(selection, selection_path)
        result["selection_artifact_path"] = str(selection_path)
        if score_units:
            score = ScoreArtifact(
                artifact_id=f"score:{artifact_slug}:{snapshot_id[:16]}",
                snapshot=snapshot,
                method=canonical_method(method),
                score_type=self._geometry_score_source(canonical_method(method))
                or self._attention_score_source(canonical_method(method))
                or "method_score",
                score_source=(
                    self._geometry_score_source(canonical_method(method))
                    or self._attention_score_source(canonical_method(method))
                    or "method"
                ),
                units=tuple(score_units),
                definition={
                    "attention": self._attention_score_source(canonical_method(method)),
                    "geometry": self._geometry_score_source(canonical_method(method)),
                    "head_aggregation_for_shared_selection": "mean_after_independent_head_scoring",
                },
                estimator={"events": estimator_events},
                metadata={"implementation": "mlx_cache_evictor_v2"},
            )
            score_path = out_dir / "artifacts" / f"score_{artifact_slug}.json"
            save_artifact(score, score_path)
            result["score_artifact_path"] = str(score_path)
        result["artifact_schema_version"] = 2
        result["mechanism_artifacts_alignment_safe"] = bool(score_units)
        result["mechanism_artifacts_stale_reason"] = (
            None if score_units else "selection is phase-locked but this method has no score"
        )
        return result

    @staticmethod
    def evidence_stats(
        selected: Dict[str, List[int]],
        evidence_positions: List[int],
        selected_by_head: Optional[Dict[str, Dict[str, List[int]]]] = None,
    ) -> Dict[str, Any]:
        evidence = set(int(x) for x in evidence_positions)
        units: Dict[str, set] = {}
        for layer, heads in (selected_by_head or {}).items():
            for head, values in (heads or {}).items():
                units[f"layer={layer},head={head}"] = {int(x) for x in values}
        if not units:
            units = {
                f"layer={layer},head=shared": {int(x) for x in values}
                for layer, values in selected.items()
            }
        per_unit = {
            key: (len(evidence & values) / len(evidence) if evidence else 0.0)
            for key, values in units.items()
        }
        all_selected = set().union(*units.values()) if units else set()
        overlap = evidence & all_selected
        recall = sum(per_unit.values()) / len(per_unit) if per_unit else 0.0
        unit_precisions = [
            len(evidence & values) / len(values) if values else 0.0
            for values in units.values()
        ]
        precision = sum(unit_precisions) / len(unit_precisions) if unit_precisions else 0.0
        return {
            "evidence_recall": recall,
            "evidence_precision": precision,
            "evidence_overlap_count": len(overlap),
            "evidence_any_unit_recall": (
                len(overlap) / len(evidence) if evidence else 0.0
            ),
            "evidence_recall_by_unit": per_unit,
        }

    @staticmethod
    def path_stats(
        selected: Dict[str, List[int]],
        selected_by_head: Dict[str, Dict[str, List[int]]],
        annotation: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not annotation or not annotation.get("target_edge_ids"):
            return {
                "assignment_node_recall": None,
                "dependency_edge_recall": None,
                "complete_path_rate": None,
                "distractor_retention": None,
                "path_coverage_by_unit": {},
            }
        assignments = {
            int(edge["edge_id"]): edge for edge in annotation.get("assignments", [])
        }
        target_edges = [int(value) for value in annotation.get("target_edge_ids", [])]
        target_nodes = set(int(value) for value in annotation.get("assignment_node_positions", []))
        distractors = set(int(value) for value in annotation.get("distractor_positions", []))
        units: Dict[str, set] = {}
        for layer, heads in (selected_by_head or {}).items():
            for head, values in (heads or {}).items():
                units[f"layer={layer},head={head}"] = {int(x) for x in values}
        if not units:
            units = {
                f"layer={layer},head=shared": {int(x) for x in values}
                for layer, values in selected.items()
            }
        coverage: Dict[str, Dict[str, float]] = {}
        for key, kept in units.items():
            edge_token_recalls = []
            complete_edges = []
            for edge_id in target_edges:
                positions = set(int(x) for x in assignments[edge_id]["token_positions"])
                edge_token_recalls.append(
                    len(positions & kept) / len(positions) if positions else 0.0
                )
                complete_edges.append(bool(positions) and positions.issubset(kept))
            coverage[key] = {
                "assignment_node_recall": (
                    len(target_nodes & kept) / len(target_nodes) if target_nodes else 0.0
                ),
                "dependency_edge_recall": (
                    sum(edge_token_recalls) / len(edge_token_recalls)
                    if edge_token_recalls else 0.0
                ),
                "complete_path": float(bool(complete_edges) and all(complete_edges)),
                "distractor_retention": (
                    len(distractors & kept) / len(distractors) if distractors else 0.0
                ),
            }
        def macro(field: str) -> float:
            values = [value[field] for value in coverage.values()]
            return sum(values) / len(values) if values else 0.0
        return {
            "assignment_node_recall": macro("assignment_node_recall"),
            "dependency_edge_recall": macro("dependency_edge_recall"),
            "complete_path_rate": macro("complete_path"),
            "distractor_retention": macro("distractor_retention"),
            "path_coverage_by_unit": coverage,
        }

    def selected_token_texts(
        self,
        selected: Dict[str, List[int]],
        stream_ids: List[int],
        limit: int = 256,
    ) -> List[Dict[str, Any]]:
        union = sorted({int(x) for vals in selected.values() for x in vals})
        rows = []
        for pos in union[:limit]:
            if 0 <= pos < len(stream_ids):
                text = self.tokenizer.decode([stream_ids[pos]], skip_special_tokens=False)
                rows.append({"position": pos, "text": text, "type": token_type(text)})
        return rows

    def selected_token_types(
        self,
        selected: Dict[str, List[int]],
        stream_ids: List[int],
    ) -> Dict[str, int]:
        counts: Counter = Counter()
        union = sorted({int(x) for vals in selected.values() for x in vals})
        for pos in union:
            if 0 <= pos < len(stream_ids):
                text = self.tokenizer.decode([stream_ids[pos]], skip_special_tokens=False)
                counts[token_type(text)] += 1
        return dict(counts)

    @staticmethod
    def distances_to_target(
        selected: Dict[str, List[int]],
        target_pos: Optional[int],
        limit: int = 512,
    ) -> List[int]:
        if target_pos is None:
            return []
        union = sorted({int(x) for vals in selected.values() for x in vals})
        return [abs(pos - int(target_pos)) for pos in union[:limit]]

    @staticmethod
    def distances_to_evidence(
        selected: Dict[str, List[int]],
        evidence: List[int],
        limit: int = 512,
    ) -> List[int]:
        if not evidence:
            return []
        ev = [int(x) for x in evidence]
        union = sorted({int(x) for vals in selected.values() for x in vals})
        return [min(abs(pos - e) for e in ev) for pos in union[:limit]]

    @staticmethod
    def oracle_positions_for_sample(sample: Dict[str, Any], method_key: str) -> List[int]:
        if method_key == "oracle_evidence":
            return [int(x) for x in sample.get("evidence_positions") or []]
        if method_key == "oracle_answer_region":
            metadata = sample.get("metadata", {}) or {}
            start = metadata.get("answer_token_start")
            end = metadata.get("answer_token_end")
            if start is not None and end is not None:
                return list(range(int(start), int(end)))
            return [int(x) for x in sample.get("answer_positions") or sample.get("eval_positions") or []]
        return []

    @staticmethod
    def score_stats_from_layers(scores: Dict[str, List[float]]) -> Dict[str, Any]:
        values: List[float] = []
        for layer_values in (scores or {}).values():
            values.extend(float(x) for x in layer_values)
        return list_stats(values)

    @staticmethod
    def estimator_metric_by_unit(
        events: List[Dict[str, Any]], metric: str
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for event in events or []:
            if event.get("failed") or event.get(metric) is None:
                continue
            label = (
                f"layer={event.get('layer')},head="
                f"{'shared' if event.get('head') is None else event.get('head')}"
            )
            result[label] = event.get(metric)
        return result

    @staticmethod
    def score_concentration_by_unit(
        scores_by_head: Dict[str, Dict[str, List[float]]]
    ) -> Dict[str, float]:
        """Top-10% absolute score mass, reported separately per layer/head."""
        result: Dict[str, float] = {}
        for layer, heads in (scores_by_head or {}).items():
            for head, values in (heads or {}).items():
                array = np.abs(np.asarray(values, dtype=np.float64))
                if array.size == 0 or not np.isfinite(array).all() or float(array.sum()) <= 0:
                    value = 0.0
                else:
                    take = max(1, int(math.ceil(0.1 * array.size)))
                    value = float(np.partition(array, array.size - take)[-take:].sum() / array.sum())
                result[f"layer={layer},head={head}"] = value
        return result

    @staticmethod
    def normalized_score_stats_from_layers(
        scores: Dict[str, List[float]],
        normalization: str,
    ) -> Dict[str, Any]:
        values: List[float] = []
        for layer_values in (scores or {}).values():
            values.extend(float(x) for x in layer_values)
        if not values:
            return {}
        arr = np.asarray(values, dtype=np.float32)
        finite = np.isfinite(arr)
        if not finite.any():
            return {"numel": int(arr.size), "all_non_finite": True}
        mode = str(normalization or "none").lower()
        out = np.zeros_like(arr)
        vals = arr[finite]
        if mode == "minmax":
            denom = max(float(vals.max() - vals.min()), 1e-8)
            out[finite] = (vals - vals.min()) / denom
        elif mode == "zscore":
            out[finite] = (vals - vals.mean()) / max(float(vals.std()), 1e-8)
        elif mode == "softmax":
            shifted = vals - vals.max()
            exp = np.exp(shifted)
            out[finite] = exp / max(float(exp.sum()), 1e-8)
        elif mode == "rank":
            order = np.argsort(vals)
            ranks = np.zeros_like(vals)
            if vals.size > 1:
                ranks[order] = np.arange(vals.size, dtype=np.float32) / float(vals.size - 1)
            else:
                ranks[order] = 1.0
            out[finite] = ranks
        else:
            out[finite] = vals
        return list_stats(out.tolist())

    @staticmethod
    def vector_shape_summary(cache_shape_summary: Dict[str, Any], method_key: str) -> Optional[Dict[str, Any]]:
        if method_key not in {
            "l1_leverage",
            "l1_prefill_only",
            "l1_decode_only",
            "l2_leverage",
            "l2_prefill_only",
            "l2_key_prefill_only",
            "l2_decode_only",
            "compactor",
            "key_l2_norm",
            "value_l2_norm",
            "key_l1_norm",
            "value_l1_norm",
            "knorm",
            "keydiff",
            "vnorml1",
            "vnorml2",
            "vatp",
            "curdkv",
            "adakv",
            "conditional_v_leverage",
            "conditional_k_leverage",
            "attention_residual_v_leverage",
            "window_residual_v_leverage",
            "attention_weighted_v_leverage",
            "window_weighted_v_leverage",
            "joint_kv_leverage",
            "ridge_v_allocation",
            "ridge_v_fixed",
            "ridge_v_shared",
            "diversity_v_leverage",
            "sink_recent_l1",
            "sink_recent_l2",
            "attention_l1",
            "attention_l2",
        }:
            return None
        layers = (cache_shape_summary or {}).get("layers") or []
        if not layers:
            return None
        first = layers[0]
        return {
            "keys_shape": first.get("keys_shape"),
            "values_shape": first.get("values_shape"),
            "num_layers": len(layers),
        }

    def skipped_result(
        self,
        sample: Dict[str, Any],
        sample_idx: int,
        method: str,
        budget: int,
        reason: str,
        spec: Any = None,
    ) -> Dict[str, Any]:
        metadata = sample.get("metadata", {}) or {}
        prompt = sample.get("prompt")
        if spec is None:
            try:
                spec = get_method_spec(method)
            except Exception:
                spec = None
        return {
            "label": f"{method}_b{budget}_s{sample_idx}",
            "experiment_name": self.cfg.experiment_name,
            "run_id": self.cfg.run_id,
            "sample_id": sample_idx,
            "sample_idx": sample_idx,
            "method": method,
            "canonical_method": canonical_method(method),
            "method_family": getattr(spec, "family", "unknown"),
            "budget": budget,
            "cache_budget": budget,
            "model": self.cfg.model.name,
            "model_name": self.cfg.model.name,
            "model_family": self.model_info.get("model_family"),
            "backend": "mlx",
            "quant_bits": self.cfg.model.quant_bits,
            "benchmark": self.cfg.benchmark.name,
            "context_length": metadata.get("seq_len"),
            "prompt_hash": text_hash(prompt),
            "prediction": None,
            "generated_text": None,
            "ground_truth": sample.get("ground_truth"),
            "correct": None,
            "contains_ground_truth": None,
            "exact_match": None,
            "answer_f1": None,
            "ppl": None,
            "mean_nll": None,
            "official_score": None,
            "official_correct": None,
            "official_metric_name": metadata.get("official_metric_name"),
            "official_metric_implementation": None,
            "dataset_official": metadata.get("dataset_official"),
            "primary_metric": None,
            "primary_score": None,
            "evidence_positions": sample.get("evidence_positions") or [],
            "selected_tokens": {},
            "selected_tokens_by_layer": {},
            "evidence_recall": None,
            "evidence_precision": None,
            "score_stats": {},
            "score_normalization": self.cfg.eviction.score_normalization,
            "seed": self.cfg.seed,
            "score_update_count": 0,
            "max_kv_len": None,
            "final_kv_len": None,
            "cache_shape_summary": {},
            "total_time_s": 0.0,
            "prefill_time_s": 0.0,
            "decode_time_s": 0.0,
            "score_time_s": 0.0,
            "eviction_time_s": 0.0,
            "topk_time_s": 0.0,
            "cache_rebuild_time_s": 0.0,
            "tokens_per_second": None,
            "skipped": True,
            "skipped_reason": reason,
            "unsupported_reason": reason,
            "oracle": bool(getattr(spec, "oracle", False)),
            "metadata": metadata,
        }

    def error_result(
        self,
        sample: Dict[str, Any],
        sample_idx: int,
        method: str,
        budget: int,
        exc: Exception,
    ) -> Dict[str, Any]:
        prompt = sample.get("prompt")
        return {
            "label": f"{method}_b{budget}_s{sample_idx}",
            "sample_id": sample_idx,
            "sample_idx": sample_idx,
            "method": method,
            "budget": budget,
            "model": self.cfg.model.name,
            "model_name": self.cfg.model.name,
            "backend": "mlx",
            "quant_bits": self.cfg.model.quant_bits,
            "benchmark": self.cfg.benchmark.name,
            "prompt_hash": text_hash(prompt),
            "ground_truth": sample.get("ground_truth"),
            "evidence_positions": sample.get("evidence_positions"),
            "metadata": sample.get("metadata", {}),
            "error": str(exc),
        }
