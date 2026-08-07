"""Cache-core selectors and stable value-geometry helpers."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from kvbench.cache.budget import stable_topk
from statekv.config import DiscoveryConfig
from kvbench.types import AttentionSignals, CacheSnapshot


@dataclass
class LayerSelection:
    layer: int
    selected_positions: List[int]
    eligible_positions: List[int]
    aggregate_scores: List[float]
    score_components: Dict[str, List[float]] = field(default_factory=dict)
    scores_by_kv_head: Dict[str, List[List[float]]] = field(default_factory=dict)
    boundary_margin: Optional[float] = None
    ridge_parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CoreSelection:
    strategy: str
    horizon_condition: Optional[int]
    by_layer: Dict[int, LayerSelection]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OnlineRidgeFactor:
    """Reusable Cholesky factor for v^T(V^TV + lambda I)^-1 v."""

    cholesky: torch.Tensor
    ridge: float
    diagnostics: Dict[str, Any]

    def score(self, vectors: torch.Tensor) -> torch.Tensor:
        values = vectors.detach().to(
            dtype=self.cholesky.dtype, device=self.cholesky.device
        )
        squeeze = values.ndim == 1
        if squeeze:
            values = values.unsqueeze(0)
        if values.ndim != 2 or int(values.shape[-1]) != int(
            self.cholesky.shape[-1]
        ):
            raise ValueError("online ridge vectors must have [token, feature]")
        solved = torch.cholesky_solve(values.T, self.cholesky).T
        scores = (values * solved).sum(dim=-1).clamp_min(0.0)
        if not torch.isfinite(scores).all():
            raise FloatingPointError("online ridge leverage contains NaN/Inf")
        return scores[0] if squeeze else scores


def mandatory_and_eligible(
    positions: Sequence[int],
    sink_size: int,
    recent_size: int,
) -> Tuple[List[int], List[int], List[int]]:
    normalized = [int(value) for value in positions]
    sink = normalized[: min(len(normalized), int(sink_size))]
    recent = normalized[max(0, len(normalized) - int(recent_size)) :]
    mandatory = sorted(set(sink + recent))
    mandatory_set = set(mandatory)
    eligible = [position for position in normalized if position not in mandatory_set]
    return sink, recent, eligible


def _relative_ridge(rows: torch.Tensor, coefficient: float, mode: str) -> float:
    if mode == "absolute":
        return max(float(coefficient), 1e-12)
    trace = float((rows * rows).sum().item())
    dim = max(1, int(rows.shape[-1]))
    return max(float(coefficient) * max(trace / dim, 1e-12), 1e-12)


def ridge_leverage(
    rows: torch.Tensor,
    coefficient: float,
    mode: str = "relative",
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Compute ridge row leverage using a stable Hermitian eigensolve."""
    values = rows.detach().float()
    if values.ndim != 2:
        raise ValueError("ridge leverage expects [token, feature]")
    n, dim = values.shape
    if n == 0:
        return values.new_zeros((0,)), {
            "ridge": None,
            "effective_dimension": 0.0,
            "condition_number": None,
            "condition_warning": False,
        }
    if not torch.isfinite(values).all():
        raise FloatingPointError("ridge leverage input contains NaN/Inf")
    ridge = _relative_ridge(values, coefficient, mode)
    gram = values.T @ values
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    eigenvalues = eigenvalues.clamp_min(0.0)
    inv_sqrt = torch.rsqrt(eigenvalues + ridge)
    transform = eigenvectors * inv_sqrt.unsqueeze(0)
    whitened = values @ transform
    scores = (whitened * whitened).sum(dim=-1).clamp_min(0.0)
    if not torch.isfinite(scores).all():
        raise FloatingPointError("ridge leverage output contains NaN/Inf")
    largest = float(eigenvalues.max().item()) if eigenvalues.numel() else 0.0
    smallest = float(eigenvalues.min().item()) if eigenvalues.numel() else 0.0
    condition = (largest + ridge) / max(smallest + ridge, 1e-30)
    return scores, {
        "calculation": "ridge_eigh_no_inverse",
        "ridge": float(ridge),
        "ridge_coefficient": float(coefficient),
        "ridge_mode": mode,
        "effective_dimension": float(scores.sum().item()),
        "regularized_condition_number": float(condition),
        "condition_warning": bool(not math.isfinite(condition) or condition > 1e8),
    }


def fit_online_ridge_factor(
    rows: torch.Tensor,
    coefficient: float,
    mode: str = "relative",
) -> OnlineRidgeFactor:
    """Factor V^TV + lambda I once; never form an explicit inverse."""

    values = rows.detach().to(dtype=torch.float64, device="cpu")
    if values.ndim != 2 or not len(values):
        raise ValueError("online ridge history must be non-empty [token, feature]")
    if not torch.isfinite(values).all():
        raise FloatingPointError("online ridge history contains NaN/Inf")
    ridge = _relative_ridge(values, coefficient, mode)
    gram = values.T @ values
    regularized = gram + float(ridge) * torch.eye(
        int(values.shape[-1]), dtype=values.dtype
    )
    factor, info = torch.linalg.cholesky_ex(regularized)
    if int(info.max().item()) != 0:
        raise FloatingPointError("online ridge Cholesky factorization failed")
    eigenvalues = torch.linalg.eigvalsh(gram).clamp_min(0.0)
    largest = float(eigenvalues.max().item())
    smallest = float(eigenvalues.min().item())
    condition = (largest + ridge) / max(smallest + ridge, 1e-30)
    return OnlineRidgeFactor(
        cholesky=factor,
        ridge=float(ridge),
        diagnostics={
            "calculation": "cholesky_solve_no_inverse",
            "history_rows": int(values.shape[0]),
            "feature_dimension": int(values.shape[1]),
            "ridge": float(ridge),
            "ridge_coefficient": float(coefficient),
            "ridge_mode": mode,
            "regularized_condition_number": float(condition),
            "condition_warning": bool(
                not math.isfinite(condition) or condition > 1e8
            ),
        },
    )


def _pool_attention(
    scores_by_head: torch.Tensor,
    kernel: int,
    mode: str,
) -> torch.Tensor:
    if scores_by_head.ndim != 2:
        raise ValueError("SnapKV scores must have [kv_head, token]")
    from src.runners.mlx_runner import snapkv_pool_scores_numpy

    shared = scores_by_head.detach().float().mean(dim=0).cpu().numpy()
    pooled = snapkv_pool_scores_numpy(shared, int(kernel), mode)
    return torch.from_numpy(pooled)


def _top_positions(
    aggregate: torch.Tensor,
    all_positions: Sequence[int],
    eligible_positions: Sequence[int],
    budget: int,
) -> Tuple[List[int], Optional[float]]:
    position_to_row = {int(position): row for row, position in enumerate(all_positions)}
    eligible_rows = torch.tensor(
        [position_to_row[int(position)] for position in eligible_positions],
        dtype=torch.long,
        device=aggregate.device,
    )
    take = min(max(0, int(budget)), len(eligible_positions))
    rows = stable_topk(aggregate, take, eligible_rows)
    selected = sorted(int(all_positions[int(row)]) for row in rows.tolist())
    if take <= 0 or len(eligible_positions) <= take:
        return selected, None
    eligible_scores = aggregate.index_select(0, eligible_rows).detach().float()
    ordered = torch.sort(eligible_scores, descending=True, stable=True).values
    margin = float((ordered[take - 1] - ordered[take]).item())
    return selected, margin


def _scores_full(
    length: int,
    eligible_rows: torch.Tensor,
    eligible_scores: torch.Tensor,
) -> torch.Tensor:
    result = torch.full(
        (length,), float("nan"), dtype=torch.float32, device=eligible_scores.device
    )
    result.index_copy_(0, eligible_rows.to(result.device), eligible_scores.float())
    return result


class CoreSelector:
    """Select shared token cores per layer while retaining per-head diagnostics."""

    def __init__(self, cfg: DiscoveryConfig):
        self.cfg = cfg

    def select(
        self,
        snapshot: CacheSnapshot,
        strategy: str,
        future_attention: Optional[Dict[int, torch.Tensor]] = None,
        horizon: Optional[int] = None,
    ) -> CoreSelection:
        by_layer: Dict[int, LayerSelection] = {}
        for layer, value in enumerate(snapshot.values):
            positions = [
                int(item) for item in snapshot.position_maps[layer].detach().cpu().tolist()
            ]
            _, _, eligible_positions = mandatory_and_eligible(
                positions, self.cfg.cache.sink_size, self.cfg.cache.recent_size
            )
            position_to_row = {position: row for row, position in enumerate(positions)}
            eligible_rows = torch.tensor(
                [position_to_row[position] for position in eligible_positions],
                dtype=torch.long,
            )
            if strategy == "snapkv":
                layer_selection = self._snapkv(
                    snapshot.attention, layer, positions, eligible_positions
                )
            elif strategy == "v_ridge_leverage":
                layer_selection = self._v_ridge(
                    value, layer, positions, eligible_positions, eligible_rows
                )
            elif strategy == "attention_weighted_v_ridge_leverage":
                layer_selection = self._attention_weighted(
                    snapshot.attention,
                    value,
                    layer,
                    positions,
                    eligible_positions,
                    eligible_rows,
                )
            elif strategy == "future_attention_oracle":
                if future_attention is None or layer not in future_attention:
                    raise ValueError("future oracle attention is missing for layer=%d" % layer)
                layer_selection = self._oracle(
                    future_attention[layer],
                    layer,
                    positions,
                    eligible_positions,
                )
            else:
                raise ValueError("unknown discovery strategy: %s" % strategy)
            by_layer[layer] = layer_selection
        return CoreSelection(
            strategy=strategy,
            horizon_condition=horizon,
            by_layer=by_layer,
            metadata={
                "shared_token_selection": True,
                "selection_granularity": "per_layer_shared_across_kv_heads",
            },
        )

    def _snapkv(
        self,
        attention: AttentionSignals,
        layer: int,
        positions: List[int],
        eligible_positions: List[int],
    ) -> LayerSelection:
        raw = attention.observation_by_layer.get(layer)
        if raw is None or int(raw.shape[-1]) != len(positions):
            raise ValueError("SnapKV observation attention is unavailable at layer=%d" % layer)
        raw = raw.detach().float().cpu()
        pooled = _pool_attention(
            raw,
            self.cfg.selectors.snapkv_pooling_kernel,
            self.cfg.selectors.snapkv_pooling,
        )
        aggregate = pooled
        selected, margin = _top_positions(
            aggregate,
            positions,
            eligible_positions,
            self.cfg.cache.selected_core_budget,
        )
        return LayerSelection(
            layer=layer,
            selected_positions=selected,
            eligible_positions=eligible_positions,
            aggregate_scores=aggregate.tolist(),
            score_components={"raw_observation_attention": raw.mean(dim=0).tolist()},
            scores_by_kv_head={
                "raw_observation_attention": raw.tolist(),
                "shared_pooled_observation_attention": pooled.tolist(),
            },
            boundary_margin=margin,
            metadata={
                "observation_window": int(self.cfg.selectors.observation_window),
                "attention_aggregation": (
                    "sum_queries_then_mean_query_and_kv_heads_before_pooling"
                ),
                "pooling_implementation": (
                    "src.runners.mlx_runner.snapkv_pool_scores_numpy"
                ),
                "pooling": self.cfg.selectors.snapkv_pooling,
                "pooling_kernel": int(
                    min(self.cfg.selectors.snapkv_pooling_kernel, len(positions))
                ),
                "per_layer": True,
                "per_head_selection": False,
            },
        )

    def _v_ridge(
        self,
        value: torch.Tensor,
        layer: int,
        positions: List[int],
        eligible_positions: List[int],
        eligible_rows: torch.Tensor,
    ) -> LayerSelection:
        values = value.detach()[0].float().cpu()
        head_scores = []
        diagnostics = []
        for head in range(int(values.shape[0])):
            score, diagnostic = ridge_leverage(
                values[head].index_select(0, eligible_rows),
                self.cfg.selectors.ridge_lambda,
                self.cfg.selectors.ridge_lambda_mode,
            )
            head_scores.append(score)
            diagnostics.append(diagnostic)
        by_head_eligible = torch.stack(head_scores, dim=0)
        eligible_score = by_head_eligible.mean(dim=0)
        full = _scores_full(len(positions), eligible_rows, eligible_score)
        selected, margin = _top_positions(
            full, positions, eligible_positions, self.cfg.cache.selected_core_budget
        )
        return LayerSelection(
            layer=layer,
            selected_positions=selected,
            eligible_positions=eligible_positions,
            aggregate_scores=full.tolist(),
            score_components={"raw_leverage": full.tolist()},
            scores_by_kv_head={"raw_leverage": by_head_eligible.tolist()},
            boundary_margin=margin,
            ridge_parameters={"kv_head_diagnostics": diagnostics},
            metadata={
                "value_matrix_scope": "selectable_history_only",
                "aggregation": "mean_after_independent_kv_head_scoring",
            },
        )

    def _attention_weighted(
        self,
        attention: AttentionSignals,
        value: torch.Tensor,
        layer: int,
        positions: List[int],
        eligible_positions: List[int],
        eligible_rows: torch.Tensor,
    ) -> LayerSelection:
        accumulated = attention.accumulated_by_layer.get(layer)
        if accumulated is None or int(accumulated.shape[-1]) != len(positions):
            raise ValueError(
                "accumulated causal attention is unavailable at layer=%d" % layer
            )
        raw_attention = accumulated.detach().float().cpu().index_select(
            1, eligible_rows
        )
        values = value.detach()[0].float().cpu()
        weighted_scores = []
        raw_leverage_scores = []
        normalized_attention = []
        normalized_leverage = []
        diagnostics = []
        epsilon = float(self.cfg.selectors.attention_weight_epsilon)
        for head in range(int(values.shape[0])):
            attn = raw_attention[head].clamp_min(0.0)
            attn_normalized = attn / max(float(attn.mean().item()), epsilon)
            rows = values[head].index_select(0, eligible_rows)
            raw_leverage, raw_diag = ridge_leverage(
                rows,
                self.cfg.selectors.attention_weighted_ridge_lambda,
                self.cfg.selectors.ridge_lambda_mode,
            )
            leverage_scale = max(float(raw_leverage.mean().item()), epsilon)
            leverage_normalized = raw_leverage / leverage_scale
            weighted_rows = rows * torch.sqrt(attn_normalized + epsilon).unsqueeze(1)
            hybrid, hybrid_diag = ridge_leverage(
                weighted_rows,
                self.cfg.selectors.attention_weighted_ridge_lambda,
                self.cfg.selectors.ridge_lambda_mode,
            )
            weighted_scores.append(hybrid)
            raw_leverage_scores.append(raw_leverage)
            normalized_attention.append(attn_normalized)
            normalized_leverage.append(leverage_normalized)
            diagnostics.append(
                {"raw_v_ridge": raw_diag, "attention_weighted_v_ridge": hybrid_diag}
            )
        hybrid_by_head = torch.stack(weighted_scores, dim=0)
        raw_leverage_by_head = torch.stack(raw_leverage_scores, dim=0)
        norm_attention_by_head = torch.stack(normalized_attention, dim=0)
        norm_leverage_by_head = torch.stack(normalized_leverage, dim=0)
        hybrid_eligible = hybrid_by_head.mean(dim=0)
        hybrid_full = _scores_full(len(positions), eligible_rows, hybrid_eligible)
        raw_attn_full = _scores_full(
            len(positions), eligible_rows, raw_attention.mean(dim=0)
        )
        raw_lev_full = _scores_full(
            len(positions), eligible_rows, raw_leverage_by_head.mean(dim=0)
        )
        norm_attn_full = _scores_full(
            len(positions), eligible_rows, norm_attention_by_head.mean(dim=0)
        )
        norm_lev_full = _scores_full(
            len(positions), eligible_rows, norm_leverage_by_head.mean(dim=0)
        )
        selected, margin = _top_positions(
            hybrid_full,
            positions,
            eligible_positions,
            self.cfg.cache.selected_core_budget,
        )
        return LayerSelection(
            layer=layer,
            selected_positions=selected,
            eligible_positions=eligible_positions,
            aggregate_scores=hybrid_full.tolist(),
            score_components={
                "raw_attention": raw_attn_full.tolist(),
                "raw_leverage": raw_lev_full.tolist(),
                "normalized_attention": norm_attn_full.tolist(),
                "normalized_leverage": norm_lev_full.tolist(),
                "hybrid_score": hybrid_full.tolist(),
            },
            scores_by_kv_head={
                "raw_attention": raw_attention.tolist(),
                "raw_leverage": raw_leverage_by_head.tolist(),
                "normalized_attention": norm_attention_by_head.tolist(),
                "normalized_leverage": norm_leverage_by_head.tolist(),
                "hybrid_score": hybrid_by_head.tolist(),
            },
            boundary_margin=margin,
            ridge_parameters={"kv_head_diagnostics": diagnostics},
            metadata={
                "formula": (
                    "ridge_leverage(diag(attention/mean(attention)+epsilon)^0.5 V)"
                ),
                "attention_epsilon": epsilon,
                "attention_source": "all_prefill_and_decode_causal_attention",
                "aggregation": "mean_after_independent_kv_head_scoring",
            },
        )

    def _oracle(
        self,
        future: torch.Tensor,
        layer: int,
        positions: List[int],
        eligible_positions: List[int],
    ) -> LayerSelection:
        if future.ndim != 2 or int(future.shape[-1]) < len(positions):
            raise ValueError("future attention has invalid shape at layer=%d" % layer)
        by_head = future.detach().float().cpu()[:, : len(positions)].clamp_min(0.0)
        aggregate = by_head.mean(dim=0)
        selected, margin = _top_positions(
            aggregate,
            positions,
            eligible_positions,
            self.cfg.cache.selected_core_budget,
        )
        return LayerSelection(
            layer=layer,
            selected_positions=selected,
            eligible_positions=eligible_positions,
            aggregate_scores=aggregate.tolist(),
            score_components={"future_attention": aggregate.tolist()},
            scores_by_kv_head={"future_attention": by_head.tolist()},
            boundary_margin=margin,
            metadata={
                "attention_aggregation": "sum_future_queries_then_mean_kv_heads",
                "anchor_history_only": True,
            },
        )


def selection_overlap(left: CoreSelection, right: CoreSelection) -> Dict[str, Any]:
    layers = sorted(set(left.by_layer) & set(right.by_layer))
    records = []
    for layer in layers:
        a = set(left.by_layer[layer].selected_positions)
        b = set(right.by_layer[layer].selected_positions)
        union = a | b
        records.append(
            {
                "layer": int(layer),
                "intersection": len(a & b),
                "jaccard": len(a & b) / len(union) if union else 1.0,
                "left_retention_ratio": len(a & b) / len(a) if a else 1.0,
                "right_retention_ratio": len(a & b) / len(b) if b else 1.0,
            }
        )
    return {
        "by_layer": records,
        "mean_jaccard": (
            float(np.mean([row["jaccard"] for row in records])) if records else None
        ),
        "mean_left_retention_ratio": (
            float(np.mean([row["left_retention_ratio"] for row in records]))
            if records
            else None
        ),
    }
