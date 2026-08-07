"""Raw set, temporal-score, query/attention, and value-geometry signals."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.stats import spearmanr

from statekv.backend import AnchorState, ReferenceTrajectory, ScoreState
from statekv.selectors import (
    CoreSelection,
    LayerSelection,
    _pool_attention,
    mandatory_and_eligible,
    ridge_leverage,
)


def _finite(values: Sequence[float], label: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if not np.isfinite(result).all():
        raise FloatingPointError("%s contains NaN/Inf" % label)
    return result


def _cosine(left: np.ndarray, right: np.ndarray, epsilon: float = 1e-12) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= epsilon:
        return 1.0 if np.linalg.norm(left - right) <= epsilon else 0.0
    return float(np.dot(left, right) / denominator)


def _jaccard(left: Sequence[int], right: Sequence[int]) -> float:
    a, b = set(int(value) for value in left), set(int(value) for value in right)
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def candidate_layer_records(
    selection: CoreSelection,
    anchor: AnchorState,
    token_ids: Sequence[int],
    sink_size: int,
    recent_size: int,
) -> List[Dict[str, Any]]:
    output = []
    for layer, selected in selection.by_layer.items():
        positions = [int(value) for value in anchor.position_maps[layer].tolist()]
        score_by_position = {
            int(position): float(score)
            for position, score in zip(positions, selected.aggregate_scores)
            if math.isfinite(float(score))
        }
        eligible_scores = _finite(
            [score_by_position[position] for position in selected.eligible_positions],
            "candidate selector scores",
        )
        order = np.argsort(-eligible_scores, kind="stable")
        rank_by_position = {
            int(selected.eligible_positions[int(row)]): rank + 1
            for rank, row in enumerate(order.tolist())
        }
        sink, recent, _ = mandatory_and_eligible(
            positions, sink_size=sink_size, recent_size=recent_size
        )
        records = []
        selected_set = set(selected.selected_positions)
        for position in selected.eligible_positions:
            records.append(
                {
                    "position": int(position),
                    "token_id": (
                        int(token_ids[position]) if 0 <= position < len(token_ids) else None
                    ),
                    "age": int(anchor.logical_length - 1 - position),
                    "score": score_by_position[position],
                    "rank": rank_by_position[position],
                    "cache_role": "core" if position in selected_set else "unselected",
                }
            )
        mandatory_records = []
        for position in sorted(set(sink + recent)):
            roles = []
            if position in sink:
                roles.append("sink")
            if position in recent:
                roles.append("recent")
            mandatory_records.append(
                {
                    "position": int(position),
                    "token_id": (
                        int(token_ids[position])
                        if 0 <= position < len(token_ids)
                        else None
                    ),
                    "age": int(anchor.logical_length - 1 - position),
                    "cache_role": "+".join(roles),
                }
            )
        component_names = (
            set(selected.score_components)
            if selection.strategy
            == "attention_weighted_v_ridge_leverage"
            else (
                {"raw_observation_attention"}
                if selection.strategy == "snapkv"
                else set()
            )
        )
        position_to_row = {
            int(position): row for row, position in enumerate(positions)
        }
        score_components = {}
        for name in sorted(component_names):
            values = selected.score_components[name]
            score_components[name] = [
                (
                    None
                    if values[position_to_row[position]] is None
                    or not math.isfinite(
                        float(values[position_to_row[position]])
                    )
                    else float(values[position_to_row[position]])
                )
                for position in selected.eligible_positions
            ]
        output.append(
            {
                "layer": int(layer),
                "sink_positions": [int(value) for value in sink],
                "recent_positions": [int(value) for value in recent],
                "mandatory_token_records": mandatory_records,
                "selected_positions": list(selected.selected_positions),
                "selected_token_ids": [
                    int(token_ids[position])
                    for position in selected.selected_positions
                    if 0 <= position < len(token_ids)
                ],
                "boundary_margin": selected.boundary_margin,
                "eligible_token_records": records,
                "score_component_positions": list(
                    selected.eligible_positions
                ),
                "score_components": score_components,
                "ridge_parameters": selected.ridge_parameters,
                "metadata": {
                    **selected.metadata,
                    "per_kv_head_scores_persisted": False,
                    "selector_score_in_eligible_token_records": True,
                },
            }
        )
    return output


def _score_vector(
    state: ScoreState,
    layer: int,
    strategy: str,
    ridge_coefficient: float,
    ridge_mode: str,
    weighted_ridge_coefficient: float,
    attention_epsilon: float,
    pooling_kernel: int,
    pooling_mode: str,
    sink_size: int,
    recent_size: int,
    core_budget: int,
) -> Tuple[List[int], np.ndarray, Optional[float]]:
    positions = [int(value) for value in state.position_maps[layer].tolist()]
    _, _, eligible = mandatory_and_eligible(positions, sink_size, recent_size)
    row_by_position = {position: row for row, position in enumerate(positions)}
    eligible_rows = torch.tensor([row_by_position[value] for value in eligible])
    if strategy == "snapkv":
        raw = state.attention.observation_by_layer[layer].float()
        aggregate = _pool_attention(raw, pooling_kernel, pooling_mode)
        values = np.asarray(
            [float(aggregate[row_by_position[value]].item()) for value in eligible],
            dtype=np.float64,
        )
    elif strategy == "v_ridge_leverage":
        heads = state.values[layer][0].float()
        scored = []
        for head in range(int(heads.shape[0])):
            score, _ = ridge_leverage(
                heads[head].index_select(0, eligible_rows),
                ridge_coefficient,
                ridge_mode,
            )
            scored.append(score)
        values = torch.stack(scored).mean(dim=0).numpy().astype(np.float64)
    elif strategy == "attention_weighted_v_ridge_leverage":
        heads = state.values[layer][0].float()
        attention = state.attention.accumulated_by_layer[layer].float().index_select(
            1, eligible_rows
        )
        scored = []
        for head in range(int(heads.shape[0])):
            attn = attention[head].clamp_min(0.0)
            normalized = attn / max(float(attn.mean().item()), attention_epsilon)
            rows = heads[head].index_select(0, eligible_rows)
            score, _ = ridge_leverage(
                rows * torch.sqrt(normalized + attention_epsilon).unsqueeze(1),
                weighted_ridge_coefficient,
                ridge_mode,
            )
            scored.append(score)
        values = torch.stack(scored).mean(dim=0).numpy().astype(np.float64)
    else:
        raise ValueError("unsupported temporal score strategy: %s" % strategy)
    _finite(values, "temporal selector scores")
    take = min(len(values), max(0, len(values)))
    boundary = None
    if len(values) > 1:
        ordered = np.sort(values)[::-1]
        boundary_index = min(len(ordered) - 1, int(core_budget) - 1)
        if boundary_index + 1 < len(ordered):
            boundary = float(
                ordered[boundary_index] - ordered[boundary_index + 1]
            )
    return eligible, values, boundary


def score_drift_rows(
    reference: ReferenceTrajectory,
    selections: Mapping[str, CoreSelection],
    anchor_step: int,
    lags: Sequence[int],
    config: Any,
    base: Dict[str, Any],
) -> List[Dict[str, Any]]:
    rows = []
    anchor_state = reference.score_states.get(anchor_step)
    if anchor_state is None:
        return rows
    deployable = [
        "snapkv",
        "v_ridge_leverage",
        "attention_weighted_v_ridge_leverage",
    ]
    for strategy in deployable:
        for layer in reference.selected_layers:
            anchor_positions, anchor_values, anchor_margin = _score_vector(
                anchor_state,
                layer,
                strategy,
                config.selectors.ridge_lambda,
                config.selectors.ridge_lambda_mode,
                config.selectors.attention_weighted_ridge_lambda,
                config.selectors.attention_weight_epsilon,
                config.selectors.snapkv_pooling_kernel,
                config.selectors.snapkv_pooling,
                config.cache.sink_size,
                config.cache.recent_size,
                config.cache.selected_core_budget,
            )
            anchor_map = dict(zip(anchor_positions, anchor_values.tolist()))
            common_universe = list(anchor_positions)
            core_budget = min(
                int(config.cache.selected_core_budget), len(common_universe)
            )
            anchor_top = [
                common_universe[index]
                for index in np.argsort(-anchor_values, kind="stable")[:core_budget]
            ]
            for lag in lags:
                future_step = int(anchor_step + lag)
                future_state = reference.score_states.get(future_step)
                if future_state is None:
                    continue
                future_positions, future_values, future_margin = _score_vector(
                    future_state,
                    layer,
                    strategy,
                    config.selectors.ridge_lambda,
                    config.selectors.ridge_lambda_mode,
                    config.selectors.attention_weighted_ridge_lambda,
                    config.selectors.attention_weight_epsilon,
                    config.selectors.snapkv_pooling_kernel,
                    config.selectors.snapkv_pooling,
                    config.cache.sink_size,
                    config.cache.recent_size,
                    config.cache.selected_core_budget,
                )
                future_map = dict(zip(future_positions, future_values.tolist()))
                common = [
                    position for position in common_universe if position in future_map
                ]
                if len(common) < 2:
                    continue
                left = _finite([anchor_map[position] for position in common], "anchor score")
                right = _finite([future_map[position] for position in common], "future score")
                future_top = [
                    common[index]
                    for index in np.argsort(-right, kind="stable")[:core_budget]
                ]
                rank = spearmanr(left, right).statistic
                rows.append(
                    {
                        **base,
                        "anchor": int(anchor_step),
                        "strategy": strategy,
                        "signal_kind": "score_drift",
                        "layer": int(layer),
                        "head": None,
                        "lag": int(lag),
                        "normalized_l2_drift": float(
                            np.linalg.norm(right - left)
                            / max(np.linalg.norm(left), 1e-12)
                        ),
                        "cosine_similarity": _cosine(left, right),
                        "spearman_rank_correlation": (
                            float(rank) if math.isfinite(float(rank)) else None
                        ),
                        "top_core_jaccard": _jaccard(anchor_top, future_top),
                        "selection_boundary_margin_anchor": anchor_margin,
                        "selection_boundary_margin_future": future_margin,
                        "score_mean": float(right.mean()),
                        "score_std": float(right.std()),
                        "score_min": float(right.min()),
                        "score_max": float(right.max()),
                        "score_autocorrelation": float(
                            np.corrcoef(left, right)[0, 1]
                        )
                        if left.std() > 0 and right.std() > 0
                        else None,
                    }
                )
    return rows


def query_attention_rows(
    reference: ReferenceTrajectory,
    selections: Mapping[str, CoreSelection],
    anchor_step: int,
    lags: Sequence[int],
    base: Dict[str, Any],
    core_budget: int,
) -> List[Dict[str, Any]]:
    rows = []
    anchor_record = reference.query_records[anchor_step]
    anchor_length = reference.prompt_length + int(anchor_step)
    for lag in lags:
        index = int(anchor_step + lag)
        if index >= len(reference.query_records):
            continue
        future = reference.query_records[index]
        for key, anchor_query in anchor_record.queries.items():
            if key not in future.queries:
                continue
            layer, head = [int(value) for value in key.split(":")]
            q0 = anchor_query.numpy().astype(np.float64)
            q1 = future.queries[key].numpy().astype(np.float64)
            a0 = (
                anchor_record.attention_distributions[key][:anchor_length]
                .float()
                .numpy()
                .astype(np.float64)
            )
            a1 = (
                future.attention_distributions[key][:anchor_length]
                .float()
                .numpy()
                .astype(np.float64)
            )
            a0 = np.maximum(a0, 0.0)
            a1 = np.maximum(a1, 0.0)
            a0 /= max(a0.sum(), 1e-12)
            a1 /= max(a1.sum(), 1e-12)
            positive = a1[a1 > 0]
            entropy = float(-(positive * np.log(positive)).sum())
            rank = spearmanr(a0, a1).statistic
            query_common = {
                "query_cosine_to_anchor": _cosine(q0, q1),
                "query_norm_change": float(
                    (np.linalg.norm(q1) - np.linalg.norm(q0))
                    / max(np.linalg.norm(q0), 1e-12)
                ),
                "attention_entropy": entropy,
                "attention_top1_mass": float(np.sort(a1)[-1:].sum()),
                "attention_top8_mass": float(np.sort(a1)[-8:].sum()),
                "attention_distribution_cosine": _cosine(a0, a1),
                "attention_rank_correlation": (
                    float(rank) if math.isfinite(float(rank)) else None
                ),
            }
            top_count = min(int(core_budget), len(a0))
            top0 = np.argsort(-a0, kind="stable")[:top_count]
            top1 = np.argsort(-a1, kind="stable")[:top_count]
            for strategy, selection in selections.items():
                selected_positions = selection.by_layer[layer].selected_positions
                selected_valid = [
                    position
                    for position in selected_positions
                    if 0 <= position < len(a1)
                ]
                rows.append(
                    {
                        **base,
                        "anchor": int(anchor_step),
                        "strategy": strategy,
                        "signal_kind": "query_attention_drift",
                        "layer": layer,
                        "head": head,
                        "lag": int(lag),
                        **query_common,
                        "selected_core_attention_mass": float(
                            a1[selected_valid].sum()
                        )
                        if selected_valid
                        else 0.0,
                        "attention_top_core_overlap": _jaccard(
                            top0.tolist(), top1.tolist()
                        ),
                    }
                )
    return rows


def _spectrum(rows: torch.Tensor) -> Dict[str, Any]:
    values = rows.detach().float()
    singular = torch.linalg.svdvals(values)
    if not torch.isfinite(singular).all():
        raise FloatingPointError("singular values contain NaN/Inf")
    squared = singular.square()
    total = float(squared.sum().item())
    leading = float(squared.max().item()) if squared.numel() else 0.0
    stable_rank = total / max(leading, 1e-12)
    probabilities = singular / singular.sum().clamp_min(1e-12)
    positive = probabilities[probabilities > 0]
    effective_rank = float(torch.exp(-(positive * positive.log()).sum()).item())
    largest = float(singular.max().item()) if singular.numel() else 0.0
    floor = max(largest * 1e-6, 1e-12)
    condition = largest / max(float(singular.min().item()), floor)
    return {
        "stable_rank": stable_rank,
        "effective_rank": effective_rank,
        "leading_singular_values": singular[:8].tolist(),
        "condition_number_clipped": condition,
        "condition_clip_floor": floor,
        "condition_warning": bool(
            singular.numel() and float(singular.min().item()) < floor
        ),
    }


def _span_basis(rows: torch.Tensor) -> torch.Tensor:
    if rows.numel() == 0:
        return rows.new_zeros((0, int(rows.shape[-1])))
    _, singular, vh = torch.linalg.svd(rows.float(), full_matrices=False)
    largest = float(singular.max().item()) if singular.numel() else 0.0
    tolerance = max(rows.shape) * torch.finfo(torch.float32).eps * largest
    rank = int((singular > tolerance).sum().item())
    return vh[:rank]


def _relative_residual(rows: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    values = rows.float()
    reconstructed = (
        values @ basis.T @ basis if basis.numel() else torch.zeros_like(values)
    )
    return torch.linalg.vector_norm(values - reconstructed, dim=-1) / (
        torch.linalg.vector_norm(values, dim=-1) + 1e-8
    )


def geometry_rows(
    reference: ReferenceTrajectory,
    selections: Mapping[str, CoreSelection],
    anchor_step: int,
    base: Dict[str, Any],
    sink_size: int,
    recent_size: int,
) -> Tuple[List[Dict[str, Any]], Dict[Tuple[str, int], float]]:
    rows = []
    residual_by_strategy_step: Dict[Tuple[str, int], List[float]] = {}
    anchor = reference.anchors[anchor_step]
    for strategy, selection in selections.items():
        for layer in reference.selected_layers:
            positions = [int(value) for value in anchor.position_maps[layer].tolist()]
            row_by_position = {position: row for row, position in enumerate(positions)}
            _, _, eligible = mandatory_and_eligible(
                positions, sink_size, recent_size
            )
            eligible_rows = torch.tensor([row_by_position[value] for value in eligible])
            selected_positions = selection.by_layer[layer].selected_positions
            selected_rows = torch.tensor(
                [row_by_position[value] for value in selected_positions]
            )
            unselected_positions = [
                value for value in eligible if value not in set(selected_positions)
            ]
            unselected_rows = torch.tensor(
                [row_by_position[value] for value in unselected_positions],
                dtype=torch.long,
            )
            values = anchor.values[layer][0].float()
            for kv_head in range(int(values.shape[0])):
                selectable = values[kv_head].index_select(0, eligible_rows)
                selected = values[kv_head].index_select(0, selected_rows)
                basis = _span_basis(selected)
                selectable_residual = _relative_residual(selectable, basis)
                unselected_residual = (
                    _relative_residual(
                        values[kv_head].index_select(0, unselected_rows), basis
                    )
                    if unselected_rows.numel()
                    else torch.empty(0)
                )
                normalized = selected / (
                    torch.linalg.vector_norm(selected, dim=-1, keepdim=True) + 1e-8
                )
                cosine = normalized @ normalized.T
                off_diagonal = cosine[
                    ~torch.eye(int(cosine.shape[0]), dtype=torch.bool)
                ]
                spectrum = _spectrum(selectable)
                rows.append(
                    {
                        **base,
                        "anchor": int(anchor_step),
                        "strategy": strategy,
                        "signal_kind": "value_geometry",
                        "layer": int(layer),
                        "head": int(kv_head),
                        "lag": 0,
                        **spectrum,
                        "selected_span_reconstruction_residual_mean": float(
                            selectable_residual.mean().item()
                        ),
                        "unselected_residual_mean": (
                            float(unselected_residual.mean().item())
                            if unselected_residual.numel()
                            else None
                        ),
                        "selected_pairwise_cosine_mean": (
                            float(off_diagonal.mean().item())
                            if off_diagonal.numel()
                            else None
                        ),
                        "selected_pairwise_cosine_std": (
                            float(off_diagonal.std(unbiased=False).item())
                            if off_diagonal.numel()
                            else None
                        ),
                        "selected_core_diversity": (
                            float(1.0 - off_diagonal.abs().mean().item())
                            if off_diagonal.numel()
                            else None
                        ),
                    }
                )
                for future_step in range(
                    1, len(reference.generated_token_ids) - anchor_step + 1
                ):
                    record_index = anchor_step + future_step
                    if record_index >= len(reference.query_records):
                        break
                    key = "%d:%d" % (layer, kv_head)
                    new_value = reference.query_records[record_index].new_values.get(key)
                    if new_value is None:
                        continue
                    residual = float(
                        _relative_residual(new_value.unsqueeze(0), basis)[0].item()
                    )
                    residual_by_strategy_step.setdefault(
                        (strategy, future_step), []
                    ).append(residual)
                    rows.append(
                        {
                            **base,
                            "anchor": int(anchor_step),
                            "strategy": strategy,
                            "signal_kind": "future_new_token_value_residual",
                            "layer": int(layer),
                            "head": int(kv_head),
                            "lag": int(future_step),
                            "future_new_token_residual": residual,
                        }
                    )
    means = {
        key: float(np.mean(values))
        for key, values in residual_by_strategy_step.items()
        if values
    }
    return rows, means
