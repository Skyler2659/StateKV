"""Functional feature maps and ridge-coverage measurements for Stage 1."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch

from statekv.backend import AnchorState


FeatureKey = Tuple[str, str, Optional[int]]


def _relative_ridge(
    rows: torch.Tensor, coefficient: float, mode: str
) -> float:
    if mode == "absolute":
        return max(float(coefficient), 1e-12)
    trace = float((rows * rows).sum().item())
    dimension = max(1, int(rows.shape[-1]))
    return max(
        float(coefficient) * max(trace / dimension, 1e-12),
        1e-12,
    )


@dataclass
class RidgeCoverageFactor:
    """Stable primal/dual ridge row-span projection without an inverse."""

    history: torch.Tensor
    factor: Optional[torch.Tensor]
    ridge: float
    calculation: str
    regularized_condition_number: Optional[float]

    @classmethod
    def fit(
        cls,
        history: torch.Tensor,
        coefficient: float,
        mode: str = "relative",
    ) -> "RidgeCoverageFactor":
        values = history.detach().to(dtype=torch.float64, device="cpu")
        if values.ndim != 2:
            raise ValueError("ridge coverage history must be [token, feature]")
        if not torch.isfinite(values).all():
            raise FloatingPointError("ridge coverage history contains NaN/Inf")
        if int(values.shape[0]) == 0:
            return cls(
                history=values,
                factor=None,
                ridge=max(float(coefficient), 1e-12),
                calculation="empty_history_zero_projection",
                regularized_condition_number=None,
            )
        ridge = _relative_ridge(values, coefficient, mode)
        rows, dimension = map(int, values.shape)
        if dimension <= rows:
            gram = values.T @ values
            regularized = gram + ridge * torch.eye(
                dimension, dtype=torch.float64
            )
            factor, info = torch.linalg.cholesky_ex(regularized)
            calculation = "primal_cholesky_solve"
            eigenvalues = torch.linalg.eigvalsh(gram).clamp_min(0.0)
        else:
            gram = values @ values.T
            regularized = gram + ridge * torch.eye(
                rows, dtype=torch.float64
            )
            factor, info = torch.linalg.cholesky_ex(regularized)
            calculation = "dual_cholesky_solve"
            eigenvalues = torch.linalg.eigvalsh(gram).clamp_min(0.0)
        if int(info.max().item()) != 0:
            raise FloatingPointError("ridge coverage Cholesky failed")
        largest = float(eigenvalues.max().item())
        smallest = float(eigenvalues.min().item())
        condition = (largest + ridge) / max(smallest + ridge, 1e-30)
        return cls(
            history=values,
            factor=factor,
            ridge=float(ridge),
            calculation=calculation,
            regularized_condition_number=float(condition),
        )

    def project(self, vectors: torch.Tensor) -> torch.Tensor:
        values = vectors.detach().to(dtype=torch.float64, device="cpu")
        if values.ndim != 2 or int(values.shape[1]) != int(
            self.history.shape[1]
        ):
            raise ValueError("ridge coverage vectors do not match feature dimension")
        if self.factor is None:
            return torch.zeros_like(values)
        if self.calculation.startswith("primal"):
            solved = torch.cholesky_solve(values.T, self.factor).T
            return values - self.ridge * solved
        coefficients = torch.cholesky_solve(
            self.history @ values.T, self.factor
        )
        return coefficients.T @ self.history

    def residuals(
        self, vectors: torch.Tensor, epsilon: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        values = vectors.detach().to(dtype=torch.float64, device="cpu")
        residual = values - self.project(values)
        residual_sq = (residual * residual).sum(dim=-1).clamp_min(0.0)
        energy = (values * values).sum(dim=-1).clamp_min(0.0)
        normalized = residual_sq / (energy + float(epsilon))
        if not (
            torch.isfinite(residual_sq).all()
            and torch.isfinite(normalized).all()
        ):
            raise FloatingPointError("ridge coverage residual is NaN/Inf")
        return residual_sq, normalized, energy

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "ridge": float(self.ridge),
            "ridge_calculation": self.calculation,
            "ridge_history_rows": int(self.history.shape[0]),
            "feature_dimension": int(self.history.shape[1]),
            "regularized_condition_number": (
                self.regularized_condition_number
            ),
            "condition_warning": bool(
                self.regularized_condition_number is not None
                and (
                    not math.isfinite(self.regularized_condition_number)
                    or self.regularized_condition_number > 1e8
                )
            ),
        }


@dataclass
class LayerFeatureMatrices:
    layer: int
    positions: List[int]
    matrices: Dict[FeatureKey, torch.Tensor]
    observation_window_queries: int
    observation_weight_source: str


def build_layer_features(
    model: Any,
    anchor: AnchorState,
    layer: int,
    diagnostic_heads: Sequence[int],
    variants: Iterable[str],
) -> LayerFeatureMatrices:
    """Build exact Raw-V/OV/AOV/AOR rows for one layer and one anchor."""

    requested = set(str(value) for value in variants)
    values = anchor.values[int(layer)].detach()[0].float().cpu()
    positions = [
        int(value)
        for value in anchor.position_maps[int(layer)].detach().cpu().tolist()
    ]
    if int(values.shape[1]) != len(positions):
        raise ValueError("feature values and positions are not aligned")
    kv_heads, _, head_dim = map(int, values.shape)
    query_heads = int(model.model_info["num_attention_heads"])
    if query_heads % kv_heads:
        raise ValueError("query/KV head counts are not GQA-compatible")
    group = query_heads // kv_heads
    query_observation = anchor.query_head_observation.get(int(layer))
    if query_observation is None:
        kv_observation = anchor.attention.observation_by_layer.get(int(layer))
        if kv_observation is None:
            raise ValueError(
                "attention observation is unavailable at layer=%d" % layer
            )
        query_observation = (
            kv_observation.detach().float().cpu().repeat_interleave(
                group, dim=0
            )
        )
        source = "kv_head_group_mean_fallback"
    else:
        query_observation = query_observation.detach().float().cpu()
        source = "exact_query_head_mean_over_retained_observation_window"
    if tuple(query_observation.shape) != (query_heads, len(positions)):
        raise ValueError(
            "query-head observation shape mismatch at layer=%d: %s"
            % (layer, tuple(query_observation.shape))
        )
    weights = query_observation.clamp_min(0.0)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-30)
    sqrt_weights = torch.sqrt(weights)
    mapped_values = values.repeat_interleave(group, dim=0)
    means = torch.einsum("hn,hnd->hd", weights, mapped_values)
    layer_raw = mapped_values.permute(1, 0, 2).reshape(
        len(positions), query_heads * head_dim
    )
    layer_aov = (
        sqrt_weights.unsqueeze(-1) * mapped_values
    ).permute(1, 0, 2).reshape(len(positions), query_heads * head_dim)
    layer_aor = (
        sqrt_weights.unsqueeze(-1)
        * (mapped_values - means.unsqueeze(1))
    ).permute(1, 0, 2).reshape(len(positions), query_heads * head_dim)
    matrices: Dict[FeatureKey, torch.Tensor] = {}
    if "raw_v" in requested:
        matrices[("raw_v", "layer", None)] = layer_raw
    if "projected_v" in requested:
        matrices[("projected_v", "layer", None)] = model.project_features(
            int(layer), layer_raw
        )
    if "aov" in requested:
        matrices[("aov", "layer", None)] = model.project_features(
            int(layer), layer_aov
        )
    if "aor" in requested:
        matrices[("aor", "layer", None)] = model.project_features(
            int(layer), layer_aor
        )
    for head in diagnostic_heads:
        head = int(head)
        if head < 0 or head >= query_heads:
            raise ValueError("diagnostic query head is out of range")
        raw = mapped_values[head]
        aov = sqrt_weights[head].unsqueeze(-1) * raw
        aor = sqrt_weights[head].unsqueeze(-1) * (
            raw - means[head].unsqueeze(0)
        )
        if "raw_v" in requested:
            matrices[("raw_v", "diagnostic_head", head)] = raw
        if "projected_v" in requested:
            matrices[
                ("projected_v", "diagnostic_head", head)
            ] = model.project_features(int(layer), raw, head=head)
        if "aov" in requested:
            matrices[("aov", "diagnostic_head", head)] = (
                model.project_features(int(layer), aov, head=head)
            )
        if "aor" in requested:
            matrices[("aor", "diagnostic_head", head)] = (
                model.project_features(int(layer), aor, head=head)
            )
    return LayerFeatureMatrices(
        layer=int(layer),
        positions=positions,
        matrices=matrices,
        observation_window_queries=int(
            model.cfg.selectors.observation_window
        ),
        observation_weight_source=source,
    )


def _rows_for_positions(
    matrix: torch.Tensor,
    all_positions: Sequence[int],
    selected_positions: Iterable[int],
) -> torch.Tensor:
    row_by_position = {
        int(position): row for row, position in enumerate(all_positions)
    }
    rows = [
        row_by_position[int(position)]
        for position in sorted(set(int(value) for value in selected_positions))
        if int(position) in row_by_position
    ]
    if not rows:
        return matrix.new_zeros((0, int(matrix.shape[1])))
    return matrix.index_select(0, torch.tensor(rows, dtype=torch.long))


def _summarize(
    residual: torch.Tensor,
    normalized: torch.Tensor,
    energy: torch.Tensor,
) -> Dict[str, float]:
    return {
        "raw_sum": float(residual.sum().item()),
        "normalized_sum": float(normalized.sum().item()),
        "energy_sum": float(energy.sum().item()),
        "energy_normalized_sum": float(
            residual.sum().item() / max(float(energy.sum().item()), 1e-30)
        ),
        "mean": float(residual.mean().item()) if residual.numel() else 0.0,
        "normalized_mean": (
            float(normalized.mean().item()) if normalized.numel() else 0.0
        ),
        "token_count": int(residual.numel()),
    }


def functional_measurement(
    *,
    base_features: LayerFeatureMatrices,
    current_features: LayerFeatureMatrices,
    key: FeatureKey,
    old_history_positions: Iterable[int],
    fresh_history_positions: Iterable[int],
    base_old_history_positions: Iterable[int],
    epsilon: float,
    ridge_coefficient: float,
    ridge_mode: str,
) -> Dict[str, Any]:
    """Measure full-history coverage, D_new, D_rew and old-fresh Delta E."""

    base_matrix = base_features.matrices[key]
    current_matrix = current_features.matrices[key]
    old_history = _rows_for_positions(
        current_matrix,
        current_features.positions,
        old_history_positions,
    )
    fresh_history = _rows_for_positions(
        current_matrix,
        current_features.positions,
        fresh_history_positions,
    )
    base_history = _rows_for_positions(
        base_matrix,
        base_features.positions,
        base_old_history_positions,
    )
    old_factor = RidgeCoverageFactor.fit(
        old_history, ridge_coefficient, ridge_mode
    )
    fresh_factor = RidgeCoverageFactor.fit(
        fresh_history, ridge_coefficient, ridge_mode
    )
    base_factor = RidgeCoverageFactor.fit(
        base_history, ridge_coefficient, ridge_mode
    )
    old_residual, old_normalized, energy = old_factor.residuals(
        current_matrix, epsilon
    )
    fresh_residual, fresh_normalized, _ = fresh_factor.residuals(
        current_matrix, epsilon
    )
    current_row = {
        int(position): row
        for row, position in enumerate(current_features.positions)
    }
    base_row = {
        int(position): row
        for row, position in enumerate(base_features.positions)
    }
    boundary = max(base_features.positions)
    new_rows = [
        current_row[position]
        for position in current_features.positions
        if position > boundary
    ]
    common_old_positions = [
        position
        for position in base_features.positions
        if position in current_row
    ]
    common_current_rows = torch.tensor(
        [current_row[position] for position in common_old_positions],
        dtype=torch.long,
    )
    common_base_rows = torch.tensor(
        [base_row[position] for position in common_old_positions],
        dtype=torch.long,
    )
    if new_rows:
        new_index = torch.tensor(new_rows, dtype=torch.long)
        new_summary = _summarize(
            old_residual.index_select(0, new_index),
            old_normalized.index_select(0, new_index),
            energy.index_select(0, new_index),
        )
        arrival_residual, arrival_normalized, arrival_energy = (
            base_factor.residuals(
                current_matrix.index_select(0, new_index), epsilon
            )
        )
        arrival_summary = _summarize(
            arrival_residual, arrival_normalized, arrival_energy
        )
    else:
        empty = old_residual.new_zeros((0,))
        new_summary = _summarize(empty, empty, empty)
        arrival_summary = _summarize(empty, empty, empty)
    base_old_residual, base_old_normalized, base_energy = (
        base_factor.residuals(base_matrix, epsilon)
    )
    if common_old_positions:
        current_common = old_residual.index_select(
            0, common_current_rows
        )
        base_common = base_old_residual.index_select(0, common_base_rows)
        reweight = torch.abs(current_common - base_common)
        current_normalized_common = old_normalized.index_select(
            0, common_current_rows
        )
        base_normalized_common = base_old_normalized.index_select(
            0, common_base_rows
        )
        reweight_normalized = torch.abs(
            current_normalized_common - base_normalized_common
        )
        reweight_energy = energy.index_select(0, common_current_rows)
        reweight_summary = _summarize(
            reweight, reweight_normalized, reweight_energy
        )
        retained_positions = set(
            int(value) for value in old_history_positions
        )
        retained_local = [
            local
            for local, position in enumerate(common_old_positions)
            if position in retained_positions
        ]
        if retained_local:
            retained_index = torch.tensor(
                retained_local, dtype=torch.long
            )
            retained_summary = _summarize(
                reweight.index_select(0, retained_index),
                reweight_normalized.index_select(0, retained_index),
                reweight_energy.index_select(0, retained_index),
            )
        else:
            empty = old_residual.new_zeros((0,))
            retained_summary = _summarize(empty, empty, empty)
    else:
        empty = old_residual.new_zeros((0,))
        reweight_summary = _summarize(empty, empty, empty)
        retained_summary = _summarize(empty, empty, empty)
    old_summary = _summarize(old_residual, old_normalized, energy)
    fresh_summary = _summarize(fresh_residual, fresh_normalized, energy)
    delta = {
        metric: float(old_summary[metric] - fresh_summary[metric])
        for metric in (
            "raw_sum",
            "normalized_sum",
            "energy_normalized_sum",
            "mean",
            "normalized_mean",
        )
    }
    return {
        "old_coverage": old_summary,
        "fresh_coverage": fresh_summary,
        "delta_e": delta,
        "d_new": new_summary,
        "d_rew": reweight_summary,
        "arrival_residual": arrival_summary,
        "retained_reweighting": retained_summary,
        "deployable_approx_raw_sum": float(
            arrival_summary["raw_sum"] + retained_summary["raw_sum"]
        ),
        "deployable_approx_normalized_sum": float(
            arrival_summary["normalized_sum"]
            + retained_summary["normalized_sum"]
        ),
        "d_func_raw_sum": float(
            new_summary["raw_sum"] + reweight_summary["raw_sum"]
        ),
        "d_func_normalized_sum": float(
            new_summary["normalized_sum"]
            + reweight_summary["normalized_sum"]
        ),
        "old_factor": old_factor.diagnostics(),
        "fresh_factor": fresh_factor.diagnostics(),
        "base_factor": base_factor.diagnostics(),
        "base_feature_energy_sum": float(base_energy.sum().item()),
    }
