"""Matched per-layer/head selection overlap analysis.

The aggregate is a macro average over layer/head units. Token positions are
never unioned across layers, because a token selected once and a token selected
in every layer are scientifically different events.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch

from src.analysis.alignment import align_selection_units
from src.analysis.rank_correlation import _spearman
from src.artifacts.schema import SelectionArtifact


@dataclass(frozen=True)
class UnitOverlapResult:
    layer: int
    head: Optional[int]
    jaccard: float
    intersection_ratio: float
    overlap_coefficient: float
    recall_a_by_b: float
    recall_b_by_a: float
    overlap_count: int
    union_count: int
    a_only_count: int
    b_only_count: int
    a_count: int
    b_count: int


@dataclass
class OverlapResult:
    method_a: str
    method_b: str
    jaccard: float
    micro_jaccard: float
    intersection_ratio: float
    overlap_coefficient: float
    recall_a_by_b: float
    recall_b_by_a: float
    overlap_count: int
    union_count: int
    a_only_count: int
    b_only_count: int
    layer_wise_jaccard: Dict[int, float] = field(default_factory=dict)
    head_wise_jaccard: Dict[Tuple[int, int], float] = field(default_factory=dict)
    unit_results: Dict[str, UnitOverlapResult] = field(default_factory=dict)
    n_units: int = 0
    aggregation: str = "macro_over_layer_head_units"


@dataclass(frozen=True)
class SelectionFrequencyResult:
    method_a: str
    method_b: str
    spearman: float
    n_positions: int
    frequency_a: Dict[int, float]
    frequency_b: Dict[int, float]


def _safe_ratio(numerator: int, denominator: int, both_empty: bool = False) -> float:
    if denominator:
        return numerator / denominator
    return 1.0 if both_empty else 0.0


def _unit_overlap(
    layer: int,
    head: Optional[int],
    selected_a: Set[int],
    selected_b: Set[int],
) -> UnitOverlapResult:
    intersection = selected_a & selected_b
    union = selected_a | selected_b
    both_empty = not selected_a and not selected_b
    return UnitOverlapResult(
        layer=layer,
        head=head,
        jaccard=_safe_ratio(len(intersection), len(union), both_empty),
        # For equal top-k sets this is |A∩B|/k. With unequal sets the larger
        # selection is the denominator, preventing a smaller set from appearing
        # to have complete top-k agreement.
        intersection_ratio=_safe_ratio(
            len(intersection), max(len(selected_a), len(selected_b)), both_empty
        ),
        overlap_coefficient=_safe_ratio(
            len(intersection), min(len(selected_a), len(selected_b)), both_empty
        ),
        recall_a_by_b=_safe_ratio(len(intersection), len(selected_a), both_empty),
        recall_b_by_a=_safe_ratio(len(intersection), len(selected_b), both_empty),
        overlap_count=len(intersection),
        union_count=len(union),
        a_only_count=len(selected_a - selected_b),
        b_only_count=len(selected_b - selected_a),
        a_count=len(selected_a),
        b_count=len(selected_b),
    )


def _mean(values: List[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _aggregate(
    method_a: str,
    method_b: str,
    units: Dict[str, UnitOverlapResult],
) -> OverlapResult:
    values = list(units.values())
    per_layer: Dict[int, List[float]] = {}
    per_head: Dict[Tuple[int, int], float] = {}
    for result in values:
        per_layer.setdefault(result.layer, []).append(result.jaccard)
        if result.head is not None:
            per_head[(result.layer, result.head)] = result.jaccard
    overlap_count = sum(result.overlap_count for result in values)
    union_count = sum(result.union_count for result in values)
    return OverlapResult(
        method_a=method_a,
        method_b=method_b,
        jaccard=_mean([result.jaccard for result in values]),
        micro_jaccard=_safe_ratio(overlap_count, union_count, union_count == 0),
        intersection_ratio=_mean([result.intersection_ratio for result in values]),
        overlap_coefficient=_mean([result.overlap_coefficient for result in values]),
        recall_a_by_b=_mean([result.recall_a_by_b for result in values]),
        recall_b_by_a=_mean([result.recall_b_by_a for result in values]),
        overlap_count=overlap_count,
        union_count=union_count,
        a_only_count=sum(result.a_only_count for result in values),
        b_only_count=sum(result.b_only_count for result in values),
        layer_wise_jaccard={layer: _mean(scores) for layer, scores in per_layer.items()},
        head_wise_jaccard=per_head,
        unit_results=units,
        n_units=len(values),
    )


def selection_frequency(artifact: SelectionArtifact) -> Dict[int, float]:
    """Fraction of eligible layer/head units selecting each original token."""

    eligible: Dict[int, int] = {}
    selected: Dict[int, int] = {}
    for unit in artifact.units:
        selected_set = set(unit.selected_positions)
        for position in unit.universe_positions:
            eligible[position] = eligible.get(position, 0) + 1
            if position in selected_set:
                selected[position] = selected.get(position, 0) + 1
    return {
        position: selected.get(position, 0) / count
        for position, count in sorted(eligible.items())
    }


class OverlapAnalyzer:
    """Compute overlap on matched layer/head units and physical budgets."""

    def pairwise_overlap(
        self,
        selected_a: Dict[int, torch.Tensor],
        selected_b: Dict[int, torch.Tensor],
        method_a: str = "method_a",
        method_b: str = "method_b",
    ) -> OverlapResult:
        """Legacy per-layer helper.

        This no longer forms a cross-layer union. It requires identical layers
        and equal selection sizes. Formal analysis must use
        :meth:`artifact_pairwise` to verify snapshot, head, universe, and scope.
        """

        if set(selected_a) != set(selected_b):
            raise ValueError("layer sets must match exactly")
        units: Dict[str, UnitOverlapResult] = {}
        for layer in sorted(selected_a):
            left = set(int(value) for value in selected_a[layer].reshape(-1).tolist())
            right = set(int(value) for value in selected_b[layer].reshape(-1).tolist())
            if len(left) != len(right):
                raise ValueError(f"selected budgets differ for layer {layer}")
            label = f"layer={layer},head=shared"
            units[label] = _unit_overlap(layer, None, left, right)
        return _aggregate(method_a, method_b, units)

    def artifact_pairwise(
        self,
        selected_a: SelectionArtifact,
        selected_b: SelectionArtifact,
    ) -> OverlapResult:
        aligned = align_selection_units(selected_a, selected_b)
        units: Dict[str, UnitOverlapResult] = {}
        for (layer, head), pair in aligned.items():
            label = f"layer={layer},head={'shared' if head is None else head}"
            units[label] = _unit_overlap(
                layer,
                head,
                set(pair.selected_a),
                set(pair.selected_b),
            )
        return _aggregate(selected_a.method, selected_b.method, units)

    def frequency_correlation(
        self,
        selected_a: SelectionArtifact,
        selected_b: SelectionArtifact,
    ) -> SelectionFrequencyResult:
        # Alignment validates snapshot, units, universes, scope, and budget.
        align_selection_units(selected_a, selected_b)
        frequency_a = selection_frequency(selected_a)
        frequency_b = selection_frequency(selected_b)
        if set(frequency_a) != set(frequency_b):
            raise ValueError("eligible token positions differ across artifacts")
        positions = sorted(frequency_a)
        correlation = _spearman(
            np.asarray([frequency_a[position] for position in positions]),
            np.asarray([frequency_b[position] for position in positions]),
        )
        return SelectionFrequencyResult(
            method_a=selected_a.method,
            method_b=selected_b.method,
            spearman=correlation,
            n_positions=len(positions),
            frequency_a=frequency_a,
            frequency_b=frequency_b,
        )

    def budget_curve(
        self,
        artifact_pairs: List[Tuple[SelectionArtifact, SelectionArtifact]],
    ) -> Dict[int, OverlapResult]:
        curve: Dict[int, OverlapResult] = {}
        for left, right in artifact_pairs:
            budget = int(left.requested_budget)
            if budget in curve:
                raise ValueError(f"duplicate artifact pair for budget {budget}")
            curve[budget] = self.artifact_pairwise(left, right)
        return dict(sorted(curve.items()))

    def full_matrix(
        self,
        all_selected: Dict[str, SelectionArtifact],
    ) -> Dict[str, Dict[str, float]]:
        methods = sorted(all_selected)
        matrix: Dict[str, Dict[str, float]] = {}
        for left in methods:
            matrix[left] = {}
            for right in methods:
                if left == right:
                    matrix[left][right] = 1.0
                elif right in matrix and left in matrix[right]:
                    matrix[left][right] = matrix[right][left]
                else:
                    matrix[left][right] = self.artifact_pairwise(
                        all_selected[left], all_selected[right]
                    ).jaccard
        return matrix

    @staticmethod
    def to_numpy_matrix(
        matrix: Dict[str, Dict[str, float]],
    ) -> Tuple[np.ndarray, List[str]]:
        labels = sorted(matrix)
        array = np.zeros((len(labels), len(labels)))
        for row, left in enumerate(labels):
            for column, right in enumerate(labels):
                array[row, column] = matrix[left].get(right, float("nan"))
        return array, labels
