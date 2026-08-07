"""Tie-aware, position-aligned score correlation analysis."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.stats import kendalltau, rankdata

from src.analysis.alignment import align_score_units
from src.artifacts.schema import ScoreArtifact


@dataclass(frozen=True)
class UnitCorrelationResult:
    layer: int
    head: Optional[int]
    pearson: float
    spearman: float
    kendall_tau_b: float
    n_tokens: int
    status: str = "ok"


@dataclass
class CorrelationResult:
    """Macro aggregate plus explicit layer/head correlation results."""

    method_a: str
    method_b: str
    spearman: float
    kendall_tau: float
    pearson: float
    layer_wise_spearman: Dict[int, float] = field(default_factory=dict)
    head_wise_spearman: Dict[Tuple[int, int], float] = field(default_factory=dict)
    unit_results: Dict[str, UnitCorrelationResult] = field(default_factory=dict)
    n_tokens: int = 0
    n_units: int = 0
    aggregation: str = "macro_over_layer_head_units"


def _as_numpy(values) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().double().reshape(-1).numpy()
    return np.asarray(values, dtype=np.float64).reshape(-1)


def _paired_numpy(a, b) -> Tuple[np.ndarray, np.ndarray]:
    left = _as_numpy(a)
    right = _as_numpy(b)
    if left.size != right.size:
        raise ValueError(
            "score vectors must already be position-aligned and have equal length; "
            f"got {left.size} and {right.size}"
        )
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("score vectors must contain only finite values")
    return left, right


def _ranks(scores: torch.Tensor) -> torch.Tensor:
    """Return descending, one-indexed average ranks for tied scores."""

    values = _as_numpy(scores)
    ranks = rankdata(-values, method="average")
    return torch.tensor(ranks, dtype=torch.float64, device=scores.device)


def _pearson_numpy(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2:
        return float("nan")
    a_centered = a - a.mean()
    b_centered = b - b.mean()
    denominator = math.sqrt(float(np.dot(a_centered, a_centered) * np.dot(b_centered, b_centered)))
    if denominator == 0.0:
        return float("nan")
    return float(np.dot(a_centered, b_centered) / denominator)


def _pearson(a, b) -> float:
    left, right = _paired_numpy(a, b)
    return _pearson_numpy(left, right)


def _spearman(a, b) -> float:
    """Spearman correlation using average ranks for ties."""

    left, right = _paired_numpy(a, b)
    if left.size < 2:
        return float("nan")
    left_ranks = rankdata(left, method="average")
    right_ranks = rankdata(right, method="average")
    return _pearson_numpy(left_ranks, right_ranks)


def _kendall_tau(a, b, max_n: Optional[int] = None, seed: int = 0) -> float:
    """Kendall tau-b, with optional deterministic paired subsampling."""

    left, right = _paired_numpy(a, b)
    if left.size < 2:
        return float("nan")
    if max_n is not None and max_n > 0 and left.size > max_n:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(left.size, size=max_n, replace=False))
        left = left[indices]
        right = right[indices]
    result = kendalltau(left, right, variant="b", nan_policy="raise")
    return float(result.statistic)


def _unit_result(layer: int, head: Optional[int], left, right) -> UnitCorrelationResult:
    a, b = _paired_numpy(left, right)
    pearson = _pearson_numpy(a, b)
    spearman = _spearman(a, b)
    tau_b = _kendall_tau(a, b)
    status = "constant_or_undefined" if not all(math.isfinite(x) for x in (pearson, spearman, tau_b)) else "ok"
    return UnitCorrelationResult(
        layer=layer,
        head=head,
        pearson=pearson,
        spearman=spearman,
        kendall_tau_b=tau_b,
        n_tokens=int(a.size),
        status=status,
    )


def _finite_mean(values: List[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def _aggregate(
    method_a: str,
    method_b: str,
    units: Dict[str, UnitCorrelationResult],
) -> CorrelationResult:
    values = list(units.values())
    per_layer: Dict[int, List[float]] = {}
    per_head: Dict[Tuple[int, int], float] = {}
    for result in values:
        per_layer.setdefault(result.layer, []).append(result.spearman)
        if result.head is not None:
            per_head[(result.layer, result.head)] = result.spearman
    return CorrelationResult(
        method_a=method_a,
        method_b=method_b,
        pearson=_finite_mean([result.pearson for result in values]),
        spearman=_finite_mean([result.spearman for result in values]),
        kendall_tau=_finite_mean([result.kendall_tau_b for result in values]),
        layer_wise_spearman={layer: _finite_mean(scores) for layer, scores in per_layer.items()},
        head_wise_spearman=per_head,
        unit_results=units,
        n_tokens=sum(result.n_tokens for result in values),
        n_units=len(values),
    )


class RankCorrelationAnalyzer:
    """Compute correlations only after explicit position alignment."""

    def pairwise(
        self,
        scores_a: torch.Tensor,
        scores_b: torch.Tensor,
        method_a: str = "method_a",
        method_b: str = "method_b",
    ) -> CorrelationResult:
        """Strict raw-vector helper for unit tests and already-aligned callers."""

        unit = _unit_result(0, None, scores_a, scores_b)
        return _aggregate(method_a, method_b, {"layer=0,head=shared": unit})

    def artifact_pairwise(
        self,
        scores_a: ScoreArtifact,
        scores_b: ScoreArtifact,
    ) -> CorrelationResult:
        aligned = align_score_units(scores_a, scores_b)
        units: Dict[str, UnitCorrelationResult] = {}
        for (layer, head), pair in aligned.items():
            label = f"layer={layer},head={'shared' if head is None else head}"
            units[label] = _unit_result(layer, head, pair.scores_a, pair.scores_b)
        return _aggregate(scores_a.method, scores_b.method, units)

    def layer_wise(
        self,
        scores_a_by_layer: Dict[int, torch.Tensor],
        scores_b_by_layer: Dict[int, torch.Tensor],
        method_a: str = "method_a",
        method_b: str = "method_b",
    ) -> CorrelationResult:
        """Legacy helper with strict layer and length equality.

        This helper cannot verify original token positions. Formal analysis must
        use :meth:`artifact_pairwise`.
        """

        if set(scores_a_by_layer) != set(scores_b_by_layer):
            raise ValueError("layer sets must match exactly")
        units: Dict[str, UnitCorrelationResult] = {}
        for layer in sorted(scores_a_by_layer):
            left, right = _paired_numpy(scores_a_by_layer[layer], scores_b_by_layer[layer])
            label = f"layer={layer},head=shared"
            units[label] = _unit_result(layer, None, left, right)
        return _aggregate(method_a, method_b, units)

    def full_matrix(
        self,
        all_scores: Dict[str, ScoreArtifact],
    ) -> Dict[str, Dict[str, float]]:
        methods = sorted(all_scores)
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
                        all_scores[left], all_scores[right]
                    ).spearman
        return matrix
