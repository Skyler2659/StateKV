"""Rank-aware exact L2 row leverage using a small Gram eigensystem."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class LeverageDiagnostics:
    calculation: str
    n_rows: int
    n_features: int
    effective_rank: int
    condition_number: Optional[float]
    rank_tolerance: float
    fallback: bool = False
    fallback_reason: Optional[str] = None
    fit_count: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def l2_leverage_numpy(
    rows: np.ndarray,
    rcond: Optional[float] = None,
) -> Tuple[np.ndarray, LeverageDiagnostics]:
    """Return exact statistical row leverage for possibly rank-deficient rows.

    For ``A = U Σ Vᵀ``, row leverage is ``diag(U_r U_rᵀ)`` where ``r``
    contains only singular directions above the numerical rank tolerance. The
    implementation diagonalizes ``AᵀA`` so memory is ``O(nd + d²)`` rather
    than storing a full ``n×n`` projection.

    Numerical failure raises an exception. It never substitutes a norm score.
    """

    matrix = np.asarray(rows, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"rows must be 2-D, got shape={matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError("rows contain non-finite values")
    n_rows, n_features = matrix.shape
    if n_rows == 0:
        diagnostics = LeverageDiagnostics(
            calculation="exact_gram_eigh",
            n_rows=0,
            n_features=n_features,
            effective_rank=0,
            condition_number=None,
            rank_tolerance=0.0,
        )
        return np.empty(0, dtype=np.float32), diagnostics
    if n_features == 0:
        diagnostics = LeverageDiagnostics(
            calculation="exact_gram_eigh",
            n_rows=n_rows,
            n_features=0,
            effective_rank=0,
            condition_number=None,
            rank_tolerance=0.0,
        )
        return np.zeros(n_rows, dtype=np.float32), diagnostics

    transform, diagnostics = l2_whitener_numpy(matrix, rcond=rcond)
    if diagnostics.effective_rank == 0:
        scores = np.zeros(n_rows, dtype=np.float64)
    else:
        whitened = matrix @ transform
        scores = np.sum(whitened * whitened, axis=1)
        scores = np.clip(scores, 0.0, 1.0)
    return scores.astype(np.float32), diagnostics


def l2_whitener_numpy(
    rows: np.ndarray,
    rcond: Optional[float] = None,
) -> Tuple[np.ndarray, LeverageDiagnostics]:
    """Fit the right-side whitening transform used by L2 leverage."""

    matrix = np.asarray(rows, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"rows must be 2-D, got shape={matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError("rows contain non-finite values")
    n_rows, n_features = matrix.shape
    if n_rows == 0 or n_features == 0:
        diagnostics = LeverageDiagnostics(
            calculation="exact_gram_eigh",
            n_rows=n_rows,
            n_features=n_features,
            effective_rank=0,
            condition_number=None,
            rank_tolerance=0.0,
        )
        return np.zeros((n_features, 0), dtype=np.float32), diagnostics

    gram = matrix.T @ matrix
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    largest = float(eigenvalues[-1]) if eigenvalues.size else 0.0
    if rcond is None:
        rcond = max(n_rows, n_features) * np.finfo(np.float64).eps
    if rcond < 0:
        raise ValueError("rcond must be non-negative")
    tolerance = float(rcond) * largest
    keep = eigenvalues > tolerance
    effective_rank = int(np.count_nonzero(keep))

    if effective_rank:
        kept_values = eigenvalues[keep]
        basis = eigenvectors[:, keep]
        transform = basis / np.sqrt(kept_values).reshape(1, -1)
        condition = float(np.sqrt(kept_values.max() / kept_values.min()))
    else:
        transform = np.zeros((n_features, 0), dtype=np.float64)
        condition = None

    diagnostics = LeverageDiagnostics(
        calculation="exact_gram_eigh",
        n_rows=n_rows,
        n_features=n_features,
        effective_rank=effective_rank,
        condition_number=condition,
        rank_tolerance=tolerance,
    )
    return transform.astype(np.float32), diagnostics
