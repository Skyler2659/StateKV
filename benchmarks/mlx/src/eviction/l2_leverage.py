"""L2 leverage score eviction with explicit rank-aware diagnostics."""
from __future__ import annotations
from typing import Optional, Union, Tuple, Dict, Any
import torch
from src.eviction.base import BaseEviction
from src.eviction.kv_utils import mean_heads


def l2_row_leverage_scores(
    rows: torch.Tensor,
    return_diagnostics: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
    """Compute exact L2 row leverage for full-rank or rank-deficient rows.

    Failure is explicit; this function never falls back to a row norm.
    """
    rows_f = rows.to(dtype=torch.float32)
    n, d = rows_f.shape
    if n == 0:
        scores = torch.empty(0, dtype=torch.float32, device=rows.device)
        diagnostics = {
            "calculation": "exact_svd",
            "n_rows": 0,
            "n_features": d,
            "effective_rank": 0,
            "condition_number": None,
            "rank_tolerance": 0.0,
            "fallback": False,
            "fallback_reason": None,
            "fit_count": 1,
        }
        return (scores, diagnostics) if return_diagnostics else scores
    if not torch.isfinite(rows_f).all():
        raise ValueError("rows contain non-finite values")
    u, singular_values, _ = torch.linalg.svd(rows_f, full_matrices=False)
    largest = float(singular_values.max().item()) if singular_values.numel() else 0.0
    tolerance = max(n, d) * torch.finfo(rows_f.dtype).eps * largest
    keep = singular_values > tolerance
    effective_rank = int(keep.sum().item())
    if effective_rank:
        scores = u[:, keep].pow(2).sum(dim=1)
        kept = singular_values[keep]
        condition = float((kept.max() / kept.min()).item())
    else:
        scores = torch.zeros(n, dtype=torch.float32, device=rows.device)
        condition = None
    scores = scores.clamp(0.0, 1.0).to(device=rows.device)
    diagnostics = {
        "calculation": "exact_svd",
        "n_rows": n,
        "n_features": d,
        "effective_rank": effective_rank,
        "condition_number": condition,
        "rank_tolerance": tolerance,
        "fallback": False,
        "fallback_reason": None,
        "fit_count": 1,
    }
    return (scores, diagnostics) if return_diagnostics else scores


class L2LeverageEviction(BaseEviction):
    """Select tokens by L2 (statistical) leverage score.

    τ_i^(2)(A) = ||q_i||_2^2 where Q is the orthonormal basis of A's column space.
    Computed from the effective-rank left singular subspace, so rank-deficient
    matrices follow the Moore--Penrose definition.
    """
    name = "l2_leverage"
    method_family = "geometry"
    supports_backends = ("torch", "mlx")
    requires_scores = True
    score_source = "value"

    def __init__(self, score_source="v", **kwargs):
        super().__init__(**kwargs)
        self.score_source = str(score_source).lower()
        if self.score_source not in {"v", "k", "kv", "value", "key", "key_value_concat"}:
            raise ValueError(f"unsupported L2 score_source={score_source!r}")
        self.estimator_diagnostics = {}

    def compute_scores(self, layer_k, layer_v, layer_idx, **kw):
        rows = self._get_rows(layer_k, layer_v)
        if rows is None:
            return None
        scores, diagnostics = l2_row_leverage_scores(rows, return_diagnostics=True)
        self.estimator_diagnostics[int(layer_idx)] = diagnostics
        return scores

    def _get_rows(self, layer_k, layer_v):
        source = {"value": "v", "key": "k", "key_value_concat": "kv"}.get(
            self.score_source, self.score_source
        )
        v_rows = mean_heads(layer_v, self.v_seq_dim)
        if source == "v":
            return v_rows
        k_rows = mean_heads(layer_k, self.k_seq_dim)
        if source == "k":
            return k_rows
        if k_rows is None or v_rows is None or k_rows.shape[0] != v_rows.shape[0]:
            raise ValueError("K and V rows must align for score_source='kv'")
        return torch.cat([k_rows.float(), v_rows.float()], dim=-1)

    def get_debug_info(self):
        info = super().get_debug_info()
        info["estimator_diagnostics"] = dict(self.estimator_diagnostics)
        return info

    def reset(self):
        super().reset()
        self.estimator_diagnostics.clear()

    def select_indices(self, scores, seq_len, budget, device):
        if seq_len <= budget:
            return torch.arange(seq_len, dtype=torch.long, device=device)
        reserved = self._reserved_indices(seq_len, budget, device)
        if scores is None:
            return self._fill_budget(reserved, seq_len, budget, device)
        fill = budget - int(reserved.numel())
        if fill <= 0:
            return self._ensure_budget(reserved, seq_len, budget, device, reserved=reserved)
        masked = scores[:seq_len].clone().to(device)
        if reserved.numel() > 0:
            masked[reserved] = -float("inf")
        valid = torch.isfinite(masked)
        if not valid.any():
            return self._fill_budget(reserved, seq_len, budget, device)
        topk = min(fill, int(valid.sum().item()))
        idx = torch.topk(masked, topk).indices
        return self._ensure_budget(
            torch.cat([reserved, idx]),
            seq_len,
            budget,
            device,
            scores=scores,
            reserved=reserved,
        )


def ridge_row_leverage_scores(rows: torch.Tensor, ridge_lambda: float = 1e-3) -> torch.Tensor:
    """Compute ridge leverage scores diag(A(A^T A + λI)^-1 A^T)."""
    rows_f = rows.to(dtype=torch.float32)
    n, d = rows_f.shape
    if n == 0:
        return torch.empty(0, dtype=torch.float32, device=rows.device)
    if n == 1:
        return torch.ones(1, dtype=torch.float32, device=rows.device)
    lam = max(float(ridge_lambda), 1e-8)
    gram = rows_f.T @ rows_f
    inv = torch.linalg.inv(gram + lam * torch.eye(d, device=rows.device, dtype=rows_f.dtype))
    scores = torch.sum((rows_f @ inv) * rows_f, dim=1)
    if not torch.isfinite(scores).all():
        raise RuntimeError("ridge leverage produced non-finite scores")
    return torch.clamp(scores, min=0.0).to(device=rows.device)


class RidgeLeverageEviction(L2LeverageEviction):
    """Ridge-regularized leverage score baseline."""

    name = "ridge_leverage"
    approximate = False

    def __init__(self, ridge_lambda: float = 1e-3, **kwargs):
        super().__init__(**kwargs)
        self.ridge_lambda = float(ridge_lambda)

    def compute_scores(self, layer_k, layer_v, layer_idx, **kw):
        rows = self._get_rows(layer_k, layer_v)
        if rows is None:
            return None
        return ridge_row_leverage_scores(rows, self.ridge_lambda)


class ApproximateL2LeverageEviction(L2LeverageEviction):
    """Random-projection approximate L2 leverage score baseline."""

    name = "approximate_l2_leverage"
    approximate = True

    def __init__(self, sketch_dim: int = 256, seed: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.sketch_dim = int(sketch_dim)
        self.seed = int(seed)

    def compute_scores(self, layer_k, layer_v, layer_idx, **kw):
        rows = self._get_rows(layer_k, layer_v)
        if rows is None:
            return None
        rows_f = rows.float()
        n, d = rows_f.shape
        if d <= self.sketch_dim:
            return l2_row_leverage_scores(rows_f)
        generator = torch.Generator(device="cpu").manual_seed(self.seed + int(layer_idx))
        proj = torch.randn(d, self.sketch_dim, generator=generator, dtype=torch.float32)
        proj = proj.to(rows_f.device) / max(self.sketch_dim, 1) ** 0.5
        return l2_row_leverage_scores(rows_f @ proj)
