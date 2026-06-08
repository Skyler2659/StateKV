"""L2 leverage score eviction — exact L2 leverage via QR decomposition."""
from __future__ import annotations
from typing import Optional
import torch
from src.eviction.base import BaseEviction
from src.eviction.kv_utils import mean_heads


def l2_row_leverage_scores(rows: torch.Tensor) -> torch.Tensor:
    """Compute standard L2 row leverage scores for A with shape [n, d].

    For rank-r matrix A = U S V^T, the row leverage score is
    ||U[i, :r]||_2^2. Computation is done in float32 for numerical stability
    and the result is returned on the input device.
    """
    rows_f = rows.to(dtype=torch.float32)
    n, d = rows_f.shape
    if n == 0:
        return torch.empty(0, dtype=torch.float32, device=rows.device)
    if n == 1:
        return torch.ones(1, dtype=torch.float32, device=rows.device)
    try:
        u, s, _ = torch.linalg.svd(rows_f, full_matrices=False)
        if s.numel() == 0 or not torch.isfinite(s).all():
            return torch.zeros(n, dtype=torch.float32, device=rows.device)
        eps = torch.finfo(rows_f.dtype).eps
        tol = max(n, d) * eps * s.max().clamp_min(1.0)
        rank = int((s > tol).sum().item())
        if rank <= 0:
            return torch.zeros(n, dtype=torch.float32, device=rows.device)
        return (u[:, :rank].pow(2).sum(dim=1)).to(device=rows.device)
    except Exception:
        return torch.norm(rows_f, p=2, dim=1).to(device=rows.device)


class L2LeverageEviction(BaseEviction):
    """Select tokens by L2 (statistical) leverage score.

    τ_i^(2)(A) = ||q_i||_2^2 where Q is the orthonormal basis of A's column space.
    Computed via thin QR: A = QR → scores = squared row norms of Q.
    """
    name = "l2_leverage"

    def __init__(self, score_source="v", **kwargs):
        super().__init__(**kwargs)
        self.score_source = score_source

    def compute_scores(self, layer_k, layer_v, layer_idx, **kw):
        rows = self._get_rows(layer_k, layer_v)
        if rows is None:
            return None
        return l2_row_leverage_scores(rows)

    def _get_rows(self, layer_k, layer_v):
        v_rows = mean_heads(layer_v, self.v_seq_dim)
        if v_rows is None or self.score_source == "v":
            return v_rows
        k_rows = mean_heads(layer_k, self.k_seq_dim)
        if k_rows is None or k_rows.shape[0] != v_rows.shape[0]:
            return v_rows
        return torch.cat([k_rows.float(), v_rows.float()], dim=-1)

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
