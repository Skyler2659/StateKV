"""Norm-based eviction: Key norm, Value norm, KV norm."""
from __future__ import annotations
from typing import Optional
import torch
from src.eviction.base import BaseEviction
from src.eviction.kv_utils import mean_heads


class _NormBase(BaseEviction):
    """Shared logic for norm-based selection."""

    def _get_rows(self, layer_k, layer_v):
        raise NotImplementedError

    def compute_scores(self, layer_k, layer_v, layer_idx, **kw):
        rows = self._get_rows(layer_k, layer_v)
        if rows is None:
            return None
        return torch.norm(rows.float(), p=2, dim=1)

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


class KeyNormEviction(_NormBase):
    name = "key_norm"
    def _get_rows(self, layer_k, layer_v):
        return mean_heads(layer_k, self.k_seq_dim)


class ValueNormEviction(_NormBase):
    name = "value_norm"
    def _get_rows(self, layer_k, layer_v):
        return mean_heads(layer_v, self.v_seq_dim)


class KVNormEviction(_NormBase):
    name = "kv_norm"
    def _get_rows(self, layer_k, layer_v):
        k_rows = mean_heads(layer_k, self.k_seq_dim)
        v_rows = mean_heads(layer_v, self.v_seq_dim)
        if k_rows is None or v_rows is None:
            return v_rows if v_rows is not None else k_rows
        if k_rows.shape[0] != v_rows.shape[0]:
            return v_rows
        return torch.cat([k_rows.float(), v_rows.float()], dim=-1)
