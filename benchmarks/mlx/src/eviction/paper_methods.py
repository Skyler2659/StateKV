"""Paper-defined KV-cache eviction scores used by the generic Torch API.

The production experiment path in this repository is MLX and has backend-
specific implementations in :mod:`src.runners.mlx_runner`.  These classes keep
the central eviction API usable for unit tests and Torch-side score inspection.
Where the generic cache interface forces one shared token set across heads, the
registry marks the implementation as an approximation of the paper method.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F

from src.eviction.attention import AttentionEviction, LastTokenAttentionEviction
from src.eviction.base import BaseEviction


def _rows_by_head(tensor: torch.Tensor, seq_dim: int) -> torch.Tensor:
    """Return ``[batch*heads, token, feature]`` rows without averaging heads."""
    rows = tensor.movedim(seq_dim, -2).float()
    return rows.reshape(-1, rows.shape[-2], rows.shape[-1])


class _TopScoreEviction(BaseEviction):
    """Shared top-score selector without implicit sink/recent protection."""

    def select_indices(self, scores, seq_len, budget, device):
        if seq_len <= budget:
            return torch.arange(seq_len, dtype=torch.long, device=device)
        if scores is None:
            raise RuntimeError(f"{self.name} requires a score for every cached token")
        take = min(int(budget), int(seq_len))
        return torch.topk(scores[:seq_len].to(device), take).indices.sort().values


class TOVAEviction(LastTokenAttentionEviction):
    """TOVA: evict the token least attended by the latest query."""

    name = "tova"
    method_family = "attention"
    requires_attention = True
    requires_scores = True
    score_source = "last_query_attention"

    def select_indices(self, scores, seq_len, budget, device):
        if seq_len <= budget:
            return torch.arange(seq_len, dtype=torch.long, device=device)
        if scores is None:
            raise RuntimeError("TOVA requires latest-query attention weights")
        return torch.topk(scores[:seq_len].to(device), int(budget)).indices.sort().values


class KNormEviction(_TopScoreEviction):
    """Devoto et al. KNorm: retain keys with the *lowest* L2 norm."""

    name = "knorm"
    method_family = "geometry"
    requires_scores = True
    score_source = "negative_key_l2_norm"

    def compute_scores(self, layer_k, layer_v, layer_idx, **kwargs):
        rows = _rows_by_head(layer_k, self.k_seq_dim)
        # The original method scores each head independently.  The generic
        # cache API has a shared token axis, so average the per-head scores.
        return -torch.linalg.vector_norm(rows, ord=2, dim=-1).mean(dim=0)


class KeyDiffEviction(_TopScoreEviction):
    """Efficient KeyDiff score: negative cosine similarity to the mean key."""

    name = "keydiff"
    method_family = "geometry"
    requires_scores = True
    score_source = "negative_cosine_to_mean_key"

    def compute_scores(self, layer_k, layer_v, layer_idx, **kwargs):
        rows = _rows_by_head(layer_k, self.k_seq_dim)
        anchor = rows.mean(dim=1, keepdim=True)
        cosine = F.cosine_similarity(rows, anchor, dim=-1, eps=1e-8)
        return -cosine.mean(dim=0)


class VATPEviction(AttentionEviction):
    """H2O+VATP: accumulated attention multiplied by value-vector L1 norm."""

    name = "vatp"
    method_family = "hybrid"
    requires_attention = True
    requires_scores = True
    score_source = "accumulated_attention_times_value_l1_norm"

    def compute_scores(self, layer_k, layer_v, layer_idx, **kwargs):
        attention = super().compute_scores(layer_k, layer_v, layer_idx, **kwargs)
        values = _rows_by_head(layer_v, self.v_seq_dim)
        value_l1 = torch.linalg.vector_norm(values, ord=1, dim=-1).mean(dim=0)
        return attention[: value_l1.numel()] * value_l1


class CurDKVEviction(_TopScoreEviction):
    """CurDKV Gaussian-projection key/value product score.

    CurDKV calls the projected row squared norms approximate leverage scores,
    multiplies the key and value scores, and retains the largest products while
    protecting an initial sink prefix.
    """

    name = "curdkv"
    method_family = "geometry"
    requires_scores = True
    score_source = "gaussian_projected_key_value_row_norm_product"

    def __init__(
        self,
        projection_dim: int = 20,
        curdkv_projection_dim: Optional[int] = None,
        curdkv_num_sink: Optional[int] = None,
        seed: int = 0,
        **kwargs,
    ):
        if curdkv_num_sink is not None:
            kwargs["sink_size"] = int(curdkv_num_sink)
        super().__init__(**kwargs)
        self.projection_dim = max(1, int(curdkv_projection_dim or projection_dim))
        self.seed = int(seed)

    def compute_scores(self, layer_k, layer_v, layer_idx, **kwargs):
        keys = _rows_by_head(layer_k, self.k_seq_dim)
        values = _rows_by_head(layer_v, self.v_seq_dim)
        if keys.shape[:2] != values.shape[:2]:
            raise ValueError("CurDKV requires aligned key and value token rows")
        dim = min(int(keys.shape[-1]), int(values.shape[-1]))
        rank = min(self.projection_dim, dim)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + int(layer_idx) * 1009 + int(keys.shape[1]))
        projection = torch.randn((dim, rank), generator=generator, dtype=torch.float32)
        projection = projection.to(keys.device) / math.sqrt(float(rank))
        key_score = torch.sum((keys[..., :dim] @ projection) ** 2, dim=-1)
        value_score = torch.sum((values[..., :dim] @ projection) ** 2, dim=-1)
        score = key_score * value_score
        score = score / score.sum(dim=1, keepdim=True).clamp_min(1e-12)
        return score.mean(dim=0)

    def select_indices(self, scores, seq_len, budget, device):
        if seq_len <= budget:
            return torch.arange(seq_len, dtype=torch.long, device=device)
        sink = min(self.sink_size, int(budget), int(seq_len))
        parts = [torch.arange(sink, device=device, dtype=torch.long)] if sink else []
        take = max(0, int(budget) - sink)
        if take:
            tail_scores = scores[sink:seq_len].to(device)
            selected = torch.topk(tail_scores, min(take, tail_scores.numel())).indices + sink
            parts.append(selected)
        return torch.cat(parts).unique(sorted=True)
