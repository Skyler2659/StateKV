"""Per-KV-head geometry scoring and stable residual projection."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from kvbench.config import MethodConfig
from kvbench.errors import SignalUnavailableError


def normalize_scores(scores: torch.Tensor, mode: str) -> torch.Tensor:
    values = scores.detach().float().flatten()
    finite = torch.isfinite(values)
    if not finite.any():
        raise SignalUnavailableError("score vector has no finite values")
    mode = str(mode or "none").lower()
    if mode == "none":
        return torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    work = values.clone()
    replacement = work[finite].min()
    work[~finite] = replacement
    if mode == "minmax":
        lo, hi = work.min(), work.max()
        return (work - lo) / (hi - lo).clamp_min(1e-8)
    if mode == "zscore":
        return (work - work.mean()) / work.std(unbiased=False).clamp_min(1e-8)
    if mode in {"log", "log_score"}:
        shifted = work - work.min()
        logged = torch.log1p(shifted)
        return (logged - logged.min()) / (logged.max() - logged.min()).clamp_min(1e-8)
    if mode == "rank":
        # Average ranks for ties; highest score receives 1.0.
        sorted_values, order = torch.sort(work, stable=True)
        sorted_ranks = torch.empty_like(sorted_values)
        start = 0
        count = int(sorted_values.numel())
        while start < count:
            end = start + 1
            while end < count and bool(sorted_values[end] == sorted_values[start]):
                end += 1
            average = 0.5 * float(start + end - 1)
            sorted_ranks[start:end] = average
            start = end
        ranks = torch.empty_like(sorted_ranks)
        ranks[order] = sorted_ranks
        if count == 1:
            return torch.ones_like(ranks)
        return ranks / float(count - 1)
    raise ValueError("unsupported score normalization: %s" % mode)


def _rows_by_head(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim != 4:
        raise SignalUnavailableError("expected KV tensor [batch, head, token, feature]")
    if tensor.shape[0] != 1:
        raise SignalUnavailableError("paper runner currently supports batch_size=1")
    return tensor[0].float()


def _effective_l2_leverage(rows: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
    n, d = rows.shape
    if n == 0:
        return rows.new_zeros((0,)), {"effective_rank": 0, "condition_number": None}
    if not torch.isfinite(rows).all():
        raise SignalUnavailableError("non-finite rows in L2 leverage")
    u, singular, _ = torch.linalg.svd(rows, full_matrices=False)
    largest = float(singular.max().item()) if singular.numel() else 0.0
    tolerance = max(n, d) * torch.finfo(rows.dtype).eps * largest
    keep = singular > tolerance
    rank = int(keep.sum().item())
    if rank == 0:
        score = rows.new_zeros((n,))
        condition = None
    else:
        score = u[:, keep].square().sum(dim=1).clamp(0.0, 1.0)
        selected = singular[keep]
        condition = float((selected.max() / selected.min()).item())
    return score, {
        "calculation": "exact_svd",
        "n_rows": int(n),
        "n_features": int(d),
        "effective_rank": rank,
        "condition_number": condition,
        "rank_tolerance": float(tolerance),
    }


def _ridge_l2_leverage(
    rows: torch.Tensor,
    coefficient: float,
    mode: str,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Return diag(A(A^T A + lambda I)^-1 A^T) without an inverse."""
    n, d = rows.shape
    if n == 0:
        return rows.new_zeros((0,)), {
            "calculation": "ridge_l2",
            "effective_dimension": 0.0,
            "ridge": None,
        }
    gram = rows.T @ rows
    if mode == "relative":
        scale = float(torch.trace(gram).item()) / max(1, d)
        ridge = float(coefficient) * max(scale, 1e-12)
    else:
        ridge = float(coefficient)
    ridge = max(ridge, 1e-12)
    identity = torch.eye(d, device=rows.device, dtype=rows.dtype)
    whitened = torch.linalg.solve(gram + ridge * identity, rows.T).T
    score = (rows * whitened).sum(dim=-1).clamp_min(0.0)
    if not torch.isfinite(score).all():
        raise SignalUnavailableError("ridge L2 leverage produced non-finite scores")
    return score, {
        "calculation": "ridge_l2",
        "n_rows": int(n),
        "n_features": int(d),
        "ridge": float(ridge),
        "lambda_mode": mode,
        "effective_dimension": float(score.sum().item()),
    }


def _woodruff_l1(
    rows: torch.Tensor,
    sketch_dim: int,
    seed: int,
    weight_floor: float = 1e-3,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    n, d = rows.shape
    if n <= 1:
        return rows.new_ones((n,)), {
            "calculation": "degenerate_l1",
            "effective_rank": int(n > 0),
        }
    rng = np.random.default_rng(int(seed) + int(n) * 9176 + int(d))
    uniform = rng.uniform(1e-8, 1.0 - 1e-8, size=(n, 1)).astype(np.float32)
    weights_np = np.maximum(-np.log(1.0 - uniform), weight_floor)
    weights = torch.from_numpy(weights_np).to(rows.device)
    if n < sketch_dim:
        weighted = rows / weights
        used_count_sketch = False
    else:
        buckets = torch.from_numpy(
            rng.integers(0, sketch_dim, size=n, dtype=np.int64)
        ).to(rows.device)
        signs = torch.from_numpy(
            rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=n)
        ).to(rows.device)
        weighted_rows = rows / weights * signs[:, None]
        weighted = rows.new_zeros((sketch_dim, d))
        weighted.index_add_(0, buckets, weighted_rows)
        used_count_sketch = True
    _, r = torch.linalg.qr(weighted, mode="reduced")
    singular = torch.linalg.svdvals(r)
    largest = float(singular.max().item()) if singular.numel() else 0.0
    tolerance = max(weighted.shape) * torch.finfo(rows.dtype).eps * largest
    kept = singular[singular > tolerance]
    if kept.numel() == 0:
        raise SignalUnavailableError("L1 sketch produced a zero-rank basis")
    condition = float((kept.max() / kept.min()).item())
    if not math.isfinite(condition) or condition > 1e6:
        raise SignalUnavailableError("L1 sketch is ill-conditioned: %.6g" % condition)
    transform = torch.linalg.pinv(r)
    score = torch.linalg.vector_norm(rows @ transform, ord=1, dim=1)
    if not torch.isfinite(score).all():
        raise SignalUnavailableError("L1 leverage produced non-finite scores")
    return score, {
        "calculation": "approximate_l1_woodruff",
        "n_rows": int(n),
        "n_features": int(d),
        "effective_rank": int(kept.numel()),
        "condition_number": condition,
        "sketch_dim": int(sketch_dim),
        "used_count_sketch": used_count_sketch,
        "seed": int(seed),
    }


def _sketched_l2(
    rows: torch.Tensor,
    sketch_dim: int,
    seed: int,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Approximate row leverage using a seeded CountSketch subspace embedding."""
    n, d = rows.shape
    if n == 0:
        return rows.new_zeros((0,)), {
            "calculation": "countsketch_l2",
            "effective_rank": 0,
            "sketch_dim": int(sketch_dim),
        }
    size = min(n, max(int(sketch_dim), d))
    if n <= size:
        sketched = rows
        used_count_sketch = False
    else:
        rng = np.random.default_rng(int(seed) + n * 10007 + d * 101)
        buckets = torch.from_numpy(
            rng.integers(0, size, size=n, dtype=np.int64)
        ).to(rows.device)
        signs = torch.from_numpy(
            rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=n)
        ).to(rows.device)
        sketched = rows.new_zeros((size, d))
        sketched.index_add_(0, buckets, rows * signs[:, None])
        used_count_sketch = True
    _, r = torch.linalg.qr(sketched, mode="reduced")
    singular = torch.linalg.svdvals(r)
    largest = float(singular.max().item()) if singular.numel() else 0.0
    tolerance = max(r.shape) * torch.finfo(rows.dtype).eps * largest
    rank = int((singular > tolerance).sum().item())
    if rank == 0:
        return rows.new_zeros((n,)), {
            "calculation": "countsketch_l2",
            "effective_rank": 0,
            "sketch_dim": size,
            "used_count_sketch": used_count_sketch,
            "seed": int(seed),
        }
    transformed = rows @ torch.linalg.pinv(r, rtol=max(r.shape) * torch.finfo(rows.dtype).eps)
    score = transformed.square().sum(dim=1).clamp_min(0.0)
    if not torch.isfinite(score).all():
        raise SignalUnavailableError("sketched L2 leverage produced non-finite scores")
    return score, {
        "calculation": "countsketch_l2",
        "n_rows": int(n),
        "n_features": int(d),
        "effective_rank": rank,
        "sketch_dim": size,
        "used_count_sketch": used_count_sketch,
        "seed": int(seed),
    }


class ScoreEngine:
    """Canonical scorer: fit each KV head independently, then aggregate scores."""

    def __init__(self, cfg: MethodConfig, seed: int):
        self.cfg = cfg
        self.seed = int(seed)

    def geometry(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        layer: int,
        source: Optional[str] = None,
        estimator: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        source = str(source or self.cfg.score_source).lower()
        k_rows = _rows_by_head(key)
        v_rows = _rows_by_head(value)
        if source == "k":
            rows = k_rows
        elif source == "v":
            rows = v_rows
        elif source in {"kv", "joint"}:
            rows = torch.cat([k_rows, v_rows], dim=-1)
        else:
            raise SignalUnavailableError("unsupported geometry source: %s" % source)
        kind = str(estimator or self.cfg.leverage_estimator).lower()
        head_scores: List[torch.Tensor] = []
        head_diagnostics: List[Dict[str, Any]] = []
        for head in range(int(rows.shape[0])):
            if kind in {"l2", "l2_exact", "exact_l2"}:
                score, diagnostics = _effective_l2_leverage(rows[head])
            elif kind in {"l2_sketch", "sketched_l2", "approximate_l2"}:
                score, diagnostics = _sketched_l2(
                    rows[head],
                    int(self.cfg.sketch_dim),
                    self.seed + int(layer) * 1009 + int(head) * 9173,
                )
            elif kind in {"l1", "l1_approx", "approximate_l1"}:
                score, diagnostics = _woodruff_l1(
                    rows[head],
                    int(self.cfg.sketch_dim),
                    self.seed + int(layer) * 1009 + int(head) * 9173,
                )
            elif kind == "l1_norm":
                score = torch.linalg.vector_norm(rows[head], ord=1, dim=-1)
                diagnostics = {"calculation": "row_l1_norm"}
            elif kind == "l2_norm":
                score = torch.linalg.vector_norm(rows[head], ord=2, dim=-1)
                diagnostics = {"calculation": "row_l2_norm"}
            else:
                raise SignalUnavailableError("unsupported leverage estimator: %s" % kind)
            head_scores.append(score)
            head_diagnostics.append(diagnostics)
        by_head = torch.stack(head_scores, dim=0)
        return by_head.mean(dim=0), by_head, {
            "source": source,
            "estimator": kind,
            "head_diagnostics": head_diagnostics,
            "aggregation": "mean_of_independently_computed_kv_head_scores",
        }

    def residual_v(
        self,
        value: torch.Tensor,
        attention_rows: List[int],
        layer: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        values = _rows_by_head(value)
        if not attention_rows:
            raise SignalUnavailableError("residual leverage requires a non-empty attention core")
        selected_index = torch.tensor(attention_rows, device=values.device, dtype=torch.long)
        head_scores: List[torch.Tensor] = []
        diagnostics: List[Dict[str, Any]] = []
        for head in range(int(values.shape[0])):
            rows = values[head]
            selected = rows.index_select(0, selected_index)
            dim = int(rows.shape[-1])
            gram = selected.T @ selected
            if self.cfg.residual_lambda_mode == "relative":
                scale = float(torch.trace(gram).item()) / max(1, dim)
                ridge = float(self.cfg.residual_lambda) * max(scale, 1e-12)
            else:
                ridge = float(self.cfg.residual_lambda)
            identity = torch.eye(dim, device=rows.device, dtype=rows.dtype)
            # Feature-space ridge projector. This avoids an |S| x |S| inverse
            # when the attention core is larger than the head dimension.
            projector = torch.linalg.solve(gram + ridge * identity, gram)
            residual = rows - rows @ projector
            score, leverage_diag = _ridge_l2_leverage(
                residual,
                float(self.cfg.residual_lambda),
                self.cfg.residual_lambda_mode,
            )
            selected_residual = torch.linalg.vector_norm(
                residual.index_select(0, selected_index), ord=2, dim=-1
            )
            head_scores.append(score)
            diagnostics.append({
                **leverage_diag,
                "ridge": ridge,
                "lambda_mode": self.cfg.residual_lambda_mode,
                "attention_core_size": len(attention_rows),
                "selected_residual_max": float(selected_residual.max().item()),
            })
        by_head = torch.stack(head_scores, dim=0)
        return by_head.mean(dim=0), by_head, {
            "calculation": "attention_core_ridge_residual_v_l2_leverage",
            "head_diagnostics": diagnostics,
            "aggregation": "mean_of_independently_computed_kv_head_scores",
        }

    def curdkv(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        layer: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """CurDKV Algorithm-1 Gaussian projected K/V norm product.

        This is deliberately not implemented as a product of statistical
        leverage scores: CurDKV calls the two projected squared row norms its
        approximate leverage signals before multiplying them.
        """
        keys = _rows_by_head(key)
        values = _rows_by_head(value)
        if keys.shape[:2] != values.shape[:2]:
            raise SignalUnavailableError("CurDKV requires aligned K/V token rows")
        dim = min(int(keys.shape[-1]), int(values.shape[-1]))
        rank = max(1, min(int(self.cfg.curdkv_projection_dim), dim))
        generator = torch.Generator(device="cpu")
        generator.manual_seed(
            self.seed + int(layer) * 1009 + int(keys.shape[1]) * 9173
        )
        projection = torch.randn(
            dim, rank, generator=generator, dtype=torch.float32
        ).to(keys.device) / math.sqrt(float(rank))
        key_score = (keys[..., :dim] @ projection).square().sum(dim=-1)
        value_score = (values[..., :dim] @ projection).square().sum(dim=-1)
        by_head = key_score * value_score
        by_head = by_head / by_head.sum(dim=1, keepdim=True).clamp_min(1e-12)
        return by_head.mean(dim=0), by_head, {
            "calculation": "curdkv_gaussian_projected_kv_row_norm_product",
            "projection_dim": rank,
            "aggregation": "shared_token_mean_of_per_head_scores",
            "seed": int(generator.initial_seed()),
        }

    @staticmethod
    def attention_by_layer(
        values: Dict[int, torch.Tensor],
        layer: int,
        seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        score = values.get(layer)
        if score is None or score.ndim != 2 or int(score.shape[-1]) < seq_len:
            raise SignalUnavailableError(
                "aligned per-KV-head attention is unavailable for layer=%d" % layer
            )
        by_head = score[:, :seq_len].float()
        return by_head.mean(dim=0), by_head
