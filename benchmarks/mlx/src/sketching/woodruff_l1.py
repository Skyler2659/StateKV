"""Woodruff-style approximate L1 leverage with explicit diagnostics."""
from __future__ import annotations
import torch
from typing import Dict, Optional


def _make_cpu_generator(seed):
    if seed is None:
        return None
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    return g


def _safe_exp_samples(size, device, dtype, generator=None):
    """Generate Exp(1) samples in float32 for numerical safety."""
    sample_device = "cpu" if generator is not None else device
    u = torch.rand(size, device=sample_device, dtype=torch.float32, generator=generator)
    u.clamp_(1e-8, 1 - 1e-8)
    return (-torch.log(1.0 - u)).to(device=device, dtype=dtype)


class CountSketch:
    """Hash-based dimensionality reduction for L1 sketching."""

    def __init__(self, sketch_dim: int, seed=None):
        self.sketch_dim = int(sketch_dim)
        self.seed = seed
        self.hash_buckets = None
        self.signs = None
        self._gen = _make_cpu_generator(seed)

    def _ensure_capacity(self, n, device):
        if self.hash_buckets is not None and self.hash_buckets.device != device:
            self.hash_buckets = self.hash_buckets.to(device)
            self.signs = self.signs.to(device)
        if self.hash_buckets is not None and self.hash_buckets.numel() >= n:
            return
        old_n = 0 if self.hash_buckets is None else self.hash_buckets.numel()
        grow = int(n - old_n)
        new_b = torch.randint(0, self.sketch_dim, (grow,), generator=self._gen).to(device)
        new_s = (torch.randint(0, 2, (grow,), generator=self._gen, dtype=torch.int64) * 2 - 1)
        new_s = new_s.to(device=device, dtype=torch.float32)
        if self.hash_buckets is None:
            self.hash_buckets, self.signs = new_b, new_s
        else:
            self.hash_buckets = torch.cat([self.hash_buckets, new_b])
            self.signs = torch.cat([self.signs, new_s])

    def apply(self, rows: torch.Tensor) -> torch.Tensor:
        n, d = rows.shape
        self._ensure_capacity(n, rows.device)
        b = self.hash_buckets[:n]
        s = self.signs[:n].to(rows.dtype)
        sketch = torch.zeros(self.sketch_dim, d, device=rows.device, dtype=rows.dtype)
        sketch.scatter_add_(0, b.unsqueeze(-1).expand(-1, d), rows * s.unsqueeze(-1))
        return sketch


class L1SubspaceEmbedding:
    """Exp(1) reweighting + CountSketch."""

    def __init__(self, sketch_dim, seed=None, exp_generator=None):
        self.sketch_dim = int(sketch_dim)
        self.count_sketch = CountSketch(self.sketch_dim, seed=seed)
        self._exp_gen = exp_generator or _make_cpu_generator(seed)

    def embed(self, v_rows: torch.Tensor) -> torch.Tensor:
        n = v_rows.shape[0]
        w = _safe_exp_samples((n, 1), v_rows.device, v_rows.dtype, self._exp_gen)
        weighted = v_rows / w
        return self.count_sketch.apply(weighted)


class WoodruffL1Estimator:
    """Approximate L1 leverage score estimator.

    Pipeline: Exp(1) reweight → CountSketch → QR → row L1 norms of A·R⁻¹.
    """

    def __init__(self, sketch_dim=1024, seed=None, condition_limit: float = 1e6):
        self.sketch_dim = int(sketch_dim)
        self.seed = seed
        self._exp_gen = _make_cpu_generator(seed)
        self.embedding = L1SubspaceEmbedding(self.sketch_dim, seed=seed, exp_generator=self._exp_gen)
        self.r_inv: Optional[torch.Tensor] = None
        self.last_dim: Optional[int] = None
        self._last_scores: Optional[torch.Tensor] = None
        self.condition_limit = float(condition_limit)
        self.fit_count = 0
        self.last_diagnostics: Dict[str, object] = {
            "calculation": "approximate_l1_woodruff",
            "fit_count": 0,
            "fallback": False,
            "fallback_reason": None,
        }

    def fit(self, rows: torch.Tensor) -> None:
        """Build the sketch basis (QR of weighted+sketched matrix)."""
        n, d = rows.shape
        if n <= 1:
            self.r_inv = None
            self.last_dim = d
            return
        self.fit_count += 1
        if n < self.embedding.count_sketch.sketch_dim:
            w = _safe_exp_samples((n, 1), rows.device, torch.float32, self._exp_gen)
            weighted = rows.float() / w
        else:
            weighted = self.embedding.embed(rows).float()
        if not torch.isfinite(weighted).all():
            raise RuntimeError("non_finite_l1_weighted_sketch")
        _, r = torch.linalg.qr(weighted, mode="reduced")
        if not torch.isfinite(r).all():
            raise RuntimeError("non_finite_l1_qr_factor")
        singular_values = torch.linalg.svdvals(r.float())
        if singular_values.numel() == 0 or not torch.isfinite(singular_values).all():
            raise RuntimeError("invalid_l1_basis_singular_values")
        largest = float(singular_values.max().item())
        tolerance = max(r.shape) * torch.finfo(torch.float32).eps * largest
        keep = singular_values > tolerance
        effective_rank = int(keep.sum().item())
        if effective_rank == 0:
            raise RuntimeError("zero_rank_l1_basis")
        smallest = float(singular_values[keep].min().item())
        condition = largest / max(smallest, 1e-12)
        if condition > self.condition_limit:
            raise RuntimeError(
                f"l1_basis_condition_exceeds_limit:{condition:.6g}>{self.condition_limit:.6g}"
            )
        self.r_inv = torch.linalg.pinv(r.float())
        self.last_dim = d
        self.last_diagnostics = {
            "calculation": "approximate_l1_woodruff",
            "n_rows": n,
            "n_features": d,
            "effective_rank": effective_rank,
            "condition_number": condition,
            "rank_tolerance": tolerance,
            "fit_count": self.fit_count,
            "refit": True,
            "sketch_dim": self.sketch_dim,
            "used_count_sketch": bool(n >= self.sketch_dim),
            "fallback": False,
            "fallback_reason": None,
        }

    def scores(self, rows: torch.Tensor, force_refit: bool = False) -> torch.Tensor:
        """Compute L1 leverage scores for each row."""
        _, d = rows.shape
        if rows.shape[0] <= 1:
            norms = torch.norm(rows.float(), p=1, dim=1)
            scores = (norms > 0).to(dtype=rows.dtype)
            self.last_diagnostics = {
                "calculation": "degenerate_single_row_l1_leverage",
                "n_rows": int(rows.shape[0]),
                "n_features": int(d),
                "effective_rank": int(bool(rows.numel()) and bool((norms > 0).any())),
                "condition_number": 1.0 if bool((norms > 0).any()) else None,
                "fit_count": self.fit_count,
                "refit": False,
                "fallback": False,
                "fallback_reason": None,
            }
            return scores
        before = self.fit_count
        if self.r_inv is None or force_refit or self.last_dim != d:
            try:
                self.fit(rows)
            except Exception as exc:
                self.r_inv = None
                self.last_dim = d
                self.last_diagnostics = {
                    "calculation": "approximate_l1_woodruff",
                    "n_rows": int(rows.shape[0]),
                    "n_features": int(d),
                    "fit_count": self.fit_count,
                    "refit": self.fit_count > before,
                    "failed": True,
                    "fallback": False,
                    "fallback_reason": str(exc),
                }
                raise RuntimeError(f"L1 leverage estimator failed: {exc}") from exc
        if self.r_inv is None:
            raise RuntimeError("L1 leverage estimator has no fitted basis")
        if self.fit_count == before:
            self.last_diagnostics = {
                **self.last_diagnostics,
                "n_rows": int(rows.shape[0]),
                "n_features": int(d),
                "refit": False,
                "fallback": False,
                "fallback_reason": None,
            }
        proj = rows.float() @ self.r_inv
        s = torch.norm(proj, p=1, dim=1).to(rows.dtype)
        if not torch.isfinite(s).all():
            raise RuntimeError("L1 leverage estimator produced non-finite scores")
        self._last_scores = s
        return s
