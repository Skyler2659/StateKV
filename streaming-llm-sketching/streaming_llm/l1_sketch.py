import torch


def _safe_exp_samples(size, device, dtype):
    # Always generate in float32 — 1e-8 clamp is subnormal in float16
    u = torch.rand(size, device=device, dtype=torch.float32).clamp_(1e-8, 1 - 1e-8)
    return -torch.log(1.0 - u).to(dtype)


class CountSketch:
    def __init__(self, sketch_dim, seed=None):
        self.sketch_dim = int(sketch_dim)
        self.seed = seed
        self.hash_buckets = None
        self.signs = None
        self._generator = None
        if self.seed is not None:
            self._generator = torch.Generator(device="cpu")
            self._generator.manual_seed(int(self.seed))

    def _ensure_capacity(self, n, device):
        if self.hash_buckets is not None and self.hash_buckets.numel() >= n:
            return
        old_n = 0 if self.hash_buckets is None else int(self.hash_buckets.numel())
        grow_n = int(n - old_n)
        new_buckets = torch.randint(
            0, self.sketch_dim, (grow_n,), generator=self._generator)
        new_signs = (
            torch.randint(0, 2, (grow_n,), generator=self._generator, dtype=torch.int64)
            * 2 - 1)
        new_buckets = new_buckets.to(device)
        new_signs = new_signs.to(device=device, dtype=torch.float32)
        if self.hash_buckets is None:
            self.hash_buckets = new_buckets
            self.signs = new_signs
        else:
            self.hash_buckets = torch.cat([self.hash_buckets, new_buckets], dim=0)
            self.signs = torch.cat([self.signs, new_signs], dim=0)

    def apply(self, rows_x_d):
        n, d = rows_x_d.shape
        device = rows_x_d.device
        self._ensure_capacity(n, device)
        buckets = self.hash_buckets[:n]
        signs = self.signs[:n].to(rows_x_d.dtype)
        sketch = torch.zeros(self.sketch_dim, d, device=device, dtype=rows_x_d.dtype)
        sketch.scatter_add_(
            0, buckets.unsqueeze(-1).expand(-1, d),
            rows_x_d * signs.unsqueeze(-1))
        return sketch


class L1SubspaceEmbedding:
    def __init__(self, sketch_dim, seed=None):
        self.sketch_dim = int(sketch_dim)
        self.count_sketch = CountSketch(self.sketch_dim, seed=seed)

    def embed(self, v_rows):
        n = v_rows.shape[0]
        weighted = v_rows / _safe_exp_samples((n, 1), v_rows.device, v_rows.dtype)
        return self.count_sketch.apply(weighted)


class L1LeverageScoreEstimator:
    def __init__(self, sketch_dim=1024, seed=None):
        self.sketch_dim = int(sketch_dim)
        self.seed = seed
        self.embedding = L1SubspaceEmbedding(self.sketch_dim, seed=seed)
        self.r_inv = None
        self.last_dim = None

    def update_basis(self, v_rows):
        n, d = v_rows.shape
        if n <= 1:
            self.r_inv = None
            return
        # When n < sketch_dim most sketch buckets are empty → rank-deficient.
        # Skip CountSketch and directly weight V with Exp(1) for exact ℓ₁ basis.
        if n < self.embedding.count_sketch.sketch_dim:
            # Force float32 for Exp samples — 1e-8 is subnormal in float16
            w = v_rows.float() / _safe_exp_samples((n, 1), v_rows.device, torch.float32)
        else:
            w = self.embedding.embed(v_rows).float()
        # Guard against NaN from upstream FP16 operations
        if torch.isnan(w).any():
            self.r_inv = None
            return
        _, r = torch.linalg.qr(w, mode="reduced")
        if torch.isnan(r).any():
            self.r_inv = None
            return
        # Adaptive jitter — strong enough for near-square matrices
        jit = max(1e-4, r.diag().abs().max().item() * 1e-6)
        r = r + torch.eye(r.shape[0], device=r.device, dtype=r.dtype) * jit
        try:
            self.r_inv = torch.linalg.inv(r)
        except torch._C._LinAlgError:
            r = r + torch.eye(r.shape[0], device=r.device, dtype=r.dtype) * 1e-2
            self.r_inv = torch.linalg.inv(r)
        self.last_dim = d

    def scores(self, v_rows, force_refit=False):
        _, d = v_rows.shape
        if self.r_inv is None or force_refit or self.last_dim != d:
            self.update_basis(v_rows)
        if self.r_inv is None:
            return torch.norm(v_rows, p=1, dim=1)
        proj = v_rows.float() @ self.r_inv
        return torch.norm(proj, p=1, dim=1).to(v_rows.dtype)


def compute_reweight(scores, keep_indices):
    probs = scores / (scores.sum() + 1e-8)
    picked = probs[keep_indices].clamp_min(1e-8)
    return 1.0 / picked
