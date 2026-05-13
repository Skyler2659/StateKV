"""H2O: Heavy-Hitter Oracle — cumulative attention-score based KV cache eviction.

Zhang et al., NeurIPS 2023.
"""
import torch


def _is_dynamic_cache(pkv):
    return pkv is not None and not isinstance(pkv, (list, tuple))


def _to_legacy(pkv):
    if pkv is None:
        return None
    if _is_dynamic_cache(pkv):
        lyrs = getattr(pkv, "layers", None)
        if isinstance(lyrs, (list, tuple)) and len(lyrs) > 0:
            return tuple((lyr.keys, lyr.values) for lyr in lyrs)
        kc = getattr(pkv, "key_cache", None)
        vc = getattr(pkv, "value_cache", None)
        if isinstance(kc, list) and isinstance(vc, list):
            return tuple((kc[i], vc[i]) for i in range(len(kc)))
        if hasattr(pkv, "to_legacy_cache"):
            return pkv.to_legacy_cache()
    return pkv


def _back_to_original(original, items):
    if original is None:
        return None
    if _is_dynamic_cache(original):
        if hasattr(original, "layers"):
            for i, (k, v) in enumerate(items):
                if i < len(original.layers):
                    original.layers[i].keys = k
                    original.layers[i].values = v
                else:
                    original.update(k, v, i)
            return original
        if hasattr(original, "key_cache"):
            new_cache = type(original)()
            new_cache.key_cache = [k for k, v in items]
            new_cache.value_cache = [v for k, v in items]
            return new_cache
        if hasattr(type(original), "from_legacy_cache"):
            return type(original).from_legacy_cache(items)
    if isinstance(original, tuple):
        return tuple(items)
    return items


class H2OKVCache:
    """Heavy-Hitter Oracle: accumulated attention score per token per layer."""

    def __init__(self, cache_size=512, k_seq_dim=2, v_seq_dim=2):
        self.cache_size = int(cache_size)
        self.k_seq_dim = int(k_seq_dim)
        self.v_seq_dim = int(v_seq_dim)
        # Per-layer accumulated attention scores: layer_idx -> torch.Tensor [numel]
        self._acc_scores = {}
        self._steps = 0

    def _compute_attn_weights(self, layer_k, layer_v, layer_idx):
        """Compute current-step attention weights from Q_last (if available)."""
        import math

        import shared_q
        q_h = shared_q.LAST_QUERY_STATES.get(layer_idx)
        if q_h is None:
            return None
        head_dim = layer_v.shape[-1]
        q_vec = q_h.mean(dim=0).to(layer_v.device)       # [D]
        k_rows = layer_k[0].mean(dim=0)                   # [S, D]
        logits = torch.matmul(q_vec, k_rows.T) / max(head_dim ** 0.5, 1e-6)
        return torch.softmax(logits, dim=0)                # [S]

    def __call__(self, past_key_values):
        if past_key_values is None:
            return None
        pkv = _to_legacy(past_key_values)
        self._steps += 1
        items = []
        for layer_idx, (k, v) in enumerate(pkv):
            seq_len = k.size(self.k_seq_dim)
            # Accumulate attention weights for this layer
            attn = self._compute_attn_weights(k, v, layer_idx)
            if attn is not None:
                prev = self._acc_scores.get(layer_idx)
                if prev is None or prev.numel() < seq_len:
                    new_prev = torch.zeros(seq_len, device=k.device, dtype=attn.dtype)
                    if prev is not None:
                        new_prev[:prev.numel()] = prev
                    prev = new_prev
                prev[:seq_len] += attn.to(prev.device)
                self._acc_scores[layer_idx] = prev

            if seq_len <= self.cache_size:
                items.append((k, v))
                continue
            # Use accumulated scores to select top-k
            scores = self._acc_scores.get(layer_idx)
            if scores is None or scores.numel() < seq_len:
                items.append((k, v))
                continue
            keep = torch.topk(scores[:seq_len], self.cache_size).indices
            keep = keep.sort().values
            keep_k = keep.to(k.device)
            keep_v = keep.to(v.device)
            new_k = torch.index_select(k, self.k_seq_dim, keep_k)
            new_v = torch.index_select(v, self.v_seq_dim, keep_v)
            # Prune accumulated scores to match kept indices
            self._acc_scores[layer_idx] = scores[:seq_len][keep].clone()
            items.append((new_k, new_v))
        return _back_to_original(past_key_values, items)

    def evict_for_space(self, past_key_values, num_coming):
        if past_key_values is None:
            return None
        pkv = _to_legacy(past_key_values)
        seq_len = pkv[0][0].size(self.k_seq_dim)
        budget = max(1, self.cache_size - int(num_coming))
        if seq_len <= budget:
            return past_key_values
        old = self.cache_size
        try:
            self.cache_size = budget
            out = self.__call__(past_key_values)
        finally:
            self.cache_size = old
        return out

    def evict_range(self, past_key_values, start, end):
        if past_key_values is None:
            return None
        pkv = _to_legacy(past_key_values)
        seq_len = pkv[0][0].size(self.k_seq_dim)
        keep = torch.cat([
            torch.arange(0, start, device=pkv[0][0].device),
            torch.arange(end, seq_len, device=pkv[0][0].device),
        ])
        items = []
        for k, v in pkv:
            new_k = torch.index_select(k, self.k_seq_dim, keep.to(k.device))
            new_v = torch.index_select(v, self.v_seq_dim, keep.to(v.device))
            items.append((new_k, new_v))
        return _back_to_original(past_key_values, items)
