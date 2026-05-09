"""Cache helpers + SlidingWindowKVCache baseline — no model or PyTorch dependency beyond torch."""

import torch


# ── DynamicCache ↦ legacy converters (multiversion HF compat) ──────────

def get_kv_seq_len(past_key_values, k_seq_dim: int):
    if hasattr(past_key_values, "get_seq_length"):
        return int(past_key_values.get_seq_length())
    return int(past_key_values[0][0].size(k_seq_dim))


def _to_legacy_cache(past_key_values):
    if past_key_values is None or isinstance(past_key_values, (list, tuple)):
        return past_key_values, None
    if hasattr(past_key_values, "to_legacy_cache"):
        try:
            return past_key_values.to_legacy_cache(), past_key_values
        except Exception:
            pass
    lyrs = getattr(past_key_values, "layers", None)
    if isinstance(lyrs, (list, tuple)) and len(lyrs) > 0:
        return tuple((lyr.keys, lyr.values) for lyr in lyrs), past_key_values
    for k_name, v_name in (("key_cache", "value_cache"),
                            ("_key_cache", "_value_cache")):
        kc = getattr(past_key_values, k_name, None)
        vc = getattr(past_key_values, v_name, None)
        if isinstance(kc, (list, tuple)) and isinstance(vc, (list, tuple)) and len(kc) == len(vc):
            return tuple((kc[i], vc[i]) for i in range(len(kc))), past_key_values
    return past_key_values, None


def _restore_cache_type(original_cache, legacy_cache):
    if original_cache is None or isinstance(original_cache, (list, tuple)):
        return legacy_cache
    if hasattr(type(original_cache), "from_legacy_cache"):
        try:
            return type(original_cache).from_legacy_cache(legacy_cache)
        except Exception:
            pass
    if hasattr(original_cache, "layers"):
        for i, (k, v) in enumerate(legacy_cache):
            if i < len(original_cache.layers):
                original_cache.layers[i].keys = k
                original_cache.layers[i].values = v
            else:
                original_cache.update(k, v, i)
        return original_cache
    return legacy_cache


def slice_tensor_by_dim(x: torch.Tensor, dim: int, start: int, end: int):
    idx = [slice(None)] * x.dim()
    idx[dim] = slice(start, end)
    return x[tuple(idx)]


# ── Sliding-window baseline ────────────────────────────────────────────

class SlidingWindowKVCache:
    """Recent-only baseline: keep the latest *cache_size* tokens, no sinks."""

    def __init__(self, cache_size: int, k_seq_dim: int, v_seq_dim: int):
        self.cache_size = int(cache_size)
        self.k_seq_dim = int(k_seq_dim)
        self.v_seq_dim = int(v_seq_dim)

    def __call__(self, past_key_values):
        if past_key_values is None:
            return None
        legacy_cache, original_cache = _to_legacy_cache(past_key_values)
        seq_len = get_kv_seq_len(legacy_cache, self.k_seq_dim)
        if seq_len <= self.cache_size:
            return past_key_values
        keep_from = seq_len - self.cache_size
        items = [
            (
                slice_tensor_by_dim(k, self.k_seq_dim, keep_from, seq_len),
                slice_tensor_by_dim(v, self.v_seq_dim, keep_from, seq_len),
            )
            for k, v in legacy_cache
        ]
        return _restore_cache_type(original_cache, tuple(items))

    def evict_for_space(self, past_key_values, num_coming):
        if past_key_values is None:
            return None
        legacy_cache, original_cache = _to_legacy_cache(past_key_values)
        seq_len = get_kv_seq_len(legacy_cache, self.k_seq_dim)
        if seq_len + num_coming <= self.cache_size:
            return past_key_values
        keep_from = max(0, seq_len - self.cache_size + num_coming)
        items = [
            (
                slice_tensor_by_dim(k, self.k_seq_dim, keep_from, seq_len),
                slice_tensor_by_dim(v, self.v_seq_dim, keep_from, seq_len),
            )
            for k, v in legacy_cache
        ]
        return _restore_cache_type(original_cache, tuple(items))
