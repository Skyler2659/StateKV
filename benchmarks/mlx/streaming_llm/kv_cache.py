import torch


def slice2d(x, start, end):
    return x[:, :, start:end, ...]


def slice3d(x, start, end):
    return x[:, :, :, start:end, ...]


def slice1d(x, start, end):
    return x[:, start:end, ...]


DIM_TO_SLICE = {
    1: slice1d,
    2: slice2d,
    3: slice3d,
}


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
    if original_cache is None:
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


def _preserve_container_type(original_cache, legacy_items):
    if isinstance(original_cache, tuple):
        legacy_cache = tuple(legacy_items)
    else:
        legacy_cache = list(legacy_items)
    return _restore_cache_type(original_cache if not isinstance(original_cache, (list, tuple)) else None, legacy_cache)


class StartRecentKVCache:
    def __init__(
        self,
        start_size=4,
        recent_size=512,
        k_seq_dim=2,
        v_seq_dim=2,
    ):
        print(f"StartRecentKVCache: {start_size}, {recent_size}")
        self.start_size = start_size
        self.recent_size = recent_size
        self.cache_size = start_size + recent_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.k_slice = DIM_TO_SLICE[k_seq_dim]
        self.v_slice = DIM_TO_SLICE[v_seq_dim]

    def __call__(self, past_key_values):
        if past_key_values is None:
            return None
        legacy_cache, original_cache = _to_legacy_cache(past_key_values)
        seq_len = legacy_cache[0][0].size(self.k_seq_dim)
        if seq_len <= self.cache_size:
            return past_key_values
        items = [
            [
                torch.cat(
                    [
                        self.k_slice(k, 0, self.start_size),
                        self.k_slice(k, seq_len - self.recent_size, seq_len),
                    ],
                    dim=self.k_seq_dim,
                ),
                torch.cat(
                    [
                        self.v_slice(v, 0, self.start_size),
                        self.v_slice(v, seq_len - self.recent_size, seq_len),
                    ],
                    dim=self.v_seq_dim,
                ),
            ]
            for k, v in legacy_cache
        ]
        return _preserve_container_type(legacy_cache if original_cache is None else original_cache, items)

    def evict_for_space(self, past_key_values, num_coming):
        if past_key_values is None:
            return None
        legacy_cache, original_cache = _to_legacy_cache(past_key_values)
        seq_len = legacy_cache[0][0].size(self.k_seq_dim)
        if seq_len + num_coming <= self.cache_size:
            return past_key_values
        items = [
            [
                torch.cat(
                    [
                        self.k_slice(k, 0, self.start_size),
                        self.k_slice(
                            k, seq_len - self.recent_size + num_coming, seq_len
                        ),
                    ],
                    dim=self.k_seq_dim,
                ),
                torch.cat(
                    [
                        self.v_slice(v, 0, self.start_size),
                        self.v_slice(
                            v, seq_len - self.recent_size + num_coming, seq_len
                        ),
                    ],
                    dim=self.v_seq_dim,
                ),
            ]
            for k, v in legacy_cache
        ]
        return _preserve_container_type(legacy_cache if original_cache is None else original_cache, items)

    def evict_range(self, past_key_values, start, end):
        if past_key_values is None:
            return None
        legacy_cache, original_cache = _to_legacy_cache(past_key_values)
        seq_len = legacy_cache[0][0].size(self.k_seq_dim)
        assert start <= end and end <= seq_len
        items = [
            [
                torch.cat(
                    [
                        self.k_slice(k, 0, start),
                        self.k_slice(k, end, seq_len),
                    ],
                    dim=self.k_seq_dim,
                ),
                torch.cat(
                    [
                        self.v_slice(v, 0, start),
                        self.v_slice(v, end, seq_len),
                    ],
                    dim=self.v_seq_dim,
                ),
            ]
            for k, v in legacy_cache
        ]
        return _preserve_container_type(legacy_cache if original_cache is None else original_cache, items)
