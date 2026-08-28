"""State-copy primitives for branch-based causal evaluations."""
from __future__ import annotations

from typing import Any

import numpy as np


def clone_mlx_state(state: Any) -> Any:
    """Deep-copy a mutable MLX cache before evaluating a branch."""

    import mlx.core as mx
    from mlx_lm.models.cache import KVCache

    from statekv.backend_mlx import MLXReplayState

    caches = []
    for cache in state.cache:
        offset = int(cache.offset)
        cloned = KVCache()
        cloned.state = (
            mx.array(np.asarray(cache.keys[:, :, :offset, :]).copy()),
            mx.array(np.asarray(cache.values[:, :, :offset, :]).copy()),
        )
        cloned.logical_offset = int(cache.logical_offset)
        caches.append(cloned)
    return MLXReplayState(
        cache=caches,
        position_maps={
            int(layer): positions.detach().clone()
            for layer, positions in state.position_maps.items()
        },
        logical_next_position=int(state.logical_next_position),
    )
