"""V-precision tiering for the QK-route, V-tier method (Family C).

Cold-tier V rows are stored at low precision: symmetric per-token-per-head
absmax quantization over groups of the head dimension, applied as
quantize->dequantize so the cache layout is unchanged while the values
carry exactly ``bits``-bit precision (algorithmic quality cost of the cold
tier; the memory claim is arithmetic, see the gate doc).
"""
from __future__ import annotations

from typing import Dict, FrozenSet, Mapping, Sequence, Set, Tuple

import torch


def quantize_dequantize(
    values: torch.Tensor, bits: int = 4, group: int = 64
) -> torch.Tensor:
    """Symmetric absmax quantize->dequantize over the last dimension.

    values: [..., dim]; each row is split into groups of ``group`` channels;
    per-group scale = absmax / qmax; integers clamped to [-qmax, qmax].
    Returns a tensor of the same shape and dtype whose values carry exactly
    ``bits``-bit precision (at most 2*qmax+1 distinct levels per group).
    """

    if int(bits) <= 0 or int(bits) > 8:
        raise ValueError("bits must be in [1, 8]")
    if int(group) <= 0:
        raise ValueError("group must be positive")
    qmax = float(2 ** (int(bits) - 1) - 1)
    dim = int(values.shape[-1])
    flat = values.reshape(-1, dim).float()
    pad = (int(group) - dim % int(group)) % int(group)
    if pad:
        flat = torch.nn.functional.pad(flat, (0, pad))
    grouped = flat.reshape(flat.shape[0], -1, int(group))
    scale = (grouped.abs().amax(dim=-1, keepdim=True) / qmax).clamp(min=1.0e-12)
    quantized = torch.round(grouped / scale).clamp(-qmax, qmax)
    restored = (quantized * scale).reshape(flat.shape[0], -1)
    if pad:
        restored = restored[:, :dim]
    return restored.reshape(values.shape).to(values.dtype)


def hot_cold_partition(
    selected_positions: Sequence[int],
    score_by_position: Mapping[int, float],
    hot_count: int,
    mandatory: Sequence[int],
) -> Tuple[FrozenSet[int], FrozenSet[int]]:
    """Partition the retained core into hot (FP16 V) and cold (tiered V).

    Hot = mandatory (sink/recent) union the top-``hot_count`` selected
    positions by score; cold = the remaining selected positions.
    Mandatory positions are never cold.
    """

    selected = {int(value) for value in selected_positions}
    protected = {int(value) for value in mandatory}
    ranked = sorted(
        selected - protected,
        key=lambda position: (
            -float(score_by_position.get(int(position), 0.0)),
            int(position),
        ),
    )
    hot = set(protected) | set(ranked[: int(hot_count)])
    cold = selected - hot
    return frozenset(hot & selected), frozenset(cold)


def tiered_bytes_per_token(
    kv_heads: int,
    head_dim: int,
    bits: int,
    group: int,
    key_bytes: int = 2,
) -> Tuple[float, float]:
    """Arithmetic active-memory model: (hot, cold) bytes per token per layer.

    K stays FP16 (key_bytes precision).  Hot V is FP16; cold V is ``bits``-
    bit plus one FP16 scale per ``group`` channels.
    """

    k_bytes = kv_heads * head_dim * key_bytes
    hot_v = kv_heads * head_dim * 2
    groups = -(-head_dim // int(group))
    cold_v = kv_heads * (head_dim * int(bits) / 8.0 + groups * 2)
    return float(k_bytes + hot_v), float(k_bytes + cold_v)
