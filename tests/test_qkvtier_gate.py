"""QK-route, V-tier method gate machinery tests."""
from __future__ import annotations

import pytest
import torch

from statekv.value_tier import (
    hot_cold_partition,
    quantize_dequantize,
    tiered_bytes_per_token,
)


def test_quantize_dequantize_precision_and_shape() -> None:
    torch.manual_seed(0)
    values = torch.randn(1, 8, 17, 128, dtype=torch.float16)
    restored = quantize_dequantize(values, bits=4, group=64)
    assert restored.shape == values.shape
    assert restored.dtype == values.dtype
    # at most 2*7+1 distinct levels per (row, group)
    flat_v = values.reshape(-1, 128).float()
    flat_r = restored.reshape(-1, 128).float()
    for row in range(flat_v.shape[0]):
        for group_index in range(2):
            distinct = torch.unique(
                flat_r[row, group_index * 64 : (group_index + 1) * 64]
            )
            assert len(distinct) <= 15
            scale = flat_v[row, group_index * 64 : (group_index + 1) * 64].abs().max() / 7.0
            error = (
                flat_r[row, group_index * 64 : (group_index + 1) * 64]
                - flat_v[row, group_index * 64 : (group_index + 1) * 64]
            ).abs().max()
            # quantization error <= scale/2 plus one fp16 ulp from the
            # final dtype cast
            assert float(error) <= float(scale) / 2 + 0.004


def test_quantize_dequantize_idempotent_and_validates() -> None:
    torch.manual_seed(1)
    values = torch.randn(4, 128)
    once = quantize_dequantize(values, bits=4, group=64)
    twice = quantize_dequantize(once, bits=4, group=64)
    assert torch.allclose(once, twice, atol=1e-6)
    with pytest.raises(ValueError, match="bits"):
        quantize_dequantize(values, bits=0)
    with pytest.raises(ValueError, match="group"):
        quantize_dequantize(values, bits=4, group=0)
    # fp16 passthrough at 8 bits is nearly lossless
    near = quantize_dequantize(values, bits=8, group=64)
    assert (near - values).abs().max() < 0.02


def test_hot_cold_partition() -> None:
    selected = [4, 5, 6, 7, 8, 9]
    scores = {4: 0.9, 5: 0.1, 6: 0.8, 7: 0.7, 8: 0.6, 9: 0.5}
    hot, cold = hot_cold_partition(selected, scores, 2, mandatory=[9])
    assert 9 in hot  # mandatory protected regardless of score
    assert hot == frozenset({4, 6, 9})  # top-2 by score + mandatory
    assert cold == frozenset({5, 7, 8})
    assert hot.isdisjoint(cold)
    assert hot | cold == frozenset(selected)
    # deterministic ties
    tie_scores = {4: 0.5, 5: 0.5, 6: 0.5, 7: 0.5, 8: 0.5, 9: 0.5}
    hot2, _ = hot_cold_partition(selected, tie_scores, 3, mandatory=[])
    assert hot2 == frozenset({4, 5, 6})


def test_tiered_bytes_model_matches_gate_doc() -> None:
    hot, cold = tiered_bytes_per_token(8, 128, 4, 64)
    assert hot == pytest.approx(4096.0)
    assert cold == pytest.approx(2592.0)
    # 352 tiered (96 hot) memory-matched to 256 fp16 within 1%
    baseline = 256 * hot
    tiered = 96 * hot + 256 * cold
    assert tiered / baseline == pytest.approx(1.008, abs=0.001)
