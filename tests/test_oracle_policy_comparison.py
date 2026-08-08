from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from statekv.oracle_policy_comparison import AttentionPolicyMemory


def _record(values):
    return SimpleNamespace(
        oracle_attention_by_layer={
            0: torch.tensor([values], dtype=torch.float32)
        }
    )


def test_attention_memory_separates_latest_window_and_h2o() -> None:
    memory = AttentionPolicyMemory((0,), window_size=1)
    memory.update_record(_record([0.2, 0.8]), {0: (10, 11)})
    memory.update_record(_record([0.7, 0.3]), {0: (10, 11)})
    positions = [10, 11]
    latest = memory.score(0, positions, "attention", 1, "max")
    h2o = memory.score(0, positions, "h2o", 1, "max")
    snapkv = memory.score(0, positions, "snapkv", 1, "max")
    assert latest.tolist() == pytest.approx([0.7, 0.3])
    assert h2o.tolist() == pytest.approx([0.9, 1.1])
    assert snapkv.tolist() == pytest.approx([0.7, 0.3])


def test_attention_memory_rejects_position_misalignment() -> None:
    memory = AttentionPolicyMemory((0,), window_size=2)
    with pytest.raises(RuntimeError, match="misaligned"):
        memory.update_record(_record([0.2, 0.8]), {0: (10,)})
