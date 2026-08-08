from pathlib import Path

import numpy as np
import torch
import yaml

from statekv.cheap_policy import CHEAP_POLICIES, CheapPolicyContext
from statekv.oracle_policy_comparison import (
    AttentionPolicyMemory,
    _selection_from_scores,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/stages/cheap_policy_freegen_qwen3_8b_n10_protocol.yaml"


class _Backing:
    def __init__(self, values: torch.Tensor) -> None:
        self._values = values

    def positions(self):
        return list(range(int(self._values.shape[2])))

    def layer_arrays(self, layer):
        return torch.zeros_like(self._values), self._values


def _memory() -> AttentionPolicyMemory:
    memory = AttentionPolicyMemory((0,), 4)
    positions = range(6)
    values = np.asarray([0.05, 0.40, 0.30, 0.10, 0.10, 0.05])
    memory.latest[0] = dict(zip(positions, values.tolist()))
    memory.latest_by_head[0] = {
        position: np.asarray([value, value], dtype=np.float64)
        for position, value in zip(positions, values)
    }
    memory.cumulative[0] = dict(memory.latest[0])
    memory.window[0] = [dict(memory.latest[0]), dict(memory.latest[0])]
    return memory


def test_protocol_runs_all_paths_without_old_baselines() -> None:
    protocol = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    assert tuple(protocol["policies"]) == CHEAP_POLICIES
    assert protocol["comparison_contract"]["old_policies_rerun"] is False
    assert protocol["model_overrides"]["name"] == "mlx-community/Qwen3-8B-4bit"
    assert len(protocol["sample_ids"]) == 10


def test_adaptive_budget_preserves_global_core_count() -> None:
    scores = {
        0: np.asarray([0.9, 0.1, 0.0, 0.0]),
        1: np.asarray([0.25, 0.25, 0.25, 0.25]),
        2: np.asarray([0.4, 0.3, 0.2, 0.1]),
    }
    budgets = CheapPolicyContext._adaptive_budgets(
        scores,
        np.arange(4, dtype=np.int64),
        core_budget=2,
        maximum_delta=1,
        eligible_count=4,
    )
    assert sum(budgets.values()) == 6
    assert min(budgets.values()) >= 1
    assert max(budgets.values()) <= 3


def test_set_output_proxy_prefers_retaining_high_attention_set() -> None:
    memory = _memory()
    values = torch.tensor(
        [[[[0.0], [3.0], [2.0], [1.0], [-1.0], [0.0]]]],
        dtype=torch.float32,
    )
    backing = _Backing(values)
    positions = list(range(6))
    eligible = [1, 2, 3, 4]
    scores = {0: np.zeros(6, dtype=np.float64)}
    panel = {
        "high": _selection_from_scores(
            "high", positions, eligible, {0: (1, 2)}, scores
        ),
        "low": _selection_from_scores(
            "low", positions, eligible, {0: (3, 4)}, scores
        ),
    }
    context = CheapPolicyContext(
        core_budget=2,
        sink_size=1,
        recent_size=1,
        pooling_kernel=1,
        pooling_method="max",
        output_diagnostic_layers=(0,),
    )
    risks = context._set_output_scores(
        panel, memory, backing, positions, eligible
    )
    assert risks["high"] < risks["low"]
