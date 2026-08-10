import json

import numpy as np
import torch

from statekv.budget_dynamics import (
    DirectBudgetController,
    LayerCacheView,
    allocate_layer_budgets,
    average_static_budgets,
    mask_overlap,
)
from statekv.oracle_policy_comparison import AttentionPolicyMemory


def _memory() -> AttentionPolicyMemory:
    memory = AttentionPolicyMemory((0, 1, 2), 4)
    layer_values = {
        0: [0.01, 0.85, 0.04, 0.04, 0.04, 0.02],
        1: [0.01, 0.20, 0.20, 0.20, 0.19, 0.20],
        2: [0.01, 0.50, 0.25, 0.12, 0.08, 0.04],
    }
    for layer, values in layer_values.items():
        memory.latest[layer] = dict(enumerate(values))
        memory.latest_by_head[layer] = {
            position: np.asarray([value, value], dtype=np.float64)
            for position, value in enumerate(values)
        }
        memory.cumulative[layer] = dict(memory.latest[layer])
        memory.window[layer] = [
            {position: value * 0.5 for position, value in enumerate(values)},
            dict(memory.latest[layer]),
        ]
    return memory


def _view() -> LayerCacheView:
    positions = tuple(range(6))
    values = {
        layer: torch.arange(6, dtype=torch.float32).reshape(1, 1, 6, 1)
        for layer in range(3)
    }
    return LayerCacheView(
        positions_by_layer={layer: positions for layer in range(3)},
        values_by_layer=values,
    )


def _controller(**kwargs) -> DirectBudgetController:
    return DirectBudgetController(
        core_budget=2,
        sink_size=1,
        recent_size=1,
        pooling_kernel=1,
        pooling_method="max",
        maximum_delta=1,
        **kwargs,
    )


def test_budget_projection_and_static_average_preserve_global_total() -> None:
    dynamic = allocate_layer_budgets({0: 1.0, 1: 4.0, 2: 2.0}, 2, 1)
    assert sum(dynamic.values()) == 6
    assert min(dynamic.values()) >= 1
    assert max(dynamic.values()) <= 3
    static = average_static_budgets(
        [dynamic, {0: 2, 1: 3, 2: 1}], 2, 1
    )
    assert sum(static.values()) == 6


def test_p1_variants_keep_direct_token_utility_fixed() -> None:
    memory = _memory()
    view = _view()
    static = {0: 1, 1: 2, 2: 3}
    decisions = {
        policy: _controller(static_budgets=static).select(
            policy, memory, view, cycle=3, sample_id="sample"
        )
        for policy in (
            "b2_uniform",
            "static_adaptive",
            "dynamic_b3",
            "layer_shuffled_b3",
            "stale_b3",
        )
    }
    reference = decisions["b2_uniform"].scores_by_layer
    for decision in decisions.values():
        assert sum(decision.requested_budgets.values()) == 6
        for layer in reference:
            np.testing.assert_allclose(
                decision.scores_by_layer[layer], reference[layer]
            )
    assert sorted(decisions["layer_shuffled_b3"].requested_budgets.values()) == sorted(
        decisions["layer_shuffled_b3"].dynamic_budgets.values()
    )


def test_stale_budget_age_and_overlap_are_auditable() -> None:
    controller = _controller(stale_lag=2)
    memory = _memory()
    view = _view()
    ages = [
        controller.select("stale_b3", memory, view, cycle, "sample").diagnostics[
            "budget_age"
        ]
        for cycle in range(4)
    ]
    assert ages == [0, 1, 2, 2]
    left = _controller().select("dynamic_b3", memory, view, 0, "sample")
    right = _controller().select("b2_uniform", memory, view, 0, "sample")
    overlap = mask_overlap(left.selection, right.selection)
    assert 0.0 <= overlap["mean_jaccard"] <= 1.0
    assert len(json.loads(overlap["jaccard_by_layer_json"])) == 3


def test_active_universe_caps_realized_but_not_requested_budget() -> None:
    view = LayerCacheView(
        positions_by_layer={
            0: tuple(range(6)),
            1: tuple(range(4)),
            2: tuple(range(5)),
        },
        values_by_layer={
            0: torch.ones((1, 1, 6, 1)),
            1: torch.ones((1, 1, 4, 1)),
            2: torch.ones((1, 1, 5, 1)),
        },
    )
    memory = _memory()
    decision = _controller().select("dynamic_b3", memory, view, 0, "sample")
    assert sum(decision.requested_budgets.values()) == 6
    assert sum(decision.realized_budgets.values()) <= 6
    assert decision.diagnostics["core_budget_shortfall"] >= 0
