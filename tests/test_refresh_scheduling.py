import numpy as np
import torch

from statekv.budget_dynamics import (
    DirectBudgetController,
    LayerCacheView,
    boundary_margin_by_layer,
    core_churn_by_layer,
    coverage_mass_by_layer,
    score_tv_by_layer,
)
from statekv.oracle_policy_comparison import AttentionPolicyMemory
from statekv.selectors import CoreSelection, LayerSelection


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


def _view(positions=tuple(range(6))) -> LayerCacheView:
    return LayerCacheView(
        positions_by_layer={layer: tuple(positions) for layer in range(3)},
        values_by_layer={
            layer: torch.arange(6, dtype=torch.float32).reshape(1, 1, 6, 1)[
                :, :, : len(positions), :
            ]
            for layer in range(3)
        },
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


def _cores(decision) -> dict:
    return {
        layer: set(int(value) for value in layer_selection.selected_positions)
        for layer, layer_selection in decision.selection.by_layer.items()
    }


def _selection(layer_cores, eligible) -> CoreSelection:
    return CoreSelection(
        strategy="toy",
        horizon_condition=None,
        by_layer={
            layer: LayerSelection(
                layer=layer,
                selected_positions=sorted(cores),
                eligible_positions=sorted(eligible[layer]),
                aggregate_scores=[],
            )
            for layer, cores in layer_cores.items()
        },
    )


def test_stale_selection_requires_a_refreshed_ranking() -> None:
    controller = _controller()
    try:
        controller.stale_selection(_view(), 1, "sample")
    except RuntimeError:
        return
    raise AssertionError("stale selection without refresh must fail")


def test_refresh_freezes_and_stale_replays_same_core_on_same_view() -> None:
    controller = _controller()
    memory = _memory()
    view = _view()
    fresh = controller.refresh_scores("attention", memory, view, 0, "sample")
    assert controller.frozen is not None
    assert controller.frozen.cycle == 0
    stale = controller.stale_selection(view, 1, "sample")
    assert _cores(stale) == _cores(fresh)
    assert stale.diagnostics["budget_age"] == 1
    assert stale.diagnostics["ranking_refresh_cycle"] == 0
    assert sum(stale.requested_budgets.values()) == 3 * controller.core_budget


def test_stale_selection_respects_budget_and_irreversibility_after_eviction() -> None:
    controller = _controller()
    memory = _memory()
    controller.refresh_scores("attention", memory, _view(), 0, "sample")
    # Position 1 (the top-scored token) was evicted; positions 6 and 7 were
    # created after the freeze and carry no frozen score.
    shifted = _view((0, 2, 3, 4, 5, 6, 7))
    stale = controller.stale_selection(shifted, 4, "sample")
    for layer in range(3):
        active = set(shifted.positions_by_layer[layer])
        core = _cores(stale)[layer]
        assert core <= active
        assert len(core) <= controller.core_budget
        assert 6 not in core and 7 not in core
        # The stale core is the top-2 by frozen score among covered eligible
        # positions {2, 3, 4, 5}; the frozen ranking puts 5 last.
        assert 5 not in core
    assert stale.diagnostics["budget_age"] == 4


def test_stale_core_is_stable_when_only_the_recent_window_slides() -> None:
    controller = _controller()
    memory = _memory()
    fresh = controller.refresh_scores("attention", memory, _view(), 0, "sample")
    # One new token appended; the position that slid out of the mandatory
    # recent window (5) keeps its frozen (zero) score and stays eligible.
    extended = _view((0, 1, 2, 3, 4, 5, 6))
    stale = controller.stale_selection(extended, 1, "sample")
    assert _cores(stale) == _cores(fresh)


def test_feature_helpers_match_hand_computed_values() -> None:
    positions = {0: (1, 2, 3, 4)}
    scores = {0: np.asarray([0.5, 0.4, 0.3, 0.1])}
    eligible = {0: (1, 2, 3, 4)}
    margins = boundary_margin_by_layer(scores, positions, eligible, core_budget=2)
    assert abs(margins[0] - 0.1) < 1.0e-12

    previous = _selection({0: {1, 2}}, {0: (1, 2, 3)})
    current = _selection({0: {2, 3}}, {0: (1, 2, 3)})
    churn = core_churn_by_layer(previous, current)
    assert abs(churn[0] - (1.0 - 1.0 / 3.0)) < 1.0e-12

    tv = score_tv_by_layer(
        {0: np.asarray([0.5, 0.5])},
        {0: (1, 2)},
        {0: np.asarray([1.0, 0.0])},
        {0: (1, 2)},
    )
    assert abs(tv[0] - 0.5) < 1.0e-12
    shifted = score_tv_by_layer(
        {0: np.asarray([0.5, 0.5])},
        {0: (1, 2)},
        {0: np.asarray([0.25, 0.75])},
        {0: (2, 3)},
    )
    assert abs(shifted[0] - 0.75) < 1.0e-12

    coverage = coverage_mass_by_layer(
        {0: np.asarray([0.5, 0.3, 0.2])},
        {0: (1, 2, 3)},
        _selection({0: {1, 2}}, {0: (1, 2, 3)}),
    )
    assert abs(coverage[0] - 0.8) < 1.0e-12


def test_shallow_clone_prune_and_forward_do_not_touch_source_state() -> None:
    import mlx.core as mx
    from mlx_lm.models.cache import KVCache

    from statekv.backend_mlx import MLXReplayState, MLXTemporalModel
    from statekv.config import CacheDiscoveryConfig

    layer_cache = KVCache()
    layer_cache.update_and_fetch(
        mx.random.normal((1, 1, 8, 2)), mx.random.normal((1, 1, 8, 2))
    )
    layer_cache.logical_offset = 8
    state = MLXReplayState(
        [layer_cache],
        {0: torch.arange(8, dtype=torch.long)},
        8,
    )
    keys_before = np.asarray(layer_cache.keys[:, :, :8, :].astype(mx.float32))
    values_before = np.asarray(layer_cache.values[:, :, :8, :].astype(mx.float32))

    clone = MLXTemporalModel.shallow_clone_state(state)
    assert clone.cache[0].keys is layer_cache.keys
    selection = CoreSelection(
        strategy="toy",
        horizon_condition=None,
        by_layer={
            0: LayerSelection(
                layer=0,
                selected_positions=[2, 3],
                eligible_positions=[2, 3, 4, 5, 6],
                aggregate_scores=[],
            )
        },
    )
    cache_config = CacheDiscoveryConfig(
        total_budget=8, sink_size=1, recent_size=2, selected_core_budget=2
    )
    model = MLXTemporalModel.__new__(MLXTemporalModel)
    model.apply_selection_in_place(clone, selection, cache_config=cache_config)
    assert clone.cache[0].keys is not layer_cache.keys
    assert clone.position_maps[0].tolist() == [0, 2, 3, 7]
    # A decode step on the pruned clone must allocate fresh buffers instead of
    # writing into the source state's shared storage.
    clone.cache[0].update_and_fetch(
        mx.zeros((1, 1, 1, 2)), mx.zeros((1, 1, 1, 2))
    )

    assert int(layer_cache.offset) == 8
    assert int(layer_cache.logical_offset) == 8
    assert state.logical_next_position == 8
    assert state.position_maps[0].tolist() == list(range(8))
    np.testing.assert_array_equal(
        np.asarray(layer_cache.keys[:, :, :8, :].astype(mx.float32)), keys_before
    )
    np.testing.assert_array_equal(
        np.asarray(layer_cache.values[:, :, :8, :].astype(mx.float32)),
        values_before,
    )
