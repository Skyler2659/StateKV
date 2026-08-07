import torch

from kvbench.backends.huggingface import AttentionAccumulator, HFCacheState, HuggingFaceBackend
from kvbench.types import SelectionDecision


def test_observation_rows_are_padded_before_cache_pruning():
    accumulator = AttentionAccumulator(num_kv_heads=1, observation_window=8)
    accumulator.update((torch.ones(1, 1, 3, 3),))
    accumulator.update((torch.ones(1, 1, 2, 5),))
    accumulator.prune({0: torch.tensor([0, 4])})
    signals = accumulator.signals()
    assert signals.observation_by_layer[0].shape == (1, 2)
    # Queries from the first chunk had no future key 4 and therefore contribute zero.
    assert signals.observation_by_layer[0][0, 1].item() == 2.0


def test_identity_cache_decision_is_a_true_noop():
    backend = object.__new__(HuggingFaceBackend)
    backend.device = torch.device("cpu")
    key = torch.randn(1, 1, 4, 2)
    value = torch.randn(1, 1, 4, 2)
    state = HFCacheState(
        past_key_values=((key, value),),
        position_maps={0: torch.arange(4)},
        logical_next_position=4,
        attention=AttentionAccumulator(1, 2),
    )
    decision = SelectionDecision(
        layer=0,
        universe_positions=[0, 1, 2, 3],
        selected_rows=[0, 1, 2, 3],
        selected_positions=[0, 1, 2, 3],
        requested_budget=4,
        effective_budget=4,
        mandatory_positions=[],
        selectable_budget=4,
        budget_scope="total_kv",
        budget_unit="shared_token_positions",
    )
    assert backend.apply_decisions(state, [decision]) == 0.0
    assert state.past_key_values[0][0] is key
    assert state.past_key_values[0][1] is value

