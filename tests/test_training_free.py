import itertools

import pytest
import torch

from statekv.training_free import (
    direct_retained_set,
    fixed_rademacher_projection,
    query_continuity_decay,
    retained_action_signature,
    sketch_action_risk,
    update_state_sketch,
)


def _tiny_problem():
    attention = torch.tensor(
        [[0.08, 0.12, 0.18, 0.27, 0.35]], dtype=torch.float64
    )
    values = torch.tensor(
        [
            [
                [0.0, 1.0],
                [1.0, -1.0],
                [2.0, 0.5],
                [-1.0, 2.0],
                [3.0, -0.5],
            ]
        ],
        dtype=torch.float64,
    )
    return attention, values


def test_fixed_projection_is_deterministic_and_scaled() -> None:
    first = fixed_rademacher_projection(7, 5, seed=42)
    second = fixed_rademacher_projection(7, 5, seed=42)
    other = fixed_rademacher_projection(7, 5, seed=43)
    assert torch.equal(first, second)
    assert not torch.equal(first, other)
    assert torch.allclose(
        first.abs(), torch.full_like(first, 1.0 / (5.0**0.5))
    )


def test_state_sketch_uses_query_continuity_without_training() -> None:
    assert query_continuity_decay(
        torch.tensor([1.0, 0.0]), torch.tensor([2.0, 0.0])
    ) == pytest.approx(1.0)
    assert query_continuity_decay(
        torch.tensor([1.0, 0.0]), torch.tensor([-1.0, 0.0])
    ) == pytest.approx(0.0)
    updated = update_state_sketch(
        torch.tensor([2.0, -1.0]), torch.tensor([0.5, 0.25]), 0.5
    )
    assert torch.allclose(updated, torch.tensor([1.5, -0.25]))


def test_history_term_can_reverse_action_only_ordering() -> None:
    state = torch.tensor([-2.0, 0.0], dtype=torch.float64)
    smaller_energy = torch.tensor([0.5, 0.0], dtype=torch.float64)
    larger_energy = torch.tensor([1.0, 0.0], dtype=torch.float64)
    zero = torch.zeros_like(state)
    assert sketch_action_risk(zero, smaller_energy) < sketch_action_risk(
        zero, larger_energy
    )
    assert sketch_action_risk(state, larger_energy) < sketch_action_risk(
        state, smaller_energy
    )


def test_retained_signature_matches_explicit_attention_output() -> None:
    attention, values = _tiny_problem()
    retained = [0, 2, 4]
    signature = retained_action_signature(attention, values, retained)
    full = torch.sum(attention.unsqueeze(-1) * values, dim=-2)
    kept_attention = attention[:, retained]
    kept = torch.sum(
        kept_attention.unsqueeze(-1) * values[:, retained], dim=-2
    ) / kept_attention.sum(dim=-1, keepdim=True)
    assert torch.allclose(signature, (kept - full).reshape(-1))


def test_direct_search_reaches_tiny_exhaustive_optimum() -> None:
    attention, values = _tiny_problem()
    state = torch.tensor([0.25, -0.5], dtype=torch.float64)
    decision = direct_retained_set(
        attention,
        values,
        budget=3,
        state_sketch=state,
        mandatory_positions=[0],
        shortlist_ratio=10.0,
        max_swaps=20,
    )
    exhaustive = []
    for remainder in itertools.combinations(range(1, 5), 2):
        retained = (0,) + remainder
        signature = retained_action_signature(attention, values, retained)
        exhaustive.append(
            (float(sketch_action_risk(state, signature).item()), retained)
        )
    expected = min(exhaustive, key=lambda item: (item[0], item[1]))
    assert decision.retained_positions == expected[1]
    assert decision.objective == pytest.approx(expected[0])


def test_switch_penalty_can_keep_previous_set() -> None:
    attention, values = _tiny_problem()
    previous = (0, 1, 3)
    no_penalty = direct_retained_set(
        attention,
        values,
        budget=3,
        state_sketch=torch.zeros(2, dtype=torch.float64),
        previous_retained=previous,
        mandatory_positions=[0],
        shortlist_ratio=10.0,
        switch_penalty=0.0,
        max_swaps=20,
    )
    protected = direct_retained_set(
        attention,
        values,
        budget=3,
        state_sketch=torch.zeros(2, dtype=torch.float64),
        previous_retained=previous,
        mandatory_positions=[0],
        shortlist_ratio=10.0,
        switch_penalty=1000.0,
        max_swaps=20,
    )
    assert no_penalty.retained_positions != previous
    assert protected.retained_positions == previous
    assert not protected.refreshed
