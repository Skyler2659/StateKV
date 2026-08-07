import pytest
import torch

from statekv.core.actions import (
    functional_history_state,
    set_level_attention_delta,
)


def test_functional_history_state_is_boundary_displacement() -> None:
    reference = torch.tensor([[1.0, 2.0, 3.0]])
    history = torch.tensor([[1.5, 1.0, 4.0]])
    assert torch.equal(
        functional_history_state(history, reference),
        torch.tensor([[0.5, -1.0, 1.0]]),
    )


def test_functional_history_state_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="equal shape"):
        functional_history_state(torch.zeros(2), torch.zeros(3))


def test_set_level_attention_delta_matches_explicit_renormalization() -> None:
    attention = torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float64)
    values = torch.tensor(
        [[[1.0, 0.0], [0.0, 2.0], [3.0, 1.0], [2.0, 4.0]]],
        dtype=torch.float64,
    )
    retained = [0, 2]
    predicted = set_level_attention_delta(attention, values, retained)

    full = torch.sum(attention.unsqueeze(-1) * values, dim=-2)
    retained_attention = attention[:, retained]
    retained_output = torch.sum(
        retained_attention.unsqueeze(-1) * values[:, retained, :], dim=-2
    ) / retained_attention.sum(dim=-1, keepdim=True)
    assert torch.allclose(predicted, retained_output - full, atol=1.0e-12)


def test_set_level_attention_delta_rejects_zero_retained_mass() -> None:
    attention = torch.tensor([0.0, 1.0])
    values = torch.tensor([[1.0], [2.0]])
    with pytest.raises(ValueError, match="numerically zero"):
        set_level_attention_delta(attention, values, [0])
