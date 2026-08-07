import torch

from statekv.core.risk import (
    fisher_vector_product,
    midpoint_path_response,
    reference_kl,
    reference_kl_increment,
    state_conditioned_quadratic_risk,
)


def test_reference_kl_increment_uses_current_state_as_baseline() -> None:
    reference = torch.tensor([1.0, 0.0, -1.0], dtype=torch.float64)
    state = torch.tensor([0.8, 0.2, -1.0], dtype=torch.float64)
    candidate = torch.tensor([0.7, 0.1, -0.8], dtype=torch.float64)
    assert torch.allclose(
        reference_kl_increment(reference, state, candidate),
        reference_kl(reference, candidate) - reference_kl(reference, state),
    )


def test_fisher_vector_product_matches_dense_matrix() -> None:
    probability = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
    vector = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64)
    dense = torch.diag(probability) - torch.outer(probability, probability)
    assert torch.allclose(
        fisher_vector_product(probability, vector), dense @ vector
    )


def test_two_midpoints_exactly_integrate_quadratic_response() -> None:
    state = torch.tensor([1.0], dtype=torch.float64)
    action = torch.tensor([2.0], dtype=torch.float64)

    def jvp(point: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
        return 2.0 * point * direction

    response = midpoint_path_response(jvp, state, action, segments=2)
    assert torch.allclose(response, torch.tensor([8.0], dtype=torch.float64))


def test_quadratic_risk_matches_explicit_fisher_form() -> None:
    reference = torch.tensor([1.0, 0.0, -0.5], dtype=torch.float64)
    state = torch.tensor([0.7, 0.2, -0.4], dtype=torch.float64)
    delta = torch.tensor([0.1, -0.2, 0.3], dtype=torch.float64)
    p_ref = torch.softmax(reference, dim=-1)
    p_state = torch.softmax(state, dim=-1)
    fisher = torch.diag(p_state) - torch.outer(p_state, p_state)
    expected = (p_state - p_ref) @ delta + 0.5 * delta @ fisher @ delta
    assert torch.allclose(
        state_conditioned_quadratic_risk(reference, state, delta), expected
    )
