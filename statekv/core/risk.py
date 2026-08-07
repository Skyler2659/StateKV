"""Finite-action response and state-conditioned output-risk primitives."""
from __future__ import annotations

from typing import Callable

import torch


JacobianVectorProduct = Callable[
    [torch.Tensor, torch.Tensor], torch.Tensor
]


def reference_kl(
    reference_logits: torch.Tensor,
    candidate_logits: torch.Tensor,
) -> torch.Tensor:
    """Return ``KL(p_reference || p_candidate)`` along the vocabulary axis."""

    if reference_logits.shape != candidate_logits.shape:
        raise ValueError("reference and candidate logits must have equal shape")
    log_reference = torch.log_softmax(reference_logits, dim=-1)
    log_candidate = torch.log_softmax(candidate_logits, dim=-1)
    probability_reference = torch.exp(log_reference)
    return torch.sum(
        probability_reference * (log_reference - log_candidate), dim=-1
    )


def reference_kl_increment(
    reference_logits: torch.Tensor,
    state_logits: torch.Tensor,
    candidate_logits: torch.Tensor,
) -> torch.Tensor:
    """Measure candidate risk as the increment over the current state."""

    return reference_kl(reference_logits, candidate_logits) - reference_kl(
        reference_logits, state_logits
    )


def fisher_vector_product(
    probability: torch.Tensor,
    vector: torch.Tensor,
) -> torch.Tensor:
    """Apply the categorical Fisher matrix without materializing it."""

    if probability.shape != vector.shape:
        raise ValueError("probability and vector must have equal shape")
    if torch.any(probability < 0):
        raise ValueError("probabilities must be non-negative")
    mass = probability.sum(dim=-1)
    if not torch.allclose(mass, torch.ones_like(mass), atol=1.0e-6, rtol=1.0e-6):
        raise ValueError("probabilities must sum to one")
    mean = torch.sum(probability * vector, dim=-1, keepdim=True)
    return probability * (vector - mean)


def midpoint_path_response(
    jvp: JacobianVectorProduct,
    state: torch.Tensor,
    action_response: torch.Tensor,
    *,
    segments: int = 2,
) -> torch.Tensor:
    """Approximate a finite downstream response with midpoint JVP probes."""

    if state.shape != action_response.shape:
        raise ValueError("state and action response must have equal shape")
    if int(segments) < 1:
        raise ValueError("segments must be positive")
    nodes = (
        torch.arange(
            int(segments), device=state.device, dtype=state.dtype
        )
        + 0.5
    ) / float(segments)
    responses = [
        jvp(state + node * action_response, action_response)
        for node in nodes
    ]
    if not responses:
        raise RuntimeError("midpoint response produced no probes")
    return torch.stack(responses, dim=0).mean(dim=0)


def state_conditioned_quadratic_risk(
    reference_logits: torch.Tensor,
    state_logits: torch.Tensor,
    delta_logits: torch.Tensor,
) -> torch.Tensor:
    """Return the local second-order KL-risk increment at the current state."""

    if not (
        reference_logits.shape == state_logits.shape == delta_logits.shape
    ):
        raise ValueError("reference, state, and delta logits must have equal shape")
    probability_reference = torch.softmax(reference_logits, dim=-1)
    probability_state = torch.softmax(state_logits, dim=-1)
    gradient = probability_state - probability_reference
    linear = torch.sum(gradient * delta_logits, dim=-1)
    curvature = torch.sum(
        delta_logits
        * fisher_vector_product(probability_state, delta_logits),
        dim=-1,
    )
    return linear + 0.5 * curvature
