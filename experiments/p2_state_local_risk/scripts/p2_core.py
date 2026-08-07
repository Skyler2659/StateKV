"""Mathematical primitives for P2 state-local risk and attribution."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
P0_DIR = ROOT / "experiments/p0_v2_fixed_boundary/scripts"
P1_DIR = ROOT / "experiments/p1_state_conditioned/scripts"
for value in (P0_DIR, P1_DIR):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from p0_v2_core import (  # noqa: E402
    atomic_frame,
    atomic_json,
    exact_kl,
    fisher_variance,
    prefixed_metrics,
    ranking_metrics,
    sha256_array,
    sha256_file,
    stable_softmax,
    vector_metrics,
)
from p1_core import (  # noqa: E402
    downstream_jvp_at,
    history_state_key,
    required_reference_anchors,
    select_fd_radius,
    validate_split_isolation,
)


FACTORIAL_REGISTRY: Dict[str, Tuple[str, str, str]] = {
    "p1_reference_state_fisher": (
        "reference_linear",
        "reference",
        "reference",
    ),
    "gradient_updated_only": (
        "state_local",
        "reference",
        "reference",
    ),
    "jacobian_updated_only": (
        "reference_linear",
        "state_local",
        "reference",
    ),
    "fisher_updated_only": (
        "reference_linear",
        "reference",
        "state_local",
    ),
    "gradient_jacobian_updated": (
        "state_local",
        "state_local",
        "reference",
    ),
    "gradient_fisher_updated": (
        "state_local",
        "reference",
        "state_local",
    ),
    "jacobian_fisher_updated": (
        "reference_linear",
        "state_local",
        "state_local",
    ),
    "full_state_local": (
        "state_local",
        "state_local",
        "state_local",
    ),
}


def fisher_vector_product(probability: Any, direction: Any) -> np.ndarray:
    """Apply the categorical Fisher matrix without constructing it."""
    p = np.asarray(probability, dtype=np.float64).reshape(-1)
    v = np.asarray(direction, dtype=np.float64).reshape(-1)
    return p * (v - float(np.dot(p, v)))


def fisher_inner(
    probability: Any, left: Any, right: Any
) -> float:
    left_value = np.asarray(left, dtype=np.float64).reshape(-1)
    return float(
        np.dot(
            left_value,
            fisher_vector_product(probability, right),
        )
    )


def exact_kl_gradient(
    reference_probability: Any, state_probability: Any
) -> np.ndarray:
    """Gradient in state logits of KL(p_reference || softmax(z_state))."""
    p0 = np.asarray(
        reference_probability, dtype=np.float64
    ).reshape(-1)
    ps = np.asarray(state_probability, dtype=np.float64).reshape(-1)
    return ps - p0


def geometry_score(
    gradient: Any,
    action_direction: Any,
    fisher_probability: Any,
) -> float:
    """Candidate score; exact KL is deliberately absent from the API."""
    g = np.asarray(gradient, dtype=np.float64).reshape(-1)
    c = np.asarray(action_direction, dtype=np.float64).reshape(-1)
    return float(
        np.dot(g, c)
        + 0.5 * fisher_variance(fisher_probability, c)
    )


def geometry_scores(
    *,
    reference_probability: Any,
    state_probability: Any,
    reference_linear_gradient: Any,
    state_local_gradient: Any,
    reference_action_direction: Any,
    state_local_action_direction: Any,
    nonlinear_action_direction: Any,
) -> Dict[str, float]:
    """Return the frozen baselines, factorial, and retrospective score."""
    p0 = np.asarray(reference_probability, dtype=np.float64)
    ps = np.asarray(state_probability, dtype=np.float64)
    gradients = {
        "reference_linear": np.asarray(
            reference_linear_gradient, dtype=np.float64
        ),
        "state_local": np.asarray(
            state_local_gradient, dtype=np.float64
        ),
    }
    actions = {
        "reference": np.asarray(
            reference_action_direction, dtype=np.float64
        ),
        "state_local": np.asarray(
            state_local_action_direction, dtype=np.float64
        ),
    }
    probabilities = {"reference": p0, "state_local": ps}
    output = {
        "reference_action_fisher": 0.5
        * fisher_variance(p0, actions["reference"])
    }
    for name, (gradient, action, fisher) in FACTORIAL_REGISTRY.items():
        output[name] = geometry_score(
            gradients[gradient],
            actions[action],
            probabilities[fisher],
        )
    output["nonlinear_direction_quadratic"] = geometry_score(
        gradients["state_local"],
        nonlinear_action_direction,
        ps,
    )
    return {name: float(value) for name, value in output.items()}


def score_registry_rows() -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {
        "reference_action_fisher": {
            "gradient": "zero",
            "jacobian": "reference",
            "fisher": "reference",
            "factorial": False,
            "retrospective": False,
        }
    }
    for name, (gradient, jacobian, fisher) in FACTORIAL_REGISTRY.items():
        rows[name] = {
            "gradient": gradient,
            "jacobian": jacobian,
            "fisher": fisher,
            "factorial": True,
            "retrospective": False,
        }
    rows["nonlinear_direction_quadratic"] = {
        "gradient": "state_local",
        "jacobian": "nonlinear_exact",
        "fisher": "state_local",
        "factorial": False,
        "retrospective": True,
    }
    return rows


def probability_entropy(
    probability: Any, floor: float = 1.0e-12
) -> float:
    p = np.asarray(probability, dtype=np.float64).reshape(-1)
    safe = np.clip(p, float(floor), 1.0)
    return float(-np.dot(p, np.log(safe)))


def probability_drift(
    reference_probability: Any, state_probability: Any
) -> Dict[str, float]:
    p0 = np.asarray(
        reference_probability, dtype=np.float64
    ).reshape(-1)
    ps = np.asarray(state_probability, dtype=np.float64).reshape(-1)
    return {
        "probability_total_variation": float(
            0.5 * np.abs(ps - p0).sum()
        ),
        "reference_entropy": probability_entropy(p0),
        "state_entropy": probability_entropy(ps),
        "entropy_delta": probability_entropy(ps)
        - probability_entropy(p0),
        "probability_l2": float(np.linalg.norm(ps - p0)),
    }


def state_local_symmetric_fd(
    downstream_map: Any,
    state_delta: Any,
    action_direction: Any,
    epsilon_relative: float,
    norm_floor: float = 1.0e-12,
) -> Dict[str, Any]:
    """Central FD around x0+delta, scaled by that workpoint norm."""
    delta = np.asarray(state_delta, dtype=np.float64).reshape(-1)
    action = np.asarray(
        action_direction, dtype=np.float64
    ).reshape(-1)
    base_input = np.asarray(
        downstream_map.base_input, dtype=np.float64
    ).reshape(-1)
    state_workpoint_norm = float(np.linalg.norm(base_input + delta))
    direction_norm = float(np.linalg.norm(action))
    epsilon_absolute = (
        float(epsilon_relative)
        * state_workpoint_norm
        / max(direction_norm, float(norm_floor))
    )
    plus = downstream_map.evaluate(delta + epsilon_absolute * action)
    minus = downstream_map.evaluate(delta - epsilon_absolute * action)
    derivative = (plus - minus) / (2.0 * epsilon_absolute)
    return {
        "epsilon_relative": float(epsilon_relative),
        "epsilon_absolute": float(epsilon_absolute),
        "state_workpoint_norm": state_workpoint_norm,
        "direction_norm": direction_norm,
        "plus": plus,
        "minus": minus,
        "derivative": derivative,
        "fd_norm": float(np.linalg.norm(derivative)),
    }


def factorial_effects(
    values: Mapping[Tuple[int, int, int], float]
) -> Dict[str, float]:
    """Balanced 2^3 contrasts; descriptive, not causal."""
    expected = {
        (gradient, jacobian, fisher)
        for gradient in (0, 1)
        for jacobian in (0, 1)
        for fisher in (0, 1)
    }
    if set(values) != expected:
        raise ValueError("factorial effect requires all eight cells")

    def contrast(mask: Tuple[int, ...]) -> float:
        total = 0.0
        for key, value in values.items():
            sign = 1.0
            for index in mask:
                sign *= 1.0 if key[index] else -1.0
            total += sign * float(value)
        return total / 4.0

    return {
        "gradient_main": contrast((0,)),
        "jacobian_main": contrast((1,)),
        "fisher_main": contrast((2,)),
        "gradient_jacobian_interaction": contrast((0, 1)),
        "gradient_fisher_interaction": contrast((0, 2)),
        "jacobian_fisher_interaction": contrast((1, 2)),
        "three_way_interaction": contrast((0, 1, 2)),
    }


__all__ = [
    "FACTORIAL_REGISTRY",
    "atomic_frame",
    "atomic_json",
    "downstream_jvp_at",
    "exact_kl",
    "exact_kl_gradient",
    "factorial_effects",
    "fisher_inner",
    "fisher_variance",
    "fisher_vector_product",
    "geometry_score",
    "geometry_scores",
    "history_state_key",
    "prefixed_metrics",
    "probability_drift",
    "ranking_metrics",
    "required_reference_anchors",
    "score_registry_rows",
    "select_fd_radius",
    "sha256_array",
    "sha256_file",
    "stable_softmax",
    "state_local_symmetric_fd",
    "validate_split_isolation",
    "vector_metrics",
]
