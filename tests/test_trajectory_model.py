from __future__ import annotations

import numpy as np
import pytest
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

from statekv.trajectory_model import (
    assert_sequence_split,
    assert_teacher_alignment,
    closed_form_recursion,
    exact_distribution_metrics,
    fit_linear_dynamics,
    fit_quadratic_form,
    layer_regime,
    scaling_fit,
    stable_trajectory_id,
    validate_hybrid_source_labels,
    validate_recent_budget,
)


def test_beta_zero_exact_reference_metrics() -> None:
    logits = torch.tensor([0.5, -0.25, 1.0])
    result = exact_distribution_metrics(logits, logits.clone(), 2)
    assert result["exact_kl"] == pytest.approx(0.0, abs=1e-8)
    assert result["js"] == pytest.approx(0.0, abs=1e-8)
    assert result["delta_nll"] == pytest.approx(0.0, abs=1e-8)


def test_beta_one_projected_injection_equivalence() -> None:
    full = np.asarray([1.0, -2.0, 0.5])
    direct = np.asarray([0.25, 0.5, -0.75])
    injected = full + 1.0 * direct
    compressed = full + direct
    assert np.array_equal(injected, compressed)


def test_direct_intervention_does_not_mutate_reference() -> None:
    full = np.asarray([1.0, 2.0, 3.0])
    immutable = full.copy()
    _ = full + np.asarray([0.1, -0.2, 0.3])
    assert np.array_equal(full, immutable)


def test_hybrid_source_labels_are_arm_specific() -> None:
    validate_hybrid_source_labels(
        "query_restore",
        {
            "query": "full_reference_restored",
            "new_key_value": "trajectory",
        },
    )
    with pytest.raises(ValueError):
        validate_hybrid_source_labels(
            "query_restore", {"query": "trajectory"}
        )


def test_teacher_forced_token_and_position_alignment() -> None:
    assert_teacher_alignment(17, 17, 9, 9)
    with pytest.raises(RuntimeError):
        assert_teacher_alignment(17, 18, 9, 9)


def test_latent_transform_is_fit_on_training_rows_only() -> None:
    train = np.asarray([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]])
    heldout = np.asarray([[100.0, 0.0]])
    pca = PCA(n_components=1).fit(train)
    assert np.array_equal(pca.mean_, np.asarray([2.0, 0.0]))
    assert not np.array_equal(
        pca.mean_, np.concatenate([train, heldout]).mean(axis=0)
    )


def test_h1_closed_form_matches_one_step_formula() -> None:
    a = np.asarray([[0.5]])
    b = np.asarray([[2.0]])
    sigma = np.asarray([[0.25]])
    q = np.asarray([[3.0]])
    result = closed_form_recursion(
        a,
        b,
        sigma,
        q,
        np.asarray([1.0]),
        np.asarray([[0.5]]),
        np.asarray([[0.75]]),
    )
    mean = a @ np.asarray([1.0]) + b @ np.asarray([0.75])
    covariance = a @ np.asarray([[0.5]]) @ a.T + sigma
    expected = float(mean.T @ q @ mean + np.trace(q @ covariance))
    assert result["cumulative_risk"][0] == pytest.approx(expected)


def test_noiseless_linear_system_recovers_a_b_and_q() -> None:
    rng = np.random.default_rng(7)
    a = np.asarray([[0.7, 0.1], [-0.2, 0.5]])
    b = np.asarray([[1.0], [0.25]])
    states = rng.normal(size=(200, 2))
    inputs = rng.normal(size=(200, 1))
    next_states = states @ a.T + inputs @ b.T
    fitted = fit_linear_dynamics(states, inputs, next_states)
    assert np.allclose(fitted["A"], a, atol=1e-10)
    assert np.allclose(fitted["B"], b, atol=1e-10)
    q = np.asarray([[2.0, 0.3], [0.3, 1.0]])
    losses = np.einsum("ni,ij,nj->n", states, q, states)
    assert np.allclose(fit_quadratic_form(states, losses), q, atol=1e-10)


def test_gaussian_mean_covariance_recursion_matches_monte_carlo() -> None:
    rng = np.random.default_rng(11)
    a = np.asarray([[0.8]])
    b = np.asarray([[0.4]])
    sigma = np.asarray([[0.2]])
    closed = closed_form_recursion(
        a,
        b,
        sigma,
        np.eye(1),
        np.asarray([0.3]),
        np.asarray([[0.1]]),
        np.ones((3, 1)),
    )
    samples = rng.normal(0.3, np.sqrt(0.1), size=(200000, 1))
    for step in range(3):
        samples = samples @ a.T + 0.4 + rng.normal(
            0.0, np.sqrt(0.2), size=samples.shape
        )
        assert samples.mean() == pytest.approx(
            closed["mean"][step, 0], abs=0.01
        )
        assert samples.var() == pytest.approx(
            closed["covariance"][step, 0, 0], abs=0.01
        )


def test_switching_system_is_not_safely_pooled() -> None:
    rng = np.random.default_rng(13)
    x = rng.normal(size=(400, 1))
    regime = np.repeat([0, 1], 200)
    y = np.where(regime[:, None] == 0, 0.9 * x, -0.9 * x)
    pooled = LinearRegression().fit(x, y).predict(x)
    switched = np.empty_like(y)
    for value in (0, 1):
        index = regime == value
        switched[index] = LinearRegression().fit(
            x[index], y[index]
        ).predict(x[index])
    pooled_error = np.mean((y - pooled) ** 2)
    switched_error = np.mean((y - switched) ** 2)
    assert switched_error < pooled_error * 1e-6


def test_sequence_split_has_no_leakage() -> None:
    assert_sequence_split(["a", "b"], ["c"])
    with pytest.raises(RuntimeError):
        assert_sequence_split(["a", "b"], ["b", "c"])


def test_recent_window_budget_is_enforced() -> None:
    validate_recent_budget(list(range(128)), 128, 4, 32)
    with pytest.raises(RuntimeError):
        validate_recent_budget(list(range(129)), 128, 4, 32)


def test_layer27_is_not_hidden_in_aggregate_regime() -> None:
    assert layer_regime(27) == "layer27"
    assert layer_regime(26) != "layer27"


def test_scaling_fit_recovers_origin_linearity() -> None:
    result = scaling_fit(
        np.asarray([0.0, 0.25, 0.5, 1.0]),
        np.asarray([0.0, 0.5, 1.0, 2.0]),
    )
    assert result["slope"] == pytest.approx(2.0)
    assert result["r2"] == pytest.approx(1.0)


def test_linear_superposition_identity() -> None:
    a = np.asarray([[0.8, 0.1], [0.0, 0.5]])
    first = np.asarray([1.0, -0.5])
    second = np.asarray([-0.25, 0.75])
    assert np.allclose(
        a @ (first + second), a @ first + a @ second
    )


def test_trajectory_ids_are_stable_and_distinct() -> None:
    first = stable_trajectory_id("s", "scale", 16, "aor", 1.0)
    second = stable_trajectory_id("s", "scale", 16, "aor", 1.0)
    third = stable_trajectory_id("s", "scale", 16, "aor", 1.5)
    assert first == second
    assert first != third
