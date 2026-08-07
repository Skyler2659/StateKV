from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "experiments/predictive_closure/scripts"
sys.path.insert(0, str(SCRIPTS))

from predictive_core import (
    adaptive_path_fisher,
    exact_kl,
    fisher_variance,
    stable_softmax,
)


def test_fixed_qkv_deletion_identity() -> None:
    rng = np.random.default_rng(20260726)
    for _ in range(64):
        logits = rng.normal(size=31)
        alpha = stable_softmax(logits)
        values = rng.normal(size=(31, 7))
        deleted = np.sort(rng.choice(31, size=13, replace=False))
        keep = np.asarray(
            [index for index in range(31) if index not in set(deleted)]
        )
        full = alpha @ values
        masked = alpha[keep] @ values[keep] / alpha[keep].sum()
        closed = (
            alpha[deleted, None] * (full[None, :] - values[deleted])
        ).sum(axis=0) / (1.0 - alpha[deleted].sum())
        np.testing.assert_allclose(masked - full, closed, rtol=1e-12, atol=1e-12)


def test_fisher_variance_matches_explicit_matrix() -> None:
    rng = np.random.default_rng(12)
    probability = stable_softmax(rng.normal(size=19))
    direction = rng.normal(size=19)
    fisher = np.diag(probability) - np.outer(probability, probability)
    expected = float(direction @ fisher @ direction)
    np.testing.assert_allclose(
        fisher_variance(probability, direction),
        expected,
        rtol=1e-12,
        atol=1e-12,
    )


def test_fisher_is_common_shift_invariant() -> None:
    rng = np.random.default_rng(17)
    probability = stable_softmax(rng.normal(size=23))
    direction = rng.normal(size=23)
    left = fisher_variance(probability, direction)
    right = fisher_variance(probability, direction + 193.0)
    np.testing.assert_allclose(left, right, rtol=1e-11, atol=1e-11)


def test_adaptive_path_identity() -> None:
    rng = np.random.default_rng(31)
    full = rng.normal(size=127)
    delta = 2.5 * rng.normal(size=127)
    result = adaptive_path_fisher(full, delta)
    assert result["absolute_error"] <= 1e-9
    np.testing.assert_allclose(
        result["exact_kl"], exact_kl(full, full + delta), rtol=0, atol=1e-12
    )


def test_state_action_cauchy_schwarz() -> None:
    rng = np.random.default_rng(47)
    matrix = rng.normal(size=(17, 8))
    q = matrix @ matrix.T
    state = rng.normal(size=17)
    action = rng.normal(size=17)
    r_state = 0.5 * float(state @ q @ state)
    r_action = 0.5 * float(action @ q @ action)
    cross = float(state @ q @ action)
    assert abs(cross) <= 2.0 * np.sqrt(r_state * r_action) + 1e-10


def test_preregistered_sequence_splits_are_disjoint() -> None:
    train = set(range(24, 44))
    calibration = set(range(44, 56))
    test = set(range(56, 72))
    prior = set(range(24))
    assert train.isdisjoint(calibration)
    assert train.isdisjoint(test)
    assert calibration.isdisjoint(test)
    assert (train | calibration | test).isdisjoint(prior)

