from __future__ import annotations

import numpy as np

from statekv.shared_jvp import (
    categorical_fisher_gram,
    randomized_action_basis,
    randomized_fisher_gram,
    state_action_scores,
)


def test_shared_basis_and_gram_recover_linear_pullback_scores() -> None:
    actions = np.asarray([[1.0, 0.0], [0.0, 2.0], [1.0, 2.0]])
    jacobian = np.asarray([[1.0, 2.0], [0.5, -1.0], [-1.0, 0.0]])
    probability = np.asarray([0.2, 0.3, 0.5])
    basis = randomized_action_basis(actions, 2, seed=7)
    output_basis = jacobian @ basis
    gram = categorical_fisher_gram(probability, output_basis)
    scores = state_action_scores(actions, np.zeros_like(actions), basis, gram)
    fisher = np.diag(probability) - np.outer(probability, probability)
    expected = np.asarray(
        [0.5 * (jacobian @ row) @ fisher @ (jacobian @ row) for row in actions]
    )
    assert np.allclose(scores, expected)


def test_randomized_fisher_gram_converges_with_large_sketch() -> None:
    rng = np.random.default_rng(4)
    probability = rng.uniform(size=17)
    probability /= probability.sum()
    directions = rng.normal(size=(17, 3))
    exact = categorical_fisher_gram(probability, directions)
    approximate = randomized_fisher_gram(
        probability, directions, 20000, seed=9
    )
    assert np.allclose(approximate, exact, atol=0.02, rtol=0.08)
