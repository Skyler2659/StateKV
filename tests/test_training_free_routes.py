import numpy as np

from statekv.training_free_routes import (
    attention_output,
    deletion_output,
    entropy_cotangent,
    margin_cotangents,
    merge_output_with_assignments,
    nearest_value_assignments,
    nearest_value_merge,
    scenario_token_scores,
    select_top_with_mandatory,
    softmax,
    symmetric_quantize,
    vjp_action_scores,
)


def test_vjp_action_scores_match_explicit_quadratic() -> None:
    rng = np.random.default_rng(3)
    actions = rng.normal(size=(5, 4))
    gradients = rng.normal(size=(4, 7))
    expected = 0.5 * np.einsum(
        "ni,ij,nj->n",
        actions,
        gradients @ gradients.T / gradients.shape[1],
        actions,
    )
    assert np.allclose(vjp_action_scores(actions, gradients), expected)


def test_margin_and_entropy_cotangents_are_shift_invariant() -> None:
    logits = np.asarray([1.0, 3.0, 2.0, -1.0])
    margins, competitors = margin_cotangents(logits, 2)
    assert competitors.tolist() == [2, 0]
    assert np.allclose(margins.sum(axis=1), 0.0)
    assert np.allclose(entropy_cotangent(logits), entropy_cotangent(logits + 11.0))
    assert abs(float(entropy_cotangent(logits).sum())) < 1.0e-12


def test_nearest_merge_preserves_mass_and_bound_holds() -> None:
    attention = np.asarray([[0.1, 0.2, 0.3, 0.4]])
    values = np.asarray([[[0.0], [1.0], [2.0], [3.0]]])
    full = attention_output(attention, values)
    deleted = deletion_output(attention, values, [0, 3])
    merged = nearest_value_merge(attention, values, [0, 3])
    assignments = nearest_value_assignments(values, [0, 3])
    repeated = merge_output_with_assignments(attention, values, assignments)
    error = np.linalg.norm(merged["output"] - full, axis=1)
    assert np.allclose(repeated["output"], merged["output"])
    assert np.all(error <= merged["bound"] + 1.0e-12)
    assert np.linalg.norm(merged["output"] - full) <= np.linalg.norm(deleted - full)


def test_scenario_selector_honors_mandatory_and_quantization_monotonic_example() -> None:
    attentions = np.asarray(
        [
            [[0.50, 0.10, 0.30, 0.10]],
            [[0.10, 0.50, 0.30, 0.10]],
        ]
    )
    values = np.asarray([[[0.0], [2.0], [1.0], [4.0]]])
    scores = scenario_token_scores(attentions, values, "max")
    ema_scores = scenario_token_scores(attentions, values, "ema")
    q75_scores = scenario_token_scores(attentions, values, "q75")
    selected = select_top_with_mandatory(scores, 3, [3])
    assert 3 in selected
    assert len(selected) == 3
    assert ema_scores.shape == scores.shape
    assert q75_scores.shape == scores.shape
    vector = np.asarray([[[-1.0, -0.43, 0.19, 0.77]]])
    err2 = np.linalg.norm(symmetric_quantize(vector, 2) - vector)
    err4 = np.linalg.norm(symmetric_quantize(vector, 4) - vector)
    assert err4 < err2
    assert np.allclose(softmax(np.asarray([1.0, 2.0])).sum(), 1.0)
