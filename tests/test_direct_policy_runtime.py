import numpy as np

from statekv.direct_policy_runtime import (
    RollingDirectPolicy,
    RollingTemporalVolatilityPolicy,
    batch_blend_score,
    batch_temporal_volatility_score,
    contribution_token_score,
    protected_rescue_score,
    stable_topk_rows,
)
from statekv.training_free_routes import scenario_token_scores


def _normalized_attention(rng: np.random.Generator, heads: int, tokens: int) -> np.ndarray:
    attention = rng.random((heads, tokens))
    return attention / attention.sum(axis=1, keepdims=True)


def test_grouped_contribution_matches_reference_kernel() -> None:
    rng = np.random.default_rng(3)
    attention = _normalized_attention(rng, 8, 11)
    values = rng.normal(size=(2, 11, 7))
    expected = scenario_token_scores(
        attention[None], values, "mean", contribution_weighted=True
    )
    observed = contribution_token_score(attention, values, dtype=np.float64)
    np.testing.assert_allclose(observed, expected, rtol=1.0e-12, atol=1.0e-12)


def test_rolling_score_matches_batch_and_selection() -> None:
    rng = np.random.default_rng(7)
    layers = (0, 2)
    banks = {
        layer: np.stack(
            [_normalized_attention(rng, 8, 13) for _ in range(4)], axis=0
        )
        for layer in layers
    }
    values = {layer: rng.normal(size=(2, 13, 5)) for layer in layers}
    eligible = np.arange(2, 10, dtype=np.int64)
    rolling = RollingDirectPolicy(
        layers=layers, window=4, contribution_weight=0.25, dtype=np.float64
    )
    for query in range(4):
        for layer in layers:
            rolling.update_layer(layer, banks[layer][query], values[layer])
    expected = batch_blend_score(
        banks,
        values,
        eligible,
        contribution_weight=0.25,
        dtype=np.float64,
    )
    np.testing.assert_allclose(rolling.score(eligible), expected, atol=1.0e-12)
    np.testing.assert_array_equal(
        rolling.select(eligible, 4), stable_topk_rows(expected, eligible, 4)
    )


def test_rolling_state_pads_old_scores_as_context_grows() -> None:
    rng = np.random.default_rng(11)
    rolling = RollingDirectPolicy(layers=(0,), window=2)
    attention_5 = _normalized_attention(rng, 4, 5)
    values_5 = rng.normal(size=(2, 5, 3))
    rolling.update_layer(0, attention_5, values_5)
    attention_6 = _normalized_attention(rng, 4, 6)
    values_6 = rng.normal(size=(2, 6, 3))
    rolling.update_layer(0, attention_6, values_6)
    score = rolling.score(np.arange(6, dtype=np.int64))
    assert score.shape == (6,)
    assert np.isclose(score.sum(), 1.0)
    assert rolling.working_set_bytes == 3 * 6 * np.dtype(np.float32).itemsize


def test_rolling_temporal_volatility_matches_batch() -> None:
    rng = np.random.default_rng(13)
    layers = (0, 2)
    banks = {
        layer: np.stack(
            [_normalized_attention(rng, 8, 13) for _ in range(4)], axis=0
        )
        for layer in layers
    }
    eligible = np.arange(2, 10, dtype=np.int64)
    rolling = RollingTemporalVolatilityPolicy(
        layers=layers, window=4, dtype=np.float64
    )
    for query in range(4):
        for layer in layers:
            rolling.update_layer(layer, banks[layer][query])
    expected = batch_temporal_volatility_score(
        banks, eligible, dtype=np.float64
    )
    np.testing.assert_allclose(rolling.score(eligible), expected, atol=1.0e-12)
    np.testing.assert_array_equal(
        rolling.select(eligible, 4), stable_topk_rows(expected, eligible, 4)
    )
    assert rolling.working_set_bytes == 2 * 4 * 13 * 8


def test_protected_rescue_bounds_attention_core_changes() -> None:
    attention = np.asarray([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    contribution = np.asarray([0.1, 0.2, 0.3, 0.4, 0.95, 1.0])
    eligible = np.arange(6, dtype=np.int64)
    priority = protected_rescue_score(
        attention, contribution, eligible, core_budget=4, rescue_slots=1
    )
    selected = set(stable_topk_rows(priority, eligible, 4).tolist())
    assert {0, 1, 2}.issubset(selected)
    assert selected == {0, 1, 2, 5}
