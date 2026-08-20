import numpy as np

from statekv.adaptive_temporal import (
    AdaptiveTemporalConfig,
    adaptive_temporal_scores,
    additional_state_bytes,
    cumulative_attention,
    fixed_ema,
    future_attention_utility,
    percentile_rank_observations,
    rank_jump_dual_scores,
)


def test_fixed_ema_is_causal_and_initializes_new_tokens_on_first_sight():
    observations = np.asarray(
        [[1.0, np.nan], [0.0, 0.4], [0.0, 0.8]], dtype=np.float64
    )
    result = fixed_ema(observations, 0.5)
    np.testing.assert_allclose(result[:, 0], [1.0, 0.5, 0.25])
    assert np.isnan(result[0, 1])
    np.testing.assert_allclose(result[1:, 1], [0.4, 0.6])


def test_cumulative_attention_matches_sum_of_available_history():
    observations = np.asarray([[0.2, np.nan], [0.3, 0.4], [0.1, 0.2]])
    result = cumulative_attention(observations)
    np.testing.assert_allclose(result[:, 0], [0.2, 0.5, 0.6])
    np.testing.assert_allclose(result[1:, 1], [0.4, 0.6])


def test_adaptive_gate_shortens_memory_after_regime_change():
    observations = np.zeros((12, 1), dtype=np.float64)
    observations[:6] = 0.1
    observations[6:] = 0.9
    config = AdaptiveTemporalConfig(
        fast_rho=0.0,
        slow_rho=0.95,
        variance_rho=0.9,
        rho_short=0.0,
        rho_long=0.99,
        threshold=1.0,
    )
    states = adaptive_temporal_scores(observations, config)
    assert states["rho_discrete"][6, 0] == 0.0
    assert states["adaptive_discrete"][6, 0] > fixed_ema(observations, 0.99)[6, 0]


def test_future_utility_excludes_current_step_and_is_noncausal_label_only():
    observations = np.asarray([[1.0], [2.0], [4.0], [8.0]])
    utility = future_attention_utility(observations, 2)
    np.testing.assert_allclose(utility[:3, 0], [6.0, 12.0, 8.0])
    assert np.isnan(utility[3, 0])


def test_additional_state_bytes():
    assert additional_state_bytes(
        scalars_per_token=4, dtype_bytes=2, layers=36, tokens_per_layer=256
    ) == 73_728


def test_rank_jump_gate_uses_only_present_and_past():
    observations = np.asarray(
        [[0.1, 0.2, np.nan], [0.3, 0.1, 0.2], [0.2, 0.4, 0.1]],
        dtype=np.float64,
    )
    ranks = percentile_rank_observations(observations)
    assert np.isnan(ranks[0, 2])
    first = rank_jump_dual_scores(
        observations,
        fast_rho=0.5,
        slow_rho=0.9,
        rank_memory_rho=0.8,
        jump_threshold=0.1,
        gate_alpha=10.0,
    )
    changed_future = observations.copy()
    changed_future[2] = [0.9, 0.01, 0.02]
    second = rank_jump_dual_scores(
        changed_future,
        fast_rho=0.5,
        slow_rho=0.9,
        rank_memory_rho=0.8,
        jump_threshold=0.1,
        gate_alpha=10.0,
    )
    np.testing.assert_allclose(first["score"][:2], second["score"][:2], equal_nan=True)
