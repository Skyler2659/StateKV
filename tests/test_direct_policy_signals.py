import numpy as np

from statekv.direct_policy_signals import (
    adjacent_value_change_score,
    attention_head_peak_score,
    attention_temporal_volatility_score,
    diagonal_leverage_score,
    uniform_position_coverage_score,
)


def test_access_signals_preserve_rare_heads_and_temporal_changes() -> None:
    bank = np.asarray(
        [
            [[0.8, 0.1, 0.1], [0.1, 0.1, 0.8]],
            [[0.2, 0.1, 0.7], [0.1, 0.1, 0.8]],
        ]
    )
    peak = attention_head_peak_score(bank)
    volatility = attention_temporal_volatility_score(bank)
    assert np.allclose(peak, [0.8, 0.1, 0.8])
    assert volatility[0] > volatility[1]
    assert volatility[2] > volatility[1]


def test_diagonal_leverage_highlights_feature_outlier() -> None:
    features = np.asarray([[[0.0], [0.0], [1.0], [8.0]]])
    score = diagonal_leverage_score(features, np.asarray([0, 1, 2, 3]))
    assert int(np.argmax(score)) == 3


def test_value_change_and_uniform_coverage_are_deterministic() -> None:
    values = np.asarray([[[0.0], [0.0], [3.0], [3.0], [7.0]]])
    change = adjacent_value_change_score(values)
    assert np.allclose(change, [0.0, 0.0, 9.0, 0.0, 16.0])
    coverage = uniform_position_coverage_score(
        10, np.arange(2, 10, dtype=np.int64), 4
    )
    assert np.flatnonzero(coverage).tolist() == [3, 5, 7, 9]
