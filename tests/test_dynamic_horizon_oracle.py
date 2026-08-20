import numpy as np

from scripts.analyze_dynamic_horizon_oracle import (
    _fixed_metrics,
    _token_time_oracle_metrics,
)


def test_token_time_oracle_never_worse_on_constructed_rank_switch() -> None:
    target = np.asarray([[3.0, 2.0, 1.0], [1.0, 2.0, 3.0]])
    scores = {
        "ema_rho_0": np.asarray([[3.0, 2.0, 1.0], [3.0, 2.0, 1.0]]),
        "ema_rho_0.9": np.asarray([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]),
    }
    eligible = np.ones_like(target, dtype=bool)
    oracle, choices = _token_time_oracle_metrics(scores, target, eligible, 1)
    fixed = _fixed_metrics(scores["ema_rho_0"], target, eligible, 1)
    assert oracle["future_topk_recall"] >= fixed["future_topk_recall"]
    assert sum(choices.values()) == target.size


def test_fixed_metrics_has_per_step_decisions() -> None:
    values = np.asarray([[0.1, 0.2, 0.3], [0.3, 0.2, 0.1]])
    result = _fixed_metrics(values, values, np.ones_like(values, dtype=bool), 2)
    assert result["decisions"] == 2
    assert result["future_topk_recall"] == 1.0

