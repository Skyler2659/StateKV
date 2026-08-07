import numpy as np
import pandas as pd

from experiments.local_truncated_jacobian.scripts.run_l1_formal import (
    _normalized_regret,
    _pairwise_sign_accuracy,
    _spearman,
    ranking_row,
)


def test_perfect_rank_metrics() -> None:
    predicted = np.arange(8, dtype=np.float64)
    truth = 3.0 * predicted + 2.0
    assert _spearman(predicted, truth) == 1.0
    assert _pairwise_sign_accuracy(predicted, truth) == 1.0
    assert _normalized_regret(predicted, truth) == 0.0


def test_constant_ranking_is_undefined() -> None:
    assert np.isnan(_spearman(np.ones(8), np.arange(8)))


def test_ranking_row_is_eight_candidate_group() -> None:
    frame = pd.DataFrame(
        {
            "sample_id": ["sample"] * 8,
            "mask_hash": [f"mask-{index}" for index in range(8)],
            "predicted_energy": np.arange(8, dtype=np.float64),
            "true_energy": np.arange(8, dtype=np.float64),
        }
    )
    result = ranking_row(frame, ("sample_id",))
    assert result["energy_spearman"] == 1.0
    assert result["top1_recall"] == 1.0
    assert result["top3_recall"] == 1.0
    assert result["mean_normalized_regret"] == 0.0
