from __future__ import annotations

import numpy as np
import pandas as pd

from statekv.metric_repair_analysis import (
    trajectory_risks,
    unsupervised_sensitivity_scales,
)


def test_unsupervised_scales_are_normalized_and_block_constant() -> None:
    diagonal, block = unsupervised_sensitivity_scales(
        np.asarray([1.0, 1.0, 4.0, 4.0]), 1, blocks=2
    )
    assert np.isclose(np.mean(np.square(diagonal)), 1.0)
    assert np.isclose(np.mean(np.square(block)), 1.0)
    assert block[0] > block[1]


def test_innovation_does_not_reaccumulate_a_constant_action() -> None:
    metadata = pd.DataFrame(
        {
            "anchor": [0, 0, 0],
            "candidate_id": ["a", "a", "a"],
            "horizon_offset": [1, 2, 3],
        }
    )
    signatures = np.asarray([[2.0], [2.0], [2.0]])
    scores = trajectory_risks(metadata, signatures, decay=1.0)
    assert np.allclose(scores["action"], [2.0, 2.0, 2.0])
    assert np.allclose(scores["repeated"], [2.0, 6.0, 10.0])
    assert np.allclose(scores["innovation"], [2.0, 6.0, 6.0])
    assert np.allclose(scores["ema"], scores["action"])
