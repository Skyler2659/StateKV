import numpy as np
import pandas as pd

from experiments.local_truncated_jacobian.scripts.run_l3_formal import (
    group_row,
)


def test_l3_group_perfect_rankings() -> None:
    energy = np.arange(1, 9, dtype=np.float64)
    frame = pd.DataFrame(
        {
            "sample_id": ["sample"] * 8,
            "task": ["task"] * 8,
            "split": ["train"] * 8,
            "anchor": [16] * 8,
            "layer": [0] * 8,
            "mask_hash": [f"mask-{index}" for index in range(8)],
            "sj1_predicted_energy": energy,
            "strue1_physical_energy": energy * 2,
            "s0_direct_energy": energy * 3,
            "exact_kl_full_to_physical": energy * 4,
        }
    )
    result = group_row(frame)
    assert result["rho_sj1_strue1"] == 1.0
    assert result["rho_strue1_exact_kl"] == 1.0
    assert result["rho_sj1_exact_kl"] == 1.0
    assert result["rho_s0_exact_kl"] == 1.0
    assert result["delta_rho_j1_minus_s0_to_kl"] == 0.0
