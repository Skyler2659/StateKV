from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "experiments/local_truncated_jacobian/scripts"
sys.path.insert(0, str(SCRIPTS))

from local_core import (
    choose_radius,
    cosine,
    load_candidates,
    relative_l2,
    symmetric_norm_ratio,
)


def test_vector_metrics() -> None:
    left = np.array([1.0, 2.0, 3.0])
    assert abs(cosine(left, left) - 1.0) < 1.0e-12
    assert relative_l2(left, left) == 0.0
    assert abs(symmetric_norm_ratio(left, left) - 1.0) < 1.0e-12


def test_frozen_radius_rule_selects_first_plateau() -> None:
    rows = []
    for radius, cosine_value, relative in (
        (1.0e-6, 0.98, 0.20),
        (3.0e-6, 0.9960, 0.04),
        (1.0e-5, 0.9965, 0.03),
        (3.0e-5, 0.9960, 0.04),
    ):
        for index in range(10):
            rows.append(
                {
                    "radius": radius,
                    "fd_norm": 1.0,
                    "noise_norm": 0.0,
                    "finite": True,
                    "jvp_fd_cosine": cosine_value,
                    "jvp_fd_relative_l2": relative,
                }
            )
    result = choose_radius(pd.DataFrame(rows))
    assert result["calibration_passed"]
    assert result["selected_radius"] == 3.0e-6


def test_candidate_registry_reuse_is_exact() -> None:
    registry = (
        ROOT
        / "experiments/predictive_closure/raw/p0_alignment"
        / "formal_4bit_retry1/candidate_registry_rows.parquet"
    )
    candidates = load_candidates(
        registry,
        "synthetic_niah_24",
        16,
        1281,
        "f2d06b2732a2a0bf8baac6694ef35aa2ed4393a19e75400564a545786d787307",
    )
    assert len(candidates) == 8
    assert len({candidate.mask_hash for candidate in candidates}) == 8
    assert {candidate.source for candidate in candidates} == {
        "attention_only",
        "aov",
        "aor",
        "v_ridge",
        "snapkv",
        "old_stale_core",
        "fresh_core",
        "random_reference",
    }
    assert all(len(candidate.retained_positions) == 128 for candidate in candidates)
    assert all(len(candidate.core_positions) == 92 for candidate in candidates)

