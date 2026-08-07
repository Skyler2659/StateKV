from __future__ import annotations

import math
from pathlib import Path

import torch

from statekv.config import load_discovery_config
from statekv.functional_features import (
    LayerFeatureMatrices,
    RidgeCoverageFactor,
    functional_measurement,
)
from statekv.functional_probe import _condition_cache


ROOT = Path(__file__).resolve().parents[1]


def test_functional_config_captures_dense_refresh_anchors() -> None:
    cfg = load_discovery_config(
        str(
            ROOT
            / "configs"
            / "discovery"
            / "functional_probe_stage1_4bit.yaml"
        )
    )
    assert cfg.captured_anchor_steps() == [
        0,
        1,
        8,
        16,
        24,
        32,
        40,
        48,
        49,
        56,
        64,
        72,
        80,
        88,
        96,
        112,
    ]


def test_recent_zero_reserves_only_the_transient_query_slot() -> None:
    cfg = load_discovery_config(
        str(
            ROOT
            / "configs"
            / "discovery"
            / "functional_probe_stage1_4bit.yaml"
        )
    )
    zero = _condition_cache(cfg, 128, 0)
    default = _condition_cache(cfg, 128, 32)
    assert zero.recent_size == 1
    assert zero.selected_core_budget == 123
    assert zero.sink_size + zero.recent_size + zero.selected_core_budget == 128
    assert default.recent_size == 32
    assert default.selected_core_budget == 92


def test_ridge_factor_matches_direct_projector_in_primal_and_dual() -> None:
    generator = torch.Generator().manual_seed(123)
    for rows, dimension in ((9, 4), (4, 9)):
        history = torch.randn(rows, dimension, generator=generator)
        probes = torch.randn(7, dimension, generator=generator)
        factor = RidgeCoverageFactor.fit(history, 1e-3, "relative")
        gram = history.double().T @ history.double()
        direct = (
            probes.double()
            @ torch.linalg.solve(
                gram
                + factor.ridge
                * torch.eye(dimension, dtype=torch.float64),
                gram,
            )
        )
        assert torch.allclose(
            factor.project(probes), direct, rtol=1e-9, atol=1e-9
        )


def test_functional_measurement_is_zero_for_identical_old_and_fresh() -> None:
    generator = torch.Generator().manual_seed(7)
    base_matrix = torch.randn(8, 5, generator=generator)
    current_matrix = torch.cat(
        [base_matrix, torch.randn(2, 5, generator=generator)], dim=0
    )
    key = ("raw_v", "layer", None)
    base = LayerFeatureMatrices(
        layer=0,
        positions=list(range(8)),
        matrices={key: base_matrix},
        observation_window_queries=4,
        observation_weight_source="test",
    )
    current = LayerFeatureMatrices(
        layer=0,
        positions=list(range(10)),
        matrices={key: current_matrix},
        observation_window_queries=4,
        observation_weight_source="test",
    )
    result = functional_measurement(
        base_features=base,
        current_features=current,
        key=key,
        old_history_positions={0, 1, 3, 6},
        fresh_history_positions={0, 1, 3, 6},
        base_old_history_positions={0, 1, 3, 6},
        epsilon=1e-12,
        ridge_coefficient=1e-3,
        ridge_mode="relative",
    )
    assert math.isclose(result["delta_e"]["raw_sum"], 0.0, abs_tol=1e-12)
    assert math.isclose(
        result["delta_e"]["normalized_sum"], 0.0, abs_tol=1e-12
    )
    assert result["arrival_residual"]["token_count"] == 2
    assert math.isclose(
        result["deployable_approx_raw_sum"],
        result["arrival_residual"]["raw_sum"]
        + result["retained_reweighting"]["raw_sum"],
        rel_tol=1e-12,
    )


def test_fixed_qkv_deletion_identity() -> None:
    generator = torch.Generator().manual_seed(99)
    attention = torch.softmax(torch.randn(17, generator=generator), dim=0).double()
    values = torch.randn(17, 8, generator=generator).double()
    deleted = torch.tensor([1, 4, 5, 11, 16])
    output = (attention.unsqueeze(-1) * values).sum(dim=0)
    mass = attention.index_select(0, deleted).sum()
    deleted_value = (
        attention.index_select(0, deleted).unsqueeze(-1)
        * values.index_select(0, deleted)
    ).sum(dim=0)
    masked = (output - deleted_value) / (1.0 - mass)
    identity = (mass * output - deleted_value) / (1.0 - mass)
    assert torch.allclose(masked - output, identity, rtol=1e-12, atol=1e-12)
