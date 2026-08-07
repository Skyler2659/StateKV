from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch

from statekv.functional_probe import ProbeStep
from statekv.theory_closing import (
    NoBackingMonitoringState,
    assert_replay_alignment,
    cumulative_rows,
    enumerate_fixed_subsets,
    exhaustive_ridge_subset_risk,
    fixed_query_feature_matrices,
    fixed_qkv_subset_metrics,
    leave_one_sequence_out,
    paired_gap,
    validate_anchor_predictor_columns,
    validate_recent_budget,
)


def test_subset_exhaustive_search_matches_brute_force() -> None:
    generator = torch.Generator().manual_seed(7)
    base = torch.randn(3, 7, generator=generator)
    pool = torch.randn(6, 7, generator=generator)
    combinations = enumerate_fixed_subsets(6, 2)
    actual, diagnostics = exhaustive_ridge_subset_risk(
        base, pool, combinations, 1e-3
    )
    expected = []
    ridge = float(diagnostics["ridge"])
    for combination in combinations:
        history = torch.cat([base, pool[combination]], dim=0).double()
        regularized = (
            history @ history.T
            + ridge * torch.eye(len(history), dtype=torch.float64)
        )
        projection = (
            torch.linalg.solve(
                regularized, history @ pool.double().T
            ).T
            @ history
        )
        expected.append(
            float(((pool.double() - projection) ** 2).sum().item())
        )
    assert np.allclose(actual, expected, atol=1e-10, rtol=1e-10)
    assert int(np.argmin(actual)) == int(np.argmin(expected))


def test_fixed_qkv_deletion_identity() -> None:
    generator = torch.Generator().manual_seed(11)
    logits = torch.randn(8, generator=generator)
    attention = torch.softmax(logits, dim=0)
    values = torch.randn(8, 4, generator=generator)
    basis = torch.randn(4, 9, generator=generator)
    combinations = enumerate_fixed_subsets(5, 2)
    result = fixed_qkv_subset_metrics(
        attention,
        values,
        basis,
        base_rows=[0, 1, 2],
        pool_rows=[3, 4, 5, 6, 7],
        combinations=combinations,
    )
    assert float(np.nanmax(result["identity_relative_error"])) < 1e-10
    assert np.allclose(
        result["true_head_risk"],
        result["identity_head_risk"],
        atol=1e-10,
        rtol=1e-10,
    )


def test_feature_shapes_and_gqa_mapping() -> None:
    generator = torch.Generator().manual_seed(13)
    attention = torch.softmax(
        torch.randn(12, 9, generator=generator), dim=1
    )
    values = torch.randn(2, 9, 4, generator=generator)
    bases = {
        head: torch.randn(4, 7, generator=generator)
        for head in range(12)
    }
    features = fixed_query_feature_matrices(attention, values, bases, 6)
    assert len(features) == 12
    for head in range(12):
        assert features[head]["raw_v"].shape == (9, 4)
        assert features[head]["projected_v"].shape == (9, 7)
        assert features[head]["aov"].shape == (9, 7)
        assert features[head]["aor"].shape == (9, 7)
        expected_kv = 0 if head < 6 else 1
        assert torch.equal(features[head]["raw_v"], values[expected_kv])


def test_old_equals_fresh_all_gaps_zero() -> None:
    values = np.asarray([0.0, 1.5, -2.0, 7.0])
    assert np.array_equal(paired_gap(values, values.copy()), np.zeros(4))


def test_cumulative_h1_equals_single_step() -> None:
    steps = pd.DataFrame(
        {
            "sample_id": ["a", "a", "a"],
            "arm": ["fresh", "fresh", "fresh"],
            "horizon_offset": [1, 2, 3],
            "benefit": [0.25, -0.1, 0.5],
        }
    )
    cumulative = cumulative_rows(
        steps, [1, 3], ["benefit"], ["sample_id", "arm"]
    )
    assert cumulative.loc[cumulative["horizon"] == 1, "benefit"].item() == (
        steps.loc[steps["horizon_offset"] == 1, "benefit"].item()
    )


def _probe(index: int, token: int, position: int) -> ProbeStep:
    return ProbeStep(
        logits=torch.zeros(3),
        diagnostic=None,  # alignment test does not inspect diagnostics
        position_maps={0: torch.arange(2)},
        target_index=index,
        target_token_id=token,
        target_token_position=position,
        active_cache_tokens=2,
        forward_time_s=0.0,
    )


def test_direct_stateful_token_alignment() -> None:
    assert_replay_alignment(_probe(4, 9, 20), _probe(4, 9, 20))
    with pytest.raises(RuntimeError):
        assert_replay_alignment(_probe(4, 9, 20), _probe(5, 9, 20))


def test_recent_rolling_window_budget() -> None:
    validate_recent_budget({0: list(range(8)), 1: list(range(7))}, 8)
    with pytest.raises(RuntimeError):
        validate_recent_budget({0: list(range(9))}, 8)


def test_no_backing_monitor_does_not_store_evicted_kv() -> None:
    state = NoBackingMonitoringState(gamma=0.9)
    state.observe_arrival(10, 1.25)
    state.update_retained([10], {10: 0.75})
    state.evict([10])
    assert state.retained_scores == {}
    assert state.schema()["stores_evicted_kv"] == "false"
    assert all(
        not isinstance(value, torch.Tensor)
        for mapping in (state.arrival_scores, state.retained_scores)
        for value in mapping.values()
    )


def test_loso_split_has_no_sequence_leakage() -> None:
    frame = pd.DataFrame(
        {
            "sample_id": ["a", "a", "b", "b", "c"],
            "value": range(5),
        }
    )
    values = frame["sample_id"].to_numpy()
    folds = leave_one_sequence_out(frame)
    assert len(folds) == 3
    for train, test in folds:
        assert set(values[train]).isdisjoint(set(values[test]))


def test_future_oracle_not_allowed_in_anchor_predictor() -> None:
    validate_anchor_predictor_columns(
        ["horizon", "task", "anchor_obs_w8_mean"]
    )
    with pytest.raises(ValueError):
        validate_anchor_predictor_columns(
            ["anchor_obs_w8_mean", "future_aor_gap"]
        )
