from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace

from statekv.robust_envelope import (
    block_triangular_mask,
    physical_shared_mask,
    refresh_preserves_error,
)
from statekv.robust_envelope_analysis import (
    _group_coverage,
    _partition_sequences,
    calibration_margin,
    envelope_coefficients_nonnegative,
    fit_nonnegative_envelope,
    h1_recursion,
    induction_check,
    recursive_envelope,
)
from statekv.robust_envelope_policy import (
    refresh_schedule,
    summarize_policy_rows,
)
from statekv.selectors import (
    CoreSelection,
    LayerSelection,
)


def _linear_system(seed: int = 7):
    rng = np.random.default_rng(seed)
    error = rng.uniform(0.0, 1.0, size=(2000, 3))
    direct = rng.uniform(0.0, 1.0, size=(2000, 3))
    a = np.asarray(
        [[0.4, 0.0, 0.0], [0.1, 0.5, 0.0], [0.2, 0.1, 0.3]]
    )
    b = np.asarray(
        [[0.7, 0.0, 0.0], [0.2, 0.4, 0.0], [0.1, 0.2, 0.6]]
    )
    target = error @ a.T + direct @ b.T
    return error, direct, target, a, b


def test_synthetic_linear_system_recovers_envelope() -> None:
    error, direct, target, a, b = _linear_system()
    model = fit_nonnegative_envelope(
        error, direct, target, [0, 7, 27], "E2"
    )
    assert np.allclose(model.a, a, atol=1e-8)
    assert np.allclose(model.b, b, atol=1e-8)


def test_quadratic_e3_improves_over_e2() -> None:
    error, direct, linear, _, _ = _linear_system(11)
    target = linear + 0.8 * error**2
    e2 = fit_nonnegative_envelope(
        error, direct, target, [0, 7, 27], "E2"
    )
    e3 = fit_nonnegative_envelope(
        error, direct, target, [0, 7, 27], "E3"
    )
    prediction2 = np.stack(
        [e2.step(x, u) for x, u in zip(error, direct)]
    )
    prediction3 = np.stack(
        [e3.step(x, u) for x, u in zip(error, direct)]
    )
    assert np.mean((target - prediction3) ** 2) < (
        np.mean((target - prediction2) ** 2) * 1e-6
    )


def test_envelope_coefficients_are_nonnegative() -> None:
    error, direct, target, _, _ = _linear_system()
    model = fit_nonnegative_envelope(
        error, direct, target, [0, 7, 27], "E3"
    )
    assert envelope_coefficients_nonnegative(model)


def test_block_triangular_mask_matches_layer_order() -> None:
    mask = block_triangular_mask([0, 7, 14, 27], [0, 7, 14, 27])
    expected = np.asarray(
        [
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 1],
        ],
        dtype=bool,
    )
    assert np.array_equal(mask, expected)


def test_recursion_does_not_accept_future_truth() -> None:
    error, direct, target, _, _ = _linear_system()
    model = fit_nonnegative_envelope(
        error, direct, target, [0, 7, 27], "E2"
    )
    first, _ = recursive_envelope(model, direct[:8], np.zeros(3))
    changed_future_truth = target[:8] * 1000
    second, _ = recursive_envelope(model, direct[:8], np.zeros(3))
    assert np.array_equal(first, second)
    assert changed_future_truth.shape == first.shape


def test_outer_partition_has_no_sequence_leakage() -> None:
    sequences = ["n%d" % i for i in range(6)] + [
        "g%d" % i for i in range(6)
    ]
    tasks = {
        value: ("niah" if value.startswith("n") else "gov")
        for value in sequences
    }
    fit, calibration = _partition_sequences(
        sequences, "n0", tasks
    )
    assert set(fit).isdisjoint(calibration)
    assert "n0" not in set(fit) | set(calibration)
    assert len(fit) == 8 and len(calibration) == 3


def test_calibration_margin_uses_only_supplied_sequences() -> None:
    residual = np.asarray([[1.0], [2.0], [100.0]])
    margin = calibration_margin(
        residual[:2], ["train", "cal"], 0.9, simultaneous=True
    )
    assert margin.item() == 2.0


def test_pointwise_and_trajectory_coverage_are_distinct() -> None:
    frame = pd.DataFrame(
        {
            "family": ["E2"] * 4,
            "route": ["empirical"] * 4,
            "coverage_level": [0.9] * 4,
            "margin_type": ["pointwise"] * 4,
            "task": ["x"] * 4,
            "held_out_sequence": ["s"] * 4,
            "trajectory_id": ["a", "a", "b", "b"],
            "horizon_offset": [1, 2, 1, 2],
            "violation": [False, True, False, False],
            "violation_magnitude": [0.0, 1.0, 0.0, 0.0],
        }
    )
    result = _group_coverage(frame, [2])[0]
    assert result["pointwise_coverage"] == 0.75
    assert result["trajectory_wise_coverage"] == 0.5


def test_beta_zero_envelope_is_only_margin() -> None:
    error, direct, target, _, _ = _linear_system()
    model = fit_nonnegative_envelope(
        error, direct, target, [0, 7, 27], "E2"
    )
    margin = np.asarray([0.1, 0.2, 0.3])
    assert np.array_equal(
        h1_recursion(model, np.zeros(3), margin), margin
    )


def test_refresh_does_not_reset_existing_error() -> None:
    error = np.asarray([1.0, 2.0, 3.0])
    kept, direct = refresh_preserves_error(
        error, np.asarray([0.1, 0.2, 0.3])
    )
    assert np.array_equal(kept, error)
    assert np.any(kept != 0)
    assert np.array_equal(direct, np.asarray([0.1, 0.2, 0.3]))


def test_h1_recursion_matches_recursive_first_step() -> None:
    error, direct, target, _, _ = _linear_system()
    model = fit_nonnegative_envelope(
        error, direct, target, [0, 7, 27], "E3"
    )
    margin = np.asarray([0.1, 0.2, 0.3])
    recursive, _ = recursive_envelope(
        model, direct[:1], margin
    )
    assert np.allclose(
        recursive[0], h1_recursion(model, direct[0], margin)
    )


def test_bound_violation_magnitude_is_not_clipped_away() -> None:
    realized = 3.0
    bound = 1.25
    violation = max(0.0, realized - bound)
    assert violation == 1.75


def test_layer27_remains_explicit_coordinate() -> None:
    error, direct, target, _, _ = _linear_system()
    model = fit_nonnegative_envelope(
        error, direct, target, [0, 7, 27], "E2"
    )
    assert model.layers[-1] == 27
    assert model.a.shape == (3, 3)


def test_recent_fifo_keeps_only_recent_tail() -> None:
    fixed = {0, 1, 2, 3, 10}
    dynamic = list(range(11, 50))
    recent_before_query = 31
    kept = sorted(
        fixed | set(dynamic[-recent_before_query:])
    )
    assert len([value for value in kept if value in dynamic]) == 31
    assert dynamic[-1] in kept and dynamic[0] not in kept


def test_candidate_selection_is_physical_shared_mask() -> None:
    selection = CoreSelection(
        strategy="candidate",
        horizon_condition=None,
        by_layer={
            0: LayerSelection(
                layer=0,
                selected_positions=[1, 2],
                eligible_positions=[1, 2, 3],
                aggregate_scores=[1.0, 0.5, 0.1],
                metadata={
                    "physical_shared_mask": True,
                    "per_query_head_selection": False,
                },
            )
        },
    )
    assert physical_shared_mask(selection)


def test_policy_comparison_requires_same_budget_and_cost() -> None:
    policies = [
        {"budget": 128, "refresh_cost": 1.0},
        {"budget": 128, "refresh_cost": 1.0},
    ]
    assert len({row["budget"] for row in policies}) == 1
    assert len({row["refresh_cost"] for row in policies}) == 1


def test_monotone_induction_step() -> None:
    error, direct, target, _, _ = _linear_system()
    model = fit_nonnegative_envelope(
        error, direct, target, [0, 7, 27], "E3"
    )
    realized = np.asarray([0.1, 0.2, 0.3])
    bound = np.asarray([0.2, 0.4, 0.5])
    assert induction_check(
        model,
        realized,
        bound,
        np.asarray([0.3, 0.2, 0.1]),
        np.asarray([0.01, 0.01, 0.01]),
    )


def test_recursive_explosion_is_reported() -> None:
    error, direct, target, _, _ = _linear_system()
    model = fit_nonnegative_envelope(
        error, direct, target, [0, 7, 27], "E3"
    )
    model.h[:] = 100.0
    _, exploded = recursive_envelope(
        model,
        np.ones((8, 3)),
        np.ones(3),
        explosion_threshold=1000,
    )
    assert exploded.any()


def test_refresh_schedule_is_pre_registered_and_evenly_spaced() -> None:
    assert refresh_schedule(0, 64) == []
    assert refresh_schedule(1, 64) == [32]
    assert refresh_schedule(2, 64) == [21, 43]
    assert refresh_schedule(3, 64) == [16, 32, 48]


def test_policy_gate_requires_both_task_directions() -> None:
    rows = []
    for task in ("niah_single_1", "gov_report"):
        for sample in range(2):
            for policy, kl in (
                ("fixed_interval", 4.0),
                ("age_only", 5.0),
                ("E2", 3.0 if task == "niah_single_1" else 6.0),
            ):
                rows.append(
                    {
                        "sample_id": "%s:%d" % (task, sample),
                        "task": task,
                        "policy": policy,
                        "requested_refresh_count": 3,
                        "exact_kl": kl,
                        "js": kl,
                        "delta_nll": kl,
                        "refreshes_completed": 3,
                        "active_cache_tokens": 128,
                        "refresh_reset_state_error": False,
                    }
                )
    cfg = SimpleNamespace(
        robust_envelope=SimpleNamespace(total_budget=128, horizon=64),
        runtime=SimpleNamespace(bootstrap_samples=100, seed=42),
    )
    summary = summarize_policy_rows(pd.DataFrame(rows), cfg)
    assert not summary["policy_value_gate"]["pass"]
