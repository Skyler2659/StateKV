from __future__ import annotations

import pytest

from statekv.oracle_closed_loop import (
    aggregate_teacher_scores,
    deterministic_uniform_core,
)


def test_uniform_core_is_deterministic_unique_and_budgeted() -> None:
    first = deterministic_uniform_core(list(range(20)), 7)
    second = deterministic_uniform_core(list(range(20)), 7)
    assert first == second
    assert len(first) == len(set(first)) == 7
    assert first[0] == 0
    assert first[-1] == 19


def test_teacher_aggregations_preserve_declared_objective() -> None:
    exact = {"stale": [2.0, 4.0], "fresh": [1.0, 5.0]}
    dense = {"stale": [0.0, 0.0], "fresh": [-0.5, 0.25]}
    assert aggregate_teacher_scores("exact_mean", exact, dense) == {
        "stale": 3.0,
        "fresh": 3.0,
    }
    assert aggregate_teacher_scores("exact_max", exact, dense) == {
        "stale": 4.0,
        "fresh": 5.0,
    }
    assert aggregate_teacher_scores("dense_quadratic_h1", exact, dense)[
        "fresh"
    ] == -0.5
    assert aggregate_teacher_scores("dense_quadratic_mean", exact, dense)[
        "fresh"
    ] == pytest.approx(-0.125)


def test_teacher_aggregation_rejects_mismatched_panels() -> None:
    with pytest.raises(ValueError, match="panels must match"):
        aggregate_teacher_scores(
            "exact_mean", {"stale": [0.0]}, {"other": [0.0]}
        )
