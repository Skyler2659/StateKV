"""CPU-only tests for the counterfactual MVP diagnostic."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statekv.counterfactual_diagnostic import (
    aggregate_diagnostic_rows,
    attention_value_damage,
    ranking_metrics,
    stratified_candidate_groups,
    teacher_position_scores,
)


def _utilities(eligible, seed=0):
    rng = np.random.default_rng(seed)
    return {
        name: {int(p): float(value) for p, value in zip(eligible, rng.random(len(eligible)))}
        for name in ("A", "B", "C")
    }


def test_stratified_groups_deterministic_and_well_formed() -> None:
    eligible = list(range(4, 244))
    utilities = _utilities(eligible)
    first = stratified_candidate_groups(
        eligible=eligible, utility_scores=utilities, seed=42
    )
    second = stratified_candidate_groups(
        eligible=eligible, utility_scores=utilities, seed=42
    )
    assert [g["positions"] for g in first] == [g["positions"] for g in second]
    assert len(first) == 2 * 12  # two sizes x groups_per_size
    sizes = {4: 0, 8: 0}
    strata = set()
    for group in first:
        assert len(group["positions"]) == group["size"]
        assert len(set(group["positions"])) == group["size"]
        assert set(group["positions"]) <= set(eligible)
        sizes[group["size"]] += 1
        strata.add(group["stratum"])
    assert sizes == {4: 12, 8: 12}
    # 9 age x salience strata plus the random stratum are all covered.
    assert "random" in strata
    assert sum(1 for s in strata if s.startswith("age")) == 9


def test_stratified_groups_cover_low_salience_tokens() -> None:
    eligible = list(range(300))
    utilities = _utilities(eligible, seed=3)
    groups = stratified_candidate_groups(
        eligible=eligible, utility_scores=utilities, seed=1
    )
    consensus = np.zeros(len(eligible))
    for name in utilities:
        values = np.asarray([utilities[name][p] for p in eligible])
        ranks = np.empty(len(eligible))
        ranks[np.argsort(values, kind="stable")] = np.arange(len(eligible))
        consensus += ranks / (len(eligible) - 1)
    consensus /= len(utilities)
    row = {int(p): consensus[i] for i, p in enumerate(eligible)}
    sampled = [row[p] for g in groups for p in g["positions"]]
    # Not only salient tokens: the bottom salience tercile is represented.
    assert np.mean([value < 1.0 / 3.0 for value in sampled]) > 0.2


def test_stratified_groups_fail_loudly_on_tiny_pool() -> None:
    with pytest.raises(RuntimeError):
        stratified_candidate_groups(
            eligible=[1, 2, 3], utility_scores=_utilities([1, 2, 3]), seed=0
        )
    utilities = _utilities(list(range(100)))
    utilities["A"][7] = float("nan")
    with pytest.raises(RuntimeError, match="non-finite"):
        stratified_candidate_groups(
            eligible=list(range(100)), utility_scores=utilities, seed=0
        )


def test_ranking_metrics_extremes() -> None:
    damage = np.arange(8, dtype=np.float64)
    perfect = ranking_metrics(damage, damage.copy())
    assert perfect["spearman"] == pytest.approx(1.0)
    assert perfect["pairwise_accuracy"] == pytest.approx(1.0)
    assert perfect["top_damage_recall"] == pytest.approx(1.0)
    reversed_scores = damage[::-1].copy()
    worst = ranking_metrics(damage, reversed_scores)
    assert worst["spearman"] == pytest.approx(-1.0)
    assert worst["pairwise_accuracy"] == pytest.approx(0.0)
    assert worst["top_damage_recall"] == pytest.approx(0.0)


def test_ranking_metrics_partial_overlap() -> None:
    damage = np.arange(8, dtype=np.float64)
    # Top-quartile (k=2) of damage is {6, 7}; scores put 6 first and 0 second.
    scores = np.asarray([6.5, 1, 2, 3, 4, 5, 7.0, 0.5])
    metrics = ranking_metrics(damage, scores)
    assert metrics["top_damage_recall"] == pytest.approx(0.5)
    with pytest.raises(RuntimeError):
        ranking_metrics(damage[:3], damage[:3])
    with pytest.raises(RuntimeError):
        ranking_metrics(damage, np.where(damage > 3, np.nan, damage))


def test_attention_value_damage_synthetic() -> None:
    # One step, one layer, one head, three candidates, head_dim 2.
    attention = np.zeros((1, 1, 1, 3))
    values = np.asarray([[[[3.0, 4.0], [1.0, 0.0], [0.0, 2.0]]]])
    assert attention_value_damage(attention, values, [[0]])[0] == 0.0
    attention[0, 0, 0] = [0.5, 0.25, 0.25]
    # Removing token 0 alone: ||0.5 * [3, 4]|| = 2.5.
    assert attention_value_damage(attention, values, [[0]])[0] == pytest.approx(2.5)
    # Group contribution is the norm of the SUM, not the sum of norms:
    # ||0.5*[3,4] + 0.25*[0,2]|| = ||[1.5, 2.5]||.
    expected = float(np.linalg.norm([1.5, 2.5]))
    assert attention_value_damage(attention, values, [[0, 2]])[0] == pytest.approx(expected)
    with pytest.raises(RuntimeError):
        attention_value_damage(attention, values, [[7]])
    with pytest.raises(RuntimeError):
        attention_value_damage(attention[0], values, [[0]])


def _teacher_fixture():
    eligible = [5, 6, 7]
    scores = np.arange(24, dtype=np.float32).reshape(1, 2, 2, 2, 3)
    teacher = {
        "cycles": np.asarray([8], dtype=np.int16),
        "horizons": np.asarray([1, 16], dtype=np.int16),
        "position_ids": np.asarray([eligible + [-1, -1]], dtype=np.int32),
        "position_lengths": np.asarray([3], dtype=np.int32),
        "scores": np.pad(scores, ((0, 0), (0, 0), (0, 0), (0, 0), (0, 2)), constant_values=np.nan),
    }
    return teacher, eligible, scores


def test_teacher_position_scores_join() -> None:
    teacher, eligible, scores = _teacher_fixture()
    joined = teacher_position_scores(teacher, 8, 16, eligible)
    np.testing.assert_allclose(
        joined, scores[0, 1].mean(axis=(0, 1)), rtol=1.0e-6
    )
    with pytest.raises(RuntimeError, match="cycle"):
        teacher_position_scores(teacher, 24, 16, eligible)
    with pytest.raises(RuntimeError, match="horizon"):
        teacher_position_scores(teacher, 8, 4, eligible)
    with pytest.raises(RuntimeError, match="eligible"):
        teacher_position_scores(teacher, 8, 16, [5, 6, 8])


def test_aggregate_diagnostic_rows() -> None:
    rows = []
    for sample_id, task, cycle, spearman in (
        ("s1", "task_a", 8, 0.5),
        ("s1", "task_a", 24, 1.0),
        ("s2", "task_a", 8, 0.25),
        ("s3", "task_b", 8, 0.9),
    ):
        rows.append(
            {
                "sample_id": sample_id,
                "task": task,
                "cycle": cycle,
                "utility": "A_current_qk",
                "damage_metric": "realized_kl",
                "spearman": spearman,
                "pairwise_accuracy": 0.5,
                "top_damage_recall": 0.5,
                "groups": 24,
            }
        )
    summary = aggregate_diagnostic_rows(pd.DataFrame(rows))
    family_a = summary[
        (summary["task"] == "task_a") & (summary["utility"] == "A_current_qk")
    ].iloc[0]
    # Per-sample means first: s1 -> 0.75, s2 -> 0.25; family mean -> 0.5.
    assert family_a["spearman_mean"] == pytest.approx(0.5)
    assert int(family_a["sequences"]) == 2
    overall = summary[summary["task"] == "ALL"].iloc[0]
    assert overall["spearman_mean"] == pytest.approx((0.75 + 0.25 + 0.9) / 3.0)
    assert int(overall["sequences"]) == 3
