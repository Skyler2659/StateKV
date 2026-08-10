"""Model-free invariants of the pure cache-selection primitives.

These tests pin the contract shared by every closed-loop controller: budgets
are exact, sink/recent positions are mandatory, and selection is
deterministic.  No model, backend, or benchmark harness is involved.
"""

import numpy as np

from statekv.oracle_closed_loop import (
    _top_core,
    deterministic_uniform_core,
    quest_like_core,
    recency_core,
)
from statekv.oracle_policy_comparison import token_rarity_scores
from statekv.selectors import mandatory_and_eligible


POSITIONS = list(range(16))
ELIGIBLE = [2, 3, 5, 7, 8, 11, 13]


def test_mandatory_and_eligible_partitions_positions() -> None:
    sink, recent, eligible = mandatory_and_eligible(POSITIONS, 2, 3)
    assert sink == [0, 1]
    assert recent == [13, 14, 15]
    mandatory = set(sink) | set(recent)
    assert mandatory == {0, 1, 13, 14, 15}
    assert set(eligible) == set(POSITIONS) - mandatory
    assert not mandatory.intersection(eligible)
    assert sorted(mandatory | set(eligible)) == POSITIONS


def test_mandatory_and_eligible_edge_cases() -> None:
    # sink + recent covers the whole stream: nothing remains eligible.
    sink, recent, eligible = mandatory_and_eligible([0, 1, 2, 3], 2, 2)
    assert sink == [0, 1]
    assert recent == [2, 3]
    assert eligible == []
    # sink + recent overlapping: mandatory stays a disjoint union.
    sink, recent, eligible = mandatory_and_eligible([0, 1, 2], 2, 2)
    assert set(sink) | set(recent) == {0, 1, 2}
    assert eligible == []
    # oversized requests clamp to the stream length.
    sink, recent, eligible = mandatory_and_eligible([0, 1], 8, 8)
    assert eligible == []
    # zero sizes leave everything eligible.
    sink, recent, eligible = mandatory_and_eligible(POSITIONS, 0, 0)
    assert sink == [] and recent == []
    assert eligible == POSITIONS


def test_top_core_exact_budget_from_eligible() -> None:
    score = np.zeros(len(POSITIONS), dtype=np.float64)
    score[7] = 0.9
    score[3] = 0.8
    score[0] = 1.0  # not eligible: must never be selected
    selected = _top_core(POSITIONS, ELIGIBLE, score, 2)
    assert selected == (3, 7)
    assert len(set(selected)) == 2
    assert set(selected) <= set(ELIGIBLE)


def test_top_core_clamps_to_eligible_size() -> None:
    score = np.arange(len(POSITIONS), dtype=np.float64)
    selected = _top_core(POSITIONS, ELIGIBLE, score, 100)
    assert selected == tuple(sorted(ELIGIBLE))
    assert _top_core(POSITIONS, ELIGIBLE, score, 0) == ()


def test_top_core_deterministic_under_ties() -> None:
    score = np.ones(len(POSITIONS), dtype=np.float64)
    first = _top_core(POSITIONS, ELIGIBLE, score, 3)
    second = _top_core(POSITIONS, ELIGIBLE, score, 3)
    assert first == second
    assert len(first) == 3
    # ties break toward the lower position.
    assert first == (2, 3, 5)


def test_deterministic_uniform_core_contract() -> None:
    first = deterministic_uniform_core(ELIGIBLE, 3)
    second = deterministic_uniform_core(ELIGIBLE, 3)
    assert first == second
    assert len(first) == 3
    assert len(set(first)) == 3
    assert set(first) <= set(ELIGIBLE)
    assert deterministic_uniform_core(ELIGIBLE, 100) == tuple(sorted(ELIGIBLE))
    assert deterministic_uniform_core(ELIGIBLE, 0) == ()


def test_recency_core_contract() -> None:
    first = recency_core(ELIGIBLE, 3)
    second = recency_core(ELIGIBLE, 3)
    assert first == second == (8, 11, 13)
    assert set(first) <= set(ELIGIBLE)
    assert recency_core(ELIGIBLE, 100) == tuple(ELIGIBLE)
    assert recency_core(ELIGIBLE, 0) == ()


def test_quest_like_core_page_granularity_and_budget() -> None:
    page_size = 3
    eligible = list(range(2, 17))  # 15 positions -> 5 full pages
    scores = {position: 0.0 for position in eligible}
    # page [8, 9, 10] has the strongest token, then page [2, 3, 4].
    scores[9] = 10.0
    scores[3] = 5.0
    first = quest_like_core(eligible, scores, page_size, 5)
    second = quest_like_core(eligible, scores, page_size, 5)
    assert first == second
    assert len(first) == 5
    assert set(first) <= set(eligible)
    # the best page is retained as a whole block of page_size positions.
    assert {8, 9, 10} <= set(first)
    # the budget-crossing page contributes only its top-scored tokens.
    assert 3 in first


def test_quest_like_core_edge_cases() -> None:
    scores = {position: float(position) for position in ELIGIBLE}
    assert quest_like_core(ELIGIBLE, scores, 4, 100) == tuple(sorted(ELIGIBLE))
    assert quest_like_core(ELIGIBLE, scores, 4, 0) == ()
    try:
        quest_like_core(ELIGIBLE, scores, 0, 2)
    except ValueError:
        pass
    else:  # pragma: no cover - defensive
        raise AssertionError("non-positive page size must raise")


def test_token_rarity_scores_deterministic_and_bounded() -> None:
    stream = [5, 5, 5, 9, 5, 5, 9, 5]
    positions = [0, 3, 8, -1]
    first = token_rarity_scores(stream, positions)
    second = token_rarity_scores(stream, positions)
    np.testing.assert_array_equal(first, second)
    # out-of-range positions score exactly 0.
    assert first[2] == 0.0
    assert first[3] == 0.0
    # the rarer token (9 at index 3) scores above the repeated token (5).
    assert first[1] > first[0]
    assert first[1] > 0.0


def test_token_rarity_scores_empty_stream() -> None:
    scores = token_rarity_scores([], [0, 1])
    np.testing.assert_array_equal(scores, np.zeros(2, dtype=np.float64))
