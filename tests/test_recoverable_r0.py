"""Recoverable Gate R0 semantics tests.

Covers the fairness invariants of analysis/statekv_recoverable_r0_protocol.md:
re-entry legality, shared candidate universe, budget fill, same-input KL,
and the new recoverable selection rules (recency / qk_pool / quest_like).
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from statekv.oracle_closed_loop import (
    KVBackingStore,
    quest_like_core,
    recency_core,
)
from statekv.oracle_policy_comparison import (
    AttentionPolicyMemory,
    _physical_candidate_panel,
)
from statekv.oracle_policy_freegen import (
    _advance_full_state,
    _aggregate_free_results,
)
from statekv.selectors import mandatory_and_eligible


POSITIONS = list(range(60))
SINK = 4
RECENT = 8
CORE = 20
LAYERS = (0, 1)


def _fake_backing() -> KVBackingStore:
    backing = KVBackingStore()
    for layer in LAYERS:
        backing.keys[int(layer)] = {
            int(position): torch.zeros(1, 2, 1, 4) for position in POSITIONS
        }
        backing.values[int(layer)] = {
            int(position): torch.zeros(1, 2, 1, 4) for position in POSITIONS
        }
    return backing


def _fake_memory() -> AttentionPolicyMemory:
    memory = AttentionPolicyMemory(LAYERS, window_size=2)
    for layer in LAYERS:
        row = {int(position): 0.01 * (position % 7) for position in POSITIONS}
        memory.latest[int(layer)] = dict(row)
        memory.cumulative[int(layer)] = dict(row)
        memory.window[int(layer)] = [dict(row)]
    return memory


def _pool_scores(favorite: int) -> dict:
    scores = {}
    for layer in LAYERS:
        row = {int(position): 0.001 for position in POSITIONS}
        row[int(favorite)] = 5.0
        scores[int(layer)] = row
    return scores


def _panel(candidate_names, pool_scores=None):
    return _physical_candidate_panel(
        None,
        SimpleNamespace(cache=[None for _ in LAYERS]),
        _fake_backing(),
        _fake_memory(),
        None,
        candidate_names,
        SINK,
        RECENT,
        CORE,
        63,
        "max",
        pool_scores=pool_scores,
        quest_page_size=16,
    )


def test_recency_core_selects_most_recent_eligible() -> None:
    assert recency_core([1, 2, 3, 4, 5], 3) == (3, 4, 5)
    assert recency_core([1, 2], 5) == (1, 2)
    assert recency_core([1, 2], 0) == tuple()


def test_quest_like_core_page_rule_and_budget() -> None:
    eligible = list(range(32))
    scores = {position: 0.0 for position in eligible}
    # Page 2 (positions 16-31) dominates; inside page 1 the late tokens win.
    for position in range(16, 32):
        scores[position] = 1.0
    scores[15] = 0.9
    scores[14] = 0.8
    scores[13] = 0.7
    scores[12] = 0.6
    core = quest_like_core(eligible, scores, 16, 20)
    assert len(core) == 20
    assert set(range(16, 32)).issubset(set(core))
    assert set(core) - set(range(16, 32)) == {12, 13, 14, 15}
    # Deterministic tie-breaks: identical call, identical core.
    assert quest_like_core(eligible, scores, 16, 20) == core
    # Whole-page case: budget exactly one page.
    assert quest_like_core(eligible, scores, 16, 16) == tuple(range(16, 32))
    with pytest.raises(ValueError, match="page size"):
        quest_like_core(eligible, scores, 0, 20)


def test_panel_new_candidates_share_universe_and_fill_budget() -> None:
    candidates = ["stale", "uniform", "recency", "qk_pool", "quest_like"]
    panel = _panel(candidates, pool_scores=_pool_scores(favorite=10))
    _, _, expected_eligible = mandatory_and_eligible(POSITIONS, SINK, RECENT)
    for name in candidates:
        selection = panel[name]
        for layer in LAYERS:
            layer_selection = selection.by_layer[int(layer)]
            # Candidate universe equality: every candidate sees the same
            # eligible set derived from the shared backing pool.
            assert [int(v) for v in layer_selection.eligible_positions] == [
                int(v) for v in expected_eligible
            ]
            core = [int(v) for v in layer_selection.selected_positions]
            assert len(core) == min(CORE, len(expected_eligible))
            assert set(core) <= set(int(v) for v in expected_eligible)


def test_qk_pool_core_is_topk_of_pool_scores() -> None:
    pool = _pool_scores(favorite=10)
    panel = _panel(["qk_pool"], pool_scores=pool)
    _, _, eligible = mandatory_and_eligible(POSITIONS, SINK, RECENT)
    expected = sorted(
        sorted(eligible, key=lambda p: (-pool[0][int(p)], int(p)))[:CORE]
    )
    core = [int(v) for v in panel["qk_pool"].by_layer[0].selected_positions]
    assert core == expected
    assert 10 in core


def test_full_pool_scores_allow_reentry_of_inactive_positions() -> None:
    # Position 10 is eligible but pretend it was evicted long ago: it is in
    # the backing universe yet not in the "active" set {40..59}.  A pool
    # score that favors it must pull it back into the core (re-entry is
    # legal for every recoverable policy, cheap ones included).
    active = set(range(40, 60))
    assert 10 not in active
    panel = _panel(["qk_pool", "quest_like"], pool_scores=_pool_scores(10))
    for name in ("qk_pool", "quest_like"):
        core = {
            int(v) for v in panel[name].by_layer[0].selected_positions
        }
        assert 10 in core
        # The core is drawn from the full backing universe, not the active
        # set: most core members are inactive positions.
        assert len(core - active) > 0


def test_recency_core_in_panel_matches_tail_of_eligible() -> None:
    panel = _panel(["recency"])
    _, _, eligible = mandatory_and_eligible(POSITIONS, SINK, RECENT)
    core = [int(v) for v in panel["recency"].by_layer[1].selected_positions]
    assert core == [int(v) for v in eligible[-CORE:]]


def test_pool_candidates_require_pool_scores() -> None:
    with pytest.raises(ValueError, match="requires full-pool scores"):
        _panel(["qk_pool"], pool_scores=None)
    with pytest.raises(ValueError, match="requires full-pool scores"):
        _panel(["quest_like"], pool_scores=None)


class _TokenRecordingModel:
    def __init__(self) -> None:
        self.seen: list = []

    def forward_one(self, state, token, capture_attention=True):
        self.seen.append(int(token))
        logits = torch.zeros(8, dtype=torch.float32)
        logits[int(token) % 8] = 1.0
        return logits, None, 0.0


def test_advance_full_state_consumes_compressed_tokens_same_input() -> None:
    # The trajectory KL evaluator must advance the full-cache state on the
    # compressed arm's own tokens (same-input semantics), never on the
    # reference generation.
    model = _TokenRecordingModel()
    runner = SimpleNamespace(model=model)
    compressed_logits = [
        torch.tensor([0.1, 0.2, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0]),
        torch.tensor([0.0, 0.0, 0.1, 0.8, 0.1, 0.0, 0.0, 0.0]),
    ]
    rows = _advance_full_state(
        runner, SimpleNamespace(), 5, [2, 3], compressed_logits
    )
    assert model.seen == [5, 2]
    assert len(rows) == 2
    for row in rows:
        assert "exact_kl" in row
        assert float(row["exact_kl"]) >= 0.0


def test_aggregate_derives_recoverable_baselines_excluding_full_cache() -> None:
    rows = []
    for policy, kl in (
        ("statekv_exact_mean", 0.05),
        ("qk_pool", 0.10),
        ("uniform", 0.20),
        ("full_cache", 0.0),
    ):
        rows.append(
            {
                "sample_id": "synthetic_niah_86",
                "policy": policy,
                "task_bucket": "NIAH",
                "official_score": 100.0,
                "mean_trajectory_exact_kl": kl,
                "rouge_l": None,
                "needle_retrieval_accuracy": 1.0,
            }
        )
        rows.append(
            {
                "sample_id": "gov_report:86",
                "policy": policy,
                "task_bucket": "GovReport",
                "official_score": 10.0,
                "mean_trajectory_exact_kl": kl,
                "rouge_l": 10.0,
                "needle_retrieval_accuracy": None,
            }
        )
    result = _aggregate_free_results(pd.DataFrame(rows))
    baselines = {row["baseline"] for row in result["paired_comparisons"]}
    assert baselines == {"qk_pool", "uniform"}
    assert result["overall_comparisons"]["best_fixed_kl_policy"] == "qk_pool"


def test_observation_window_tokens_past_only() -> None:
    from statekv.oracle_policy_freegen import _observation_window_tokens

    prompt = list(range(100))
    generated = [100, 101, 102]
    window = _observation_window_tokens(prompt, generated, 32)
    assert window == list(range(71, 103))
    # never contains future tokens
    assert max(window) <= 102
    # window larger than history degrades gracefully
    assert _observation_window_tokens([5, 6], [], 32) == [5, 6]


def test_mean_score_rows_position_keyed() -> None:
    from statekv.oracle_policy_freegen import _mean_score_rows

    rows = [
        {0: {10: 0.2, 11: 0.8}},
        {0: {10: 0.4, 12: 1.0}},
    ]
    out = _mean_score_rows(rows)
    assert out[0][10] == 0.30000000000000004
    assert out[0][11] == 0.8  # absent from row 2 -> mean over present rows
    assert out[0][12] == 1.0
