"""Contract tests for the no-gate retest panel and the token_rarity policy."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from statekv.oracle_policy_comparison import token_rarity_scores
from statekv.retest_freegen import (
    _aggregates,
    _paired_comparisons,
    classify_policies,
)


def test_classify_policies_splits_groups() -> None:
    groups = classify_policies(
        [
            "attention",
            "qk_pool",
            "qk_tiered_v",
            "token_rarity",
            "a2_temporal_volatility",
            "b3_layer_adaptive_budget",
        ]
    )
    assert groups["cheap"] == ["a2_temporal_volatility", "b3_layer_adaptive_budget"]
    assert groups["tiered"] == ["qk_tiered_v"]
    assert groups["panel"] == ["attention", "qk_pool", "token_rarity"]


def test_token_rarity_scores_match_frozen_formula() -> None:
    # stream of 6 tokens with one repeated id; span is clipped to +-2
    ids = [10, 11, 10, 12, 13, 10]
    positions = list(range(6))
    scores = token_rarity_scores(ids, positions)
    total = len(ids)
    from collections import Counter

    counts = Counter(ids)
    for position in positions:
        start = max(0, position - 2)
        end = min(len(ids), position + 3)
        expected = sum(
            math.log((total + 1.0) / (counts[ids[j]] + 1.0))
            for j in range(start, end)
        ) / (end - start)
        assert scores[position] == pytest.approx(expected, rel=1e-12)
    # rare tokens (id 12/13, count 1) score above the frequent one (id 10)
    assert scores[3] > scores[0]
    # out-of-stream positions score zero
    assert token_rarity_scores(ids, [9])[0] == 0.0


def _frame() -> pd.DataFrame:
    rows = []
    for index, sample in enumerate(["s1", "s2", "s3", "s4"]):
        bucket = "GovReport" if index < 2 else "NIAH"
        for policy, kl, score in (
            ("full_cache", 0.0, 10.0),
            ("attention", 0.3, 8.0 + index),
            ("qk_pool", 0.1, 9.0),
        ):
            rows.append(
                {
                    "sample_id": sample,
                    "task_bucket": bucket,
                    "policy": policy,
                    "official_score": float(score),
                    "mean_trajectory_exact_kl": float(kl),
                    "repetition_4gram_rate": 0.01,
                    "rouge_l": 5.0 if bucket == "GovReport" else None,
                    "needle_retrieval_accuracy": (
                        None if bucket == "GovReport" else 1.0
                    ),
                    "all_budgets_respected": True,
                }
            )
    return pd.DataFrame(rows)


def test_aggregates_report_bucket_means_without_verdicts() -> None:
    frame = _frame()
    steps = pd.DataFrame(
        {
            "policy": ["attention", "attention", "qk_pool", "qk_pool"],
            "delta_nll": [0.01, 0.03, 0.0, 0.02],
        }
    )
    rows = {row["policy"]: row for row in _aggregates(frame, steps)}
    assert rows["attention"]["mean_govreport_rouge_l"] == pytest.approx(5.0)
    assert rows["attention"]["mean_niah_retrieval"] == pytest.approx(1.0)
    assert rows["attention"]["mean_reasoning_official"] is None
    assert rows["attention"]["mean_delta_nll"] == pytest.approx(0.02)
    assert not any("pass" in key or "gate" in key for key in rows["attention"])


def test_paired_comparisons_are_all_pairs_and_continuous() -> None:
    rows = _paired_comparisons(_frame(), seed=0, bootstrap_samples=100)
    pairs = {
        (row["policy"], row["baseline"], row["metric"], row["task_bucket"])
        for row in rows
    }
    assert ("attention", "qk_pool", "official_score", "all") in pairs
    assert ("full_cache", "qk_pool", "mean_trajectory_exact_kl", "all") in pairs
    # bucket-specific metrics only pair within their bucket
    assert ("attention", "qk_pool", "rouge_l", "GovReport") in pairs
    assert ("attention", "qk_pool", "needle_retrieval_accuracy", "NIAH") in pairs
    for row in rows:
        assert row["paired_samples"] > 0
        assert math.isfinite(row["mean_delta_policy_minus_baseline"])
        assert row["wins"] + row["ties"] + row["losses"] == row["paired_samples"]
