from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statekv.proxy_alignment import (
    proxy_action_risk,
    repair_stale_core,
    summarize_alignment,
    summarize_refresh,
)


def test_proxy_action_risk_is_deleted_mass_on_the_same_action() -> None:
    score = np.asarray([0.1, 0.4, 0.3, 0.2])
    assert proxy_action_risk(score, [10, 11, 12, 13], [10, 11, 12, 13], [11, 12]) == pytest.approx(0.3)


def test_repair_stale_core_preserves_legal_members_then_fills_by_current_score() -> None:
    repaired = repair_stale_core(
        previous_core=[1, 2, 8],
        positions=list(range(10)),
        score=np.asarray([0.0, 0.1, 0.2, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]),
        core_budget=3,
        sink_size=1,
        recent_size=2,
    )
    assert repaired == (1, 2, 3)


def test_alignment_summary_keeps_decision_units_separate() -> None:
    rows = pd.DataFrame(
        {
            "sample_id": ["s"] * 3,
            "task": ["t"] * 3,
            "anchor": [16] * 3,
            "horizon": [4] * 3,
            "proxy": ["p"] * 3,
            "candidate_policy": ["a", "b", "c"],
            "candidate_action_id": ["a", "b", "c"],
            "proxy_risk": [0.1, 0.2, 0.3],
            "teacher_risk": [0.1, 0.2, 0.3],
        }
    )
    units, summary = summarize_alignment(rows)
    assert units.iloc[0]["spearman"] == pytest.approx(1.0)
    assert summary[summary["task"] == "all"].iloc[0][
        "mean_normalized_regret"
    ] == pytest.approx(0.0)


def test_refresh_summary_uses_same_proxy_regret_against_teacher_benefit() -> None:
    rows = pd.DataFrame(
        {
            "proxy": ["p", "p", "p"],
            "task": ["t", "t", "t"],
            "proxy_regret": [0.1, 0.2, 0.3],
            "teacher_refresh_benefit": [-0.1, 0.0, 0.2],
        }
    )
    summary = summarize_refresh(rows)
    overall = summary[summary["task"] == "all"].iloc[0]
    assert overall["proxy_teacher_spearman"] == pytest.approx(1.0)
    assert overall["beneficial_refresh_fraction"] == pytest.approx(1.0 / 3.0)
