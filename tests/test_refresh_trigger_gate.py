"""Unit tests for the R2b trigger arm, capture metric, and gate verdict."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from statekv.refresh_trigger import (
    CHEAP_TRIGGER_FEATURES,
    TriggerRule,
    decide_trigger_refresh,
    load_trigger_rule,
)
from statekv.statekv_gate_analysis import (
    R2B_PREDECLARED_GATE,
    _capture,
    _evaluate_r2b_gate,
    _interpolated_fixed_k_capture,
    _r2b_per_sample_table,
    analyze_r2b,
)
from statekv.statekv_gate_runner import _r2b_arms, _scheduled_refresh


def _rule(clauses) -> TriggerRule:
    return load_trigger_rule({"name": "toy", "clauses": clauses})


# ---------------------------------------------------------------------------
# trigger rule parsing + allowlist enforcement
# ---------------------------------------------------------------------------
def test_rule_parses_single_and_double_clause_conjunctions() -> None:
    single = _rule([{"feature": "churn_jaccard_mean", "op": ">=", "threshold": 0.4}])
    assert single.evaluate({"churn_jaccard_mean": 0.5})
    assert not single.evaluate({"churn_jaccard_mean": 0.3})

    double = _rule(
        [
            {"feature": "churn_jaccard_mean", "op": ">=", "threshold": 0.4},
            {"feature": "boundary_margin_mean", "op": "<=", "threshold": 0.05},
        ]
    )
    features = {"churn_jaccard_mean": 0.5, "boundary_margin_mean": 0.04}
    assert double.evaluate(features)
    assert not double.evaluate({**features, "boundary_margin_mean": 0.06})
    assert not double.evaluate({**features, "churn_jaccard_mean": 0.1})


def test_rule_round_trips_through_a_json_file(tmp_path: Path) -> None:
    path = tmp_path / "rule.json"
    payload = {
        "name": "frozen_v1",
        "clauses": [{"feature": "score_tv_mean", "op": ">=", "threshold": 0.25}],
        "loso": {"auc_mean": 0.7},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    rule = load_trigger_rule(path)
    assert rule.name == "frozen_v1"
    assert rule.provenance["loso"]["auc_mean"] == 0.7
    assert rule.evaluate({"score_tv_mean": 0.25})


@pytest.mark.parametrize(
    "feature",
    [
        "exact_kl",
        "stale_exact_kl_lag4",
        "refresh_benefit_lag4",
        "full_probability_token_17",
        "compressed_entropy",
        "churn_jaccard_min",
        "",
    ],
)
def test_teacher_side_or_unlisted_features_are_rejected(feature: str) -> None:
    with pytest.raises(ValueError, match="cheap"):
        _rule([{"feature": feature, "op": ">=", "threshold": 0.1}])
    for allowed in CHEAP_TRIGGER_FEATURES:
        _rule([{"feature": allowed, "op": ">=", "threshold": 0.1}])


def test_malformed_rules_are_rejected() -> None:
    with pytest.raises(ValueError):
        _rule([])  # zero clauses
    with pytest.raises(ValueError):
        _rule(
            [
                {"feature": "score_tv_mean", "op": ">=", "threshold": 0.1},
                {"feature": "score_tv_mean", "op": ">=", "threshold": 0.2},
                {"feature": "score_tv_mean", "op": ">=", "threshold": 0.3},
            ]
        )  # three clauses
    with pytest.raises(ValueError):
        _rule([{"feature": "score_tv_mean", "op": "==", "threshold": 0.1}])
    with pytest.raises(ValueError):
        _rule([{"feature": "score_tv_mean", "op": ">=", "threshold": "high"}])
    with pytest.raises(ValueError):
        load_trigger_rule({"no_clauses": True})


def test_nan_features_never_fire() -> None:
    rule = _rule([{"feature": "coverage_mass_mean", "op": "<=", "threshold": 0.5}])
    assert not rule.evaluate({"coverage_mass_mean": float("nan")})
    assert rule.evaluate({"coverage_mass_mean": 0.5})
    with pytest.raises(KeyError):
        rule.evaluate({"churn_jaccard_mean": 0.9})


# ---------------------------------------------------------------------------
# trigger arm refresh pattern
# ---------------------------------------------------------------------------
def test_scheduled_refresh_matches_the_calendar_arms() -> None:
    assert all(_scheduled_refresh("every", cycle, 4) for cycle in range(9))
    assert [_scheduled_refresh("never", cycle, 4) for cycle in range(9)] == [
        True,
        *([False] * 8),
    ]
    assert [_scheduled_refresh("fixed_k", cycle, 4) for cycle in range(9)] == [
        True,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        True,
    ]
    assert all(_scheduled_refresh("never", cycle, 4, label_mode=True) for cycle in range(4))


def test_trigger_arm_refresh_pattern_on_a_toy_rule() -> None:
    rule = _rule([{"feature": "churn_jaccard_mean", "op": ">=", "threshold": 0.5}])
    churn = [0.9, 0.1, 0.7, 0.2, 0.8, 0.1]
    fired, refreshed = [], []
    for cycle in range(6):
        features = {
            "churn_jaccard_mean": churn[cycle],
            "boundary_margin_mean": float("nan"),
            "score_tv_mean": float("nan"),
            "coverage_mass_mean": float("nan"),
        }
        fired_now, refresh_now = decide_trigger_refresh(rule, features, cycle)
        fired.append(fired_now)
        refreshed.append(refresh_now)
    # cycle 0 refreshes unconditionally even though the rule would fire anyway;
    # later cycles refresh iff the rule fires.
    assert fired == [False, False, True, False, True, False]
    assert refreshed == [True, False, True, False, True, False]
    # cycle 0 never counts as a rule firing even with a NaN-gated rule
    fired0, refresh0 = decide_trigger_refresh(
        rule, {"churn_jaccard_mean": float("nan")}, 0
    )
    assert (fired0, refresh0) == (False, True)


# ---------------------------------------------------------------------------
# capture metric + matched-refresh interpolation
# ---------------------------------------------------------------------------
def test_capture_math_and_denominator_guard() -> None:
    assert _capture(1.0, 1.0, 0.0) == 0.0  # never arm recovers nothing
    assert _capture(1.0, 0.0, 0.0) == 1.0  # fresh arm recovers everything
    assert abs(_capture(1.0, 0.25, 0.0) - 0.75) < 1.0e-12
    assert _capture(1.0, 0.5, 1.0) != _capture(1.0, 0.5, 1.0)  # NaN guard
    assert np.isnan(_capture(1.0, 0.5, 1.0 - 1.0e-12))
    assert np.isnan(_capture(float("nan"), 0.5, 0.0))


def test_fixed_k_interpolation_inside_range() -> None:
    points = [(16.0, 0.80), (8.0, 0.60), (4.0, 0.30)]  # unsorted on purpose
    matched = _interpolated_fixed_k_capture(points, 6.0)
    assert matched["interpolated"] is True
    assert matched["outside_fixed_k_range"] is False
    # halfway between (4, 0.30) and (8, 0.60)
    assert abs(matched["reference_capture"] - 0.45) < 1.0e-12


def test_fixed_k_out_of_range_uses_nearest_arm_and_flags() -> None:
    points = [(4.0, 0.30), (8.0, 0.60), (16.0, 0.80)]
    low = _interpolated_fixed_k_capture(points, 2.0)
    assert low["interpolated"] is False
    assert low["outside_fixed_k_range"] is True
    assert low["reference_capture"] == 0.30
    high = _interpolated_fixed_k_capture(points, 64.0)
    assert high["outside_fixed_k_range"] is True
    assert high["reference_capture"] == 0.80
    edge = _interpolated_fixed_k_capture(points, 8.0)
    assert edge["interpolated"] is True
    assert edge["reference_capture"] == 0.60


# ---------------------------------------------------------------------------
# gate verdict logic on synthetic arm tables
# ---------------------------------------------------------------------------
def _synthetic_run(
    tmp_path: Path,
    trigger_capture: float = 0.75,
    trigger_refreshes: int = 8,
    trigger_score: float = 0.55,
    every_score: float = 0.60,
    flags_hold: bool = True,
    include_trigger: bool = True,
):
    """Build a synthetic 4-arm r2b run dir (2 samples x 1 policy)."""
    arms = ["every", "never", "fixed_k4", "fixed_k8", "fixed_k16"]
    if include_trigger:
        arms.append("trigger")
    sample_ids = ["gov_report:101", "synthetic_niah_101"]
    kl_never, kl_every = 1.0, 0.0
    arm_kl = {
        "every": kl_every,
        "never": kl_never,
        "fixed_k4": kl_never - 0.30 * (kl_never - kl_every),
        "fixed_k8": kl_never - 0.60 * (kl_never - kl_every),
        "fixed_k16": kl_never - 0.80 * (kl_never - kl_every),
        "trigger": kl_never - trigger_capture * (kl_never - kl_every),
    }
    arm_refreshes = {
        "every": 64,
        "never": 1,
        "fixed_k4": 17,
        "fixed_k8": 9,
        "fixed_k16": 5,
        "trigger": trigger_refreshes,
    }
    cycles = 64
    sample_rows, step_rows, cycle_rows = [], [], []
    for sample_id in sample_ids:
        for arm in arms:
            score = every_score if arm != "trigger" else trigger_score
            sample_rows.append(
                {
                    "sample_id": sample_id,
                    "task": "govreport_or_qmsum",
                    "task_bucket": "GovReport",
                    "policy": "attention",
                    "arm": arm,
                    "official_score": score,
                    "rouge_l": score,
                    "needle_retrieval_accuracy": float("nan"),
                    "mean_trajectory_exact_kl": arm_kl[arm],
                    "refresh_count": arm_refreshes[arm],
                }
            )
            for cycle in range(cycles):
                refreshed = cycle < arm_refreshes[arm]
                step_rows.append(
                    {
                        "sample_id": sample_id,
                        "policy": "attention",
                        "arm": arm,
                        "cycle": cycle,
                        "exact_kl": arm_kl[arm],
                    }
                )
                cycle_rows.append(
                    {
                        "sample_id": sample_id,
                        "policy": "attention",
                        "arm": arm,
                        "cycle": cycle,
                        "ranking_refreshed": bool(refreshed),
                        "trigger_fired": bool(arm == "trigger" and refreshed and cycle > 0),
                        "refresh_count": int(cycle + 1 if refreshed else arm_refreshes[arm]),
                        "irreversible_set_inclusion": bool(flags_hold),
                        "global_kv_budget_respected": True,
                    }
                )
    run_dir = tmp_path / "r2b_run"
    run_dir.mkdir(parents=True)
    pd.DataFrame(sample_rows).to_csv(run_dir / "sample_results.csv", index=False)
    pd.DataFrame(step_rows).to_parquet(run_dir / "step_rows.parquet")
    pd.DataFrame(cycle_rows).to_parquet(run_dir / "cycle_rows.parquet")
    config = {
        "r2b_gate": {
            "output_run": str(run_dir.relative_to(tmp_path)),
            "sample_ids": sample_ids,
        }
    }
    config_path = tmp_path / "r2b.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path, run_dir


def test_per_sample_table_computes_capture_and_counts(tmp_path: Path) -> None:
    config_path, run_dir = _synthetic_run(tmp_path)
    table = _r2b_per_sample_table(
        pd.read_csv(run_dir / "sample_results.csv"),
        pd.read_parquet(run_dir / "step_rows.parquet"),
        pd.read_parquet(run_dir / "cycle_rows.parquet"),
    )
    trigger = table[table["arm"] == "trigger"]
    assert set(table["arm"].unique()) == {
        "every", "never", "fixed_k4", "fixed_k8", "fixed_k16", "trigger",
    }
    assert abs(trigger["capture"].mean() - 0.75) < 1.0e-12
    assert int(trigger["refresh_count"].max()) == 8
    assert abs(table[table["arm"] == "never"]["capture"].mean() - 0.0) < 1.0e-12
    assert abs(table[table["arm"] == "every"]["capture"].mean() - 1.0) < 1.0e-12


def test_gate_passes_when_trigger_beats_matched_fixed_k(tmp_path: Path) -> None:
    config_path, run_dir = _synthetic_run(tmp_path, trigger_capture=0.75, trigger_refreshes=8)
    table = _r2b_per_sample_table(
        pd.read_csv(run_dir / "sample_results.csv"),
        pd.read_parquet(run_dir / "step_rows.parquet"),
        pd.read_parquet(run_dir / "cycle_rows.parquet"),
    )
    gate = _evaluate_r2b_gate(table, pd.read_parquet(run_dir / "cycle_rows.parquet"))
    assert gate["verdict"] == "pass"
    row = gate["per_policy"][0]
    # trigger at 8 refreshes interpolates between fixed_k16 (5 refreshes,
    # capture 0.80) and fixed_k8 (9 refreshes, capture 0.60) in refresh-count
    # space: 0.60 + (0.80 - 0.60) * (8 - 9) / (5 - 9) = 0.65
    assert row["matched_fixed_k"]["interpolated"] is True
    assert abs(row["matched_fixed_k"]["reference_capture"] - 0.65) < 1.0e-12
    assert row["condition_a_capture_positive"]
    assert row["condition_b_beats_matched_fixed_k"]
    assert row["condition_c_quality_within_tolerance"]
    assert row["condition_d_budget_flags_hold"]
    assert row["gate_pass"]


def test_gate_fails_each_condition_independently(tmp_path: Path) -> None:
    cycles = pd.read_parquet(_synthetic_run(tmp_path)[1] / "cycle_rows.parquet")

    # (b) trigger capture (0.62) below interpolated fixed-k (0.70) + 0.05
    _, run_dir = _synthetic_run(tmp_path / "b", trigger_capture=0.62, trigger_refreshes=8)
    gate = _evaluate_r2b_gate(
        _r2b_per_sample_table(
            pd.read_csv(run_dir / "sample_results.csv"),
            pd.read_parquet(run_dir / "step_rows.parquet"),
            pd.read_parquet(run_dir / "cycle_rows.parquet"),
        ),
        cycles,
    )
    row = gate["per_policy"][0]
    assert row["condition_a_capture_positive"]
    assert not row["condition_b_beats_matched_fixed_k"]
    assert gate["verdict"] == "fail"

    # (a) trigger capture <= 0
    _, run_dir = _synthetic_run(tmp_path / "a", trigger_capture=-0.1, trigger_refreshes=8)
    gate = _evaluate_r2b_gate(
        _r2b_per_sample_table(
            pd.read_csv(run_dir / "sample_results.csv"),
            pd.read_parquet(run_dir / "step_rows.parquet"),
            pd.read_parquet(run_dir / "cycle_rows.parquet"),
        ),
        cycles,
    )
    assert not gate["per_policy"][0]["condition_a_capture_positive"]
    assert gate["verdict"] == "fail"

    # (c) trigger quality more than 0.5 below the every arm
    _, run_dir = _synthetic_run(
        tmp_path / "c", trigger_capture=0.9, trigger_score=0.05, every_score=0.60
    )
    gate = _evaluate_r2b_gate(
        _r2b_per_sample_table(
            pd.read_csv(run_dir / "sample_results.csv"),
            pd.read_parquet(run_dir / "step_rows.parquet"),
            pd.read_parquet(run_dir / "cycle_rows.parquet"),
        ),
        cycles,
    )
    assert not gate["per_policy"][0]["condition_c_quality_within_tolerance"]
    assert gate["verdict"] == "fail"

    # (d) budget flags broken -> fail even when a/b/c hold
    _, run_dir = _synthetic_run(tmp_path / "d", trigger_capture=0.9, flags_hold=False)
    gate = _evaluate_r2b_gate(
        _r2b_per_sample_table(
            pd.read_csv(run_dir / "sample_results.csv"),
            pd.read_parquet(run_dir / "step_rows.parquet"),
            pd.read_parquet(run_dir / "cycle_rows.parquet"),
        ),
        pd.read_parquet(run_dir / "cycle_rows.parquet"),
    )
    assert not gate["per_policy"][0]["condition_d_budget_flags_hold"]
    assert gate["verdict"] == "fail"

    # trigger outside the fixed-k refresh range -> nearest arm, flagged
    _, run_dir = _synthetic_run(tmp_path / "e", trigger_capture=0.85, trigger_refreshes=2)
    gate = _evaluate_r2b_gate(
        _r2b_per_sample_table(
            pd.read_csv(run_dir / "sample_results.csv"),
            pd.read_parquet(run_dir / "step_rows.parquet"),
            pd.read_parquet(run_dir / "cycle_rows.parquet"),
        ),
        cycles,
    )
    matched = gate["per_policy"][0]["matched_fixed_k"]
    assert matched["outside_fixed_k_range"] is True
    assert matched["reference_capture"] == 0.80  # nearest = fixed_k16


def test_no_trigger_arm_yields_curve_only_verdict(tmp_path: Path) -> None:
    _, run_dir = _synthetic_run(tmp_path, include_trigger=False)
    gate = _evaluate_r2b_gate(
        _r2b_per_sample_table(
            pd.read_csv(run_dir / "sample_results.csv"),
            pd.read_parquet(run_dir / "step_rows.parquet"),
            pd.read_parquet(run_dir / "cycle_rows.parquet"),
        ),
        pd.read_parquet(run_dir / "cycle_rows.parquet"),
    )
    assert gate["verdict"] == "no-trigger-frozen-interval-curve-only"
    assert gate["per_policy"] == []
    assert len(gate["refresh_efficiency"]) == 5  # 5 arms x 1 policy


def test_analyze_r2b_end_to_end_on_a_synthetic_run(tmp_path: Path) -> None:
    config_path, run_dir = _synthetic_run(tmp_path, trigger_capture=0.9)
    output = analyze_r2b(config_path, tmp_path)
    assert output == run_dir / "gate_analysis.json"
    analysis = json.loads(output.read_text(encoding="utf-8"))
    assert analysis["stage"] == "r2b_selective_refresh_gate"
    assert analysis["gate"]["verdict"] == "pass"
    assert analysis["predeclared_gate"] == R2B_PREDECLARED_GATE
    assert analysis["execution_valid"] is True
    assert (run_dir / "r2b_capture_by_sample.csv").exists()
    assert (run_dir / "r2b_refresh_efficiency.csv").exists()
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["predeclared_gate"] == R2B_PREDECLARED_GATE
    assert summary["gate"]["verdict"] == "pass"


# ---------------------------------------------------------------------------
# arm construction (trigger skipped without a frozen rule file)
# ---------------------------------------------------------------------------
def test_r2b_arms_skip_trigger_when_rule_file_is_missing(tmp_path: Path) -> None:
    arms, skipped = _r2b_arms({"refresh_k": [4, 8], "trigger_rule": "missing.json"}, tmp_path)
    assert [arm["arm"] for arm in arms] == ["every", "never", "fixed_k4", "fixed_k8"]
    assert skipped["skipped"] is True
    assert "missing" in skipped["reason"]
    arms, skipped = _r2b_arms({}, tmp_path)
    assert [arm["arm"] for arm in arms] == [
        "every", "never", "fixed_k4", "fixed_k8", "fixed_k16",
    ]
    assert skipped["reason"] == "no trigger_rule configured"


def test_r2b_arms_include_trigger_when_rule_file_exists(tmp_path: Path) -> None:
    rule_path = tmp_path / "rule.json"
    rule_path.write_text(
        json.dumps(
            {"clauses": [{"feature": "score_tv_mean", "op": ">=", "threshold": 0.2}]}
        ),
        encoding="utf-8",
    )
    arms, skipped = _r2b_arms({"trigger_rule": "rule.json"}, tmp_path)
    assert skipped is None
    assert arms[-1]["arm"] == "trigger"
    assert arms[-1]["refresh_mode"] == "trigger"
    assert arms[-1]["trigger_rule_path"] == str(rule_path)
