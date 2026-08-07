from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = (
    ROOT / "experiments/p3_decision_validity/scripts"
)
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from p3_core import (  # noqa: E402
    LOW_COST_FEATURES,
    ZERO_COST_FEATURES,
    all_numeric_finite,
    assert_no_future_columns,
    atomic_frame,
    atomic_json,
    choose_harmful_epsilon,
    component_swap_scores,
    decision_event,
    detector_metrics,
    deterministic_projection,
    forward_cost,
    isolation_check,
    mean_rank_disagreement,
    normalized_regret,
    pairwise_accuracy,
    physical_alignment,
    prefilter_coverage,
    projected_l2,
    ranking_spearman,
    retained_overlap,
    scalar_risk,
    sequence_first,
    threshold_decision,
    token_age_statistics,
    validate_feature_schema,
    sha256_file,
)


def components():
    p0 = np.array([0.2, 0.3, 0.5])
    p1 = np.array([0.3, 0.4, 0.3])
    g0 = np.array([0.1, -0.2, 0.1])
    g1 = np.array([0.2, -0.1, -0.1])
    q0 = np.array([0.2, -0.1, 0.0])
    q1 = np.array([0.1, -0.2, 0.1])
    q2 = np.array([0.4, 0.0, -0.1])
    q3 = np.array([0.2, 0.1, -0.2])
    return p0, p1, g0, g1, q0, q1, q2, q3


def test_fresh_reused_component_isolation():
    p0, p1, g0, g1, q0, q1, q2, q3 = components()
    scores = component_swap_scores(g0, p0, q0, q1, g1, p1, q2, q3)
    assert scores["risk_all_old"] == pytest.approx(
        scalar_risk(g0, p0, 0.5 * (q0 + q1))
    )
    assert scores["risk_full_fresh"] == pytest.approx(
        scalar_risk(g1, p1, 0.5 * (q2 + q3))
    )


def test_component_g_swap_identity():
    p0, p1, g0, g1, q0, q1, q2, q3 = components()
    scores = component_swap_scores(g0, p0, q0, q1, g1, p1, q2, q3)
    assert scores["risk_update_g"] == pytest.approx(
        scalar_risk(g1, p0, 0.5 * (q0 + q1))
    )


def test_component_f_swap_identity():
    p0, p1, g0, g1, q0, q1, q2, q3 = components()
    scores = component_swap_scores(g0, p0, q0, q1, g1, p1, q2, q3)
    assert scores["risk_update_f"] == pytest.approx(
        scalar_risk(g0, p1, 0.5 * (q0 + q1))
    )


def test_component_path_swap_identity():
    p0, p1, g0, g1, q0, q1, q2, q3 = components()
    scores = component_swap_scores(g0, p0, q0, q1, g1, p1, q2, q3)
    assert scores["risk_update_path"] == pytest.approx(
        scalar_risk(g0, p0, 0.5 * (q2 + q3))
    )


def test_metric_only_candidate_consistency():
    ids = ["a", "b", "c"]
    fresh = pd.DataFrame({"candidate_id": ids, "score": [1, 2, 3]})
    reused = pd.DataFrame({"candidate_id": ids, "score": [3, 2, 1]})
    assert list(fresh.candidate_id) == list(reused.candidate_id)


def test_full_selection_candidate_distinction():
    current = {"a": (1, 2), "b": (2, 3)}
    stale = {"a": (1, 4), "b": (3, 4)}
    assert current != stale


def test_harmful_regret_label():
    event = decision_event([0, 1, 2], [0, 1, 2], [2, 0, 1], 0.1)
    assert event["harmful_stale"]
    assert event["reuse_normalized_regret"] == pytest.approx(0.5)


def test_exact_kl_leakage_prohibited():
    with pytest.raises(ValueError):
        validate_feature_schema(["exact_kl"])


def test_full_reference_observable_prohibited():
    with pytest.raises(ValueError):
        validate_feature_schema(["full_reference_state"])


def test_detector_feature_schema_freeze():
    assert validate_feature_schema(ZERO_COST_FEATURES) == ZERO_COST_FEATURES


def test_low_cost_schema_switch():
    assert validate_feature_schema(LOW_COST_FEATURES) == LOW_COST_FEATURES
    with pytest.raises(ValueError):
        validate_feature_schema(LOW_COST_FEATURES, allow_low_cost=False)


def test_random_projection_freeze():
    left = deterministic_projection(7, 3, 22)
    right = deterministic_projection(7, 3, 22)
    assert np.array_equal(left, right)


def test_random_projection_seed_changes():
    assert not np.array_equal(
        deterministic_projection(7, 3, 22),
        deterministic_projection(7, 3, 23),
    )


def test_projected_l2_zero_identity():
    assert projected_l2(
        np.ones(9), np.ones(9), output_dimension=3, seed=7
    ) == 0.0


def test_detector_threshold_freeze():
    values = np.array([0.1, 0.5, 0.9])
    assert threshold_decision(values, 0.5, "gt").tolist() == [
        False, False, True
    ]


def test_horizon_token_position_alignment():
    tau = 32
    horizons = [0, 1, 2, 4, 8, 16, 32]
    targets = [32, 33, 34, 36, 40, 48, 64]
    assert [tau + value for value in horizons] == targets


def test_no_future_token_leakage():
    with pytest.raises(ValueError):
        assert_no_future_columns(["query_norm", "future_token_id"])


def test_no_future_attention_leakage():
    with pytest.raises(ValueError):
        assert_no_future_columns(["future_attention_entropy"])


def test_data_ledger_exclusion():
    result = isolation_check(
        {"formal": ["a", "b"], "rep": ["c"]}, ["old"]
    )
    assert result["passed"]


def test_formal_replication_isolation():
    result = isolation_check(
        {"formal": ["a"], "rep": ["a"]}, []
    )
    assert not result["passed"]


def test_minimal_refresh_cost_accounting():
    assert forward_cost("full_fresh", candidate_count=8) == 32
    assert forward_cost(
        "single_midpoint", candidate_count=8, probed_count=3
    ) == 6


def test_candidate_top_k_coverage():
    result = prefilter_coverage([0, 1, 2, 3], [2, 0, 3, 4], 2)
    assert result["coverage"] == 1.0


def test_candidate_false_elimination():
    result = prefilter_coverage([0, 1, 2, 3], [3, 2, 0, 4], 2)
    assert result["false_elimination"] == 1.0


def test_selective_probing_integrity():
    with pytest.raises(ValueError):
        forward_cost("full_fresh", candidate_count=8, probed_count=9)


def test_physical_history_alignment():
    keys = ["sample_id", "candidate_id"]
    left = pd.DataFrame(
        {"sample_id": ["s"], "candidate_id": ["c"], "x": [1]}
    )
    right = pd.DataFrame(
        {"sample_id": ["s"], "candidate_id": ["c"], "y": [2]}
    )
    assert len(physical_alignment(left, right, keys)) == 1


def test_physical_history_mismatch_detected():
    keys = ["sample_id", "candidate_id"]
    left = pd.DataFrame(
        {"sample_id": ["s"], "candidate_id": ["c"], "x": [1]}
    )
    right = pd.DataFrame(
        {"sample_id": ["s"], "candidate_id": ["d"], "y": [2]}
    )
    with pytest.raises(ValueError):
        physical_alignment(left, right, keys)


def test_controlled_physical_ranking_comparison():
    assert ranking_spearman([0, 1, 2], [0, 1, 2]) == pytest.approx(1.0)


def test_sequence_first_aggregation():
    rows = pd.DataFrame(
        {
            "sample_id": ["s", "s"],
            "task": ["t", "t"],
            "stage": ["f", "f"],
            "metric": [0.0, 1.0],
        }
    )
    result = sequence_first(rows, ["metric"])
    assert result.metric.iloc[0] == pytest.approx(0.5)


def test_task_layer_history_stratification():
    rows = pd.DataFrame(
        {
            "task": ["a", "b"],
            "layer": [0, 1],
            "history_id": ["h0", "h1"],
            "x": [1, 2],
        }
    )
    assert len(rows.groupby(["task", "layer", "history_id"])) == 2


def test_all_vectors_finite():
    assert all_numeric_finite(pd.DataFrame({"x": [1.0, 2.0]}))
    assert not all_numeric_finite(pd.DataFrame({"x": [np.inf]}))


def test_atomic_writes(tmp_path):
    json_path = tmp_path / "a.json"
    frame_path = tmp_path / "a.parquet"
    atomic_json(json_path, {"x": 1})
    atomic_frame(frame_path, pd.DataFrame({"x": [1]}))
    assert json.loads(json_path.read_text()) == {"x": 1}
    assert pd.read_parquet(frame_path).x.iloc[0] == 1


def test_pairwise_accuracy():
    assert pairwise_accuracy([0, 1, 2], [0, 1, 2]) == 1.0


def test_normalized_regret():
    assert normalized_regret([0, 1, 2], [2, 0, 1]) == 0.5


def test_rank_disagreement():
    assert mean_rank_disagreement([[0, 1], [0, 1]]) == 0.0


def test_cache_overlap():
    assert retained_overlap([1, 2], [2, 3]) == pytest.approx(1 / 3)


def test_token_age_statistics():
    result = token_age_statistics([1, 3], 5)
    assert result["token_age_mean"] == 3.0
    assert result["token_age_std"] == 1.0


def test_epsilon_mechanical_selection():
    result = choose_harmful_epsilon(
        [0, 0.015, 0.03, 0.2], [0.01, 0.02, 0.05, 0.1]
    )
    assert result["selected"] == 0.02


def test_detector_metrics_by_task():
    result = detector_metrics(
        [True, True, False],
        [True, False, False],
        [0.1, 0.2, 0.0],
        task=["a", "b", "a"],
    )
    assert result["harmful_recall"] == 0.5
    assert result["task_recall"]["a"] == 1.0
    assert result["task_recall"]["b"] == 0.0


def test_config_target_horizon_alignment():
    config = yaml.safe_load(
        (
            ROOT
            / "experiments/p3_decision_validity/p3_config.yaml"
        ).read_text()
    )
    trajectory = config["trajectory"]
    assert [
        trajectory["calibration_anchor"] + value
        for value in trajectory["horizons"]
    ] == trajectory["target_anchors"]


def test_p0_p1_p2_recovery_manifests_unchanged():
    config = yaml.safe_load(
        (
            ROOT
            / "experiments/p3_decision_validity/p3_config.yaml"
        ).read_text()
    )
    paths = {
        "p0": ROOT
        / "experiments/p0_v2_fixed_boundary/P0_V2_MANIFEST.yaml",
        "p1": ROOT
        / "experiments/p1_state_conditioned/"
        "P1_STATE_CONDITIONED_MANIFEST.yaml",
        "p2": ROOT
        / "experiments/p2_state_local_risk/P2_STATE_LOCAL_MANIFEST.yaml",
        "p2_recovery": ROOT
        / "experiments/p2_recovery/P2_RECOVERY_MANIFEST.yaml",
    }
    assert all(
        sha256_file(path)
        == config["source"][f"{name}_manifest_sha256"]
        for name, path in paths.items()
    )


def test_formula_render_audit():
    path = (
        ROOT
        / "experiments/p3_decision_validity/results/"
        "formula_render_audit.json"
    )
    if not path.exists():
        pytest.skip("formula audit is a finalization-stage artifact")
    audit = json.loads(path.read_text())
    assert audit["passed"]
    assert audit["total_warning_count"] == 0
    assert audit["total_raw_math_leftover_count"] == 0


def test_manifest_integrity():
    path = (
        ROOT
        / "experiments/p3_decision_validity/results/"
        "checksum_verification.json"
    )
    if not path.exists():
        pytest.skip("manifest is a finalization-stage artifact")
    verification = json.loads(path.read_text())
    assert verification["passed"]
    assert all(verification["prior_manifest_checks"].values())
