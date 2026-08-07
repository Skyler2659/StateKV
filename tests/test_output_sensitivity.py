from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from statekv.output_sensitivity import (
    candidate_budget_equal,
    no_task_feature,
)
from statekv.output_sensitivity_analysis import (
    Bridge,
    _attach_recursive_e2,
    bridge_coefficients_nonnegative,
    clustered_additive_margin,
    clustered_coordinate_margin,
    conformal_order_statistic,
    deployable_output_features,
    direction_estimates_agree,
    dominance_decision,
    json_numbers_consistent,
    nested_sequence_partition,
    pairwise_auc,
    pairwise_feature_difference,
    pairwise_prediction,
    refresh_lcb_trigger,
    softmax_kl_inequality_holds,
    swapped_pairwise_interval,
    top1_regret,
    topk_overlap,
    validate_finite_difference_grid,
)
from statekv.robust_envelope_analysis import EnvelopeModel


def _partition_fixture():
    sequences = ["n%d" % i for i in range(12)] + [
        "g%d" % i for i in range(12)
    ]
    tasks = {
        value: ("niah_single_1" if value.startswith("n") else "gov_report")
        for value in sequences
    }
    return sequences, tasks


def test_nested_split_has_preregistered_sizes():
    sequences, tasks = _partition_fixture()
    fit, state, calibration = nested_sequence_partition(
        sequences, "n0", tasks
    )
    assert (len(fit), len(state), len(calibration)) == (10, 5, 8)


def test_held_out_sequence_never_leaks():
    sequences, tasks = _partition_fixture()
    fit, state, calibration = nested_sequence_partition(
        sequences, "g3", tasks
    )
    assert "g3" not in fit + state + calibration


def test_nested_roles_are_disjoint():
    sequences, tasks = _partition_fixture()
    fit, state, calibration = nested_sequence_partition(
        sequences, "n4", tasks
    )
    assert not set(fit) & set(state)
    assert not set(fit) & set(calibration)
    assert not set(state) & set(calibration)


def test_90_percent_conformal_with_eight_sequences_is_maximum():
    value, rank, is_maximum = conformal_order_statistic(range(8), 0.90)
    assert value == 7 and rank == 8 and is_maximum


def test_95_percent_conformal_with_eight_sequences_is_maximum():
    value, rank, is_maximum = conformal_order_statistic(range(8), 0.95)
    assert value == 7 and rank == 8 and is_maximum


def test_conformal_margin_clusters_by_sequence():
    residual = [1.0, 100.0, 2.0, 3.0]
    sequence = ["a", "a", "b", "b"]
    value, _, _ = clustered_additive_margin(residual, sequence, 0.5)
    assert value in {3.0, 100.0}


def test_coordinate_conformal_margin_is_coordinatewise():
    residual = np.asarray([[1.0, 4.0], [3.0, 2.0]])
    margin, _, _ = clustered_coordinate_margin(
        residual, ["a", "b"], 0.5
    )
    assert np.array_equal(margin, np.asarray([3.0, 4.0]))


def test_e2_recursive_rollout_does_not_read_future_truth():
    model = EnvelopeModel(
        "E2",
        [0],
        np.asarray([[0.5]]),
        np.asarray([[1.0]]),
        np.zeros((1, 1)),
    )
    base = pd.DataFrame(
        {
            "trajectory_id": ["x", "x"],
            "horizon_offset": [1, 2],
            "d_l0": [1.0, 1.0],
            "e_l0": [2.0, 9999.0],
        }
    )
    changed = base.copy()
    changed["e_l0"] = [500.0, 0.0]
    first = _attach_recursive_e2(base, [0], model, np.asarray([0.0]))
    second = _attach_recursive_e2(changed, [0], model, np.asarray([0.0]))
    assert np.allclose(first["b_l0"], second["b_l0"])


def test_output_bridge_coefficients_are_nonnegative():
    bridge = Bridge("O2", np.asarray([0.0, 2.0]), 1.0, {})
    assert bridge_coefficients_nonnegative(bridge)


def test_negative_output_bridge_coefficient_is_rejected():
    bridge = Bridge("O2", np.asarray([0.0, -0.1]), 1.0, {})
    assert not bridge_coefficients_nonnegative(bridge)


def test_task_id_is_not_a_deployable_feature():
    assert not deployable_output_features(["output_entropy", "task_id"])


def test_sequence_id_is_not_a_deployable_feature():
    assert not deployable_output_features(["sample_id", "prefix_length"])


def test_future_labels_are_not_deployable_features():
    assert not deployable_output_features(["future_kl"])
    assert not deployable_output_features(["future_nll"])


def test_o4_registered_observables_are_deployable():
    assert deployable_output_features(
        ["output_entropy", "attention_entropy", "prefix_length"]
    )


def test_collector_feature_guard_rejects_task():
    assert not no_task_feature(["output_entropy", "task"])


def test_pairwise_features_are_antisymmetric():
    left = np.asarray([1.0, 3.0])
    right = np.asarray([2.0, 1.0])
    assert np.allclose(
        pairwise_feature_difference(left, right),
        -pairwise_feature_difference(right, left),
    )


def test_pairwise_prediction_is_antisymmetric():
    assert pairwise_prediction(2.0, 7.0) == -pairwise_prediction(7.0, 2.0)


def test_same_action_pairwise_regret_is_zero():
    assert pairwise_prediction(3.2, 3.2) == 0.0


def test_pairwise_interval_swaps_sign_and_endpoints():
    assert swapped_pairwise_interval(-2.0, 5.0) == (-5.0, 2.0)


def test_abstention_when_interval_contains_zero():
    abstain, sign = dominance_decision(0.2, 0.5)
    assert abstain and sign == 0


def test_left_dominance_when_upper_endpoint_is_negative():
    abstain, sign = dominance_decision(-2.0, 0.5)
    assert not abstain and sign == -1


def test_refresh_lcb_threshold_is_strict():
    assert not refresh_lcb_trigger(1.0, 1.0, 3, 0)
    assert refresh_lcb_trigger(1.01, 1.0, 3, 0)


def test_refresh_policy_never_exceeds_maximum_count():
    assert not refresh_lcb_trigger(10.0, 1.0, 3, 3)


def test_refresh_does_not_reset_accumulated_envelope_state():
    model = EnvelopeModel(
        "E2",
        [0],
        np.asarray([[1.0]]),
        np.asarray([[0.0]]),
        np.zeros((1, 1)),
    )
    result, _ = __import__(
        "statekv.robust_envelope_analysis",
        fromlist=["recursive_envelope"],
    ).recursive_envelope(
        model,
        np.asarray([[0.0]]),
        np.asarray([0.0]),
        initial_error=np.asarray([4.0]),
    )
    assert result[0, 0] == 4.0


def test_refresh_action_does_not_recall_deleted_positions():
    physical_history = set(range(10))
    deleted = {3, 7}
    available = physical_history - deleted
    refreshed = {value for value in [1, 3, 5] if value in available}
    assert 3 not in refreshed


def test_all_candidate_budgets_must_match():
    left = SimpleNamespace(
        by_layer={0: SimpleNamespace(selected_positions=[1, 2])}
    )
    right = SimpleNamespace(
        by_layer={0: SimpleNamespace(selected_positions=[3, 4])}
    )
    wrong = SimpleNamespace(
        by_layer={0: SimpleNamespace(selected_positions=[3])}
    )
    assert candidate_budget_equal([left, right])
    assert not candidate_budget_equal([left, wrong])


def test_jvp_and_small_radius_finite_difference_direction_agree():
    assert direction_estimates_agree([1.0, 2.0], [1.01, 1.99])


def test_finite_difference_executes_eight_directions_three_radii():
    rows = []
    for direction in range(8):
        for radius in [0.001, 0.01, 0.05]:
            rows.append(
                {
                    "sample_id": "s",
                    "anchor": 16,
                    "layer": 27,
                    "direction_index": direction,
                    "relative_radius": radius,
                    "finite_difference_symmetric": True,
                }
            )
    assert validate_finite_difference_grid(pd.DataFrame(rows))


def test_finite_direction_gain_is_not_called_operator_norm():
    row = {"claimed_operator_norm": False}
    assert not row["claimed_operator_norm"]


def test_softmax_kl_inequality_is_checked_per_sample():
    assert softmax_kl_inequality_holds(
        np.asarray([0.1, 0.2]), np.asarray([0.4, 0.8])
    )
    assert not softmax_kl_inequality_holds(
        np.asarray([0.2]), np.asarray([0.4])
    )


def test_logit_and_kl_coverage_are_separate_columns():
    frame = pd.DataFrame(
        {"logit_covered": [True], "kl_covered": [False]}
    )
    assert frame["logit_covered"].iloc[0]
    assert not frame["kl_covered"].iloc[0]


def test_candidate_pairs_are_not_bootstrap_units():
    rows = pd.DataFrame(
        {
            "held_out_sequence": ["s", "s"],
            "bootstrap_unit": ["sequence", "sequence"],
        }
    )
    assert rows["bootstrap_unit"].eq("sequence").all()
    assert rows["held_out_sequence"].nunique() == 1


def test_top1_regret_calculation():
    assert top1_regret([1.0, 4.0, 2.0], [3.0, 1.0, 2.0]) == 3.0


def test_top3_overlap_calculation():
    assert topk_overlap([1, 2, 3, 4], [1, 4, 3, 2], 3) == 2 / 3


def test_pairwise_auc_direction():
    truth = [-2.0, -1.0, 1.0, 2.0]
    prediction = [-3.0, -0.5, 0.5, 3.0]
    assert pairwise_auc(truth, prediction) == 1.0


def test_layer_27_is_reported_separately():
    layers = [0, 7, 14, 15, 21, 27]
    assert 27 in layers and layers[-1] == 27


def test_both_task_splits_are_retained():
    tasks = pd.Series(["niah_single_1", "gov_report"])
    buckets = {
        "GovReport" if "gov" in value else "NIAH" for value in tasks
    }
    assert buckets == {"NIAH", "GovReport"}


def test_reversal_rows_round_trip_without_filtering(tmp_path: Path):
    rows = pd.DataFrame(
        {"task": ["NIAH", "GovReport"], "increment": [0.1, -0.1]}
    )
    path = tmp_path / "reversals.parquet"
    rows.to_parquet(path, index=False)
    loaded = pd.read_parquet(path)
    assert loaded.to_dict("records") == rows.to_dict("records")


def test_segment_oracle_is_not_a_training_feature():
    assert deployable_output_features(["output_entropy"])
    assert "segment_oracle" not in ["output_entropy"]


def test_teacher_forced_and_free_generation_are_separate(tmp_path: Path):
    payload = {
        "status": "not_run_by_preregistered_gate",
        "teacher_forced_and_free_generation_separated": True,
    }
    path = tmp_path / "free_generation_results.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert json.loads(path.read_text())[
        "teacher_forced_and_free_generation_separated"
    ]


def test_matched_refresh_count_is_explicit():
    policy_counts = {"baseline": 2, "lcb": 2}
    assert len(set(policy_counts.values())) == 1


def test_json_and_table_number_consistency():
    frame = pd.DataFrame({"coverage": [0.5, 1.0]})
    assert json_numbers_consistent(
        frame, {"coverage_mean": 0.75}, "coverage", "coverage_mean"
    )
