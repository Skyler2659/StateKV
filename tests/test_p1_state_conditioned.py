from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "experiments/p1_state_conditioned/scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import p1_core as CORE  # noqa: E402


def protocol() -> dict:
    return yaml.safe_load(
        (ROOT / "configs/frozen/p1_state_conditioned_config.yaml").read_text(
            encoding="utf-8"
        )
    )


def test_fisher_inner_is_symmetric() -> None:
    p = np.array([0.2, 0.3, 0.5])
    a = np.array([-0.2, 0.1, 0.8])
    b = np.array([1.1, -0.4, 0.3])
    assert np.isclose(
        CORE.fisher_inner(p, a, b),
        CORE.fisher_inner(p, b, a),
        atol=1.0e-12,
    )


def test_fisher_inner_is_common_shift_invariant() -> None:
    p = np.array([0.2, 0.3, 0.5])
    a = np.array([-0.2, 0.1, 0.8])
    b = np.array([1.1, -0.4, 0.3])
    assert np.isclose(
        CORE.fisher_inner(p, a, b),
        CORE.fisher_inner(p, a + 100.0, b - 31.0),
        atol=1.0e-12,
    )


def test_cross_term_equals_polarization() -> None:
    p = np.array([0.2, 0.3, 0.5])
    state = np.array([-0.2, 0.1, 0.8])
    action = np.array([1.1, -0.4, 0.3])
    assert np.isclose(
        CORE.fisher_inner(p, state, action),
        CORE.polarization_cross(p, state, action),
        atol=1.0e-12,
    )


def test_total_score_decomposes_exactly() -> None:
    p = np.array([0.2, 0.3, 0.5])
    values = CORE.state_action_scores(
        p,
        np.array([-0.2, 0.1, 0.8]),
        np.array([1.1, -0.4, 0.3]),
    )
    assert abs(values["decomposition_error"]) < 1.0e-12


def test_h0_reduces_state_score_to_action_score() -> None:
    p = np.array([0.2, 0.3, 0.5])
    action = np.array([1.1, -0.4, 0.3])
    values = CORE.state_action_scores(p, np.zeros(3), action)
    assert np.isclose(values["cross_fisher_score"], 0.0)
    assert np.isclose(
        values["state_fisher_score"], values["action_fisher_score"]
    )


def test_state_and_total_scores_have_identical_candidate_ranking() -> None:
    p = np.array([0.2, 0.3, 0.5])
    state = np.array([-0.2, 0.1, 0.8])
    actions = [
        np.array([1.1, -0.4, 0.3]),
        np.array([0.1, 0.4, -0.8]),
        np.array([-0.3, 0.2, 0.7]),
    ]
    rows = [CORE.state_action_scores(p, state, action) for action in actions]
    state_order = np.argsort([row["state_fisher_score"] for row in rows])
    total_order = np.argsort([row["total_fisher_score"] for row in rows])
    assert np.array_equal(state_order, total_order)


def test_required_anchors_cover_starts_refreshes_and_targets() -> None:
    anchors = CORE.required_reference_anchors(
        [48, 64], protocol()["history_conditions"]
    )
    assert anchors == (16, 24, 32, 40, 48, 56, 64)


def test_history_key_changes_with_condition_or_vector() -> None:
    zero = np.zeros(4, dtype=np.float32)
    one = np.ones(4, dtype=np.float32)
    first = CORE.history_state_key("s", 48, 15, "H1", zero)
    assert first != CORE.history_state_key("s", 48, 15, "H2", zero)
    assert first != CORE.history_state_key("s", 48, 15, "H1", one)


def test_euclidean_cosine_identity() -> None:
    value = np.array([1.0, -2.0, 3.0])
    assert np.isclose(CORE.euclidean_cosine(value, value), 1.0)


def test_fisher_cosine_identity() -> None:
    p = np.array([0.2, 0.3, 0.5])
    value = np.array([1.0, -2.0, 3.0])
    assert np.isclose(CORE.fisher_cosine(p, value, value), 1.0)


def test_fd_radius_rule_uses_minimum_error_and_larger_tie() -> None:
    rows = pd.DataFrame(
        [
            {
                "epsilon_relative": radius,
                "finite": True,
                "fd_norm": 1.0,
                "cosine": 1.0,
                "relative_l2": error,
                "symmetric_norm_ratio": 1.0,
            }
            for radius, error in (
                (1.0e-3, 0.01),
                (3.0e-4, 0.01),
                (1.0e-4, 0.02),
            )
        ]
    )
    selected, summary = CORE.select_fd_radius(
        rows, protocol()["numeric"]["fd_selection_rule"]
    )
    assert selected == 1.0e-3
    assert summary["passes"].all()


def test_split_isolation_passes_frozen_protocol() -> None:
    audit = CORE.validate_split_isolation(protocol())
    assert all(audit["checks"].values())


def test_formal_row_count_is_preregistered_minimum() -> None:
    cfg = protocol()
    evaluation = cfg["data"]["evaluation"]
    expected = (
        4
        * len(evaluation["target_anchors"])
        * len(evaluation["layers"])
        * len(
            [
                key
                for key in cfg["history_conditions"]
                if key.startswith("H") and key[1:].isdigit()
            ]
        )
        * evaluation["candidates_per_unit"]
    )
    assert expected == 768
    assert expected == evaluation["expected_candidate_history_rows"]


def test_candidates_are_history_independent_and_no_oracle() -> None:
    cfg = protocol()
    assert cfg["candidates"]["shared_across_histories"]
    assert "future_attention_oracle" not in cfg["candidates"]["sources"]
    assert len(cfg["candidates"]["sources"]) == 8


def test_scope_forbids_policy_and_multistep_risk() -> None:
    prohibited = set(protocol()["scope"]["prohibited"])
    assert "refresh_policy_evaluation" in prohibited
    assert "multi_step_horizon_risk" in prohibited
    assert "free_generation" in prohibited
    assert "joint_current_multilayer_mask" in prohibited


def test_h3_is_matched_reset_control_not_policy() -> None:
    cfg = protocol()
    assert (
        cfg["history_conditions"]["periodic_refresh_semantics"]
        == "matched_full_reference_anchor_reset"
    )
    assert cfg["history_conditions"]["H3"]["expected_refresh_count"] == 3


def formal_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    result = ROOT / "experiments/p1_state_conditioned/results"
    return (
        pd.read_parquet(result / "response_rows.parquet"),
        pd.read_parquet(result / "state_registry.parquet"),
        pd.read_parquet(result / "unit_audit.parquet"),
    )


def test_formal_h0_is_zero_and_scores_are_identical() -> None:
    response, _state, _audit = formal_tables()
    h0 = response[response["history_id"] == "H0"]
    assert h0["state_norm"].max() == 0.0
    assert h0["cross_fisher_score"].abs().max() <= 1.0e-12
    assert np.allclose(
        h0["state_fisher_score"], h0["action_fisher_score"]
    )


def test_formal_state_hash_is_shared_across_candidates() -> None:
    response, _state, _audit = formal_tables()
    sizes = response.groupby(
        ["sample_id", "anchor", "layer", "history_id"]
    )["state_hash"].nunique()
    assert sizes.eq(1).all()


def test_formal_candidate_sets_are_equal_across_histories() -> None:
    response, _state, _audit = formal_tables()
    sets = (
        response.groupby(
            ["sample_id", "anchor", "layer", "history_id"]
        )["mask_hash"]
        .apply(lambda values: tuple(sorted(set(values))))
        .reset_index()
    )
    assert (
        sets.groupby(["sample_id", "anchor", "layer"])[
            "mask_hash"
        ]
        .nunique()
        .eq(1)
        .all()
    )


def test_formal_zero_map_reproduces_baseline() -> None:
    _response, _state, audit = formal_tables()
    assert audit["boundary_map_baseline_relative_l2"].eq(0.0).all()
    assert np.allclose(audit["boundary_map_baseline_cosine"], 1.0)


def test_formal_combination_adds_state_and_action_once() -> None:
    response, _state, _audit = formal_tables()
    expected = np.sqrt(
        response["state_norm"] ** 2
        + response["action_r_norm"] ** 2
        + 2.0
        * response["state_norm"]
        * response["action_r_norm"]
        * response["state_action_euclidean_cosine"]
    )
    assert np.allclose(
        response["combined_boundary_norm"],
        expected,
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_formal_metrics_are_finite_and_negative_cross_is_retained() -> None:
    response, state, _audit = formal_tables()
    numeric = response.select_dtypes(include=[np.number])
    assert np.isfinite(numeric.to_numpy()).all()
    assert np.isfinite(
        state.select_dtypes(include=[np.number]).to_numpy()
    ).all()
    assert (response["cross_fisher_score"] < 0.0).any()


def test_formal_state_and_total_ranking_identity() -> None:
    response, _state, _audit = formal_tables()
    grouping = [
        "sample_id",
        "anchor",
        "layer",
        "history_id",
    ]
    for _key, group in response.groupby(grouping):
        assert np.array_equal(
            np.argsort(group["state_fisher_score"].to_numpy()),
            np.argsort(group["total_fisher_score"].to_numpy()),
        )


def test_formal_protocol_identity_and_sequence_first_cardinality() -> None:
    result = ROOT / "experiments/p1_state_conditioned/results"
    metadata = json.loads(
        (result / "evaluation_metadata.json").read_text()
    )
    config = protocol()
    assert config["experiment"] == "p1_state_conditioned_fixed_boundary_risk_closure"
    assert metadata["completed"]
    assert metadata["stage"] == "formal_evaluation"
    assert metadata["row_counts"]["response_rows"] == config["data"][
        "evaluation"
    ]["expected_candidate_history_rows"]
    sequence = pd.read_parquet(
        result / "sequence_first_ranking.parquet"
    )
    assert sequence["sample_id"].nunique() == 4
    assert len(sequence) == 4 * 5
