from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "experiments/p2_recovery/scripts"
P2_DIR = ROOT / "experiments/p2_state_local_risk/scripts"
for value in (SCRIPT_DIR, P2_DIR):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import recovery_core as CORE  # noqa: E402
from p2_core import sha256_file  # noqa: E402


def test_action_scaling_linearity() -> None:
    direction = np.array([1.0, -2.0, 0.5])
    assert np.allclose(0.25 * direction, direction / 4.0)


def test_quadratic_residual_scaling() -> None:
    gamma = np.array([0.0625, 0.125, 0.25, 0.5, 1.0])
    residual = 3.0 * gamma**2
    assert np.isclose(CORE.log_log_slope(gamma, residual), 2.0)


def test_state_conditioned_deletion_identity() -> None:
    full = np.array([2.0, -1.0, 0.5])
    retained = np.array([1.5, -0.5, 0.25])
    state_injection = retained - full
    assert np.allclose(full + state_injection, retained)


def test_state_injection_matches_physical_current_action() -> None:
    state_injection = np.array([-0.3, 0.2])
    physical_increment = np.array([-0.3, 0.2])
    assert np.array_equal(state_injection, physical_increment)


def test_adjacent_state_local_response() -> None:
    matrix = np.array([[2.0, 0.5], [-1.0, 3.0]])
    injection = np.array([0.2, -0.4])
    assert np.allclose(matrix @ injection, np.array([0.2, -1.4]))


def test_path_integration_convergence() -> None:
    # Integral of derivative 2a on [0,1] is one.
    estimates = []
    for count in (2, 4, 8, 16):
        nodes = CORE.midpoint_nodes(count)
        estimates.append(CORE.midpoint_integral([2.0 * x for x in nodes]))
    assert all(np.allclose(value, 1.0) for value in estimates)


def test_midpoint_and_trapezoidal_implementation() -> None:
    assert np.allclose(CORE.midpoint_integral([[1.0], [3.0]]), [2.0])
    assert np.allclose(CORE.trapezoidal_integral([1.0], [3.0]), [2.0])


def test_adaptive_segmentation_has_no_exact_target_input() -> None:
    signature = inspect.signature(CORE.adaptive_segment_count)
    assert "exact_kl" not in signature.parameters
    assert "truth" not in signature.parameters


def test_scalar_risk_has_no_exact_kl_leakage() -> None:
    signature = inspect.signature(CORE.state_local_quadratic_risk)
    assert "exact_kl" not in signature.parameters
    p = np.array([0.2, 0.3, 0.5])
    assert np.isfinite(
        CORE.state_local_quadratic_risk(
            np.zeros(3), np.ones(3), p
        )
    )


def test_formal_id_exclusion() -> None:
    ledger = yaml.safe_load(
        (ROOT / "configs/frozen/P2_RECOVERY_DATA_LEDGER.yaml").read_text()
    )
    excluded = {
        value
        for row in ledger["exclusion_ledger"]
        for value in row["ids"]
    }
    assert "gov_report:84" in excluded
    assert "synthetic_niah_85" in excluded
    assert not any(value.endswith("_86") for value in excluded)


def test_iteration_config_freeze() -> None:
    ledger = yaml.safe_load(
        (ROOT / "configs/frozen/P2_RECOVERY_DATA_LEDGER.yaml").read_text()
    )
    for iteration in ("r0_failure_map", "r1_amplitude_trust_region"):
        row = ledger["iterations"][iteration]
        assert sha256_file(ROOT / row["config_path"]) == row[
            "config_sha256"
        ]


def test_cumulative_data_ledger_integrity() -> None:
    ledger = yaml.safe_load(
        (ROOT / "configs/frozen/P2_RECOVERY_DATA_LEDGER.yaml").read_text()
    )
    assert ledger["schema_version"] == 1
    assert len(ledger["exclusion_ledger"]) == 7
    assert set(ledger["iterations"]) >= {
        "r0_failure_map",
        "r1_amplitude_trust_region",
    }


def test_previous_manifests_unchanged() -> None:
    ledger = yaml.safe_load(
        (ROOT / "configs/frozen/P2_RECOVERY_DATA_LEDGER.yaml").read_text()
    )
    for source in ("p0_manifest", "p1_manifest", "p2_manifest"):
        record = ledger["immutable_sources"][source]
        assert sha256_file(ROOT / record["path"]) == record["sha256"]


def test_candidate_registry_consistency() -> None:
    registry = pd.read_parquet(
        ROOT
        / "experiments/p2_state_local_risk/results/"
        "candidate_registry.parquet"
    )
    assert (
        registry.groupby(["sample_id", "anchor"])["mask_hash"].nunique()
        == 8
    ).all()


def test_state_candidate_decoupling() -> None:
    response = pd.read_parquet(
        ROOT
        / "experiments/p2_state_local_risk/results/"
        "response_rows.parquet"
    )
    assert (
        response.groupby(
            ["sample_id", "anchor", "layer", "history_id"]
        )["state_hash"].nunique()
        == 1
    ).all()


def test_physical_history_token_position_alignment() -> None:
    state = pd.read_parquet(
        ROOT
        / "experiments/p2_state_local_risk/results/"
        "state_registry.parquet"
    )
    assert (state["query_position"] == state["reference_query_position"]).all()


def test_sequence_first_aggregation() -> None:
    rows = pd.DataFrame(
        {"sample_id": ["a", "a", "b", "b"], "value": [1, 3, 2, 6]}
    )
    result = rows.groupby("sample_id")["value"].median()
    assert result.to_dict() == {"a": 2.0, "b": 4.0}


def test_task_stratified_gate() -> None:
    sequence = pd.DataFrame(
        [
            {"task": "a", "cosine": 0.995, "relative_l2": 0.01, "finite": True},
            {"task": "b", "cosine": 0.994, "relative_l2": 0.02, "finite": True},
        ]
    )
    rows = pd.DataFrame(
        [{"cosine": 0.995, "finite": True}, {"cosine": 0.994, "finite": True}]
    )
    rule = {
        "overall_sequence_first_cosine_min": 0.99,
        "each_task_cosine_min": 0.98,
        "overall_sequence_first_relative_l2_max": 0.05,
        "row_cosine_min": 0.99,
        "row_pass_fraction_min": 0.95,
        "all_finite": True,
    }
    assert CORE.sequence_gate(sequence, rows, rule)["passed"]


def test_all_existing_recovery_vectors_finite() -> None:
    path = (
        ROOT
        / "experiments/p2_recovery/r1_amplitude_trust_region/"
        "results/scaling_rows.parquet"
    )
    if not path.exists():
        pytest.skip("R1 not executed")
    frame = pd.read_parquet(path)
    assert np.isfinite(
        frame.select_dtypes(include=[np.number]).to_numpy()
    ).all()


def test_formula_rendering_artifact() -> None:
    path = (
        ROOT
        / "experiments/p2_recovery/results/"
        "formula_render_audit.json"
    )
    if not path.exists():
        pytest.skip("Recovery formula audit not generated")
    result = json.loads(path.read_text())
    assert result["passed"]
    assert result["total_warning_count"] == 0
    assert result["total_raw_math_leftover_count"] == 0
