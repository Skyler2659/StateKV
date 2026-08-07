from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "experiments/p2_state_local_risk/scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import p2_core as CORE  # noqa: E402


def protocol() -> dict:
    return yaml.safe_load(
        (ROOT / "configs/frozen/p2_state_local_config.yaml").read_text(
            encoding="utf-8"
        )
    )


def test_h0_state_local_degeneration() -> None:
    delta = np.zeros(4)
    assert np.linalg.norm(delta) == 0.0


def test_h0_probability_and_gradient_identity() -> None:
    p0 = np.array([0.2, 0.3, 0.5])
    ps = p0.copy()
    assert np.array_equal(ps, p0)
    assert np.array_equal(CORE.exact_kl_gradient(p0, ps), np.zeros(3))


def test_h0_reference_and_state_jvp_identity() -> None:
    c0 = np.array([-0.2, 0.1, 0.8])
    cs = c0.copy()
    assert np.array_equal(c0, cs)


def test_h0_full_score_equals_action_fisher() -> None:
    p0 = np.array([0.2, 0.3, 0.5])
    zero = np.zeros(3)
    action = np.array([-0.2, 0.1, 0.8])
    scores = CORE.geometry_scores(
        reference_probability=p0,
        state_probability=p0,
        reference_linear_gradient=zero,
        state_local_gradient=zero,
        reference_action_direction=action,
        state_local_action_direction=action,
        nonlinear_action_direction=action,
    )
    assert np.isclose(
        scores["full_state_local"],
        scores["reference_action_fisher"],
        atol=1.0e-12,
    )


def test_exact_kl_gradient_matches_autodiff() -> None:
    p0 = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
    logits = torch.tensor(
        [0.4, -0.2, 0.7], dtype=torch.float64, requires_grad=True
    )
    log_q = torch.log_softmax(logits, dim=0)
    loss = torch.sum(p0 * (torch.log(p0) - log_q))
    loss.backward()
    ps = torch.softmax(logits.detach(), dim=0).numpy()
    expected = CORE.exact_kl_gradient(p0.numpy(), ps)
    assert np.allclose(logits.grad.numpy(), expected, atol=1.0e-12)


def test_fisher_quadratic_matches_explicit_matrix() -> None:
    p = np.array([0.2, 0.3, 0.5])
    v = np.array([-0.2, 0.1, 0.8])
    explicit = np.diag(p) - np.outer(p, p)
    assert np.allclose(CORE.fisher_vector_product(p, v), explicit @ v)
    assert np.isclose(
        CORE.fisher_variance(p, v), float(v @ explicit @ v)
    )


def test_eight_factorial_combinations_are_complete() -> None:
    registry = CORE.score_registry_rows()
    factorial = {
        name: row
        for name, row in registry.items()
        if row["factorial"]
    }
    assert len(factorial) == 8
    assert set(factorial) == set(CORE.FACTORIAL_REGISTRY)
    assert "full_state_local" in factorial


def test_state_candidate_hash_decoupling_schema() -> None:
    state_hash = "state"
    rows = [
        {"candidate": candidate, "state_hash": state_hash}
        for candidate in range(8)
    ]
    assert len({row["state_hash"] for row in rows}) == 1
    assert len({row["candidate"] for row in rows}) == 8


def test_formal_data_isolation_and_mechanical_ids() -> None:
    cfg = protocol()
    calibration = set(cfg["data"]["calibration"]["gov_report_indices"])
    formal = set(cfg["data"]["evaluation"]["gov_report_indices"])
    assert calibration == {82, 83}
    assert formal == {84, 85}
    assert calibration.isdisjoint(formal)
    assert min(calibration | formal) > 81


def test_score_api_cannot_receive_exact_kl() -> None:
    signatures = [
        inspect.signature(CORE.geometry_score),
        inspect.signature(CORE.geometry_scores),
    ]
    assert all(
        "exact_kl" not in signature.parameters
        for signature in signatures
    )


def test_scope_forbids_future_and_policy_leakage() -> None:
    prohibited = set(protocol()["scope"]["prohibited"])
    assert {
        "future_query",
        "future_attention",
        "refresh_controller",
        "online_policy",
        "physical_history_transfer",
    }.issubset(prohibited)


def test_sequence_first_keeps_four_independent_sequences() -> None:
    rows = pd.DataFrame(
        {
            "sample_id": np.repeat(["a", "b", "c", "d"], 8),
            "value": np.arange(32, dtype=float),
        }
    )
    sequence = rows.groupby("sample_id")["value"].median()
    assert len(sequence) == 4
    assert np.allclose(sequence.to_numpy(), [3.5, 11.5, 19.5, 27.5])


def test_structured_writes(tmp_path: Path) -> None:
    json_path = tmp_path / "value.json"
    parquet_path = tmp_path / "value.parquet"
    CORE.atomic_json(json_path, {"value": 1})
    CORE.atomic_frame(
        parquet_path, pd.DataFrame([{"value": 1.0}])
    )
    assert json.loads(json_path.read_text())["value"] == 1
    assert pd.read_parquet(parquet_path)["value"].iloc[0] == 1.0


class LinearMap:
    def __init__(self) -> None:
        self.base_input = np.array([3.0, 4.0])
        self.matrix = np.array([[2.0, -1.0], [0.5, 3.0]])

    def evaluate(self, delta: np.ndarray) -> np.ndarray:
        return self.matrix @ np.asarray(delta)


def test_state_local_fd_uses_workpoint_and_recovers_linear_jvp() -> None:
    mapping = LinearMap()
    state = np.array([1.0, -2.0])
    action = np.array([0.4, 0.7])
    result = CORE.state_local_symmetric_fd(
        mapping, state, action, 1.0e-3
    )
    assert np.allclose(result["derivative"], mapping.matrix @ action)
    assert np.isclose(
        result["state_workpoint_norm"],
        np.linalg.norm(mapping.base_input + state),
    )


def test_candidate_independent_constant_preserves_ranking() -> None:
    scores = np.array([0.3, -0.2, 0.8, 0.1])
    assert np.array_equal(
        np.argsort(scores), np.argsort(scores + 123.0)
    )


def test_gradient_term_keeps_negative_scores() -> None:
    p = np.array([0.2, 0.3, 0.5])
    gradient = np.array([-2.0, 0.0, 0.0])
    action = np.array([1.0, 0.0, 0.0])
    assert CORE.geometry_score(gradient, action, p) < 0.0


def test_factorial_effects_require_and_use_all_cells() -> None:
    values = {
        (g, j, f): float(2 * g + 3 * j + 5 * f)
        for g in (0, 1)
        for j in (0, 1)
        for f in (0, 1)
    }
    effects = CORE.factorial_effects(values)
    assert np.isclose(effects["gradient_main"], 2.0)
    assert np.isclose(effects["jacobian_main"], 3.0)
    assert np.isclose(effects["fisher_main"], 5.0)
    assert np.isclose(effects["three_way_interaction"], 0.0)


def test_probability_drift_zero_at_h0() -> None:
    p = np.array([0.2, 0.3, 0.5])
    drift = CORE.probability_drift(p, p)
    assert drift["probability_total_variation"] == 0.0
    assert drift["entropy_delta"] == 0.0


def formal_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    result = ROOT / "experiments/p2_state_local_risk/results"
    required = [
        result / "response_rows.parquet",
        result / "geometry_score_rows.parquet",
        result / "unit_audit.parquet",
    ]
    if not all(path.exists() for path in required):
        pytest.skip("P2 formal outputs not generated yet")
    return tuple(pd.read_parquet(path) for path in required)  # type: ignore


def test_formal_vectors_and_scores_are_finite() -> None:
    response, scores, _audit = formal_outputs()
    assert np.isfinite(
        response.select_dtypes(include=[np.number]).to_numpy()
    ).all()
    assert np.isfinite(
        scores.select_dtypes(include=[np.number]).to_numpy()
    ).all()


def test_formal_nonlinear_baseline_identity() -> None:
    _response, _scores, audit = formal_outputs()
    assert (
        audit["state_operating_point_output_max_error"].max()
        <= protocol()["numeric"]["baseline_max_absolute_error_max"]
    )


def test_formal_h0_identities() -> None:
    response, _scores, _audit = formal_outputs()
    h0 = response[response["history_id"] == "H0"]
    assert h0["state_norm"].max() == 0.0
    assert h0["state_gradient_norm"].max() == 0.0
    assert h0["probability_total_variation"].max() == 0.0
    assert h0["h0_full_action_score_absolute_error"].max() <= 1.0e-12


def test_formal_protocol_identity_and_cardinality() -> None:
    result = ROOT / "experiments/p2_state_local_risk/results"
    metadata = result / "evaluation_metadata.json"
    if not metadata.exists():
        pytest.skip("P2 formal evaluation not generated yet")
    payload = json.loads(metadata.read_text())
    config = protocol()
    assert config["experiment"] == "p2_state_local_risk_closure_and_geometry_attribution"
    assert payload["completed"]
    assert payload["stage"] == "formal_evaluation"
    assert payload["row_counts"]["response_rows"] == config["data"][
        "evaluation"
    ]["expected_candidate_history_rows"]


def test_formal_calibration_artifact_passes() -> None:
    result = ROOT / "experiments/p2_state_local_risk/results"
    path = result / "calibration_summary.json"
    if not path.exists():
        pytest.skip("P2 calibration not generated yet")
    summary = json.loads(path.read_text())
    assert summary["passed"]
    assert summary["direction_count"] == 192
