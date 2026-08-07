from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from statekv.repository_layout import verify_repository_checksum


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/p3_physical_recovery"
SCRIPT_DIR = EXPERIMENT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import p3pr_core as CORE  # noqa: E402
import run_p3pr as RUNNER  # noqa: E402


STAGES = (
    "diagnostic",
    "calibration",
    "formal",
    "replication",
    "recovery_formal",
    "recovery_replication",
    "scope_budget",
    "scope_history_anchor",
    "scope_candidate_pool",
    "disagreement_calibration",
    "disagreement_formal",
    "disagreement_replication",
)


def evaluation() -> dict:
    return json.loads(
        (
            EXPERIMENT / "results/P3PR_EVALUATION_SUMMARY.json"
        ).read_text()
    )


def test_physical_baseline_candidate_clone_identity() -> None:
    units = pd.read_parquet(
        EXPERIMENT / "results/diagnostic/unit_rows.parquet"
    )
    assert units["prequery_clone_isolated"].all()
    assert units["baseline_repeat_max_abs_error"].max() == 0.0


def test_all_layer_cache_isolation_source() -> None:
    source = inspect.getsource(CORE.clone_mlx_state)
    assert "for cache in state.cache" in source
    assert "positions.detach().clone()" in source


def test_physical_no_op_candidate() -> None:
    summary = evaluation()
    assert summary["integrity"]["maximum_no_op_exact_kl"] == 0.0


def test_physical_exact_kl_reproducibility() -> None:
    left = np.array([0.1, 0.2, -0.3])
    right = np.array([0.0, 0.25, -0.2])
    assert CORE.exact_kl(left, right) == CORE.exact_kl(left, right)
    assert CORE.exact_kl(left, left) == pytest.approx(0.0)


def test_token_position_rope_alignment() -> None:
    summary = evaluation()["integrity"]
    assert summary["all_query_aligned"]
    assert summary["all_token_aligned"]


def test_state_conditioned_deletion_identity() -> None:
    assert (
        evaluation()["integrity"][
            "maximum_stable_identity_relative_l2"
        ]
        < 1.0e-4
    )


def test_physical_qkv_extraction_schema() -> None:
    rows = pd.read_parquet(
        EXPERIMENT / "results/diagnostic/layer_rows.parquet"
    )
    assert {
        "deleted_key_norm",
        "deleted_value_norm",
        "selector_key_query",
        "deleted_attention_mass_mean",
    }.issubset(rows.columns)


def test_physical_adjacent_response_finite() -> None:
    rows = pd.read_parquet(
        EXPERIMENT / "results/diagnostic/layer_rows.parquet"
    )
    assert np.isfinite(
        rows[
            [
                "adjacent_cosine",
                "adjacent_relative_l2",
                "local_r_norm",
            ]
        ].to_numpy()
    ).all()


def test_controlled_and_physical_candidate_identity_distinguished() -> None:
    audit = json.loads(
        (EXPERIMENT / "results/source_target_audit.json").read_text()
    )
    assert not audit["p3_target_code_audit"][
        "is_same_current_physical_state_clone_target"
    ]
    assert audit["p3pr_target"]["does_not_replace_p3_target"]
    registry = pd.read_parquet(
        EXPERIMENT / "results/diagnostic/candidate_registry.parquet"
    )
    assert registry["candidate_id"].str.startswith("physical_delete_").all()


def test_candidate_pool_coverage() -> None:
    for stage in STAGES:
        registry = pd.read_parquet(
            EXPERIMENT / f"results/{stage}/candidate_registry.parquet"
        )
        counts = registry.groupby(
            ["sample_id", "target_anchor"]
        )["deleted_position"].nunique()
        assert counts.eq(8).all()


def test_multi_boundary_schema() -> None:
    rows = pd.read_parquet(
        EXPERIMENT / "results/calibration/candidate_rows.parquet"
    )
    assert {
        "multi_all_endpoint_risk",
        "multi_uniform8_endpoint_risk",
        "multi_inherited3_endpoint_risk",
        "multi_pair_endpoint_risk",
        "multi_three_endpoint_risk",
    }.issubset(rows.columns)


def test_boundary_selection_freeze() -> None:
    frozen = json.loads(
        (
            EXPERIMENT / "results/frozen_disagreement_model.json"
        ).read_text()
    )
    assert frozen["boundary"] == 27
    assert frozen["probe_layer"] == 26
    assert frozen["boundary_count"] == 1


def test_dense_sparse_zero_boundary_equivalence() -> None:
    units = pd.read_parquet(
        EXPERIMENT / "results/diagnostic/unit_rows.parquet"
    )
    assert units["multi_reconstruction_max_abs_error"].max() == 0.0
    assert units["readout_reconstruction_max_abs_error"].max() == 0.0


def test_kv_summary_correctness() -> None:
    rows = pd.read_parquet(
        EXPERIMENT / "results/calibration/layer_rows.parquet"
    )
    assert (rows["key_norm_variance"] >= 0).all()
    assert (rows["value_norm_variance"] >= 0).all()
    assert (rows["attention_entropy"] >= 0).all()
    assert rows["attention_concentration"].between(0, 1).all()


def test_fixed_projection_freeze() -> None:
    left = CORE.deterministic_projection(12, 4, 2026072811)
    right = CORE.deterministic_projection(12, 4, 2026072811)
    assert np.array_equal(left, right)


def test_pca_calibration_only_fit() -> None:
    source = (
        EXPERIMENT / "scripts/analyze_p3pr_calibration.py"
    ).read_text()
    assert "groups != held_out" in source
    assert "PCA(n_components=dimension" in source
    assert "results/formal" not in source


def test_no_formal_feature_fitting() -> None:
    frozen = json.loads(
        (
            EXPERIMENT / "results/frozen_disagreement_model.json"
        ).read_text()
    )
    assert frozen["formal_fit_allowed"] is False
    assert frozen["parameter_count"] == 0


def test_cross_layer_interaction_rank() -> None:
    config = yaml.safe_load(
        (EXPERIMENT / "p3pr_config.yaml").read_text()
    )
    assert config["representations"]["interaction_ranks"] == [2, 4, 8]
    models = pd.read_parquet(
        EXPERIMENT / "results/calibration/model_class_results.parquet"
    )
    ranks = set(
        models.loc[models["model_class"].eq("M6"), "interaction_rank"]
        .dropna()
        .astype(int)
    )
    assert ranks == {2, 4, 8}


def test_exact_kl_leakage_prohibition() -> None:
    with pytest.raises(ValueError):
        CORE.validate_feature_names(["exact_physical_kl"])


def test_future_token_leakage_prohibition() -> None:
    with pytest.raises(ValueError):
        CORE.validate_feature_names(["future_token"])


def test_future_attention_leakage_prohibition() -> None:
    with pytest.raises(ValueError):
        CORE.validate_feature_names(["future_attention"])


def test_physical_path_midpoint_correctness() -> None:
    class QuadraticReadout:
        base_input = np.array([1.0])

        @staticmethod
        def evaluate(value):
            vector = np.asarray(value, dtype=np.float64)
            return np.array([np.dot(vector, vector)])

    for count in (1, 2, 4, 8):
        result = RUNNER._path_delta(
            QuadraticReadout(), np.array([2.0]), count, 1.0e-3
        )
        assert result[0] == pytest.approx(4.0, rel=1.0e-8)


def test_candidate_specific_probe_isolation() -> None:
    formal = pd.read_parquet(
        EXPERIMENT / "results/disagreement_formal/candidate_rows.parquet"
    )
    assert formal["probe_b27_path_k1_risk"].notna().all()
    assert (
        formal.groupby("sample_id")["candidate_id"].nunique().eq(8).all()
    )


def test_full_reference_feature_prohibition() -> None:
    with pytest.raises(ValueError):
        CORE.validate_feature_names(["full_reference_state"])


def test_sequence_first_aggregation() -> None:
    rows = pd.DataFrame(
        {
            "sample_id": ["a"] * 3 + ["b"] * 3,
            "task": ["x"] * 3 + ["y"] * 3,
            "target_anchor": [1] * 6,
            "exact_physical_kl": [0, 1, 2, 0, 1, 2],
            "score": [0, 1, 2, 0, 2, 1],
        }
    )
    sequence, summary = CORE.sequence_first_metrics(rows, "score")
    assert len(sequence) == 2
    assert summary["sequence_count"] == 2


def test_task_layer_history_stratification() -> None:
    rows = pd.read_parquet(
        EXPERIMENT / "results/diagnostic/layer_rows.parquet"
    )
    assert set(rows["task"]) == {"gov_report", "niah_single_1"}
    assert set(rows["layer"]) == set(range(28))
    assert set(rows["history_length"]) == {8, 32}


def test_formal_replication_separation() -> None:
    ledger = yaml.safe_load(
        (EXPERIMENT / "P3PR_DATA_LEDGER.yaml").read_text()
    )
    formal = set(ledger["allocations"]["disagreement_formal"])
    replication = set(
        ledger["allocations"]["disagreement_replication"]
    )
    calibration = set(
        ledger["allocations"]["disagreement_calibration"]
    )
    assert formal.isdisjoint(replication)
    assert formal.isdisjoint(calibration)
    assert replication.isdisjoint(calibration)


def test_model_class_ledger_integrity() -> None:
    ledger = yaml.safe_load(
        (EXPERIMENT / "P3PR_MODEL_CLASS_LEDGER.yaml").read_text()
    )
    assert set(ledger["classes"]) >= {
        "M1_oracle_ladder",
        "M2_state_conditioned_injection",
        "M3_single_boundary",
        "M4_sparse_multi_boundary",
        "M5_kv_augmented",
        "M6_low_rank_interaction",
        "M7_physical_candidate_generation",
        "M8_mechanistic_and_diagnostic",
        "M9_dense_oracle_and_probe",
        "physical_path",
    }


def test_data_ledger_exclusion() -> None:
    scan = json.loads(
        (
            EXPERIMENT / "results/data_scan_all_allocated.json"
        ).read_text()
    )
    assert scan["all_pass"]
    assert scan["new_old_overlap"] == []
    assert len(scan["rows"]) == 48


def test_atomic_writes(tmp_path: Path) -> None:
    json_path = tmp_path / "a.json"
    frame_path = tmp_path / "a.parquet"
    CORE.atomic_json(json_path, {"x": 1})
    CORE.atomic_frame(frame_path, pd.DataFrame({"x": [1]}))
    assert json.loads(json_path.read_text()) == {"x": 1}
    assert pd.read_parquet(frame_path)["x"].tolist() == [1]


def test_all_vectors_finite() -> None:
    for stage in STAGES:
        for name in ("candidate_rows", "layer_rows", "unit_rows"):
            frame = pd.read_parquet(
                EXPERIMENT / f"results/{stage}/{name}.parquet"
            )
            numeric = frame.select_dtypes(include=[np.number])
            for column in numeric:
                assert np.isfinite(numeric[column].dropna().to_numpy()).all()


def test_checksum_verification() -> None:
    verification = json.loads(
        (
            EXPERIMENT / "results/checksum_verification.json"
        ).read_text()
    )
    assert verification["passed"]
    manifest = yaml.safe_load(
        (
            EXPERIMENT / "P3_PHYSICAL_RECOVERY_MANIFEST.yaml"
        ).read_text()
    )
    for relative, expected in manifest["checksums"].items():
        assert verify_repository_checksum(ROOT, relative, expected)


def test_formula_rendering() -> None:
    audit = json.loads(
        (EXPERIMENT / "results/formula_render_audit.json").read_text()
    )
    assert audit["passed"]
    assert audit["document_count"] == 25
    assert audit["mathml_node_count"] > 0
    assert audit["warning_count"] == 0
    assert audit["raw_math_leftover_count"] == 0


def test_p0_through_p3_manifests_unchanged() -> None:
    summary = evaluation()
    assert summary["all_old_manifests_unchanged"]
    assert all(
        row["unchanged"]
        for row in summary["old_manifest_checks"].values()
    )


def test_candidate_generator_rejects_forbidden_inputs() -> None:
    rows = [
        {
            "candidate_id": str(index),
            "action_score": float(index),
            "dense_score": float(8 - index),
            "exact_physical_kl": 0.0,
        }
        for index in range(8)
    ]
    with pytest.raises(ValueError):
        CORE.select_mechanism_disagreement(rows, 8)


def test_candidate_generator_argmin_disagreement() -> None:
    rows = [
        {
            "candidate_id": str(index),
            "action_score": float(index),
            "dense_score": float((index - 1) % 9),
        }
        for index in range(9)
    ]
    selected, audit = CORE.select_mechanism_disagreement(rows, 8)
    assert len(selected) == 8
    assert (
        audit["action_argmin_candidate_id"]
        != audit["dense_argmin_candidate_id"]
    )
    assert not audit["exact_physical_kl_used"]


def test_candidate_generator_artifact_label_free() -> None:
    units = pd.read_parquet(
        EXPERIMENT
        / "results/disagreement_formal/unit_rows.parquet"
    )
    assert units["candidate_generator_seed_count"].eq(24).all()
    assert (~units["candidate_generator_exact_kl_used"]).all()
    assert (~units["candidate_generator_endpoint_logits_used"]).all()
    assert (~units["candidate_generator_task_id_used"]).all()


def test_fresh_formal_gate_passes() -> None:
    assert evaluation()["formal_gate"]["passed"]


def test_frozen_replication_gate_passes() -> None:
    assert evaluation()["replication_gate"]["passed"]


def test_minimality_gate_passes() -> None:
    result = evaluation()["minimality"]
    assert result["passed"]
    assert result["formal_dense_spearman_gap"] <= 0.02
    assert result["replication_dense_spearman_gap"] <= 0.02
    assert result["formal_action_only_gain_retention"] >= 0.90


def test_both_tasks_strict_regret_gain() -> None:
    result = evaluation()
    for split in ("formal", "replication"):
        model = result[split]["task_normalized_regret"]
        action = result[f"{split}_action_only"][
            "task_normalized_regret"
        ]
        assert all(model[task] < action[task] for task in model)


def test_terminal_success_is_mechanical() -> None:
    result = evaluation()
    assert result["outcome"] == "P3PR-S"
    assert result["terminal_condition"] == "Terminal Success"
    assert result["terminal_success"]


def test_root_readme_preserves_physical_scope() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "same-state physical evaluator" in text
    assert "candidate-specific teacher" in text
    assert "online policy" in text


def test_required_artifacts_exist() -> None:
    retired_reports = {
        "P3PR_MASTER_EXPERIMENT_PLAN.md",
        "P3PR_TARGET_AUDIT.md",
        "P3PR_DISCREPANCY_DECOMPOSITION.md",
        "P3PR_PHYSICAL_INJECTION_RESULTS.md",
        "P3PR_SINGLE_BOUNDARY_RESULTS.md",
        "P3PR_MULTI_BOUNDARY_RESULTS.md",
        "P3PR_KV_SUMMARY_RESULTS.md",
        "P3PR_CROSS_LAYER_RESULTS.md",
        "P3PR_PHYSICAL_CANDIDATE_RESULTS.md",
        "P3PR_REPRESENTATION_SUFFICIENCY.md",
        "P3PR_DENSE_ORACLE_RESULTS.md",
        "P3PR_PHYSICAL_PATH_RESULTS.md",
        "P3PR_FORMAL_RESULTS.md",
        "P3PR_REPLICATION_RESULTS.md",
        "P3PR_MINIMALITY_RESULTS.md",
        "P3PR_SCOPE_EXPANSION.md",
        "P3PR_FAILURE_ANALYSIS.md",
        "P3PR_CODE_AUDIT.md",
        "P3PR_CUMULATIVE_RESULTS.md",
        "P3PR_FINAL_RECOMMENDATION.md",
    }
    structured = {
        "P3PR_DATA_LEDGER.yaml",
        "P3PR_MODEL_CLASS_LEDGER.yaml",
    }
    assert all((EXPERIMENT / name).is_file() for name in structured)
    ledger = yaml.safe_load(
        (ROOT / "experiments/retired_documents.yaml").read_text()
    )
    for name in retired_reports:
        path = EXPERIMENT / name
        relative = path.relative_to(ROOT).as_posix()
        assert relative in ledger["documents"]
        assert not path.exists()
