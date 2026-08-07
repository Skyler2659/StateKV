#!/usr/bin/env python
"""Validate independent-Fisher artifacts, gates, schemas, and data separation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
IMPORT_ROOTS = (
    REPOSITORY_ROOT,
    REPOSITORY_ROOT / "benchmarks" / "torch",
    REPOSITORY_ROOT / "benchmarks" / "mlx",
)
for import_root in IMPORT_ROOTS:
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from statekv.config import load_discovery_config
from statekv.trajectory_analysis import atomic_json


REQUIRED = (
    "independent_fisher_geometry_rows.parquet",
    "independent_fisher_replication_summary.json",
    "adaptive_curvature_rows.parquet",
    "adaptive_curvature_summary.json",
    "fisher_trust_region_summary.json",
    "pullback_operating_point_rows.parquet",
    "pullback_jvp_validation_summary.json",
    "fisher_direct_ranking_summary.json",
    "oracle_midpoint_recovery_summary.json",
    "state_action_cross_term_rows.parquet",
    "state_action_cross_term_summary.json",
    "pullback_low_rank_rows.parquet",
    "pullback_low_rank_summary.json",
    "pullback_subspace_drift_summary.json",
    "q_state_envelope_rows.parquet",
    "q_state_envelope_coverage_summary.json",
    "q_state_envelope_tightness_summary.json",
    "q_state_action_summary.json",
    "interaction_q_envelope_summary.json",
    "spectral_band_q_envelope_summary.json",
    "q_pairwise_calibration_summary.json",
    "q_refresh_policy_rows.parquet",
    "q_refresh_policy_summary.json",
    "q_free_generation_results.json",
)


def _finite_frame(frame: pd.DataFrame) -> bool:
    numeric = frame.select_dtypes(include=[np.number])
    return bool(np.isfinite(numeric.to_numpy(dtype=np.float64)).all())


def validate(cfg: Any, root: Path) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []

    def add(name: str, passed: bool, detail: Any = None) -> None:
        checks.append(
            {"name": str(name), "passed": bool(passed), "detail": detail}
        )

    add(
        "required_artifacts_exist",
        all((root / name).exists() for name in REQUIRED),
        [name for name in REQUIRED if not (root / name).exists()],
    )
    geometry = pd.read_parquet(
        root / "independent_fisher_geometry_rows.parquet"
    )
    curvature = pd.read_parquet(root / "adaptive_curvature_rows.parquet")
    inventory = pd.read_parquet(root / "independent_candidate_inventory.parquet")
    vectors = pd.read_parquet(root / "independent_vector_index.parquet")
    manifest = pd.read_parquet(root / "new_sequence_manifest.parquet")
    add("new_sequence_count_24", manifest["sample_id"].nunique() == 24)
    add(
        "task_split_12_12",
        sorted(manifest.groupby("task")["sample_id"].nunique().tolist())
        == [12, 12],
    )
    gov = manifest[manifest["task"] == "gov_report"]
    add(
        "official_govreport_loader",
        len(gov) == 12 and gov["dataset_official"].astype(bool).all(),
    )
    old_ids = set()
    for run_id in cfg.independent_fisher.prior_run_ids:
        prior = root.parent / str(run_id)
        for name in (
            "oracle_geometry_rows.parquet",
            "output_candidate_inventory.parquet",
        ):
            path = prior / name
            if path.exists():
                old_ids.update(
                    pd.read_parquet(path, columns=["sample_id"])[
                        "sample_id"
                    ].astype(str)
                )
                break
    new_ids = set(manifest["sample_id"].astype(str))
    add("new_old_sequence_id_disjoint", not bool(new_ids & old_ids))
    add(
        "prompt_hashes_unique",
        manifest["prompt_sha256"].nunique() == 24,
    )
    add(
        "preregistered_generation_rule",
        manifest["generation_rule_preregistered"].astype(bool).all(),
    )
    expected_rows = 24 * 3 * 24 * 16
    add("geometry_row_count", len(geometry) == expected_rows, len(geometry))
    add("curvature_row_count", len(curvature) == expected_rows, len(curvature))
    add("vector_index_row_count", len(vectors) == expected_rows, len(vectors))
    add("inventory_row_count", len(inventory) == 24 * 3 * 24, len(inventory))
    mask_counts = inventory.groupby(["sample_id", "anchor"])[
        "mask_hash"
    ].nunique()
    add("24_distinct_masks_per_anchor", mask_counts.eq(24).all())
    add(
        "candidate_budget_exact",
        inventory["total_budget"].eq(128).all()
        and geometry["active_cache_tokens"].eq(128).all(),
    )
    add(
        "physical_layer_shared_masks",
        inventory["physical_layer_shared_mask"].astype(bool).all(),
    )
    add("gqa_shared_masks", inventory["gqa_shared"].astype(bool).all())
    add(
        "token_positions_aligned",
        geometry["token_position_aligned"].astype(bool).all(),
    )
    add(
        "no_future_compressed_truth_stage_a",
        (~geometry["uses_future_compressed_truth"].astype(bool)).all(),
    )
    add(
        "task_id_not_feature",
        (~geometry["uses_task_feature"].astype(bool)).all(),
    )
    add("geometry_no_nan_inf", _finite_frame(geometry))
    add("curvature_no_nan_inf", _finite_frame(curvature))
    add(
        "exact_kl_identity",
        geometry["kl_cumulant_identity_abs_error"].max() <= 1.0e-10,
        float(geometry["kl_cumulant_identity_abs_error"].max()),
    )
    tolerance = np.maximum(
        float(cfg.independent_fisher.adaptive_absolute_tolerance),
        float(cfg.independent_fisher.adaptive_relative_tolerance)
        * curvature["exact_kl"].abs().to_numpy(dtype=np.float64),
    )
    adaptive_error = curvature[
        "adaptive_weighted_abs_error_vs_exact"
    ].to_numpy(dtype=np.float64)
    add(
        "adaptive_integration_matches_exact",
        bool(np.all(adaptive_error <= 10.0 * tolerance)),
        float(np.max(adaptive_error / np.maximum(tolerance, 1.0e-30))),
    )
    add(
        "adaptive_no_warnings",
        curvature["adaptive_warning_count"].eq(0).all(),
        int(curvature["adaptive_warning_count"].sum()),
    )
    add(
        "curvature_peak_location_valid",
        curvature["curvature_peak_location"].between(0.0, 1.0).all(),
    )
    add(
        "curvature_width_nonnegative",
        (curvature["effective_curvature_width"] >= 0.0).all(),
    )
    add(
        "top_switch_count_nonnegative",
        (curvature["top1_change_count"] >= 0).all(),
    )
    replication = json.loads(
        (root / "independent_fisher_replication_summary.json").read_text()
    )
    add(
        "replication_json_row_count",
        int(replication["row_count"]) == len(geometry),
    )
    add(
        "replication_json_sequence_count",
        int(replication["sequence_count"]) == 24,
    )
    trust = json.loads((root / "fisher_trust_region_summary.json").read_text())
    add("trust_threshold_no_task_id", not trust["thresholds_use_task_id"])
    add(
        "trust_heldout_no_leakage",
        not trust["heldout_sequence_leakage"],
    )
    gate = replication["gate"]
    add("fixed_gl5_not_gate", not gate["fixed_gl5_is_blocking_gate"])
    add("no_post_hoc_gate_relaxation", not gate["post_hoc_gate_relaxation"])
    add(
        "sequence_direction_reported",
        set(gate["sequence_direction"])
        == set(geometry["task"].astype(str).unique()),
    )
    add(
        "anchor_split_reported",
        set(replication["pointwise"]["anchor"]) == {"16", "32", "48"},
    )
    add(
        "horizon_split_reported",
        set(replication["action"]["horizon"]) == {"1", "4", "8", "16"},
    )
    stage_a_passed = bool(gate["stage_a_prime_replication_passed"])
    pullback = pd.read_parquet(root / "pullback_operating_point_rows.parquet")
    cross = pd.read_parquet(root / "state_action_cross_term_rows.parquet")
    if stage_a_passed:
        add("stage_b_executed_when_authorized", len(pullback) > 0)
        add("pullback_no_nan_inf", _finite_frame(pullback))
        add(
            "pure_map_no_cache_mutation",
            pullback["pure_map_cache_unchanged"].astype(bool).all(),
        )
        add(
            "pure_map_repeated_calls_equal",
            pullback["pure_map_repeated_equal"].astype(bool).all(),
        )
        add(
            "oracle_midpoint_marked_non_deployable",
            (
                ~pullback[
                    pullback["pullback_mode"] == "B1_ORACLE_MIDPOINT"
                ]["deployable"].astype(bool)
            ).all(),
        )
        add(
            "candidate_direct_no_future_truth",
            (
                ~pullback[
                    pullback["pullback_mode"]
                    == "B2_CANDIDATE_DIRECT_MIDPOINT"
                ]["uses_future_compressed_truth"].astype(bool)
            ).all(),
        )
        add(
            "predicted_midpoint_training_excludes_test",
            pullback[
                pullback["pullback_mode"] == "B3_PREDICTED_MIDPOINT"
            ][
                "predicted_response_training_excludes_test_sequence"
            ].astype(bool).all(),
        )
        add(
            "predicted_midpoint_radius_supported",
            pullback[
                pullback["pullback_mode"] == "B3_PREDICTED_MIDPOINT"
            ]["predicted_response_scale"].between(0.0, 1.0).all(),
        )
        add("cross_rows_present", len(cross) > 0)
        add(
            "cross_decomposition_identity",
            cross["decomposition_abs_error"].max() <= 1.0e-8,
        )
        add(
            "cross_cauchy_schwarz",
            cross["cauchy_schwarz_holds"].astype(bool).all(),
        )
        add(
            "scalar_cross_bound",
            cross["scalar_bound_holds"].astype(bool).all(),
        )
    else:
        add("stage_b_skipped_when_blocked", len(pullback) == 0)
        for name in (
            "pullback_jvp_validation_summary.json",
            "oracle_midpoint_recovery_summary.json",
            "fisher_direct_ranking_summary.json",
        ):
            value = json.loads((root / name).read_text())
            add(
                "explicit_skip_%s" % name,
                value["status"] == "not_run_by_preregistered_gate",
            )
    required_reports = (
        REPOSITORY_ROOT / "INDEPENDENT_FISHER_VALIDATION_RESULTS_ZH.md",
        REPOSITORY_ROOT / "CANDIDATE_CONDITIONED_PULLBACK_DERIVATION_ZH.md",
        REPOSITORY_ROOT / "Q_STATE_ENVELOPE_RESULTS_ZH.md",
        REPOSITORY_ROOT / "THEORY_MODEL_UPDATE_AFTER_PULLBACK_ZH.md",
        REPOSITORY_ROOT / "NEW_SEQUENCE_DATA_AUDIT_ZH.md",
    )
    add(
        "required_reports_exist",
        all(path.exists() for path in required_reports),
        [str(path) for path in required_reports if not path.exists()],
    )
    result = {
        "status": "passed" if all(row["passed"] for row in checks) else "failed",
        "checks_passed": int(sum(row["passed"] for row in checks)),
        "checks_total": int(len(checks)),
        "checks": checks,
    }
    atomic_json(root / "independent_fisher_artifact_validation.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_discovery_config(args.config)
    root = (
        REPOSITORY_ROOT
        / cfg.runtime.output_root
        / str(cfg.runtime.run_id)
    )
    result = validate(cfg, root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
