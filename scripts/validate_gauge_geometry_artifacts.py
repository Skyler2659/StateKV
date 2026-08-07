#!/usr/bin/env python
"""Cross-check gauge geometry parquet/JSON/report artifacts."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

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


ROOT_ARTIFACTS = [
    "GAUGE_AWARE_OUTPUT_GEOMETRY_DESIGN_ZH.md",
    "configs/stages/gauge_geometry_config.yaml",
    "GAUGE_GEOMETRY_DATA_AUDIT_ZH.md",
    "GAUGE_AWARE_OUTPUT_GEOMETRY_RESULTS_ZH.md",
    "FISHER_PULLBACK_ANALYTICAL_DERIVATION_ZH.md",
    "FISHER_PULLBACK_STATE_ENVELOPE_RESULTS_ZH.md",
    "THEORY_MODEL_UPDATE_AFTER_GAUGE_GEOMETRY_ZH.md",
]

RUN_ARTIFACTS = [
    "oracle_geometry_rows.parquet",
    "oracle_geometry_kl_summary.json",
    "oracle_geometry_action_summary.json",
    "oracle_geometry_decomposition_summary.json",
    "path_fisher_quadrature_summary.json",
    "topk_gap_geometry_summary.json",
    "cumulant_geometry_summary.json",
    "pullback_jvp_rows.parquet",
    "pullback_linearization_summary.json",
    "pullback_low_rank_summary.json",
    "pullback_subspace_drift_summary.json",
    "q_state_envelope_rows.parquet",
    "q_state_envelope_coverage_summary.json",
    "q_state_envelope_tightness_summary.json",
    "q_state_action_ranking_summary.json",
    "spectral_band_envelope_summary.json",
    "pairwise_q_calibration_summary.json",
    "q_refresh_policy_summary.json",
    "q_free_generation_results.json",
]


def balanced_display_math(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    return (
        text.count("$$") % 2 == 0
        and "\\[" not in text
        and "\\]" not in text
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_discovery_config(args.config)
    run_dir = (
        REPOSITORY_ROOT
        / cfg.runtime.output_root
        / str(cfg.runtime.run_id)
    )
    checks = {}
    for name in ROOT_ARTIFACTS:
        checks["artifact:%s" % name] = (REPOSITORY_ROOT / name).exists()
    for name in RUN_ARTIFACTS:
        checks["artifact:%s" % name] = (run_dir / name).exists()

    rows = pd.read_parquet(run_dir / "oracle_geometry_rows.parquet")
    index = pd.read_parquet(run_dir / "gauge_vector_index.parquet")
    rows["task_bucket"] = rows["task"].map(
        lambda value: "GovReport"
        if "gov" in str(value).lower()
        else "NIAH"
    )
    checks["24_independent_sequences"] = rows["sample_id"].nunique() == 24
    task_counts = rows.groupby("task_bucket")["sample_id"].nunique()
    checks["12_sequences_per_task"] = bool(
        task_counts.to_dict() == {"GovReport": 12, "NIAH": 12}
    )
    group = rows.groupby(["sample_id", "anchor"])
    checks["24_candidates_per_anchor"] = bool(
        group["candidate_id"].nunique().eq(24).all()
    )
    checks["32_steps_per_candidate"] = bool(
        rows.groupby(
            ["sample_id", "anchor", "candidate_id"]
        )["horizon_offset"].nunique().eq(32).all()
    )
    checks["55296_geometry_rows"] = len(rows) == 55296
    checks["vector_index_aligned"] = len(index) == len(rows)
    checks["exact_budget_128"] = rows["total_budget"].eq(128).all()
    checks["active_budget_128"] = rows["active_cache_tokens"].eq(128).all()
    checks["token_position_alignment"] = rows[
        "token_position_aligned"
    ].all()
    checks["no_future_compressed_truth"] = not rows[
        "uses_future_compressed_truth"
    ].any()
    checks["no_task_feature"] = not rows["task_feature_used"].any()
    checks["full_vocab_streamed"] = rows[
        "full_vocabulary_streamed"
    ].all()
    checks["full_logits_not_claimed_stored"] = not rows[
        "full_logits_stored"
    ].any()
    checks["kl_identity"] = (
        float(rows["kl_cumulant_identity_abs_error"].max()) < 1.0e-10
    )
    checks["fisher_identity"] = (
        float(rows["fisher_variance_identity_abs_error"].max()) < 1.0e-10
    )
    checks["source_exact_kl_reproduced"] = (
        float(rows["source_exact_kl_abs_error"].max()) < 1.0e-3
    )
    source_relative_l2 = (
        rows["source_logit_l2_sq_abs_error"].to_numpy()
        / np.maximum(rows["source_logit_l2_sq"].to_numpy(), 1.0)
    )
    checks["source_logit_l2_reproduced"] = (
        float(np.max(source_relative_l2)) < 1.0e-5
    )
    numeric = rows.select_dtypes(include=[np.number])
    checks["all_numeric_columns_finite"] = bool(
        np.isfinite(numeric.to_numpy(dtype=np.float64)).all()
    )
    checks["g4b_range_bound_coverage_checked"] = bool(
        rows["g4b_range_covered"].notna().all()
    )
    checks["topk_mass_monotone"] = bool(
        (
            rows[
                [
                    "topk_mass_4",
                    "topk_mass_8",
                    "topk_mass_16",
                    "topk_mass_32",
                    "topk_mass_64",
                    "topk_mass_128",
                    "topk_mass_256",
                ]
            ]
            .diff(axis=1)
            .iloc[:, 1:]
            >= -1.0e-12
        ).all().all()
    )
    gate = json.loads(
        (run_dir / "gauge_geometry_gate_decision.json").read_text()
    )
    checks["gate_frozen"] = bool(gate["gate_frozen_before_formal_run"])
    if not bool(gate["stage_a_passed"]):
        for name in (
            "pullback_linearization_summary.json",
            "q_state_envelope_coverage_summary.json",
            "pairwise_q_calibration_summary.json",
            "q_refresh_policy_summary.json",
            "q_free_generation_results.json",
        ):
            content = json.loads((run_dir / name).read_text())
            checks["skip:%s" % name] = (
                content["status"] == "not_run_by_preregistered_gate"
            )
        checks["pullback_rows_empty_when_skipped"] = pd.read_parquet(
            run_dir / "pullback_jvp_rows.parquet"
        ).empty
        checks["q_state_rows_empty_when_skipped"] = pd.read_parquet(
            run_dir / "q_state_envelope_rows.parquet"
        ).empty

    for name in ROOT_ARTIFACTS:
        path = REPOSITORY_ROOT / name
        if path.suffix == ".md" and path.exists():
            checks["math_render:%s" % name] = balanced_display_math(path)
    normalized_checks = {
        key: bool(value) for key, value in checks.items()
    }
    result = {
        "passed": bool(all(normalized_checks.values())),
        "check_count": int(len(normalized_checks)),
        "passed_count": int(
            sum(bool(value) for value in normalized_checks.values())
        ),
        "checks": normalized_checks,
    }
    atomic_json(run_dir / "gauge_geometry_artifact_validation.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
