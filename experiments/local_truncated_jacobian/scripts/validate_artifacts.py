#!/usr/bin/env python3
"""Validate the complete local truncated-Jacobian artifact tree."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "experiments/local_truncated_jacobian"
PREREG_SHA256 = (
    "d21532527849ad9cf644458bbb622ba6afe9088443556d289401ece6e4c0b28e"
)
REGISTRY_SHA256 = (
    "f2d06b2732a2a0bf8baac6694ef35aa2ed4393a19e75400564a545786d787307"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_frame(
    checks: Dict[str, bool],
    name: str,
    path: Path,
    expected_rows: int,
) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    checks[f"{name}_rows"] = len(frame) == expected_rows
    if "finite" in frame:
        checks[f"{name}_finite"] = bool(frame["finite"].all())
    return frame


def run() -> Dict[str, Any]:
    checks: Dict[str, bool] = {}
    checks["preregistration_checksum"] = (
        sha256_file(EXPERIMENT / "PREREGISTRATION.md")
        == PREREG_SHA256
    )
    registry = (
        ROOT
        / "experiments/predictive_closure/raw/p0_alignment"
        / "formal_4bit_retry1/candidate_registry_rows.parquet"
    )
    checks["candidate_registry_checksum"] = (
        sha256_file(registry) == REGISTRY_SHA256
    )
    manifest = yaml.safe_load(
        (EXPERIMENT / "EXPERIMENT_MANIFEST.yaml").read_text(
            encoding="utf-8"
        )
    )
    checks["manifest_complete"] = (
        manifest["experiment"]["status"] == "complete"
        and manifest["experiment"]["formal_outcome"]
        == "local_outcome_l_a"
    )
    checks["manifest_no_heldout"] = not bool(
        manifest["integrity"]["calibration_or_test_loaded"]
    )
    checks["failure_report_absent_on_success"] = not (
        EXPERIMENT / "FAILURE_REPORT.md"
    ).exists()
    for report in (
        "AUDIT.md",
        "PREREGISTRATION.md",
        "LOCAL_TRUNCATED_JACOBIAN_RESULTS_ZH.md",
        "LOCAL_TRUNCATED_JACOBIAN_RESULTS_EN.md",
        "THEORY_UPDATE.md",
        "EXPERIMENT_MANIFEST.yaml",
        "figures/FIGURE_INDEX.md",
    ):
        checks[f"required_{report}"] = (
            EXPERIMENT / report
        ).is_file()

    radius = check_frame(
        checks,
        "radius",
        EXPERIMENT / "raw/radius_calibration/radius_rows.parquet",
        90,
    )
    checks["radius_units"] = bool(
        radius[
            [
                "sample_id",
                "anchor",
                "candidate_id",
                "layer",
            ]
        ]
        .drop_duplicates()
        .shape[0]
        == 10
    )
    l0 = EXPERIMENT / "raw/l0_boundary/formal"
    check_frame(checks, "l0_registry", l0 / "candidate_registry_rows.parquet", 128)
    check_frame(checks, "l0_direct", l0 / "direct_rows.parquet", 640)
    check_frame(
        checks,
        "l0_boundary",
        l0 / "native_boundary_rows.parquet",
        640,
    )
    check_frame(checks, "l0_jvp", l0 / "jvp_fd_rows.parquet", 1280)
    check_frame(
        checks, "l0_baseline", l0 / "local_baseline_rows.parquet", 640
    )
    check_frame(checks, "l0_audit", l0 / "anchor_audit_rows.parquet", 16)

    l1 = EXPERIMENT / "raw/l1_local_linearization/formal"
    l1_vectors = check_frame(
        checks, "l1_vector", l1 / "local_vector_rows.parquet", 6400
    )
    check_frame(
        checks, "l1_ranking", l1 / "local_ranking_rows.parquet", 800
    )
    check_frame(
        checks,
        "l1_transfer_vector",
        l1 / "native_transfer_vector_rows.parquet",
        1920,
    )
    check_frame(
        checks,
        "l1_transfer_ranking",
        l1 / "native_transfer_ranking_rows.parquet",
        240,
    )
    checks["l1_groups_eight"] = bool(
        l1_vectors.groupby(
            ["sample_id", "anchor", "layer", "direction", "scale"]
        )
        .size()
        .eq(8)
        .all()
    )

    l2 = EXPERIMENT / "raw/l2_depth_ablation/formal"
    l2_vectors = check_frame(
        checks, "l2_vector", l2 / "depth_vector_rows.parquet", 2432
    )
    check_frame(
        checks, "l2_ranking", l2 / "depth_ranking_rows.parquet", 304
    )
    check_frame(
        checks, "l2_physical", l2 / "physical_output_rows.parquet", 640
    )
    checks["l2_depth_counts"] = (
        l2_vectors.groupby("depth").size().to_dict()
        == {0: 640, 1: 640, 2: 640, 4: 512}
    )
    checks["l2_baseline_exact"] = bool(
        l2_vectors[l2_vectors["depth"].gt(0)][
            "baseline_relative_l2"
        ].max()
        <= 1.0e-12
    )

    l3 = EXPERIMENT / "raw/l3_output_diagnostic/formal"
    check_frame(checks, "l3_candidate", l3 / "candidate_rows.parquet", 640)
    check_frame(
        checks, "l3_group", l3 / "group_ranking_rows.parquet", 80
    )
    check_frame(
        checks, "l3_sequence", l3 / "sequence_aggregate.parquet", 4
    )
    check_frame(checks, "l3_task", l3 / "task_aggregate.parquet", 2)

    summary_paths = {
        "radius": EXPERIMENT
        / "raw/radius_calibration/radius_calibration_summary.json",
        "l0": l0 / "l0_summary.json",
        "l1": l1 / "l1_summary.json",
        "l2": l2 / "l2_summary.json",
        "l3": l3 / "l3_summary.json",
    }
    summaries = {
        key: json.loads(path.read_text(encoding="utf-8"))
        for key, path in summary_paths.items()
    }
    checks["all_stage_integrity"] = bool(
        summaries["l0"]["integrity"]["passed"]
        and summaries["l1"]["integrity"]["passed"]
        and summaries["l2"]["integrity"]["passed"]
        and summaries["l3"]["integrity"]["passed"]
    )
    checks["all_stage_gates"] = bool(
        summaries["radius"]["calibration_passed"]
        and summaries["l0"]["formal_l0_passed"]
        and summaries["l1"]["formal_l1_local_passed"]
        and summaries["l1"]["formal_l1_native_transfer_passed"]
        and summaries["l2"]["formal_l2_passed"]
    )
    checks["all_stage_train_only"] = all(
        not bool(summary.get("calibration_or_test_loaded", False))
        for summary in summaries.values()
    )

    figures = sorted((EXPERIMENT / "figures").glob("*.png"))
    checks["figure_count"] = len(figures) == 10
    figure_shapes = {}
    for path in figures:
        with Image.open(path) as image:
            width, height = image.size
        checks[f"figure_{path.name}_valid"] = (
            width >= 600 and height >= 350 and path.stat().st_size > 10_000
        )
        figure_shapes[path.name] = [width, height]

    result = {
        "status": "passed" if all(checks.values()) else "failed",
        "all_checks_passed": bool(all(checks.values())),
        "check_count": len(checks),
        "checks": checks,
        "figure_shapes": figure_shapes,
        "formal_outcome": manifest["experiment"]["formal_outcome"],
        "calibration_or_test_loaded": False,
    }
    output = EXPERIMENT / "summaries/final_validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
