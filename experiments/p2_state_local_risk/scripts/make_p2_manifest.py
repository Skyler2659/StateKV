#!/usr/bin/env python3
"""Build the non-self-referential P2 SHA-256 manifest."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Dict

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "experiments/p2_state_local_risk"
RESULTS = EXPERIMENT / "results"
MANIFEST = EXPERIMENT / "P2_STATE_LOCAL_MANIFEST.yaml"
MIRROR = RESULTS / "p2_checksums.json"
ROOT_FILES = [
    ROOT / "experiments/p2_state_local_risk/docs/code_audit.md",
    ROOT / "experiments/p2_state_local_risk/docs/experiment_plan.md",
    ROOT / "experiments/p2_state_local_risk/docs/calibration.md",
    ROOT / "experiments/p2_state_local_risk/docs/results.md",
    ROOT / "experiments/p2_state_local_risk/docs/failure_analysis.md",
    ROOT / "configs/frozen/p2_state_local_config.yaml",
    ROOT / "tests/test_p2_state_local.py",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_state() -> Dict[str, object]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    ).stdout.strip()
    return {"commit": commit, "worktree_dirty": bool(status)}


def main() -> None:
    exclusions = {MANIFEST.resolve(), MIRROR.resolve()}
    files = set(ROOT_FILES)
    files.update(
        path
        for path in EXPERIMENT.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.resolve() not in exclusions
    )
    checksums = {
        str(path.relative_to(ROOT)): sha256(path)
        for path in sorted(files)
    }
    gate = json.loads(
        (RESULTS / "p2_gate_outcome.json").read_text(
            encoding="utf-8"
        )
    )
    metadata = json.loads(
        (RESULTS / "evaluation_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    calibration = json.loads(
        (RESULTS / "calibration_summary.json").read_text(
            encoding="utf-8"
        )
    )
    tests = json.loads(
        (RESULTS / "test_summary.json").read_text(
            encoding="utf-8"
        )
    )
    protocol = yaml.safe_load(
        (ROOT / "configs/frozen/p2_state_local_config.yaml").read_text(
            encoding="utf-8"
        )
    )
    calibration_section = protocol["data"]["calibration"]
    calibration_ids = [
        *[
            f"gov_report:{int(value)}"
            for value in calibration_section["gov_report_indices"]
        ],
        *[
            f"synthetic_niah_{int(value)}"
            for value in calibration_section["niah_offsets"]
        ],
    ]
    response = pd.read_parquet(RESULTS / "response_rows.parquet")
    state = pd.read_parquet(RESULTS / "state_registry.parquet")
    ranking = pd.read_parquet(
        RESULTS / "unit_ranking_rows.parquet"
    )
    geometry = pd.read_parquet(
        RESULTS / "geometry_score_rows.parquet"
    )
    manifest = {
        "schema_version": 1,
        "experiment": protocol["experiment"],
        "outcome": gate["outcome"],
        "seed": int(protocol["numeric"]["seed"]),
        "config": {
            "path": "p2_state_local_config.yaml",
            "sha256_at_evaluation": metadata["config_sha256"],
            "sha256_current": sha256(
                ROOT / "configs/frozen/p2_state_local_config.yaml"
            ),
            "numeric_calibration_status": protocol[
                "numeric_calibration_status"
            ],
            "fd_selected_relative_radius": calibration[
                "selected_relative_radius"
            ],
        },
        "source_integrity": {
            "p0_manifest_sha256": protocol["source_integrity"][
                "p0"
            ]["manifest_sha256"],
            "p0_outcome": tests["source_manifests"]["p0"][
                "outcome"
            ],
            "p0_entries": tests["source_manifests"]["p0"][
                "entry_count"
            ],
            "p1_manifest_sha256": protocol["source_integrity"][
                "p1"
            ]["manifest_sha256"],
            "p1_outcome": tests["source_manifests"]["p1"][
                "outcome"
            ],
            "p1_entries": tests["source_manifests"]["p1"][
                "entry_count"
            ],
        },
        "data": {
            "calibration_ids": calibration_ids,
            "evaluation_ids": metadata["sequence_ids"],
            "sequence_count": len(metadata["sequence_ids"]),
            "calibration_direction_count": calibration[
                "direction_count"
            ],
            "calibration_row_count": calibration["row_count"],
            "response_row_count": len(response),
            "geometry_score_row_count": len(geometry),
            "state_unit_count": len(state),
            "ranking_row_count": len(ranking),
        },
        "gates": {
            "gate0": gate["gate0"]["passed"],
            "gate1": gate["gate1"]["passed"],
            "gate2": gate["gate2"]["passed"],
            "gate2_eligible": gate["gate2"]["eligible"],
            "gate3": gate["gate3"]["passed"],
            "gate3_eligible": gate["gate3"]["eligible"],
        },
        "verification": {
            "passed_test_count": tests["passed_test_count"],
            "skipped_test_count": tests["skipped_test_count"],
            "formula_render_passed": tests[
                "formula_audit_passed"
            ],
            "formula_warning_count": tests[
                "formula_warning_count"
            ],
            "formula_raw_math_leftover_count": tests[
                "formula_raw_math_leftover_count"
            ],
        },
        "model": {
            "source": metadata["model_info"]["model_name"],
            "execution": "fully_dequantized_float32",
            "quantized_modules_after_dequantization": metadata[
                "model_info"
            ]["dequantization"]["after_quantized_modules_total"],
        },
        "git": git_state(),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "pandas": pd.__version__,
        },
        "checksum_exclusions": [
            str(MANIFEST.relative_to(ROOT)),
            str(MIRROR.relative_to(ROOT)),
        ],
        "checksum_exclusion_reason": (
            "Manifest and JSON mirror are excluded to avoid "
            "self-referential checksums."
        ),
        "checksums": checksums,
    }
    MANIFEST.write_text(
        yaml.safe_dump(
            manifest,
            allow_unicode=True,
            sort_keys=False,
            width=100,
        ),
        encoding="utf-8",
    )
    MIRROR.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "experiment": manifest["experiment"],
                "outcome": manifest["outcome"],
                "checksums": checksums,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "outcome": manifest["outcome"],
                "checksum_entry_count": len(checksums),
                "manifest": str(MANIFEST),
                "mirror": str(MIRROR),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
