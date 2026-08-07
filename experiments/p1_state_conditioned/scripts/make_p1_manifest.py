#!/usr/bin/env python3
"""Build the non-self-referential P1 checksum manifest."""
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
EXPERIMENT = ROOT / "experiments/p1_state_conditioned"
RESULTS = EXPERIMENT / "results"
MANIFEST = EXPERIMENT / "P1_STATE_CONDITIONED_MANIFEST.yaml"
MIRROR = RESULTS / "p1_checksums.json"
ROOT_FILES = [
    ROOT / "experiments/p1_state_conditioned/docs/code_audit.md",
    ROOT / "experiments/p1_state_conditioned/docs/experiment_plan.md",
    ROOT / "experiments/p1_state_conditioned/docs/calibration.md",
    ROOT / "experiments/p1_state_conditioned/docs/results.md",
    ROOT / "experiments/p1_state_conditioned/docs/failure_analysis.md",
    ROOT / "configs/frozen/p1_state_conditioned_config.yaml",
    ROOT / "tests/test_p1_state_conditioned.py",
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
        (RESULTS / "p1_gate_outcome.json").read_text(encoding="utf-8")
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
    protocol = yaml.safe_load(
        (ROOT / "configs/frozen/p1_state_conditioned_config.yaml").read_text(
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
    ranking = pd.read_parquet(RESULTS / "ranking_rows.parquet")
    manifest = {
        "schema_version": 1,
        "experiment": "p1_state_conditioned_fixed_boundary_risk_closure",
        "outcome": gate["outcome"],
        "seed": 20260727,
        "config": {
            "path": "p1_state_conditioned_config.yaml",
            "sha256_at_evaluation": metadata["config_sha256"],
            "sha256_current": sha256(
                ROOT / "configs/frozen/p1_state_conditioned_config.yaml"
            ),
            "fd_selected_relative_radius": calibration[
                "selected_relative_radius"
            ],
        },
        "data": {
            "calibration_ids": calibration_ids,
            "evaluation_ids": metadata["sequence_ids"],
            "sequence_count": len(metadata["sequence_ids"]),
            "response_row_count": len(response),
            "state_unit_count": len(state),
            "ranking_row_count": len(ranking),
        },
        "gates": {
            "gate0": gate["gate0_p0_regression"]["passed"],
            "gate1": gate["gate1_history_validity"]["passed"],
            "gate2": gate["gate2_combined_readout"]["passed"],
            "gate3": gate["gate3_decision_gain"]["passed"],
            "state_operating_point_diagnostic": gate[
                "state_operating_point_diagnostic"
            ]["passed"],
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
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
