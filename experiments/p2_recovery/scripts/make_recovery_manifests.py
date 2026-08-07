#!/usr/bin/env python3
"""Create non-self-referential manifests for every Recovery iteration."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "experiments/p2_recovery"
ITERATIONS = {
    "r0_failure_map": "retrospective_failure_map",
    "r1_amplitude_trust_region": "D2",
    "r3_path_integrated_readout": "C",
    "r4_scalar_decision_risk": "A",
}
PRIOR_MANIFESTS = {
    "p0": ROOT
    / "experiments/p0_v2_fixed_boundary/P0_V2_MANIFEST.yaml",
    "p1": ROOT
    / "experiments/p1_state_conditioned/"
    "P1_STATE_CONDITIONED_MANIFEST.yaml",
    "p2": ROOT
    / "experiments/p2_state_local_risk/P2_STATE_LOCAL_MANIFEST.yaml",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_state() -> Dict[str, Any]:
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


def iteration_outcome(name: str, fallback: str) -> str:
    path = EXPERIMENT / name / "results/iteration_outcome.json"
    if not path.exists():
        return fallback
    return str(
        json.loads(path.read_text(encoding="utf-8"))["outcome"]
    )


def file_checksums(
    directory: Path, exclusions: set[Path]
) -> Dict[str, str]:
    return {
        str(path.relative_to(ROOT)): sha256(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.resolve() not in exclusions
    }


def write_iteration_manifest(name: str, fallback: str) -> None:
    directory = EXPERIMENT / name
    manifest_path = directory / f"{name.upper()}_MANIFEST.yaml"
    mirror_path = directory / "results/checksum_verification.json"
    exclusions = {manifest_path.resolve(), mirror_path.resolve()}
    checksums = file_checksums(directory, exclusions)
    verification = {
        path: sha256(ROOT / path) == digest
        for path, digest in checksums.items()
    }
    formula_path = directory / "results/formula_render_audit.json"
    formula = json.loads(formula_path.read_text(encoding="utf-8"))
    manifest = {
        "schema_version": 1,
        "program": "p2_recovery",
        "iteration": name,
        "outcome": iteration_outcome(name, fallback),
        "formula_render_audit": {
            "passed": formula["passed"],
            "document_count": formula["document_count"],
            "mathml_node_count": formula[
                "total_mathml_node_count"
            ],
            "warning_count": formula["total_warning_count"],
            "raw_math_leftover_count": formula[
                "total_raw_math_leftover_count"
            ],
        },
        "prior_manifest_sha256": {
            key: sha256(path)
            for key, path in PRIOR_MANIFESTS.items()
        },
        "checksum_entry_count": len(checksums),
        "checksum_verification_passed": all(
            verification.values()
        ),
        "checksum_exclusions": [
            str(path.relative_to(ROOT))
            for path in sorted(exclusions)
        ],
        "checksums": checksums,
    }
    manifest_path.write_text(
        yaml.safe_dump(
            manifest,
            allow_unicode=True,
            sort_keys=False,
            width=100,
        ),
        encoding="utf-8",
    )
    mirror_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "iteration": name,
                "passed": all(verification.values()),
                "entry_count": len(verification),
                "verification": verification,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def write_cumulative_manifest() -> None:
    manifest_path = EXPERIMENT / "P2_RECOVERY_MANIFEST.yaml"
    mirror_path = EXPERIMENT / "results/checksum_verification.json"
    iteration_manifests = {
        name: EXPERIMENT / name / f"{name.upper()}_MANIFEST.yaml"
        for name in ITERATIONS
    }
    exclusions = {
        manifest_path.resolve(),
        mirror_path.resolve(),
        *(
            path.resolve()
            for path in iteration_manifests.values()
        ),
        *(
            (
                EXPERIMENT
                / name
                / "results/checksum_verification.json"
            ).resolve()
            for name in ITERATIONS
        ),
    }
    checksums = file_checksums(EXPERIMENT, exclusions)
    root_files = [
        ROOT / "experiments/p2_recovery/docs/r0_failure_map.md",
        ROOT / "experiments/p2_recovery/docs/master_log.md",
        ROOT / "experiments/p2_recovery/docs/decision_tree.md",
        ROOT / "experiments/p2_recovery/docs/cumulative_results.md",
        ROOT / "configs/frozen/P2_RECOVERY_DATA_LEDGER.yaml",
        ROOT / "experiments/p2_recovery/docs/final_recommendation.md",
        ROOT / "experiments/p2_recovery/docs/results_summary.md",
        ROOT / "tests/test_p2_recovery.py",
    ]
    checksums.update(
        {
            str(path.relative_to(ROOT)): sha256(path)
            for path in root_files
        }
    )
    verification = {
        path: sha256(ROOT / path) == digest
        for path, digest in checksums.items()
    }
    outcome_path = EXPERIMENT / "results/cumulative_outcome.json"
    outcome = json.loads(
        outcome_path.read_text(encoding="utf-8")
    )
    formula = json.loads(
        (
            EXPERIMENT / "results/formula_render_audit.json"
        ).read_text(encoding="utf-8")
    )
    manifest = {
        "schema_version": 1,
        "program": "p2_recovery",
        "cumulative_outcome": outcome["outcome"],
        "p3_eligible": outcome["p3_eligible"],
        "iterations": {
            name: {
                "outcome": iteration_outcome(name, fallback),
                "manifest_path": str(
                    path.relative_to(ROOT)
                ),
                "manifest_sha256": sha256(path),
            }
            for (name, fallback), path in zip(
                ITERATIONS.items(), iteration_manifests.values()
            )
        },
        "prior_manifest_sha256": {
            key: sha256(path)
            for key, path in PRIOR_MANIFESTS.items()
        },
        "formula_render_audit": {
            "passed": formula["passed"],
            "document_count": formula["document_count"],
            "mathml_node_count": formula[
                "total_mathml_node_count"
            ],
            "warning_count": formula["total_warning_count"],
            "raw_math_leftover_count": formula[
                "total_raw_math_leftover_count"
            ],
        },
        "checksum_entry_count": len(checksums),
        "checksum_verification_passed": all(
            verification.values()
        ),
        "git": git_state(),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "pandas": pd.__version__,
        },
        "checksum_exclusions": [
            str(path.relative_to(ROOT))
            for path in sorted(exclusions)
        ],
        "checksums": checksums,
    }
    manifest_path.write_text(
        yaml.safe_dump(
            manifest,
            allow_unicode=True,
            sort_keys=False,
            width=100,
        ),
        encoding="utf-8",
    )
    mirror_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "program": "p2_recovery",
                "passed": all(verification.values()),
                "entry_count": len(verification),
                "verification": verification,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    for name, fallback in ITERATIONS.items():
        write_iteration_manifest(name, fallback)
    write_cumulative_manifest()
    print(
        json.dumps(
            {
                "iterations": list(ITERATIONS),
                "cumulative_manifest": str(
                    EXPERIMENT / "P2_RECOVERY_MANIFEST.yaml"
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
