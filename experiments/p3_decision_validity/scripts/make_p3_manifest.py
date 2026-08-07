#!/usr/bin/env python3
"""Create and verify the non-self-referential P3 manifest."""
from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from p3_core import atomic_json, sha256_file  # noqa: E402


EXPERIMENT = ROOT / "experiments/p3_decision_validity"
MANIFEST = EXPERIMENT / "P3_DECISION_VALIDITY_MANIFEST.yaml"
VERIFICATION = EXPERIMENT / "results/checksum_verification.json"
PRIOR = {
    "p0": ROOT
    / "experiments/p0_v2_fixed_boundary/P0_V2_MANIFEST.yaml",
    "p1": ROOT
    / "experiments/p1_state_conditioned/"
    "P1_STATE_CONDITIONED_MANIFEST.yaml",
    "p2": ROOT
    / "experiments/p2_state_local_risk/P2_STATE_LOCAL_MANIFEST.yaml",
    "p2_recovery": ROOT
    / "experiments/p2_recovery/P2_RECOVERY_MANIFEST.yaml",
}


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


def main() -> None:
    exclusions = {MANIFEST.resolve(), VERIFICATION.resolve()}
    files = [
        path
        for path in sorted(EXPERIMENT.rglob("*"))
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.resolve() not in exclusions
    ]
    files.append(ROOT / "tests/test_p3_decision_validity.py")
    checksums = {
        str(path.relative_to(ROOT)): sha256_file(path)
        for path in files
    }
    verification = {
        relative: (
            (ROOT / relative).exists()
            and sha256_file(ROOT / relative) == digest
        )
        for relative, digest in checksums.items()
    }
    formula = json.loads(
        (
            EXPERIMENT / "results/formula_render_audit.json"
        ).read_text(encoding="utf-8")
    )
    outcome = json.loads(
        (
            EXPERIMENT / "results/cumulative_outcome.json"
        ).read_text(encoding="utf-8")
    )
    tests = json.loads(
        (EXPERIMENT / "results/test_summary.json").read_text(
            encoding="utf-8"
        )
    )
    prior_expected = yaml.safe_load(
        (EXPERIMENT / "p3_config.yaml").read_text(encoding="utf-8")
    )["source"]
    prior_hashes = {
        key: sha256_file(path) for key, path in PRIOR.items()
    }
    prior_checks = {
        key: prior_hashes[key]
        == str(prior_expected[f"{key}_manifest_sha256"])
        for key in PRIOR
    }
    manifest = {
        "schema_version": 1,
        "program": "p3_decision_validity",
        "cumulative_outcome": outcome["outcome"],
        "iteration_count": outcome["iteration_count"],
        "p4_allowed": outcome["p4_allowed"],
        "prior_manifest_sha256": prior_hashes,
        "prior_manifests_unchanged": all(prior_checks.values()),
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
        "tests": tests,
        "checksum_entry_count": len(checksums),
        "checksum_verification_passed": all(verification.values()),
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
    MANIFEST.write_text(
        yaml.safe_dump(
            manifest,
            allow_unicode=True,
            sort_keys=False,
            width=100,
        ),
        encoding="utf-8",
    )
    atomic_json(
        VERIFICATION,
        {
            "schema_version": 1,
            "passed": all(verification.values()),
            "entry_count": len(verification),
            "verification": verification,
            "prior_manifest_checks": prior_checks,
        },
    )
    if (
        not manifest["checksum_verification_passed"]
        or not manifest["prior_manifests_unchanged"]
        or not formula["passed"]
        or not tests["passed"]
    ):
        raise RuntimeError("P3 manifest integrity failed")
    print(
        json.dumps(
            {
                "manifest": str(MANIFEST.relative_to(ROOT)),
                "entry_count": len(checksums),
                "passed": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
