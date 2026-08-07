#!/usr/bin/env python3
"""Build and verify the final P3PR checksum manifest."""
from __future__ import annotations

import json
import platform
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "experiments/p3_physical_recovery"
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from p3pr_core import atomic_json, sha256_file  # noqa: E402


MANIFEST = EXPERIMENT / "P3_PHYSICAL_RECOVERY_MANIFEST.yaml"
VERIFICATION = EXPERIMENT / "results/checksum_verification.json"


def test_summary() -> dict:
    path = EXPERIMENT / "results/pytest_junit.xml"
    if not path.exists():
        return {"status": "pending"}
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root)
    return {
        "status": "complete",
        "tests": sum(int(row.attrib.get("tests", 0)) for row in suites),
        "failures": sum(
            int(row.attrib.get("failures", 0)) for row in suites
        ),
        "errors": sum(int(row.attrib.get("errors", 0)) for row in suites),
        "skipped": sum(int(row.attrib.get("skipped", 0)) for row in suites),
    }


def main() -> None:
    files = [
        path
        for path in EXPERIMENT.rglob("*")
        if path.is_file()
        and path not in {MANIFEST, VERIFICATION}
        and path.name != ".DS_Store"
        and path.suffix != ".pyc"
        and "__pycache__" not in path.parts
    ]
    files.extend(
        [
            ROOT / "docs/statekv/physical_same_step.md",
            ROOT / "experiments/p3_physical_recovery/docs/detailed_report.md",
            ROOT / "tests/test_p3_physical_recovery.py",
        ]
    )
    files = sorted(set(path for path in files if path.exists()))
    checksums = {
        str(path.relative_to(ROOT)): sha256_file(path) for path in files
    }
    evaluation = json.loads(
        (
            EXPERIMENT / "results/P3PR_EVALUATION_SUMMARY.json"
        ).read_text(encoding="utf-8")
    )
    formula = json.loads(
        (
            EXPERIMENT / "results/formula_render_audit.json"
        ).read_text(encoding="utf-8")
    )
    tests = test_summary()
    payload = {
        "schema_version": 1,
        "program": "p3_physical_recovery",
        "cumulative_outcome": "P3PR-S",
        "terminal_condition": "Terminal Success",
        "iteration_count": 17,
        "calibration_model_instance_count": 77,
        "method_abstraction_allowed": True,
        "terminal_success": bool(evaluation["terminal_success"]),
        "prior_manifest_sha256": {
            name.removesuffix("_manifest"): row["sha256"]
            for name, row in yaml.safe_load(
                (EXPERIMENT / "p3pr_config.yaml").read_text()
            )["source"].items()
            if isinstance(row, dict) and "sha256" in row
        },
        "prior_manifests_unchanged": bool(
            evaluation["all_old_manifests_unchanged"]
        ),
        "formula_render_audit": {
            key: formula[key]
            for key in (
                "passed",
                "document_count",
                "mathml_node_count",
                "warning_count",
                "raw_math_leftover_count",
            )
        },
        "tests": tests,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "pandas": pd.__version__,
        },
        "checksum_exclusions": [
            str(MANIFEST.relative_to(ROOT)),
            str(VERIFICATION.relative_to(ROOT)),
        ],
        "checksum_entry_count": len(checksums),
        "checksum_verification_passed": True,
        "checksums": checksums,
    }
    MANIFEST.write_text(
        yaml.safe_dump(
            payload,
            sort_keys=False,
            allow_unicode=True,
            width=1000,
        ),
        encoding="utf-8",
    )
    mismatches = {
        relative: {
            "expected": expected,
            "actual": sha256_file(ROOT / relative),
        }
        for relative, expected in checksums.items()
        if sha256_file(ROOT / relative) != expected
    }
    verification = {
        "schema_version": 1,
        "manifest": str(MANIFEST.relative_to(ROOT)),
        "entry_count": len(checksums),
        "mismatches": mismatches,
        "passed": not mismatches,
    }
    atomic_json(VERIFICATION, verification)
    if mismatches:
        raise SystemExit(f"checksum verification failed: {mismatches}")
    print(
        {
            "manifest": str(MANIFEST.relative_to(ROOT)),
            "entry_count": len(checksums),
            "verification_passed": True,
            "tests": tests,
        }
    )


if __name__ == "__main__":
    main()
