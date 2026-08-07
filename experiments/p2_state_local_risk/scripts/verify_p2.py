#!/usr/bin/env python3
"""Run all P0/P1/P2 tests and recheck source manifests."""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import yaml


ROOT = Path(__file__).resolve().parents[3]
P2 = ROOT / "experiments/p2_state_local_risk"
RESULTS = P2 / "results"
SCRIPT_DIR = P2 / "scripts"
for value in (ROOT, SCRIPT_DIR):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from p2_core import sha256_file  # noqa: E402
from statekv.repository_layout import (  # noqa: E402
    verify_repository_checksum,
)


def manifest_status(path: Path) -> Dict[str, Any]:
    manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
    checks = {}
    for relative, expected in manifest["checksums"].items():
        checks[str(relative)] = verify_repository_checksum(
            ROOT, relative, str(expected)
        )
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256_file(path),
        "outcome": manifest["outcome"],
        "entry_count": len(checks),
        "matched_entry_count": int(sum(checks.values())),
        "all_match": all(checks.values()),
    }


def main() -> None:
    command = [
        str(ROOT / ".venv/bin/python"),
        "-m",
        "pytest",
        "-q",
        str(ROOT / "tests/test_p0_v2.py"),
        str(ROOT / "tests/test_p1_state_conditioned.py"),
        str(ROOT / "tests/test_p2_state_local.py"),
    ]
    tests = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    match = re.search(
        r"(\d+) passed(?:, (\d+) skipped)?",
        tests.stdout,
    )
    passed_count = int(match.group(1)) if match else 0
    skipped_count = (
        int(match.group(2))
        if match and match.group(2) is not None
        else 0
    )
    manifests = {
        "p0": manifest_status(
            ROOT
            / "experiments/p0_v2_fixed_boundary/"
            "P0_V2_MANIFEST.yaml"
        ),
        "p1": manifest_status(
            ROOT
            / "experiments/p1_state_conditioned/"
            "P1_STATE_CONDITIONED_MANIFEST.yaml"
        ),
    }
    formula = json.loads(
        (RESULTS / "formula_render_audit.json").read_text(
            encoding="utf-8"
        )
    )
    result = {
        "passed": bool(
            tests.returncode == 0
            and skipped_count == 0
            and all(
                value["all_match"]
                for value in manifests.values()
            )
            and formula["passed"]
        ),
        "test_command": command,
        "test_exit_code": tests.returncode,
        "test_stdout": tests.stdout,
        "test_stderr": tests.stderr,
        "passed_test_count": passed_count,
        "skipped_test_count": skipped_count,
        "source_manifests": manifests,
        "formula_audit_passed": formula["passed"],
        "formula_warning_count": formula[
            "total_warning_count"
        ],
        "formula_raw_math_leftover_count": formula[
            "total_raw_math_leftover_count"
        ],
    }
    destination = RESULTS / "test_summary.json"
    destination.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    if not result["passed"]:
        raise SystemExit(
            json.dumps(result, ensure_ascii=False, indent=2)
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
