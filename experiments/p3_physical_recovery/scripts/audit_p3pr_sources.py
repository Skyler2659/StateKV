#!/usr/bin/env python3
"""Create the immutable-source and P3 physical-target implementation audit."""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from p3pr_core import (  # noqa: E402
    atomic_json,
    discover_old_ids,
    sha256_file,
    source_integrity,
)


EXPERIMENT = ROOT / "experiments/p3_physical_recovery"


def function_source_lines(path: Path, function_name: str) -> tuple[int, int]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
            node.name == function_name
        ):
            return int(node.lineno), int(node.end_lineno or node.lineno)
    raise ValueError(f"{function_name} missing from {path}")


def main() -> None:
    config = yaml.safe_load(
        (EXPERIMENT / "p3pr_config.yaml").read_text(encoding="utf-8")
    )
    checks = source_integrity(config)
    p3_runner = (
        ROOT
        / "experiments/p3_decision_validity/scripts/run_p3_trajectory.py"
    )
    physical_lines = function_source_lines(p3_runner, "physical_trajectories")
    run_lines = function_source_lines(p3_runner, "run_sequence")
    result = {
        "schema_version": 1,
        "passed": all(checks.values()),
        "source_checks": checks,
        "source_hashes": {
            name: {
                "path": payload["path"],
                "expected": payload["sha256"],
                "actual": sha256_file(ROOT / payload["path"]),
            }
            for name, payload in config["source"].items()
            if isinstance(payload, dict) and "path" in payload
        },
        "automatic_old_id_scan": discover_old_ids(),
        "p3_target_code_audit": {
            "runner": str(p3_runner.relative_to(ROOT)),
            "physical_trajectories_lines": list(physical_lines),
            "run_sequence_lines": list(run_lines),
            "baseline": "full_replay_at_target",
            "candidate": (
                "tau_selected_all_layer_mask_propagated_by_teacher_forcing"
            ),
            "target": (
                "KL(full_reference_target_logits, propagated_candidate_logits)"
            ),
            "is_same_current_physical_state_clone_target": False,
            "retained_without_redefinition": True,
        },
        "p3pr_target": {
            "baseline": (
                "same_prequery_compressed_state_clone_without_current_eviction"
            ),
            "candidate": (
                "same_prequery_state_clone_with_one_shared_physical_position "
                "deleted_before_the_same_query"
            ),
            "target": (
                "KL(unpruned_current_baseline_logits, "
                "current_candidate_eviction_logits)"
            ),
            "compared_on_independent_new_ids": True,
            "does_not_replace_p3_target": True,
        },
    }
    atomic_json(EXPERIMENT / "results/source_target_audit.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

