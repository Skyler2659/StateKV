#!/usr/bin/env python
"""Run the conditionally authorized output-sensitivity refresh replay."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
IMPORT_ROOTS = (
    REPOSITORY_ROOT,
    REPOSITORY_ROOT / "benchmarks" / "torch",
    REPOSITORY_ROOT / "benchmarks" / "mlx",
)
for import_root in IMPORT_ROOTS:
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import pandas as pd

from statekv.config import load_discovery_config
from statekv.output_sensitivity_policy import (
    OutputSensitivityPolicyRunner,
    summarize_output_policy,
)
from statekv.trajectory_analysis import atomic_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    cfg = load_discovery_config(args.config)
    run_dir = Path(args.run_dir).resolve()
    gate = json.load(
        (run_dir / "output_sensitivity_gate_decision.json").open()
    )
    if not (
        gate["stage_b_partial_passed"]
        and gate["stage_c_partial_passed"]
    ):
        raise RuntimeError("Stage B/C partial gate did not authorize Stage D")
    path = OutputSensitivityPolicyRunner(
        cfg, REPOSITORY_ROOT, run_dir
    ).run()
    summary = summarize_output_policy(pd.read_parquet(path))
    atomic_json(run_dir / "refresh_lcb_policy_summary.json", summary)
    gate["policy_replay_executed"] = True
    gate["policy_gate_passed"] = bool(
        summary["policy_gate"][
            "both_tasks_nonworse_at_some_matched_actual_count"
        ]
        and summary["policy_gate"]["at_least_one_task_strictly_improves"]
        and summary["policy_gate"]["not_almost_always_max_trigger"]
    )
    atomic_json(run_dir / "output_sensitivity_gate_decision.json", gate)
    print(path)


if __name__ == "__main__":
    main()
