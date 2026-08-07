#!/usr/bin/env python
"""Run nested output-sensitivity and decision-calibration analysis."""
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

from statekv.config import load_discovery_config
from statekv.output_sensitivity_analysis import (
    analyze_output_sensitivity,
)
from statekv.trajectory_analysis import atomic_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    cfg = load_discovery_config(args.config)
    run_dir = Path(args.run_dir)
    decision = analyze_output_sensitivity(cfg, run_dir)
    status_path = run_dir / "status.json"
    if status_path.exists():
        status = json.load(status_path.open())
        status["analysis_complete"] = True
        status["state"] = (
            "stage_d_pending"
            if decision["stage_d_requires_conditional_stateful_runner"]
            else "complete"
        )
        atomic_json(status_path, status)
    print(decision)


if __name__ == "__main__":
    main()
