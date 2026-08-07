#!/usr/bin/env python
"""Run conditionally authorized free-generation external validation."""
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
from statekv.output_sensitivity_freegen import (
    OutputSensitivityFreeGenerationRunner,
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
    payload = OutputSensitivityFreeGenerationRunner(
        cfg, REPOSITORY_ROOT, run_dir
    ).run()
    output = run_dir / "free_generation_results.json"
    atomic_json(output, payload)
    gate["free_generation_executed"] = True
    gate["free_generation_gate"] = payload["external_validity_gate"]
    gate["stage_d_executed"] = True
    atomic_json(run_dir / "output_sensitivity_gate_decision.json", gate)
    status_path = run_dir / "status.json"
    if status_path.exists():
        status = json.load(status_path.open())
        status["state"] = "complete"
        status["stage_d_complete"] = True
        atomic_json(status_path, status)
    print(output)


if __name__ == "__main__":
    main()
