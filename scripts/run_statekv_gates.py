#!/usr/bin/env python3
"""Run the P1 mechanism, P2 pure-eviction, and P3 telemetry stages."""
from __future__ import annotations

import argparse
from pathlib import Path

from statekv.statekv_gate_runner import (
    run_calibration,
    run_ladder,
    run_p1,
    run_p2,
    run_p2_profile,
    run_p3,
    run_r2a,
    run_r2b,
    run_teacher_gate,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=(
            "calibration",
            "p1",
            "p2",
            "p2-profile",
            "p3",
            "r2a-labels",
            "r2b-gate",
            "teacher-gate",
            "ladder",
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stages/statekv_p1_p3_gates_qwen3_8b.yaml"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    runners = {
        "calibration": run_calibration,
        "p1": run_p1,
        "p2": run_p2,
        "p2-profile": run_p2_profile,
        "p3": run_p3,
        "r2a-labels": run_r2a,
        "r2b-gate": run_r2b,
        "teacher-gate": run_teacher_gate,
        "ladder": run_ladder,
    }
    output = runners[args.stage](args.config.resolve(), root)
    print(output, flush=True)


if __name__ == "__main__":
    main()
