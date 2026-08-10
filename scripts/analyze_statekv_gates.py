#!/usr/bin/env python3
"""Analyze completed StateKV P1-P3 gate runs without rerunning the model."""
from __future__ import annotations

import argparse
from pathlib import Path

from statekv.statekv_gate_analysis import (
    analyze_p1,
    analyze_p2,
    analyze_p3,
    analyze_r2b,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("p1", "p2", "p3", "r2b", "analyze-r2b", "all"))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stages/statekv_p1_p3_gates_qwen3_8b.yaml"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    functions = {
        "p1": analyze_p1,
        "p2": analyze_p2,
        "p3": analyze_p3,
        "r2b": analyze_r2b,
        "analyze-r2b": analyze_r2b,
    }
    stages = ("p1", "p2", "p3") if args.stage == "all" else (args.stage,)
    for stage in stages:
        print(functions[stage](args.config.resolve(), root), flush=True)


if __name__ == "__main__":
    main()
