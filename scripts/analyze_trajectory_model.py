#!/usr/bin/env python
"""Analyze a completed trajectory stochastic-model run."""
from __future__ import annotations

import argparse
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
from statekv.trajectory_analysis import run_trajectory_analysis


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    cfg = load_discovery_config(args.config)
    outputs = run_trajectory_analysis(
        Path(args.run_dir), cfg
    )
    for path in outputs.values():
        print(path)


if __name__ == "__main__":
    main()
