#!/usr/bin/env python
"""Run controlled trajectory stochastic-model identification."""
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
from statekv.trajectory_model import TrajectoryModelRunner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_discovery_config(args.config)
    run_dir = TrajectoryModelRunner(config, REPOSITORY_ROOT).run()
    print(run_dir)


if __name__ == "__main__":
    main()
