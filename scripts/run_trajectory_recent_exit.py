#!/usr/bin/env python
"""Append real recent-FIFO exit trajectories to a completed run."""
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
    print(
        TrajectoryModelRunner(
            config, REPOSITORY_ROOT
        ).run_recent_exit_extension()
    )


if __name__ == "__main__":
    main()
