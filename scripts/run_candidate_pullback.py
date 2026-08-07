#!/usr/bin/env python
"""Run gated final-boundary candidate-conditioned Fisher pullbacks."""
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

from statekv.candidate_pullback import CandidatePullbackRunner
from statekv.config import load_discovery_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_discovery_config(args.config)
    runner = CandidatePullbackRunner(cfg, REPOSITORY_ROOT)
    print(runner.run_pullback())


if __name__ == "__main__":
    main()
