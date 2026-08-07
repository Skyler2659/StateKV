#!/usr/bin/env python
"""Run robust perturbation-envelope trajectory collection."""
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
from statekv.robust_envelope import RobustEnvelopeRunner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_discovery_config(args.config)
    print(RobustEnvelopeRunner(cfg, REPOSITORY_ROOT).run())


if __name__ == "__main__":
    main()
