#!/usr/bin/env python
"""Analyze Stage A and write the immutable gate decision."""
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
from statekv.gauge_geometry_analysis import analyze_stage_a


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_discovery_config(args.config)
    run_dir = (
        REPOSITORY_ROOT
        / cfg.runtime.output_root
        / str(cfg.runtime.run_id)
    )
    print(analyze_stage_a(cfg, run_dir))


if __name__ == "__main__":
    main()
