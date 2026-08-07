#!/usr/bin/env python
"""Run the opt-in temporal cache discovery experiment."""
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
from statekv.plotting import generate_plots
from statekv.runner import TemporalDiscoveryRunner
from statekv.statistics import compute_descriptive_statistics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="only generate raw fragments and Parquet tables",
    )
    args = parser.parse_args()
    config = load_discovery_config(args.config)
    repository_root = REPOSITORY_ROOT
    run_dir = TemporalDiscoveryRunner(config, repository_root).run()
    if not args.skip_analysis:
        compute_descriptive_statistics(
            run_dir,
            seed=config.runtime.seed,
            bootstrap_samples=config.runtime.bootstrap_samples,
        )
        generate_plots(run_dir)
    print(run_dir)


if __name__ == "__main__":
    main()
