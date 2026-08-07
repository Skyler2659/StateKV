#!/usr/bin/env python
"""Recompute descriptive statistics from an existing discovery run."""
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

from statekv.statistics import compute_descriptive_statistics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    args = parser.parse_args()
    output = compute_descriptive_statistics(
        Path(args.run_dir), args.seed, args.bootstrap_samples
    )
    print(Path(args.run_dir) / "descriptive_statistics.json")
    print("rank_reversal_total=%d" % output["rank_reversal_total"])


if __name__ == "__main__":
    main()
