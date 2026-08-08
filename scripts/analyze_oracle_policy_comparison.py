#!/usr/bin/env python
"""Consolidate StateKV teacher-versus-fixed-policy experiments."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from statekv.oracle_policy_comparison_analysis import (
    analyze_oracle_policy_comparison,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=(
            "results/temporal_cache_discovery/"
            "statekv_oracle_policy_freegen_independent_p30_v1/analysis"
        ),
    )
    args = parser.parse_args()
    output = analyze_oracle_policy_comparison(
        REPOSITORY_ROOT, Path(args.output_dir).resolve()
    )
    print(output)


if __name__ == "__main__":
    main()
