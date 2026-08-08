#!/usr/bin/env python
"""Run the matched StateKV-oracle versus fixed-policy closed-loop test."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
MLX_ROOT = REPOSITORY_ROOT / "benchmarks" / "mlx"
if str(MLX_ROOT) not in sys.path:
    sys.path.insert(0, str(MLX_ROOT))

from statekv.oracle_policy_comparison import run_oracle_policy_comparison


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/stages/oracle_policy_comparison_protocol.yaml",
    )
    args = parser.parse_args()
    output = run_oracle_policy_comparison(
        Path(args.config).resolve(), REPOSITORY_ROOT
    )
    print(output)


if __name__ == "__main__":
    main()
