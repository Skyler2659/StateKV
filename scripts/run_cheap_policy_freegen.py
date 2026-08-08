#!/usr/bin/env python
"""Run A1--B3 cheap-controller closed-loop generation."""
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

from statekv.cheap_policy_freegen import run_cheap_policy_freegen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/stages/cheap_policy_freegen_qwen3_8b_n10_protocol.yaml",
    )
    args = parser.parse_args()
    output = run_cheap_policy_freegen(
        Path(args.config).resolve(), REPOSITORY_ROOT
    )
    print(output)


if __name__ == "__main__":
    main()
