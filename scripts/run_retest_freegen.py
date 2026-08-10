#!/usr/bin/env python
"""Run the no-gate multi-policy retrospective retest panel."""
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

from statekv.retest_freegen import run_retest_freegen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/stages/retest_freegen_qwen3_8b_n20_protocol.yaml",
    )
    args = parser.parse_args()
    output = run_retest_freegen(Path(args.config).resolve(), REPOSITORY_ROOT)
    print(output)


if __name__ == "__main__":
    main()
