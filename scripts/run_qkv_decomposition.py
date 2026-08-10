#!/usr/bin/env python
"""Build the QK-V decomposition dataset (discovery protocol Phase 1)."""
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

from statekv.qkv_decomposition import run_qkv_decomposition


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/stages/statekv_qkv_decomposition_qwen3_8b.yaml",
    )
    args = parser.parse_args()
    output = run_qkv_decomposition(
        Path(args.config).resolve(), REPOSITORY_ROOT
    )
    print(output)


if __name__ == "__main__":
    main()
