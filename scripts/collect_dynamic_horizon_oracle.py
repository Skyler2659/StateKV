#!/usr/bin/env python3
"""Collect matched per-KV-head trajectories for the dynamic-horizon oracle."""
from __future__ import annotations

import argparse
from pathlib import Path

from statekv.dynamic_horizon_oracle import collect_dynamic_horizon_trajectories


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/adaptive_temporal/dynamic_horizon_oracle_qwen3_8b.yaml",
    )
    arguments = parser.parse_args()
    output = collect_dynamic_horizon_trajectories(ROOT / arguments.config, ROOT)
    print(output)


if __name__ == "__main__":
    main()

