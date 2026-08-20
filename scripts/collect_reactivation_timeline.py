#!/usr/bin/env python3
"""Collect full-causal-timeline attention trajectories for reactivation RI."""
from __future__ import annotations

import argparse
from pathlib import Path

from statekv.reactivation_timeline import collect_reactivation_timeline_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", action="append", dest="splits")
    parser.add_argument("--sample-id", action="append", dest="sample_ids")
    parser.add_argument("--cycle-limit", type=int)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = collect_reactivation_timeline_dataset(
        root / args.config,
        root,
        splits=args.splits,
        sample_ids=args.sample_ids,
        cycle_limit=args.cycle_limit,
    )
    print(output)


if __name__ == "__main__":
    main()
