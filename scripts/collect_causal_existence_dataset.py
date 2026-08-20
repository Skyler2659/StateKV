#!/usr/bin/env python3
"""Collect strictly causal state features and offline future-attention labels."""
from __future__ import annotations

import argparse
from pathlib import Path

from statekv.causal_existence import collect_causal_existence_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/statekv_existence/causal_existence_qwen3_8b.yaml",
    )
    parser.add_argument("--split", action="append", dest="splits")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--cycle-limit", type=int)
    parser.add_argument("--sample-prefix", action="append", dest="sample_prefixes")
    parser.add_argument("--sample-id", action="append", dest="sample_ids")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = collect_causal_existence_dataset(
        root / args.config,
        root,
        splits=args.splits,
        max_samples=args.max_samples,
        cycle_limit=args.cycle_limit,
        sample_prefixes=args.sample_prefixes,
        sample_ids=args.sample_ids,
    )
    print(output)


if __name__ == "__main__":
    main()
