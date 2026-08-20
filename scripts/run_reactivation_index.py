#!/usr/bin/env python3
"""Compute Reactivation Index over collected full-cache artifacts.

Development/validation tool: runs the RI computation over a split directory
of collected .npz artifacts and writes per-sequence RI summaries to CSV.
Parameters must come from the frozen protocol for any test use.
"""

import argparse
from pathlib import Path

import pandas as pd

from statekv.reactivation_index import (
    ReactivationParams,
    compute_sequence_reactivation,
    load_artifact,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--dormant-window", type=int, default=16)
    parser.add_argument("--dormant-rank-quantile", type=float, default=0.1)
    parser.add_argument("--min-cycle", type=int, default=1)
    args = parser.parse_args()

    params = ReactivationParams(
        top_k=args.top_k,
        dormant_window=args.dormant_window,
        dormant_rank_quantile=args.dormant_rank_quantile,
        min_cycle=args.min_cycle,
    )
    rows = []
    for path in sorted(args.artifacts.glob("*.npz")):
        artifact = load_artifact(str(path))
        rows.append(compute_sequence_reactivation(artifact, params).summary())
        print(f"[ri] {path.name}: done")
    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(frame.groupby("task")["ri_fraction"].describe())
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
