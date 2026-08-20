#!/usr/bin/env python3
"""Build the StateKV PK task-score table from strict closed-loop outputs.

Reads ``sample_summary.csv`` from one or more closed-loop run directories
(e.g. the synthetic and LongBench dev panels) and pivots task x policy per
budget on the official task score.  This is the primary deliverable table:
StateKV Student vs R2/SnapKV/H2O/fixed-EMA on real task scores, with KL kept
as a secondary diagnostic column in the long-format output.
"""
from pathlib import Path
import argparse

import pandas as pd


SCORE_COLUMN = "official_score"


def load_summaries(paths):
    frames = []
    for path in paths:
        summary = Path(path) / "sample_summary.csv"
        if not summary.exists():
            raise RuntimeError(f"missing closed-loop summary: {summary}")
        frame = pd.read_csv(summary)
        frame["source_run"] = str(Path(path))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", help="closed-loop split directories")
    parser.add_argument("--output", help="optional csv output path")
    args = parser.parse_args()

    frame = load_summaries(args.runs)
    frame["policy_budget"] = (
        frame["policy"] + " @ " + frame["budget"].astype(str)
    )
    pivot = frame.pivot_table(
        index="task",
        columns="policy_budget",
        values=SCORE_COLUMN,
        aggfunc="mean",
    ).round(2)
    # Column order: full cache reference first, then matched budgets.
    ordered = sorted(
        pivot.columns,
        key=lambda name: (name.split(" @ ")[1] != "0", name),
    )
    pivot = pivot[ordered]
    print(pivot.to_string())
    print()
    kl_pivot = frame.pivot_table(
        index="task",
        columns="policy_budget",
        values="mean_trajectory_exact_kl",
        aggfunc="mean",
    ).round(4)[ordered]
    print("mean trajectory exact KL (secondary diagnostic):")
    print(kl_pivot.to_string())
    print()
    counts = (
        frame.groupby(["task", "policy", "budget"], as_index=False)
        .size()
        .rename(columns={"size": "sequences"})
    )
    print("sequence counts:")
    print(counts.to_string(index=False))
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        pivot.to_csv(output)
        print(f"\nwrote {output}")


if __name__ == "__main__":
    main()
