#!/usr/bin/env python3
"""Reactivation Index analysis over collected timeline artifacts.

Development/validation tool for the counterfactual phase. Computes
per-sequence RI summaries for a split directory and aggregates by task
family. Two dormancy rules are reported side by side (parameter selection
happens on train/validation only, per preregistration):

- rule A (quantile streak): the rows immediately before a top-K entry must
  all have rank fraction >= dormant_rank_quantile for >= L rows
  (statekv/reactivation_timeline.compute_timeline_reactivation).
- rule B (top-k gap): the number of consecutive non-top-K rows immediately
  before the entry must be >= L. Threshold-free except (top_k, L); robust
  to reactivation ramps that rule A misses.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from statekv.reactivation_timeline import (
    TimelineReactivationParams,
    compute_timeline_reactivation,
    load_artifact,
    timeline_importance,
)


def rule_b_events(
    artifact, top_k: int, dormant_window_rows: int, min_row: int = 1
) -> dict:
    """Dormancy = consecutive non-top-K rows immediately before the entry."""
    importance, lengths, n_blocks = timeline_importance(artifact)
    n_rows, n_positions = importance.shape
    in_top_k = np.zeros((n_rows, n_positions), dtype=bool)
    for row in range(n_rows):
        active = int(lengths[row])
        if active == 0:
            continue
        order = np.argsort(-importance[row, :active], kind="stable")
        in_top_k[row, order[: min(top_k, active)]] = True

    entries = 0
    events = []
    for position in range(n_positions):
        first_row = int(np.searchsorted(lengths, position + 1))
        previous = -1
        for row in range(max(first_row, min_row), n_rows):
            if not in_top_k[row, position]:
                continue
            if row > 0 and in_top_k[row - 1, position]:
                previous = row
                continue  # continuation
            entries += 1
            gap = row - previous - 1 if previous >= 0 else row - first_row
            if previous < 0 and first_row > 0:
                gap = row - first_row + 1  # never important since cache entry
            if gap >= dormant_window_rows and row - first_row >= dormant_window_rows:
                events.append(
                    {
                        "position": position,
                        "row": row,
                        "event_type": "I" if row < n_blocks else "II",
                        "dormancy_duration": gap,
                    }
                )
            previous = row
    return {
        "entries": entries,
        "events": events,
        "ri_count": len(events),
        "ri_fraction": (len(events) / entries) if entries else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timelines", required=True, type=Path, nargs="+")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--dormant-window-rows", type=int, default=8)
    parser.add_argument("--dormant-rank-quantile", type=float, default=0.1)
    parser.add_argument("--grid", action="store_true",
                        help="parameter sensitivity grid (train only)")
    args = parser.parse_args()

    paths = []
    for root in args.timelines:
        paths.extend(sorted(root.glob("*.npz")))
    if not paths:
        raise SystemExit(f"no timeline artifacts under {args.timelines}")

    def summarize(params: TimelineReactivationParams) -> pd.DataFrame:
        rows = []
        for path in paths:
            artifact = load_artifact(str(path))
            result = compute_timeline_reactivation(artifact, params)
            rule_b = rule_b_events(
                artifact, params.top_k, params.dormant_window_rows
            )
            type_i = sum(1 for e in result.events if e.event_type == "I")
            type_ii = sum(1 for e in result.events if e.event_type == "II")
            rows.append({
                "sample_id": result.sample_id,
                "task": result.task,
                "ri_fraction_a": result.ri_fraction,
                "ri_count_a": result.n_reactivation_events,
                "type_i_a": type_i,
                "type_ii_a": type_ii,
                "entries_a": result.n_entry_events,
                "ri_fraction_b": rule_b["ri_fraction"],
                "ri_count_b": rule_b["ri_count"],
                "type_i_b": sum(1 for e in rule_b["events"] if e["event_type"] == "I"),
                "type_ii_b": sum(1 for e in rule_b["events"] if e["event_type"] == "II"),
                "entries_b": rule_b["entries"],
            })
        return pd.DataFrame(rows)

    if args.grid:
        grid_rows = []
        for top_k in (5, 10, 20):
            for window in (4, 8, 16):
                for quantile in (0.05, 0.1, 0.25):
                    params = TimelineReactivationParams(
                        top_k=top_k,
                        dormant_window_rows=window,
                        dormant_rank_quantile=quantile,
                    )
                    frame = summarize(params)
                    by_task = frame.groupby("task")[
                        ["ri_fraction_a", "ri_fraction_b"]
                    ].mean()
                    for task, row in by_task.iterrows():
                        grid_rows.append({
                            "top_k": top_k, "window": window,
                            "quantile": quantile, "task": task,
                            "ri_fraction_a": row["ri_fraction_a"],
                            "ri_fraction_b": row["ri_fraction_b"],
                        })
        grid = pd.DataFrame(grid_rows)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        grid.to_csv(args.out, index=False)
        print(grid.to_string())
        return

    params = TimelineReactivationParams(
        top_k=args.top_k,
        dormant_window_rows=args.dormant_window_rows,
        dormant_rank_quantile=args.dormant_rank_quantile,
    )
    frame = summarize(params)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    agg = frame.groupby("task").agg(
        n=("sample_id", "count"),
        ri_a_mean=("ri_fraction_a", "mean"),
        ri_a_median=("ri_fraction_a", "median"),
        type_i_a=("type_i_a", "sum"),
        type_ii_a=("type_ii_a", "sum"),
        ri_b_mean=("ri_fraction_b", "mean"),
        ri_b_median=("ri_fraction_b", "median"),
        type_i_b=("type_i_b", "sum"),
        type_ii_b=("type_ii_b", "sum"),
    )
    print(agg.to_string())
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
