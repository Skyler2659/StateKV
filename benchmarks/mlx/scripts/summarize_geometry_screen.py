#!/usr/bin/env python3
"""Summarize the paired 7B geometry screen against SnapKV."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return statistics.mean(values) if values else math.nan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_root", type=Path)
    parser.add_argument("--baseline", default="snapkv")
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for path in sorted(args.results_root.glob("*/samples/*.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"warning: skipped {path}: {exc}")
            continue
        row["task"] = "gov_report" if int(row["sample_idx"]) < 8 else "qmsum"
        rows.append(row)

    baseline = {
        int(row["sample_idx"]): row
        for row in rows
        if row.get("method") == args.baseline
    }
    if len(baseline) != 16:
        raise SystemExit(f"Expected 16 {args.baseline} rows, found {len(baseline)}")

    header = (
        "method\ttask\tn\tmean_rouge\tmedian_rouge\tpaired_delta"
        "\twins/ties/losses\taggregate_ppl\tmean_ppl\tgeneration_s"
        "\ttotal_s\tscore_s\tppl_score_s\ttokens_per_s"
    )
    print(header)
    methods = sorted({str(row["method"]) for row in rows})
    for method in methods:
        for task, lower, upper in (("gov_report", 0, 8), ("qmsum", 8, 16)):
            group = sorted(
                (
                    row
                    for row in rows
                    if row.get("method") == method
                    and lower <= int(row["sample_idx"]) < upper
                ),
                key=lambda row: int(row["sample_idx"]),
            )
            if not group:
                continue
            scores = [float(row["primary_score"]) for row in group]
            deltas = [
                float(row["primary_score"])
                - float(baseline[int(row["sample_idx"])]["primary_score"])
                for row in group
            ]
            wins = sum(delta > 1e-9 for delta in deltas)
            ties = sum(abs(delta) <= 1e-9 for delta in deltas)
            losses = sum(delta < -1e-9 for delta in deltas)
            nlls = [float(row["mean_nll"]) for row in group if row.get("mean_nll") is not None]
            aggregate_ppl = math.exp(statistics.mean(nlls)) if nlls else math.nan
            print(
                f"{method}\t{task}\t{len(group)}\t"
                f"{statistics.mean(scores):.4f}\t{statistics.median(scores):.4f}\t"
                f"{statistics.mean(deltas):+.4f}\t{wins}/{ties}/{losses}\t"
                f"{aggregate_ppl:.3f}\t{_mean(group, 'ppl'):.3f}\t"
                f"{_mean(group, 'generation_time_s'):.2f}\t{_mean(group, 'total_time_s'):.2f}\t"
                f"{_mean(group, 'score_time_s'):.3f}\t{_mean(group, 'ppl_score_time_s'):.3f}\t"
                f"{_mean(group, 'tokens_per_second'):.2f}"
            )

    print(f"\nloaded_rows\t{len(rows)}")


if __name__ == "__main__":
    main()
