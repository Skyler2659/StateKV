#!/usr/bin/env python
"""Offline error decomposition for refresh-trigger design.

Splits per-event teacher risk into a staleness gap (stale action vs fresh
action of the same selector) and a selector gap (fresh action vs panel
oracle), on existing P23b/P22/P24 experiment outputs. Read-only with
respect to result directories; writes compact CSVs to analysis/tables/.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPOSITORY_ROOT / "results" / "temporal_cache_discovery"
TABLES_DIR = REPOSITORY_ROOT / "analysis" / "tables"

ATTENTION_PROXY = "attention_mean_w1_shared"
JOIN_KEYS = ["sample_id", "anchor", "horizon"]

DATASETS = {
    "p23b": "statekv_risk_consistent_proxy_independent_p23b_v1",
    "p22": "statekv_risk_consistent_proxy_alignment_p22_v1",
    "p24": "statekv_risk_consistent_output_aware_proxy_p24_v1",
}

SCOPES = [
    ("p23b", None, "refresh_gap_decomposition_p23b.csv"),
    ("p22", ATTENTION_PROXY, "refresh_gap_decomposition_p22_attention.csv"),
    ("p22", "ALL", "refresh_gap_decomposition_p22_all_proxies.csv"),
    ("p24", ATTENTION_PROXY, "refresh_gap_decomposition_p24_attention.csv"),
    ("p24", "ALL", "refresh_gap_decomposition_p24_all_proxies.csv"),
]


def _fmt(value: float) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "nan"
    return f"{value:.4f}"


def build_event_table(dataset: str, proxy_scope: Optional[str]) -> pd.DataFrame:
    """Per refresh event: staleness/selector/total gaps plus panel context."""
    base = RESULTS_ROOT / DATASETS[dataset]
    refresh = pd.read_parquet(base / "refresh_regret_rows.parquet")
    cross = pd.read_parquet(base / "cross_action_rows.parquet")

    if proxy_scope == "ALL":
        events = refresh.copy()
    else:
        events = refresh[refresh["proxy"] == (proxy_scope or ATTENTION_PROXY)].copy()

    # The candidate panel (and therefore its teacher risks) is identical
    # across proxies; use the attention-proxy rows as the canonical panel.
    panel = cross[cross["proxy"] == ATTENTION_PROXY]
    oracle = (
        panel.groupby(JOIN_KEYS)["teacher_risk"]
        .agg(oracle_teacher_risk="min", panel_max_teacher_risk="max", n_candidates="count")
        .reset_index()
    )
    events = events.merge(oracle, on=JOIN_KEYS, how="left")
    if events["oracle_teacher_risk"].isna().any():
        missing = events[events["oracle_teacher_risk"].isna()]
        raise ValueError(f"{dataset}: {len(missing)} events without a matching candidate panel")

    events["staleness_gap"] = events["stale_teacher_risk"] - events["fresh_teacher_risk"]
    events["selector_gap"] = events["fresh_teacher_risk"] - events["oracle_teacher_risk"]
    events["total_gap"] = events["stale_teacher_risk"] - events["oracle_teacher_risk"]
    events["identity_residual"] = (
        events["total_gap"] - events["staleness_gap"] - events["selector_gap"]
    )
    events["staleness_positive"] = events["staleness_gap"] > 0
    events["staleness_exceeds_selector"] = events["staleness_gap"] > events["selector_gap"]
    events["stale_worse_than_worst_candidate"] = (
        events["stale_teacher_risk"] > events["panel_max_teacher_risk"]
    )
    return events


def verify_oracle(dataset: str, events: pd.DataFrame) -> pd.DataFrame:
    """Check panel-derived oracle against alignment_units.oracle_teacher_risk."""
    base = RESULTS_ROOT / DATASETS[dataset]
    units = pd.read_parquet(base / "alignment_units.parquet")
    merged = units.merge(
        events[JOIN_KEYS + ["proxy", "oracle_teacher_risk"]],
        on=JOIN_KEYS + ["proxy"],
        how="inner",
        suffixes=("_units", "_panel"),
    )
    diff = (merged["oracle_teacher_risk_units"] - merged["oracle_teacher_risk_panel"]).abs()
    return pd.DataFrame(
        {
            "dataset": [dataset],
            "units_compared": [len(merged)],
            "mismatches_gt_1e-9": [int((diff > 1e-9).sum())],
            "max_abs_diff": [float(diff.max()) if len(diff) else float("nan")],
        }
    )


def summarize(events: pd.DataFrame, group_cols: List[str], scope: str) -> pd.DataFrame:
    def _agg(frame: pd.DataFrame) -> pd.Series:
        return pd.Series(
            {
                "n_events": len(frame),
                "staleness_mean": frame["staleness_gap"].mean(),
                "staleness_median": frame["staleness_gap"].median(),
                "staleness_p90": frame["staleness_gap"].quantile(0.9),
                "selector_mean": frame["selector_gap"].mean(),
                "selector_median": frame["selector_gap"].median(),
                "selector_p90": frame["selector_gap"].quantile(0.9),
                "frac_staleness_positive": frame["staleness_positive"].mean(),
                "frac_staleness_exceeds_selector": frame["staleness_exceeds_selector"].mean(),
                "frac_stale_worse_than_worst": frame["stale_worse_than_worst_candidate"].mean(),
            }
        )

    if group_cols:
        rows = [
            {**dict(zip(group_cols, key if isinstance(key, tuple) else (key,))),
             **_agg(frame)}
            for key, frame in events.groupby(group_cols)
        ]
    else:
        rows = [_agg(events)]
    grouped = pd.DataFrame(rows)
    grouped.insert(0, "scope", scope)
    return grouped


def print_block(title: str, frame: pd.DataFrame, group_cols: List[str]) -> None:
    print(f"  [{title}]")
    header = (
        group_cols
        + ["n", "stale_mean", "stale_med", "stale_p90", "sel_mean", "sel_med",
           "sel_p90", "f>0", "f>sel", "f>worst"]
    )
    print("    " + " | ".join(f"{h:>9s}" for h in header))
    for _, row in frame.iterrows():
        cells = [str(row[c]) for c in group_cols]
        cells += [
            str(int(row["n_events"])),
            _fmt(row["staleness_mean"]),
            _fmt(row["staleness_median"]),
            _fmt(row["staleness_p90"]),
            _fmt(row["selector_mean"]),
            _fmt(row["selector_median"]),
            _fmt(row["selector_p90"]),
            _fmt(row["frac_staleness_positive"]),
            _fmt(row["frac_staleness_exceeds_selector"]),
            _fmt(row["frac_stale_worse_than_worst"]),
        ]
        print("    " + " | ".join(f"{c:>9s}" for c in cells))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=TABLES_DIR,
        help="directory for the compact CSV outputs",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    event_columns = [
        "sample_id", "task", "previous_anchor", "anchor", "horizon", "proxy",
        "staleness_gap", "selector_gap", "total_gap", "identity_residual",
        "proxy_regret", "fresh_teacher_risk", "stale_teacher_risk",
        "oracle_teacher_risk", "panel_max_teacher_risk",
        "fresh_action_id", "stale_action_id",
    ]
    summary_frames: List[pd.DataFrame] = []
    verification_frames: List[pd.DataFrame] = []

    for dataset, proxy_scope, csv_name in SCOPES:
        scope = f"{dataset}:{proxy_scope or 'attention'}"
        events = build_event_table(dataset, proxy_scope)
        events[event_columns].to_csv(args.output_dir / csv_name, index=False)

        verification_frames.append(verify_oracle(dataset, events))

        residual = events["identity_residual"].abs()
        pooled = summarize(events, [], scope)
        by_horizon = summarize(events, ["horizon"], scope)
        by_task = summarize(events, ["task"], scope)
        summary_frames += [pooled.assign(split="pooled"),
                           by_horizon.assign(split="by_horizon"),
                           by_task.assign(split="by_task")]

        rho, pval = spearmanr(events["staleness_gap"], events["selector_gap"])

        print(f"\n=== {scope} ({len(events)} events) ===")
        print(
            f"  identity check: max|residual|={residual.max():.3e} "
            f"mean|residual|={residual.mean():.3e}"
        )
        print(
            f"  spearman(staleness_gap, selector_gap) = {_fmt(rho)} "
            f"(p={pval:.3e})"
        )
        print_block("pooled", pooled, [])
        print_block("by horizon", by_horizon, ["horizon"])
        print_block("by task", by_task, ["task"])

        top = events.nlargest(5, "staleness_gap")[
            ["sample_id", "task", "previous_anchor", "anchor", "horizon",
             "staleness_gap", "selector_gap", "proxy_regret"]
        ].copy()
        regret_rank = events["proxy_regret"].rank(ascending=False)
        top["proxy_regret_rank_of_n"] = [
            f"{int(regret_rank[idx])}/{len(events)}" for idx in top.index
        ]
        print("  top-5 staleness-gap events:")
        for _, row in top.iterrows():
            print(
                f"    {row['sample_id']:>18s} {row['task']:>13s} "
                f"{int(row['previous_anchor'])}->{int(row['anchor'])} "
                f"h={int(row['horizon']):>2d} "
                f"staleness={_fmt(row['staleness_gap'])} "
                f"selector={_fmt(row['selector_gap'])} "
                f"proxy_regret={_fmt(row['proxy_regret'])} "
                f"(rank {row['proxy_regret_rank_of_n']})"
            )

    summary = pd.concat(summary_frames, ignore_index=True)
    summary.to_csv(args.output_dir / "refresh_gap_decomposition_summary.csv", index=False)
    verification = pd.concat(verification_frames, ignore_index=True)
    verification.to_csv(
        args.output_dir / "refresh_gap_decomposition_oracle_verification.csv", index=False
    )

    print("\n=== oracle verification (panel min vs alignment_units) ===")
    print(verification.to_string(index=False))
    print(f"\nCSV outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
