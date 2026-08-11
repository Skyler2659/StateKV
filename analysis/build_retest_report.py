#!/usr/bin/env python
"""Build the no-gate retest report across all retest tracks.

Reads whichever retest artifacts exist and emits
``docs/evidence/statekv_retest_report.md``.  The report contains only continuous
effect sizes (point estimates, paired bootstrap CIs, win/tie/loss) — no
pass/fail verdicts.  Sections preserve the original phase grouping so each
stage's content stays visible.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPOSITORY_ROOT / "results" / "temporal_cache_discovery"
OUTPUT = REPOSITORY_ROOT / "analysis" / "statekv_retest_report.md"

TRACK_A = RESULTS / "statekv_retest_replay_era1_n24_v1"
TRACK_B = RESULTS / "statekv_retest_freegen_qwen3_8b_n20_v1"
TRACK_D = RESULTS / "statekv_retest_vjp_rademacher_replication_v1"

NON_INFERIORITY = {"mean_delta_nll": 0.01, "mean_govreport_rouge_l": -0.5}


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    return ("%." + str(digits) + "f") % float(value)


def _track_a_section(lines: List[str]) -> None:
    metrics = TRACK_A / "metrics.csv"
    if not metrics.exists():
        lines.append("_Track A has not run yet._")
        return
    frame = pd.read_csv(metrics)
    frame = frame.sort_values("mean_exact_kl")
    lines.extend(
        [
            "Era-1 teacher-forced replay (Qwen2.5-1.5B, shared mask, budget "
            "128/core 92, 24 fresh sequences at offsets 106-117, anchors "
            "16/32/48).  This re-screens the P7/P9/P13/P14/P15 policy "
            "families with no selection gate.",
            "",
            "| policy | mean KL ↓ | P95 KL ↓ | CVaR95 ↓ | KL≥1 rate ↓ | "
            "mean ΔNLL ↓ | seq win vs attn |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in frame.itertuples(index=False):
        lines.append(
            "| %s | %s | %s | %s | %s | %s | %s |"
            % (
                row.policy,
                _fmt(row.mean_exact_kl, 4),
                _fmt(row.p95_exact_kl, 3),
                _fmt(row.cvar95_exact_kl, 3),
                _fmt(row.large_loss_rate, 4),
                _fmt(row.mean_delta_nll, 4),
                _fmt(row.sequence_win_rate_vs_baseline, 3),
            )
        )
    lines.append("")


def _track_b_section(lines: List[str]) -> None:
    aggregates = TRACK_B / "policy_aggregates.csv"
    if not aggregates.exists():
        lines.append("_Track B has not run yet._")
        return
    frame = pd.read_csv(aggregates)
    frame = frame.sort_values("mean_official_score", ascending=False)
    lines.extend(
        [
            "Era-2 recoverable free generation (Qwen3-8B-4bit, budget "
            "256/core 220, 20 fresh sequences at offsets 118-127, 64 "
            "tokens, horizon 1).  Primary endpoint: task scores; KL and "
            "ΔNLL are diagnostics.  Non-inferiority reference lines: "
            "ΔNLL +%s, ROUGE-L %s."
            % (NON_INFERIORITY["mean_delta_nll"], NON_INFERIORITY["mean_govreport_rouge_l"]),
            "",
            "| policy | n | official ↑ | GovReport ROUGE-L ↑ | NIAH ↑ | "
            "mean KL ↓ | mean ΔNLL ↓ |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in frame.itertuples(index=False):
        lines.append(
            "| %s | %d | %s | %s | %s | %s | %s |"
            % (
                row.policy,
                int(row.samples),
                _fmt(row.mean_official_score),
                _fmt(row.mean_govreport_rouge_l, 3),
                _fmt(row.mean_niah_retrieval, 3),
                _fmt(row.mean_trajectory_exact_kl, 6),
                _fmt(row.mean_delta_nll, 5),
            )
        )
    lines.append("")
    comparisons = TRACK_B / "paired_comparisons.csv"
    if comparisons.exists():
        paired = pd.read_csv(comparisons)
        for reference in ("attention", "qk_pool"):
            current = paired.loc[
                (paired["baseline"] == reference)
                & (paired["metric"] == "official_score")
                & (paired["task_bucket"] == "all")
            ].sort_values("mean_delta_policy_minus_baseline", ascending=False)
            if current.empty:
                continue
            lines.extend(
                [
                    "Paired official-score deltas vs **%s**:" % reference,
                    "",
                    "| policy | Δ | CI95 | wins/ties/losses |",
                    "|---|---:|---|---|",
                ]
            )
            for row in current.itertuples(index=False):
                lines.append(
                    "| %s | %+.4f | [%+.4f, %+.4f] | %d/%d/%d |"
                    % (
                        row.policy,
                        row.mean_delta_policy_minus_baseline,
                        row.delta_ci95_low,
                        row.delta_ci95_high,
                        int(row.wins),
                        int(row.ties),
                        int(row.losses),
                    )
                )
            lines.append("")


def _track_d_section(lines: List[str]) -> None:
    metrics = TRACK_D / "metrics.csv"
    if not metrics.exists():
        lines.append("_Track D has not run yet._")
        return
    frame = pd.read_csv(metrics)
    frame = frame.loc[
        (frame["width"] == 8) & (frame["refresh_interval"] == 4)
    ]
    agg = (
        frame.groupby("method")[
            [
                "median_spearman_gain",
                "mean_pairwise_accuracy_gain",
                "mean_normalized_regret_gain",
            ]
        ]
        .mean()
        .sort_values("mean_normalized_regret_gain", ascending=False)
        .reset_index()
    )
    lines.extend(
        [
            "Rademacher VJP independent-sample replication (8 untouched "
            "sequences, offsets 14-17, from the frozen P3 source run).  "
            "Gains vs the `hidden_l2_action` baseline at width 8 / "
            "refresh 4, averaged over both tasks.  The original P3 post-hoc "
            "variant had normalized regret 0.1945 → 0.0696 (gain +0.125) "
            "with pairwise accuracy -0.0026 on the development sequences.",
            "",
            "| method | Δ median spearman | Δ pairwise acc | Δ normalized regret |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in agg.itertuples(index=False):
        lines.append(
            "| %s | %+.4f | %+.4f | %+.4f |"
            % (
                row.method,
                row.median_spearman_gain,
                row.mean_pairwise_accuracy_gain,
                row.mean_normalized_regret_gain,
            )
        )
    lines.append("")


def main() -> None:
    lines: List[str] = [
        "# StateKV gate-retest report (no verdicts)",
        "",
        "Date: 2026-08-10.  Re-tests of policies that earlier phases "
        "rejected through preregistered joint gates, under the no-hard-gate "
        "contract: task scores are the primary endpoint, KL/NLL are "
        "diagnostics, and every number is reported continuously (point "
        "estimate, paired bootstrap CI, win/tie/loss).  Fresh sample "
        "offsets throughout (Track A 106-117, Track B 118-127, Track D "
        "14-17); recoverable backing-pool semantics in Track B.",
        "",
        "## Track A — Era-1 replay policy families (P7–P15)",
        "",
    ]
    _track_a_section(lines)
    lines.extend(["## Track B — Era-2 recoverable freegen panel (P16–P32, QKV-tier)", ""])
    _track_b_section(lines)
    lines.extend(["## Track D — Rademacher VJP replication (P3)", ""])
    _track_d_section(lines)
    lines.extend(
        [
            "## Standing caveats",
            "",
            "- Era-1 and Era-2 substrates are not comparable (different "
            "model, mask semantics, budget); cross-era rows are never "
            "averaged.",
            "- Bootstrap CIs at these sample sizes are wide; treat "
            "reference-line crossings as descriptive, not adjudicated.",
            "- Raw artifacts: `%s`, `%s`, `%s`."
            % (TRACK_A.name, TRACK_B.name, TRACK_D.name),
            "",
        ]
    )
    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    sys.exit(main())
