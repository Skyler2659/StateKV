#!/usr/bin/env python
"""Create presentation-ready static plots from robust-envelope artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "E1": "#7A7A7A",
    "E2": "#1769AA",
    "E3": "#C65D21",
    "baseline": "#8A8F98",
    "oracle": "#2E7D32",
}


def _json(path: Path):
    with path.open() as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve()
    figure_dir = run_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    coverage = pd.DataFrame(
        _json(run_dir / "envelope_coverage_summary.json")["primary_rows"]
    )
    tightness = pd.DataFrame(
        _json(run_dir / "envelope_tightness_summary.json")["summary"]
    )
    tightness = tightness[
        (tightness["coverage_level"] == 0.9)
        & (tightness["margin_type"] == "simultaneous")
        & (tightness["route"] == "empirical_nonnegative")
    ]
    subset = pd.DataFrame(
        _json(run_dir / "envelope_subset_ranking_summary.json")["summary"]
    )

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.7))
    horizons = [1, 2, 4, 8, 16, 32, 64]
    for family in ("E1", "E2", "E3"):
        current = (
            coverage[coverage["family"] == family]
            .groupby("horizon")["pointwise_coverage"]
            .min()
            .reindex(horizons)
        )
        axes[0].plot(
            horizons,
            current,
            marker="o",
            label=family,
            color=COLORS[family],
        )
    axes[0].axhline(0.9, color="#B33A3A", linestyle="--", linewidth=1)
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(horizons, [str(value) for value in horizons])
    axes[0].set_ylim(0.89, 1.005)
    axes[0].set_xlabel("Horizon")
    axes[0].set_ylabel("Worst-task pointwise coverage")
    axes[0].set_title("Held-out coverage")
    axes[0].legend(frameon=False)

    for family in ("E1", "E2", "E3"):
        current = (
            tightness[tightness["family"] == family]
            .groupby("horizon_offset")["median_looseness"]
            .median()
            .reindex(horizons)
        )
        axes[1].plot(
            horizons,
            current,
            marker="o",
            label=family,
            color=COLORS[family],
        )
    axes[1].axhline(5.0, color="#B33A3A", linestyle="--", linewidth=1)
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks(horizons, [str(value) for value in horizons])
    axes[1].set_xlabel("Horizon")
    axes[1].set_ylabel("Median bound / realized")
    axes[1].set_title("Envelope tightness")

    order = [
        "attention_objective",
        "aov_objective",
        "dynamic_direct",
        "E1_objective",
        "E2_objective",
        "E3_objective",
    ]
    labels = ["Attention", "AOV", "Direct", "E1", "E2", "E3"]
    current = subset.set_index("objective").reindex(order)
    colors = [
        COLORS["baseline"],
        COLORS["baseline"],
        COLORS["baseline"],
        COLORS["E1"],
        COLORS["E2"],
        COLORS["E3"],
    ]
    axes[2].bar(
        np.arange(len(order)),
        current["median_spearman_kl"],
        color=colors,
    )
    axes[2].axhline(0.0, color="#555555", linewidth=0.8)
    axes[2].set_xticks(np.arange(len(order)), labels, rotation=30, ha="right")
    axes[2].set_ylabel("Median sequence Spearman vs KL")
    axes[2].set_title("Physical-subset ranking")
    fig.tight_layout()
    fig.savefig(
        figure_dir / "robust_envelope_validity_and_ranking.png",
        bbox_inches="tight",
    )
    plt.close(fig)

    policy = pd.read_parquet(run_dir / "envelope_refresh_policy_rows.parquet")
    per_sequence = (
        policy.groupby(
            ["sample_id", "task", "policy", "requested_refresh_count"]
        )["exact_kl"]
        .sum()
        .reset_index()
    )
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7))
    baseline = float(
        per_sequence[
            (per_sequence["policy"] == "no_refresh_static")
            & (per_sequence["requested_refresh_count"] == 0)
        ]["exact_kl"].median()
    )
    for name, label, color in (
        ("aov_trigger", "AOV", COLORS["baseline"]),
        ("E2", "E2 envelope", COLORS["E2"]),
    ):
        current = (
            per_sequence[per_sequence["policy"] == name]
            .groupby("requested_refresh_count")["exact_kl"]
            .median()
        )
        x = [0] + sorted(int(value) for value in current.index)
        y = [baseline] + [float(current.loc[value]) for value in x[1:]]
        axes[0].plot(x, y, marker="o", label=label, color=color)
    threshold = per_sequence[
        per_sequence["policy"] == "E2_threshold"
    ]
    if not threshold.empty:
        actual = (
            policy[policy["policy"] == "E2_threshold"]
            .groupby("sample_id")["refreshes_completed"]
            .max()
            .median()
        )
        axes[0].scatter(
            [actual],
            [threshold["exact_kl"].median()],
            marker="D",
            s=55,
            color=COLORS["E3"],
            label="E2 threshold",
            zorder=4,
        )
    axes[0].set_xlabel("Refresh count")
    axes[0].set_ylabel("Median cumulative KL")
    axes[0].set_title("Quality–refresh trade-off")
    axes[0].legend(frameon=False)

    selected = per_sequence[
        per_sequence["requested_refresh_count"].isin([0, 3])
        & per_sequence["policy"].isin(
            [
                "no_refresh_static",
                "fixed_interval",
                "age_only",
                "aov_trigger",
                "aor_trigger",
                "direct_trigger",
                "E2",
                "stateful_oracle",
            ]
        )
    ]
    rows = []
    for task, group in selected.groupby("task"):
        baseline_value = (
            group[
                ~group["policy"].isin(["E2", "stateful_oracle"])
            ]
            .groupby("policy")["exact_kl"]
            .median()
            .min()
        )
        for policy_name, label in (
            ("baseline", "Best baseline"),
            ("E2", "E2"),
            ("stateful_oracle", "Oracle"),
        ):
            absolute = (
                baseline_value
                if policy_name == "baseline"
                else group[group["policy"] == policy_name]["exact_kl"].median()
            )
            value = 100.0 * (float(absolute) - float(baseline_value)) / float(
                baseline_value
            )
            rows.append((task, label, value))
    task_frame = pd.DataFrame(rows, columns=["task", "policy", "kl"])
    tasks = list(task_frame["task"].drop_duplicates())
    x = np.arange(len(tasks))
    width = 0.24
    for index, (label, color) in enumerate(
        (
            ("Best baseline", COLORS["baseline"]),
            ("E2", COLORS["E2"]),
            ("Oracle", COLORS["oracle"]),
        )
    ):
        values = [
            float(
                task_frame[
                    (task_frame["task"] == task)
                    & (task_frame["policy"] == label)
                ]["kl"].iloc[0]
            )
            for task in tasks
        ]
        axes[1].bar(
            x + (index - 1) * width,
            values,
            width,
            label=label,
            color=color,
        )
    axes[1].set_xticks(x, ["GovReport" if value == "gov_report" else "NIAH" for value in tasks])
    axes[1].axhline(0.0, color="#555555", linewidth=0.8)
    axes[1].set_ylabel("Cumulative KL vs best baseline (%)")
    axes[1].set_title("Matched 3-refresh policy delta")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(
        figure_dir / "robust_envelope_refresh_policy.png",
        bbox_inches="tight",
    )
    plt.close(fig)
    for path in sorted(figure_dir.glob("robust_envelope_*.png")):
        print(path)


if __name__ == "__main__":
    main()
