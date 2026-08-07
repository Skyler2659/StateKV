#!/usr/bin/env python3
"""Rebuild P0 summaries and figures from immutable row-level artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]


def load_tables(raw_dir: Path) -> Dict[str, pd.DataFrame]:
    return {
        name: pd.read_parquet(raw_dir / f"{name}_rows.parquet")
        for name in (
            "candidate_registry",
            "deletion_identity",
            "projection_block",
            "single_layer",
            "jvp_fd",
            "anchor_audit",
        )
    }


def sequence_summary(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    identity = tables["deletion_identity"]
    fp32 = identity[identity["dtype"].eq("float32")]
    single = tables["single_layer"]
    jvp = tables["jvp_fd"]
    audit = tables["anchor_audit"]
    rows = []
    for (sample_id, task), group in audit.groupby(["sample_id", "task"]):
        fp = fp32[fp32["sample_id"].eq(sample_id)]
        sl = single[single["sample_id"].eq(sample_id)]
        jv = jvp[jvp["sample_id"].eq(sample_id)]
        rows.append(
            {
                "sample_id": sample_id,
                "task": task,
                "split": "train",
                "anchors": int(group["anchor"].nunique()),
                "candidate_groups": int(len(group)),
                "fp32_identity_max_relative_error": float(
                    fp["relative_error"].max()
                ),
                "fp32_identity_median_relative_error": float(
                    fp["relative_error"].median()
                ),
                "single_layer_cosine_median": float(
                    sl["physical_manual_cosine"].median()
                ),
                "jvp_fd_cosine_median": float(
                    jv["jvp_fd_cosine"].median()
                ),
                "jvp_fd_relative_l2_median": float(
                    jv["jvp_fd_relative_l2"].median()
                ),
                "base_repeat_max_absolute_error": float(
                    group["repeat_max_absolute_error"].max()
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_identity(
    identity: pd.DataFrame, output_path: Path
) -> None:
    fp32 = identity[identity["dtype"].eq("float32")].copy()
    rng = np.random.default_rng(20260726)
    if len(fp32) > 15000:
        fp32 = fp32.iloc[
            np.sort(rng.choice(len(fp32), size=15000, replace=False))
        ]
    x = np.maximum(fp32["direct_norm"].to_numpy(), 1.0e-30)
    y = np.maximum(fp32["relative_error"].to_numpy(), 1.0e-30)
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    for task, color in (
        ("gov_report", "#3b82f6"),
        ("niah_single_1", "#f97316"),
    ):
        mask = fp32["task"].eq(task).to_numpy()
        axis.scatter(
            x[mask],
            y[mask],
            s=7,
            alpha=0.22,
            linewidths=0,
            label=task,
            color=color,
        )
    axis.axhline(1.0e-6, color="black", linestyle="--", linewidth=1)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("FP32 direct-direction norm")
    axis.set_ylabel("Deletion-identity relative error")
    axis.set_title("P0 identity error is condition-number dependent")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_single_layer(
    single: pd.DataFrame, output_path: Path
) -> None:
    grouped = (
        single.groupby(["task", "layer"])["physical_manual_cosine"]
        .median()
        .unstack(0)
        .sort_index()
    )
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    grouped.plot(kind="bar", ax=axis, width=0.78)
    axis.axhline(0.999, color="black", linestyle="--", linewidth=1)
    axis.set_ylim(min(0.5, float(grouped.min().min()) - 0.05), 1.005)
    axis.set_xlabel("Intervention layer")
    axis.set_ylabel("Median final-logit cosine")
    axis.set_title("Single-layer physical vs manual boundary alignment")
    axis.legend(frameon=False, title=None)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_radius(
    diagnosis_path: Path,
    cosine_path: Path,
    error_path: Path,
) -> None:
    diagnosis = json.loads(diagnosis_path.read_text(encoding="utf-8"))
    frame = pd.DataFrame(diagnosis["radius_rows"])
    for metric, ylabel, path, threshold in (
        ("cosine", "JVP / symmetric-FD cosine", cosine_path, 0.99),
        (
            "relative_l2",
            "JVP / symmetric-FD relative L2",
            error_path,
            None,
        ),
    ):
        figure, axis = plt.subplots(figsize=(7.2, 4.6))
        for candidate_id, group in frame.groupby("candidate_id"):
            group = group.sort_values("radius")
            axis.plot(
                group["radius"],
                group[metric],
                marker="o",
                label=candidate_id,
            )
        if threshold is not None:
            axis.axhline(
                threshold, color="black", linestyle="--", linewidth=1
            )
        axis.axvline(
            1.0e-4,
            color="#dc2626",
            linestyle=":",
            linewidth=1.3,
            label="frozen P0 radius",
        )
        axis.set_xscale("log")
        axis.set_xlabel("Relative intervention radius")
        axis.set_ylabel(ylabel)
        axis.set_title("Native-4bit numerical radius diagnosis (synthetic)")
        axis.legend(frameon=False)
        figure.tight_layout()
        figure.savefig(path, dpi=180)
        plt.close(figure)


def run(raw_dir: Path, smoke_dir: Path, summary_dir: Path, figure_dir: Path) -> None:
    summary_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    tables = load_tables(raw_dir)
    sequence = sequence_summary(tables)
    sequence.to_csv(summary_dir / "p0_sequence_summary.csv", index=False)
    formal = json.loads(
        (raw_dir / "p0_formal_summary.json").read_text(encoding="utf-8")
    )
    compact = {
        "formal_p0": True,
        "formal_p0_passed": formal["formal_p0_passed"],
        "stop_p1_plus": formal["stop_p1_plus"],
        "heldout_touched": formal["heldout_touched"],
        "checks": formal["checks"],
        "metrics": formal["metrics"],
        "row_counts": formal["row_counts"],
        "sequence_ids": formal["sequence_ids"],
    }
    (summary_dir / "p0_gate_summary.json").write_text(
        json.dumps(compact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    plot_identity(
        tables["deletion_identity"],
        figure_dir / "p0_identity_conditioning.png",
    )
    plot_single_layer(
        tables["single_layer"],
        figure_dir / "p0_single_layer_alignment.png",
    )
    plot_radius(
        smoke_dir / "numeric_diagnosis.json",
        figure_dir / "p0_jvp_cosine_vs_radius.png",
        figure_dir / "p0_jvp_relative_error_vs_radius.png",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=ROOT
        / "experiments/predictive_closure/raw/p0_alignment"
        / "formal_4bit_retry1",
    )
    parser.add_argument(
        "--smoke-dir",
        type=Path,
        default=ROOT
        / "experiments/predictive_closure/raw/p0_alignment/smoke_4bit",
    )
    parser.add_argument(
        "--summary-dir",
        type=Path,
        default=ROOT / "experiments/predictive_closure/summaries",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=ROOT / "experiments/predictive_closure/figures",
    )
    args = parser.parse_args()
    run(args.raw_dir, args.smoke_dir, args.summary_dir, args.figure_dir)


if __name__ == "__main__":
    main()
