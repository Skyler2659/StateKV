#!/usr/bin/env python3
"""Write stratified machine summaries and the preregistered figure set."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "experiments/local_truncated_jacobian"
RAW = EXPERIMENT / "raw"
FIGURES = EXPERIMENT / "figures"
SUMMARIES = EXPERIMENT / "summaries"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def save_figure(figure: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        FIGURES / name,
        dpi=180,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)


def task_label(value: str) -> str:
    return "GovReport" if value == "gov_report" else "NIAH"


def write_summaries(
    radius: pd.DataFrame,
    boundary: pd.DataFrame,
    direct: pd.DataFrame,
    l1_vector: pd.DataFrame,
    l1_ranking: pd.DataFrame,
    depth_vector: pd.DataFrame,
    depth_ranking: pd.DataFrame,
    l3_group: pd.DataFrame,
) -> None:
    SUMMARIES.mkdir(parents=True, exist_ok=True)
    radius.groupby("radius").agg(
        row_count=("radius", "size"),
        median_jvp_fd_cosine=("jvp_fd_cosine", "median"),
        median_jvp_fd_relative_l2=("jvp_fd_relative_l2", "median"),
        median_fd_norm=("fd_norm", "median"),
        median_noise_norm=("noise_norm", "median"),
        minimum_fd_to_noise_ratio=("fd_to_noise_ratio", "min"),
    ).reset_index().to_parquet(
        SUMMARIES / "radius_aggregate.parquet", index=False
    )
    boundary.groupby(["task", "layer", "anchor"]).agg(
        row_count=("physical_manual_cosine", "size"),
        median_cosine=("physical_manual_cosine", "median"),
        minimum_cosine=("physical_manual_cosine", "min"),
        median_relative_l2=("physical_manual_relative_l2", "median"),
        maximum_relative_l2=("physical_manual_relative_l2", "max"),
    ).reset_index().to_parquet(
        SUMMARIES / "l0_boundary_stratified.parquet", index=False
    )
    direct.groupby(
        ["task", "layer", "anchor", "candidate_source"]
    ).agg(
        row_count=("theory_phys_cosine", "size"),
        median_theory_phys_cosine=("theory_phys_cosine", "median"),
        median_theory_phys_relative_l2=(
            "theory_phys_relative_l2",
            "median",
        ),
        median_retained_mass=("retained_mass_mean", "median"),
        median_deleted_mass=("deleted_mass_mean", "median"),
    ).reset_index().to_parquet(
        SUMMARIES / "l0_direct_stratified.parquet", index=False
    )
    l1_vector.groupby(
        ["sample_id", "task", "direction", "scale"]
    ).agg(
        row_count=("vector_cosine", "size"),
        median_vector_cosine=("vector_cosine", "median"),
        median_vector_relative_l2=("vector_relative_l2", "median"),
        median_norm_ratio=("vector_symmetric_norm_ratio", "median"),
    ).reset_index().to_parquet(
        SUMMARIES / "l1_sequence_vector_aggregate.parquet",
        index=False,
    )
    l1_ranking.groupby(
        ["sample_id", "task", "direction", "scale"]
    ).agg(
        group_count=("energy_spearman", "size"),
        median_energy_spearman=("energy_spearman", "median"),
        median_pairwise_accuracy=(
            "pairwise_sign_accuracy",
            "median",
        ),
        median_top1_recall=("top1_recall", "median"),
        median_top3_recall=("top3_recall", "median"),
        median_regret=("mean_normalized_regret", "median"),
    ).reset_index().to_parquet(
        SUMMARIES / "l1_sequence_ranking_aggregate.parquet",
        index=False,
    )
    depth_vector.groupby(["sample_id", "task", "depth"]).agg(
        row_count=("vector_cosine", "size"),
        median_vector_cosine=("vector_cosine", "median"),
        median_vector_relative_l2=("vector_relative_l2", "median"),
        median_jvp_seconds=("jvp_seconds", "median"),
        median_relative_forward_cost=(
            "relative_forward_cost",
            "median",
        ),
        median_peak_memory_bytes=("mlx_peak_memory_bytes", "median"),
    ).reset_index().to_parquet(
        SUMMARIES / "l2_sequence_vector_aggregate.parquet",
        index=False,
    )
    depth_ranking.groupby(["sample_id", "task", "depth"]).agg(
        group_count=("energy_spearman", "size"),
        median_energy_spearman=("energy_spearman", "median"),
        median_pairwise_accuracy=(
            "pairwise_sign_accuracy",
            "median",
        ),
        median_regret=("mean_normalized_regret", "median"),
    ).reset_index().to_parquet(
        SUMMARIES / "l2_sequence_ranking_aggregate.parquet",
        index=False,
    )
    l3_group.groupby(["sample_id", "task"]).median(
        numeric_only=True
    ).reset_index().to_parquet(
        SUMMARIES / "l3_sequence_ranking_aggregate.parquet",
        index=False,
    )


def plot_radius(radius: pd.DataFrame) -> None:
    aggregate = radius.groupby("radius").agg(
        cosine=("jvp_fd_cosine", "median"),
        relative=("jvp_fd_relative_l2", "median"),
    )
    figure, axis = plt.subplots(figsize=(6.2, 3.8))
    axis.semilogx(
        aggregate.index,
        aggregate["cosine"],
        marker="o",
        label="Median JVP/FD cosine",
    )
    axis.axvline(3.0e-6, color="#c44e52", linestyle="--", label="Frozen radius")
    axis.axhline(0.995, color="#777777", linestyle=":", label="Eligibility")
    axis.set(xlabel="Relative FD radius", ylabel="Cosine", ylim=(0.996, 1.0002))
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    save_figure(figure, "01_jvp_fd_cosine_vs_radius.png")

    aggregate_noise = radius.groupby("radius").agg(
        fd_norm=("fd_norm", "median"),
        noise_norm=("noise_norm", "median"),
    )
    floor = np.maximum(aggregate_noise["noise_norm"], 1.0e-12)
    figure, axis = plt.subplots(figsize=(6.2, 3.8))
    axis.loglog(
        aggregate_noise.index,
        aggregate_noise["fd_norm"],
        marker="o",
        label="Median FD derivative norm",
    )
    axis.loglog(
        aggregate_noise.index,
        100.0 * floor,
        linestyle="--",
        label="100× noise floor",
    )
    axis.axvline(3.0e-6, color="#c44e52", linestyle=":")
    axis.set(xlabel="Relative FD radius", ylabel="L2 norm")
    axis.grid(alpha=0.25, which="both")
    axis.legend(frameon=False)
    save_figure(figure, "02_fd_norm_vs_noise_floor.png")


def plot_boundary(boundary: pd.DataFrame) -> None:
    aggregate = (
        boundary.groupby(["task", "layer"])[
            "physical_manual_cosine"
        ]
        .median()
        .unstack(0)
    )
    figure, axis = plt.subplots(figsize=(6.4, 3.8))
    for task in aggregate.columns:
        axis.plot(
            aggregate.index,
            aggregate[task],
            marker="o",
            label=task_label(task),
        )
    axis.axhline(0.99, color="#777777", linestyle=":", label="Layer gate")
    axis.set(
        xlabel="Layer",
        ylabel="Median physical/manual cosine",
        ylim=(0.989, 1.0003),
    )
    axis.set_xticks(sorted(boundary["layer"].unique()))
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    save_figure(figure, "03_native_boundary_by_task_layer.png")


def plot_l1(
    vectors: pd.DataFrame, rankings: pd.DataFrame
) -> None:
    primary = vectors[vectors["direction"].eq("u_phys")]
    aggregate = (
        primary.groupby(["task", "scale"])["vector_cosine"]
        .median()
        .unstack(0)
    )
    figure, axis = plt.subplots(figsize=(6.4, 3.8))
    for task in aggregate.columns:
        axis.plot(
            aggregate.index,
            aggregate[task],
            marker="o",
            label=task_label(task),
        )
    axis.axhline(0.95, color="#777777", linestyle=":", label="Natural gate")
    axis.set(
        xscale="log",
        xlabel="Natural direction scale η",
        ylabel="Median JVP/nonlinear cosine",
        ylim=(0.94, 1.002),
    )
    axis.set_xticks([0.125, 0.25, 0.5, 1.0, 2.0])
    axis.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    save_figure(figure, "04_l1_cosine_by_scale.png")

    natural = primary[primary["scale"].eq(1.0)]
    figure, axis = plt.subplots(figsize=(5.2, 4.6))
    for task, group in natural.groupby("task"):
        axis.scatter(
            group["predicted_energy"],
            group["true_energy"],
            s=12,
            alpha=0.55,
            label=task_label(task),
        )
    lower = max(
        min(
            natural["predicted_energy"].min(),
            natural["true_energy"].min(),
        ),
        1.0e-5,
    )
    upper = max(
        natural["predicted_energy"].max(),
        natural["true_energy"].max(),
    )
    axis.plot([lower, upper], [lower, upper], color="#777777", linestyle=":")
    axis.set(
        xscale="log",
        yscale="log",
        xlabel="Predicted energy ||J₁u||²",
        ylabel="FP32 nonlinear energy",
    )
    axis.grid(alpha=0.2, which="both")
    axis.legend(frameon=False)
    save_figure(figure, "05_predicted_vs_true_energy.png")


def plot_depth(
    vectors: pd.DataFrame, rankings: pd.DataFrame
) -> None:
    aggregate = (
        rankings.groupby(["task", "depth"])["energy_spearman"]
        .median()
        .unstack(0)
    )
    figure, axis = plt.subplots(figsize=(6.4, 3.8))
    for task in aggregate.columns:
        axis.plot(
            aggregate.index,
            aggregate[task],
            marker="o",
            label=task_label(task),
        )
    axis.set(
        xlabel="Truncation depth k",
        ylabel="Median candidate-energy Spearman",
        ylim=(0.94, 1.005),
    )
    axis.set_xticks([0, 1, 2, 4])
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    save_figure(figure, "06_depth_vs_ranking_accuracy.png")

    aggregate_cost = vectors.groupby("depth").agg(
        latency=("jvp_seconds", "median"),
        memory=("mlx_peak_memory_bytes", "median"),
        cost=("relative_forward_cost", "median"),
    )
    figure, axes = plt.subplots(1, 2, figsize=(8.4, 3.7))
    axes[0].plot(
        aggregate_cost.index,
        aggregate_cost["cost"],
        marker="o",
    )
    axes[0].set(
        xlabel="Truncation depth k",
        ylabel="Median cost / one-token forward",
    )
    axes[0].set_xticks([0, 1, 2, 4])
    axes[0].grid(alpha=0.25)
    axes[1].plot(
        aggregate_cost.index,
        aggregate_cost["memory"] / (1024**3),
        marker="s",
        color="#c44e52",
    )
    axes[1].set(
        xlabel="Truncation depth k",
        ylabel="Median MLX peak memory (GiB)",
    )
    axes[1].set_xticks([0, 1, 2, 4])
    axes[1].grid(alpha=0.25)
    save_figure(figure, "07_depth_vs_latency_memory.png")

    deltas = aggregate.subtract(aggregate.loc[0], axis=1)
    figure, axis = plt.subplots(figsize=(6.4, 3.8))
    for task in deltas.columns:
        axis.plot(
            deltas.index,
            deltas[task],
            marker="o",
            label=task_label(task),
        )
    axis.axhline(0.0, color="#777777", linestyle=":")
    axis.set(
        xlabel="Truncation depth k",
        ylabel="Δρ relative to k=0",
    )
    axis.set_xticks([0, 1, 2, 4])
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    save_figure(figure, "08_taskwise_delta_rho_by_depth.png")


def plot_direct(direct: pd.DataFrame) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(8.5, 3.8), sharey=True)
    for axis, column, label in (
        (
            axes[0],
            "retained_mass_mean",
            "Mean retained attention mass",
        ),
        (
            axes[1],
            "deleted_mass_mean",
            "Mean deleted attention mass",
        ),
    ):
        for task, group in direct.groupby("task"):
            axis.scatter(
                group[column],
                group["theory_phys_relative_l2"],
                s=10,
                alpha=0.5,
                label=task_label(task),
            )
        axis.set(xlabel=label, yscale="log")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Theory/physical injection relative L2")
    axes[1].legend(frameon=False)
    save_figure(figure, "09_injection_error_vs_attention_mass.png")


def plot_l3(candidates: pd.DataFrame) -> None:
    figure, axis = plt.subplots(figsize=(5.4, 4.6))
    for task, group in candidates.groupby("task"):
        axis.scatter(
            group["strue1_physical_energy"],
            group["exact_kl_full_to_physical"],
            s=12,
            alpha=0.55,
            label=task_label(task),
        )
    axis.set(
        xscale="log",
        yscale="log",
        xlabel="True adjacent hidden energy",
        ylabel="Exact KL(full || physical)",
    )
    axis.grid(alpha=0.2, which="both")
    axis.legend(frameon=False)
    save_figure(figure, "10_adjacent_hidden_energy_vs_exact_kl.png")


def run() -> Dict[str, str]:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    radius = pd.read_parquet(
        RAW / "radius_calibration/radius_rows.parquet"
    )
    boundary = pd.read_parquet(
        RAW / "l0_boundary/formal/native_boundary_rows.parquet"
    )
    direct = pd.read_parquet(
        RAW / "l0_boundary/formal/direct_rows.parquet"
    )
    l1_vector = pd.read_parquet(
        RAW
        / "l1_local_linearization/formal/local_vector_rows.parquet"
    )
    l1_ranking = pd.read_parquet(
        RAW
        / "l1_local_linearization/formal/local_ranking_rows.parquet"
    )
    depth_vector = pd.read_parquet(
        RAW / "l2_depth_ablation/formal/depth_vector_rows.parquet"
    )
    depth_ranking = pd.read_parquet(
        RAW / "l2_depth_ablation/formal/depth_ranking_rows.parquet"
    )
    l3_candidates = pd.read_parquet(
        RAW / "l3_output_diagnostic/formal/candidate_rows.parquet"
    )
    l3_group = pd.read_parquet(
        RAW
        / "l3_output_diagnostic/formal/group_ranking_rows.parquet"
    )
    write_summaries(
        radius,
        boundary,
        direct,
        l1_vector,
        l1_ranking,
        depth_vector,
        depth_ranking,
        l3_group,
    )
    plot_radius(radius)
    plot_boundary(boundary)
    plot_l1(l1_vector, l1_ranking)
    plot_depth(depth_vector, depth_ranking)
    plot_direct(direct)
    plot_l3(l3_candidates)
    artifacts = sorted(
        list(FIGURES.glob("*.png")) + list(SUMMARIES.glob("*.parquet"))
    )
    checksums = {
        str(path.relative_to(EXPERIMENT)): sha256_file(path)
        for path in artifacts
    }
    with (SUMMARIES / "generated_artifact_checksums.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(checksums, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return checksums


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
