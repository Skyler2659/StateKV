"""One-file-per-figure matplotlib diagnostics for temporal discovery."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _finish(path: Path, xlabel: str, ylabel: str, title: str) -> None:
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _line_by_strategy(
    frame: pd.DataFrame,
    x: str,
    y: str,
    path: Path,
    xlabel: str,
    ylabel: str,
    title: str,
) -> None:
    plt.figure(figsize=(7.2, 4.8))
    for strategy, group in frame.dropna(subset=[x, y]).groupby("strategy"):
        values = group.groupby(x, as_index=False)[y].mean().sort_values(x)
        plt.plot(values[x], values[y], marker="o", label=strategy)
    if not frame.empty:
        plt.legend(fontsize=8)
    _finish(path, xlabel, ylabel, title)


def generate_plots(run_dir: Path) -> Dict[str, str]:
    run_dir = Path(run_dir)
    figure_dir = run_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    horizon = pd.read_parquet(run_dir / "horizon_losses.parquet")
    step = pd.read_parquet(run_dir / "step_losses.parquet")
    candidate = pd.read_parquet(run_dir / "candidate_sets.parquet")
    temporal = pd.read_parquet(run_dir / "temporal_signals.parquet")
    horizon = horizon[horizon.get("valid", False) == True]
    step = step[step.get("valid", False) == True]
    outputs: Dict[str, str] = {}

    path = figure_dir / "01_loss_vs_horizon.png"
    _line_by_strategy(
        horizon, "horizon", "avg_delta_nll", path, "Horizon", "Average ΔNLL",
        "Loss versus horizon",
    )
    outputs["loss_vs_horizon"] = str(path)

    for number, column, name, ylabel in [
        (2, "cumulative_delta_nll", "cumulative_delta_nll_vs_step", "Cumulative ΔNLL"),
        (3, "approx_kl", "approximate_kl_vs_step", "Approximate KL"),
        (4, "attention_output_error_mean", "attention_output_error_vs_step", "Attention-output relative error"),
    ]:
        path = figure_dir / ("%02d_%s.png" % (number, name))
        _line_by_strategy(
            step, "future_step", column, path, "Future step", ylabel,
            ylabel + " versus future step",
        )
        outputs[name] = str(path)

    path = figure_dir / "05_strategy_horizon_loss_heatmap.png"
    matrix = horizon.pivot_table(
        index="strategy", columns="horizon", values="avg_delta_nll", aggfunc="mean"
    )
    plt.figure(figsize=(7.2, 4.8))
    if not matrix.empty:
        image = plt.imshow(matrix.to_numpy(), aspect="auto", interpolation="nearest")
        plt.colorbar(image, label="Average ΔNLL")
        plt.xticks(range(len(matrix.columns)), matrix.columns)
        plt.yticks(range(len(matrix.index)), matrix.index)
    _finish(path, "Horizon", "Strategy", "Strategy × horizon loss")
    outputs["strategy_horizon_loss_heatmap"] = str(path)

    oracle_pairs_path = run_dir / "oracle_horizon_overlap_pairs.parquet"
    oracle_pairs = (
        pd.read_parquet(oracle_pairs_path)
        if oracle_pairs_path.exists()
        else pd.DataFrame()
    )
    path = figure_dir / "06_oracle_set_overlap_across_horizons.png"
    plt.figure(figsize=(6.2, 5.2))
    if not oracle_pairs.empty:
        matrix = oracle_pairs.pivot_table(
            index="left_horizon",
            columns="right_horizon",
            values="jaccard",
            aggfunc="mean",
        )
        image = plt.imshow(matrix.to_numpy(), vmin=0, vmax=1, interpolation="nearest")
        plt.colorbar(image, label="Jaccard")
        plt.xticks(range(len(matrix.columns)), matrix.columns)
        plt.yticks(range(len(matrix.index)), matrix.index)
    _finish(path, "Oracle horizon", "Oracle horizon", "Oracle-set overlap")
    outputs["oracle_set_overlap"] = str(path)

    path = figure_dir / "07_deployable_oracle_overlap_vs_horizon.png"
    deployable = horizon[horizon["strategy"] != "future_attention_oracle"]
    _line_by_strategy(
        deployable, "horizon", "oracle_overlap", path, "Horizon", "Mean Jaccard",
        "Deployable-strategy overlap with future oracle",
    )
    outputs["deployable_oracle_overlap"] = str(path)

    score = temporal[temporal.get("signal_kind", "").eq("score_drift")]
    path = figure_dir / "08_score_rank_correlation_vs_lag.png"
    _line_by_strategy(
        score, "lag", "spearman_rank_correlation", path, "Lag",
        "Spearman correlation", "Selector score rank correlation versus lag",
    )
    outputs["score_rank_correlation"] = str(path)

    path = figure_dir / "09_top_core_overlap_vs_lag.png"
    _line_by_strategy(
        score, "lag", "top_core_jaccard", path, "Lag", "Jaccard",
        "Top-core overlap versus lag",
    )
    outputs["top_core_overlap"] = str(path)

    join_keys = ["run_id", "task", "sample_id", "anchor", "strategy"]
    query = temporal[temporal.get("signal_kind", "").eq("query_attention_drift")]
    query_loss = query.merge(
        step,
        left_on=join_keys + ["lag"],
        right_on=join_keys + ["future_step"],
        suffixes=("", "_loss"),
    )
    path = figure_dir / "10_query_drift_vs_future_loss.png"
    plt.figure(figsize=(6.4, 4.8))
    if not query_loss.empty:
        plt.scatter(
            1.0 - query_loss["query_cosine_to_anchor"],
            query_loss["delta_nll"],
            alpha=0.45,
        )
    _finish(path, "1 − query cosine to anchor", "ΔNLL", "Query drift versus future loss")
    outputs["query_drift_vs_loss"] = str(path)

    residual = temporal[
        temporal.get("signal_kind", "").eq("future_new_token_value_residual")
    ]
    residual = (
        residual.groupby(join_keys + ["lag"], as_index=False)[
            "future_new_token_residual"
        ].mean()
        if not residual.empty
        else residual
    )
    residual_loss = residual.merge(
        step,
        left_on=join_keys + ["lag"],
        right_on=join_keys + ["future_step"],
    ) if not residual.empty else pd.DataFrame()
    path = figure_dir / "11_new_token_value_residual_vs_future_loss.png"
    plt.figure(figsize=(6.4, 4.8))
    if not residual_loss.empty:
        plt.scatter(
            residual_loss["future_new_token_residual"],
            residual_loss["delta_nll"],
            alpha=0.45,
        )
    _finish(path, "New-token value residual", "ΔNLL", "New-token residual versus future loss")
    outputs["new_value_residual_vs_loss"] = str(path)

    validity_points = []
    for _, row in horizon.iterrows():
        try:
            observations = json.loads(row["validity_horizons"])
        except Exception:
            continue
        match = [
            item for item in observations
            if item["metric"] == "avg_delta_nll" and item["threshold"] == 0.1
        ]
        if match:
            validity_points.append(
                (float(row["avg_delta_nll"]), float(match[0]["observed_horizon"]))
            )
    path = figure_dir / "12_snapshot_loss_vs_validity_horizon.png"
    plt.figure(figsize=(6.4, 4.8))
    if validity_points:
        plt.scatter(
            [value[0] for value in validity_points],
            [value[1] for value in validity_points],
            alpha=0.55,
        )
    _finish(path, "Snapshot average ΔNLL", "Observed validity horizon", "Snapshot loss versus validity horizon")
    outputs["snapshot_loss_vs_validity"] = str(path)

    ages = []
    for raw in candidate[candidate.get("valid", False) == True].get("layers", []):
        for layer in json.loads(raw):
            ages.extend(
                record["age"]
                for record in layer.get("eligible_token_records", [])
                if record.get("cache_role") == "core"
            )
    path = figure_dir / "13_selected_token_age_distribution.png"
    plt.figure(figsize=(6.4, 4.8))
    if ages:
        plt.hist(ages, bins=min(40, max(5, int(np.sqrt(len(ages))))), alpha=0.8)
    _finish(path, "Token age at anchor", "Count", "Selected token age distribution")
    outputs["selected_token_age_distribution"] = str(path)

    geometry = temporal[temporal.get("signal_kind", "").eq("value_geometry")]
    geometry = (
        geometry.groupby(join_keys, as_index=False)["effective_rank"].mean()
        if not geometry.empty
        else geometry
    )
    geometry_loss = geometry.merge(horizon, on=join_keys) if not geometry.empty else pd.DataFrame()
    path = figure_dir / "14_value_effective_rank_vs_future_loss.png"
    plt.figure(figsize=(6.4, 4.8))
    if not geometry_loss.empty:
        plt.scatter(
            geometry_loss["effective_rank"],
            geometry_loss["avg_delta_nll"],
            alpha=0.5,
        )
    _finish(path, "Value effective rank", "Average ΔNLL", "Value effective rank versus future loss")
    outputs["effective_rank_vs_loss"] = str(path)

    with open(figure_dir / "figures.json", "w", encoding="utf-8") as handle:
        json.dump(outputs, handle, indent=2, sort_keys=True)
    return outputs
