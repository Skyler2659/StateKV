#!/usr/bin/env python3
"""Create the 23 requested standalone figures and their source data."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import cluster_bootstrap_statistic, ensure_directory, write_dual


SEED = 20260724
COLORS = {
    "snapkv": "#4477AA",
    "v_ridge_leverage": "#EE6677",
    "attention_weighted_v_ridge_leverage": "#228833",
    "future_attention_oracle": "#AA3377",
}
LABELS = {
    "snapkv": "SnapKV",
    "v_ridge_leverage": "V-ridge",
    "attention_weighted_v_ridge_leverage": "Attn-weighted V-ridge",
    "future_attention_oracle": "Future oracle (H-conditioned)",
}


class FigureWriter:
    def __init__(self, analysis_dir: Path):
        self.figure_dir = ensure_directory(Path(analysis_dir) / "figures")
        self.data_dir = ensure_directory(self.figure_dir / "data")
        self.records: list[dict] = []

    def save(self, number: int, slug: str, data: pd.DataFrame, note: str = "") -> None:
        stem = f"{number:02d}_{slug}"
        write_dual(data, self.data_dir / stem)
        path = self.figure_dir / f"{stem}.png"
        plt.tight_layout()
        plt.savefig(path, dpi=180, bbox_inches="tight")
        plt.close()
        self.records.append(
            {
                "figure": number,
                "slug": slug,
                "png": str(path),
                "data_parquet": str(self.data_dir / f"{stem}.parquet"),
                "data_csv": str(self.data_dir / f"{stem}.csv"),
                "availability": (
                    str(data["availability"].iloc[0])
                    if "availability" in data and len(data)
                    else "available"
                ),
                "note": note,
            }
        )

    def unavailable(self, number: int, slug: str, title: str, reason: str) -> None:
        data = pd.DataFrame([{"availability": "unavailable", "reason": reason}])
        plt.figure(figsize=(8.5, 4.8))
        plt.axis("off")
        plt.title(title, fontsize=13, pad=18)
        plt.text(
            0.5,
            0.5,
            "UNAVAILABLE FROM SAVED OUTPUTS\n\n" + reason,
            ha="center",
            va="center",
            wrap=True,
            fontsize=11,
        )
        self.save(number, slug, data, note=reason)

    def finish(self) -> None:
        write_dual(pd.DataFrame(self.records), self.figure_dir / "figure_manifest")


def _summary(
    frame: pd.DataFrame,
    groups: list[str],
    value: str,
    statistic: str = "mean",
    draws: int = 500,
) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    for key, group in frame.groupby(groups, dropna=False, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        point, low, high, n = cluster_bootstrap_statistic(
            group, "sample_cluster", value, statistic, rng, draws
        )
        rows.append(
            {
                **dict(zip(groups, key)),
                "metric": value,
                "estimate": point,
                "ci_low": low,
                "ci_high": high,
                "n_samples": n,
                "n_rows": len(group),
                "uncertainty": "95% sample-cluster bootstrap CI",
            }
        )
    return pd.DataFrame(rows)


def _line_by_strategy(data: pd.DataFrame, x: str, y_label: str, title: str) -> None:
    plt.figure(figsize=(8.4, 5.2))
    for strategy, group in data.groupby("strategy"):
        group = group.sort_values(x)
        color = COLORS.get(strategy)
        plt.plot(
            group[x],
            group["estimate"],
            marker="o",
            color=color,
            label=LABELS.get(strategy, strategy),
        )
        plt.fill_between(
            group[x],
            group["ci_low"],
            group["ci_high"],
            color=color,
            alpha=0.15,
        )
    plt.xlabel(x.replace("_", " ").title())
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(alpha=0.2)
    plt.legend(fontsize=8)


def _scatter_with_bins(
    raw: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    x_label: str,
    y_label: str,
) -> pd.DataFrame:
    clean = raw.dropna(subset=[x, y]).copy()
    clean["x_bin"] = pd.qcut(clean[x], 5, duplicates="drop")
    summary = (
        clean.groupby("x_bin", observed=True)
        .agg(
            x_mean=(x, "mean"),
            y_mean=(y, "mean"),
            y_sd=(y, "std"),
            n_rows=(y, "size"),
            n_samples=("sample_id", "nunique"),
        )
        .reset_index(drop=True)
    )
    summary["ci_low"] = summary["y_mean"] - 1.96 * summary["y_sd"] / np.sqrt(
        summary["n_samples"].clip(lower=1)
    )
    summary["ci_high"] = summary["y_mean"] + 1.96 * summary["y_sd"] / np.sqrt(
        summary["n_samples"].clip(lower=1)
    )
    plt.figure(figsize=(8.2, 5.2))
    sample = clean.sample(min(2500, len(clean)), random_state=SEED)
    plt.scatter(sample[x], sample[y], s=10, alpha=0.12, color="#777777")
    plt.plot(summary["x_mean"], summary["y_mean"], marker="o", color="#CC3311")
    plt.fill_between(
        summary["x_mean"], summary["ci_low"], summary["ci_high"], color="#CC3311", alpha=0.18
    )
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(alpha=0.2)
    raw_out = clean.copy()
    raw_out["x_bin"] = raw_out["x_bin"].astype(str)
    raw_out["record_type"] = "raw"
    summary["record_type"] = "binned_summary"
    return pd.concat([raw_out, summary], ignore_index=True, sort=False)


def build(analysis_dir: Path) -> None:
    table = Path(analysis_dir) / "tables"
    writer = FigureWriter(analysis_dir)
    horizon = pd.read_parquet(table / "per_horizon_metrics.parquet")
    step = pd.read_parquet(table / "per_step_metrics.parquet")
    refresh = pd.read_parquet(table / "refresh_benefit_analysis.parquet")
    validity = pd.read_parquet(table / "validity_horizon_sensitivity.parquet")
    rankings = pd.read_parquet(table / "selector_horizon_rankings.parquet")
    score = pd.read_parquet(table / "score_stability.parquet")
    sets = pd.read_parquet(table / "set_stability.parquet")
    oracle = pd.read_parquet(table / "future_oracle_horizon_overlap.parquet")
    residual = pd.read_parquet(table / "future_selected_core_residuals.parquet")
    geometry = pd.read_parquet(table / "geometry_anchor_summaries.parquet")
    spectrum = pd.read_parquet(table / "singular_spectrum.parquet")
    links = pd.read_parquet(table / "mechanism_links.parquet")
    events = pd.read_parquet(table / "direction_shift_events.parquet")

    # 1
    data = _summary(horizon, ["strategy", "horizon"], "avg_delta_nll")
    _line_by_strategy(
        data,
        "horizon",
        "Mean ΔNLL (nats/token)",
        "Stale-cache loss vs replay horizon (15 samples, 3 tasks)",
    )
    writer.save(1, "stale_cache_loss_vs_horizon", data)

    # 2
    global_refresh = refresh[refresh["record_scope"].eq("global_output")].copy()
    data = _summary(global_refresh, ["strategy", "refresh_lag"], "refresh_benefit")
    _line_by_strategy(
        data,
        "refresh_lag",
        "Stale − refreshed ΔNLL (nats/token)",
        "Sparse refresh benefit at saved cross-anchor lags (15 samples)",
    )
    writer.save(
        2,
        "refresh_benefit_vs_horizon",
        data,
        note="x-axis is refresh lag; dense horizon counterfactual is unavailable",
    )

    # 3
    chosen = validity[
        validity["definition"].eq("absolute_average_delta_nll")
        & validity["threshold"].eq(0.1)
    ]
    data = _summary(chosen, ["strategy"], "observed_horizon", statistic="median")
    plt.figure(figsize=(8.4, 5.2))
    x = np.arange(len(data))
    plt.bar(
        x,
        data["estimate"],
        yerr=[data["estimate"] - data["ci_low"], data["ci_high"] - data["estimate"]],
        color=[COLORS.get(s, "#777777") for s in data["strategy"]],
        capsize=4,
    )
    plt.xticks(x, [LABELS.get(s, s) for s in data["strategy"]], rotation=20, ha="right")
    plt.ylabel("Observed validity horizon (steps)")
    plt.title("Empirical validity distribution (avg ΔNLL ≤ 0.1; 15 samples)")
    plt.grid(axis="y", alpha=0.2)
    writer.save(3, "empirical_validity_horizon_distribution", data)

    # 4
    data = rankings[rankings["task"].eq("ALL")].copy()
    plt.figure(figsize=(8.4, 5.2))
    for strategy, group in data.groupby("strategy"):
        plt.plot(
            group["horizon"],
            group["loss_rank"],
            marker="o",
            color=COLORS.get(strategy),
            label=LABELS.get(strategy, strategy),
        )
    plt.gca().invert_yaxis()
    plt.yticks([1, 2, 3, 4])
    plt.xlabel("Replay horizon")
    plt.ylabel("Mean-loss rank (1 = lower)")
    plt.title("Selector rank vs horizon (15 samples; descriptive ranks)")
    plt.grid(alpha=0.2)
    plt.legend(fontsize=8)
    writer.save(
        4,
        "selector_rank_vs_horizon",
        data,
        note="Ranks have no CI; source means are based on 15 sample clusters",
    )

    # 5
    primary = step[
        step["analysis_primary"]
        & step["strategy"].eq("attention_weighted_v_ridge_leverage")
    ].copy()
    trajectories = (
        primary.groupby(
            ["sample_id", "sample_cluster", "task", "future_step"], as_index=False
        )["delta_nll"]
        .mean()
        .sort_values("future_step")
    )
    summary = _summary(trajectories, ["future_step"], "delta_nll")
    plt.figure(figsize=(8.8, 5.4))
    task_colors = dict(
        zip(sorted(trajectories["task"].unique()), ["#4477AA", "#228833", "#EE6677"])
    )
    for (sample, task), group in trajectories.groupby(["sample_id", "task"]):
        plt.plot(
            group["future_step"],
            group["delta_nll"],
            color=task_colors[task],
            alpha=0.22,
            linewidth=0.8,
        )
    plt.plot(summary["future_step"], summary["estimate"], color="black", linewidth=2, label="cluster mean")
    plt.fill_between(
        summary["future_step"], summary["ci_low"], summary["ci_high"], color="black", alpha=0.12
    )
    for task, color in task_colors.items():
        plt.plot([], [], color=color, label=task)
    plt.xlabel("Future step")
    plt.ylabel("ΔNLL (nats/token)")
    plt.title("Per-sample loss trajectories: attn-weighted V-ridge (15 samples)")
    plt.grid(alpha=0.2)
    plt.legend(fontsize=8)
    out_data = trajectories.copy()
    out_data["record_type"] = "sample_trajectory"
    summary["record_type"] = "cluster_summary"
    writer.save(5, "per_sample_loss_trajectories", pd.concat([out_data, summary], sort=False))

    # 6
    data = _summary(score, ["strategy", "lag"], "spearman_rank_correlation", statistic="median")
    _line_by_strategy(
        data,
        "lag",
        "Median Spearman rank correlation",
        "Old-token score rank correlation vs lag (15 samples; layers 0/14/27)",
    )
    writer.save(6, "score_rank_correlation_vs_lag", data)

    # 7
    data = _summary(sets, ["strategy", "lag"], "top_core_jaccard", statistic="median")
    _line_by_strategy(
        data,
        "lag",
        "Median selected-core Jaccard",
        "Old-token core Jaccard vs lag (15 samples; excludes sink/recent)",
    )
    writer.save(7, "core_jaccard_vs_lag", data)

    # 8
    raw = sets.rename(
        columns={"selection_boundary_margin_future": "selection_margin"}
    )
    data = _scatter_with_bins(
        raw,
        "selection_margin",
        "selected_core_turnover",
        "Selection margin vs old-token set turnover (15 samples)",
        "Future selection boundary margin",
        "Selected-core turnover",
    )
    writer.save(8, "selection_margin_vs_set_turnover", data)

    # 9
    oracle["horizon_pair"] = (
        oracle["horizon_left"].astype(str) + "→" + oracle["horizon_right"].astype(str)
    )
    data = _summary(oracle, ["horizon_pair"], "mean_jaccard", statistic="median")
    pair_order = {
        "1→4": 1,
        "1→16": 2,
        "1→64": 3,
        "4→16": 4,
        "4→64": 5,
        "16→64": 6,
    }
    data["pair_order"] = data["horizon_pair"].map(pair_order)
    data = data.sort_values("pair_order")
    plt.figure(figsize=(8.2, 5.1))
    x = np.arange(len(data))
    plt.errorbar(
        x,
        data["estimate"],
        yerr=[data["estimate"] - data["ci_low"], data["ci_high"] - data["estimate"]],
        fmt="o-",
        capsize=4,
        color="#AA3377",
    )
    plt.xticks(x, data["horizon_pair"])
    plt.ylabel("Median oracle-core Jaccard across layers")
    plt.xlabel("Oracle horizon pair")
    plt.title("Future-oracle content overlap across horizons (15 samples)")
    plt.grid(alpha=0.2)
    writer.save(9, "future_oracle_overlap_across_horizons", data)

    # 10
    data = _summary(
        residual, ["strategy", "lag"], "future_new_token_residual", statistic="median"
    )
    _line_by_strategy(
        data,
        "lag",
        "Median relative residual",
        "New-token residual to selected-core span (not full history; 15 samples)",
    )
    writer.save(10, "new_direction_residual_vs_decoding_step", data)

    # 11-13
    writer.unavailable(
        11,
        "online_leverage_vs_decoding_step",
        "Online ridge leverage of new tokens vs decoding step",
        "Future V vectors and the anchor Gram/factor were not persisted.",
    )
    writer.unavailable(
        12,
        "covariance_drift_vs_lag",
        "Covariance drift vs lag",
        "Time-resolved V matrices or sufficient covariance sketches were not persisted.",
    )
    writer.unavailable(
        13,
        "principal_angle_drift_vs_lag",
        "Principal-angle drift vs lag",
        "Anchor and future-window subspace bases were not persisted.",
    )

    # 14
    data = _summary(geometry, ["strategy", "anchor"], "effective_rank", statistic="median")
    _line_by_strategy(
        data,
        "anchor",
        "Median effective rank",
        "Selected-core effective rank at saved anchors (not step-resolved; 15 samples)",
    )
    writer.save(14, "effective_rank_vs_decoding_step", data)

    # 15
    data = (
        spectrum.groupby(["anchor", "singular_value_rank"], as_index=False)
        .agg(
            estimate=("singular_value", "median"),
            ci_low=("singular_value", lambda x: x.quantile(0.10)),
            ci_high=("singular_value", lambda x: x.quantile(0.90)),
            n_samples=("sample_id", "nunique"),
        )
    )
    plt.figure(figsize=(8.3, 5.2))
    for anchor, group in data.groupby("anchor"):
        plt.plot(
            group["singular_value_rank"],
            group["estimate"],
            marker="o",
            label=f"anchor {anchor}",
        )
        plt.fill_between(
            group["singular_value_rank"], group["ci_low"], group["ci_high"], alpha=0.12
        )
    plt.yscale("log")
    plt.xlabel("Singular-value rank")
    plt.ylabel("Selected-core singular value")
    plt.title("Leading selected-core spectrum at anchors (10–90% interval; 15 samples)")
    plt.grid(alpha=0.2)
    plt.legend()
    writer.save(15, "singular_spectrum_selected_anchors", data)

    # 16
    rel = links[
        links["relationship"].eq("selected-core residual -> sparse refresh benefit")
    ].copy()
    agg = rel.groupby(
        ["sample_id", "sample_cluster", "task", "anchor", "strategy", "lag"],
        as_index=False,
    ).agg(selected_core_residual=("x_value", "mean"), refresh_benefit=("y_value", "mean"))
    data = _scatter_with_bins(
        agg,
        "selected_core_residual",
        "refresh_benefit",
        "Selected-core residual vs sparse refresh benefit (15 samples)",
        "Mean residual to selected-core full-rank span",
        "Stale − refreshed ΔNLL",
    )
    writer.save(16, "new_direction_residual_vs_refresh_benefit", data)

    writer.unavailable(
        17,
        "online_leverage_vs_refresh_benefit",
        "Online leverage vs refresh benefit",
        "Online leverage cannot be recovered without future V vectors and anchor factors.",
    )
    writer.unavailable(
        18,
        "subspace_drift_vs_refresh_benefit",
        "Subspace drift vs refresh benefit",
        "Time-resolved subspaces were not persisted; no proxy is substituted.",
    )

    # 19
    rel = links[
        links["relationship"].eq("selected-set turnover -> sparse refresh benefit")
    ].copy()
    data = _scatter_with_bins(
        rel.rename(columns={"x_value": "turnover", "y_value": "benefit"}),
        "turnover",
        "benefit",
        "Old-token core turnover vs sparse refresh benefit (15 samples)",
        "Mean selected-core turnover",
        "Stale − refreshed ΔNLL",
    )
    writer.save(19, "set_turnover_vs_refresh_benefit", data)

    # 20
    rel = links[links["relationship"].eq("attention-output error -> delta NLL")].copy()
    agg = rel.groupby(
        ["sample_id", "sample_cluster", "task", "anchor", "strategy", "lag"],
        as_index=False,
    ).agg(attention_error=("x_value", "mean"), delta_nll=("y_value", "mean"))
    data = _scatter_with_bins(
        agg,
        "attention_error",
        "delta_nll",
        "Attention-output error vs ΔNLL (15 samples; six diagnostic heads)",
        "Mean diagnostic attention-output relative error",
        "ΔNLL (nats/token)",
    )
    writer.save(20, "attention_output_error_vs_delta_nll", data)

    # 21
    rel = links[
        links["relationship"].eq("selection margin -> empirical validity horizon")
    ].copy()
    data = _scatter_with_bins(
        rel.rename(columns={"x_value": "margin", "y_value": "validity"}),
        "margin",
        "validity",
        "Anchor selection margin vs empirical validity (threshold 0.1; 15 samples)",
        "Median anchor boundary margin",
        "Observed validity horizon",
    )
    writer.save(21, "selection_margin_vs_empirical_lifetime", data)

    # 22
    aligned = events.dropna(subset=["refresh_benefit"]).copy()
    aligned["sample_cluster"] = (
        aligned["task"].astype(str) + "::" + aligned["sample_id"].astype(str)
    )
    aligned = aligned.drop_duplicates(
        ["sample_id", "anchor", "strategy", "lag", "event_signal"]
    )
    data = _summary(aligned, ["event_signal"], "refresh_benefit")
    plt.figure(figsize=(9.0, 5.3))
    x = np.arange(len(data))
    plt.bar(
        x,
        data["estimate"],
        yerr=[data["estimate"] - data["ci_low"], data["ci_high"] - data["estimate"]],
        color="#CC6677",
        capsize=4,
    )
    plt.xticks(x, data["event_signal"], rotation=25, ha="right")
    plt.ylabel("Mean sparse refresh benefit")
    plt.title("Event-aligned benefit at saved refresh boundaries (15 samples)")
    plt.grid(axis="y", alpha=0.2)
    writer.save(
        22,
        "event_aligned_refresh_benefit_curve",
        data,
        note="No dense event-time curve is recoverable; bars are boundary-aligned point estimates",
    )

    writer.unavailable(
        23,
        "recent_window_exit_lag_vs_refresh_benefit",
        "Recent-window exit lag vs refresh benefit",
        "Event token identity and dense refreshed replays around the 32-token exit were not saved.",
    )
    writer.finish()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, default=Path("analysis"))
    args = parser.parse_args()
    build(args.analysis_dir)


if __name__ == "__main__":
    main()
