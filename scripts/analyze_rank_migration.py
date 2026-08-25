#!/usr/bin/env python3
"""Rank-migration mechanism analysis (pure numpy/pandas; no model).

Reads the per-arm diagnostic npz files written by
``scripts/run_rank_migration_diagnostic.py`` plus the fresh-run
``sample_summary.csv`` files (for Gain_s = official_score(R2) -
official_score(QK) at budget 256) and produces the rank-migration tables
under ``results/statekv_counterfactual/rank_migration_v1/``.

npz schema (per arm, see statekv/causal_closed_loop._rank_migration_payload):
``cycle_%04d_positions`` (int32), ``cycle_%04d_eligible`` (int32),
``cycle_%04d_current_shared`` (float32, aligned with positions),
``cycle_%04d_selected_core`` (int32), ``cycle_%04d_generated_token_id``
(int32 scalar), ``cycle_%04d_refreshed`` (bool scalar); refresh cycles add
``cycle_%04d_rollout_per_step`` (horizon x n_eligible float32, aligned with
eligible) and ``cycle_%04d_rollout_generated`` (int32). Arm scalars:
``budget``, ``sink_size``, ``recent_size``, ``core_budget``,
``refresh_frequency``, ``rollout_horizon``, ``needle_spans`` (n,2 int32),
``official_score``, ``mean_trajectory_exact_kl``.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

BOOTSTRAP_SEED = 20260820
BOOTSTRAP_REPS = 20000

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

NPZ_ROOTS = {
    "multikey": "results/statekv_counterfactual/cheapr2_fresh_multikey_v1/rank_migration/_shards",
    "longbench": "results/statekv_counterfactual/cheapr2_fresh_longbench_v1/rank_migration/_shards",
}
SUMMARY_CSVS = [
    "results/statekv_counterfactual/cheapr2_fresh_multikey_v1/closed_loop/train/_shards/s0/sample_summary.csv",
    "results/statekv_counterfactual/cheapr2_fresh_multikey_v1/closed_loop/train/_shards/s1/sample_summary.csv",
    "results/statekv_counterfactual/cheapr2_fresh_longbench_v1/closed_loop/train/_shards/s0/sample_summary.csv",
    "results/statekv_counterfactual/cheapr2_fresh_longbench_v1/closed_loop/train/_shards/s0b/sample_summary.csv",
    "results/statekv_counterfactual/cheapr2_fresh_longbench_v1/closed_loop/train/_shards/s1/sample_summary.csv",
]
WORKLOAD_COLUMNS = ["multikey", "hotpotqa", "2wikimqa"]
R2_POLICY = "STRICT_CAUSAL_ROLLOUT_R2"
QK_POLICY = "STRICT_QK_CURRENT"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _task_group(task: str) -> str:
    task = str(task).lower()
    if "multikey" in task:
        return "multikey"
    if "hotpotqa" in task:
        return "hotpotqa"
    if "2wikimqa" in task:
        return "2wikimqa"
    if "gov" in task:
        return "gov_report"
    return str(task)


def _load_gains(root: Path) -> pd.DataFrame:
    """Per-sample Gain_s = official_score(R2) - official_score(QK), budget 256."""

    frames = []
    for relative in SUMMARY_CSVS:
        path = root / relative
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        raise RuntimeError("no fresh sample_summary.csv files found")
    summary = pd.concat(frames, ignore_index=True)
    summary = summary[summary["budget"].astype(int) == 256]
    pivot = summary.pivot_table(
        index="sample_id", columns="policy", values="official_score", aggfunc="first"
    )
    gains = (pivot[R2_POLICY] - pivot[QK_POLICY]).rename("gain_s").reset_index()
    return gains


def _iter_npz(root: Path) -> List[Path]:
    paths: List[Path] = []
    for name, relative in NPZ_ROOTS.items():
        base = root / relative
        if base.exists():
            paths.extend(sorted(base.glob("*/train/*.npz")))
    return paths


# ---------------------------------------------------------------------------
# Per-cycle mechanism metrics
# ---------------------------------------------------------------------------


def _ranks_desc(values: np.ndarray) -> np.ndarray:
    """Rank within the given set, 1 = highest."""

    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(-values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.float64)
    return ranks


def _cycle_metrics(
    eligible: np.ndarray,
    current_eligible: np.ndarray,
    rollout_per_step: np.ndarray,
    selected_core: np.ndarray,
    needle_mask: np.ndarray,
    core_budget: int,
) -> Dict[str, float]:
    """Mechanism metrics for one refresh cycle, at B = core_budget and B = 256."""

    n_eligible = int(len(eligible))
    horizon = int(rollout_per_step.shape[0])
    rank_future = np.stack(
        [_ranks_desc(rollout_per_step[step]) for step in range(horizon)], axis=0
    )
    rank_future_best = rank_future.min(axis=0)
    rollout_sum = rollout_per_step.sum(axis=0)
    total_rollout_mass = float(rollout_sum.sum())
    core_set = {int(value) for value in selected_core}

    out: Dict[str, float] = {}
    for budget_core, suffix in ((int(core_budget), ""), (256, "_b256")):
        rank_now = _ranks_desc(current_eligible)
        cold = rank_now > budget_core
        reactivated = cold & (rank_future_best <= budget_core)
        n_cold = int(cold.sum())
        n_reactivated = int(reactivated.sum())
        topb_now = rank_now <= budget_core
        overlaps = []
        for step in range(horizon):
            topb_future = rank_future[step] <= budget_core
            overlaps.append(
                float(np.logical_and(topb_now, topb_future).sum()) / float(budget_core)
            )
        prefix = "" if suffix == "" else suffix
        out[f"reactivation_rate{prefix}"] = (
            float(n_reactivated) / n_cold if n_cold else float("nan")
        )
        out[f"reactivation_count_norm{prefix}"] = float(n_reactivated) / float(
            budget_core
        )
        out[f"reactivation_mass{prefix}"] = (
            float(rollout_sum[reactivated].sum()) / total_rollout_mass
            if total_rollout_mass > 0.0
            else float("nan")
        )
        severe = (rank_now > 2 * budget_core) & (rank_future_best <= budget_core)
        out[f"severe_reactivation_rate{prefix}"] = (
            float(severe.sum()) / n_cold if n_cold else float("nan")
        )
        for cutoff in (512, 1024):
            if n_eligible > cutoff:
                severe_abs = (rank_now > cutoff) & (rank_future_best <= budget_core)
                out[f"severe_reactivation_rate_{cutoff}{prefix}"] = (
                    float(severe_abs.sum()) / n_cold if n_cold else float("nan")
                )
            else:
                out[f"severe_reactivation_rate_{cutoff}{prefix}"] = float("nan")
        magnitudes = rank_now[reactivated] - rank_future_best[reactivated]
        out[f"migration_magnitude_median{prefix}"] = (
            float(np.median(magnitudes)) if n_reactivated else float("nan")
        )
        out[f"migration_magnitude_pct_median{prefix}"] = (
            float(np.median(magnitudes / n_eligible)) if n_reactivated else float("nan")
        )
        out[f"topb_overlap_min{prefix}"] = float(np.min(overlaps))
        out[f"topb_overlap_mean{prefix}"] = float(np.mean(overlaps))
        for label, mask in (("needle", needle_mask), ("nonneedle", ~needle_mask)):
            n_pool = int(mask.sum())
            if n_pool == 0:
                for name in ("cold_frac", "reactivated_frac", "retained_r2", "retained_qk"):
                    out[f"{label}_{name}{prefix}"] = float("nan")
                continue
            retained_r2 = np.array(
                [int(position) in core_set for position in eligible[mask]], dtype=bool
            )
            out[f"{label}_cold_frac{prefix}"] = float(cold[mask].mean())
            out[f"{label}_reactivated_frac{prefix}"] = float(reactivated[mask].mean())
            out[f"{label}_retained_r2{prefix}"] = float(retained_r2.mean())
            out[f"{label}_retained_qk{prefix}"] = float(topb_now[mask].mean())
    return out


def _needle_mask(eligible: np.ndarray, needle_spans: np.ndarray) -> np.ndarray:
    mask = np.zeros(len(eligible), dtype=bool)
    for start, end in np.asarray(needle_spans, dtype=np.int64).reshape(-1, 2):
        mask |= (eligible >= start) & (eligible < end)
    return mask


# ---------------------------------------------------------------------------
# Per-sample aggregation
# ---------------------------------------------------------------------------


def _per_sample_rows(path: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Return (per-sample aggregated row, per-refresh-cycle rows)."""

    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    cycles = sorted(
        int(key.split("_")[1])
        for key in arrays
        if key.startswith("cycle_") and key.endswith("_positions")
    )
    core_budget = int(arrays["core_budget"])
    needle_spans = np.asarray(arrays["needle_spans"], dtype=np.int64).reshape(-1, 2)
    cycle_rows: List[Dict[str, Any]] = []
    for cycle in cycles:
        prefix = "cycle_%04d_" % cycle
        if not bool(arrays[prefix + "refreshed"]):
            continue
        positions = np.asarray(arrays[prefix + "positions"], dtype=np.int64)
        eligible = np.asarray(arrays[prefix + "eligible"], dtype=np.int64)
        current_shared = np.asarray(arrays[prefix + "current_shared"], dtype=np.float64)
        value_by_position = {
            int(position): float(value)
            for position, value in zip(positions, current_shared)
        }
        current_eligible = np.asarray(
            [value_by_position[int(position)] for position in eligible],
            dtype=np.float64,
        )
        rollout_per_step = np.asarray(
            arrays[prefix + "rollout_per_step"], dtype=np.float64
        )
        selected_core = np.asarray(arrays[prefix + "selected_core"], dtype=np.int64)
        mask = _needle_mask(eligible, needle_spans)
        metrics = _cycle_metrics(
            eligible, current_eligible, rollout_per_step, selected_core, mask, core_budget
        )
        cycle_rows.append({"cycle": int(cycle), **metrics})
    cycle_frame = pd.DataFrame(cycle_rows)
    metric_columns = [column for column in cycle_frame.columns if column != "cycle"]
    row: Dict[str, Any] = {
        "sample_id": str(arrays["sample_id"]),
        "task": str(arrays["task"]),
        "group": _task_group(str(arrays["task"])),
        "budget": int(arrays["budget"]),
        "core_budget": core_budget,
        "n_refresh_cycles": int(len(cycle_rows)),
        "n_eligible_last": int(
            len(arrays["cycle_%04d_eligible" % cycles[-1]])
        ),
        "official_score_r2": float(arrays["official_score"]),
    }
    for column in metric_columns:
        values = cycle_frame[column].to_numpy(dtype=np.float64)
        finite = values[np.isfinite(values)]
        row[f"{column}_mean"] = (
            float(finite.mean()) if len(finite) else float("nan")
        )
        row[f"{column}_max"] = float(finite.max()) if len(finite) else float("nan")
    return row, cycle_rows


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def _bootstrap_mean_ci(
    values: np.ndarray, rng: np.random.Generator, reps: int
) -> Tuple[float, float, float, int]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    n = int(len(values))
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    estimate = float(values.mean())
    if n < 2:
        return estimate, estimate, estimate, n
    index = rng.integers(0, n, size=(int(reps), n))
    means = values[index].mean(axis=1)
    return (
        estimate,
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
        n,
    )


def _rank_rows(matrix: np.ndarray) -> np.ndarray:
    """Average ranks (ties share the mean rank) along axis 1, 0-based."""

    matrix = np.asarray(matrix, dtype=np.float64)
    n = matrix.shape[1]
    less = (matrix[:, :, None] > matrix[:, None, :]).sum(axis=2)
    leq = (matrix[:, :, None] >= matrix[:, None, :]).sum(axis=2)
    return (less + leq - 1) / 2.0


def _bootstrap_spearman(
    x: np.ndarray, y: np.ndarray, rng: np.random.Generator, reps: int
) -> Tuple[float, float, float, int]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    n = int(len(x))
    if n < 4 or np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan"), float("nan"), float("nan"), n
    estimate = float(spearmanr(x, y).statistic)
    index = rng.integers(0, n, size=(int(reps), n))
    resampled_x = _rank_rows(x[index])
    resampled_y = _rank_rows(y[index])
    resampled_x -= resampled_x.mean(axis=1, keepdims=True)
    resampled_y -= resampled_y.mean(axis=1, keepdims=True)
    numerator = (resampled_x * resampled_y).sum(axis=1)
    denominator = np.sqrt(
        (resampled_x**2).sum(axis=1) * (resampled_y**2).sum(axis=1)
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        statistics = numerator / denominator
    statistics = statistics[np.isfinite(statistics)]
    if len(statistics) == 0:
        return estimate, float("nan"), float("nan"), n
    return (
        estimate,
        float(np.percentile(statistics, 2.5)),
        float(np.percentile(statistics, 97.5)),
        n,
    )


def _bootstrap_diff_ci(
    a: np.ndarray, b: np.ndarray, rng: np.random.Generator, reps: int
) -> Tuple[float, float, float]:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return float("nan"), float("nan"), float("nan")
    estimate = float(a.mean() - b.mean())
    index_a = rng.integers(0, len(a), size=(int(reps), len(a)))
    index_b = rng.integers(0, len(b), size=(int(reps), len(b)))
    diffs = a[index_a].mean(axis=1) - b[index_b].mean(axis=1)
    return (
        estimate,
        float(np.percentile(diffs, 2.5)),
        float(np.percentile(diffs, 97.5)),
    )


# ---------------------------------------------------------------------------
# Case studies
# ---------------------------------------------------------------------------


def _case_study_frame(path: Path, needle_only: bool) -> pd.DataFrame:
    """Per-refresh, per-token migration table for one arm.

    ``realized_usage`` is the mean ``current_shared`` the token received in
    the 16 cycles after the refresh, over cycles where it is still in
    ``positions``. With ``needle_only`` the rows are needle tokens; without
    needles the rows are the reactivated tokens (rank_now > B, best future
    rank <= B) of each refresh cycle.
    """

    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    core_budget = int(arrays["core_budget"])
    refresh_frequency = int(arrays["refresh_frequency"])
    needle_spans = np.asarray(arrays["needle_spans"], dtype=np.int64).reshape(-1, 2)
    cycles = sorted(
        int(key.split("_")[1])
        for key in arrays
        if key.startswith("cycle_") and key.endswith("_positions")
    )
    value_by_cycle: Dict[int, Dict[int, float]] = {}
    for cycle in cycles:
        prefix = "cycle_%04d_" % cycle
        positions = np.asarray(arrays[prefix + "positions"], dtype=np.int64)
        current_shared = np.asarray(
            arrays[prefix + "current_shared"], dtype=np.float64
        )
        value_by_cycle[cycle] = {
            int(position): float(value)
            for position, value in zip(positions, current_shared)
        }
    rows: List[Dict[str, Any]] = []
    for cycle in cycles:
        prefix = "cycle_%04d_" % cycle
        if not bool(arrays[prefix + "refreshed"]):
            continue
        eligible = np.asarray(arrays[prefix + "eligible"], dtype=np.int64)
        current_eligible = np.asarray(
            [value_by_cycle[cycle][int(position)] for position in eligible]
        )
        rollout_per_step = np.asarray(
            arrays[prefix + "rollout_per_step"], dtype=np.float64
        )
        rank_now = _ranks_desc(current_eligible)
        rank_future_best = np.stack(
            [_ranks_desc(rollout_per_step[step]) for step in range(rollout_per_step.shape[0])],
            axis=0,
        ).min(axis=0)
        core_set = {int(value) for value in arrays[prefix + "selected_core"]}
        needle_mask = _needle_mask(eligible, needle_spans)
        if needle_only:
            chosen = np.flatnonzero(needle_mask)
        else:
            chosen = np.flatnonzero(
                (rank_now > core_budget) & (rank_future_best <= core_budget)
            )[:20]
        after_cycles = [
            later
            for later in cycles
            if cycle < later <= cycle + refresh_frequency
        ]
        for index in chosen:
            position = int(eligible[index])
            usages = [
                value_by_cycle[later][position]
                for later in after_cycles
                if position in value_by_cycle[later]
            ]
            rows.append(
                {
                    "cycle": int(cycle),
                    "position": position,
                    "is_needle": bool(needle_mask[index]),
                    "rank_now": float(rank_now[index]),
                    "rank_future_best": float(rank_future_best[index]),
                    "retained_by_r2": position in core_set,
                    "retained_by_counterfactual_qk": bool(
                        rank_now[index] <= core_budget
                    ),
                    "realized_usage": (
                        float(np.mean(usages)) if usages else float("nan")
                    ),
                    "cycles_observed": int(len(usages)),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=str(REPOSITORY_ROOT / "results/statekv_counterfactual/rank_migration_v1"),
    )
    parser.add_argument("--bootstrap-reps", type=int, default=BOOTSTRAP_REPS)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args()
    root = REPOSITORY_ROOT
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reps = int(args.bootstrap_reps)
    rng = np.random.default_rng(int(args.seed))

    npz_paths = _iter_npz(root)
    if not npz_paths:
        raise RuntimeError("no rank-migration npz files found")
    print(f"[rank-migration-analysis] {len(npz_paths)} arm npz files")

    sample_rows: List[Dict[str, Any]] = []
    cycle_rows: List[Dict[str, Any]] = []
    npz_by_sample: Dict[str, Path] = {}
    for path in npz_paths:
        row, per_cycle = _per_sample_rows(path)
        if int(row["budget"]) != 256:
            continue
        sample_rows.append(row)
        for entry in per_cycle:
            cycle_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "group": row["group"],
                    **entry,
                }
            )
        npz_by_sample[str(row["sample_id"])] = path
    samples = pd.DataFrame(sample_rows)
    cycles_frame = pd.DataFrame(cycle_rows)
    gains = _load_gains(root)
    samples = samples.merge(gains, on="sample_id", how="left")
    n_missing_gain = int(samples["gain_s"].isna().sum())
    if n_missing_gain:
        print(
            f"[rank-migration-analysis] WARNING: {n_missing_gain} samples lack Gain_s"
        )

    metric_columns = [
        column
        for column in samples.columns
        if column.endswith("_mean") or column.endswith("_max")
    ]
    base_metrics = [column[: -len("_mean")] for column in metric_columns if column.endswith("_mean")]

    # 1) workload table: mean over samples of the per-sample mean metric.
    workload_rows: List[Dict[str, Any]] = []
    for metric in base_metrics:
        column = f"{metric}_mean"
        row: Dict[str, Any] = {"metric": metric}
        for group in WORKLOAD_COLUMNS:
            values = samples.loc[samples["group"] == group, column].to_numpy()
            estimate, low, high, n = _bootstrap_mean_ci(values, rng, reps)
            row[f"{group}"] = estimate
            row[f"{group}_ci_low"] = low
            row[f"{group}_ci_high"] = high
            row[f"{group}_n"] = n
        workload_rows.append(row)
    workload = pd.DataFrame(workload_rows)
    workload.to_csv(output_dir / "workload_table.csv", index=False)

    # 2) sample level: Spearman(metric, Gain_s) per task with bootstrap CI.
    level_rows: List[Dict[str, Any]] = []
    for group, group_frame in samples.groupby("group"):
        gain = group_frame["gain_s"].to_numpy(dtype=np.float64)
        for column in metric_columns:
            rho, low, high, n = _bootstrap_spearman(
                group_frame[column].to_numpy(dtype=np.float64), gain, rng, reps
            )
            level_rows.append(
                {
                    "task": group,
                    "metric": column,
                    "spearman_rho": rho,
                    "ci_low": low,
                    "ci_high": high,
                    "n": n,
                }
            )
    pd.DataFrame(level_rows).to_csv(output_dir / "sample_level.csv", index=False)

    # 3) win vs tie within multikey.
    win_tie_rows: List[Dict[str, Any]] = []
    multikey = samples[samples["group"] == "multikey"]
    win = multikey[multikey["gain_s"] > 0]
    tie = multikey[multikey["gain_s"] == 0]
    for column in metric_columns:
        win_values = win[column].to_numpy(dtype=np.float64)
        tie_values = tie[column].to_numpy(dtype=np.float64)
        win_est, win_low, win_high, n_win = _bootstrap_mean_ci(win_values, rng, reps)
        tie_est, tie_low, tie_high, n_tie = _bootstrap_mean_ci(tie_values, rng, reps)
        diff, diff_low, diff_high = _bootstrap_diff_ci(win_values, tie_values, rng, reps)
        win_tie_rows.append(
            {
                "metric": column,
                "mean_win": win_est,
                "win_ci_low": win_low,
                "win_ci_high": win_high,
                "n_win": n_win,
                "mean_tie": tie_est,
                "tie_ci_low": tie_low,
                "tie_ci_high": tie_high,
                "n_tie": n_tie,
                "diff_win_minus_tie": diff,
                "diff_ci_low": diff_low,
                "diff_ci_high": diff_high,
            }
        )
    pd.DataFrame(win_tie_rows).to_csv(output_dir / "win_tie_comparison.csv", index=False)

    # 4) cycle level: metrics by refresh-cycle index per task (B = core only).
    cycle_base = [
        column
        for column in cycles_frame.columns
        if column not in {"sample_id", "group", "cycle"} and not column.endswith("_b256")
    ]
    cycle_long = (
        cycles_frame.groupby(["group", "cycle"])[cycle_base]
        .mean()
        .reset_index()
        .melt(
            id_vars=["group", "cycle"],
            value_vars=cycle_base,
            var_name="metric",
            value_name="mean",
        )
    )
    counts = cycles_frame.groupby(["group", "cycle"]).size().rename("n").reset_index()
    cycle_long = cycle_long.merge(counts, on=["group", "cycle"])
    cycle_long.to_csv(output_dir / "cycle_level.csv", index=False)

    # 5) case studies.
    cases: List[str] = []
    multikey_with_gain = multikey.dropna(subset=["gain_s"]).sort_values(
        "gain_s", ascending=False
    )
    chosen = list(multikey_with_gain.head(3)["sample_id"])
    tie_samples = list(multikey_with_gain[multikey_with_gain["gain_s"] == 0]["sample_id"])[:2]
    chosen.extend(tie_samples)
    hotpot = samples[samples["group"] == "hotpotqa"].dropna(subset=["gain_s"])
    if len(hotpot):
        median_gain = float(hotpot["gain_s"].median())
        chosen.append(
            str(
                hotpot.iloc[(hotpot["gain_s"] - median_gain).abs().argsort().iloc[0]][
                    "sample_id"
                ]
            )
        )
    for sample_id in chosen:
        path = npz_by_sample.get(str(sample_id))
        if path is None:
            continue
        group = _task_group(str(sample_id))
        frame = _case_study_frame(path, needle_only=(group == "multikey"))
        safe = str(sample_id).replace(":", "__").replace("/", "_")
        frame.to_csv(output_dir / f"case_{safe}.csv", index=False)
        cases.append(str(sample_id))

    # 6) compact stdout summary.
    print("\n=== rank-migration summary ===")
    print(f"samples: {len(samples)} " + ", ".join(
        f"{group}={int((samples['group'] == group).sum())}"
        for group in sorted(samples["group"].unique())
    ))
    print("\n-- workload table (mean [95% CI]) --")
    show = workload[workload["metric"].isin([
        "reactivation_rate",
        "reactivation_count_norm",
        "reactivation_mass",
        "severe_reactivation_rate",
        "migration_magnitude_pct_median",
        "topb_overlap_mean",
        "needle_reactivated_frac",
        "needle_retained_r2",
        "needle_retained_qk",
        "nonneedle_reactivated_frac",
    ])]
    for _, row in show.iterrows():
        cells = []
        for group in WORKLOAD_COLUMNS:
            estimate, low, high = row[group], row[f"{group}_ci_low"], row[f"{group}_ci_high"]
            cells.append(
                f"{group}={estimate:.3f} [{low:.3f},{high:.3f}]"
                if np.isfinite(estimate)
                else f"{group}=NA"
            )
        print(f"  {row['metric']:<36} " + "  ".join(cells))
    print("\n-- top |rho| with Gain_s (per-sample mean metrics) --")
    level_frame = pd.DataFrame(level_rows)
    level_mean = level_frame[level_frame["metric"].str.endswith("_mean")]
    for group in sorted(level_mean["task"].unique()):
        sub = level_mean[level_mean["task"] == group].dropna(subset=["spearman_rho"])
        sub = sub.reindex(sub["spearman_rho"].abs().sort_values(ascending=False).index)
        for _, row in sub.head(3).iterrows():
            print(
                f"  {group:<10} {row['metric']:<44} rho={row['spearman_rho']:+.3f} "
                f"[{row['ci_low']:+.3f},{row['ci_high']:+.3f}] n={int(row['n'])}"
            )
    print("\n-- multikey win vs tie (largest |diff|) --")
    win_tie_frame = pd.DataFrame(win_tie_rows)
    wt = win_tie_frame[win_tie_frame["metric"].str.endswith("_mean")].dropna(
        subset=["diff_win_minus_tie"]
    )
    wt = wt.reindex(wt["diff_win_minus_tie"].abs().sort_values(ascending=False).index)
    for _, row in wt.head(5).iterrows():
        print(
            f"  {row['metric']:<44} win={row['mean_win']:.3f} tie={row['mean_tie']:.3f} "
            f"diff={row['diff_win_minus_tie']:+.3f} "
            f"[{row['diff_ci_low']:+.3f},{row['diff_ci_high']:+.3f}]"
        )
    print(f"\ncase studies written: {', '.join(cases)}")
    print(f"outputs: {output_dir}")


if __name__ == "__main__":
    main()
