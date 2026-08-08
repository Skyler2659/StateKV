"""Analysis for expensive physical-oracle closed-loop runs."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy import stats

from statekv.core.decision import select_lowest_risk
from statekv.storage import atomic_frame, atomic_json


def _finite_spearman(left: pd.Series, right: pd.Series) -> float:
    x = left.to_numpy(dtype=np.float64)
    y = right.to_numpy(dtype=np.float64)
    if len(x) < 2 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    return float(stats.spearmanr(x, y).statistic)


def analyze_oracle_closed_loop(run_dir: Path) -> Path:
    cycles = pd.read_parquet(run_dir / "cycle_rows.parquet")
    steps = pd.read_parquet(run_dir / "candidate_step_rows.parquet")
    if cycles.empty or steps.empty:
        raise ValueError("closed-loop artifacts are empty")
    cycles = cycles.copy()
    cycles["exact_mean_improvement"] = (
        cycles["stale_exact_kl_mean"] - cycles["selected_exact_kl_mean"]
    )
    strategy_rows: List[Dict[str, Any]] = []
    for strategy, current in cycles.groupby("strategy", sort=True):
        strategy_rows.append(
            {
                "strategy": str(strategy),
                "sample_loops": int(current["sample_id"].nunique()),
                "control_cycles": int(len(current)),
                "refresh_events": int(current["refresh"].sum()),
                "post_initial_refresh_events": int(
                    current[current["cycle"].astype(int) > 0]["refresh"].sum()
                ),
                "recovery_events": int(
                    (current["selected_recovered_core_tokens"] > 0).sum()
                ),
                "mean_selected_exact_kl": float(
                    current["selected_exact_kl_mean"].mean()
                ),
                "mean_stale_exact_kl": float(
                    current["stale_exact_kl_mean"].mean()
                ),
                "mean_exact_kl_improvement": float(
                    current["exact_mean_improvement"].mean()
                ),
                "harmful_exact_mean_cycles": int(
                    (current["exact_mean_improvement"] < -1.0e-12).sum()
                ),
                "minimum_unique_candidate_cores": int(
                    current["unique_candidate_cores"].min()
                ),
                "maximum_active_cache_tokens": int(
                    current["maximum_active_cache_tokens"].max()
                ),
            }
        )
    strategy_frame = pd.DataFrame(strategy_rows)

    units = (
        steps.groupby(
            ["sample_id", "task", "strategy", "cycle", "candidate"],
            as_index=False,
        )
        .agg(
            exact_mean=("exact_kl", "mean"),
            dense_mean=("dense_quadratic_risk", "mean"),
            dense_h1=("dense_quadratic_risk", "first"),
        )
    )
    alignment_rows: List[Dict[str, Any]] = []
    for keys, current in units.groupby(
        ["sample_id", "task", "strategy", "cycle"], sort=True
    ):
        sample_id, task, strategy, cycle = keys
        if str(strategy) not in {
            "dense_quadratic_h1",
            "dense_quadratic_mean",
        }:
            continue
        signal = (
            "dense_h1"
            if str(strategy) == "dense_quadratic_h1"
            else "dense_mean"
        )
        predicted = dict(zip(current["candidate"], current[signal]))
        exact = dict(zip(current["candidate"], current["exact_mean"]))
        alignment_rows.append(
            {
                "sample_id": str(sample_id),
                "task": str(task),
                "strategy": str(strategy),
                "cycle": int(cycle),
                "candidate_count": int(len(current)),
                "spearman": _finite_spearman(
                    current[signal], current["exact_mean"]
                ),
                "predicted_top1": select_lowest_risk(predicted).candidate_id,
                "exact_top1": select_lowest_risk(exact).candidate_id,
                "top1_agreement": bool(
                    select_lowest_risk(predicted).candidate_id
                    == select_lowest_risk(exact).candidate_id
                ),
            }
        )
    alignment = pd.DataFrame(alignment_rows)
    alignment_summary = []
    for strategy, current in alignment.groupby("strategy", sort=True):
        finite = current[np.isfinite(current["spearman"])]
        alignment_summary.append(
            {
                "strategy": str(strategy),
                "decision_units": int(len(current)),
                "finite_spearman_units": int(len(finite)),
                "median_spearman": (
                    float(finite["spearman"].median())
                    if not finite.empty
                    else None
                ),
                "mean_spearman": (
                    float(finite["spearman"].mean())
                    if not finite.empty
                    else None
                ),
                "top1_agreement": float(current["top1_agreement"].mean()),
            }
        )
    result = {
        "strategy_aggregates": strategy_rows,
        "dense_alignment": alignment_summary,
        "closed_loop_mechanics_passed": bool(
            cycles["state_continuity"].all()
            and (cycles["state_advanced_by"] > 0).all()
            and cycles["budget_respected"].all()
            and (cycles["unique_candidate_cores"] >= 2).all()
        ),
        "post_initial_refresh_events": int(
            cycles[cycles["cycle"].astype(int) > 0]["refresh"].sum()
        ),
        "recovery_events": int(
            (cycles["selected_recovered_core_tokens"] > 0).sum()
        ),
    }
    atomic_frame(strategy_frame, run_dir / "analysis_strategy_aggregate.csv")
    atomic_frame(alignment, run_dir / "analysis_dense_alignment_units.csv")
    atomic_json(run_dir / "analysis.json", result)
    return run_dir / "analysis.json"


__all__ = ["analyze_oracle_closed_loop"]
