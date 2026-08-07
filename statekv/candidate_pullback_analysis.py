"""Analysis and frozen gates for final-boundary Fisher pullbacks."""
from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd

from statekv.independent_fisher_analysis import (
    LOW_RANK_SCHEMA,
    Q_STATE_SCHEMA,
    REFRESH_SCHEMA,
    _action_metric_row,
    _correlation,
    _empty_frame,
)
from statekv.theory_closing import _atomic_frame
from statekv.trajectory_analysis import atomic_json


def _point_metrics(
    frame: pd.DataFrame, prediction: str, truth: str
) -> Dict[str, float]:
    predicted = frame[prediction].to_numpy(dtype=np.float64)
    realized = frame[truth].to_numpy(dtype=np.float64)
    epsilon = 1.0e-12
    ratio = np.maximum(
        (predicted + epsilon) / (realized + epsilon),
        (realized + epsilon) / (predicted + epsilon),
    )
    relative = np.abs(predicted - realized) / np.maximum(
        np.abs(realized), epsilon
    )
    return {
        "rows": int(len(frame)),
        "spearman": _correlation(predicted, realized, "spearman"),
        "pearson": _correlation(predicted, realized, "pearson"),
        "median_symmetric_ratio": float(np.median(ratio)),
        "p90_symmetric_ratio": float(np.quantile(ratio, 0.90)),
        "median_relative_error": float(np.median(relative)),
        "p90_relative_error": float(np.quantile(relative, 0.90)),
    }


def _action_rows(
    rows: pd.DataFrame,
    score_columns: Mapping[str, str],
    horizons: Sequence[int],
) -> pd.DataFrame:
    output: List[Dict[str, Any]] = []
    keys = ["task", "sample_id", "anchor", "candidate_id"]
    columns = ["exact_kl"] + sorted(set(score_columns.values()))
    for horizon in horizons:
        step = rows[rows["horizon_offset"] <= int(horizon)]
        cumulative = step.groupby(keys, as_index=False)[columns].sum()
        for _, group in cumulative.groupby(
            ["task", "sample_id", "anchor"], sort=False
        ):
            for family, column in score_columns.items():
                output.append(
                    _action_metric_row(
                        group, column, family, int(horizon)
                    )
                )
    return pd.DataFrame(output)


def _task_action_summary(action: pd.DataFrame) -> Dict[str, Any]:
    result = {}
    for (task, family), current in action.groupby(["task", "family"]):
        result.setdefault(str(task), {})[str(family)] = {
            column: float(current[column].median())
            for column in (
                "spearman",
                "kendall",
                "normalized_regret",
                "top1_regret",
                "top3_overlap",
                "pairwise_sign_accuracy",
            )
        }
    return result


def summarize_jvp_validation(rows: pd.DataFrame) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "task_radius": {},
        "pure_map": {},
    }
    for (task, radius), current in rows.groupby(
        ["task", "relative_radius"]
    ):
        summary["task_radius"].setdefault(str(task), {})[
            str(float(radius))
        ] = {
            "rows": int(len(current)),
            "median_cosine": float(current["jvp_fd_cosine"].median()),
            "p10_cosine": float(
                current["jvp_fd_cosine"].quantile(0.10)
            ),
            "median_relative_norm_error": float(
                current["relative_norm_error"].median()
            ),
            "median_fisher_energy_relative_error": float(
                current["fisher_energy_relative_error"].median()
            ),
            "median_plus_minus_asymmetry": float(
                current["plus_minus_asymmetry"].median()
            ),
        }
    reliable_task = {}
    for task, values in summary["task_radius"].items():
        small = [
            value
            for radius, value in values.items()
            if float(radius) <= 1.0e-2
        ]
        reliable_task[task] = bool(
            any(
                value["median_cosine"] >= 0.99
                and value["median_relative_norm_error"] <= 0.10
                and value["median_fisher_energy_relative_error"] <= 0.10
                for value in small
            )
        )
    summary["task_reliable"] = reliable_task
    summary["jvp_fd_reliable"] = bool(
        reliable_task and all(reliable_task.values())
    )
    return summary


def summarize_pullback(
    rows: pd.DataFrame, action: pd.DataFrame
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "task_mode": {},
        "action": _task_action_summary(action),
    }
    for (task, mode), current in rows.groupby(
        ["task", "pullback_mode"]
    ):
        truth = (
            "true_g2" if str(mode) == "B0_BASE" else "true_g3"
        )
        result["task_mode"].setdefault(str(task), {})[str(mode)] = {
            "vs_geometry_target": _point_metrics(
                current, "actual_state_energy", truth
            ),
            "vs_exact_kl": _point_metrics(
                current, "actual_state_energy", "exact_kl"
            ),
        }
    return result


def summarize_cross_term(rows: pd.DataFrame) -> Dict[str, Any]:
    result: Dict[str, Any] = {"task": {}}
    for task, current in rows.groupby("task"):
        flip_values = []
        for _, group in current.groupby(
            ["sample_id", "anchor", "horizon_offset"]
        ):
            without = (
                group["state_energy"].to_numpy(dtype=np.float64)
                + group["direct_energy"].to_numpy(dtype=np.float64)
            )
            with_cross = group["total_energy"].to_numpy(dtype=np.float64)
            flip_values.append(
                int(np.argmin(without) != np.argmin(with_cross))
            )
        result["task"][str(task)] = {
            "rows": int(len(current)),
            "median_cross_ratio": float(
                current["cross_ratio"].median()
            ),
            "p90_cross_ratio": float(
                current["cross_ratio"].quantile(0.90)
            ),
            "positive_fraction": float(
                (current["cross_energy"] > 0.0).mean()
            ),
            "negative_fraction": float(
                (current["cross_energy"] < 0.0).mean()
            ),
            "candidate_order_flip_rate": float(np.mean(flip_values)),
            "decomposition_max_abs_error": float(
                current["decomposition_abs_error"].max()
            ),
            "cauchy_schwarz_violations": int(
                (~current["cauchy_schwarz_holds"].astype(bool)).sum()
            ),
            "scalar_bound_coverage": float(
                current["scalar_bound_holds"].astype(bool).mean()
            ),
            "scalar_bound_median_looseness": float(
                current["scalar_bound_looseness"].median()
            ),
            "scalar_bound_action_spearman": _correlation(
                current["scalar_safe_bound"],
                current["exact_kl"],
                "spearman",
            ),
            "cross_exact_kl_spearman": _correlation(
                current["cross_energy"],
                current["exact_kl"],
                "spearman",
            ),
        }
    median_cross = max(
        value["median_cross_ratio"]
        for value in result["task"].values()
    )
    max_flip = max(
        value["candidate_order_flip_rate"]
        for value in result["task"].values()
    )
    if median_cross <= 0.10 and max_flip <= 0.05:
        route = "scalar"
    elif median_cross > 0.25 or max_flip > 0.20:
        route = "spectral_directional"
    else:
        route = "mixed"
    result["registered_q_state_route"] = route
    return result


def _sequence_positive_count(
    action: pd.DataFrame, task: str, family: str, baseline: str
) -> int:
    current = action[action["task"].astype(str) == str(task)]
    pivot = (
        current.groupby(["sample_id", "family"])["spearman"]
        .median()
        .unstack("family")
    )
    return int((pivot[family] > pivot[baseline]).sum())


def _oracle_midpoint_gate(
    summary: Mapping[str, Any],
    cfg: Any,
) -> Dict[str, Any]:
    checks: Dict[str, bool] = {}
    tasks = sorted(summary["task_mode"])
    for task in tasks:
        point = summary["task_mode"][task]["B1_ORACLE_MIDPOINT"][
            "vs_exact_kl"
        ]
        actions = summary["action"][task]
        b1 = actions["B1_ORACLE_MIDPOINT"]
        g3 = actions["TRUE_LOGIT_G3_ORACLE"]
        checks["%s_kl_spearman" % task] = bool(
            point["spearman"]
            >= float(cfg.oracle_midpoint_kl_spearman_gate)
        )
        checks["%s_action_spearman" % task] = bool(
            b1["spearman"]
            >= float(cfg.oracle_midpoint_action_spearman_gate)
        )
        checks["%s_median_ratio" % task] = bool(
            point["median_symmetric_ratio"]
            <= float(cfg.g3_median_symmetric_ratio_gate)
        )
        checks["%s_g3_action_drop" % task] = bool(
            g3["spearman"] - b1["spearman"] <= 0.05
        )
        checks["%s_g3_regret_gap" % task] = bool(
            b1["normalized_regret"] - g3["normalized_regret"] <= 0.05
        )
    return {
        "passed": bool(checks and all(checks.values())),
        "checks": checks,
    }


def _fisher_direct_gate(
    action: pd.DataFrame,
    summary: Mapping[str, Any],
    cfg: Any,
) -> Dict[str, Any]:
    candidates = (
        "B0_BASE_FISHER_DIRECT",
        "B2_CANDIDATE_DIRECT_MIDPOINT_FISHER_DIRECT",
        "B3_PREDICTED_MIDPOINT_FISHER_DIRECT",
    )
    family_checks = {}
    tasks = sorted(summary)
    for family in candidates:
        checks: Dict[str, bool] = {}
        increments = []
        retention = []
        for task in tasks:
            values = summary[task]
            baseline = values["EUCLIDEAN_DIRECT_ONLY"]
            current = values[family]
            oracle = values["TRUE_LOGIT_G3_ORACLE"]
            increment = current["spearman"] - baseline["spearman"]
            oracle_increment = (
                oracle["spearman"] - baseline["spearman"]
            )
            increments.append(increment)
            checks["%s_spearman_noninferior" % task] = bool(
                increment >= 0.0
            )
            checks["%s_regret" % task] = bool(
                current["normalized_regret"]
                < baseline["normalized_regret"]
            )
            checks["%s_top1" % task] = bool(
                current["top1_regret"] <= baseline["top1_regret"]
            )
            retention.append(
                float(
                    increment / max(oracle_increment, 1.0e-12)
                )
                if oracle_increment > 0.0
                else -np.inf
            )
            checks["%s_positive_sequences" % task] = bool(
                _sequence_positive_count(
                    action,
                    task,
                    family,
                    "EUCLIDEAN_DIRECT_ONLY",
                )
                >= 8
            )
            non_h1 = action[
                (action["task"].astype(str) == str(task))
                & (action["family"] == family)
                & (action["horizon"] > 1)
            ]
            baseline_h = action[
                (action["task"].astype(str) == str(task))
                & (action["family"] == "EUCLIDEAN_DIRECT_ONLY")
                & (action["horizon"] > 1)
            ]
            merged = non_h1.merge(
                baseline_h,
                on=["task", "sample_id", "anchor", "horizon"],
                suffixes=("_new", "_base"),
            )
            checks["%s_not_h1_only" % task] = bool(
                (
                    merged.groupby("horizon")["spearman_new"].median()
                    >= merged.groupby("horizon")[
                        "spearman_base"
                    ].median()
                ).sum()
                >= 2
            )
        checks["one_task_increment_005"] = bool(
            max(increments) >= float(cfg.fisher_direct_action_increment_gate)
        )
        checks["retains_80pct_oracle_increment"] = bool(
            min(retention) >= 0.80
        )
        family_checks[family] = {
            "passed": bool(all(checks.values())),
            "checks": checks,
            "task_spearman_increments": {
                task: float(increment)
                for task, increment in zip(tasks, increments)
            },
            "task_oracle_increment_retention": {
                task: float(value)
                for task, value in zip(tasks, retention)
            },
        }
    passing = [
        family
        for family, value in family_checks.items()
        if value["passed"]
    ]
    return {
        "families": family_checks,
        "passing_non_oracle_families": passing,
        "passed": bool(passing),
    }


def write_post_b_skips(
    root: Path, blocking_stage: str, reason: str
) -> None:
    _atomic_frame(
        _empty_frame(LOW_RANK_SCHEMA),
        root / "pullback_low_rank_rows.parquet",
    )
    _atomic_frame(
        _empty_frame(Q_STATE_SCHEMA),
        root / "q_state_envelope_rows.parquet",
    )
    _atomic_frame(
        _empty_frame(REFRESH_SCHEMA),
        root / "q_refresh_policy_rows.parquet",
    )
    skipped = {
        "status": "not_run_by_preregistered_gate",
        "blocking_stage": str(blocking_stage),
        "reason": str(reason),
        "rows": 0,
        "post_hoc_gate_relaxation": False,
    }
    for filename in (
        "pullback_low_rank_summary.json",
        "pullback_subspace_drift_summary.json",
        "q_state_envelope_coverage_summary.json",
        "q_state_envelope_tightness_summary.json",
        "q_state_action_summary.json",
        "interaction_q_envelope_summary.json",
        "spectral_band_q_envelope_summary.json",
        "q_pairwise_calibration_summary.json",
        "q_refresh_policy_summary.json",
        "q_free_generation_results.json",
    ):
        atomic_json(root / filename, dict(skipped, artifact=filename))


def analyze_candidate_pullback(cfg: Any, run_dir: Path) -> Dict[str, Any]:
    root = Path(run_dir)
    rows = pd.read_parquet(root / "pullback_operating_point_rows.parquet")
    fd = pd.read_parquet(root / "pullback_jvp_validation_rows.parquet")
    cross = pd.read_parquet(root / "state_action_cross_term_rows.parquet")
    wide = rows.pivot_table(
        index=[
            "task",
            "sample_id",
            "anchor",
            "horizon_offset",
            "candidate_id",
            "candidate_source",
        ],
        columns="pullback_mode",
        values=["actual_state_energy", "direct_q_energy"],
        aggfunc="first",
    )
    wide.columns = [
        "%s__%s" % (left, right) for left, right in wide.columns
    ]
    wide = wide.reset_index()
    base = rows[rows["pullback_mode"] == "B0_BASE"][
        [
            "task",
            "sample_id",
            "anchor",
            "horizon_offset",
            "candidate_id",
            "exact_kl",
            "true_g3",
            "direct_direction_norm",
        ]
    ]
    wide = wide.merge(
        base,
        on=[
            "task",
            "sample_id",
            "anchor",
            "horizon_offset",
            "candidate_id",
        ],
        how="left",
        validate="one_to_one",
    )
    score_columns = {
        "EUCLIDEAN_DIRECT_ONLY": "euclidean_direct",
        "TRUE_LOGIT_G3_ORACLE": "true_g3",
        "EXACT_KL_ORACLE": "exact_kl",
        "B0_BASE": "actual_state_energy__B0_BASE",
        "B1_ORACLE_MIDPOINT": (
            "actual_state_energy__B1_ORACLE_MIDPOINT"
        ),
        "B2_CANDIDATE_DIRECT_MIDPOINT": (
            "actual_state_energy__B2_CANDIDATE_DIRECT_MIDPOINT"
        ),
        "B3_PREDICTED_MIDPOINT": (
            "actual_state_energy__B3_PREDICTED_MIDPOINT"
        ),
        "B0_BASE_FISHER_DIRECT": "direct_q_energy__B0_BASE",
        "B2_CANDIDATE_DIRECT_MIDPOINT_FISHER_DIRECT": (
            "direct_q_energy__B2_CANDIDATE_DIRECT_MIDPOINT"
        ),
        "B3_PREDICTED_MIDPOINT_FISHER_DIRECT": (
            "direct_q_energy__B3_PREDICTED_MIDPOINT"
        ),
    }
    wide["euclidean_direct"] = (
        wide["direct_direction_norm"].to_numpy(dtype=np.float64) ** 2
    )
    action = _action_rows(
        wide,
        score_columns,
        cfg.independent_fisher.evaluation_horizons,
    )
    _atomic_frame(action, root / "fisher_pullback_action_rows.parquet")
    summary = summarize_pullback(rows, action)
    jvp = summarize_jvp_validation(fd)
    # Pure-map checks live on operating-point rows, not FD rows.
    jvp["pure_map"] = {
        "repeated_equal_fraction": float(
            rows["pure_map_repeated_equal"].astype(bool).mean()
        ),
        "cache_unchanged_fraction": float(
            rows["pure_map_cache_unchanged"].astype(bool).mean()
        ),
        "median_reference_relative_error": float(
            rows["pure_map_reference_relative_error"].median()
        ),
        "p90_reference_relative_error": float(
            rows["pure_map_reference_relative_error"].quantile(0.90)
        ),
    }
    jvp["pure_map_reliable"] = bool(
        jvp["jvp_fd_reliable"]
        and jvp["pure_map"]["repeated_equal_fraction"] == 1.0
        and jvp["pure_map"]["cache_unchanged_fraction"] == 1.0
        and jvp["pure_map"]["p90_reference_relative_error"] <= 1.0e-3
    )
    oracle_gate = _oracle_midpoint_gate(
        summary, cfg.independent_fisher
    )
    direct_gate = _fisher_direct_gate(
        action,
        summary["action"],
        cfg.independent_fisher,
    )
    cross_summary = summarize_cross_term(cross)
    atomic_json(root / "pullback_jvp_validation_summary.json", jvp)
    atomic_json(
        root / "oracle_midpoint_recovery_summary.json",
        {
            "status": "complete",
            "summary": summary,
            "gate": oracle_gate,
        },
    )
    atomic_json(
        root / "fisher_direct_ranking_summary.json",
        {
            "status": "complete",
            "action": summary["action"],
            "gate": direct_gate,
        },
    )
    atomic_json(
        root / "state_action_cross_term_summary.json", cross_summary
    )
    stage_b_passed = bool(
        jvp["pure_map_reliable"]
        and oracle_gate["passed"]
        and direct_gate["passed"]
    )
    if not stage_b_passed:
        if not jvp["pure_map_reliable"] or not oracle_gate["passed"]:
            failure = "F-B"
            reason = (
                "The pure final-boundary/oracle-midpoint pullback did not "
                "recover the true G3 geometry under the frozen gates."
            )
        else:
            failure = "F-D"
            reason = (
                "Pullback recovery passed, but no non-oracle Fisher-direct "
                "score passed the two-task action gate."
            )
        write_post_b_skips(root, "Stage B-prime", reason)
    else:
        failure = None
    gate = {
        "stage_b_prime_passed": stage_b_passed,
        "pure_map_reliable": bool(jvp["pure_map_reliable"]),
        "oracle_midpoint_passed": bool(oracle_gate["passed"]),
        "fisher_direct_passed": bool(direct_gate["passed"]),
        "failure_type_if_stopped": failure,
        "registered_q_state_route": cross_summary[
            "registered_q_state_route"
        ],
        "post_hoc_gate_relaxation": False,
    }
    atomic_json(root / "candidate_pullback_gate_decision.json", gate)
    return gate
