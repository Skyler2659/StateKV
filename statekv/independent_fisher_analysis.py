"""Grouped Stage-A′ analysis, frozen replication gate, and trust region."""
from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr

from statekv.theory_closing import _atomic_frame
from statekv.trajectory_analysis import atomic_json


FAMILIES = {
    "G0_RAW_LOGIT_NORM": "g0_raw_l2_sq",
    "G2_BASE_FISHER": "g2_base_fisher",
    "G3_MIDPOINT_FISHER": "g3_midpoint_fisher",
}


def _finite(value: Any, fallback: float = 0.0) -> float:
    try:
        result = float(value)
    except Exception:
        return float(fallback)
    return result if np.isfinite(result) else float(fallback)


def _correlation(
    left: Sequence[float], right: Sequence[float], kind: str
) -> float:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3 or np.ptp(x) <= 0.0 or np.ptp(y) <= 0.0:
        return 0.0
    if kind == "spearman":
        return _finite(spearmanr(x, y).statistic)
    if kind == "pearson":
        return _finite(pearsonr(x, y).statistic)
    if kind == "kendall":
        return _finite(kendalltau(x, y).statistic)
    raise ValueError("unknown correlation kind: %s" % kind)


def _regression_metrics(
    prediction: Sequence[float], truth: Sequence[float]
) -> Dict[str, float]:
    x = np.asarray(prediction, dtype=np.float64)
    y = np.asarray(truth, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 2:
        return {
            "slope": 0.0,
            "intercept": 0.0,
            "r2": 0.0,
        }
    design = np.stack([x, np.ones(len(x))], axis=1)
    slope, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = float(np.sum((y - (slope * x + intercept)) ** 2))
    total = float(np.sum((y - y.mean()) ** 2))
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": float(1.0 - residual / max(total, 1.0e-30)),
    }


def _pointwise_metrics(
    frame: pd.DataFrame, prediction_column: str
) -> Dict[str, float]:
    prediction = frame[prediction_column].to_numpy(dtype=np.float64)
    truth = frame["exact_kl"].to_numpy(dtype=np.float64)
    epsilon = 1.0e-12
    ratio = np.maximum(
        (prediction + epsilon) / (truth + epsilon),
        (truth + epsilon) / (prediction + epsilon),
    )
    relative = np.abs(prediction - truth) / np.maximum(
        np.abs(truth), epsilon
    )
    result = {
        "rows": int(len(frame)),
        "spearman": _correlation(prediction, truth, "spearman"),
        "pearson": _correlation(prediction, truth, "pearson"),
        "median_symmetric_ratio": float(np.median(ratio)),
        "p90_symmetric_ratio": float(np.quantile(ratio, 0.90)),
        "median_relative_error": float(np.median(relative)),
        "p90_relative_error": float(np.quantile(relative, 0.90)),
    }
    result.update(_regression_metrics(prediction, truth))
    return result


def _pairwise_sign_accuracy(
    prediction: np.ndarray, truth: np.ndarray
) -> float:
    correct = 0
    total = 0
    for left, right in itertools.combinations(range(len(prediction)), 2):
        predicted_delta = float(prediction[left] - prediction[right])
        true_delta = float(truth[left] - truth[right])
        if predicted_delta == 0.0 or true_delta == 0.0:
            continue
        correct += int(np.sign(predicted_delta) == np.sign(true_delta))
        total += 1
    return float(correct / total) if total else 0.0


def _action_metric_row(
    current: pd.DataFrame,
    prediction_column: str,
    family: str,
    horizon: int,
) -> Dict[str, Any]:
    prediction = current[prediction_column].to_numpy(dtype=np.float64)
    truth = current["exact_kl"].to_numpy(dtype=np.float64)
    predicted_order = np.argsort(prediction, kind="stable")
    true_order = np.argsort(truth, kind="stable")
    selected = int(predicted_order[0])
    best = float(truth[int(true_order[0])])
    worst = float(truth[int(true_order[-1])])
    top_k = min(3, len(current))
    overlap = len(
        set(predicted_order[:top_k].tolist())
        & set(true_order[:top_k].tolist())
    ) / float(top_k)
    first = current.iloc[0]
    return {
        "task": str(first["task"]),
        "sample_id": str(first["sample_id"]),
        "anchor": int(first["anchor"]),
        "horizon": int(horizon),
        "family": str(family),
        "candidate_count": int(len(current)),
        "spearman": _correlation(prediction, truth, "spearman"),
        "kendall": _correlation(prediction, truth, "kendall"),
        "normalized_regret": float(
            (truth[selected] - best) / max(worst - best, 1.0e-12)
        ),
        "top1_regret": float(truth[selected] - best),
        "top3_overlap": float(overlap),
        "pairwise_sign_accuracy": _pairwise_sign_accuracy(
            prediction, truth
        ),
        "selected_candidate_id": str(
            current.iloc[selected]["candidate_id"]
        ),
        "oracle_candidate_id": str(
            current.iloc[int(true_order[0])]["candidate_id"]
        ),
    }


def build_action_rows(
    rows: pd.DataFrame, horizons: Sequence[int]
) -> pd.DataFrame:
    output: List[Dict[str, Any]] = []
    keys = ["task", "sample_id", "anchor", "candidate_id"]
    for horizon in horizons:
        step = rows[rows["horizon_offset"] <= int(horizon)]
        columns = ["exact_kl"] + list(FAMILIES.values())
        cumulative = step.groupby(keys, as_index=False)[columns].sum()
        for (_, _, _), group in cumulative.groupby(
            ["task", "sample_id", "anchor"], sort=False
        ):
            for family, column in FAMILIES.items():
                output.append(
                    _action_metric_row(
                        group, column, family, int(horizon)
                    )
                )
    return pd.DataFrame(output)


def _cluster_bootstrap_interval(
    frame: pd.DataFrame,
    value_column: str,
    samples: int,
    seed: int,
) -> Dict[str, float]:
    sequence_ids = sorted(frame["sample_id"].astype(str).unique())
    if not sequence_ids:
        return {"low": 0.0, "high": 0.0}
    values = {
        sample_id: frame[
            frame["sample_id"].astype(str) == sample_id
        ][value_column].to_numpy(dtype=np.float64)
        for sample_id in sequence_ids
    }
    rng = np.random.default_rng(int(seed))
    estimates = []
    for _ in range(int(samples)):
        chosen = rng.choice(sequence_ids, size=len(sequence_ids), replace=True)
        combined = np.concatenate([values[str(value)] for value in chosen])
        estimates.append(float(np.median(combined)))
    return {
        "low": float(np.quantile(estimates, 0.025)),
        "high": float(np.quantile(estimates, 0.975)),
    }


def _action_summary(
    action: pd.DataFrame, bootstrap_samples: int, seed: int
) -> Dict[str, Any]:
    result: Dict[str, Any] = {"task": {}, "anchor": {}, "horizon": {}}
    for task, task_frame in action.groupby("task"):
        result["task"][str(task)] = {}
        for family, current in task_frame.groupby("family"):
            metrics = {
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
            metrics["spearman_cluster_bootstrap_95ci"] = (
                _cluster_bootstrap_interval(
                    current,
                    "spearman",
                    samples=bootstrap_samples,
                    seed=seed,
                )
            )
            result["task"][str(task)][str(family)] = metrics
    for anchor, current_anchor in action.groupby("anchor"):
        result["anchor"][str(int(anchor))] = {}
        for (task, family), current in current_anchor.groupby(
            ["task", "family"]
        ):
            result["anchor"][str(int(anchor))][
                "%s:%s" % (task, family)
            ] = {
                "spearman": float(current["spearman"].median()),
                "normalized_regret": float(
                    current["normalized_regret"].median()
                ),
            }
    for horizon, current_horizon in action.groupby("horizon"):
        result["horizon"][str(int(horizon))] = {}
        for (task, family), current in current_horizon.groupby(
            ["task", "family"]
        ):
            result["horizon"][str(int(horizon))][
                "%s:%s" % (task, family)
            ] = {
                "spearman": float(current["spearman"].median()),
                "normalized_regret": float(
                    current["normalized_regret"].median()
                ),
            }
    return result


def _pointwise_summary(rows: pd.DataFrame) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "task": {},
        "anchor": {},
        "horizon_offset": {},
        "kl_magnitude_bins": {},
    }
    for task, current in rows.groupby("task"):
        result["task"][str(task)] = {
            family: _pointwise_metrics(current, column)
            for family, column in FAMILIES.items()
        }
    for anchor, current in rows.groupby("anchor"):
        result["anchor"][str(int(anchor))] = {
            "%s:%s" % (task, family): _pointwise_metrics(task_rows, column)
            for task, task_rows in current.groupby("task")
            for family, column in FAMILIES.items()
        }
    for offset, current in rows.groupby("horizon_offset"):
        result["horizon_offset"][str(int(offset))] = {
            "%s:%s" % (task, family): _pointwise_metrics(task_rows, column)
            for task, task_rows in current.groupby("task")
            for family, column in FAMILIES.items()
        }
    positive = rows["exact_kl"].to_numpy(dtype=np.float64)
    edges = np.unique(
        np.quantile(positive, [0.0, 0.25, 0.5, 0.75, 1.0])
    )
    if len(edges) >= 2:
        binned = rows.copy()
        binned["kl_bin"] = pd.cut(
            binned["exact_kl"],
            bins=edges,
            include_lowest=True,
            duplicates="drop",
        ).astype(str)
        for label, current in binned.groupby("kl_bin"):
            result["kl_magnitude_bins"][str(label)] = {
                "%s:%s" % (task, family): _pointwise_metrics(
                    task_rows, column
                )
                for task, task_rows in current.groupby("task")
                for family, column in FAMILIES.items()
            }
    return result


def _trust_threshold(
    training: pd.DataFrame,
    metric: str,
    error_gate: float,
) -> Tuple[float, float, float]:
    values = training[metric].to_numpy(dtype=np.float64)
    errors = training["g3_relative_error"].to_numpy(dtype=np.float64)
    candidates = np.unique(
        np.quantile(values[np.isfinite(values)], np.linspace(0.05, 1.0, 96))
    )
    selected = float(np.min(candidates))
    best_coverage = -1.0
    best_precision = 0.0
    for threshold in candidates:
        mask = values <= float(threshold)
        if not mask.any():
            continue
        precision = float(np.mean(errors[mask] <= float(error_gate)))
        coverage = float(np.mean(mask))
        if precision >= 0.90 and coverage > best_coverage:
            selected = float(threshold)
            best_coverage = coverage
            best_precision = precision
    if best_coverage < 0.0:
        selected = float(np.min(candidates))
        mask = values <= selected
        best_coverage = float(np.mean(mask))
        best_precision = float(
            np.mean(errors[mask] <= float(error_gate))
        )
    return selected, best_precision, best_coverage


def analyze_trust_region(
    rows: pd.DataFrame, cfg: Any
) -> Dict[str, Any]:
    metrics = {
        "T0_BASE_FISHER_DISTANCE": (
            "trust_t0_base_fisher_distance",
            True,
        ),
        "T1_FISHER_MARGIN_RATIO": (
            "trust_t1_fisher_margin_ratio",
            True,
        ),
        "T2_TOP_SWITCH_MARGIN_RATIO": (
            "trust_t2_top_switch_margin_ratio",
            True,
        ),
        "T3_G2_G3_DISAGREEMENT": (
            "trust_t3_g2_g3_disagreement",
            False,
        ),
    }
    result: Dict[str, Any] = {"families": {}}
    for family, (metric, deployable) in metrics.items():
        heldout_blocks = []
        thresholds = []
        for sample_id in sorted(rows["sample_id"].astype(str).unique()):
            training = rows[rows["sample_id"].astype(str) != sample_id]
            test = rows[rows["sample_id"].astype(str) == sample_id].copy()
            threshold, train_precision, train_coverage = _trust_threshold(
                training,
                metric,
                float(cfg.trust_relative_error_gate),
            )
            thresholds.append(threshold)
            test["trusted"] = test[metric] <= threshold
            test["trust_threshold"] = threshold
            test["training_precision"] = train_precision
            test["training_coverage"] = train_coverage
            heldout_blocks.append(test)
        evaluated = pd.concat(heldout_blocks, ignore_index=True)
        task_summary = {}
        for task, current in evaluated.groupby("task"):
            trusted = current["trusted"].to_numpy(dtype=bool)
            good = (
                current["g3_relative_error"].to_numpy(dtype=np.float64)
                <= float(cfg.trust_relative_error_gate)
            )
            precision = (
                float(np.mean(good[trusted])) if trusted.any() else 0.0
            )
            recall = float(np.sum(good & trusted) / max(np.sum(good), 1))
            group_counts = (
                current.groupby(
                    ["sample_id", "anchor", "horizon_offset"]
                )["trusted"]
                .sum()
                .to_numpy(dtype=np.int64)
            )
            ranking_values = []
            for _, group in current[current["trusted"]].groupby(
                ["sample_id", "anchor", "horizon_offset"]
            ):
                if len(group) >= 3:
                    ranking_values.append(
                        _correlation(
                            group["g3_midpoint_fisher"],
                            group["exact_kl"],
                            "spearman",
                        )
                    )
            task_summary[str(task)] = {
                "precision": precision,
                "recall": recall,
                "sample_coverage": float(np.mean(trusted)),
                "abstention_rate": float(1.0 - np.mean(trusted)),
                "action_coverage": float(np.mean(group_counts >= 2)),
                "trusted_action_spearman": (
                    float(np.median(ranking_values))
                    if ranking_values
                    else 0.0
                ),
            }
        passed = bool(
            deployable
            and task_summary
            and min(
                value["precision"] for value in task_summary.values()
            )
            >= float(cfg.trust_precision_gate)
            and min(
                value["sample_coverage"] for value in task_summary.values()
            )
            >= float(cfg.trust_coverage_gate)
        )
        result["families"][family] = {
            "metric_column": metric,
            "deployable": bool(deployable),
            "outer_threshold_median": float(np.median(thresholds)),
            "task": task_summary,
            "passed": passed,
        }
    passing = [
        family
        for family, value in result["families"].items()
        if bool(value["passed"])
    ]
    result["passing_deployable_families"] = passing
    result["measurable_trust_region_exists"] = bool(passing)
    result["thresholds_use_task_id"] = False
    result["heldout_sequence_leakage"] = False
    return result


def summarize_curvature(curvature: pd.DataFrame) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "adaptive_vs_exact": {
            "maximum_absolute_error": float(
                curvature[
                    "adaptive_weighted_abs_error_vs_exact"
                ].max()
            ),
            "maximum_relative_error": float(
                curvature[
                    "adaptive_weighted_relative_error_vs_exact"
                ].max()
            ),
            "warning_rows": int(
                (curvature["adaptive_warning_count"] > 0).sum()
            ),
        },
        "task": {},
    }
    for task, current in curvature.groupby("task"):
        outlier = current["gl5_relative_error"] > 1.0e-3
        switched = current["top1_changed_along_path"].astype(bool)
        summary["task"][str(task)] = {
            "rows": int(len(current)),
            "gl3_median_relative_error": float(
                current["gl3_relative_error"].median()
            ),
            "gl3_p90_relative_error": float(
                current["gl3_relative_error"].quantile(0.90)
            ),
            "gl3_max_relative_error": float(
                current["gl3_relative_error"].max()
            ),
            "gl5_median_relative_error": float(
                current["gl5_relative_error"].median()
            ),
            "gl5_p90_relative_error": float(
                current["gl5_relative_error"].quantile(0.90)
            ),
            "gl5_max_relative_error": float(
                current["gl5_relative_error"].max()
            ),
            "gl5_outlier_fraction": float(outlier.mean()),
            "top_switch_fraction": float(switched.mean()),
            "top_switch_fraction_among_gl5_outliers": (
                float(switched[outlier].mean()) if outlier.any() else 0.0
            ),
            "median_curvature_concentration": float(
                current["curvature_concentration"].median()
            ),
            "median_curvature_concentration_gl5_outliers": (
                float(
                    current.loc[
                        outlier, "curvature_concentration"
                    ].median()
                )
                if outlier.any()
                else 0.0
            ),
            "median_effective_width": float(
                current["effective_curvature_width"].median()
            ),
            "median_effective_width_gl5_outliers": (
                float(
                    current.loc[
                        outlier, "effective_curvature_width"
                    ].median()
                )
                if outlier.any()
                else 0.0
            ),
            "gl5_error_spearman_curvature_concentration": _correlation(
                current["gl5_relative_error"],
                current["curvature_concentration"],
                "spearman",
            ),
            "gl5_error_spearman_initial_margin": _correlation(
                current["gl5_relative_error"],
                current["initial_top1_margin"],
                "spearman",
            ),
            "gl5_error_spearman_direct_norm": _correlation(
                current["gl5_relative_error"],
                current["layer27_direct_norm"],
                "spearman",
            ),
            "gl5_error_spearman_state_drift": _correlation(
                current["gl5_relative_error"],
                current["layer27_actual_residual_norm"],
                "spearman",
            ),
        }
    return summary


def _replication_gate(
    rows: pd.DataFrame,
    action: pd.DataFrame,
    pointwise: Mapping[str, Any],
    action_summary: Mapping[str, Any],
    cfg: Any,
) -> Dict[str, Any]:
    task_names = sorted(rows["task"].astype(str).unique())
    checks: Dict[str, bool] = {}
    sequence_improvements: Dict[str, Any] = {}
    for task in task_names:
        point = pointwise["task"][task]
        actions = action_summary["task"][task]
        g0 = actions["G0_RAW_LOGIT_NORM"]
        g3 = actions["G3_MIDPOINT_FISHER"]
        checks["%s_kl_spearman" % task] = bool(
            point["G3_MIDPOINT_FISHER"]["spearman"]
            >= float(cfg.g3_kl_spearman_gate)
        )
        checks["%s_action_spearman" % task] = bool(
            g3["spearman"] >= float(cfg.g3_action_spearman_gate)
        )
        checks["%s_median_symmetric_ratio" % task] = bool(
            point["G3_MIDPOINT_FISHER"]["median_symmetric_ratio"]
            <= float(cfg.g3_median_symmetric_ratio_gate)
        )
        checks["%s_action_increment" % task] = bool(
            g3["spearman"] - g0["spearman"]
            >= float(cfg.g3_action_increment_gate)
        )
        checks["%s_normalized_regret" % task] = bool(
            g3["normalized_regret"] < g0["normalized_regret"]
        )
        task_action = action[action["task"].astype(str) == task]
        pivot = (
            task_action.groupby(["sample_id", "family"])["spearman"]
            .median()
            .unstack("family")
        )
        improvements = (
            pivot["G3_MIDPOINT_FISHER"]
            - pivot["G0_RAW_LOGIT_NORM"]
        )
        positive_sequences = int((improvements > 0.0).sum())
        checks["%s_sequence_direction" % task] = bool(
            positive_sequences >= 8
        )
        positive_anchors = 0
        for anchor in sorted(task_action["anchor"].unique()):
            current = task_action[task_action["anchor"] == anchor]
            medians = current.groupby("family")["spearman"].median()
            positive_anchors += int(
                medians["G3_MIDPOINT_FISHER"]
                > medians["G0_RAW_LOGIT_NORM"]
            )
        positive_horizons = 0
        for horizon in sorted(task_action["horizon"].unique()):
            current = task_action[task_action["horizon"] == horizon]
            medians = current.groupby("family")["spearman"].median()
            positive_horizons += int(
                medians["G3_MIDPOINT_FISHER"]
                > medians["G0_RAW_LOGIT_NORM"]
            )
        checks["%s_not_single_anchor_or_horizon" % task] = bool(
            positive_anchors >= 2 and positive_horizons >= 2
        )
        sequence_improvements[task] = {
            "positive_sequences": positive_sequences,
            "total_sequences": int(len(improvements)),
            "positive_anchors": positive_anchors,
            "positive_horizons": positive_horizons,
        }
    return {
        "stage_a_prime_replication_passed": bool(all(checks.values())),
        "checks": checks,
        "sequence_direction": sequence_improvements,
        "fixed_gl5_is_blocking_gate": False,
        "post_hoc_gate_relaxation": False,
        "formal_data_only": True,
    }


PULLBACK_SCHEMA = {
    "sample_id": "string",
    "task": "string",
    "anchor": "int64",
    "horizon_offset": "int64",
    "candidate_id": "string",
    "pullback_mode": "string",
    "exact_kl": "float64",
    "pullback_energy": "float64",
}

CROSS_SCHEMA = {
    "sample_id": "string",
    "task": "string",
    "anchor": "int64",
    "horizon_offset": "int64",
    "candidate_id": "string",
    "state_energy": "float64",
    "direct_energy": "float64",
    "cross_energy": "float64",
}

LOW_RANK_SCHEMA = {
    "sample_id": "string",
    "task": "string",
    "anchor": "int64",
    "rank": "int64",
    "explained_trace": "float64",
}

Q_STATE_SCHEMA = {
    "sample_id": "string",
    "task": "string",
    "anchor": "int64",
    "horizon_offset": "int64",
    "candidate_id": "string",
    "q_family": "string",
    "realized": "float64",
    "bound": "float64",
}

REFRESH_SCHEMA = {
    "sample_id": "string",
    "task": "string",
    "policy": "string",
    "maximum_refresh_count": "int64",
    "actual_refresh_count": "int64",
    "cumulative_kl": "float64",
}


def _empty_frame(schema: Mapping[str, str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            column: pd.Series(dtype=dtype)
            for column, dtype in schema.items()
        }
    )


def write_later_stage_skips(
    run_dir: Path, blocking_stage: str, reason: str
) -> None:
    root = Path(run_dir)
    for filename, schema in (
        ("pullback_operating_point_rows.parquet", PULLBACK_SCHEMA),
        ("state_action_cross_term_rows.parquet", CROSS_SCHEMA),
        ("pullback_low_rank_rows.parquet", LOW_RANK_SCHEMA),
        ("q_state_envelope_rows.parquet", Q_STATE_SCHEMA),
        ("q_refresh_policy_rows.parquet", REFRESH_SCHEMA),
    ):
        _atomic_frame(_empty_frame(schema), root / filename)
    skipped = {
        "status": "not_run_by_preregistered_gate",
        "blocking_stage": str(blocking_stage),
        "reason": str(reason),
        "rows": 0,
        "post_hoc_gate_relaxation": False,
    }
    for filename in (
        "pullback_jvp_validation_summary.json",
        "fisher_direct_ranking_summary.json",
        "oracle_midpoint_recovery_summary.json",
        "state_action_cross_term_summary.json",
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


def analyze_independent_stage_a(cfg: Any, run_dir: Path) -> Dict[str, Any]:
    root = Path(run_dir)
    rows = pd.read_parquet(root / "independent_fisher_geometry_rows.parquet")
    curvature = pd.read_parquet(root / "adaptive_curvature_rows.parquet")
    action = build_action_rows(
        rows, cfg.independent_fisher.evaluation_horizons
    )
    _atomic_frame(action, root / "independent_fisher_action_rows.parquet")
    pointwise = _pointwise_summary(rows)
    action_summary = _action_summary(
        action,
        bootstrap_samples=int(cfg.runtime.bootstrap_samples),
        seed=int(cfg.runtime.seed),
    )
    gate = _replication_gate(
        rows,
        action,
        pointwise,
        action_summary,
        cfg.independent_fisher,
    )
    replication = {
        "status": "complete",
        "sequence_count": int(rows["sample_id"].nunique()),
        "task_sequence_counts": {
            str(key): int(value)
            for key, value in rows.groupby("task")["sample_id"].nunique().items()
        },
        "row_count": int(len(rows)),
        "pointwise": pointwise,
        "action": action_summary,
        "gate": gate,
    }
    curvature_summary = summarize_curvature(curvature)
    trust = analyze_trust_region(rows, cfg.independent_fisher)
    atomic_json(
        root / "independent_fisher_replication_summary.json",
        replication,
    )
    atomic_json(
        root / "adaptive_curvature_summary.json", curvature_summary
    )
    atomic_json(root / "fisher_trust_region_summary.json", trust)
    atomic_json(root / "independent_fisher_gate_decision.json", gate)
    if not gate["stage_a_prime_replication_passed"]:
        write_later_stage_skips(
            root,
            "Stage A-prime",
            "G3 did not pass every frozen independent-replication gate.",
        )
    status_path = root / "status.json"
    status = json.loads(status_path.read_text())
    status["stage_a_prime_gate"] = gate
    status["state"] = (
        "stage_b_prime_authorized"
        if gate["stage_a_prime_replication_passed"]
        else "stage_a_prime_gate_failed_later_stages_skipped"
    )
    atomic_json(status_path, status)
    return {
        "run_dir": str(root),
        "stage_a_prime_passed": bool(
            gate["stage_a_prime_replication_passed"]
        ),
        "rows": int(len(rows)),
        "action_rows": int(len(action)),
        "trust_region_exists": bool(
            trust["measurable_trust_region_exists"]
        ),
    }
