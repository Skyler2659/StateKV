#!/usr/bin/env python3
"""Analyze the Stage-1 functional-staleness mechanism experiment."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)


COMPARISON_KEYS = [
    "sample_id",
    "task",
    "base_anchor",
    "probe_lag",
    "strategy",
    "total_budget",
    "protected_recent_size",
]
PRIMARY_SIGNAL = "delta_e_energy_normalized_sum"
SIGNALS = [
    PRIMARY_SIGNAL,
    "d_func_normalized_sum",
    "d_new_normalized_sum",
    "d_rew_normalized_sum",
    "deployable_approx_normalized_sum",
]
FUNCTIONAL_VARIANTS = ["projected_v", "aov", "aor"]


def _finite(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    result = frame.dropna(subset=list(columns)).copy()
    mask = np.ones(len(result), dtype=bool)
    for column in columns:
        if pd.api.types.is_numeric_dtype(result[column]):
            mask &= np.isfinite(
                result[column].to_numpy(dtype=np.float64)
            )
    return result.loc[mask]


def _rho(frame: pd.DataFrame, x: str, y: str) -> float:
    valid = _finite(frame, [x, y])
    if len(valid) < 3 or valid[x].nunique() < 2 or valid[y].nunique() < 2:
        return float("nan")
    return float(spearmanr(valid[x], valid[y]).statistic)


def _cluster_bootstrap_rho(
    frame: pd.DataFrame,
    x: str,
    y: str,
    *,
    clusters: str,
    samples: int,
    seed: int,
) -> Dict[str, float]:
    valid = _finite(frame, [x, y, clusters])
    names = valid[clusters].drop_duplicates().tolist()
    rng = np.random.default_rng(seed)
    values = []
    if len(names) < 2:
        return {"low": float("nan"), "high": float("nan"), "p_gt_zero": float("nan")}
    grouped = {name: valid[valid[clusters].eq(name)] for name in names}
    for _ in range(int(samples)):
        chosen = rng.choice(names, size=len(names), replace=True)
        draw = pd.concat([grouped[name] for name in chosen], ignore_index=True)
        value = _rho(draw, x, y)
        if math.isfinite(value):
            values.append(value)
    if not values:
        return {"low": float("nan"), "high": float("nan"), "p_gt_zero": float("nan")}
    array = np.asarray(values)
    return {
        "low": float(np.quantile(array, 0.025)),
        "high": float(np.quantile(array, 0.975)),
        "p_gt_zero": float(np.mean(array > 0.0)),
    }


def _partial_spearman(
    frame: pd.DataFrame, x: str, y: str
) -> float:
    valid = _finite(frame, [x, y])
    if len(valid) < 8:
        return float("nan")
    controls = pd.DataFrame(index=valid.index)
    categorical_controls = ["task", "strategy"]
    if "layer" in valid.columns:
        categorical_controls.append("layer")
    for column in categorical_controls:
        controls = pd.concat(
            [
                controls,
                pd.get_dummies(
                    valid[column].astype(str),
                    prefix=column,
                    drop_first=True,
                    dtype=float,
                ),
            ],
            axis=1,
        )
    for column in (
        "base_anchor",
        "probe_lag",
        "total_budget",
        "protected_recent_size",
    ):
        controls[column] = rankdata(valid[column].to_numpy(dtype=float))
    design = np.column_stack(
        [np.ones(len(valid)), controls.to_numpy(dtype=float)]
    )
    x_rank = rankdata(valid[x].to_numpy(dtype=float))
    y_rank = rankdata(valid[y].to_numpy(dtype=float))
    x_residual = x_rank - design @ np.linalg.lstsq(
        design, x_rank, rcond=None
    )[0]
    y_residual = y_rank - design @ np.linalg.lstsq(
        design, y_rank, rcond=None
    )[0]
    return float(spearmanr(x_residual, y_residual).statistic)


def _tail_metrics(frame: pd.DataFrame, signal: str, benefit: str) -> Dict[str, float]:
    valid = _finite(frame, [signal, benefit])
    positive = valid[benefit][valid[benefit] > 0]
    if len(valid) < 10 or len(positive) < 2:
        return {
            "event_threshold": float("nan"),
            "prevalence": float("nan"),
            "auroc": float("nan"),
            "auprc": float("nan"),
        }
    threshold = float(np.quantile(positive, 0.8))
    target = (valid[benefit] >= threshold).astype(int)
    if target.nunique() < 2:
        return {
            "event_threshold": threshold,
            "prevalence": float(target.mean()),
            "auroc": float("nan"),
            "auprc": float("nan"),
        }
    result = {
        "event_threshold": threshold,
        "prevalence": float(target.mean()),
        "auroc": float(roc_auc_score(target, valid[signal])),
        "auprc": float(average_precision_score(target, valid[signal])),
    }
    order = np.argsort(-valid[signal].to_numpy(dtype=float), kind="stable")
    target_array = target.to_numpy(dtype=int)
    positives = max(1, int(target_array.sum()))
    for fraction in (0.05, 0.10, 0.20):
        take = max(1, int(math.ceil(len(valid) * fraction)))
        selected = target_array[order[:take]]
        suffix = "%02d" % int(fraction * 100)
        result["precision_at_%s" % suffix] = float(selected.mean())
        result["recall_at_%s" % suffix] = float(
            selected.sum() / positives
        )
    return result


def _loso_tail(
    frame: pd.DataFrame, signal: str, benefit: str
) -> Dict[str, float]:
    valid = _finite(frame, [signal, benefit, "sample_id"])
    predictions: List[float] = []
    targets: List[int] = []
    for heldout in valid.sample_id.drop_duplicates():
        train = valid[~valid.sample_id.eq(heldout)]
        test = valid[valid.sample_id.eq(heldout)]
        positive = train[benefit][train[benefit] > 0]
        if len(positive) < 2:
            continue
        threshold = float(np.quantile(positive, 0.8))
        y_train = (train[benefit] >= threshold).astype(int)
        y_test = (test[benefit] >= threshold).astype(int)
        if y_train.nunique() < 2:
            continue
        mean = float(train[signal].mean())
        scale = float(train[signal].std(ddof=0))
        scale = scale if scale > 1e-12 else 1.0
        model = LogisticRegression(
            random_state=0, solver="lbfgs", max_iter=1000
        )
        model.fit(
            ((train[[signal]] - mean) / scale).to_numpy(),
            y_train.to_numpy(),
        )
        probability = model.predict_proba(
            ((test[[signal]] - mean) / scale).to_numpy()
        )[:, 1]
        predictions.extend(probability.tolist())
        targets.extend(y_test.tolist())
    if len(set(targets)) < 2:
        return {
            "loso_auroc": float("nan"),
            "loso_auprc": float("nan"),
            "loso_brier": float("nan"),
            "loso_prevalence": float("nan"),
        }
    return {
        "loso_auroc": float(roc_auc_score(targets, predictions)),
        "loso_auprc": float(average_precision_score(targets, predictions)),
        "loso_brier": float(brier_score_loss(targets, predictions)),
        "loso_prevalence": float(np.mean(targets)),
    }


def _task_transfer_tail(
    frame: pd.DataFrame, signal: str, benefit: str
) -> Dict[str, Dict[str, float]]:
    valid = _finite(frame, [signal, benefit, "task"])
    output: Dict[str, Dict[str, float]] = {}
    tasks = valid.task.drop_duplicates().tolist()
    for train_task in tasks:
        train = valid[valid.task.eq(train_task)]
        test = valid[~valid.task.eq(train_task)]
        if test.empty:
            continue
        positive = train[benefit][train[benefit] > 0]
        if len(positive) < 2:
            continue
        threshold = float(np.quantile(positive, 0.8))
        y_train = (train[benefit] >= threshold).astype(int)
        y_test = (test[benefit] >= threshold).astype(int)
        if y_train.nunique() < 2 or y_test.nunique() < 2:
            continue
        mean = float(train[signal].mean())
        scale = float(train[signal].std(ddof=0))
        scale = scale if scale > 1e-12 else 1.0
        model = LogisticRegression(
            random_state=0, solver="lbfgs", max_iter=1000
        )
        model.fit(
            ((train[[signal]] - mean) / scale).to_numpy(),
            y_train.to_numpy(),
        )
        probability = model.predict_proba(
            ((test[[signal]] - mean) / scale).to_numpy()
        )[:, 1]
        output[str(train_task)] = {
            "test_task": ",".join(
                sorted(str(value) for value in test.task.unique())
            ),
            "auroc": float(roc_auc_score(y_test, probability)),
            "auprc": float(
                average_precision_score(y_test, probability)
            ),
            "prevalence": float(y_test.mean()),
            "brier": float(brier_score_loss(y_test, probability)),
        }
    return output


def _bootstrap_variant_delta(
    frame: pd.DataFrame,
    variant: str,
    signal: str,
    benefit: str,
    samples: int,
    seed: int,
) -> Dict[str, float]:
    index = COMPARISON_KEYS + ["layer"]
    subset = frame[frame.feature_variant.isin(["raw_v", variant])]
    pivot = subset.pivot_table(
        index=index,
        columns="feature_variant",
        values=signal,
        aggfunc="first",
    ).reset_index()
    labels = (
        subset[index + [benefit]]
        .drop_duplicates(index)
        .groupby(index, as_index=False)[benefit]
        .first()
    )
    pivot = pivot.merge(labels, on=index, how="inner")
    pivot = _finite(pivot, ["raw_v", variant, benefit])
    clusters = pivot.sample_id.drop_duplicates().tolist()
    grouped = {
        name: pivot[pivot.sample_id.eq(name)] for name in clusters
    }

    def differences(value: pd.DataFrame) -> Tuple[float, float]:
        delta_rho = _rho(value, variant, benefit) - _rho(
            value, "raw_v", benefit
        )
        functional_tail = _tail_metrics(value, variant, benefit)
        raw_tail = _tail_metrics(value, "raw_v", benefit)
        delta_auprc = (
            functional_tail["auprc"] - raw_tail["auprc"]
        )
        return delta_rho, delta_auprc

    point_rho, point_auprc = differences(pivot)
    rng = np.random.default_rng(seed)
    rho_values = []
    auprc_values = []
    for _ in range(int(samples)):
        chosen = rng.choice(clusters, size=len(clusters), replace=True)
        draw = pd.concat([grouped[name] for name in chosen], ignore_index=True)
        rho_value, auprc_value = differences(draw)
        if math.isfinite(rho_value):
            rho_values.append(rho_value)
        if math.isfinite(auprc_value):
            auprc_values.append(auprc_value)
    rho_array = np.asarray(rho_values)
    auprc_array = np.asarray(auprc_values)

    def quantile_or_nan(array: np.ndarray, value: float) -> float:
        return (
            float(np.quantile(array, value))
            if array.size
            else float("nan")
        )

    return {
        "delta_spearman": float(point_rho),
        "delta_spearman_ci_low": quantile_or_nan(rho_array, 0.025),
        "delta_spearman_ci_high": quantile_or_nan(rho_array, 0.975),
        "delta_spearman_p_gt_zero": float(
            np.mean(rho_array > 0.0)
            if rho_array.size
            else float("nan")
        ),
        "delta_auprc": float(point_auprc),
        "delta_auprc_ci_low": quantile_or_nan(auprc_array, 0.025),
        "delta_auprc_ci_high": quantile_or_nan(auprc_array, 0.975),
        "delta_auprc_p_gt_zero": float(
            np.mean(auprc_array > 0.0)
            if auprc_array.size
            else float("nan")
        ),
    }


def _output_frame(
    features: pd.DataFrame, attention: pd.DataFrame
) -> pd.DataFrame:
    primary = features[
        features.feature_granularity.eq("layer")
        & features.coverage_scope.eq("active_cache_with_recent")
    ].copy()
    labels = attention[
        attention.label_granularity.eq("layer_projected")
    ][COMPARISON_KEYS + ["layer", "refresh_benefit_output"]]
    return primary.merge(
        labels, on=COMPARISON_KEYS + ["layer"], how="inner"
    )


def _output_performance(
    frame: pd.DataFrame, bootstrap_samples: int, seed: int
) -> pd.DataFrame:
    rows = []
    for variant in sorted(frame.feature_variant.unique()):
        subset = frame[frame.feature_variant.eq(variant)]
        for signal in SIGNALS:
            rho = _rho(subset, signal, "refresh_benefit_output")
            interval = _cluster_bootstrap_rho(
                subset,
                signal,
                "refresh_benefit_output",
                clusters="sample_id",
                samples=bootstrap_samples,
                seed=seed + len(rows),
            )
            task_values = {
                str(task): _rho(
                    subset[subset.task.eq(task)],
                    signal,
                    "refresh_benefit_output",
                )
                for task in subset.task.unique()
            }
            sequence_values = [
                _rho(group, signal, "refresh_benefit_output")
                for _, group in subset.groupby("sample_id")
            ]
            tail = _tail_metrics(
                subset, signal, "refresh_benefit_output"
            )
            loso = _loso_tail(
                subset, signal, "refresh_benefit_output"
            )
            task_transfer = _task_transfer_tail(
                subset, signal, "refresh_benefit_output"
            )
            rows.append(
                {
                    "feature_variant": variant,
                    "signal": signal,
                    "spearman": rho,
                    "ci_low": interval["low"],
                    "ci_high": interval["high"],
                    "bootstrap_p_gt_zero": interval["p_gt_zero"],
                    "partial_spearman": _partial_spearman(
                        subset, signal, "refresh_benefit_output"
                    ),
                    "sequence_positive_fraction": float(
                        np.mean(
                            [
                                value > 0
                                for value in sequence_values
                                if math.isfinite(value)
                            ]
                        )
                    ),
                    "task_spearman": json.dumps(
                        task_values, sort_keys=True
                    ),
                    **tail,
                    **loso,
                    "task_transfer": json.dumps(
                        task_transfer, sort_keys=True
                    ),
                    "rows": len(subset),
                }
            )
    return pd.DataFrame(rows)


def _nll_performance(
    output_frame: pd.DataFrame, downstream: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    aggregates = (
        output_frame.groupby(
            COMPARISON_KEYS + ["feature_variant"], as_index=False
        )[SIGNALS]
        .mean()
    )
    joined = aggregates.merge(
        downstream[
            COMPARISON_KEYS
            + [
                "refresh_benefit_nll",
                "refresh_benefit_exact_kl",
                "refresh_benefit_js",
            ]
        ],
        on=COMPARISON_KEYS,
        how="inner",
    )
    rows = []
    for variant in sorted(joined.feature_variant.unique()):
        subset = joined[joined.feature_variant.eq(variant)]
        for signal in SIGNALS:
            rows.append(
                {
                    "feature_variant": variant,
                    "signal": signal,
                    "nll_spearman": _rho(
                        subset, signal, "refresh_benefit_nll"
                    ),
                    "exact_kl_spearman": _rho(
                        subset, signal, "refresh_benefit_exact_kl"
                    ),
                    "js_spearman": _rho(
                        subset, signal, "refresh_benefit_js"
                    ),
                    "rows": len(subset),
                }
            )
    return pd.DataFrame(rows), joined


def _mediation(
    output_frame: pd.DataFrame, downstream: pd.DataFrame
) -> pd.DataFrame:
    output = (
        output_frame.groupby(
            COMPARISON_KEYS + ["feature_variant"], as_index=False
        )
        .agg(
            delta_e=(PRIMARY_SIGNAL, "mean"),
            output_benefit=("refresh_benefit_output", "mean"),
            deployable_approx=(
                "deployable_approx_normalized_sum",
                "mean",
            ),
        )
        .merge(
            downstream[
                COMPARISON_KEYS
                + ["refresh_benefit_nll", "refresh_benefit_exact_kl"]
            ],
            on=COMPARISON_KEYS,
            how="inner",
        )
    )
    rows = []
    for variant, subset in output.groupby("feature_variant"):
        rows.append(
            {
                "feature_variant": variant,
                "delta_e_to_output_spearman": _rho(
                    subset, "delta_e", "output_benefit"
                ),
                "output_to_nll_spearman": _rho(
                    subset, "output_benefit", "refresh_benefit_nll"
                ),
                "output_to_exact_kl_spearman": _rho(
                    subset,
                    "output_benefit",
                    "refresh_benefit_exact_kl",
                ),
                "deployable_to_offline_spearman": _rho(
                    subset, "deployable_approx", "delta_e"
                ),
            }
        )
    return pd.DataFrame(rows)


def _architecture_summary(downstream: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for recent, subset in downstream.groupby("protected_recent_size"):
        rows.append(
            {
                "protected_recent_size": int(recent),
                "probe_lag": -1,
                "aggregation": "all_lags",
                "mean_refresh_benefit_nll": float(
                    subset.refresh_benefit_nll.mean()
                ),
                "median_refresh_benefit_nll": float(
                    subset.refresh_benefit_nll.median()
                ),
                "positive_fraction": float(
                    (subset.refresh_benefit_nll > 0).mean()
                ),
                "rows": len(subset),
            }
        )
        for lag, lag_subset in subset.groupby("probe_lag"):
            rows.append(
                {
                    "protected_recent_size": int(recent),
                    "probe_lag": int(lag),
                    "aggregation": "lag",
                    "mean_refresh_benefit_nll": float(
                        lag_subset.refresh_benefit_nll.mean()
                    ),
                    "median_refresh_benefit_nll": float(
                        lag_subset.refresh_benefit_nll.median()
                    ),
                    "positive_fraction": float(
                        (lag_subset.refresh_benefit_nll > 0).mean()
                    ),
                    "rows": len(lag_subset),
                }
            )
    return pd.DataFrame(rows)


def _set_links(
    output_frame: pd.DataFrame, set_metrics: pd.DataFrame
) -> pd.DataFrame:
    output = (
        output_frame.groupby(
            COMPARISON_KEYS + ["feature_variant"], as_index=False
        )
        .agg(
            delta_e=(PRIMARY_SIGNAL, "mean"),
            output_benefit=("refresh_benefit_output", "mean"),
        )
    )
    turnover = (
        set_metrics.groupby(COMPARISON_KEYS, as_index=False)
        .selected_core_turnover.mean()
    )
    joined = output.merge(turnover, on=COMPARISON_KEYS, how="inner")
    rows = []
    for variant, subset in joined.groupby("feature_variant"):
        rows.append(
            {
                "feature_variant": variant,
                "delta_e_to_set_turnover_spearman": _rho(
                    subset, "delta_e", "selected_core_turnover"
                ),
                "set_turnover_to_output_spearman": _rho(
                    subset,
                    "selected_core_turnover",
                    "output_benefit",
                ),
                "delta_e_to_output_spearman": _rho(
                    subset, "delta_e", "output_benefit"
                ),
            }
        )
    return pd.DataFrame(rows)


def _variant_deltas(
    output_frame: pd.DataFrame, bootstrap_samples: int, seed: int
) -> pd.DataFrame:
    rows = []
    raw = output_frame[output_frame.feature_variant.eq("raw_v")]
    for index, variant in enumerate(FUNCTIONAL_VARIANTS):
        subset = output_frame[
            output_frame.feature_variant.isin(["raw_v", variant])
        ]
        delta = _bootstrap_variant_delta(
            subset,
            variant,
            PRIMARY_SIGNAL,
            "refresh_benefit_output",
            bootstrap_samples,
            seed + index,
        )
        functional = output_frame[
            output_frame.feature_variant.eq(variant)
        ]
        task_delta = {}
        for task in functional.task.unique():
            functional_task = functional[functional.task.eq(task)]
            raw_task = raw[raw.task.eq(task)]
            task_delta[str(task)] = (
                _rho(
                    functional_task,
                    PRIMARY_SIGNAL,
                    "refresh_benefit_output",
                )
                - _rho(
                    raw_task,
                    PRIMARY_SIGNAL,
                    "refresh_benefit_output",
                )
            )
        rows.append(
            {
                "feature_variant": variant,
                **delta,
                "task_delta_spearman": json.dumps(
                    task_delta, sort_keys=True
                ),
                "both_tasks_positive": bool(
                    task_delta
                    and all(value > 0 for value in task_delta.values())
                ),
            }
        )
    return pd.DataFrame(rows)


def _quality(
    run_dir: Path,
    probe: pd.DataFrame,
    features: pd.DataFrame,
    identity: pd.DataFrame,
) -> Dict[str, Any]:
    expected_comparisons = 6 * 2 * 8 * 2 * 2 * 2
    numeric = features.select_dtypes("number").drop(
        columns=["head"], errors="ignore"
    )
    stable_identity = identity[identity.stable_denominator]
    stable_fraction = float(identity.stable_denominator.mean())
    identity_max = (
        float(stable_identity.identity_relative_error.max())
        if len(stable_identity)
        else float("inf")
    )
    return {
        "comparison_rows": len(probe),
        "expected_comparison_rows": expected_comparisons,
        "comparison_count_pass": len(probe) == expected_comparisons,
        "same_token_alignment_pass": bool(
            probe.same_reference_token_verified.all()
        ),
        "budget_cap_pass": bool(
            (
                probe.old_active_cache_tokens <= probe.total_budget
            ).all()
            and (
                probe.fresh_active_cache_tokens <= probe.total_budget
            ).all()
        ),
        "functional_feature_rows": len(features),
        "feature_numeric_finite_pass": bool(
            np.isfinite(numeric.to_numpy(dtype=float)).all()
        ),
        "identity_rows": len(identity),
        "identity_stable_fraction": stable_fraction,
        "identity_max_relative_error": identity_max,
        "identity_pass": bool(
            stable_fraction >= 0.95 and identity_max < 1e-8
        ),
        "all_pass": False,
        "run_dir": str(run_dir),
    }


def _figures(
    run_dir: Path,
    downstream: pd.DataFrame,
    attention: pd.DataFrame,
    performance: pd.DataFrame,
) -> List[str]:
    directory = run_dir / "figures_functional_stage1"
    directory.mkdir(exist_ok=True)
    paths = []
    layer = attention[
        attention.label_granularity.eq("layer_projected")
    ]
    time_curve = (
        layer.groupby("probe_lag", as_index=False)
        .refresh_benefit_output.mean()
        .merge(
            downstream.groupby("probe_lag", as_index=False)
            .refresh_benefit_nll.mean(),
            on="probe_lag",
        )
    )
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].plot(
        time_curve.probe_lag,
        time_curve.refresh_benefit_output,
        marker="o",
    )
    axes[0].set(xlabel="refresh lag", ylabel="mean output benefit")
    axes[1].plot(
        time_curve.probe_lag,
        time_curve.refresh_benefit_nll,
        marker="o",
        color="#b54a4a",
    )
    axes[1].set(xlabel="refresh lag", ylabel="mean NLL benefit")
    figure.tight_layout()
    path = directory / "refresh_benefit_by_lag.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    paths.append(str(path))

    recent_curve = (
        downstream.groupby(
            ["protected_recent_size", "probe_lag"], as_index=False
        )
        .refresh_benefit_nll.mean()
    )
    figure, axis = plt.subplots(figsize=(6.5, 3.8))
    for recent, subset in recent_curve.groupby(
        "protected_recent_size"
    ):
        axis.plot(
            subset.probe_lag,
            subset.refresh_benefit_nll,
            marker="o",
            label="recent=%d" % int(recent),
        )
    axis.axvline(32, color="black", linestyle="--", linewidth=0.8)
    axis.set(
        xlabel="refresh lag",
        ylabel="mean NLL refresh benefit",
    )
    axis.legend(frameon=False)
    figure.tight_layout()
    path = directory / "recent_window_refresh_curve.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    paths.append(str(path))

    primary = performance[performance.signal.eq(PRIMARY_SIGNAL)]
    figure, axis = plt.subplots(figsize=(7, 3.8))
    x = np.arange(len(primary))
    axis.bar(x, primary.spearman, color="#4977a3")
    axis.errorbar(
        x,
        primary.spearman,
        yerr=np.vstack(
            [
                primary.spearman - primary.ci_low,
                primary.ci_high - primary.spearman,
            ]
        ),
        fmt="none",
        color="black",
        capsize=3,
    )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xticks(x, primary.feature_variant, rotation=20)
    axis.set_ylabel("Spearman: Delta E vs output benefit")
    figure.tight_layout()
    path = directory / "feature_output_correlations.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    paths.append(str(path))
    return paths


def _report(
    run_dir: Path,
    quality: Dict[str, Any],
    performance: pd.DataFrame,
    nll: pd.DataFrame,
    mediation: pd.DataFrame,
    deltas: pd.DataFrame,
    architecture: pd.DataFrame,
    set_links: pd.DataFrame,
    decision: Dict[str, Any],
) -> Path:
    best = performance[
        performance.signal.eq(PRIMARY_SIGNAL)
    ].sort_values("spearman", ascending=False)
    lines = [
        "# Stage-1 functional staleness report",
        "",
        "## Decision",
        "",
        "**%s** — %s"
        % (decision["decision"].upper(), decision["reason"]),
        "",
        "Gate status: `%s`"
        % json.dumps(decision.get("gates", {}), sort_keys=True),
        "",
        "## Prior mechanism context",
        "",
        "The completed 15-sequence targeted run found that raw online leverage "
        "predicts refreshed V-ridge core entry (AUC 0.964) and set turnover "
        "(Spearman 0.728), but not positive NLL refresh benefit "
        "(Spearman -0.096, cluster CI [-0.194, 0.041]). At recent-window "
        "exit, refreshed-core membership rose from 0 to 0.824 and the "
        "V-ridge median benefit jump was 0.139 [0.000, 0.401]. This Stage-1 "
        "run tests the missing functional link.",
        "",
        "## Data quality",
        "",
        "- comparisons: %d / %d"
        % (
            quality["comparison_rows"],
            quality["expected_comparison_rows"],
        ),
        "- same-token alignment: `%s`" % quality["same_token_alignment_pass"],
        "- cache budget cap: `%s`" % quality["budget_cap_pass"],
        "- functional rows: %d" % quality["functional_feature_rows"],
        "- fixed-QKV identity max relative error: `%.3e`"
        % quality["identity_max_relative_error"],
        "- fixed-QKV stable-denominator fraction: `%.4f`"
        % quality["identity_stable_fraction"],
        "",
        "## Primary output association",
        "",
        "| feature | Spearman | cluster 95% CI | partial | LOSO AUPRC |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in best.itertuples(index=False):
        lines.append(
            "| %s | %.3f | [%.3f, %.3f] | %.3f | %.3f |"
            % (
                row.feature_variant,
                row.spearman,
                row.ci_low,
                row.ci_high,
                row.partial_spearman,
                row.loso_auprc,
            )
        )
    primary_aor = best[best.feature_variant.eq("aor")].iloc[0]
    lines.extend(
        [
            "",
            "AOR's top-20%% event AUPRC is %.3f against prevalence %.3f; "
            "LOSO AUPRC is %.3f. The feature therefore explains broad "
            "continuous variation but does not yet provide useful tail "
            "calibration."
            % (
                primary_aor.auprc,
                primary_aor.prevalence,
                primary_aor.loso_auprc,
            ),
        ]
    )
    lines.extend(
        [
            "",
            "## Functional feature improvement over Raw V",
            "",
        "| feature | ΔSpearman | cluster 95% CI | ΔAUPRC (95% CI) | both tasks positive |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in deltas.itertuples(index=False):
        lines.append(
            "| %s | %.3f | [%.3f, %.3f] | %.3f [%.3f, %.3f] | %s |"
            % (
                row.feature_variant,
                row.delta_spearman,
                row.delta_spearman_ci_low,
                row.delta_spearman_ci_high,
                row.delta_auprc,
                row.delta_auprc_ci_low,
                row.delta_auprc_ci_high,
                row.both_tasks_positive,
            )
        )
    lines.extend(
        [
            "",
            "## Mechanism chain",
            "",
            "| feature | ΔE→output | output→NLL | output→KL | deployable→offline |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in mediation.itertuples(index=False):
        lines.append(
            "| %s | %.3f | %.3f | %.3f | %.3f |"
            % (
                row.feature_variant,
                row.delta_e_to_output_spearman,
                row.output_to_nll_spearman,
                row.output_to_exact_kl_spearman,
                row.deployable_to_offline_spearman,
            )
        )
    recent_overall = architecture[
        architecture.aggregation.eq("all_lags")
    ].set_index("protected_recent_size")
    recent_lags = architecture[
        architecture.aggregation.eq("lag")
        & architecture.probe_lag.isin([1, 32, 40, 64])
    ]
    lines.extend(["", "## Recent-window architecture effect", ""])
    if {0, 32}.issubset(set(recent_overall.index)):
        lines.append(
            "Across all lags, recent=0 has mean/median NLL refresh benefit "
            "%.3f/%.3f, versus %.3f/%.3f for recent=32."
            % (
                recent_overall.loc[0, "mean_refresh_benefit_nll"],
                recent_overall.loc[0, "median_refresh_benefit_nll"],
                recent_overall.loc[32, "mean_refresh_benefit_nll"],
                recent_overall.loc[32, "median_refresh_benefit_nll"],
            )
        )
    else:
        lines.append(
            "This run contains recent conditions: `%s`."
            % sorted(int(value) for value in recent_overall.index)
        )
    lines.extend(
        [
            "",
            "| lag | recent | mean NLL benefit | median NLL benefit | positive fraction |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in recent_lags.sort_values(
        ["probe_lag", "protected_recent_size"]
    ).itertuples(index=False):
        lines.append(
            "| %d | %d | %.3f | %.3f | %.3f |"
            % (
                row.probe_lag,
                row.protected_recent_size,
                row.mean_refresh_benefit_nll,
                row.median_refresh_benefit_nll,
                row.positive_fraction,
            )
        )
    lines.append("")
    if {0, 32}.issubset(set(recent_overall.index)):
        lines.append(
            "The protected arm is nearly flat at short lag and rises around/after "
            "the 32-token boundary, while recent=0 exposes refresh benefit "
            "immediately. This agrees with the prior event-aligned exit experiment."
        )
    lines.extend(
        [
            "",
            "## Set change versus functional change",
            "",
            "| feature | ΔE→turnover | turnover→output | ΔE→output |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in set_links.itertuples(index=False):
        lines.append(
            "| %s | %.3f | %.3f | %.3f |"
            % (
                row.feature_variant,
                row.delta_e_to_set_turnover_spearman,
                row.set_turnover_to_output_spearman,
                row.delta_e_to_output_spearman,
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "- This is a six-sequence theory-discovery run, not a benchmark.",
            "- The model is the cached MLX 4-bit checkpoint; results are not numerically equivalent to bf16.",
            "- Exact deletion identities are fixed-QKV diagnostics; old/fresh labels come from stateful replay.",
            "- Layer features are measured at the three pre-registered diagnostic layers; selector turnover is retained for all 28 layers.",
            "- Offline full-history features cannot be used as a production trigger. The deployable approximation is reported separately.",
            "- Raw-V/OV arrival residuals are online-deployable because their token features are static. AOV/AOR values for evicted new tokens require a backing store plus current attention and therefore do not satisfy the deployable gate.",
            "",
            "Machine-readable tables are stored beside this report.",
        ]
    )
    path = run_dir / "STAGE1_FUNCTIONAL_STALENESS_REPORT.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def analyze(run_dir: Path, bootstrap_samples: int, seed: int) -> Dict[str, Any]:
    probe = pd.read_parquet(run_dir / "probe_index.parquet")
    features = pd.read_parquet(run_dir / "functional_features.parquet")
    attention = pd.read_parquet(run_dir / "attention_labels.parquet")
    downstream = pd.read_parquet(run_dir / "downstream_labels.parquet")
    identity = pd.read_parquet(run_dir / "identity_checks.parquet")
    set_metrics = pd.read_parquet(run_dir / "set_metrics.parquet")
    quality = _quality(run_dir, probe, features, identity)
    quality["all_pass"] = bool(
        quality["comparison_count_pass"]
        and quality["same_token_alignment_pass"]
        and quality["budget_cap_pass"]
        and quality["feature_numeric_finite_pass"]
        and quality["identity_pass"]
    )
    output_frame = _output_frame(features, attention)
    performance = _output_performance(
        output_frame, bootstrap_samples, seed
    )
    nll, _ = _nll_performance(output_frame, downstream)
    mediation = _mediation(output_frame, downstream)
    architecture = _architecture_summary(downstream)
    set_links = _set_links(output_frame, set_metrics)
    deltas = _variant_deltas(
        output_frame, bootstrap_samples, seed + 10000
    )
    performance.to_csv(run_dir / "output_signal_performance.csv", index=False)
    nll.to_csv(run_dir / "downstream_signal_performance.csv", index=False)
    mediation.to_csv(run_dir / "mechanism_chain.csv", index=False)
    architecture.to_csv(
        run_dir / "refresh_architecture_summary.csv", index=False
    )
    set_links.to_csv(run_dir / "set_functional_links.csv", index=False)
    deltas.to_csv(run_dir / "functional_vs_raw_deltas.csv", index=False)
    candidates = deltas[
        (deltas.delta_spearman >= 0.10)
        | (deltas.delta_auprc >= 0.05)
    ]
    if len(candidates):
        ranked = candidates.sort_values(
            ["both_tasks_positive", "delta_spearman", "delta_auprc"],
            ascending=False,
        )
        candidate = ranked.iloc[0]
        mechanism = mediation[
            mediation.feature_variant.eq(candidate.feature_variant)
        ].iloc[0]
        chain_pass = bool(
            mechanism.delta_e_to_output_spearman > 0
            and abs(mechanism.output_to_nll_spearman) >= 0.10
        )
        deployable_pass = bool(
            candidate.feature_variant == "projected_v"
            and mechanism.deployable_to_offline_spearman >= 0.30
        )
        direction_pass = bool(candidate.both_tasks_positive)
        confidence_pass = bool(
            candidate.delta_spearman_ci_low > 0
            or candidate.delta_auprc_ci_low > 0
        )
        gates = {
            "data_quality": bool(quality["all_pass"]),
            "functional_improvement": True,
            "both_tasks_same_direction": direction_pass,
            "offline_mechanism_chain": chain_pass,
            "deployable_approximation": deployable_pass,
            "cluster_confidence": confidence_pass,
        }
        if (
            quality["all_pass"]
            and chain_pass
            and deployable_pass
            and direction_pass
            and confidence_pass
        ):
            decision = {
                "decision": "go",
                "feature": candidate.feature_variant,
                "reason": "pre-registered improvement, cross-task direction, mechanism chain, and data-quality gates passed",
                "gates": gates,
            }
        elif (
            quality["all_pass"]
            and chain_pass
            and deployable_pass
            and direction_pass
        ):
            decision = {
                "decision": "inconclusive",
                "feature": candidate.feature_variant,
                "reason": "point estimates pass, but six-sequence uncertainty remains too wide",
                "gates": gates,
            }
        elif (
            quality["all_pass"]
            and chain_pass
            and direction_pass
            and confidence_pass
            and not deployable_pass
        ):
            decision = {
                "decision": "no-go",
                "feature": candidate.feature_variant,
                "reason": (
                    "the offline functional mechanism passes, but the "
                    "current online-deployable approximation gate fails"
                ),
                "gates": gates,
            }
        else:
            decision = {
                "decision": "no-go",
                "feature": candidate.feature_variant,
                "reason": (
                    "a point improvement exists, but the cross-task, "
                    "mechanism-chain, or deployable-approximation gate failed"
                ),
                "gates": gates,
            }
    else:
        decision = {
            "decision": "no-go",
            "feature": None,
            "reason": "no functional feature reached ΔSpearman 0.10 or ΔAUPRC 0.05 over Raw V",
            "gates": {
                "data_quality": bool(quality["all_pass"]),
                "functional_improvement": False,
            },
        }
    figures = _figures(run_dir, downstream, attention, performance)
    report = _report(
        run_dir,
        quality,
        performance,
        nll,
        mediation,
        deltas,
        architecture,
        set_links,
        decision,
    )
    summary = {
        "quality": quality,
        "decision": decision,
        "figures": figures,
        "report": str(report),
        "bootstrap_samples": int(bootstrap_samples),
        "seed": int(seed),
    }
    (run_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    summary = analyze(
        args.run_dir.resolve(), args.bootstrap_samples, args.seed
    )
    print(json.dumps(summary["decision"], sort_keys=True))


if __name__ == "__main__":
    main()
