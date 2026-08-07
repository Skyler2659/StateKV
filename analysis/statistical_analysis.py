#!/usr/bin/env python3
"""Exploratory, sample-clustered tests of the proposed mechanism links."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import cluster_bootstrap_correlation, write_dual


KEY = ["sample_id", "task", "anchor", "strategy", "layer", "lag"]


def _link(
    frame: pd.DataFrame,
    relationship: str,
    x: str,
    y: str,
    extra: list[str] | None = None,
) -> pd.DataFrame:
    frame = frame.copy()
    if "sample_cluster" not in frame:
        frame["sample_cluster"] = (
            frame["task"].astype(str) + "::" + frame["sample_id"].astype(str)
        )
    columns = [
        "sample_id",
        "sample_cluster",
        "task",
        "anchor",
        "strategy",
        "layer",
        "head",
        "lag",
    ]
    columns += extra or []
    available = [column for column in columns if column in frame]
    result = frame[available + [x, y]].copy()
    result = result.rename(columns={x: "x_value", y: "y_value"})
    result["relationship"] = relationship
    result["x_metric"] = x
    result["y_metric"] = y
    return result


def _sample_consistency(frame: pd.DataFrame) -> tuple[float, int]:
    signs = []
    for _, group in frame.groupby("sample_cluster"):
        clean = group[["x_value", "y_value"]].dropna()
        if len(clean) >= 3 and clean["x_value"].nunique() > 1 and clean["y_value"].nunique() > 1:
            value = spearmanr(clean["x_value"], clean["y_value"]).statistic
            if np.isfinite(value):
                signs.append(float(value))
    if not signs:
        return float("nan"), 0
    global_sign = np.sign(
        spearmanr(frame["x_value"], frame["y_value"], nan_policy="omit").statistic
    )
    return float(np.mean(np.sign(signs) == global_sign)), len(signs)


def _binned(frame: pd.DataFrame, bins: int = 5) -> pd.DataFrame:
    rows = []
    for relationship, group in frame.groupby("relationship"):
        clean = group.dropna(subset=["x_value", "y_value"]).copy()
        if len(clean) < bins:
            continue
        try:
            clean["x_bin"] = pd.qcut(clean["x_value"], bins, duplicates="drop")
        except ValueError:
            continue
        for index, (_, bucket) in enumerate(clean.groupby("x_bin", observed=True), 1):
            sample_means = bucket.groupby("sample_cluster")["y_value"].mean()
            rows.append(
                {
                    "relationship": relationship,
                    "bin": index,
                    "x_mean": float(bucket["x_value"].mean()),
                    "x_median": float(bucket["x_value"].median()),
                    "y_mean": float(bucket["y_value"].mean()),
                    "y_median": float(bucket["y_value"].median()),
                    "y_sample_mean_se": (
                        float(sample_means.std(ddof=1) / np.sqrt(len(sample_means)))
                        if len(sample_means) > 1
                        else np.nan
                    ),
                    "n_rows": int(len(bucket)),
                    "n_samples": int(bucket["sample_cluster"].nunique()),
                }
            )
    return pd.DataFrame(rows)


def _multivariable_refresh_model(
    refresh: pd.DataFrame,
    residual: pd.DataFrame,
    sets: pd.DataFrame,
    score: pd.DataFrame,
    attention_drift: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = refresh[refresh["record_scope"].eq("global_output")].copy()
    base = base[~base["strategy"].eq("future_attention_oracle")].copy()
    agg_resid = (
        residual.groupby(["sample_id", "task", "anchor", "strategy", "lag"], as_index=False)[
            "future_new_token_residual"
        ]
        .mean()
        .rename(columns={"future_new_token_residual": "mean_selected_core_residual"})
    )
    agg_set = (
        sets.groupby(["sample_id", "task", "anchor", "strategy", "lag"], as_index=False)[
            ["selected_core_turnover", "selection_boundary_margin_future"]
        ]
        .mean()
    )
    agg_attention = (
        attention_drift.groupby(
            ["sample_id", "task", "anchor", "strategy", "lag"], as_index=False
        )["attention_distribution_shift"]
        .mean()
    )
    base = base.rename(columns={"stale_anchor": "anchor", "refresh_lag": "lag"})
    model = base.merge(
        agg_resid, on=["sample_id", "task", "anchor", "strategy", "lag"], how="left"
    )
    model = model.merge(
        agg_set, on=["sample_id", "task", "anchor", "strategy", "lag"], how="left"
    )
    model = model.merge(
        agg_attention,
        on=["sample_id", "task", "anchor", "strategy", "lag"],
        how="left",
    )
    features = [
        "stale_loss",
        "mean_selected_core_residual",
        "selected_core_turnover",
        "attention_distribution_shift",
        "selection_boundary_margin_future",
    ]
    clean = model.dropna(subset=features + ["refresh_benefit"]).copy()
    if clean.empty:
        return pd.DataFrame(), pd.DataFrame()
    means = clean[features].mean()
    stds = clean[features].std(ddof=0).replace(0, 1)
    x = (clean[features] - means) / stds
    y = clean["refresh_benefit"].to_numpy(float)
    design = np.column_stack([np.ones(len(x)), x.to_numpy(float)])
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    fitted = design @ coefficients
    tss = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum((y - fitted) ** 2)) / tss if tss > 0 else np.nan
    coefficient_rows = [
        {
            "model": "standardized_ols_exploratory",
            "term": "intercept",
            "coefficient": float(coefficients[0]),
            "n_rows": len(clean),
            "n_samples": clean["sample_id"].nunique(),
            "in_sample_r2": r2,
        }
    ]
    coefficient_rows += [
        {
            "model": "standardized_ols_exploratory",
            "term": term,
            "coefficient": float(value),
            "n_rows": len(clean),
            "n_samples": clean["sample_id"].nunique(),
            "in_sample_r2": r2,
        }
        for term, value in zip(features, coefficients[1:])
    ]

    predictions = []
    for held_out in clean["sample_id"].unique():
        train = clean[~clean["sample_id"].eq(held_out)]
        test = clean[clean["sample_id"].eq(held_out)]
        train_means = train[features].mean()
        train_stds = train[features].std(ddof=0).replace(0, 1)
        x_train = (train[features] - train_means) / train_stds
        x_test = (test[features] - train_means) / train_stds
        beta = np.linalg.lstsq(
            np.column_stack([np.ones(len(train)), x_train.to_numpy(float)]),
            train["refresh_benefit"].to_numpy(float),
            rcond=None,
        )[0]
        pred = np.column_stack([np.ones(len(test)), x_test.to_numpy(float)]) @ beta
        for index, value in zip(test.index, pred):
            predictions.append(
                {
                    "sample_id": held_out,
                    "task": clean.loc[index, "task"],
                    "strategy": clean.loc[index, "strategy"],
                    "anchor": int(clean.loc[index, "anchor"]),
                    "lag": int(clean.loc[index, "lag"]),
                    "observed_refresh_benefit": float(
                        clean.loc[index, "refresh_benefit"]
                    ),
                    "predicted_refresh_benefit": float(value),
                }
            )
    return pd.DataFrame(coefficient_rows), pd.DataFrame(predictions)


def build(input_dir: Path, analysis_dir: Path, bootstrap_draws: int) -> None:
    out = Path(analysis_dir) / "tables"
    score = pd.read_parquet(out / "score_stability.parquet")
    sets = pd.read_parquet(out / "set_stability.parquet")
    residual = pd.read_parquet(out / "future_selected_core_residuals.parquet")
    attention_drift = pd.read_parquet(Path(input_dir) / "temporal_signals.parquet")
    attention_drift = attention_drift[
        attention_drift["signal_kind"].eq("query_attention_drift")
    ].copy()
    attention_drift["layer"] = attention_drift["layer"].astype(int)
    attention_drift["head"] = attention_drift["head"].astype(int)
    attention_drift["sample_cluster"] = attention_drift["task"].astype(str) + "::" + attention_drift[
        "sample_id"
    ].astype(str)
    attention_drift["attention_distribution_shift"] = 1.0 - attention_drift[
        "attention_distribution_cosine"
    ]
    refresh = pd.read_parquet(out / "refresh_benefit_analysis.parquet")
    head_error = pd.read_parquet(out / "attention_output_by_head.parquet")
    head_error["lag"] = head_error["future_step"]
    validity = pd.read_parquet(out / "validity_horizon_sensitivity.parquet")

    # Same layer/lag links.
    resid_layer = residual.groupby(KEY, as_index=False).agg(
        future_new_token_residual=("future_new_token_residual", "mean")
    )
    resid_layer["sample_cluster"] = resid_layer["task"].astype(str) + "::" + resid_layer[
        "sample_id"
    ].astype(str)
    score_set = score.merge(
        sets[
            KEY
            + [
                "selected_core_turnover",
            ]
        ],
        on=KEY,
        how="inner",
    )
    resid_score = resid_layer.merge(
        score_set,
        on=KEY,
        how="inner",
        suffixes=("", "_score"),
    )
    primary_step = pd.read_parquet(out / "per_step_metrics.parquet")
    primary_step = primary_step[primary_step["analysis_primary"]].rename(
        columns={"future_step": "lag"}
    )
    set_loss = sets.merge(
        primary_step[
            ["sample_id", "task", "anchor", "strategy", "lag", "delta_nll"]
        ],
        on=["sample_id", "task", "anchor", "strategy", "lag"],
        how="inner",
    )
    set_loss["sample_cluster"] = set_loss["task"].astype(str) + "::" + set_loss[
        "sample_id"
    ].astype(str)
    snap_attention = attention_drift.merge(
        sets[
            [
                "sample_id",
                "task",
                "anchor",
                "strategy",
                "layer",
                "lag",
                "selected_core_turnover",
            ]
        ],
        on=["sample_id", "task", "anchor", "strategy", "layer", "lag"],
        how="inner",
    )

    global_refresh = refresh[refresh["record_scope"].eq("global_output")].rename(
        columns={"stale_anchor": "anchor", "refresh_lag": "lag"}
    )
    refresh_resid = global_refresh.merge(
        resid_layer,
        on=["sample_id", "task", "anchor", "strategy", "lag"],
        how="inner",
        suffixes=("", "_resid"),
    )
    refresh_set = global_refresh.merge(
        sets.groupby(
            ["sample_id", "task", "anchor", "strategy", "lag"], as_index=False
        )["selected_core_turnover"].mean(),
        on=["sample_id", "task", "anchor", "strategy", "lag"],
        how="inner",
    )
    refresh_attention = global_refresh.merge(
        attention_drift.groupby(
            ["sample_id", "task", "anchor", "strategy", "lag"], as_index=False
        )["attention_distribution_shift"].mean(),
        on=["sample_id", "task", "anchor", "strategy", "lag"],
        how="inner",
    )
    margin = score.groupby(
        ["sample_id", "task", "anchor", "strategy"], as_index=False
    )["selection_boundary_margin_anchor"].median()
    life = validity[
        validity["definition"].eq("absolute_average_delta_nll")
        & validity["threshold"].eq(0.1)
    ].merge(margin, on=["sample_id", "task", "anchor", "strategy"], how="inner")
    life["sample_cluster"] = life["task"].astype(str) + "::" + life["sample_id"].astype(str)

    links = [
        _link(
            resid_score,
            "selected-core residual -> score instability",
            "future_new_token_residual",
            "score_relative_l2_change",
        ),
        _link(
            resid_score,
            "selected-core residual -> set turnover",
            "future_new_token_residual",
            "selected_core_turnover",
        ),
        _link(
            score_set,
            "score drift -> selected-set turnover",
            "score_relative_l2_change",
            "selected_core_turnover",
        ),
        _link(
            score_set,
            "selection margin -> selected-set turnover",
            "selection_boundary_margin_future",
            "selected_core_turnover",
        ),
        _link(
            snap_attention,
            "attention drift -> selector set turnover",
            "attention_distribution_shift",
            "selected_core_turnover",
        ),
        _link(
            set_loss,
            "selected-set turnover -> delta NLL",
            "selected_core_turnover",
            "delta_nll",
        ),
        _link(
            head_error,
            "attention-output error -> delta NLL",
            "attention_output_relative_error",
            "delta_nll",
        ),
        _link(
            refresh_resid,
            "selected-core residual -> sparse refresh benefit",
            "future_new_token_residual",
            "refresh_benefit",
        ),
        _link(
            refresh_set,
            "selected-set turnover -> sparse refresh benefit",
            "selected_core_turnover",
            "refresh_benefit",
        ),
        _link(
            refresh_attention,
            "attention drift -> sparse refresh benefit",
            "attention_distribution_shift",
            "refresh_benefit",
        ),
        _link(
            global_refresh,
            "stale loss -> sparse refresh benefit",
            "stale_loss",
            "refresh_benefit",
        ),
        _link(
            life,
            "selection margin -> empirical validity horizon",
            "selection_boundary_margin_anchor",
            "observed_horizon",
        ),
    ]
    links = pd.concat(links, ignore_index=True, sort=False)
    write_dual(links, out / "mechanism_links")

    rng = np.random.default_rng(20260724)
    summaries = []
    for relationship, group in links.groupby("relationship"):
        result = cluster_bootstrap_correlation(
            group,
            "sample_cluster",
            "x_value",
            "y_value",
            rng,
            bootstrap_draws,
        )
        consistency, n_consistency = _sample_consistency(group)
        summaries.append(
            {
                "relationship": relationship,
                **result,
                "same_direction_sample_fraction": consistency,
                "n_samples_with_estimable_within_sample_correlation": n_consistency,
                "analysis_status": "exploratory_cluster_bootstrap",
            }
        )
    correlation = pd.DataFrame(summaries)
    write_dual(correlation, out / "mechanism_correlation_summary")
    write_dual(_binned(links), out / "mechanism_binned_trends")

    coefficients, predictions = _multivariable_refresh_model(
        refresh, residual, sets, score, attention_drift
    )
    write_dual(coefficients, out / "refresh_exploratory_model_coefficients")
    write_dual(predictions, out / "refresh_loso_predictions")

    unavailable = pd.DataFrame(
        [
            {
                "relationship": "full-history residual -> score instability",
                "status": "unavailable",
                "reason": "full historical V matrices and future V vectors not persisted",
            },
            {
                "relationship": "online leverage -> turnover/refresh benefit",
                "status": "unavailable",
                "reason": "anchor Gram/factor and future V vectors not persisted",
            },
            {
                "relationship": "covariance/subspace drift -> score/refresh",
                "status": "unavailable",
                "reason": "time-resolved V matrices or sufficient sketches not persisted",
            },
            {
                "relationship": "recent-window exit -> refresh benefit",
                "status": "unavailable",
                "reason": "no dense refreshed-cache arm or token-level event identity",
            },
        ]
    )
    write_dual(unavailable, out / "unavailable_mechanism_links")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, default=Path("analysis"))
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    args = parser.parse_args()
    build(args.input_dir, args.analysis_dir, args.bootstrap_draws)


if __name__ == "__main__":
    main()
