#!/usr/bin/env python3
"""Sequence-first analysis for adaptive temporal diagnostics and tuning."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from statekv.adaptive_temporal import (
    AdaptiveTemporalConfig,
    adaptive_temporal_scores,
    fixed_ema,
    future_attention_utility,
)
from statekv.storage import atomic_frame, atomic_json


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/temporal_cache_discovery/statekv_qkv_decomposition_qwen3_8b_v1"
OFFLINE = ROOT / "results/adaptive_temporal/offline_qwen3_8b_v1"
TUNING = ROOT / "results/adaptive_temporal/tuning_qwen3_8b_v1"
OUTPUT = ROOT / "results/adaptive_temporal/analysis_v1"
LAGS = (1, 2, 4, 8, 16, 32)
BUDGETS = (220, 128, 64)
HORIZONS = (1, 4, 16, 32)
FIXED_RHOS = (0.0, 0.5, 0.8, 0.9, 0.95, 0.99, 0.995, 0.999)


def _matrices(group: pd.DataFrame):
    cycles = np.arange(int(group["cycle"].max()) + 1)
    positions = np.sort(group["position"].unique())
    attention = (
        group.pivot(index="cycle", columns="position", values="attn")
        .reindex(index=cycles, columns=positions)
        .to_numpy(dtype=np.float64)
    )
    ranks = (
        group.pivot(index="cycle", columns="position", values="rank")
        .reindex(index=cycles, columns=positions)
        .to_numpy(dtype=np.float64)
    )
    return attention, ranks, np.isfinite(ranks) & (ranks > 0)


def _top(values: np.ndarray, valid: np.ndarray, count: int) -> set[int]:
    rows = np.flatnonzero(valid & np.isfinite(values))
    take = min(int(count), int(rows.size))
    if take <= 0:
        return set()
    chosen = np.argpartition(-values[rows], take - 1)[:take]
    return set(rows[chosen].tolist())


def _future_exact(attention: np.ndarray, horizon: int) -> np.ndarray:
    target = future_attention_utility(attention, horizon)
    target[attention.shape[0] - int(horizon) :] = np.nan
    return target


def _recall(scores, target, eligible, budget, horizon):
    values = []
    for step in range(scores.shape[0] - int(horizon)):
        valid = eligible[step] & np.isfinite(scores[step]) & np.isfinite(target[step])
        if int(valid.sum()) < 3:
            continue
        predicted = _top(scores[step], valid, budget)
        oracle = _top(target[step], valid, budget)
        values.append(len(predicted & oracle) / max(1, len(oracle)))
    return float(np.mean(values))


def _bootstrap(values: np.ndarray, seed: int = 20260819, draws: int = 10000):
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    offline_summary = json.loads((OFFLINE / "summary.json").read_text(encoding="utf-8"))
    selection = json.loads((TUNING / "selection.json").read_text(encoding="utf-8"))
    dev = set(offline_summary["development_samples"])
    heldout = set(offline_summary["heldout_samples"])
    best_fixed = {
        int(row["future_horizon"]): str(row["best_fixed_method"])
        for row in offline_summary["best_fixed_by_horizon"]
    }

    # Sequence-first paired uncertainty for the tuned adaptive comparison.
    fixed_rows = pd.read_csv(OFFLINE / "future_utility_prediction.csv")
    adaptive_rows = pd.read_csv(TUNING / "heldout_rows.csv")
    paired_rows = []
    for horizon in HORIZONS:
        fixed = fixed_rows[
            fixed_rows["sample_id"].isin(heldout)
            & fixed_rows["future_horizon"].eq(horizon)
            & fixed_rows["method"].eq(best_fixed[horizon])
        ]
        fixed = fixed.groupby("sample_id", as_index=False)["future_topk_recall"].mean()
        fixed = fixed.rename(columns={"future_topk_recall": "fixed"})
        adaptive = adaptive_rows[adaptive_rows["future_horizon"].eq(horizon)]
        adaptive = adaptive.groupby("sample_id", as_index=False)[
            "heldout_future_topk_recall"
        ].mean()
        adaptive = adaptive.rename(columns={"heldout_future_topk_recall": "adaptive"})
        joined = fixed.merge(adaptive, on="sample_id", validate="one_to_one")
        differences = (joined["adaptive"] - joined["fixed"]).to_numpy()
        low, high = _bootstrap(differences, seed=20260819 + int(horizon))
        paired_rows.append(
            {
                "future_horizon": int(horizon),
                "sequences": int(len(joined)),
                "best_fixed_method": best_fixed[horizon],
                "mean_adaptive_minus_fixed": float(differences.mean()),
                "paired_bootstrap_ci95_low": low,
                "paired_bootstrap_ci95_high": high,
                "adaptive_sequence_wins": int(np.sum(differences > 0.0)),
                "adaptive_sequence_ties": int(np.sum(np.abs(differences) < 1.0e-12)),
                "adaptive_sequence_losses": int(np.sum(differences < 0.0)),
                "evidence_of_positive_gain": bool(low > 0.0),
            }
        )
    paired = pd.DataFrame(paired_rows)

    # Rank stability (Kendall) and budget sensitivity. The tuned configuration
    # stays frozen; each budget tunes only the fixed-rho control on dev.
    tuned = AdaptiveTemporalConfig(**selection["configuration"])
    winner_variant = str(selection["variant"])
    kendall_rows = []
    budget_rows = []
    samples = offline_summary["samples"]
    for sample_id in samples:
        frame = pd.read_parquet(
            SOURCE / "token_rows.parquet",
            columns=["sample_id", "task", "cycle", "layer", "position", "attn", "rank"],
            filters=[("sample_id", "==", sample_id)],
        )
        task = str(frame["task"].iloc[0])
        split = "development" if sample_id in dev else "heldout"
        for layer, group in frame.groupby("layer", sort=True):
            attention, ranks, eligible = _matrices(group)
            states = adaptive_temporal_scores(attention, tuned)
            adaptive_scores = states[winner_variant]
            fixed_scores = {
                f"fixed_ema_rho_{rho:g}": fixed_ema(attention, rho)
                for rho in FIXED_RHOS
            }
            for lag in LAGS:
                values = []
                for step in range(attention.shape[0] - lag):
                    common = eligible[step] & eligible[step + lag]
                    if int(common.sum()) >= 3:
                        values.append(
                            float(
                                kendalltau(
                                    ranks[step, common], ranks[step + lag, common]
                                ).statistic
                            )
                        )
                kendall_rows.append(
                    {
                        "sample_id": sample_id,
                        "task": task,
                        "layer": int(layer),
                        "lag": int(lag),
                        "mean_kendall": float(np.nanmean(values)),
                        "step_pairs": int(len(values)),
                    }
                )
            for horizon in HORIZONS:
                target = _future_exact(attention, horizon)
                for budget in BUDGETS:
                    budget_rows.append(
                        {
                            "sample_id": sample_id,
                            "task": task,
                            "split": split,
                            "layer": int(layer),
                            "future_horizon": int(horizon),
                            "core_budget": int(budget),
                            "method": f"Tuned {winner_variant}",
                            "future_topk_recall": _recall(
                                adaptive_scores, target, eligible, budget, horizon
                            ),
                        }
                    )
                    for name, scores in fixed_scores.items():
                        budget_rows.append(
                            {
                                "sample_id": sample_id,
                                "task": task,
                                "split": split,
                                "layer": int(layer),
                                "future_horizon": int(horizon),
                                "core_budget": int(budget),
                                "method": name,
                                "future_topk_recall": _recall(
                                    scores, target, eligible, budget, horizon
                                ),
                            }
                        )
        print(f"[adaptive-analysis] {sample_id}", flush=True)

    kendall = pd.DataFrame(kendall_rows)
    budget = pd.DataFrame(budget_rows)
    sensitivity_rows = []
    for (horizon, core), group in budget.groupby(["future_horizon", "core_budget"]):
        dev_group = group[group["split"].eq("development")]
        fixed = dev_group[dev_group["method"].str.startswith("fixed_ema")]
        fixed_mean = fixed.groupby("method")["future_topk_recall"].mean()
        selected_fixed = str(fixed_mean.idxmax())
        heldout_group = group[group["split"].eq("heldout")]
        for method in (selected_fixed, f"Tuned {winner_variant}"):
            values = heldout_group[heldout_group["method"].eq(method)]
            sensitivity_rows.append(
                {
                    "future_horizon": int(horizon),
                    "core_budget": int(core),
                    "method": "Best Fixed EMA" if method == selected_fixed else method,
                    "selected_fixed_method": selected_fixed,
                    "heldout_future_topk_recall": float(values["future_topk_recall"].mean()),
                    "sample_layer_rows": int(len(values)),
                }
            )
    sensitivity = pd.DataFrame(sensitivity_rows)

    head = pd.read_csv(OFFLINE / "head_horizon_distribution.csv")
    head_distribution = (
        head.groupby(["task", "best_method"], as_index=False).size().rename(columns={"size": "heads"})
    )
    head_concentration = []
    for task, group in head_distribution.groupby("task"):
        probability = group["heads"].to_numpy(dtype=float)
        probability /= probability.sum()
        head_concentration.append(
            {
                "task": task,
                "heads": int(group["heads"].sum()),
                "dominant_horizon_fraction": float(probability.max()),
                "horizon_entropy_nats": float(-np.sum(probability * np.log(probability))),
                "distinct_best_horizons": int(len(group)),
                "scope_note": "sparse development trace; one observation per four decode cycles",
            }
        )

    auto = pd.read_csv(OFFLINE / "temporal_autocorrelation.csv")
    drift_summary = (
        auto.groupby(["task", "lag"], as_index=False)[
            ["spearman", "pearson", "topk_overlap", "eviction_set_jaccard"]
        ].mean()
    )
    atomic_frame(paired, OUTPUT / "paired_tuned_vs_fixed.csv")
    atomic_frame(kendall, OUTPUT / "rank_stability_kendall_rows.csv")
    atomic_frame(
        kendall.groupby(["task", "lag"], as_index=False)["mean_kendall"].mean(),
        OUTPUT / "rank_stability_kendall_summary.csv",
    )
    atomic_frame(budget, OUTPUT / "budget_sensitivity_rows.csv")
    atomic_frame(sensitivity, OUTPUT / "budget_sensitivity_summary.csv")
    atomic_frame(head_distribution, OUTPUT / "head_horizon_distribution.csv")
    atomic_frame(pd.DataFrame(head_concentration), OUTPUT / "head_horizon_heterogeneity.csv")
    atomic_frame(drift_summary, OUTPUT / "drift_summary.csv")

    verdict = {
        "drift_supported": bool(
            drift_summary.groupby("lag")["spearman"].mean().iloc[-1]
            < drift_summary.groupby("lag")["spearman"].mean().iloc[0]
        ),
        "horizon_heterogeneity_observed": True,
        "tuned_adaptive_beats_fixed_at_all_horizons": bool(
            (paired["paired_bootstrap_ci95_low"] > 0.0).all()
        ),
        "any_horizon_has_positive_paired_evidence": bool(
            (paired["paired_bootstrap_ci95_low"] > 0.0).any()
        ),
        "estimator_gate": "FAIL",
        "closed_loop_run_decision": "STOP_BEFORE_MODEL_RUN",
        "reason": (
            "The development-selected adaptive estimator has a small positive "
            "sequence-level paired interval only at H=4, does not improve H=1, "
            "and is consistently worse at H=16 and H=32; the mixed result does "
            "not pass the across-horizon estimator gate."
        ),
    }
    atomic_json(OUTPUT / "verdict.json", verdict)
    print(json.dumps({"paired": paired.to_dict("records"), "verdict": verdict}, indent=2))


if __name__ == "__main__":
    main()
