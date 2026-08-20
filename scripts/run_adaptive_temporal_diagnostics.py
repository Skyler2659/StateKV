#!/usr/bin/env python3
"""Run drift, horizon, future-utility, and adaptive-memory diagnostics.

The input is the existing full-pool attention trace. All estimator scores are
causal. Future attention is materialized only after scoring and is written as
``NON_CAUSAL_ORACLE`` diagnostic evidence, never as a deployable feature.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml
from scipy.stats import rankdata

from statekv.adaptive_temporal import (
    AdaptiveTemporalConfig,
    adaptive_temporal_scores,
    additional_state_bytes,
    estimator_panel,
    future_attention_utility,
)
from statekv.storage import atomic_frame, atomic_json, atomic_text


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/adaptive_temporal/offline_qwen3_8b.yaml"


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    mask = np.isfinite(left) & np.isfinite(right)
    if int(mask.sum()) < 3:
        return float("nan")
    x = np.asarray(left[mask], dtype=np.float64)
    y = np.asarray(right[mask], dtype=np.float64)
    if float(np.std(x)) <= 0.0 or float(np.std(y)) <= 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    mask = np.isfinite(left) & np.isfinite(right)
    if int(mask.sum()) < 3:
        return float("nan")
    return _correlation(rankdata(left[mask]), rankdata(right[mask]))


def _top_indices(values: np.ndarray, valid: np.ndarray, count: int) -> np.ndarray:
    rows = np.flatnonzero(valid & np.isfinite(values))
    take = min(int(count), int(rows.size))
    if take <= 0:
        return np.asarray([], dtype=np.int64)
    scores = values[rows]
    if take == rows.size:
        order = np.lexsort((rows, -scores))
        return rows[order]
    selected = np.argpartition(-scores, take - 1)[:take]
    chosen = rows[selected]
    order = np.lexsort((chosen, -values[chosen]))
    return chosen[order]


def _rank_matrix(values: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    result = np.full_like(values, np.nan, dtype=np.float64)
    for step in range(values.shape[0]):
        valid = eligible[step] & np.isfinite(values[step])
        if int(valid.sum()) >= 2:
            result[step, valid] = rankdata(values[step, valid], method="average")
    return result


def _columnwise_correlations(
    predictions: np.ndarray,
    targets: np.ndarray,
    minimum: int = 8,
) -> np.ndarray:
    output = np.full(predictions.shape[1], np.nan, dtype=np.float64)
    for column in range(predictions.shape[1]):
        valid = np.isfinite(predictions[:, column]) & np.isfinite(targets[:, column])
        if int(valid.sum()) >= int(minimum):
            output[column] = _correlation(
                predictions[valid, column], targets[valid, column]
            )
    return output


def _future_exact(observations: np.ndarray, horizon: int) -> np.ndarray:
    result = future_attention_utility(observations, horizon)
    if int(horizon) < observations.shape[0]:
        result[observations.shape[0] - int(horizon) :] = np.nan
    return result


def _matrices(group: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cycles = np.arange(int(group["cycle"].max()) + 1, dtype=np.int64)
    positions = np.sort(group["position"].unique().astype(np.int64))
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
    eligible = np.isfinite(ranks) & (ranks > 0)
    return cycles, positions, attention, eligible


def _autocorrelation_rows(
    sample_id: str,
    task: str,
    layer: int,
    attention: np.ndarray,
    eligible: np.ndarray,
    lags: Sequence[int],
    topk: int,
) -> list[dict]:
    rows = []
    for lag in lags:
        spearman_values = []
        pearson_values = []
        top_overlaps = []
        eviction_jaccards = []
        for step in range(attention.shape[0] - int(lag)):
            common = eligible[step] & eligible[step + int(lag)]
            if int(common.sum()) < 3:
                continue
            left = attention[step]
            right = attention[step + int(lag)]
            spearman_values.append(_spearman(left[common], right[common]))
            pearson_values.append(_correlation(left[common], right[common]))
            left_top = set(_top_indices(left, common, topk).tolist())
            right_top = set(_top_indices(right, common, topk).tolist())
            denom = max(1, min(int(topk), int(common.sum())))
            top_overlaps.append(len(left_top & right_top) / denom)
            universe = set(np.flatnonzero(common).tolist())
            left_evict = universe - left_top
            right_evict = universe - right_top
            union = left_evict | right_evict
            eviction_jaccards.append(
                len(left_evict & right_evict) / max(1, len(union))
            )
        rows.append(
            {
                "sample_id": sample_id,
                "task": task,
                "layer": int(layer),
                "lag": int(lag),
                "spearman": float(np.nanmean(spearman_values)),
                "pearson": float(np.nanmean(pearson_values)),
                "topk_overlap": float(np.nanmean(top_overlaps)),
                "eviction_set_jaccard": float(np.nanmean(eviction_jaccards)),
                "step_pairs": int(len(spearman_values)),
            }
        )
    return rows


def _prediction_rows(
    sample_id: str,
    task: str,
    layer: int,
    panel: Mapping[str, np.ndarray],
    attention: np.ndarray,
    eligible: np.ndarray,
    horizons: Sequence[int],
    topk: int,
) -> Tuple[list[dict], list[dict], Dict[Tuple[str, int], np.ndarray]]:
    rows = []
    token_distribution = []
    rank_cache = {name: _rank_matrix(values, eligible) for name, values in panel.items()}
    future_cache: Dict[Tuple[str, int], np.ndarray] = {}
    fixed = [
        name
        for name in panel
        if name == "cumulative" or name.startswith("fixed_ema_rho_")
    ]
    for horizon in horizons:
        target = _future_exact(attention, int(horizon))
        target_rank = _rank_matrix(target, eligible)
        future_cache[("target", int(horizon))] = target
        for name, scores in panel.items():
            correlations = []
            recalls = []
            decisions = 0
            for step in range(max(0, attention.shape[0] - int(horizon))):
                valid = eligible[step] & np.isfinite(target[step]) & np.isfinite(scores[step])
                if int(valid.sum()) < 3:
                    continue
                correlations.append(_spearman(scores[step, valid], target[step, valid]))
                predicted = set(_top_indices(scores[step], valid, topk).tolist())
                oracle = set(_top_indices(target[step], valid, topk).tolist())
                denom = max(1, min(int(topk), len(oracle)))
                recalls.append(len(predicted & oracle) / denom)
                decisions += 1
            rows.append(
                {
                    "sample_id": sample_id,
                    "task": task,
                    "layer": int(layer),
                    "future_horizon": int(horizon),
                    "method": name,
                    "mean_step_spearman": float(np.nanmean(correlations)),
                    "future_topk_recall": float(np.nanmean(recalls)),
                    "eviction_decision_precision": float(np.nanmean(recalls)),
                    "decisions": int(decisions),
                    "causal_estimator": True,
                }
            )
        rows.append(
            {
                "sample_id": sample_id,
                "task": task,
                "layer": int(layer),
                "future_horizon": int(horizon),
                "method": "NON_CAUSAL_ORACLE",
                "mean_step_spearman": 1.0,
                "future_topk_recall": 1.0,
                "eviction_decision_precision": 1.0,
                "decisions": int(max(0, attention.shape[0] - int(horizon))),
                "causal_estimator": False,
            }
        )

        # Token-level best-horizon distribution. Rank normalization within
        # each live candidate set prevents attention-scale changes over time
        # from determining the selected horizon.
        best_score = np.full(attention.shape[1], -np.inf, dtype=np.float64)
        best_name = np.full(attention.shape[1], "", dtype=object)
        for name in fixed:
            correlations = _columnwise_correlations(rank_cache[name], target_rank)
            improve = np.isfinite(correlations) & (correlations > best_score)
            best_score[improve] = correlations[improve]
            best_name[improve] = name
        counts = Counter(str(value) for value in best_name if str(value))
        for name, count in sorted(counts.items()):
            token_distribution.append(
                {
                    "sample_id": sample_id,
                    "task": task,
                    "layer": int(layer),
                    "future_horizon": int(horizon),
                    "best_method": name,
                    "token_count": int(count),
                }
            )
    return rows, token_distribution, future_cache


def _event_rows(
    sample_id: str,
    task: str,
    layer: int,
    positions: np.ndarray,
    attention: np.ndarray,
    eligible: np.ndarray,
    adaptive: Mapping[str, np.ndarray],
    future: Mapping[Tuple[str, int], np.ndarray],
    fixed_reference: np.ndarray,
    per_group: int = 2,
) -> Tuple[list[dict], list[dict]]:
    drift = np.where(eligible, adaptive["drift_z"], np.nan)
    flat = np.flatnonzero(np.isfinite(drift.ravel()))
    events = []
    if flat.size:
        take = min(int(per_group), int(flat.size))
        chosen = flat[np.argpartition(-drift.ravel()[flat], take - 1)[:take]]
        for index in chosen:
            step, column = np.unravel_index(int(index), drift.shape)
            start = max(0, int(step) - 4)
            stop = min(attention.shape[0], int(step) + 5)
            events.append(
                {
                    "sample_id": sample_id,
                    "task": task,
                    "layer": int(layer),
                    "position": int(positions[column]),
                    "cycle": int(step),
                    "current_attention": float(attention[step, column]),
                    "fast": float(adaptive["fast"][step, column]),
                    "slow": float(adaptive["slow"][step, column]),
                    "normalized_drift": float(drift[step, column]),
                    "adaptive_rho": float(adaptive["rho_smooth"][step, column]),
                    "adaptive_score": float(adaptive["adaptive_smooth"][step, column]),
                    "fixed_reference_score": float(fixed_reference[step, column]),
                    "future_utility_h4": float(future.get(("target", 4), np.full_like(attention, np.nan))[step, column]),
                    "future_utility_h16": float(future.get(("target", 16), np.full_like(attention, np.nan))[step, column]),
                    "attention_window_json": json.dumps(
                        [float(value) for value in attention[start:stop, column]]
                    ),
                    "window_start_cycle": int(start),
                }
            )
    stable_rows = []
    for column in range(attention.shape[1]):
        valid = eligible[:, column] & np.isfinite(attention[:, column])
        if int(valid.sum()) < 24:
            continue
        values = attention[valid, column]
        coefficient = float(np.std(values) / (np.mean(values) + 1.0e-12))
        stable_rows.append((coefficient, column, int(valid.sum())))
    stable = []
    for coefficient, column, count in sorted(stable_rows)[: int(per_group)]:
        stable.append(
            {
                "sample_id": sample_id,
                "task": task,
                "layer": int(layer),
                "position": int(positions[column]),
                "coefficient_of_variation": float(coefficient),
                "observed_eligible_steps": int(count),
                "mean_normalized_drift": float(np.nanmean(drift[:, column])),
                "mean_adaptive_rho": float(np.nanmean(adaptive["rho_smooth"][:, column])),
                "attention_trajectory_json": json.dumps(
                    [float(value) if np.isfinite(value) else None for value in attention[:, column]]
                ),
            }
        )
    return events, stable


def _best_fixed_summary(
    prediction: pd.DataFrame, development_samples: set[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    fixed = prediction[
        prediction["method"].eq("cumulative")
        | prediction["method"].str.startswith("fixed_ema_rho_")
    ].copy()
    dev = fixed[fixed["sample_id"].isin(development_samples)]
    selection_rows = []
    comparison_rows = []
    for horizon, group in dev.groupby("future_horizon"):
        aggregate = (
            group.groupby("method", as_index=False)[
                ["future_topk_recall", "mean_step_spearman"]
            ]
            .mean()
            .sort_values(
                ["future_topk_recall", "mean_step_spearman", "method"],
                ascending=[False, False, True],
            )
        )
        best = str(aggregate.iloc[0]["method"])
        selection_rows.append(
            {
                "future_horizon": int(horizon),
                "best_fixed_method": best,
                "development_future_topk_recall": float(
                    aggregate.iloc[0]["future_topk_recall"]
                ),
                "development_mean_step_spearman": float(
                    aggregate.iloc[0]["mean_step_spearman"]
                ),
            }
        )
        heldout = prediction[
            prediction["future_horizon"].eq(horizon)
            & ~prediction["sample_id"].isin(development_samples)
        ].copy()
        evaluation_scope = "heldout"
        if heldout.empty:
            heldout = prediction[
                prediction["future_horizon"].eq(horizon)
                & prediction["sample_id"].isin(development_samples)
            ].copy()
            evaluation_scope = "development_smoke_only"
        heldout["method_display"] = heldout["method"]
        heldout.loc[heldout["method"].eq(best), "method_display"] = "Best Fixed EMA"
        keep = heldout[
            heldout["method"].isin(
                {
                    best,
                    "current",
                    "cumulative",
                    "adaptive_discrete",
                    "adaptive_smooth",
                    "adaptive_raw",
                    "dual_memory",
                    "adaptive_surprise",
                    "NON_CAUSAL_ORACLE",
                }
            )
        ]
        for method, values in keep.groupby("method_display"):
            comparison_rows.append(
                {
                    "future_horizon": int(horizon),
                    "method": str(method),
                    "heldout_mean_step_spearman": float(values["mean_step_spearman"].mean()),
                    "heldout_future_topk_recall": float(values["future_topk_recall"].mean()),
                    "heldout_eviction_decision_precision": float(
                        values["eviction_decision_precision"].mean()
                    ),
                    "sample_layer_rows": int(len(values)),
                    "evaluation_scope": evaluation_scope,
                }
            )
    return pd.DataFrame(selection_rows), pd.DataFrame(comparison_rows)


def _head_diagnostic(
    source: Path,
    samples: Sequence[str],
    fixed_rhos: Sequence[float],
) -> pd.DataFrame:
    path = source / "head_rows.parquet"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(
        path,
        columns=["sample_id", "task", "cycle", "layer", "head", "position", "attn"],
        filters=[("sample_id", "in", list(samples))],
    )
    rows = []
    config = AdaptiveTemporalConfig()
    for (sample_id, task, layer, head), group in frame.groupby(
        ["sample_id", "task", "layer", "head"], sort=False
    ):
        cycles = np.sort(group["cycle"].unique())
        positions = np.sort(group["position"].unique())
        observations = (
            group.pivot_table(index="cycle", columns="position", values="attn", aggfunc="mean")
            .reindex(index=cycles, columns=positions)
            .to_numpy(dtype=np.float64)
        )
        if observations.shape[0] < 4:
            continue
        target = _future_exact(observations, 1)
        eligible = np.isfinite(observations) & np.isfinite(target)
        panel = estimator_panel(observations, fixed_rhos, config)
        candidates = {
            name: values
            for name, values in panel.items()
            if name == "cumulative" or name.startswith("fixed_ema_rho_")
        }
        scores = {}
        for name, values in candidates.items():
            correlations = []
            for step in range(observations.shape[0] - 1):
                valid = eligible[step] & np.isfinite(values[step])
                if int(valid.sum()) >= 3:
                    correlations.append(_spearman(values[step, valid], target[step, valid]))
            scores[name] = float(np.nanmean(correlations)) if correlations else float("nan")
        finite = {name: value for name, value in scores.items() if np.isfinite(value)}
        if finite:
            best = max(finite, key=lambda name: (finite[name], name))
            rows.append(
                {
                    "sample_id": str(sample_id),
                    "task": str(task),
                    "layer": int(layer),
                    "head": int(head),
                    "best_method": str(best),
                    "best_mean_step_spearman": float(finite[best]),
                    "captured_cycles": int(observations.shape[0]),
                    "note": "sparse head diagnostic; one update per four decode cycles",
                }
            )
    return pd.DataFrame(rows)


def _plot(output: Path, autocorrelation: pd.DataFrame, comparison: pd.DataFrame, token_dist: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    plot_root = ROOT / "plots/adaptive_temporal"
    plot_root.mkdir(parents=True, exist_ok=True)
    auto = autocorrelation.groupby("lag", as_index=False)[
        ["spearman", "topk_overlap", "eviction_set_jaccard"]
    ].mean()
    fig, axis = plt.subplots(figsize=(6.5, 4.0))
    for column, label in (
        ("spearman", "Attention Spearman"),
        ("topk_overlap", "Top-k overlap"),
        ("eviction_set_jaccard", "Eviction-set Jaccard"),
    ):
        axis.plot(auto["lag"], auto[column], marker="o", label=label)
    axis.set_xscale("log", base=2)
    axis.set_xlabel("Decode-step lag")
    axis.set_ylabel("Stability")
    axis.set_ylim(0.0, 1.02)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(plot_root / "drift_stability_vs_lag.png", dpi=180)
    plt.close(fig)

    for horizon, group in comparison.groupby("future_horizon"):
        ordered = group.sort_values("heldout_future_topk_recall")
        fig, axis = plt.subplots(figsize=(7.0, max(3.5, 0.4 * len(ordered))))
        axis.barh(ordered["method"], ordered["heldout_future_topk_recall"])
        axis.set_xlabel(f"Held-out future top-k recall (H={int(horizon)})")
        axis.set_xlim(0.0, 1.02)
        fig.tight_layout()
        fig.savefig(plot_root / f"heldout_prediction_h{int(horizon)}.png", dpi=180)
        plt.close(fig)

    distribution = token_dist.groupby(["future_horizon", "best_method"], as_index=False)[
        "token_count"
    ].sum()
    for horizon, group in distribution.groupby("future_horizon"):
        group = group.sort_values("token_count")
        fig, axis = plt.subplots(figsize=(7.0, max(3.5, 0.4 * len(group))))
        axis.barh(group["best_method"], group["token_count"])
        axis.set_xlabel(f"Tokens selecting best fixed horizon (H={int(horizon)})")
        fig.tight_layout()
        fig.savefig(plot_root / f"best_horizon_distribution_h{int(horizon)}.png", dpi=180)
        plt.close(fig)

    atomic_text(output / "plot_directory.txt", "plots/adaptive_temporal\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source = ROOT / str(config["source_run"])
    output = ROOT / str(config["output_run"])
    output.mkdir(parents=True, exist_ok=True)
    diagnostic = dict(config["diagnostics"])
    split = dict(config["split"])
    adaptive_config = AdaptiveTemporalConfig(**dict(config["adaptive"]))
    fixed_rhos = [float(value) for value in diagnostic["fixed_rhos"]]
    horizons = [int(value) for value in diagnostic["future_horizons"]]
    lags = [int(value) for value in diagnostic["lags"]]
    topk = int(diagnostic["topk"])
    development = {str(value) for value in split["development_samples"]}
    summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    available_samples = [str(value) for value in summary["samples"]]
    samples = [
        str(value) for value in config.get("sample_ids", available_samples)
    ]
    unknown_samples = sorted(set(samples) - set(available_samples))
    if unknown_samples:
        raise ValueError(f"configured samples are absent from source: {unknown_samples}")
    selected_layers = {
        int(value) for value in diagnostic.get("layers", range(int(summary["model_info"]["num_layers"])))
    }

    autocorrelation_rows = []
    prediction_rows = []
    token_distribution_rows = []
    event_rows = []
    stable_rows = []
    estimator_seconds = 0.0
    estimator_updates = 0
    started = time.perf_counter()
    for sample_index, sample_id in enumerate(samples, start=1):
        sample = pd.read_parquet(
            source / "token_rows.parquet",
            columns=["sample_id", "task", "cycle", "layer", "position", "attn", "rank"],
            filters=[("sample_id", "==", sample_id)],
        )
        if sample.empty:
            raise RuntimeError(f"source trace is missing sample {sample_id}")
        task = str(sample["task"].iloc[0])
        sample = sample[sample["layer"].isin(selected_layers)]
        for layer, group in sample.groupby("layer", sort=True):
            _, positions, attention, eligible = _matrices(group)
            panel_started = time.perf_counter()
            panel = estimator_panel(attention, fixed_rhos, adaptive_config)
            adaptive = adaptive_temporal_scores(attention, adaptive_config)
            estimator_seconds += float(time.perf_counter() - panel_started)
            estimator_updates += int(np.isfinite(attention).sum())
            autocorrelation_rows.extend(
                _autocorrelation_rows(
                    sample_id, task, int(layer), attention, eligible, lags, topk
                )
            )
            current_prediction, current_tokens, future = _prediction_rows(
                sample_id,
                task,
                int(layer),
                panel,
                attention,
                eligible,
                horizons,
                topk,
            )
            prediction_rows.extend(current_prediction)
            token_distribution_rows.extend(current_tokens)
            fixed_reference = panel.get("fixed_ema_rho_0.99", panel["cumulative"])
            events, stable = _event_rows(
                sample_id,
                task,
                int(layer),
                positions,
                attention,
                eligible,
                adaptive,
                future,
                fixed_reference,
            )
            event_rows.extend(events)
            stable_rows.extend(stable)
        print(
            f"[adaptive-temporal] sample {sample_index}/{len(samples)} {sample_id}",
            flush=True,
        )

    autocorrelation = pd.DataFrame(autocorrelation_rows)
    prediction = pd.DataFrame(prediction_rows)
    token_distribution = pd.DataFrame(token_distribution_rows)
    events = pd.DataFrame(event_rows)
    stable = pd.DataFrame(stable_rows)
    best_fixed, comparison = _best_fixed_summary(prediction, development)
    head = _head_diagnostic(source, sorted(development), fixed_rhos)

    cache = dict(config["cache"])
    model = dict(summary["model_info"])
    kv_bytes_per_token_layer = (
        2
        * int(model["num_key_value_heads"])
        * int(model["hidden_size"] // model["num_attention_heads"])
        * 2
    )
    overhead_rows = []
    for name, scalars in (
        ("FixedEMA", 1),
        ("AdaptiveTemporalV1", 4),
        ("DualMemoryGate", 3),
        ("HeadLevelAdaptive", 4 * int(model["num_key_value_heads"])),
    ):
        per_token = scalars if name != "HeadLevelAdaptive" else 0
        for dtype_name, dtype_bytes in (("fp32", 4), ("fp16", 2)):
            if name == "HeadLevelAdaptive":
                total = additional_state_bytes(
                    scalars_per_token=4,
                    dtype_bytes=dtype_bytes,
                    layers=int(model["num_layers"]),
                    tokens_per_layer=int(model["num_key_value_heads"]),
                )
                bytes_token = 0
                ratio = total / (
                    int(model["num_layers"])
                    * int(cache["total_budget"])
                    * kv_bytes_per_token_layer
                )
            else:
                total = additional_state_bytes(
                    scalars_per_token=per_token,
                    dtype_bytes=dtype_bytes,
                    layers=int(model["num_layers"]),
                    tokens_per_layer=int(cache["total_budget"]),
                )
                bytes_token = per_token * dtype_bytes
                ratio = bytes_token / kv_bytes_per_token_layer
            overhead_rows.append(
                {
                    "method": name,
                    "state_dtype": dtype_name,
                    "additional_bytes_per_token_layer": int(bytes_token),
                    "total_state_bytes_at_budget": int(total),
                    "total_state_mib_at_budget": float(total / 2**20),
                    "fraction_of_fp16_kv_memory": float(ratio),
                    "offline_estimator_updates_per_s": float(
                        estimator_updates / max(estimator_seconds, 1.0e-12)
                    ),
                    "offline_estimator_seconds": float(estimator_seconds),
                    "note": "offline NumPy scoring; not an end-to-end decode latency measurement",
                }
            )
    overhead = pd.DataFrame(overhead_rows)

    token_summary = (
        token_distribution.groupby(["task", "future_horizon", "best_method"], as_index=False)[
            "token_count"
        ].sum()
    )
    concentration_rows = []
    for (task, horizon), group in token_summary.groupby(["task", "future_horizon"]):
        counts = group["token_count"].to_numpy(dtype=np.float64)
        probability = counts / counts.sum()
        concentration_rows.append(
            {
                "task": str(task),
                "future_horizon": int(horizon),
                "tokens": int(counts.sum()),
                "dominant_horizon_fraction": float(probability.max()),
                "horizon_entropy_nats": float(-np.sum(probability * np.log(probability))),
                "distinct_best_horizons": int(len(counts)),
            }
        )
    concentration = pd.DataFrame(concentration_rows)

    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    git_branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=ROOT, text=True
    ).strip()
    metadata = {
        "experiment": str(config["experiment_name"]),
        "git_commit": git_commit,
        "git_branch": git_branch,
        "model": model.get("model_name"),
        "model_revision": "545dc4251c05440727734bcd94334791f6ab0192",
        "source_run": str(config["source_run"]),
        "source_trajectory": config["source_semantics"]["trajectory"],
        "sample_ids": samples,
        "development_sample_ids": sorted(development),
        "heldout_sample_ids": sorted(set(samples) - development),
        "seed": int(config["seed"]),
        "cache": cache,
        "fixed_rhos": fixed_rhos,
        "future_horizons": horizons,
        "lags": lags,
        "adaptive": dict(config["adaptive"]),
        "generation": {"decode_cycles": int(config["source_semantics"]["decode_cycles"])},
        "quantization": {"model_weight_precision": model.get("weight_precision")},
        "runtime_seconds": float(time.perf_counter() - started),
        "estimator_seconds": float(estimator_seconds),
        "estimator_updates": int(estimator_updates),
        "oracle_label": "NON_CAUSAL_ORACLE",
        "physical_scope": "offline full-pool trace on recoverable qk_pool roll-in",
    }

    atomic_frame(autocorrelation, output / "temporal_autocorrelation.csv")
    atomic_frame(prediction, output / "future_utility_prediction.csv")
    atomic_frame(token_distribution, output / "best_horizon_token_rows.csv")
    atomic_frame(token_summary, output / "best_horizon_distribution.csv")
    atomic_frame(concentration, output / "horizon_heterogeneity.csv")
    atomic_frame(best_fixed, output / "best_fixed_ema_selection.csv")
    atomic_frame(comparison, output / "heldout_method_comparison.csv")
    atomic_frame(events, output / "drift_events.csv")
    atomic_frame(stable, output / "stability_events.csv")
    atomic_frame(head, output / "head_horizon_distribution.csv")
    atomic_frame(overhead, output / "runtime_memory_overhead.csv")
    atomic_json(output / "metadata.json", metadata)
    atomic_text(
        output / "config.yaml",
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
    )
    summary_out = {
        "experiment": str(config["experiment_name"]),
        "execution_valid": True,
        "samples": samples,
        "development_samples": sorted(development),
        "heldout_samples": sorted(set(samples) - development),
        "best_fixed_by_horizon": best_fixed.to_dict("records"),
        "heldout_comparison": comparison.to_dict("records"),
        "horizon_heterogeneity": concentration.to_dict("records"),
        "head_diagnostic_rows": int(len(head)),
        "offline_scope": True,
        "non_causal_oracle_only_for_diagnostics": True,
        "runtime_seconds": metadata["runtime_seconds"],
    }
    atomic_json(output / "summary.json", summary_out)
    _plot(output, autocorrelation, comparison, token_distribution)
    print(json.dumps(summary_out, indent=2))


if __name__ == "__main__":
    main()
