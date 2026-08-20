#!/usr/bin/env python3
"""Decompose global/task/head/token-time dynamic-horizon headroom.

All fixed horizon choices are selected on the three development sequences.
The token-time oracle is explicitly noncausal: on held-out sequences it uses
future-attention ranks to choose, independently for each token and decode
time, among the same fixed EMA candidates.  Sequence, not token/head, is the
unit of inference for the continuation gate.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import rankdata

from statekv.adaptive_temporal import fixed_ema, future_attention_utility
from statekv.selectors import mandatory_and_eligible
from statekv.storage import atomic_frame, atomic_json, atomic_text


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/adaptive_temporal/dynamic_horizon_oracle_qwen3_8b.yaml"


def _method(rho: float) -> str:
    return f"ema_rho_{float(rho):g}"


def _rho(method: str) -> float:
    return float(method.rsplit("_", 1)[-1])


def _top(values: np.ndarray, valid: np.ndarray, count: int) -> np.ndarray:
    rows = np.flatnonzero(valid & np.isfinite(values))
    take = min(int(count), int(rows.size))
    if take <= 0:
        return np.asarray([], dtype=np.int64)
    order = np.lexsort((rows, -values[rows]))
    return rows[order[:take]]


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 3:
        return float("nan")
    x = rankdata(left[valid], method="average")
    y = rankdata(right[valid], method="average")
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _eligible_matrix(
    position_ids: np.ndarray, sink_size: int, recent_size: int
) -> np.ndarray:
    eligible = np.zeros(position_ids.shape, dtype=bool)
    for cycle in range(position_ids.shape[0]):
        positions = position_ids[cycle, position_ids[cycle] >= 0].astype(int).tolist()
        _, _, values = mandatory_and_eligible(positions, sink_size, recent_size)
        eligible[cycle, : len(positions)] = np.isin(positions, values)
    return eligible


def _future_exact(attention: np.ndarray, horizon: int) -> np.ndarray:
    target = future_attention_utility(attention, int(horizon))
    target[max(0, target.shape[0] - int(horizon)) :] = np.nan
    return target


def _fixed_metrics(
    score: np.ndarray,
    target: np.ndarray,
    eligible: np.ndarray,
    topk: int,
) -> Dict[str, float]:
    recalls = []
    correlations = []
    for cycle in range(score.shape[0]):
        valid = eligible[cycle] & np.isfinite(score[cycle]) & np.isfinite(target[cycle])
        if int(valid.sum()) < 3:
            continue
        predicted = set(_top(score[cycle], valid, topk).tolist())
        oracle = set(_top(target[cycle], valid, topk).tolist())
        recalls.append(len(predicted & oracle) / max(1, len(oracle)))
        correlations.append(_spearman(score[cycle, valid], target[cycle, valid]))
    return {
        "future_topk_recall": float(np.nanmean(recalls)),
        "mean_step_spearman": float(np.nanmean(correlations)),
        "decisions": int(len(recalls)),
    }


def _percentile(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    output = np.full(values.shape, np.nan, dtype=np.float64)
    count = int(valid.sum())
    if count:
        output[valid] = rankdata(values[valid], method="average") / float(count)
    return output


def _token_time_oracle_metrics(
    scores: Mapping[str, np.ndarray],
    target: np.ndarray,
    eligible: np.ndarray,
    topk: int,
) -> Tuple[Dict[str, float], Counter]:
    recalls = []
    correlations = []
    choices: Counter = Counter()
    methods = list(scores)
    for cycle in range(target.shape[0]):
        valid = eligible[cycle] & np.isfinite(target[cycle])
        for values in scores.values():
            valid &= np.isfinite(values[cycle])
        if int(valid.sum()) < 3:
            continue
        target_rank = _percentile(target[cycle], valid)
        candidate_ranks = np.stack(
            [_percentile(scores[name][cycle], valid) for name in methods], axis=0
        )
        distance = np.abs(candidate_ranks - target_rank[None, :])
        selected = np.argmin(distance[:, valid], axis=0)
        oracle_score = np.full(target.shape[1], np.nan, dtype=np.float64)
        valid_rows = np.flatnonzero(valid)
        oracle_score[valid_rows] = candidate_ranks[selected, valid_rows]
        for index in selected.tolist():
            choices[methods[int(index)]] += 1
        predicted = set(_top(oracle_score, valid, topk).tolist())
        oracle = set(_top(target[cycle], valid, topk).tolist())
        recalls.append(len(predicted & oracle) / max(1, len(oracle)))
        correlations.append(_spearman(oracle_score[valid], target[cycle, valid]))
    return (
        {
            "future_topk_recall": float(np.nanmean(recalls)),
            "mean_step_spearman": float(np.nanmean(correlations)),
            "decisions": int(len(recalls)),
        },
        choices,
    )


def _load_trajectory(path: Path) -> Tuple[str, str, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return (
            str(data["sample_id"].item()),
            str(data["task"].item()),
            data["layers"].astype(int),
            data["attention"].astype(np.float64),
            data["position_ids"].astype(int),
        )


def _development_metrics(
    paths: Sequence[Path],
    rhos: Sequence[float],
    horizons: Sequence[int],
    sink_size: int,
    recent_size: int,
    topk: int,
) -> pd.DataFrame:
    rows = []
    for path in paths:
        sample_id, task, layers, attention, position_ids = _load_trajectory(path)
        eligible = _eligible_matrix(position_ids, sink_size, recent_size)
        for layer_index, layer in enumerate(layers):
            for head in range(attention.shape[2]):
                observations = attention[:, layer_index, head]
                scores = {_method(rho): fixed_ema(observations, rho) for rho in rhos}
                for horizon in horizons:
                    target = _future_exact(observations, horizon)
                    for method, values in scores.items():
                        rows.append(
                            {
                                "sample_id": sample_id,
                                "task": task,
                                "layer": int(layer),
                                "head": int(head),
                                "future_horizon": int(horizon),
                                "method": method,
                                **_fixed_metrics(values, target, eligible, topk),
                            }
                        )
    return pd.DataFrame(rows)


def _best(group: pd.DataFrame) -> str:
    ranking = (
        group.groupby("method", as_index=False)["future_topk_recall"]
        .mean()
        .assign(rho=lambda frame: frame["method"].map(_rho))
        .sort_values(["future_topk_recall", "rho"], ascending=[False, True])
    )
    return str(ranking.iloc[0]["method"])


def _assignments(development: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for horizon, group in development.groupby("future_horizon"):
        rows.append(
            {
                "level": "global_fixed",
                "task": "*",
                "layer": -1,
                "head": -1,
                "future_horizon": int(horizon),
                "method": _best(group),
            }
        )
    for (task, horizon), group in development.groupby(["task", "future_horizon"]):
        rows.append(
            {
                "level": "task_fixed",
                "task": str(task),
                "layer": -1,
                "head": -1,
                "future_horizon": int(horizon),
                "method": _best(group),
            }
        )
    for (layer, head, horizon), group in development.groupby(
        ["layer", "head", "future_horizon"]
    ):
        rows.append(
            {
                "level": "per_head_fixed",
                "task": "*",
                "layer": int(layer),
                "head": int(head),
                "future_horizon": int(horizon),
                "method": _best(group),
            }
        )
    return pd.DataFrame(rows)


def _lookup(assignments: pd.DataFrame, level: str, task: str, layer: int, head: int, horizon: int) -> str:
    values = assignments[
        (assignments["level"] == level)
        & (assignments["future_horizon"] == int(horizon))
    ]
    if level == "task_fixed":
        values = values[values["task"] == task]
    elif level == "per_head_fixed":
        values = values[(values["layer"] == layer) & (values["head"] == head)]
    if len(values) != 1:
        raise RuntimeError(f"missing {level} assignment for {task}/{layer}/{head}/H{horizon}")
    return str(values.iloc[0]["method"])


def _heldout_metrics(
    paths: Sequence[Path],
    rhos: Sequence[float],
    horizons: Sequence[int],
    assignments: pd.DataFrame,
    sink_size: int,
    recent_size: int,
    topk: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    unit_rows = []
    choice_rows = []
    for path in paths:
        sample_id, task, layers, attention, position_ids = _load_trajectory(path)
        eligible = _eligible_matrix(position_ids, sink_size, recent_size)
        for layer_index, layer in enumerate(layers):
            for head in range(attention.shape[2]):
                observations = attention[:, layer_index, head]
                scores = {_method(rho): fixed_ema(observations, rho) for rho in rhos}
                for horizon in horizons:
                    target = _future_exact(observations, horizon)
                    for level in ("global_fixed", "task_fixed", "per_head_fixed"):
                        method = _lookup(
                            assignments, level, task, int(layer), int(head), int(horizon)
                        )
                        unit_rows.append(
                            {
                                "sample_id": sample_id,
                                "task": task,
                                "layer": int(layer),
                                "head": int(head),
                                "future_horizon": int(horizon),
                                "level": level,
                                "method": method,
                                **_fixed_metrics(scores[method], target, eligible, topk),
                            }
                        )
                    metrics, choices = _token_time_oracle_metrics(
                        scores, target, eligible, topk
                    )
                    unit_rows.append(
                        {
                            "sample_id": sample_id,
                            "task": task,
                            "layer": int(layer),
                            "head": int(head),
                            "future_horizon": int(horizon),
                            "level": "token_time_dynamic_oracle",
                            "method": "NON_CAUSAL_TOKEN_TIME_ORACLE",
                            **metrics,
                        }
                    )
                    total = sum(choices.values())
                    for method, count in choices.items():
                        choice_rows.append(
                            {
                                "sample_id": sample_id,
                                "task": task,
                                "layer": int(layer),
                                "head": int(head),
                                "future_horizon": int(horizon),
                                "method": method,
                                "token_time_choices": int(count),
                                "choice_fraction": float(count / max(1, total)),
                            }
                        )
    units = pd.DataFrame(unit_rows)
    sequence_rows = []
    for keys, group in units.groupby(["sample_id", "task", "future_horizon", "level"]):
        weights = group["decisions"].to_numpy(dtype=np.float64)
        sequence_rows.append(
            {
                "sample_id": keys[0],
                "task": keys[1],
                "future_horizon": int(keys[2]),
                "level": keys[3],
                "future_topk_recall": float(np.average(group["future_topk_recall"], weights=weights)),
                "mean_step_spearman": float(np.average(group["mean_step_spearman"], weights=weights)),
                "head_step_decisions": int(weights.sum()),
            }
        )
    return pd.DataFrame(sequence_rows), pd.DataFrame(choice_rows)


def _bootstrap(
    sequence: pd.DataFrame,
    baseline: str,
    repetitions: int,
    confidence: float,
    seed: int,
) -> Dict[str, float]:
    pivot = (
        sequence.groupby(["sample_id", "level"], as_index=False)["future_topk_recall"]
        .mean()
        .pivot(index="sample_id", columns="level", values="future_topk_recall")
    )
    difference = (
        pivot["token_time_dynamic_oracle"] - pivot[baseline]
    ).to_numpy(dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    draws = rng.integers(0, len(difference), size=(int(repetitions), len(difference)))
    means = difference[draws].mean(axis=1)
    alpha = (1.0 - float(confidence)) / 2.0
    return {
        "comparison": f"token_time_dynamic_oracle_minus_{baseline}",
        "mean_gain": float(difference.mean()),
        "ci_low": float(np.quantile(means, alpha)),
        "ci_high": float(np.quantile(means, 1.0 - alpha)),
        "sequence_wins": int((difference > 0).sum()),
        "sequence_ties": int((difference == 0).sum()),
        "sequence_losses": int((difference < 0).sum()),
        "sequences": int(len(difference)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG.relative_to(ROOT)))
    arguments = parser.parse_args()
    config_path = ROOT / arguments.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output = ROOT / str(config["output_run"])
    trajectory_root = output / "trajectories"
    analysis_root = output / "analysis"
    plot_root = ROOT / "plots/adaptive_temporal/dynamic_horizon_oracle_qwen3_8b_v1"
    analysis_root.mkdir(parents=True, exist_ok=True)
    plot_root.mkdir(parents=True, exist_ok=True)

    def path_for(sample_id: str) -> Path:
        name = str(sample_id).replace(":", "__").replace("/", "_") + ".npz"
        path = trajectory_root / name
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    development_paths = [path_for(value) for value in config["development_sample_ids"]]
    heldout_paths = [path_for(value) for value in config["heldout_sample_ids"]]
    rhos = [float(value) for value in config["candidate_rhos"]]
    horizons = [int(value) for value in config["future_utility_horizons"]]
    sink_size = int(config["sink_size"])
    recent_size = int(config["recent_size"])
    topk = int(config["core_budget"])

    development = _development_metrics(
        development_paths, rhos, horizons, sink_size, recent_size, topk
    )
    assignments = _assignments(development)
    heldout, choices = _heldout_metrics(
        heldout_paths,
        rhos,
        horizons,
        assignments,
        sink_size,
        recent_size,
        topk,
    )
    decomposition = (
        heldout.groupby(["future_horizon", "level"], as_index=False)
        .agg(
            future_topk_recall=("future_topk_recall", "mean"),
            mean_step_spearman=("mean_step_spearman", "mean"),
            sequences=("sample_id", "nunique"),
        )
    )
    aggregate = (
        heldout.groupby("level", as_index=False)
        .agg(
            future_topk_recall=("future_topk_recall", "mean"),
            mean_step_spearman=("mean_step_spearman", "mean"),
        )
    )
    global_value = float(
        aggregate.loc[aggregate["level"] == "global_fixed", "future_topk_recall"].iloc[0]
    )
    aggregate["headroom_over_global"] = aggregate["future_topk_recall"] - global_value

    gate = dict(config["gate"])
    bootstrap_rows = [
        _bootstrap(
            heldout,
            baseline,
            int(gate["bootstrap_repetitions"]),
            float(gate["bootstrap_confidence"]),
            int(gate["random_seed"]),
        )
        for baseline in gate["require_gain_over"]
    ]
    minimum_gain = float(gate["minimum_absolute_gain"])
    minimum_wins = int(gate["minimum_sequence_wins"])
    passed = all(
        row["mean_gain"] >= minimum_gain
        and row["ci_low"] > 0.0
        and row["sequence_wins"] >= minimum_wins
        for row in bootstrap_rows
    )
    verdict = {
        "gate_passed": bool(passed),
        "decision": (
            "dynamic headroom exists; adaptive-horizon exploration may continue"
            if passed
            else "stop adaptive-horizon algorithm exploration and report a negative finding"
        ),
        "oracle_label": "NON_CAUSAL_TOKEN_TIME_ORACLE",
        "inference_unit": "heldout_sequence",
        "heldout_opened_once": True,
        "criteria": gate,
        "comparisons": bootstrap_rows,
    }

    atomic_frame(development, analysis_root / "development_candidate_metrics.csv")
    atomic_frame(assignments, analysis_root / "fixed_horizon_assignments.csv")
    atomic_frame(heldout, analysis_root / "heldout_sequence_metrics.csv")
    atomic_frame(choices, analysis_root / "token_time_oracle_choices.csv")
    atomic_frame(decomposition, analysis_root / "headroom_by_future_horizon.csv")
    atomic_frame(aggregate, analysis_root / "headroom_aggregate.csv")
    atomic_frame(pd.DataFrame(bootstrap_rows), analysis_root / "paired_sequence_bootstrap.csv")
    atomic_json(analysis_root / "verdict.json", verdict)

    order = ["global_fixed", "task_fixed", "per_head_fixed", "token_time_dynamic_oracle"]
    labels = ["Global", "Task", "Per head", "Token-time\noracle"]
    plot_values = aggregate.set_index("level").reindex(order)["future_topk_recall"]
    figure, axis = plt.subplots(figsize=(7.0, 4.2))
    colors = ["#4C78A8", "#72B7B2", "#F58518", "#E45756"]
    bars = axis.bar(labels, plot_values.to_numpy(), color=colors)
    axis.bar_label(bars, fmt="%.3f", padding=3)
    axis.set_ylabel("Held-out future top-k recall")
    axis.set_title("Dynamic-horizon headroom decomposition")
    axis.set_ylim(0.0, 1.0)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(plot_root / "headroom_decomposition.pdf", bbox_inches="tight")
    figure.savefig(plot_root / "headroom_decomposition.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    atomic_text(
        analysis_root / "README.md",
        "# Dynamic-Horizon Oracle analysis\n\n"
        "All four levels use the same Qwen3-8B qk_pool trajectories, six "
        "preregistered layers, eight KV heads, future-attention labels, and "
        "held-out sequences. Global/task/per-head choices are fitted only on "
        "the three development sequences. `NON_CAUSAL_TOKEN_TIME_ORACLE` uses "
        "future ranks and is an upper bound, not a deployable method. The gate "
        "uses paired sequence bootstrap on the mean over H={1,4,16,32}.\n",
    )
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
