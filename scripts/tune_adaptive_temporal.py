#!/usr/bin/env python3
"""Development-only calibration for a bounded adaptive-temporal grid.

The selected configuration is evaluated once on held-out samples. This script
does not reopen the grid after seeing held-out metrics.
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml
from scipy.stats import rankdata

from statekv.adaptive_temporal import (
    AdaptiveTemporalConfig,
    adaptive_temporal_scores,
    future_attention_utility,
)
from statekv.storage import atomic_frame, atomic_json, atomic_text


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/adaptive_temporal/tuning_qwen3_8b.yaml"


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
    return attention, np.isfinite(ranks) & (ranks > 0)


def _rank_observations(attention: np.ndarray) -> np.ndarray:
    output = np.full_like(attention, np.nan)
    for step in range(attention.shape[0]):
        valid = np.isfinite(attention[step])
        if int(valid.sum()):
            output[step, valid] = rankdata(attention[step, valid]) / int(valid.sum())
    return output


def _top(values: np.ndarray, valid: np.ndarray, count: int) -> set[int]:
    rows = np.flatnonzero(valid & np.isfinite(values))
    take = min(int(count), int(rows.size))
    if take <= 0:
        return set()
    local = np.argpartition(-values[rows], take - 1)[:take]
    return set(rows[local].tolist())


def _recall_by_horizon(
    scores: np.ndarray,
    attention: np.ndarray,
    eligible: np.ndarray,
    horizons: Sequence[int],
    topk: int,
) -> Dict[int, float]:
    result = {}
    for horizon in horizons:
        target = future_attention_utility(attention, int(horizon))
        target[attention.shape[0] - int(horizon) :] = np.nan
        recalls = []
        for step in range(attention.shape[0] - int(horizon)):
            valid = eligible[step] & np.isfinite(target[step]) & np.isfinite(scores[step])
            if int(valid.sum()) < 3:
                continue
            predicted = _top(scores[step], valid, topk)
            oracle = _top(target[step], valid, topk)
            recalls.append(len(predicted & oracle) / max(1, len(oracle)))
        result[int(horizon)] = float(np.mean(recalls))
    return result


def _grid(config: Mapping) -> list[AdaptiveTemporalConfig]:
    result = []
    grid = dict(config["grid"])
    for pair, variance, threshold in itertools.product(
        grid["state_pairs"], grid["variance_rhos"], grid["thresholds"]
    ):
        fast, slow, short, long = [float(value) for value in pair]
        result.append(
            AdaptiveTemporalConfig(
                fast_rho=fast,
                slow_rho=slow,
                variance_rho=float(variance),
                rho_short=short,
                rho_long=long,
                threshold=float(threshold),
                smooth_alpha=float(grid["smooth_alpha"]),
                epsilon=float(grid["epsilon"]),
            )
        )
    return result


def _score_variants(attention: np.ndarray, cfg: AdaptiveTemporalConfig):
    states = adaptive_temporal_scores(attention, cfg)
    rank_states = adaptive_temporal_scores(_rank_observations(attention), cfg)
    return {
        "adaptive_discrete": states["adaptive_discrete"],
        "adaptive_smooth": states["adaptive_smooth"],
        "dual_memory": states["dual_memory"],
        "rank_adaptive_smooth": rank_states["adaptive_smooth"],
    }, states


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source = ROOT / str(config["source_run"])
    offline = ROOT / str(config["offline_run"])
    output = ROOT / str(config["output_run"])
    output.mkdir(parents=True, exist_ok=True)
    source_summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    all_samples = [str(value) for value in source_summary["samples"]]
    development = {str(value) for value in config["development_samples"]}
    heldout = [value for value in all_samples if value not in development]
    horizons = [int(value) for value in config["future_horizons"]]
    topk = int(config["topk"])
    layers = {int(value) for value in config["tuning_layers"]}
    variants = [str(value) for value in config["variants"]]
    grid = _grid(config)

    started = time.perf_counter()
    aggregates: Dict[tuple, list[float]] = {}
    gate_stats: Dict[tuple, list[float]] = {}
    for sample_id in sorted(development):
        sample = pd.read_parquet(
            source / "token_rows.parquet",
            columns=["sample_id", "task", "cycle", "layer", "position", "attn", "rank"],
            filters=[("sample_id", "==", sample_id)],
        )
        sample = sample[sample["layer"].isin(layers)]
        for layer, group in sample.groupby("layer", sort=True):
            attention, eligible = _matrices(group)
            for index, cfg in enumerate(grid):
                panel, states = _score_variants(attention, cfg)
                for variant in variants:
                    recall = _recall_by_horizon(
                        panel[variant], attention, eligible, horizons, topk
                    )
                    for horizon, value in recall.items():
                        aggregates.setdefault((index, variant, horizon), []).append(value)
                valid = eligible & np.isfinite(states["rho_smooth"])
                gate_stats.setdefault((index, "mean_rho"), []).append(
                    float(np.mean(states["rho_smooth"][valid]))
                )
                gate_stats.setdefault((index, "short_fraction"), []).append(
                    float(np.mean(states["rho_smooth"][valid] < (cfg.rho_short + cfg.rho_long) / 2.0))
                )
        print(f"[adaptive-tune] development {sample_id}", flush=True)

    tuning_rows = []
    for (index, variant, horizon), values in aggregates.items():
        cfg = grid[index]
        tuning_rows.append(
            {
                "config_id": int(index),
                "variant": variant,
                "future_horizon": int(horizon),
                "development_future_topk_recall": float(np.mean(values)),
                "sample_layer_rows": int(len(values)),
                **cfg.__dict__,
                "mean_rho": float(np.mean(gate_stats[(index, "mean_rho")])),
                "short_gate_fraction": float(np.mean(gate_stats[(index, "short_fraction")])),
            }
        )
    tuning = pd.DataFrame(tuning_rows)
    aggregate = (
        tuning.groupby(["config_id", "variant"], as_index=False)[
            "development_future_topk_recall"
        ]
        .mean()
        .sort_values(
            ["development_future_topk_recall", "variant", "config_id"],
            ascending=[False, True, True],
        )
    )
    winner = aggregate.iloc[0]
    winner_id = int(winner["config_id"])
    winner_variant = str(winner["variant"])
    winner_config = grid[winner_id]

    # One held-out evaluation after selection. The grid is never revisited.
    heldout_rows = []
    for sample_id in heldout:
        sample = pd.read_parquet(
            source / "token_rows.parquet",
            columns=["sample_id", "task", "cycle", "layer", "position", "attn", "rank"],
            filters=[("sample_id", "==", sample_id)],
        )
        task = str(sample["task"].iloc[0])
        for layer, group in sample.groupby("layer", sort=True):
            attention, eligible = _matrices(group)
            panel, _ = _score_variants(attention, winner_config)
            recall = _recall_by_horizon(
                panel[winner_variant], attention, eligible, horizons, topk
            )
            for horizon, value in recall.items():
                heldout_rows.append(
                    {
                        "sample_id": sample_id,
                        "task": task,
                        "layer": int(layer),
                        "future_horizon": int(horizon),
                        "method": f"Tuned {winner_variant}",
                        "heldout_future_topk_recall": float(value),
                    }
                )
        print(f"[adaptive-tune] heldout {sample_id}", flush=True)
    heldout_frame = pd.DataFrame(heldout_rows)
    heldout_summary = (
        heldout_frame.groupby(["future_horizon", "method"], as_index=False)[
            "heldout_future_topk_recall"
        ].mean()
    )
    fixed = pd.read_csv(offline / "heldout_method_comparison.csv")
    fixed = fixed[fixed["method"].eq("Best Fixed EMA")][
        ["future_horizon", "method", "heldout_future_topk_recall"]
    ]
    comparison = pd.concat([fixed, heldout_summary], ignore_index=True)
    pivot = comparison.pivot(
        index="future_horizon", columns="method", values="heldout_future_topk_recall"
    ).reset_index()
    adaptive_name = f"Tuned {winner_variant}"
    pivot["adaptive_minus_best_fixed"] = (
        pivot[adaptive_name] - pivot["Best Fixed EMA"]
    )

    selection = {
        "selected_on": "development samples and six preregistered diagnostic layers",
        "heldout_opened_after_selection": True,
        "config_id": winner_id,
        "variant": winner_variant,
        "development_mean_future_topk_recall": float(
            winner["development_future_topk_recall"]
        ),
        "configuration": winner_config.__dict__,
        "grid_configurations": int(len(grid)),
        "variants": variants,
        "total_candidates": int(len(grid) * len(variants)),
    }
    atomic_frame(tuning, output / "development_grid.csv")
    atomic_frame(aggregate, output / "development_ranking.csv")
    atomic_frame(heldout_frame, output / "heldout_rows.csv")
    atomic_frame(pivot, output / "heldout_tuned_vs_fixed.csv")
    atomic_json(output / "selection.json", selection)
    atomic_json(
        output / "summary.json",
        {
            "experiment": str(config["experiment_name"]),
            "selection": selection,
            "heldout_comparison": pivot.to_dict("records"),
            "runtime_seconds": float(time.perf_counter() - started),
            "test_set_retuning": False,
        },
    )
    atomic_text(
        output / "config.yaml",
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
    )
    print(
        json.dumps(
            {
                "selection": selection,
                "heldout_comparison": pivot.to_dict("records"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

