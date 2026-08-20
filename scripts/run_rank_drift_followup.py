#!/usr/bin/env python3
"""Bounded causal rank-drift follow-up after the dynamic-oracle gate."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from analyze_dynamic_horizon_oracle import (
    _eligible_matrix,
    _fixed_metrics,
    _future_exact,
    _load_trajectory,
)
from statekv.adaptive_temporal import rank_jump_dual_scores
from statekv.storage import atomic_frame, atomic_json, atomic_text


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/adaptive_temporal/rank_drift_followup_qwen3_8b.yaml"


def _grid(config: Mapping) -> list[dict]:
    rows = []
    for pair, rank_rho, threshold, alpha, output_space in itertools.product(
        config["fast_slow_pairs"],
        config["rank_memory_rhos"],
        config["jump_thresholds"],
        config["gate_alphas"],
        config["output_spaces"],
    ):
        rows.append(
            {
                "fast_rho": float(pair[0]),
                "slow_rho": float(pair[1]),
                "rank_memory_rho": float(rank_rho),
                "jump_threshold": float(threshold),
                "gate_alpha": float(alpha),
                "output_space": str(output_space),
            }
        )
    return rows


def _score(observations: np.ndarray, config: Mapping) -> Dict[str, np.ndarray]:
    return rank_jump_dual_scores(observations, **dict(config))


def _paths(oracle: Mapping, sample_ids: Sequence[str]) -> list[Path]:
    root = ROOT / str(oracle["output_run"]) / "trajectories"
    return [root / (str(value).replace(":", "__").replace("/", "_") + ".npz") for value in sample_ids]


def _development(
    paths: Sequence[Path],
    grid: Sequence[Mapping],
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
                targets = {h: _future_exact(observations, h) for h in horizons}
                for config_id, config in enumerate(grid):
                    states = _score(observations, config)
                    present = eligible & np.isfinite(states["score"])
                    diagnostics = {
                        "mean_rank_jump": float(np.nanmean(states["rank_jump"][present])),
                        "mean_stable_gate": float(np.nanmean(states["stable_gate"][present])),
                        "short_gate_fraction": float(
                            np.nanmean(states["stable_gate"][present] < 0.5)
                        ),
                    }
                    for horizon, target in targets.items():
                        rows.append(
                            {
                                "sample_id": sample_id,
                                "task": task,
                                "layer": int(layer),
                                "head": int(head),
                                "future_horizon": int(horizon),
                                "config_id": int(config_id),
                                **dict(config),
                                **diagnostics,
                                **_fixed_metrics(states["score"], target, eligible, topk),
                            }
                        )
    return pd.DataFrame(rows)


def _select(development: pd.DataFrame) -> tuple[int, dict, pd.DataFrame]:
    ranking = (
        development.groupby("config_id", as_index=False)
        .agg(
            development_future_topk_recall=("future_topk_recall", "mean"),
            development_mean_step_spearman=("mean_step_spearman", "mean"),
            mean_rank_jump=("mean_rank_jump", "mean"),
            mean_stable_gate=("mean_stable_gate", "mean"),
            short_gate_fraction=("short_gate_fraction", "mean"),
        )
        .sort_values(
            ["development_future_topk_recall", "config_id"],
            ascending=[False, True],
        )
    )
    winner_id = int(ranking.iloc[0]["config_id"])
    columns = [
        "fast_rho",
        "slow_rho",
        "rank_memory_rho",
        "jump_threshold",
        "gate_alpha",
        "output_space",
    ]
    winner = development[development["config_id"] == winner_id].iloc[0]
    return winner_id, {column: winner[column] for column in columns}, ranking


def _heldout(
    paths: Sequence[Path],
    winner_id: int,
    winner: Mapping,
    horizons: Sequence[int],
    sink_size: int,
    recent_size: int,
    topk: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    units = []
    for path in paths:
        sample_id, task, layers, attention, position_ids = _load_trajectory(path)
        eligible = _eligible_matrix(position_ids, sink_size, recent_size)
        for layer_index, layer in enumerate(layers):
            for head in range(attention.shape[2]):
                observations = attention[:, layer_index, head]
                states = _score(observations, winner)
                present = eligible & np.isfinite(states["score"])
                for horizon in horizons:
                    units.append(
                        {
                            "sample_id": sample_id,
                            "task": task,
                            "layer": int(layer),
                            "head": int(head),
                            "future_horizon": int(horizon),
                            "level": "rank_jump_dual",
                            "config_id": int(winner_id),
                            "mean_rank_jump": float(np.nanmean(states["rank_jump"][present])),
                            "mean_stable_gate": float(np.nanmean(states["stable_gate"][present])),
                            "short_gate_fraction": float(np.nanmean(states["stable_gate"][present] < 0.5)),
                            **_fixed_metrics(
                                states["score"],
                                _future_exact(observations, horizon),
                                eligible,
                                topk,
                            ),
                        }
                    )
    unit_frame = pd.DataFrame(units)
    sequence_rows = []
    for keys, group in unit_frame.groupby(["sample_id", "task", "future_horizon"]):
        weights = group["decisions"].to_numpy(dtype=np.float64)
        sequence_rows.append(
            {
                "sample_id": keys[0],
                "task": keys[1],
                "future_horizon": int(keys[2]),
                "level": "rank_jump_dual",
                "future_topk_recall": float(np.average(group["future_topk_recall"], weights=weights)),
                "mean_step_spearman": float(np.average(group["mean_step_spearman"], weights=weights)),
                "head_step_decisions": int(weights.sum()),
            }
        )
    return unit_frame, pd.DataFrame(sequence_rows)


def _paired_bootstrap(
    comparison: pd.DataFrame,
    baseline: str,
    repetitions: int,
    confidence: float,
    seed: int,
) -> dict:
    pivot = (
        comparison.groupby(["sample_id", "level"], as_index=False)["future_topk_recall"]
        .mean()
        .pivot(index="sample_id", columns="level", values="future_topk_recall")
    )
    difference = (pivot["rank_jump_dual"] - pivot[baseline]).to_numpy(dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    draws = rng.integers(0, len(difference), size=(int(repetitions), len(difference)))
    means = difference[draws].mean(axis=1)
    alpha = (1.0 - float(confidence)) / 2.0
    return {
        "comparison": f"rank_jump_dual_minus_{baseline}",
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
    config = yaml.safe_load((ROOT / arguments.config).read_text(encoding="utf-8"))
    oracle = yaml.safe_load((ROOT / str(config["oracle_config"])).read_text(encoding="utf-8"))
    oracle_analysis = ROOT / str(oracle["output_run"]) / "analysis"
    oracle_verdict = json.loads((oracle_analysis / "verdict.json").read_text(encoding="utf-8"))
    if not bool(oracle_verdict["gate_passed"]):
        raise RuntimeError("rank-drift follow-up is forbidden because the oracle gate failed")

    output = ROOT / str(config["output_run"])
    output.mkdir(parents=True, exist_ok=True)
    grid = _grid(config)
    horizons = [int(value) for value in oracle["future_utility_horizons"]]
    sink_size = int(oracle["sink_size"])
    recent_size = int(oracle["recent_size"])
    topk = int(oracle["core_budget"])
    development = _development(
        _paths(oracle, oracle["development_sample_ids"]),
        grid,
        horizons,
        sink_size,
        recent_size,
        topk,
    )
    winner_id, winner, ranking = _select(development)
    units, adaptive_sequence = _heldout(
        _paths(oracle, oracle["heldout_sample_ids"]),
        winner_id,
        winner,
        horizons,
        sink_size,
        recent_size,
        topk,
    )
    fixed_sequence = pd.read_csv(oracle_analysis / "heldout_sequence_metrics.csv")
    fixed_sequence = fixed_sequence[
        fixed_sequence["level"].isin(["global_fixed", "per_head_fixed"])
    ]
    comparison = pd.concat([fixed_sequence, adaptive_sequence], ignore_index=True)
    summary = (
        comparison.groupby(["future_horizon", "level"], as_index=False)
        .agg(
            future_topk_recall=("future_topk_recall", "mean"),
            mean_step_spearman=("mean_step_spearman", "mean"),
            sequences=("sample_id", "nunique"),
        )
    )
    gate = dict(config["followup_gate"])
    paired = _paired_bootstrap(
        comparison,
        str(gate["baseline"]),
        int(gate["bootstrap_repetitions"]),
        float(gate["bootstrap_confidence"]),
        int(gate["random_seed"]),
    )
    passed = (
        paired["mean_gain"] >= float(gate["minimum_absolute_gain"])
        and paired["ci_low"] > 0.0
        and paired["sequence_wins"] >= int(gate["minimum_sequence_wins"])
    )
    verdict = {
        "oracle_prerequisite_passed": True,
        "followup_is_conditional_exploratory_analysis": True,
        "heldout_retuning": False,
        "selected_config_id": winner_id,
        "selected_configuration": winner,
        "grid_candidates": len(grid),
        "paired_result": paired,
        "followup_gate": gate,
        "gate_passed": bool(passed),
        "decision": (
            "rank-drift gate passes future-utility estimator criterion"
            if passed
            else "rank-drift does not capture the observed dynamic-oracle headroom"
        ),
    }

    atomic_frame(development, output / "development_grid_rows.csv")
    atomic_frame(ranking, output / "development_ranking.csv")
    atomic_frame(units, output / "heldout_unit_metrics.csv")
    atomic_frame(comparison, output / "heldout_sequence_comparison.csv")
    atomic_frame(summary, output / "heldout_summary.csv")
    atomic_frame(pd.DataFrame([paired]), output / "paired_sequence_bootstrap.csv")
    atomic_json(output / "verdict.json", verdict)
    atomic_text(output / "config.yaml", yaml.safe_dump(config, sort_keys=False))
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
