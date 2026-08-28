#!/usr/bin/env python3
"""Plot per-token importance trajectories over the full causal timeline.

For one collected timeline artifact, shows how much attention a handful of
tokens (needle / filler / early-prompt) receive at every timeline row:
prefill blocks (aggregated query spans) followed by per-cycle decode rows.
The prefill|decode boundary is marked with a vertical line.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from statekv.reactivation_timeline import load_artifact, timeline_importance
from statekv.storage import safe_path_component


def _rank_fractions(importance: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    n_rows, n_positions = importance.shape
    ranks = np.full((n_rows, n_positions), np.nan)
    for row in range(n_rows):
        active = int(lengths[row])
        order = np.argsort(-importance[row, :active], kind="stable")
        rank_of = np.empty(active)
        rank_of[order] = np.arange(active, dtype=np.float64) / active
        ranks[row, :active] = rank_of
    return ranks


def _select_tokens(
    artifact: Dict[str, np.ndarray],
    importance: np.ndarray,
    n_blocks: int,
    n_filler: int,
    n_early: int,
    seed: int,
) -> List[Tuple[int, str]]:
    prompt_length = int(np.asarray(artifact["prompt_length"]))
    spans = np.asarray(artifact.get("needle_token_spans", np.zeros((0, 2))))
    selected: List[Tuple[int, str]] = []
    taken = set()
    if spans.size:
        span = spans[0]
        candidates = [
            int(span[0]),
            int((span[0] + span[1]) // 2),
            int(span[1]) - 2,
        ]
        for position in candidates:
            position = min(max(position, 0), prompt_length - 1)
            if position not in taken:
                selected.append((position, f"needle@{position}"))
                taken.add(position)
    else:
        decode_mean = importance[n_blocks:, :prompt_length].mean(axis=0)
        decode_mean[0] = -np.inf  # skip the BOS/sink position
        for position in np.argsort(-decode_mean)[:3]:
            selected.append((int(position), f"decode-max@{int(position)}"))
            taken.add(int(position))

    for index in range(n_early):
        position = min(4 * index, prompt_length - 1)
        if position not in taken:
            selected.append((position, f"early@{position}"))
            taken.add(position)

    rng = np.random.default_rng(seed)
    span_mask = np.zeros(prompt_length, dtype=bool)
    for start, end in spans:
        span_mask[int(start) : int(end)] = True
    pool = [
        position
        for position in range(16, prompt_length)
        if not span_mask[position] and position not in taken
    ]
    chosen = rng.choice(len(pool), size=min(n_filler, len(pool)), replace=False)
    for index in sorted(chosen):
        position = int(pool[int(index)])
        selected.append((position, f"filler@{position}"))
        taken.add(position)
    return selected


def plot_artifact(
    artifact_path: Path,
    out_dir: Path,
    n_filler: int = 3,
    n_early: int = 2,
    seed: int = 0,
) -> Path:
    artifact = load_artifact(str(artifact_path))
    importance, lengths, n_blocks = timeline_importance(artifact)
    n_rows = importance.shape[0]
    ranks = _rank_fractions(importance, lengths)
    tokens = _select_tokens(
        artifact, importance, n_blocks, n_filler, n_early, seed
    )
    sample_id = str(np.asarray(artifact["sample_id"]))
    rows = np.arange(n_rows)

    figure, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for position, label in tokens:
        color = None
        if label.startswith("needle") or label.startswith("decode-max"):
            (line,) = axes[0].plot(
                rows, importance[:, position], label=label, linewidth=1.8
            )
            color = line.get_color()
            axes[1].plot(
                rows, ranks[:, position], label=label, color=color, linewidth=1.8
            )
        else:
            (line,) = axes[0].plot(
                rows, importance[:, position], label=label, alpha=0.7
            )
            axes[1].plot(
                rows, ranks[:, position], label=label, alpha=0.7,
                color=line.get_color(),
            )
    for axis in axes:
        axis.axvline(
            n_blocks - 0.5, color="black", linestyle="--", linewidth=1.0
        )
        axis.text(
            n_blocks - 0.5,
            axis.get_ylim()[1],
            " prefill | decode",
            verticalalignment="top",
            fontsize=8,
        )
    axes[0].set_yscale("log")
    axes[0].set_ylabel("received attention share")
    axes[0].set_title(f"{sample_id}: full-timeline token importance")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("rank fraction (0 = most important)")
    axes[1].set_xlabel("timeline row (prefill blocks, then decode cycles)")
    axes[0].legend(fontsize=8, loc="upper right")
    figure.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / f"{safe_path_component(sample_id)}_trajectory.png"
    figure.savefig(output, dpi=150)
    plt.close(figure)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", action="append", required=True)
    parser.add_argument("--out-dir", default="plots/statekv_counterfactual")
    parser.add_argument("--n-filler", type=int, default=3)
    parser.add_argument("--n-early", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    for artifact in args.artifact:
        output = plot_artifact(
            root / artifact,
            root / args.out_dir,
            n_filler=args.n_filler,
            n_early=args.n_early,
            seed=args.seed,
        )
        print(output)


if __name__ == "__main__":
    main()
