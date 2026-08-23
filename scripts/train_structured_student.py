#!/usr/bin/env python3
"""Train + evaluate the structured R2 student and its feature ablations.

Trains ``full`` (>= --epochs) and each ablation (--ablation-epochs each,
features are assembled once and cached), then evaluates every variant plus
the legacy v2.1 student checkpoint through one cutoff-metric loop on the
validation split, and writes:

- ``structured_validation_cutoff.csv``   per-boundary metric rows
- ``structured_comparison.csv``          per-method summary
- ``structured_comparison_by_task.csv``  per-task x method top-220 recall
"""
from pathlib import Path
import argparse
import time

import pandas as pd
import yaml

from statekv.causal_student import StudentScorer, load_student_checkpoint
from statekv.structured_student import (
    ABLATIONS,
    StructuredStudentScorer,
    evaluate_structured_cutoff,
    summarize_cutoff_rows,
    train_structured_student,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/statekv_counterfactual/structured_student_qwen3_8b.yaml",
    )
    parser.add_argument(
        "--ablations", nargs="*", default=list(ABLATIONS), choices=list(ABLATIONS)
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--ablation-epochs", type=int, default=None)
    parser.add_argument(
        "--old-student",
        default="results/statekv_counterfactual/student_models/r2_student_mlp_v2.pt",
    )
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="load existing structured_student_<ablation>.pt checkpoints",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = root / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_root = root / str(config["output_models"])
    output_root.mkdir(parents=True, exist_ok=True)
    cache_dir = root / "tmp" / "structured_student_cache"

    checkpoints = {}
    if args.eval_only:
        for ablation in args.ablations:
            path = output_root / f"structured_student_{ablation}.pt"
            if not path.exists():
                raise RuntimeError(f"missing checkpoint for {ablation}: {path}")
            checkpoints[ablation] = path
    else:
        for ablation in args.ablations:
            epochs = args.epochs
            if ablation != "full" and args.ablation_epochs is not None:
                epochs = args.ablation_epochs
            checkpoints[ablation] = train_structured_student(
                config_path, root, ablation=ablation, epochs=epochs,
                cache_dir=cache_dir,
            )
    if args.skip_eval:
        return

    started = time.perf_counter()
    scorers = {
        f"structured_{ablation}": StructuredStudentScorer(
            load_student_checkpoint(path)
        )
        for ablation, path in checkpoints.items()
    }
    scorers["student_mlp_v2"] = StudentScorer(
        load_student_checkpoint(root / args.old_student)
    )
    rows = evaluate_structured_cutoff(
        config, root, scorers, ks=(220, 476), cache_dir=cache_dir
    )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_root / "structured_validation_cutoff.csv", index=False)
    summary = summarize_cutoff_rows(rows, ks=(220, 476))
    summary.to_csv(output_root / "structured_comparison.csv", index=False)
    per_task = (
        frame.groupby(["task", "method"], as_index=False)["topk_recall@220"]
        .mean()
        .pivot(index="task", columns="method", values="topk_recall@220")
        .round(4)
    )
    per_task.to_csv(output_root / "structured_comparison_by_task.csv")
    print(summary.round(4).to_string(index=False), flush=True)
    print(per_task.to_string(), flush=True)
    print(f"[structured-eval] elapsed {time.perf_counter() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
