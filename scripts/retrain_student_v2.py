#!/usr/bin/env python3
"""Retrain only the v2 student variant and compare it against the frozen v1.

Skips the GBDT and v1 MLP (their checkpoints are already on disk) so a v2
hyperparameter iteration costs one MLP training pass plus validation.  The
result overwrites ``r2_student_mlp_v2.pt`` and prints cutoff metrics at both
deployment core budgets next to the frozen v1 reference.
"""
from pathlib import Path
import argparse
import time

import pandas as pd
import yaml

from statekv.causal_distillation import _teacher_arrays
from statekv.causal_existence import _safe_sample_id, sample_id_for
from statekv.causal_student import (
    R2_TEACHER,
    StudentScorer,
    _train_student_mlp_v2,
    evaluate_students,
    load_student_checkpoint,
    save_student_checkpoint,
)
from sklearn.preprocessing import StandardScaler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/statekv_counterfactual/r2_student_qwen3_8b.yaml",
    )
    parser.add_argument("--pairwise-weight", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / args.config).read_text(encoding="utf-8"))
    horizons = [int(value) for value in config["future_utility_horizons"]]
    seed = int(config["data_seed"])
    source_run = root / str(config["source_run"])
    teacher_root = root / str(config["teacher_scores"])
    output_root = root / str(config["output_models"])

    train_ids = [
        sample_id_for(str(family), int(index))
        for family in config["task_families"]
        for index in config["distillation"]["train_indices"]
    ]
    artifact_paths = [
        source_run / "artifacts" / "train" / f"{_safe_sample_id(sample_id)}.npz"
        for sample_id in train_ids
    ]
    started = time.perf_counter()
    features, _, truth, _, boundary_ids = _teacher_arrays(
        artifact_paths, teacher_root / "train", config
    )
    print(f"[retrain-v2] {len(features)} token rows", flush=True)
    scaler = StandardScaler().fit(features)
    normalized = scaler.transform(features).astype("float32")

    model = _train_student_mlp_v2(
        normalized,
        truth,
        boundary_ids,
        len(horizons),
        seed + 43,
        epochs=int(args.epochs),
        cutoff_budgets=[
            int(value)
            for value in config["student"].get(
                "cutoff_budgets", [92, int(config["core_budget"])]
            )
        ],
        pairwise_weight=float(args.pairwise_weight),
        device=str(config["student"].get("device", "cpu")),
    )
    checkpoint = save_student_checkpoint(
        output_root / "r2_student_mlp_v2.pt",
        kind="mlp",
        models=model.state_dict(),
        scaler=scaler,
        horizons=horizons,
        projector_seed=seed,
        score_channel=int(config["student"].get("v2_score_channel", 0)),
        metadata={
            "teacher": R2_TEACHER,
            "objective": (
                "percentile BCE + log-utility regression + per-horizon "
                f"cutoff-straddling pairwise (weight {args.pairwise_weight})"
            ),
            "pairwise_weight": float(args.pairwise_weight),
            "epochs": int(args.epochs),
            "runtime_future_access": False,
        },
    )
    print(f"[retrain-v2] saved {checkpoint}", flush=True)

    scorers = {
        "v1_cls_frozen": StudentScorer(
            load_student_checkpoint(output_root / "r2_student_mlp_v1_frozen.pt")
        ),
        "v2_pct": StudentScorer(load_student_checkpoint(checkpoint)),
    }
    for core_budget in (92, int(config["core_budget"])):
        eval_config = dict(config)
        eval_config["core_budget"] = core_budget
        frame = pd.DataFrame(evaluate_students(eval_config, root, scorers))
        summary = (
            frame.groupby(["method", "future_horizon"], as_index=False)
            .agg(
                spearman=("spearman", "mean"),
                topk=("future_topk_recall", "mean"),
                ndcg=("ndcg", "mean"),
                ogap=("oracle_gap_recovery", "mean"),
            )
        )
        print(f"==== core_budget={core_budget} ====")
        print(summary.round(3).to_string(index=False), flush=True)
    print(f"[retrain-v2] elapsed {time.perf_counter() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
