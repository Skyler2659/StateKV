#!/usr/bin/env python3
"""Ranking-first R2 -> Student distillation with hard-negative mining.

Pipeline:
  v3 base  (cutoff-weighted pairwise ranking, 12 epochs)
  -> mine round 1 (FN x FP pairs at the deployment cutoff) -> retrain 6 epochs
  -> mine round 2 -> retrain 4 epochs
  -> evaluate v1-frozen / v2.1 / v3 / v3-hn1 / v3-hn2 on validation with the
     selection-fidelity battery at k=220 (budget 256) and k=476 (budget 512),
     deployment horizon H=1, layer/head-mean aggregation (as deployed).

Outputs land next to the other student checkpoints; a comparison CSV goes to
``student_models/v3_comparison.csv``.
"""
from pathlib import Path
import argparse
import time

import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler

from statekv.causal_distillation import _teacher_arrays
from statekv.causal_existence import _safe_sample_id, sample_id_for
from statekv.causal_student import (
    R2_TEACHER,
    StudentScorer,
    _train_student_mlp_v3,
    evaluate_students_cutoff,
    load_student_checkpoint,
    mine_cutoff_errors,
    save_student_checkpoint,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/statekv_counterfactual/r2_student_qwen3_8b.yaml",
    )
    parser.add_argument("--base-epochs", type=int, default=12)
    parser.add_argument("--mine-epochs", type=int, nargs="*", default=[6, 4])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / args.config).read_text(encoding="utf-8"))
    horizons = [int(v) for v in config["future_utility_horizons"]]
    seed = int(config["data_seed"])
    core = int(config["core_budget"])
    source_run = root / str(config["source_run"])
    teacher_root = root / str(config["teacher_scores"])
    output_root = root / str(config["output_models"])
    output_root.mkdir(parents=True, exist_ok=True)

    train_ids = [
        sample_id_for(str(fam), int(idx))
        for fam in config["task_families"]
        for idx in config["distillation"]["train_indices"]
    ]
    artifact_paths = [
        source_run / "artifacts" / "train" / f"{_safe_sample_id(s)}.npz"
        for s in train_ids
    ]
    started = time.perf_counter()
    features, _, truth, _, boundary_ids, sizes = _teacher_arrays(
        artifact_paths, teacher_root / "train", config, return_sizes=True
    )
    print(f"[v3] {len(features)} token rows from {len(artifact_paths)} sequences", flush=True)
    scaler = StandardScaler().fit(features)
    normalized = scaler.transform(features).astype("float32")

    variants = {}
    model = _train_student_mlp_v3(
        normalized, truth, boundary_ids, sizes, len(horizons), seed + 61,
        int(args.base_epochs), core,
    )
    variants["v3_base"] = model
    for round_index, mine_epochs in enumerate(args.mine_epochs, start=1):
        scorer = StudentScorer(
            {
                "kind": "mlp",
                "models": model.state_dict(),
                "scaler": scaler,
                "horizons": horizons,
                "score_channel": 0,
            }
        )
        mined = mine_cutoff_errors(
            scorer, normalized, truth, boundary_ids, sizes, core, horizon_column=0
        )
        if not mined:
            print(f"[v3] mining round {round_index}: no cutoff errors, stopping", flush=True)
            break
        model = _train_student_mlp_v3(
            normalized, truth, boundary_ids, sizes, len(horizons),
            seed + 61 + round_index, int(mine_epochs), core,
            init_state=model.state_dict(), extra_pairs=mined, extra_weight=5.0,
        )
        variants[f"v3_hn{round_index}"] = model

    checkpoints = {}
    for name, variant in variants.items():
        checkpoints[name] = save_student_checkpoint(
            output_root / f"r2_student_mlp_{name}.pt",
            kind="mlp",
            models=variant.state_dict(),
            scaler=scaler,
            horizons=horizons,
            projector_seed=seed,
            score_channel=0,
            metadata={
                "teacher": R2_TEACHER,
                "objective": "cutoff-weighted pairwise ranking + hard-negative mining",
                "runtime_future_access": False,
            },
        )
        print(f"[v3] saved {checkpoints[name]}", flush=True)

    scorers = {
        "v1_frozen": StudentScorer(
            load_student_checkpoint(output_root / "r2_student_mlp_v1_frozen.pt")
        ),
        "v2_1": StudentScorer(
            load_student_checkpoint(output_root / "r2_student_mlp_v2.pt")
        ),
    }
    for name, path in checkpoints.items():
        scorers[name] = StudentScorer(load_student_checkpoint(path))

    rows = evaluate_students_cutoff(config, root, scorers, ks=(220, 476))
    frame = pd.DataFrame(rows)
    summary = (
        frame.groupby("method", as_index=False)
        .agg(
            top256_recall=("topk_recall@220", "mean"),
            top512_recall=("topk_recall@476", "mean"),
            jaccard_256=("jaccard@220", "mean"),
            cutoff_pair_acc_256=("cutoff_pair_accuracy@220", "mean"),
            band_pair_acc_256=("band_pair_accuracy@220", "mean"),
            boundaries=("sample_id", "size"),
        )
        .sort_values("top256_recall", ascending=False)
    )
    print(summary.round(4).to_string(index=False), flush=True)
    frame.to_csv(output_root / "v3_boundary_metrics.csv", index=False)
    summary.to_csv(output_root / "v3_comparison.csv", index=False)
    per_task = (
        frame.groupby(["task", "method"], as_index=False)["topk_recall@220"]
        .mean()
        .pivot(index="task", columns="method", values="topk_recall@220")
        .round(3)
    )
    print(per_task.to_string(), flush=True)
    print(f"[v3] elapsed {time.perf_counter() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
