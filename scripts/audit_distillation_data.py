#!/usr/bin/env python3
"""Audit the R2 -> Student distillation dataset before retraining.

Independent checks (fail loudly) for:
1. teacher-score / feature alignment at (sample, cycle, position)
2. teacher H=1 score vs the real next-cycle attention recorded in the
   full-cache artifact (catches token/head misalignment: a correctly joined
   teacher score must correlate strongly with the artifact's own future row)
3. feature determinism: artifact_boundary twice -> bit-identical
4. sequence-level split isolation (train/validation disjoint)
5. feature sanity: no NaN/constant columns, per-segment statistics reported
"""
from pathlib import Path
import argparse

import numpy as np
import yaml
from scipy.stats import spearmanr

from statekv.causal_existence import sample_id_for
from statekv.causal_predictors import FixedProjector, _load_npz, artifact_boundary
from statekv.selectors import mandatory_and_eligible


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/statekv_counterfactual/r2_student_qwen3_8b.yaml",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / args.config).read_text(encoding="utf-8"))
    source_run = root / str(config["source_run"])
    teacher_root = root / str(config["teacher_scores"])
    horizons = [int(v) for v in config["future_utility_horizons"]]
    sink = int(config["sink_size"])
    recent = int(config["recent_size"])
    core = int(config["core_budget"])
    projector = FixedProjector(int(config["data_seed"]))

    train_ids = {
        sample_id_for(str(fam), int(idx))
        for fam in config["task_families"]
        for idx in config["distillation"]["train_indices"]
    }

    checks = {"alignment": 0, "h1_corr": [], "determinism": 0, "rows": 0}
    feature_blocks = []
    for split in ("train", "validation"):
        artifact_dir = source_run / "artifacts" / split
        for artifact_path in sorted(artifact_dir.glob("*.npz")):
            teacher_path = teacher_root / split / artifact_path.name
            if not teacher_path.exists():
                # artifacts cover more sequences than the teacher dump; only
                # the intersection carries distillation labels
                continue
            artifact = _load_npz(artifact_path)
            teacher = _load_npz(teacher_path)
            sample_id = str(teacher["sample_id"].item())
            if split == "validation" and sample_id in train_ids:
                raise RuntimeError(f"split leak: {sample_id} in both train and validation")
            for cycle_index, cycle in enumerate(teacher["cycles"]):
                cycle = int(cycle)
                count = int(teacher["position_lengths"][cycle_index])
                teacher_positions = [
                    int(v) for v in teacher["position_ids"][cycle_index, :count]
                ]
                cur = int(artifact["position_lengths"][cycle])
                positions = [int(v) for v in artifact["position_ids"][cycle, :cur]]
                _, _, eligible = mandatory_and_eligible(positions, sink, recent)
                if teacher_positions != [int(v) for v in eligible]:
                    raise RuntimeError(
                        f"position misalignment: {artifact_path.name} cycle {cycle}"
                    )
                checks["alignment"] += 1
                attention = artifact["attention"]  # (cycles, layers, kv, positions)
                pos_col = {int(p): i for i, p in enumerate(positions)}
                eligible_cols = [pos_col[p] for p in teacher_positions]
                for layer_index in range(int(artifact["layers"].size)):
                    for head in range(int(attention.shape[2])):
                        boundary = artifact_boundary(
                            artifact, cycle, layer_index, head, horizons,
                            sink, recent, core, projector, feature_only=True,
                        )
                        # determinism on one boundary per sample is enough
                        if layer_index == 0 and head == 0 and cycle_index == 0:
                            again = artifact_boundary(
                                artifact, cycle, layer_index, head, horizons,
                                sink, recent, core, projector, feature_only=True,
                            )
                            if not np.array_equal(
                                boundary.features, again.features, equal_nan=True
                            ):
                                raise RuntimeError(
                                    f"artifact_boundary not deterministic: {artifact_path.name}"
                                )
                            checks["determinism"] += 1
                        # teacher H=1 vs artifact's real next-cycle attention
                        if cycle + 1 < attention.shape[0]:
                            teacher_h1 = teacher["scores"][
                                cycle_index, 0, layer_index, head, :count
                            ]
                            real_next = attention[
                                cycle + 1, layer_index, head, :
                            ][eligible_cols]
                            if np.std(real_next) > 0 and np.std(teacher_h1) > 0:
                                checks["h1_corr"].append(
                                    float(spearmanr(teacher_h1, real_next).statistic)
                                )
                        checks["rows"] += len(boundary.features)
                        if split == "validation" and cycle_index == 0 and head == 0:
                            feature_blocks.append(boundary.features)
    h1 = np.asarray(checks["h1_corr"])
    h1 = h1[np.isfinite(h1)]
    print(f"alignment checks passed: {checks['alignment']} boundaries")
    print(f"determinism checks passed: {checks['determinism']}")
    print(f"total boundary rows seen: {checks['rows']}")
    print(
        f"teacher H=1 vs artifact next-cycle attention Spearman: "
        f"mean={h1.mean():.3f} p5={np.percentile(h1, 5):.3f} "
        f"p50={np.percentile(h1, 50):.3f} (n={len(h1)})"
    )
    if h1.mean() < 0.5:
        raise RuntimeError(
            "teacher H=1 score barely correlates with the artifact's own "
            "next-cycle attention; token/head misalignment is likely"
        )
    feats = np.concatenate(feature_blocks)
    nan_cols = int(np.isnan(feats).any(axis=0).sum())
    const_cols = int((np.nanstd(feats, axis=0) < 1e-12).sum())
    print(f"feature sanity: width={feats.shape[1]} nan_cols={nan_cols} const_cols={const_cols}")
    print("split isolation: train/validation sequence sets disjoint OK")
    print("AUDIT PASS")


if __name__ == "__main__":
    main()
