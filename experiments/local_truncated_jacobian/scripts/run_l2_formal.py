#!/usr/bin/env python3
"""Execute frozen Stage L2 depth ablation and preserve L3 inputs."""
from __future__ import annotations

import argparse
import gc
import json
import resource
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
PREDICTIVE_SCRIPTS = ROOT / "experiments/predictive_closure/scripts"
LOCAL_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks/torch"))
sys.path.insert(0, str(PREDICTIVE_SCRIPTS))
sys.path.insert(0, str(LOCAL_SCRIPTS))

from kvbench.temporal.backend_mlx import MLXTemporalModel
from kvbench.temporal.config import load_discovery_config
from kvbench.temporal.tasks import load_discovery_tasks

from mlx_predictive_core import single_layer_selection

from local_core import (
    FP32LocalBlock,
    FP32TransformerLayer,
    FP32TruncatedStack,
    atomic_frame,
    atomic_json,
    cosine,
    load_candidates,
    relative_l2,
    replay_record,
    sha256_file,
    symmetric_norm_ratio,
    to_physical_candidate,
)
from run_l1_formal import make_ranking_table


REGISTRY = (
    ROOT
    / "experiments/predictive_closure/raw/p0_alignment"
    / "formal_4bit_retry1/candidate_registry_rows.parquet"
)
REGISTRY_SHA256 = (
    "f2d06b2732a2a0bf8baac6694ef35aa2ed4393a19e75400564a545786d787307"
)
PREREGISTRATION_SHA256 = (
    "d21532527849ad9cf644458bbb622ba6afe9088443556d289401ece6e4c0b28e"
)
L1_SUMMARY = (
    ROOT
    / "experiments/local_truncated_jacobian/raw"
    / "l1_local_linearization/formal/l1_summary.json"
)
L1_SUMMARY_SHA256 = (
    "f127bfb9a0706719d04695013e3b943ebd8f92b66b17dde34778aa3efd0790c6"
)
LAYERS = (0, 7, 14, 21, 26)
COMMON_DEPTH_LAYERS = (0, 7, 14, 21)
ANCHORS = (16, 32, 48, 64)
DEPTHS = (0, 1, 2, 4)
TABLES = (
    "depth_vector",
    "depth_ranking",
    "physical_output",
    "anchor_audit",
)


def expected_sample_ids() -> List[str]:
    return [
        "gov_report:24",
        "gov_report:25",
        "synthetic_niah_24",
        "synthetic_niah_25",
    ]


def safe_id(value: str) -> str:
    return "".join(
        character
        if character.isalnum() or character in "._-"
        else "_"
        for character in value
    )


def stable_exact_kl(full_logits: Any, changed_logits: Any) -> float:
    full = np.asarray(full_logits, dtype=np.float64).reshape(-1)
    changed = np.asarray(changed_logits, dtype=np.float64).reshape(-1)
    full_max = float(np.max(full))
    changed_max = float(np.max(changed))
    full_lse = full_max + float(
        np.log(np.exp(full - full_max).sum())
    )
    changed_lse = changed_max + float(
        np.log(np.exp(changed - changed_max).sum())
    )
    log_p = full - full_lse
    log_q = changed - changed_lse
    probability = np.exp(log_p)
    return float(np.sum(probability * (log_p - log_q)))


def available_depths(layer: int, layer_count: int) -> List[int]:
    return [
        depth
        for depth in DEPTHS
        if depth == 0 or int(layer) + int(depth) <= int(layer_count)
    ]


def write_sequence_tables(
    sequence_dir: Path, tables: Mapping[str, pd.DataFrame]
) -> None:
    for name in TABLES:
        atomic_frame(sequence_dir / f"{name}.parquet", tables[name])


def load_sequence_tables(
    sequence_dir: Path,
) -> Dict[str, pd.DataFrame]:
    return {
        name: pd.read_parquet(sequence_dir / f"{name}.parquet")
        for name in TABLES
    }


def run_sequence(
    model: MLXTemporalModel,
    sample: Any,
    sequence_dir: Path,
    local_templates: Dict[int, FP32LocalBlock],
    layer_templates: Dict[int, FP32TransformerLayer],
    stack_metadata: Dict[str, Any],
) -> Dict[str, pd.DataFrame]:
    started = time.perf_counter()
    vector_rows: List[Dict[str, Any]] = []
    physical_rows: List[Dict[str, Any]] = []
    audit_rows: List[Dict[str, Any]] = []
    atomic_json(
        sequence_dir / "status.json",
        {
            "stage": "formal_l2",
            "state": "running",
            "sample_id": sample.sample_id,
            "task": sample.task,
            "split": "train",
            "calibration_or_test_loaded": False,
            "errors": [],
        },
    )
    reference = model.generate_reference(
        sample_id=sample.sample_id,
        task=sample.task,
        prompt=sample.prompt,
    )
    layer_count = int(model.model_info["num_layers"])
    for anchor_step in ANCHORS:
        anchor_started = time.perf_counter()
        anchor = reference.anchors[int(anchor_step)]
        full_record = reference.query_records[int(anchor_step)]
        full_logits = (
            reference.probe_logits[int(anchor_step)].double().numpy()
        )
        candidates = load_candidates(
            REGISTRY,
            sample.sample_id,
            anchor_step,
            int(anchor.logical_length - 1),
            REGISTRY_SHA256,
        )
        for candidate in candidates:
            physical_candidate = to_physical_candidate(candidate)
            for layer in LAYERS:
                selection, physical_cache_cfg = single_layer_selection(
                    reference,
                    anchor_step,
                    physical_candidate,
                    layer,
                )
                (
                    physical_logits,
                    physical_record,
                    physical_seconds,
                ) = replay_record(
                    model,
                    reference,
                    anchor_step,
                    selection,
                    physical_cache_cfg,
                )
                u_phys = (
                    physical_record.projected_attention_outputs[layer]
                    .float()
                    - full_record.projected_attention_outputs[layer].float()
                ).numpy()
                native_b = (
                    full_record.post_attention_residuals[layer]
                    .float()
                    .numpy()
                )
                common = {
                    "sample_id": sample.sample_id,
                    "task": sample.task,
                    "split": "train",
                    "anchor": anchor_step,
                    "layer": layer,
                    "candidate_id": candidate.candidate_id,
                    "candidate_source": candidate.source,
                    "mask_hash": candidate.mask_hash,
                }
                adjacent_target = (
                    physical_record.layer_outputs[layer].float()
                    - full_record.layer_outputs[layer].float()
                ).numpy()
                direct_prediction = np.asarray(
                    u_phys, dtype=np.float64
                )
                vector_rows.append(
                    {
                        **common,
                        "depth": 0,
                        "target_layer_output": layer,
                        "prediction_path": "identity_direct_u_phys",
                        "jvp_method": "identity_no_propagation",
                        "baseline_relative_l2": 0.0,
                        "jvp_seconds": 0.0,
                        "one_token_forward_seconds": physical_seconds,
                        "relative_forward_cost": 0.0,
                        "mlx_peak_memory_bytes": 0,
                        "prediction_norm": float(
                            np.linalg.norm(direct_prediction)
                        ),
                        "truth_norm": float(
                            np.linalg.norm(adjacent_target)
                        ),
                        "predicted_energy": float(
                            np.dot(
                                direct_prediction, direct_prediction
                            )
                        ),
                        "true_energy": float(
                            np.dot(adjacent_target, adjacent_target)
                        ),
                        "vector_cosine": cosine(
                            direct_prediction, adjacent_target
                        ),
                        "vector_relative_l2": relative_l2(
                            direct_prediction, adjacent_target
                        ),
                        "vector_symmetric_norm_ratio": (
                            symmetric_norm_ratio(
                                direct_prediction, adjacent_target
                            )
                        ),
                        "correction_norms_json": "[]",
                        "finite": bool(
                            np.isfinite(direct_prediction).all()
                            and np.isfinite(adjacent_target).all()
                        ),
                    }
                )
                depth_one_prediction = None
                for depth in available_depths(layer, layer_count):
                    if depth == 0:
                        continue
                    stack = FP32TruncatedStack(
                        model,
                        layer,
                        depth,
                        native_b,
                        full_record,
                        anchor,
                        local_templates,
                        layer_templates,
                    )
                    (
                        primal,
                        prediction,
                        method,
                        jvp_seconds,
                        peak_memory,
                    ) = stack.jvp(u_phys)
                    target_output_layer = layer + depth - 1
                    native_baseline = (
                        full_record.layer_outputs[target_output_layer]
                        .float()
                        .numpy()
                    )
                    target = (
                        physical_record.layer_outputs[target_output_layer]
                        .float()
                        - full_record.layer_outputs[target_output_layer].float()
                    ).numpy()
                    baseline_relative = relative_l2(
                        primal, native_baseline
                    )
                    corrections = [
                        float(row["correction_norm"])
                        for row in stack.metadata["corrections"]
                    ]
                    vector_rows.append(
                        {
                            **common,
                            "depth": depth,
                            "target_layer_output": target_output_layer,
                            "prediction_path": (
                                "fp32_dequantized_frozen_kv_truncated_jvp"
                            ),
                            "jvp_method": method,
                            "baseline_relative_l2": baseline_relative,
                            "jvp_seconds": jvp_seconds,
                            "one_token_forward_seconds": physical_seconds,
                            "relative_forward_cost": float(
                                jvp_seconds
                                / max(physical_seconds, 1.0e-30)
                            ),
                            "mlx_peak_memory_bytes": peak_memory,
                            "prediction_norm": float(
                                np.linalg.norm(prediction)
                            ),
                            "truth_norm": float(
                                np.linalg.norm(target)
                            ),
                            "predicted_energy": float(
                                np.dot(prediction, prediction)
                            ),
                            "true_energy": float(
                                np.dot(target, target)
                            ),
                            "vector_cosine": cosine(
                                prediction, target
                            ),
                            "vector_relative_l2": relative_l2(
                                prediction, target
                            ),
                            "vector_symmetric_norm_ratio": (
                                symmetric_norm_ratio(
                                    prediction, target
                                )
                            ),
                            "correction_norms_json": json.dumps(
                                corrections, separators=(",", ":")
                            ),
                            "finite": bool(
                                np.isfinite(prediction).all()
                                and np.isfinite(target).all()
                                and np.isfinite(primal).all()
                            ),
                        }
                    )
                    stack_metadata[
                        f"{sample.sample_id}|{anchor_step}|"
                        f"{candidate.candidate_id}|{layer}|{depth}"
                    ] = stack.metadata
                    if depth == 1:
                        depth_one_prediction = prediction
                if depth_one_prediction is None:
                    raise RuntimeError("depth-one prediction is missing")
                physical_rows.append(
                    {
                        **common,
                        "physical_seconds": physical_seconds,
                        "s0_direct_energy": float(
                            np.dot(u_phys, u_phys)
                        ),
                        "sj1_predicted_energy": float(
                            np.dot(
                                depth_one_prediction,
                                depth_one_prediction,
                            )
                        ),
                        "strue1_physical_energy": float(
                            np.dot(adjacent_target, adjacent_target)
                        ),
                        "exact_kl_full_to_physical": stable_exact_kl(
                            full_logits, physical_logits
                        ),
                        "finite": bool(
                            np.isfinite(full_logits).all()
                            and np.isfinite(physical_logits).all()
                            and np.isfinite(depth_one_prediction).all()
                            and np.isfinite(adjacent_target).all()
                        ),
                    }
                )
        audit_rows.append(
            {
                "sample_id": sample.sample_id,
                "task": sample.task,
                "split": "train",
                "anchor": anchor_step,
                "candidate_count": len(candidates),
                "candidate_distinct_count": len(
                    {candidate.mask_hash for candidate in candidates}
                ),
                "active_budget_min": min(
                    len(candidate.retained_positions)
                    for candidate in candidates
                ),
                "active_budget_max": max(
                    len(candidate.retained_positions)
                    for candidate in candidates
                ),
                "anchor_wall_seconds": (
                    time.perf_counter() - anchor_started
                ),
            }
        )
        print(
            json.dumps(
                {
                    "event": "l2_anchor_complete",
                    "sample_id": sample.sample_id,
                    "anchor": anchor_step,
                    "units": 8 * len(LAYERS),
                    "wall_seconds": audit_rows[-1][
                        "anchor_wall_seconds"
                    ],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        gc.collect()
    vector_frame = pd.DataFrame(vector_rows)
    ranking_frame = make_ranking_table(
        vector_frame,
        (
            "sample_id",
            "task",
            "split",
            "anchor",
            "layer",
            "depth",
        ),
    )
    tables = {
        "depth_vector": vector_frame,
        "depth_ranking": ranking_frame,
        "physical_output": pd.DataFrame(physical_rows),
        "anchor_audit": pd.DataFrame(audit_rows),
    }
    write_sequence_tables(sequence_dir, tables)
    atomic_json(
        sequence_dir / "status.json",
        {
            "stage": "formal_l2",
            "state": "complete",
            "sample_id": sample.sample_id,
            "task": sample.task,
            "split": "train",
            "calibration_or_test_loaded": False,
            "wall_seconds": time.perf_counter() - started,
            "prompt_length": reference.prompt_length,
            "prompt_truncated": reference.prompt_truncated,
            "errors": [],
        },
    )
    return tables


def depth_aggregates(
    vectors: pd.DataFrame, rankings: pd.DataFrame
) -> List[Dict[str, Any]]:
    output = []
    for depth in DEPTHS:
        vector = vectors[vectors["depth"].eq(depth)]
        ranking = rankings[rankings["depth"].eq(depth)]
        sequence_vector = vector.groupby("sample_id").agg(
            vector_cosine=("vector_cosine", "median"),
            vector_relative_l2=("vector_relative_l2", "median"),
            jvp_seconds=("jvp_seconds", "median"),
            relative_forward_cost=("relative_forward_cost", "median"),
            mlx_peak_memory_bytes=("mlx_peak_memory_bytes", "median"),
        )
        sequence_ranking = ranking.groupby("sample_id").agg(
            energy_spearman=("energy_spearman", "median"),
            pairwise_sign_accuracy=(
                "pairwise_sign_accuracy",
                "median",
            ),
            mean_normalized_regret=(
                "mean_normalized_regret",
                "median",
            ),
        )
        joined = sequence_vector.join(sequence_ranking)
        output.append(
            {
                "depth": depth,
                "available_layers": sorted(
                    int(value) for value in vector["layer"].unique()
                ),
                "vector_row_count": int(len(vector)),
                "ranking_group_count": int(len(ranking)),
                "median_vector_cosine": float(
                    joined["vector_cosine"].median()
                ),
                "median_vector_relative_l2": float(
                    joined["vector_relative_l2"].median()
                ),
                "median_energy_spearman": float(
                    joined["energy_spearman"].median()
                ),
                "median_pairwise_sign_accuracy": float(
                    joined["pairwise_sign_accuracy"].median()
                ),
                "median_normalized_regret": float(
                    joined["mean_normalized_regret"].median()
                ),
                "median_jvp_seconds": float(
                    joined["jvp_seconds"].median()
                ),
                "median_relative_forward_cost": float(
                    joined["relative_forward_cost"].median()
                ),
                "median_mlx_peak_memory_bytes": float(
                    joined["mlx_peak_memory_bytes"].median()
                ),
                "sequence_metrics": joined.reset_index().to_dict(
                    "records"
                ),
            }
        )
    return output


def _rho_by_depth(frame: pd.DataFrame) -> Dict[int, float]:
    sequence = (
        frame.groupby(["sample_id", "depth"])["energy_spearman"]
        .median()
        .reset_index()
    )
    return {
        int(depth): float(
            sequence[sequence["depth"].eq(depth)][
                "energy_spearman"
            ].median()
        )
        for depth in DEPTHS
    }


def adjudicate(
    frames: Mapping[str, pd.DataFrame],
    started: float,
    model_info: Mapping[str, Any],
    dataset_events: List[Dict[str, Any]],
) -> Dict[str, Any]:
    vectors = frames["depth_vector"]
    rankings = frames["depth_ranking"]
    common = rankings[rankings["layer"].isin(COMMON_DEPTH_LAYERS)]
    common_rho = _rho_by_depth(common)
    best_propagated = max(
        common_rho[depth] for depth in (1, 2, 4)
    )
    closeness = common_rho[1] >= best_propagated - 0.03
    retention = float(
        (common_rho[1] - common_rho[0])
        / max(best_propagated - common_rho[0] + 1.0e-12, 1.0e-12)
    )
    task_depth_rho: Dict[str, Dict[str, float]] = {}
    task_direction_checks = {}
    for task, task_frame in common.groupby("task"):
        values = _rho_by_depth(task_frame)
        task_depth_rho[str(task)] = {
            str(key): float(value) for key, value in values.items()
        }
        task_direction_checks[str(task)] = bool(
            np.isfinite(list(values.values())).all()
            and values[1] > 0.0
            and values[1] >= values[0]
        )
    expected_depth_counts = {0: 640, 1: 640, 2: 640, 4: 512}
    expected_group_counts = {0: 80, 1: 80, 2: 80, 4: 64}
    integrity_checks = {
        "vector_row_count": len(vectors) == 2432,
        "ranking_group_count": len(rankings) == 304,
        "physical_output_rows": len(frames["physical_output"]) == 640,
        "anchor_audit_rows": len(frames["anchor_audit"]) == 16,
        "depth_vector_counts": all(
            len(vectors[vectors["depth"].eq(depth)]) == count
            for depth, count in expected_depth_counts.items()
        ),
        "depth_group_counts": all(
            len(rankings[rankings["depth"].eq(depth)]) == count
            for depth, count in expected_group_counts.items()
        ),
        "candidate_groups": bool(
            vectors.groupby(
                ["sample_id", "anchor", "layer", "depth"]
            ).size().eq(8).all()
        ),
        "all_finite": bool(
            vectors["finite"].all()
            and rankings["finite"].all()
            and frames["physical_output"]["finite"].all()
            and np.isfinite(rankings["energy_spearman"]).all()
        ),
        "baseline_exact": bool(
            vectors[vectors["depth"].gt(0)][
                "baseline_relative_l2"
            ].max()
            <= 1.0e-12
        ),
        "sequence_ids": (
            set(vectors["sample_id"]) == set(expected_sample_ids())
        ),
        "l1_gate_source": (
            sha256_file(L1_SUMMARY) == L1_SUMMARY_SHA256
            and bool(
                json.loads(L1_SUMMARY.read_text())[
                    "formal_l1_local_passed"
                ]
            )
        ),
    }
    integrity_pass = bool(all(integrity_checks.values()))
    task_consistent = bool(
        len(task_direction_checks) == 2
        and all(task_direction_checks.values())
    )
    professor_support = bool(
        integrity_pass
        and task_consistent
        and (closeness or retention >= 0.80)
    )
    return {
        "stage": "formal_l2",
        "formal_l2_passed": professor_support,
        "professor_adjacent_depth_supported": professor_support,
        "provisional_outcome": (
            "local_outcome_l_a"
            if professor_support
            else "shortest_effective_depth_greater_than_one"
        ),
        "split": "train",
        "calibration_or_test_loaded": False,
        "integrity": {
            "passed": integrity_pass,
            "checks": integrity_checks,
        },
        "adjacent_depth_rule": {
            "common_support_layers": list(COMMON_DEPTH_LAYERS),
            "common_support_rho": {
                str(key): float(value)
                for key, value in common_rho.items()
            },
            "best_propagated_rho": float(best_propagated),
            "k1_loss_to_best": float(
                best_propagated - common_rho[1]
            ),
            "within_0_03_of_best": bool(closeness),
            "retained_fraction_of_ranking_gain": retention,
            "retains_at_least_80_percent": bool(retention >= 0.80),
            "task_depth_rho": task_depth_rho,
            "task_direction_checks": task_direction_checks,
            "task_direction_consistent": task_consistent,
        },
        "depth_aggregates": depth_aggregates(vectors, rankings),
        "row_counts": {
            name: int(len(frame)) for name, frame in frames.items()
        },
        "model_info": dict(model_info),
        "dataset_events": dataset_events,
        "runtime": {
            "wall_seconds": time.perf_counter() - started,
            "peak_rss_bytes": int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            ),
        },
    }


def run(config_path: Path, output_dir: Path) -> Dict[str, Any]:
    import mlx.core as mx

    started = time.perf_counter()
    preregistration = (
        ROOT / "experiments/local_truncated_jacobian/PREREGISTRATION.md"
    )
    if sha256_file(preregistration) != PREREGISTRATION_SHA256:
        raise RuntimeError("formal preregistration checksum mismatch")
    if sha256_file(REGISTRY) != REGISTRY_SHA256:
        raise RuntimeError("formal candidate registry checksum mismatch")
    if sha256_file(L1_SUMMARY) != L1_SUMMARY_SHA256:
        raise RuntimeError("formal L1 summary checksum mismatch")
    l1 = json.loads(L1_SUMMARY.read_text(encoding="utf-8"))
    if not bool(l1["formal_l1_local_passed"]):
        raise RuntimeError("formal L1 gate does not authorize L2")
    cfg = load_discovery_config(str(config_path))
    cfg.validate()
    cfg.independent_fisher.enabled = True
    cfg.independent_fisher.anchors = list(ANCHORS)
    cfg.independent_fisher.segment_horizon = 1
    samples, dataset_events = load_discovery_tasks(cfg)
    if [sample.sample_id for sample in samples] != expected_sample_ids():
        raise RuntimeError("formal L2 sequence isolation failed")
    output_dir.mkdir(parents=True, exist_ok=True)
    status = {
        "stage": "formal_l2",
        "state": "running",
        "split": "train",
        "calibration_or_test_loaded": False,
        "completed_sequences": [],
        "errors": [],
    }
    atomic_json(output_dir / "status.json", status)
    model = MLXTemporalModel(cfg)
    all_frames: Dict[str, List[pd.DataFrame]] = {
        name: [] for name in TABLES
    }
    local_templates: Dict[int, FP32LocalBlock] = {}
    layer_templates: Dict[int, FP32TransformerLayer] = {}
    stack_metadata: Dict[str, Any] = {}
    try:
        model_info = model.load()
        for sample in samples:
            sequence_dir = (
                output_dir / "checkpoints" / safe_id(sample.sample_id)
            )
            sequence_status_path = sequence_dir / "status.json"
            complete = False
            if cfg.runtime.resume and sequence_status_path.exists():
                sequence_status = json.loads(
                    sequence_status_path.read_text(encoding="utf-8")
                )
                complete = (
                    sequence_status.get("state") == "complete"
                    and all(
                        (sequence_dir / f"{name}.parquet").exists()
                        for name in TABLES
                    )
                )
            if complete:
                tables = load_sequence_tables(sequence_dir)
                event = "l2_sequence_resumed"
            else:
                tables = run_sequence(
                    model,
                    sample,
                    sequence_dir,
                    local_templates,
                    layer_templates,
                    stack_metadata,
                )
                event = "l2_sequence_complete"
            for name in TABLES:
                all_frames[name].append(tables[name])
            status["completed_sequences"].append(sample.sample_id)
            atomic_json(output_dir / "status.json", status)
            print(
                json.dumps(
                    {
                        "event": event,
                        "sample_id": sample.sample_id,
                        "completed_sequences": len(
                            status["completed_sequences"]
                        ),
                        "total_sequences": len(samples),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        frames = {
            name: pd.concat(parts, ignore_index=True)
            for name, parts in all_frames.items()
        }
        for name, frame in frames.items():
            atomic_frame(output_dir / f"{name}_rows.parquet", frame)
        summary = adjudicate(
            frames, started, model_info, dataset_events
        )
        summary["runtime"]["mlx_peak_memory_bytes"] = int(
            mx.get_peak_memory()
        )
        layer_metadata = {
            str(layer): weights.metadata
            for layer, weights in layer_templates.items()
        }
        atomic_json(
            output_dir / "fp32_layer_metadata.json", layer_metadata
        )
        atomic_json(
            output_dir / "stack_context_metadata.json", stack_metadata
        )
        atomic_json(output_dir / "l2_summary.json", summary)
        pd.DataFrame(
            [
                {
                    "formal_l2_passed": summary["formal_l2_passed"],
                    "professor_adjacent_depth_supported": summary[
                        "professor_adjacent_depth_supported"
                    ],
                    "k1_loss_to_best": summary["adjacent_depth_rule"][
                        "k1_loss_to_best"
                    ],
                    "retained_fraction_of_ranking_gain": summary[
                        "adjacent_depth_rule"
                    ]["retained_fraction_of_ranking_gain"],
                    "task_direction_consistent": summary[
                        "adjacent_depth_rule"
                    ]["task_direction_consistent"],
                }
            ]
        ).to_csv(output_dir / "l2_summary.csv", index=False)
        status.update(
            {
                "state": "complete",
                "formal_l2_passed": summary["formal_l2_passed"],
                "professor_adjacent_depth_supported": summary[
                    "professor_adjacent_depth_supported"
                ],
            }
        )
        atomic_json(output_dir / "status.json", status)
        return summary
    except Exception as exc:
        status["state"] = "failed"
        status["errors"].append(
            {"type": type(exc).__name__, "message": str(exc)}
        )
        atomic_json(output_dir / "status.json", status)
        raise
    finally:
        if model.runner is not None:
            model.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "experiments/predictive_closure/configs/p0_formal_4bit.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "experiments/local_truncated_jacobian/raw"
        / "l2_depth_ablation/formal",
    )
    args = parser.parse_args()
    result = run(args.config, args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
