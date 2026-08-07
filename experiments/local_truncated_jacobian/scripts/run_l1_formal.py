#!/usr/bin/env python3
"""Execute frozen Stage L1 local linearization and native transfer."""
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
from scipy.stats import rankdata

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

from mlx_predictive_core import full_selection, single_layer_selection

from local_core import (
    FP32LocalBlock,
    atomic_frame,
    atomic_json,
    cosine,
    load_candidates,
    relative_l2,
    replay_record,
    sha256_file,
    symmetric_norm_ratio,
    theoretical_injection,
    to_physical_candidate,
)


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
L0_SUMMARY = (
    ROOT
    / "experiments/local_truncated_jacobian/raw/l0_boundary"
    / "formal/l0_summary.json"
)
L0_SUMMARY_SHA256 = (
    "33c160e37677e3e1f0c343c4d16e791779d6df63f98b37c2b8cfa95554545f80"
)
LAYERS = (0, 7, 14, 21, 26)
ANCHORS = (16, 32, 48, 64)
SCALES = (0.125, 0.25, 0.5, 1.0, 2.0)
TABLES = (
    "local_vector",
    "local_ranking",
    "native_transfer_vector",
    "native_transfer_ranking",
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


def _spearman(predicted: Sequence[float], truth: Sequence[float]) -> float:
    left = np.asarray(predicted, dtype=np.float64)
    right = np.asarray(truth, dtype=np.float64)
    if (
        left.size < 2
        or not np.isfinite(left).all()
        or not np.isfinite(right).all()
        or np.all(left == left[0])
        or np.all(right == right[0])
    ):
        return float("nan")
    left_rank = rankdata(left, method="average")
    right_rank = rankdata(right, method="average")
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _pairwise_sign_accuracy(
    predicted: Sequence[float], truth: Sequence[float]
) -> float:
    left = np.asarray(predicted, dtype=np.float64)
    right = np.asarray(truth, dtype=np.float64)
    correct = 0
    total = 0
    for first in range(len(left)):
        for second in range(first + 1, len(left)):
            predicted_sign = np.sign(left[first] - left[second])
            truth_sign = np.sign(right[first] - right[second])
            correct += int(predicted_sign == truth_sign)
            total += 1
    return float(correct / total) if total else float("nan")


def _normalized_regret(
    predicted: Sequence[float], truth: Sequence[float]
) -> float:
    score = np.asarray(predicted, dtype=np.float64)
    target = np.asarray(truth, dtype=np.float64)
    selected = int(np.nanargmin(score))
    return float(
        (target[selected] - np.nanmin(target))
        / max(float(np.nanmax(target) - np.nanmin(target)), 1.0e-30)
    )


def ranking_row(
    frame: pd.DataFrame,
    group_columns: Sequence[str],
) -> Dict[str, Any]:
    if len(frame) != 8 or frame["mask_hash"].nunique() != 8:
        raise RuntimeError("L1 ranking group is not eight-distinct")
    predicted = frame["predicted_energy"].to_numpy(dtype=np.float64)
    truth = frame["true_energy"].to_numpy(dtype=np.float64)
    predicted_order = np.argsort(predicted, kind="stable")
    truth_order = np.argsort(truth, kind="stable")
    common = {
        column: frame.iloc[0][column] for column in group_columns
    }
    return {
        **common,
        "candidate_count": 8,
        "energy_spearman": _spearman(predicted, truth),
        "pairwise_sign_accuracy": _pairwise_sign_accuracy(
            predicted, truth
        ),
        "top1_recall": float(predicted_order[0] == truth_order[0]),
        "top3_recall": float(
            len(set(predicted_order[:3]) & set(truth_order[:3])) / 3.0
        ),
        "mean_normalized_regret": _normalized_regret(
            predicted, truth
        ),
        "finite": bool(
            np.isfinite(predicted).all() and np.isfinite(truth).all()
        ),
    }


def make_ranking_table(
    frame: pd.DataFrame, group_columns: Sequence[str]
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            ranking_row(group, group_columns)
            for _key, group in frame.groupby(
                list(group_columns), sort=False, dropna=False
            )
        ]
    )


def get_local_block(
    backend: Any,
    templates: Dict[int, FP32LocalBlock],
    metadata: Dict[str, Any],
    layer: int,
    native_b: np.ndarray,
) -> FP32LocalBlock:
    if layer not in templates:
        templates[layer] = FP32LocalBlock(backend, layer, native_b)
        metadata[str(layer)] = templates[layer].metadata
        return templates[layer]
    return templates[layer].with_base(native_b)


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
    templates: Dict[int, FP32LocalBlock],
    local_metadata: Dict[str, Any],
) -> Dict[str, pd.DataFrame]:
    started = time.perf_counter()
    local_rows: List[Dict[str, Any]] = []
    transfer_rows: List[Dict[str, Any]] = []
    audit_rows: List[Dict[str, Any]] = []
    atomic_json(
        sequence_dir / "status.json",
        {
            "stage": "formal_l1",
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
    for anchor_step in ANCHORS:
        anchor_started = time.perf_counter()
        anchor = reference.anchors[int(anchor_step)]
        full_record = reference.query_records[int(anchor_step)]
        candidates = load_candidates(
            REGISTRY,
            sample.sample_id,
            anchor_step,
            int(anchor.logical_length - 1),
            REGISTRY_SHA256,
        )
        full_core, full_cache_cfg = full_selection(reference, anchor_step)
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
                    _physical_logits,
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
                u_theory, _mass = theoretical_injection(
                    model,
                    reference,
                    anchor_step,
                    candidate.retained_positions,
                    layer,
                )
                physical_delta = (
                    physical_record.layer_outputs[layer].float()
                    - full_record.layer_outputs[layer].float()
                ).numpy()
                native_b = (
                    full_record.post_attention_residuals[layer]
                    .float()
                    .numpy()
                )
                local = get_local_block(
                    model,
                    templates,
                    local_metadata,
                    layer,
                    native_b,
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
                derivatives: Dict[str, np.ndarray] = {}
                directions = {
                    "u_phys": u_phys,
                    "u_theory": u_theory,
                }
                for direction_name, direction in directions.items():
                    _base, derivative, method = local.jvp(direction)
                    derivatives[direction_name] = derivative
                    for scale in SCALES:
                        prediction = float(scale) * derivative
                        truth = local.nonlinear_delta(direction, scale)
                        local_rows.append(
                            {
                                **common,
                                "direction": direction_name,
                                "scale": float(scale),
                                "jvp_method": method,
                                "prediction_norm": float(
                                    np.linalg.norm(prediction)
                                ),
                                "truth_norm": float(
                                    np.linalg.norm(truth)
                                ),
                                "predicted_energy": float(
                                    np.dot(prediction, prediction)
                                ),
                                "true_energy": float(
                                    np.dot(truth, truth)
                                ),
                                "vector_cosine": cosine(
                                    prediction, truth
                                ),
                                "vector_relative_l2": relative_l2(
                                    prediction, truth
                                ),
                                "vector_symmetric_norm_ratio": (
                                    symmetric_norm_ratio(
                                        prediction, truth
                                    )
                                ),
                                "finite": bool(
                                    np.isfinite(prediction).all()
                                    and np.isfinite(truth).all()
                                ),
                            }
                        )
                for direction_name, direction in directions.items():
                    (
                        _manual_logits,
                        manual_record,
                        manual_seconds,
                    ) = replay_record(
                        model,
                        reference,
                        anchor_step,
                        full_core,
                        full_cache_cfg,
                        injection_layer=layer,
                        injection=direction,
                    )
                    manual_delta = (
                        manual_record.layer_outputs[layer].float()
                        - full_record.layer_outputs[layer].float()
                    ).numpy()
                    targets = [("native_manual", manual_delta)]
                    if direction_name == "u_phys":
                        targets.append(("native_physical", physical_delta))
                    for target_name, target in targets:
                        prediction = derivatives[direction_name]
                        transfer_rows.append(
                            {
                                **common,
                                "direction": direction_name,
                                "scale": 1.0,
                                "target_path": target_name,
                                "physical_seconds": physical_seconds,
                                "manual_seconds": manual_seconds,
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
                                "finite": bool(
                                    np.isfinite(prediction).all()
                                    and np.isfinite(target).all()
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
                    "event": "l1_anchor_complete",
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
    local_frame = pd.DataFrame(local_rows)
    transfer_frame = pd.DataFrame(transfer_rows)
    local_groups = make_ranking_table(
        local_frame,
        (
            "sample_id",
            "task",
            "split",
            "anchor",
            "layer",
            "direction",
            "scale",
        ),
    )
    transfer_groups = make_ranking_table(
        transfer_frame,
        (
            "sample_id",
            "task",
            "split",
            "anchor",
            "layer",
            "direction",
            "scale",
            "target_path",
        ),
    )
    tables = {
        "local_vector": local_frame,
        "local_ranking": local_groups,
        "native_transfer_vector": transfer_frame,
        "native_transfer_ranking": transfer_groups,
        "anchor_audit": pd.DataFrame(audit_rows),
    }
    write_sequence_tables(sequence_dir, tables)
    atomic_json(
        sequence_dir / "status.json",
        {
            "stage": "formal_l1",
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


def _sequence_median(frame: pd.DataFrame, column: str) -> float:
    return float(
        frame.groupby("sample_id")[column].median().median()
    )


def gate_summary(
    vectors: pd.DataFrame,
    rankings: pd.DataFrame,
    target_label: str,
) -> Dict[str, Any]:
    task_rho = (
        rankings.groupby("task")["energy_spearman"].median().to_dict()
    )
    layer_rho = (
        rankings.groupby("layer")["energy_spearman"].median().to_dict()
    )
    sequence_metrics = (
        vectors.groupby("sample_id")
        .agg(
            vector_cosine=("vector_cosine", "median"),
            vector_relative_l2=("vector_relative_l2", "median"),
        )
        .join(
            rankings.groupby("sample_id").agg(
                energy_spearman=("energy_spearman", "median"),
                pairwise_sign_accuracy=(
                    "pairwise_sign_accuracy",
                    "median",
                ),
            )
        )
        .reset_index()
    )
    pooled = {
        "vector_cosine": float(
            sequence_metrics["vector_cosine"].median()
        ),
        "vector_relative_l2": float(
            sequence_metrics["vector_relative_l2"].median()
        ),
        "energy_spearman": float(
            sequence_metrics["energy_spearman"].median()
        ),
        "pairwise_sign_accuracy": float(
            sequence_metrics["pairwise_sign_accuracy"].median()
        ),
    }
    checks = {
        "median_vector_cosine": pooled["vector_cosine"] >= 0.95,
        "median_vector_relative_l2": (
            pooled["vector_relative_l2"] <= 0.30
        ),
        "candidate_energy_spearman": (
            pooled["energy_spearman"] >= 0.85
        ),
        "pairwise_sign_accuracy": (
            pooled["pairwise_sign_accuracy"] >= 0.80
        ),
        "task_direction_consistent": bool(
            len(task_rho) == 2
            and all(
                np.isfinite(value) and float(value) > 0.0
                for value in task_rho.values()
            )
        ),
        "four_of_five_layers": bool(
            sum(
                np.isfinite(value) and float(value) >= 0.75
                for value in layer_rho.values()
            )
            >= 4
        ),
        "all_finite": bool(
            vectors["finite"].all()
            and rankings["finite"].all()
            and np.isfinite(rankings["energy_spearman"]).all()
        ),
    }
    return {
        "target": target_label,
        "passed": bool(all(checks.values())),
        "checks": checks,
        "sequence_independent_aggregation": (
            "within-group metrics; median within each sequence; "
            "median across four sequences"
        ),
        "pooled_sequence_medians": pooled,
        "task_median_energy_spearman": {
            str(key): float(value) for key, value in task_rho.items()
        },
        "layer_median_energy_spearman": {
            str(key): float(value) for key, value in layer_rho.items()
        },
        "sequence_metrics": sequence_metrics.to_dict("records"),
    }


def adjudicate(
    frames: Mapping[str, pd.DataFrame],
    started: float,
    model_info: Mapping[str, Any],
    dataset_events: List[Dict[str, Any]],
) -> Dict[str, Any]:
    local_vectors = frames["local_vector"]
    local_rankings = frames["local_ranking"]
    transfer_vectors = frames["native_transfer_vector"]
    transfer_rankings = frames["native_transfer_ranking"]
    local_primary_vectors = local_vectors[
        local_vectors["direction"].eq("u_phys")
        & local_vectors["scale"].eq(1.0)
    ]
    local_primary_rankings = local_rankings[
        local_rankings["direction"].eq("u_phys")
        & local_rankings["scale"].eq(1.0)
    ]
    native_primary_vectors = transfer_vectors[
        transfer_vectors["direction"].eq("u_phys")
        & transfer_vectors["target_path"].eq("native_physical")
    ]
    native_primary_rankings = transfer_rankings[
        transfer_rankings["direction"].eq("u_phys")
        & transfer_rankings["target_path"].eq("native_physical")
    ]
    local_gate = gate_summary(
        local_primary_vectors,
        local_primary_rankings,
        "fp32_nonlinear_local_response",
    )
    native_gate = gate_summary(
        native_primary_vectors,
        native_primary_rankings,
        "native_physical_next_hidden",
    )
    integrity_checks = {
        "local_vector_rows": len(local_vectors) == 6400,
        "local_ranking_rows": len(local_rankings) == 800,
        "native_transfer_vector_rows": len(transfer_vectors) == 1920,
        "native_transfer_ranking_rows": len(transfer_rankings) == 240,
        "anchor_audit_rows": len(frames["anchor_audit"]) == 16,
        "sequence_ids": (
            set(local_vectors["sample_id"]) == set(expected_sample_ids())
        ),
        "scales": set(local_vectors["scale"]) == set(SCALES),
        "layers": set(local_vectors["layer"]) == set(LAYERS),
        "directions": set(local_vectors["direction"])
        == {"u_phys", "u_theory"},
        "candidate_groups": bool(
            local_vectors.groupby(
                [
                    "sample_id",
                    "anchor",
                    "layer",
                    "direction",
                    "scale",
                ]
            ).size().eq(8).all()
        ),
        "all_finite": bool(
            local_vectors["finite"].all()
            and transfer_vectors["finite"].all()
        ),
        "l0_gate_source": (
            sha256_file(L0_SUMMARY) == L0_SUMMARY_SHA256
            and bool(json.loads(L0_SUMMARY.read_text())["formal_l0_passed"])
        ),
    }
    integrity_pass = bool(all(integrity_checks.values()))
    local_pass = bool(integrity_pass and local_gate["passed"])
    native_pass = bool(integrity_pass and native_gate["passed"])
    if not local_pass:
        outcome = "local_outcome_l_c"
        stop_reason = "l1_natural_magnitude_local_gate_failed"
    elif not native_pass:
        outcome = "local_outcome_l_b"
        stop_reason = "l1_native_physical_transfer_failed"
    else:
        outcome = "pending_l2_l3"
        stop_reason = None
    scale_summary = (
        local_vectors.groupby(["direction", "scale"])
        .agg(
            median_vector_cosine=("vector_cosine", "median"),
            median_vector_relative_l2=("vector_relative_l2", "median"),
            median_norm_ratio=("vector_symmetric_norm_ratio", "median"),
        )
        .reset_index()
        .to_dict("records")
    )
    rank_scale_summary = (
        local_rankings.groupby(["direction", "scale"])
        .agg(
            median_energy_spearman=("energy_spearman", "median"),
            median_pairwise_accuracy=(
                "pairwise_sign_accuracy",
                "median",
            ),
            median_top1_recall=("top1_recall", "median"),
            median_top3_recall=("top3_recall", "median"),
            median_regret=("mean_normalized_regret", "median"),
        )
        .reset_index()
        .to_dict("records")
    )
    return {
        "stage": "formal_l1",
        "formal_l1_local_passed": local_pass,
        "formal_l1_native_transfer_passed": native_pass,
        "proceed_l2_l3": bool(local_pass),
        "stop_reason": stop_reason,
        "provisional_outcome": outcome,
        "split": "train",
        "calibration_or_test_loaded": False,
        "integrity": {
            "passed": integrity_pass,
            "checks": integrity_checks,
        },
        "local_primary_gate": local_gate,
        "native_primary_transfer_gate": native_gate,
        "scale_summary": scale_summary,
        "ranking_scale_summary": rank_scale_summary,
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
    if sha256_file(L0_SUMMARY) != L0_SUMMARY_SHA256:
        raise RuntimeError("formal L0 summary checksum mismatch")
    l0 = json.loads(L0_SUMMARY.read_text(encoding="utf-8"))
    if not bool(l0["formal_l0_passed"]) or bool(l0["stop_l1_plus"]):
        raise RuntimeError("formal L0 gate does not authorize L1")
    cfg = load_discovery_config(str(config_path))
    cfg.validate()
    cfg.independent_fisher.enabled = True
    cfg.independent_fisher.anchors = list(ANCHORS)
    cfg.independent_fisher.segment_horizon = 1
    samples, dataset_events = load_discovery_tasks(cfg)
    if [sample.sample_id for sample in samples] != expected_sample_ids():
        raise RuntimeError("formal L1 sequence isolation failed")
    output_dir.mkdir(parents=True, exist_ok=True)
    status = {
        "stage": "formal_l1",
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
    templates: Dict[int, FP32LocalBlock] = {}
    local_metadata: Dict[str, Any] = {}
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
                event = "l1_sequence_resumed"
            else:
                tables = run_sequence(
                    model,
                    sample,
                    sequence_dir,
                    templates,
                    local_metadata,
                )
                event = "l1_sequence_complete"
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
        atomic_json(output_dir / "local_block_metadata.json", local_metadata)
        atomic_json(output_dir / "l1_summary.json", summary)
        pd.DataFrame(
            [
                {
                    "formal_l1_local_passed": summary[
                        "formal_l1_local_passed"
                    ],
                    "formal_l1_native_transfer_passed": summary[
                        "formal_l1_native_transfer_passed"
                    ],
                    "proceed_l2_l3": summary["proceed_l2_l3"],
                    "stop_reason": summary["stop_reason"],
                    "local_vector_cosine": summary[
                        "local_primary_gate"
                    ]["pooled_sequence_medians"]["vector_cosine"],
                    "local_energy_spearman": summary[
                        "local_primary_gate"
                    ]["pooled_sequence_medians"]["energy_spearman"],
                    "native_vector_cosine": summary[
                        "native_primary_transfer_gate"
                    ]["pooled_sequence_medians"]["vector_cosine"],
                    "native_energy_spearman": summary[
                        "native_primary_transfer_gate"
                    ]["pooled_sequence_medians"]["energy_spearman"],
                }
            ]
        ).to_csv(output_dir / "l1_summary.csv", index=False)
        status.update(
            {
                "state": "complete",
                "formal_l1_local_passed": summary[
                    "formal_l1_local_passed"
                ],
                "formal_l1_native_transfer_passed": summary[
                    "formal_l1_native_transfer_passed"
                ],
                "proceed_l2_l3": summary["proceed_l2_l3"],
                "stop_reason": summary["stop_reason"],
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
        / "l1_local_linearization/formal",
    )
    args = parser.parse_args()
    result = run(args.config, args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
