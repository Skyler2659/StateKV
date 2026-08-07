#!/usr/bin/env python3
"""Execute the frozen 640-unit train-only Local Stage L0."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import resource
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
PREDICTIVE_SCRIPTS = (
    ROOT / "experiments/predictive_closure/scripts"
)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks/torch"))
sys.path.insert(0, str(PREDICTIVE_SCRIPTS))

from kvbench.temporal.backend_mlx import MLXTemporalModel
from kvbench.temporal.config import load_discovery_config
from kvbench.temporal.tasks import load_discovery_tasks

from mlx_predictive_core import (
    full_selection,
    single_layer_selection,
)

from local_core import (
    FP32LocalBlock,
    atomic_frame,
    atomic_json,
    cosine,
    load_candidates,
    max_pairwise_noise,
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
FORMAL_RADIUS = 3.0e-6
LAYERS = (0, 7, 14, 21, 26)
ANCHORS = (16, 32, 48, 64)
TABLES = (
    "candidate_registry",
    "anchor_audit",
    "direct",
    "native_boundary",
    "jvp_fd",
    "local_baseline",
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


def cache_fingerprint(anchor: Any) -> str:
    digest = hashlib.sha256()
    for layer, (key, value) in enumerate(zip(anchor.keys, anchor.values)):
        digest.update(int(layer).to_bytes(4, "big"))
        digest.update(key.numpy().tobytes())
        digest.update(value.numpy().tobytes())
        digest.update(anchor.position_maps[layer].numpy().tobytes())
    digest.update(str(anchor.logical_length).encode("utf-8"))
    digest.update(str(anchor.query_token_id).encode("utf-8"))
    return digest.hexdigest()


def write_sequence_tables(
    sequence_dir: Path, tables: Mapping[str, List[Dict[str, Any]]]
) -> None:
    for name in TABLES:
        atomic_frame(
            sequence_dir / f"{name}.parquet",
            pd.DataFrame(tables[name]),
        )


def load_sequence_tables(
    sequence_dir: Path,
) -> Dict[str, List[Dict[str, Any]]]:
    return {
        name: pd.read_parquet(
            sequence_dir / f"{name}.parquet"
        ).to_dict("records")
        for name in TABLES
    }


def get_local_block(
    backend: Any,
    templates: Dict[int, FP32LocalBlock],
    metadata: Dict[str, Any],
    layer: int,
    native_b: np.ndarray,
) -> FP32LocalBlock:
    if layer not in templates:
        templates[layer] = FP32LocalBlock(
            backend, layer, native_b
        )
        metadata[str(layer)] = templates[layer].metadata
        return templates[layer]
    return templates[layer].with_base(native_b)


def run_sequence(
    model: MLXTemporalModel,
    sample: Any,
    cfg: Any,
    sequence_dir: Path,
    templates: Dict[int, FP32LocalBlock],
    local_metadata: Dict[str, Any],
) -> Dict[str, List[Dict[str, Any]]]:
    started = time.perf_counter()
    tables: Dict[str, List[Dict[str, Any]]] = {
        name: [] for name in TABLES
    }
    status = {
        "stage": "formal_l0",
        "state": "running",
        "sample_id": sample.sample_id,
        "task": sample.task,
        "split": "train",
        "calibration_or_test_loaded": False,
        "errors": [],
    }
    atomic_json(sequence_dir / "status.json", status)
    reference = model.generate_reference(
        sample_id=sample.sample_id,
        task=sample.task,
        prompt=sample.prompt,
    )
    for anchor_step in ANCHORS:
        anchor_started = time.perf_counter()
        anchor = reference.anchors[int(anchor_step)]
        full_record = reference.query_records[int(anchor_step)]
        before = cache_fingerprint(anchor)
        candidates = load_candidates(
            REGISTRY,
            sample.sample_id,
            anchor_step,
            int(anchor.logical_length - 1),
            REGISTRY_SHA256,
        )
        full_core, full_cache_cfg = full_selection(
            reference, anchor_step
        )
        full_logits, full_replay_record, full_seconds = replay_record(
            model,
            reference,
            anchor_step,
            full_core,
            full_cache_cfg,
        )
        reference_logits = (
            reference.probe_logits[anchor_step].double().numpy()
        )
        layer_output_errors = [
            relative_l2(
                full_replay_record.layer_outputs[layer].double().numpy(),
                full_record.layer_outputs[layer].double().numpy(),
            )
            for layer in LAYERS
        ]
        tables["anchor_audit"].append(
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
                "full_logits_cosine": cosine(
                    full_logits, reference_logits
                ),
                "full_logits_relative_l2": relative_l2(
                    full_logits, reference_logits
                ),
                "full_layer_output_max_relative_l2": max(
                    layer_output_errors
                ),
                "full_replay_seconds": full_seconds,
                "cache_fingerprint_before": before,
                "cache_fingerprint_after": None,
                "cache_fingerprint_invariant": None,
                "anchor_wall_seconds": None,
            }
        )
        for candidate in candidates:
            physical_candidate = to_physical_candidate(candidate)
            registry_common = {
                "sample_id": sample.sample_id,
                "task": sample.task,
                "split": "train",
                "anchor": anchor_step,
                "candidate_id": candidate.candidate_id,
                "candidate_source": candidate.source,
                "mask_hash": candidate.mask_hash,
            }
            tables["candidate_registry"].append(
                {
                    **registry_common,
                    "candidate_seed": candidate.seed,
                    "active_budget": len(
                        candidate.retained_positions
                    ),
                    "core_budget": len(candidate.core_positions),
                    "retained_positions_json": json.dumps(
                        candidate.retained_positions,
                        separators=(",", ":"),
                    ),
                    "core_positions_json": json.dumps(
                        candidate.core_positions,
                        separators=(",", ":"),
                    ),
                }
            )
            for layer in LAYERS:
                common = {**registry_common, "layer": layer}
                u_theory, mass = theoretical_injection(
                    model,
                    reference,
                    anchor_step,
                    candidate.retained_positions,
                    layer,
                )
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
                physical_delta = (
                    physical_record.layer_outputs[layer].float()
                    - full_record.layer_outputs[layer].float()
                ).numpy()
                physical_next_input_delta = (
                    physical_record.residual_inputs[layer + 1].float()
                    - full_record.residual_inputs[layer + 1].float()
                ).numpy()
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
                    injection=u_phys,
                )
                manual_delta = (
                    manual_record.layer_outputs[layer].float()
                    - full_record.layer_outputs[layer].float()
                ).numpy()
                effective_manual_u = (
                    manual_record.projected_attention_outputs[layer].float()
                    - full_record.projected_attention_outputs[layer].float()
                ).numpy()
                tables["direct"].append(
                    {
                        **common,
                        **mass,
                        "u_theory_norm": float(
                            np.linalg.norm(u_theory)
                        ),
                        "u_phys_norm": float(np.linalg.norm(u_phys)),
                        "theory_phys_cosine": cosine(
                            u_theory, u_phys
                        ),
                        "theory_phys_relative_l2": relative_l2(
                            u_theory, u_phys
                        ),
                        "theory_phys_symmetric_norm_ratio": (
                            symmetric_norm_ratio(u_theory, u_phys)
                        ),
                        "finite": bool(
                            np.isfinite(u_theory).all()
                            and np.isfinite(u_phys).all()
                        ),
                    }
                )
                tables["native_boundary"].append(
                    {
                        **common,
                        "physical_seconds": physical_seconds,
                        "manual_seconds": manual_seconds,
                        "physical_delta_norm": float(
                            np.linalg.norm(physical_delta)
                        ),
                        "manual_delta_norm": float(
                            np.linalg.norm(manual_delta)
                        ),
                        "physical_manual_cosine": cosine(
                            manual_delta, physical_delta
                        ),
                        "physical_manual_relative_l2": relative_l2(
                            manual_delta, physical_delta
                        ),
                        "physical_manual_symmetric_norm_ratio": (
                            symmetric_norm_ratio(
                                manual_delta, physical_delta
                            )
                        ),
                        "layer_output_vs_next_input_relative_l2": (
                            relative_l2(
                                physical_next_input_delta,
                                physical_delta,
                            )
                        ),
                        "effective_manual_u_vs_requested_cosine": (
                            cosine(effective_manual_u, u_phys)
                        ),
                        "effective_manual_u_vs_requested_relative_l2": (
                            relative_l2(effective_manual_u, u_phys)
                        ),
                        "pre_target_input_relative_l2": relative_l2(
                            physical_record.residual_inputs[
                                layer
                            ].double().numpy(),
                            full_record.residual_inputs[
                                layer
                            ].double().numpy(),
                        ),
                        "finite": bool(
                            np.isfinite(physical_delta).all()
                            and np.isfinite(manual_delta).all()
                        ),
                    }
                )
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
                local_repeated = [
                    local.baseline(), local.baseline()
                ]
                local_base = local_repeated[0]
                native_next = (
                    full_record.layer_outputs[layer].double().numpy()
                )
                noise = max_pairwise_noise(local_repeated)
                tables["local_baseline"].append(
                    {
                        **common,
                        "noise_norm": noise,
                        "fp32_base_norm": float(
                            np.linalg.norm(local_base)
                        ),
                        "native_base_norm": float(
                            np.linalg.norm(native_next)
                        ),
                        "fp32_native_cosine": cosine(
                            local_base, native_next
                        ),
                        "fp32_native_relative_l2": relative_l2(
                            local_base, native_next
                        ),
                        "fp32_native_symmetric_norm_ratio": (
                            symmetric_norm_ratio(
                                local_base, native_next
                            )
                        ),
                        "finite": bool(
                            np.isfinite(local_base).all()
                            and np.isfinite(native_next).all()
                        ),
                    }
                )
                for direction_name, direction in (
                    ("u_phys", u_phys),
                    ("u_theory", u_theory),
                ):
                    base_jvp, derivative, method = local.jvp(
                        direction
                    )
                    fd = local.symmetric_fd(
                        direction, FORMAL_RADIUS
                    )
                    fd_derivative = fd["derivative"]
                    tables["jvp_fd"].append(
                        {
                            **common,
                            "direction": direction_name,
                            "radius": FORMAL_RADIUS,
                            "epsilon_absolute": fd[
                                "epsilon_absolute"
                            ],
                            "direction_norm": float(
                                np.linalg.norm(direction)
                            ),
                            "noise_norm": noise,
                            "jvp_method": method,
                            "jvp_primal_vs_base_relative_l2": (
                                relative_l2(base_jvp, local_base)
                            ),
                            "jvp_norm": float(
                                np.linalg.norm(derivative)
                            ),
                            "fd_norm": float(
                                np.linalg.norm(fd_derivative)
                            ),
                            "jvp_fd_cosine": cosine(
                                derivative, fd_derivative
                            ),
                            "jvp_fd_relative_l2": relative_l2(
                                derivative, fd_derivative
                            ),
                            "jvp_fd_symmetric_norm_ratio": (
                                symmetric_norm_ratio(
                                    derivative, fd_derivative
                                )
                            ),
                            "fd_to_noise_ratio": float(
                                np.linalg.norm(fd_derivative)
                                / max(noise, 1.0e-30)
                            ),
                            "finite": bool(
                                np.isfinite(derivative).all()
                                and np.isfinite(fd_derivative).all()
                            ),
                        }
                    )
        after = cache_fingerprint(anchor)
        audit_row = tables["anchor_audit"][-1]
        audit_row["cache_fingerprint_after"] = after
        audit_row["cache_fingerprint_invariant"] = before == after
        audit_row["anchor_wall_seconds"] = (
            time.perf_counter() - anchor_started
        )
        print(
            json.dumps(
                {
                    "event": "l0_anchor_complete",
                    "sample_id": sample.sample_id,
                    "anchor": anchor_step,
                    "units": 8 * len(LAYERS),
                    "wall_seconds": audit_row[
                        "anchor_wall_seconds"
                    ],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        gc.collect()
    write_sequence_tables(sequence_dir, tables)
    status.update(
        {
            "state": "complete",
            "wall_seconds": time.perf_counter() - started,
            "prompt_length": reference.prompt_length,
            "prompt_truncated": reference.prompt_truncated,
        }
    )
    atomic_json(sequence_dir / "status.json", status)
    return tables


def adjudicate(
    frames: Mapping[str, pd.DataFrame],
    started: float,
    model_info: Mapping[str, Any],
    dataset_events: List[Dict[str, Any]],
) -> Dict[str, Any]:
    registry = frames["candidate_registry"]
    audit = frames["anchor_audit"]
    direct = frames["direct"]
    boundary = frames["native_boundary"]
    jvp = frames["jvp_fd"]
    local_base = frames["local_baseline"]
    primary_jvp = jvp[jvp["direction"].eq("u_phys")]
    task_boundary = (
        boundary.groupby("task")["physical_manual_cosine"]
        .median()
        .to_dict()
    )
    layer_boundary = (
        boundary.groupby("layer")["physical_manual_cosine"]
        .median()
        .to_dict()
    )
    task_jvp = (
        primary_jvp.groupby("task")["jvp_fd_cosine"]
        .median()
        .to_dict()
    )
    integrity_checks = {
        "sequence_count": registry["sample_id"].nunique() == 4,
        "sequence_ids": set(registry["sample_id"])
        == set(expected_sample_ids()),
        "anchor_count": registry["anchor"].nunique() == 4,
        "candidate_count_per_group": bool(
            registry.groupby(["sample_id", "anchor"]).size().eq(8).all()
        ),
        "candidate_distinct": bool(
            registry.groupby(["sample_id", "anchor"])[
                "mask_hash"
            ].nunique().eq(8).all()
        ),
        "active_budget": bool(registry["active_budget"].eq(128).all()),
        "registry_checksum": sha256_file(REGISTRY)
        == REGISTRY_SHA256,
        "cache_fingerprint": bool(
            audit["cache_fingerprint_invariant"].all()
        ),
        "full_replay_reference": bool(
            audit["full_logits_relative_l2"].max() <= 1.0e-12
            and audit["full_layer_output_max_relative_l2"].max()
            <= 1.0e-12
        ),
        "path_outputs_finite": bool(
            direct["finite"].all()
            and boundary["finite"].all()
            and jvp["finite"].all()
            and local_base["finite"].all()
        ),
    }
    native_checks = {
        "pooled_median_cosine": float(
            boundary["physical_manual_cosine"].median()
        )
        >= 0.999,
        "govreport_median_cosine": float(
            task_boundary.get("gov_report", float("-inf"))
        )
        >= 0.995,
        "niah_median_cosine": float(
            task_boundary.get("niah_single_1", float("-inf"))
        )
        >= 0.995,
        "all_layer_medians": bool(
            all(float(value) >= 0.99 for value in layer_boundary.values())
            and len(layer_boundary) == 5
        ),
        "all_finite": bool(boundary["finite"].all()),
    }
    jvp_checks = {
        "pooled_median_cosine": float(
            primary_jvp["jvp_fd_cosine"].median()
        )
        >= 0.99,
        "govreport_median_cosine": float(
            task_jvp.get("gov_report", float("-inf"))
        )
        >= 0.99,
        "niah_median_cosine": float(
            task_jvp.get("niah_single_1", float("-inf"))
        )
        >= 0.99,
        "median_relative_l2": float(
            primary_jvp["jvp_fd_relative_l2"].median()
        )
        <= 0.10,
        "all_finite": bool(primary_jvp["finite"].all()),
    }
    integrity_pass = bool(all(integrity_checks.values()))
    native_pass = bool(all(native_checks.values()))
    jvp_pass = bool(all(jvp_checks.values()))
    l0_pass = bool(integrity_pass and native_pass and jvp_pass)
    if not native_pass:
        stop_reason = "l0_native_adjacent_boundary_failed"
        outcome = "local_outcome_l_d"
    elif not jvp_pass:
        stop_reason = "l0_fp32_jvp_fd_failed"
        outcome = "local_differentiable_map_not_certified"
    elif not integrity_pass:
        stop_reason = "l0_integrity_failed"
        outcome = "invalid_l0"
    else:
        stop_reason = None
        outcome = "pending_l1"
    return {
        "stage": "formal_l0",
        "formal_l0_passed": l0_pass,
        "stop_l1_plus": not l0_pass,
        "stop_reason": stop_reason,
        "provisional_outcome": outcome,
        "formal_radius": FORMAL_RADIUS,
        "split": "train",
        "calibration_or_test_loaded": False,
        "integrity": {
            "passed": integrity_pass,
            "checks": integrity_checks,
        },
        "native_boundary": {
            "passed": native_pass,
            "checks": native_checks,
            "pooled_median_cosine": float(
                boundary["physical_manual_cosine"].median()
            ),
            "pooled_median_relative_l2": float(
                boundary["physical_manual_relative_l2"].median()
            ),
            "task_median_cosine": {
                str(key): float(value)
                for key, value in task_boundary.items()
            },
            "layer_median_cosine": {
                str(key): float(value)
                for key, value in layer_boundary.items()
            },
        },
        "jvp_fd": {
            "passed": jvp_pass,
            "checks": jvp_checks,
            "pooled_median_cosine": float(
                primary_jvp["jvp_fd_cosine"].median()
            ),
            "pooled_median_relative_l2": float(
                primary_jvp["jvp_fd_relative_l2"].median()
            ),
            "task_median_cosine": {
                str(key): float(value)
                for key, value in task_jvp.items()
            },
            "method_counts": {
                str(key): int(value)
                for key, value in primary_jvp[
                    "jvp_method"
                ].value_counts().items()
            },
        },
        "direct_diagnostic": {
            "theory_phys_median_cosine": float(
                direct["theory_phys_cosine"].median()
            ),
            "theory_phys_median_relative_l2": float(
                direct["theory_phys_relative_l2"].median()
            ),
            "theory_phys_median_symmetric_norm_ratio": float(
                direct["theory_phys_symmetric_norm_ratio"].median()
            ),
        },
        "local_baseline_diagnostic": {
            "fp32_native_median_cosine": float(
                local_base["fp32_native_cosine"].median()
            ),
            "fp32_native_median_relative_l2": float(
                local_base["fp32_native_relative_l2"].median()
            ),
        },
        "row_counts": {
            name: int(len(frame)) for name, frame in frames.items()
        },
        "sequence_ids": sorted(
            registry["sample_id"].unique().tolist()
        ),
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
    prereg_path = (
        ROOT
        / "experiments/local_truncated_jacobian/PREREGISTRATION.md"
    )
    if sha256_file(prereg_path) != PREREGISTRATION_SHA256:
        raise RuntimeError("formal preregistration checksum mismatch")
    if sha256_file(REGISTRY) != REGISTRY_SHA256:
        raise RuntimeError("formal candidate registry checksum mismatch")
    cfg = load_discovery_config(str(config_path))
    cfg.validate()
    cfg.independent_fisher.enabled = True
    cfg.independent_fisher.anchors = list(ANCHORS)
    cfg.independent_fisher.segment_horizon = 1
    samples, dataset_events = load_discovery_tasks(cfg)
    actual_ids = [sample.sample_id for sample in samples]
    if actual_ids != expected_sample_ids():
        raise RuntimeError(
            f"formal L0 sequence isolation failed: {actual_ids}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    status = {
        "stage": "formal_l0",
        "state": "running",
        "formal_radius": FORMAL_RADIUS,
        "split": "train",
        "calibration_or_test_loaded": False,
        "completed_sequences": [],
        "errors": [],
    }
    atomic_json(output_dir / "status.json", status)
    model = MLXTemporalModel(cfg)
    all_tables: Dict[str, List[Dict[str, Any]]] = {
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
                event = "l0_sequence_resumed"
            else:
                tables = run_sequence(
                    model,
                    sample,
                    cfg,
                    sequence_dir,
                    templates,
                    local_metadata,
                )
                event = "l0_sequence_complete"
            for name in TABLES:
                all_tables[name].extend(tables[name])
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
            name: pd.DataFrame(rows)
            for name, rows in all_tables.items()
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
        atomic_json(output_dir / "l0_summary.json", summary)
        pd.DataFrame(
            [
                {
                    "formal_l0_passed": summary[
                        "formal_l0_passed"
                    ],
                    "stop_l1_plus": summary["stop_l1_plus"],
                    "stop_reason": summary["stop_reason"],
                    "native_boundary_pooled_median_cosine": summary[
                        "native_boundary"
                    ]["pooled_median_cosine"],
                    "jvp_fd_pooled_median_cosine": summary[
                        "jvp_fd"
                    ]["pooled_median_cosine"],
                    "jvp_fd_pooled_median_relative_l2": summary[
                        "jvp_fd"
                    ]["pooled_median_relative_l2"],
                }
            ]
        ).to_csv(output_dir / "l0_summary.csv", index=False)
        status.update(
            {
                "state": "complete",
                "formal_l0_passed": summary["formal_l0_passed"],
                "stop_l1_plus": summary["stop_l1_plus"],
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
        / "experiments/local_truncated_jacobian/raw/l0_boundary/formal",
    )
    args = parser.parse_args()
    result = run(args.config, args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
