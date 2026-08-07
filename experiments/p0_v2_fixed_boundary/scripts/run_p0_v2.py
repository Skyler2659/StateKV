#!/usr/bin/env python3
"""Run smoke, calibration, and held-out evaluation for fixed-boundary P0-v2."""
from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
PREDICTIVE_DIR = ROOT / "experiments/predictive_closure/scripts"
for value in (ROOT, ROOT / "benchmarks/torch", PREDICTIVE_DIR, SCRIPT_DIR):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from kvbench.temporal.config import DiscoveryConfig
from kvbench.temporal.tasks import load_discovery_tasks
from mlx_predictive_core import make_selector_candidates
from precision_diagnostic import (
    count_quantized_modules,
    dequantize_reference_model,
    layer_identity_and_injection,
)

from p0_v2_core import (
    AdjacentBoundaryMap,
    FixedBoundaryReadoutMap,
    P0V2FP32TemporalModel,
    add_identity_conditioning,
    atomic_frame,
    atomic_json,
    exact_kl,
    fisher_variance,
    full_replay,
    manual_boundary_replay,
    physical_single_layer_replay,
    prefixed_metrics,
    ranking_metrics,
    sha256_file,
    stable_softmax,
    vector_metrics,
)


TABLE_NAMES = (
    "response_rows",
    "identity_rows",
    "candidate_registry",
    "unit_audit",
)


def load_protocol(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        protocol = yaml.safe_load(handle)
    required = {
        "schema_version",
        "experiment",
        "scope",
        "model",
        "data",
        "cache",
        "candidates",
        "numeric",
        "metrics",
        "gates",
        "outcomes",
        "runtime",
    }
    missing = sorted(required - set(protocol))
    if missing:
        raise ValueError(f"P0-v2 config missing fields: {missing}")
    if protocol["scope"]["prohibited"] != [
        "historical_accumulated_state",
        "temporal_transition",
        "E2",
        "E2_J1",
        "refresh_policy",
        "future_query_prediction",
        "future_attention_oracle",
        "multi_step_horizon",
        "free_generation",
        "joint_multilayer_mask",
        "multilayer_response_sum",
        "selection_refresh_controller",
        "low_rank_Q",
        "post_test_threshold_tuning",
    ]:
        raise ValueError("frozen prohibited-scope list changed")
    return protocol


def stage_indices(protocol: Mapping[str, Any], stage: str) -> Tuple[List[int], List[int]]:
    data = protocol["data"]
    if stage == "evaluation":
        section = data["evaluation"]
    elif stage in {"smoke", "calibration"}:
        section = data[stage]
    else:
        raise ValueError(f"unknown data stage: {stage}")
    return (
        [int(value) for value in section["gov_report_indices"]],
        [int(value) for value in section["niah_offsets"]],
    )


def expected_ids(protocol: Mapping[str, Any], stage: str) -> List[str]:
    gov, niah = stage_indices(protocol, stage)
    return [
        *[f"gov_report:{value}" for value in gov],
        *[f"synthetic_niah_{value}" for value in niah],
    ]


def validate_split_isolation(protocol: Mapping[str, Any]) -> Dict[str, Any]:
    calibration = set(expected_ids(protocol, "calibration"))
    evaluation = set(expected_ids(protocol, "evaluation"))
    historical = set(
        str(value)
        for value in protocol["data"]["forbidden_historical_test_ids"]
    )
    checks = {
        "calibration_evaluation_disjoint": calibration.isdisjoint(evaluation),
        "evaluation_historical_disjoint": evaluation.isdisjoint(historical),
        "two_tasks_in_evaluation": bool(
            any(value.startswith("gov_report:") for value in evaluation)
            and any(value.startswith("synthetic_niah_") for value in evaluation)
        ),
        "multiple_sequences_per_task": bool(
            sum(value.startswith("gov_report:") for value in evaluation) >= 2
            and sum(value.startswith("synthetic_niah_") for value in evaluation)
            >= 2
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"split isolation failed: {checks}")
    return {
        "checks": checks,
        "calibration_ids": sorted(calibration),
        "evaluation_ids": sorted(evaluation),
        "historical_forbidden_ids": sorted(historical),
    }


def build_discovery_config(
    protocol: Mapping[str, Any], stage: str
) -> DiscoveryConfig:
    cfg = DiscoveryConfig()
    cfg.experiment_name = f"{protocol['experiment']}_{stage}"
    cfg.model.name = str(protocol["model"]["source"])
    cfg.model.dtype = "4bit"
    cfg.model.backend = "mlx"
    cfg.model.quant_bits = 4
    cfg.model.deterministic = True
    cfg.model.temperature = 0.0
    cfg.model.do_sample = False
    cfg.model.revision = str(protocol["model"]["revision"])
    cfg.model.trust_remote_code = False
    cfg.model.local_files_only = True
    cfg.model.attn_implementation = "eager"
    cfg.model.prompt_format = str(protocol["model"]["prompt_format"])
    cfg.generation.max_new_tokens = int(
        protocol["data"]["max_new_tokens"]
    )
    cfg.generation.temperature = 0.0
    cfg.generation.do_sample = False
    cfg.generation.stop_on_eos = bool(
        protocol["data"]["stop_on_eos"]
    )
    cfg.cache.total_budget = int(protocol["cache"]["total_budget"])
    cfg.cache.sink_size = int(protocol["cache"]["sink_size"])
    cfg.cache.recent_size = int(protocol["cache"]["recent_size"])
    cfg.cache.selected_core_budget = int(
        protocol["cache"]["selected_core_budget"]
    )
    cfg.selectors.observation_window = 32
    cfg.selectors.snapkv_pooling_kernel = 63
    cfg.selectors.snapkv_pooling = "max"
    cfg.selectors.ridge_lambda = 1.0e-3
    cfg.selectors.ridge_lambda_mode = "relative"
    cfg.selectors.shared_token_selection = True
    cfg.runtime.seed = int(protocol["numeric"]["seed"])
    cfg.runtime.deterministic = True
    cfg.runtime.prefill_chunk_size = int(
        protocol["numeric"]["prefill_chunk_size"]
    )
    cfg.runtime.resume = bool(protocol["runtime"]["resume"])
    cfg.runtime.fail_on_error = True
    cfg.runtime.max_prompt_tokens = int(
        protocol["data"]["max_prompt_tokens"]
    )
    cfg.runtime.run_id = str(protocol["runtime"]["run_id"])
    gov_indices, niah_offsets = stage_indices(protocol, stage)
    if niah_offsets != list(
        range(niah_offsets[0], niah_offsets[0] + len(niah_offsets))
    ):
        raise ValueError("NIAH offsets must be contiguous")
    cfg.tasks = {
        "gov_report": {
            "num_samples": len(gov_indices),
            "dataset_name": protocol["data"]["gov_report"][
                "dataset_name"
            ],
            "dataset_config": protocol["data"]["gov_report"][
                "dataset_config"
            ],
            "split": protocol["data"]["gov_report"]["split"],
            "sample_indices": gov_indices,
            "max_words": int(
                protocol["data"]["gov_report"]["max_words"]
            ),
        },
        "ruler_niah": {
            "num_samples": len(niah_offsets),
            "context_length": int(
                protocol["data"]["ruler_niah"]["context_length"]
            ),
            "sample_offset": niah_offsets[0],
        },
    }
    section = protocol["data"][stage]
    cfg.anchor_steps = [int(value) for value in section["anchors"]]
    cfg.horizons = [1]
    cfg.signal_lags = [1]
    cfg.strategies = []
    cfg.diagnostics.num_layers = 28
    cfg.diagnostics.heads_per_layer = 12
    cfg.diagnostics.layer_selection = "explicit"
    cfg.diagnostics.explicit_layers = list(range(28))
    cfg.diagnostics.explicit_heads = list(range(12))
    # This flag suppresses quadratic all-query recording and asks the backend
    # to retain only the preregistered current-step anchors. No Fisher oracle or
    # future information is read from the resulting ReferenceTrajectory.
    cfg.independent_fisher.enabled = True
    cfg.independent_fisher.anchors = list(cfg.anchor_steps)
    cfg.independent_fisher.segment_horizon = 1
    return cfg


def load_fp32_model(
    protocol: Mapping[str, Any], stage: str
) -> Tuple[Any, Dict[str, Any], List[Any], List[Dict[str, Any]]]:
    cfg = build_discovery_config(protocol, stage)
    samples, dataset_events = load_discovery_tasks(cfg)
    actual = [sample.sample_id for sample in samples]
    expected = expected_ids(protocol, stage)
    if actual != expected:
        raise RuntimeError(
            f"{stage} sequence isolation failed: {actual} != {expected}"
        )
    model = P0V2FP32TemporalModel.create(cfg)
    model_info = model.load()
    dequantization = dequantize_reference_model(model.runner.model)
    model_info.update(
        {
            "p0_v2_execution": "dequantized_float32",
            "p0_v2_anchor_storage": "float32",
            "p0_v2_quantized_kernel_reachable": False,
            "dequantization": dequantization,
        }
    )
    if count_quantized_modules(
        model.runner.model
    )["quantized_modules_total"] != 0:
        raise RuntimeError("quantized module remains reachable")
    return model, model_info, samples, dataset_events


def safe_id(value: str) -> str:
    return "".join(
        character
        if character.isalnum() or character in "._-"
        else "_"
        for character in value
    )


def common_metadata(
    sample: Any,
    anchor: int,
    layer: int,
    candidate: Any,
    split: str,
) -> Dict[str, Any]:
    return {
        "sample_id": sample.sample_id,
        "task": sample.task,
        "split": split,
        "anchor": int(anchor),
        "layer": int(layer),
        "boundary_layer": int(layer) + 1,
        "candidate_id": candidate.candidate_id,
        "candidate_source": candidate.source,
        "mask_hash": candidate.mask_hash,
        "candidate_seed": int(candidate.seed),
    }


def select_candidates(
    candidates: Sequence[Any], sources: Sequence[str]
) -> List[Any]:
    by_source = {candidate.source: candidate for candidate in candidates}
    missing = sorted(set(sources) - set(by_source))
    if missing:
        raise RuntimeError(f"candidate sources absent: {missing}")
    return [by_source[source] for source in sources]


def verify_anchor_fp32(anchor: Any) -> Dict[str, Any]:
    dtypes = sorted(
        {
            str(value.dtype).replace("torch.", "")
            for value in list(anchor.keys) + list(anchor.values)
        }
    )
    shapes_valid = bool(
        all(value.ndim == 4 for value in anchor.keys)
        and all(value.ndim == 4 for value in anchor.values)
    )
    return {
        "anchor_cache_dtypes_json": json.dumps(dtypes),
        "anchor_cache_all_fp32": dtypes == ["float32"],
        "anchor_cache_shapes_valid": shapes_valid,
    }


def candidate_registry_rows(
    sample: Any,
    anchor: int,
    candidates: Sequence[Any],
    registry: Mapping[str, Any],
    split: str,
) -> List[Dict[str, Any]]:
    return [
        {
            "sample_id": sample.sample_id,
            "task": sample.task,
            "split": split,
            "anchor": int(anchor),
            "candidate_id": candidate.candidate_id,
            "candidate_source": candidate.source,
            "candidate_seed": int(candidate.seed),
            "mask_hash": candidate.mask_hash,
            "active_budget": len(candidate.retained_positions),
            "core_budget": len(candidate.core_positions),
            "retained_positions_json": json.dumps(
                candidate.retained_positions, separators=(",", ":")
            ),
            "core_positions_json": json.dumps(
                candidate.core_positions, separators=(",", ":")
            ),
            "dedup_event_count": len(registry["dedup_events"]),
            "layer_shared_generation": bool(
                registry["shared_across_layers"]
            ),
            "gqa_shared_generation": bool(
                registry["shared_across_gqa_heads"]
            ),
        }
        for candidate in candidates
    ]


def theoretical_pulse(
    backend: Any,
    anchor: Any,
    base_record: Any,
    candidate: Any,
    layer: int,
    identity_floors: Sequence[float],
    common: Mapping[str, Any],
) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
    pulse32, rows32, tensors = layer_identity_and_injection(
        backend,
        anchor,
        base_record,
        candidate.retained_positions,
        layer,
        torch.float32,
    )
    _pulse64, rows64, _unused = layer_identity_and_injection(
        backend,
        anchor,
        base_record,
        candidate.retained_positions,
        layer,
        torch.float64,
    )
    rows: List[Dict[str, Any]] = []
    for arithmetic, generated in (
        ("float32", rows32),
        ("float64", rows64),
    ):
        conditioned = add_identity_conditioning(
            generated, identity_floors
        )
        for row in conditioned:
            rows.append(
                {
                    **common,
                    "arithmetic": arithmetic,
                    "retained_size": len(candidate.retained_positions),
                    "deleted_size": int(
                        len(anchor.position_maps[layer])
                        - len(candidate.retained_positions)
                    ),
                    **row,
                }
            )
    return pulse32.astype(np.float64), rows, tensors


def smoke_stage(
    protocol: Mapping[str, Any],
    config_path: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    started = time.perf_counter()
    split_audit = validate_split_isolation(protocol)
    model, model_info, samples, dataset_events = load_fp32_model(
        protocol, "smoke"
    )
    checks: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []
    try:
        for sample in samples:
            reference = model.generate_reference(
                sample.sample_id, sample.task, sample.prompt
            )
            anchor_step = int(protocol["data"]["smoke"]["anchors"][0])
            layer = int(protocol["data"]["smoke"]["layers"][0])
            boundary = layer + 1
            anchor = reference.anchors[anchor_step]
            candidates, registry = make_selector_candidates(
                model,
                reference,
                anchor_step,
                model.cfg.cache,
                str(protocol["runtime"]["run_id"]),
            )
            candidates = select_candidates(
                candidates,
                protocol["data"]["smoke"]["candidate_sources"],
            )
            base_logits, base_record, base_positions, base_dtypes = full_replay(
                model, reference, anchor_step
            )
            repeated_logits, _record2, _positions2, _dtypes2 = full_replay(
                model, reference, anchor_step
            )
            downstream = FixedBoundaryReadoutMap(
                model, anchor, base_record, boundary
            )
            downstream_base = downstream.baseline()
            anchor_checks = verify_anchor_fp32(anchor)
            repeat_error = float(
                np.max(np.abs(base_logits - repeated_logits))
            )
            base_map_metrics = vector_metrics(
                downstream_base, base_logits
            )
            for candidate in candidates:
                common = common_metadata(
                    sample,
                    anchor_step,
                    layer,
                    candidate,
                    "smoke",
                )
                u_theory, identity_rows, _tensors = theoretical_pulse(
                    model,
                    anchor,
                    base_record,
                    candidate,
                    layer,
                    protocol["numeric"]["identity_norm_floors"],
                    common,
                )
                physical_logits, physical_record, physical_positions, _ = (
                    physical_single_layer_replay(
                        model,
                        reference,
                        anchor_step,
                        candidate,
                        layer,
                    )
                )
                u_physical = (
                    physical_record.projected_attention_outputs[layer]
                    - base_record.projected_attention_outputs[layer]
                ).double().numpy()
                r_physical = (
                    physical_record.layer_outputs[layer]
                    - base_record.layer_outputs[layer]
                ).double().numpy()
                adjacent = AdjacentBoundaryMap(model, layer, base_record)
                adjacent_base, r_j1, adjacent_method = adjacent.jvp(
                    u_theory
                )
                manual_hook_logits, manual_record, _manual_positions, _ = (
                    manual_boundary_replay(
                        model,
                        reference,
                        anchor_step,
                        boundary,
                        base_record.residual_inputs[boundary]
                        .double()
                        .numpy()
                        + r_physical,
                    )
                )
                map_manual_logits = downstream.evaluate(r_physical)
                _jvp_base, jvp_physical, jvp_method = downstream.jvp(
                    r_physical
                )
                rng = np.random.default_rng(
                    int(candidate.seed) + int(layer)
                )
                cotangent = rng.standard_normal(
                    base_logits.shape
                ).astype(np.float32)
                cotangent /= max(
                    float(np.linalg.norm(cotangent)), 1.0e-30
                )
                vjp = downstream.vjp(cotangent)
                row = {
                    **common,
                    **anchor_checks,
                    "quantized_module_count": count_quantized_modules(
                        model.runner.model
                    )["quantized_modules_total"],
                    "base_runtime_dtypes_json": json.dumps(
                        base_dtypes, sort_keys=True
                    ),
                    "repeat_max_absolute_error": repeat_error,
                    "boundary_map_baseline_cosine": base_map_metrics[
                        "cosine"
                    ],
                    "boundary_map_baseline_relative_l2": base_map_metrics[
                        "relative_l2"
                    ],
                    "adjacent_method": adjacent_method,
                    "jvp_method": jvp_method,
                    "vjp_finite": bool(np.isfinite(vjp).all()),
                    "vjp_norm": float(np.linalg.norm(vjp)),
                    "identity_all_finite": bool(
                        all(item["finite"] for item in identity_rows)
                    ),
                    "retained_mass_min": min(
                        float(item["retained_mass"])
                        for item in identity_rows
                    ),
                    **prefixed_metrics(
                        "pulse", u_theory, u_physical
                    ),
                    **prefixed_metrics(
                        "adjacent",
                        r_j1,
                        r_physical,
                    ),
                    **prefixed_metrics(
                        "manual_hook_vs_physical",
                        manual_hook_logits - base_logits,
                        physical_logits - base_logits,
                    ),
                    **prefixed_metrics(
                        "pure_map_vs_manual_hook",
                        map_manual_logits - downstream_base,
                        manual_hook_logits - base_logits,
                    ),
                    **prefixed_metrics(
                        "jvp_vs_manual_hook",
                        jvp_physical,
                        manual_hook_logits - base_logits,
                    ),
                    "target_positions_match_candidate": (
                        list(physical_positions[layer].tolist())
                        == list(candidate.retained_positions)
                    ),
                    "non_target_position_maps_unchanged": all(
                        torch.equal(
                            physical_positions[index],
                            base_positions[index],
                        )
                        for index in range(
                            int(model.model_info["num_layers"])
                        )
                        if index != layer
                    ),
                    "boundary_input_override_exact": bool(
                        np.max(
                            np.abs(
                                manual_record.residual_inputs[boundary]
                                .double()
                                .numpy()
                                - (
                                    base_record.residual_inputs[boundary]
                                    .double()
                                    .numpy()
                                    + r_physical
                                )
                            )
                        )
                        <= 1.0e-6
                    ),
                    "adjacent_baseline_alignment_max_abs": float(
                        np.max(
                            np.abs(
                                adjacent_base
                                - base_record.layer_outputs[layer]
                                .double()
                                .numpy()
                            )
                        )
                    ),
                }
                rows.append(row)
            del reference
            gc.collect()
        frame = pd.DataFrame(rows)
        gate = protocol["gates"]["numeric"]
        checks_dict = {
            "row_count": len(frame) == len(samples) * 2,
            "dequantized_weights": bool(
                frame["quantized_module_count"].eq(0).all()
            ),
            "fp32_anchor_cache": bool(
                frame["anchor_cache_all_fp32"].all()
            ),
            "anchor_shapes": bool(
                frame["anchor_cache_shapes_valid"].all()
            ),
            "deterministic_full_replay": bool(
                frame["repeat_max_absolute_error"].max()
                <= float(gate["repeat_max_absolute_error_max"])
            ),
            "boundary_map_baseline": bool(
                frame["boundary_map_baseline_cosine"].min()
                >= float(gate["boundary_map_baseline_cosine_min"])
                and frame[
                    "boundary_map_baseline_relative_l2"
                ].max()
                <= float(
                    gate["boundary_map_baseline_relative_l2_max"]
                )
            ),
            "identity_finite": bool(frame["identity_all_finite"].all()),
            "retained_mass_valid": bool(
                frame["retained_mass_min"].min()
                > float(gate["retained_mass_min_strict"])
            ),
            "physical_mask_target_only": bool(
                frame["target_positions_match_candidate"].all()
                and frame["non_target_position_maps_unchanged"].all()
            ),
            "manual_boundary_exact": bool(
                frame["boundary_input_override_exact"].all()
            ),
            "manual_pure_map_match": bool(
                frame["pure_map_vs_manual_hook_cosine"].min() >= 0.999999
                and frame[
                    "pure_map_vs_manual_hook_relative_l2"
                ].max()
                <= 1.0e-4
            ),
            "autodiff_jvp_vjp_finite": bool(
                frame["jvp_vs_manual_hook_finite"].all()
                and frame["vjp_finite"].all()
            ),
            "pulse_sign": bool(frame["pulse_cosine"].min() > 0.99),
            "split_isolation": bool(
                all(split_audit["checks"].values())
            ),
        }
        passed = bool(all(checks_dict.values()))
        output_dir.mkdir(parents=True, exist_ok=True)
        atomic_frame(output_dir / "smoke_rows.parquet", frame)
        summary = {
            "stage": "smoke",
            "passed": passed,
            "checks": checks_dict,
            "row_count": len(frame),
            "model_info": model_info,
            "dataset_events": dataset_events,
            "split_audit": split_audit,
            "config_sha256": sha256_file(config_path),
            "wall_seconds": time.perf_counter() - started,
        }
        atomic_json(output_dir / "smoke_summary.json", summary)
        if not passed:
            raise RuntimeError(
                "P0-v2 smoke prerequisite failed; formal stages are blocked"
            )
        return summary
    finally:
        model.close()


def calibration_stage(
    protocol: Mapping[str, Any],
    config_path: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    started = time.perf_counter()
    split_audit = validate_split_isolation(protocol)
    smoke_path = output_dir / "smoke_summary.json"
    if not smoke_path.exists():
        raise RuntimeError("smoke summary is missing")
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    if not smoke.get("passed"):
        raise RuntimeError("smoke prerequisite did not pass")
    model, model_info, samples, dataset_events = load_fp32_model(
        protocol, "calibration"
    )
    rows: List[Dict[str, Any]] = []
    try:
        section = protocol["data"]["calibration"]
        grid = [
            float(value)
            for value in protocol["numeric"]["fd_relative_radius_grid"]
        ]
        for sample in samples:
            reference = model.generate_reference(
                sample.sample_id, sample.task, sample.prompt
            )
            previous_attention_core = None
            for anchor_step in [
                int(value) for value in section["anchors"]
            ]:
                candidates, registry = make_selector_candidates(
                    model,
                    reference,
                    anchor_step,
                    model.cfg.cache,
                    str(protocol["runtime"]["run_id"]),
                    previous_attention_core=previous_attention_core,
                )
                previous_attention_core = registry["attention_core"]
                selected = select_candidates(
                    candidates, section["candidate_sources"]
                )
                anchor = reference.anchors[anchor_step]
                base_logits, base_record, _positions, _dtypes = full_replay(
                    model, reference, anchor_step
                )
                for layer in [int(value) for value in section["layers"]]:
                    boundary = layer + 1
                    adjacent = AdjacentBoundaryMap(
                        model, layer, base_record
                    )
                    downstream = FixedBoundaryReadoutMap(
                        model, anchor, base_record, boundary
                    )
                    for candidate in selected:
                        common = common_metadata(
                            sample,
                            anchor_step,
                            layer,
                            candidate,
                            "calibration",
                        )
                        u_theory, _identity, _tensors = theoretical_pulse(
                            model,
                            anchor,
                            base_record,
                            candidate,
                            layer,
                            protocol["numeric"][
                                "identity_norm_floors"
                            ],
                            common,
                        )
                        (
                            _physical_logits,
                            physical_record,
                            _physical_positions,
                            _physical_dtypes,
                        ) = physical_single_layer_replay(
                            model,
                            reference,
                            anchor_step,
                            candidate,
                            layer,
                        )
                        r_physical = (
                            physical_record.layer_outputs[layer]
                            - base_record.layer_outputs[layer]
                        ).double().numpy()
                        _adjacent_base, r_j1, adjacent_method = (
                            adjacent.jvp(u_theory)
                        )
                        for direction_name, direction in (
                            ("physical_boundary", r_physical),
                            ("J1_theory", r_j1),
                        ):
                            jvp_base, derivative, jvp_method = (
                                downstream.jvp(direction)
                            )
                            for radius in grid:
                                fd = downstream.symmetric_fd(
                                    direction,
                                    radius,
                                    float(
                                        protocol["numeric"][
                                            "vector_norm_floor"
                                        ]
                                    ),
                                )
                                metrics = vector_metrics(
                                    derivative,
                                    fd["derivative"],
                                    norm_floor=float(
                                        protocol["numeric"][
                                            "vector_norm_floor"
                                        ]
                                    ),
                                    low_norm_threshold=float(
                                        protocol["metrics"][
                                            "low_norm_threshold"
                                        ]
                                    ),
                                )
                                rows.append(
                                    {
                                        **common,
                                        "direction_type": direction_name,
                                        "epsilon_relative": radius,
                                        "epsilon_absolute": fd[
                                            "epsilon_absolute"
                                        ],
                                        "base_input_norm": fd[
                                            "base_norm"
                                        ],
                                        "direction_norm": fd[
                                            "direction_norm"
                                        ],
                                        "fd_norm": fd["fd_norm"],
                                        "jvp_norm": float(
                                            np.linalg.norm(derivative)
                                        ),
                                        "jvp_method": jvp_method,
                                        "adjacent_method": adjacent_method,
                                        "jvp_base_alignment_cosine": (
                                            vector_metrics(
                                                jvp_base, base_logits
                                            )["cosine"]
                                        ),
                                        **{
                                            f"jvp_fd_{key}": value
                                            for key, value in metrics.items()
                                        },
                                    }
                                )
            del reference
            gc.collect()
        frame = pd.DataFrame(rows)
        by_radius = (
            frame.groupby("epsilon_relative", as_index=False)
            .agg(
                row_count=("jvp_fd_cosine", "size"),
                finite_rate=("jvp_fd_finite", "mean"),
                nonzero_fd_norm_rate=(
                    "fd_norm",
                    lambda value: float(
                        np.mean(
                            value.to_numpy(dtype=np.float64)
                            > float(
                                protocol["metrics"][
                                    "low_norm_threshold"
                                ]
                            )
                        )
                    ),
                ),
                median_cosine=("jvp_fd_cosine", "median"),
                median_relative_l2=(
                    "jvp_fd_relative_l2",
                    "median",
                ),
                median_symmetric_norm_ratio=(
                    "jvp_fd_symmetric_norm_ratio",
                    "median",
                ),
            )
            .sort_values("epsilon_relative", ascending=False)
        )
        rule = protocol["numeric"]["fd_selection_rule"]
        by_radius["eligible"] = (
            by_radius["finite_rate"].ge(
                float(rule["finite_rate_min"])
            )
            & by_radius["nonzero_fd_norm_rate"].ge(
                float(rule["nonzero_fd_norm_rate_min"])
            )
            & by_radius["median_cosine"].ge(
                float(rule["median_cosine_min"])
            )
            & by_radius["median_relative_l2"].le(
                float(rule["median_relative_l2_max"])
            )
            & by_radius["median_symmetric_norm_ratio"].ge(
                float(rule["median_symmetric_norm_ratio_min"])
            )
        )
        eligible = by_radius[by_radius["eligible"]].sort_values(
            ["median_relative_l2", "epsilon_relative"],
            ascending=[True, False],
            kind="stable",
        )
        selected_radius: Optional[float] = (
            float(eligible.iloc[0]["epsilon_relative"])
            if len(eligible)
            else None
        )
        passed = selected_radius is not None
        atomic_frame(output_dir / "calibration_rows.parquet", frame)
        by_radius.to_csv(
            output_dir / "calibration_radius_summary.csv", index=False
        )
        summary = {
            "stage": "calibration",
            "passed": passed,
            "selected_relative_radius": selected_radius,
            "selection_rule": rule,
            "row_count": len(frame),
            "direction_count": int(
                frame[
                    [
                        "sample_id",
                        "anchor",
                        "layer",
                        "candidate_id",
                        "direction_type",
                    ]
                ].drop_duplicates().shape[0]
            ),
            "radius_summary": by_radius.to_dict("records"),
            "model_info": model_info,
            "dataset_events": dataset_events,
            "split_audit": split_audit,
            "config_sha256_before_radius_freeze": sha256_file(
                config_path
            ),
            "wall_seconds": time.perf_counter() - started,
        }
        atomic_json(output_dir / "calibration_summary.json", summary)
        if not passed:
            raise RuntimeError(
                "no finite-difference radius passed the frozen rule"
            )
        return summary
    finally:
        model.close()


def run_evaluation_sequence(
    model: Any,
    sample: Any,
    protocol: Mapping[str, Any],
    sequence_dir: Path,
) -> Dict[str, List[Dict[str, Any]]]:
    started = time.perf_counter()
    tables: Dict[str, List[Dict[str, Any]]] = {
        name: [] for name in TABLE_NAMES
    }
    status = {
        "state": "running",
        "stage": "formal_evaluation",
        "sample_id": sample.sample_id,
        "task": sample.task,
        "split": "evaluation",
    }
    atomic_json(sequence_dir / "status.json", status)
    reference = model.generate_reference(
        sample.sample_id, sample.task, sample.prompt
    )
    section = protocol["data"]["evaluation"]
    anchors = [int(value) for value in section["anchors"]]
    layers = [int(value) for value in section["layers"]]
    missing = [
        anchor for anchor in anchors if anchor not in reference.anchors
    ]
    if missing:
        raise RuntimeError(f"missing evaluation anchors: {missing}")
    previous_attention_core = None
    for anchor_step in anchors:
        anchor_started = time.perf_counter()
        candidates, registry = make_selector_candidates(
            model,
            reference,
            anchor_step,
            model.cfg.cache,
            str(protocol["runtime"]["run_id"]),
            previous_attention_core=previous_attention_core,
        )
        previous_attention_core = registry["attention_core"]
        tables["candidate_registry"].extend(
            candidate_registry_rows(
                sample,
                anchor_step,
                candidates,
                registry,
                "evaluation",
            )
        )
        anchor = reference.anchors[anchor_step]
        base_logits, base_record, base_positions, base_dtypes = full_replay(
            model, reference, anchor_step
        )
        repeated_logits, _repeat_record, _repeat_positions, _repeat_dtypes = (
            full_replay(model, reference, anchor_step)
        )
        repeat_error = float(
            np.max(np.abs(base_logits - repeated_logits))
        )
        probability = stable_softmax(base_logits)
        anchor_checks = verify_anchor_fp32(anchor)
        for layer in layers:
            unit_started = time.perf_counter()
            boundary = layer + 1
            adjacent = AdjacentBoundaryMap(model, layer, base_record)
            downstream = FixedBoundaryReadoutMap(
                model, anchor, base_record, boundary
            )
            downstream_fingerprint_before = downstream.cache_fingerprint()
            downstream_base = downstream.baseline()
            downstream_base_metrics = vector_metrics(
                downstream_base, base_logits
            )
            adjacent_base = adjacent.baseline()
            adjacent_base_metrics = vector_metrics(
                adjacent_base,
                base_record.layer_outputs[layer].double().numpy(),
            )
            unit_upstream_max = 0.0
            unit_position_maps_valid = True
            unit_jvp_methods = set()
            for candidate in candidates:
                common = common_metadata(
                    sample,
                    anchor_step,
                    layer,
                    candidate,
                    "evaluation",
                )
                u_theory, identity_rows, tensors = theoretical_pulse(
                    model,
                    anchor,
                    base_record,
                    candidate,
                    layer,
                    protocol["numeric"]["identity_norm_floors"],
                    common,
                )
                tables["identity_rows"].extend(identity_rows)
                (
                    physical_logits,
                    physical_record,
                    physical_positions,
                    physical_dtypes,
                ) = physical_single_layer_replay(
                    model,
                    reference,
                    anchor_step,
                    candidate,
                    layer,
                )
                u_physical = (
                    physical_record.projected_attention_outputs[layer]
                    - base_record.projected_attention_outputs[layer]
                ).double().numpy()
                r_physical = (
                    physical_record.layer_outputs[layer]
                    - base_record.layer_outputs[layer]
                ).double().numpy()
                _adjacent_output, r_j1, adjacent_method = adjacent.jvp(
                    u_theory
                )
                r_local_exact = (
                    adjacent.evaluate(u_theory) - adjacent_base
                )
                manual_physical_logits = downstream.evaluate(r_physical)
                manual_j1_logits = downstream.evaluate(r_j1)
                manual_physical_delta = (
                    manual_physical_logits - downstream_base
                )
                manual_j1_delta = manual_j1_logits - downstream_base
                _base_phys, jvp_physical, jvp_phys_method = downstream.jvp(
                    r_physical
                )
                _base_j1, jvp_j1, jvp_j1_method = downstream.jvp(r_j1)
                unit_jvp_methods.update(
                    [jvp_phys_method, jvp_j1_method]
                )
                physical_delta = physical_logits - base_logits
                exact = exact_kl(base_logits, physical_logits)
                fisher = 0.5 * fisher_variance(probability, jvp_j1)
                midpoint_probability = stable_softmax(
                    base_logits + 0.5 * physical_delta
                )
                midpoint_oracle = 0.5 * fisher_variance(
                    midpoint_probability, physical_delta
                )
                direct_score = float(np.dot(u_theory, u_theory))
                local_score = float(np.dot(r_j1, r_j1))
                retained_mass = [
                    float(item["retained_mass"])
                    for item in identity_rows
                    if item["arithmetic"] == "float64"
                ]
                upstream_max = 0.0
                for upstream in range(layer):
                    upstream_max = max(
                        upstream_max,
                        float(
                            torch.max(
                                torch.abs(
                                    physical_record.layer_outputs[upstream]
                                    - base_record.layer_outputs[upstream]
                                )
                            )
                        ),
                    )
                unit_upstream_max = max(unit_upstream_max, upstream_max)
                target_positions_valid = (
                    list(physical_positions[layer].tolist())
                    == list(candidate.retained_positions)
                )
                non_target_maps_valid = all(
                    torch.equal(
                        physical_positions[index],
                        base_positions[index],
                    )
                    for index in range(
                        int(model.model_info["num_layers"])
                    )
                    if index != layer
                )
                unit_position_maps_valid = (
                    unit_position_maps_valid
                    and target_positions_valid
                    and non_target_maps_valid
                )
                row = {
                    **common,
                    "query_token_id": int(anchor.query_token_id),
                    "logical_position": int(anchor.logical_length - 1),
                    "active_budget": len(candidate.retained_positions),
                    "core_budget": len(candidate.core_positions),
                    "retained_mass_min": min(retained_mass),
                    "retained_mass_mean": float(np.mean(retained_mass)),
                    "kappa_mass_max": 1.0 / min(retained_mass),
                    "cancellation_sensitive_head_count": int(
                        sum(
                            bool(item["cancellation_sensitive"])
                            for item in identity_rows
                            if item["arithmetic"] == "float64"
                        )
                    ),
                    "direct_score": direct_score,
                    "local_score": local_score,
                    "fisher_score": fisher,
                    "midpoint_fisher_oracle": midpoint_oracle,
                    "exact_kl": exact,
                    "physical_delta_norm": float(
                        np.linalg.norm(physical_delta)
                    ),
                    "manual_physical_delta_norm": float(
                        np.linalg.norm(manual_physical_delta)
                    ),
                    "manual_j1_delta_norm": float(
                        np.linalg.norm(manual_j1_delta)
                    ),
                    "jvp_physical_norm": float(
                        np.linalg.norm(jvp_physical)
                    ),
                    "jvp_j1_norm": float(np.linalg.norm(jvp_j1)),
                    "adjacent_method": adjacent_method,
                    "jvp_physical_method": jvp_phys_method,
                    "jvp_j1_method": jvp_j1_method,
                    "physical_runtime_dtypes_json": json.dumps(
                        physical_dtypes, sort_keys=True
                    ),
                    "target_positions_match_candidate": target_positions_valid,
                    "non_target_position_maps_unchanged": (
                        non_target_maps_valid
                    ),
                    "upstream_layer_output_max_abs": upstream_max,
                    **prefixed_metrics(
                        "pulse_theory_vs_physical",
                        u_theory,
                        u_physical,
                    ),
                    **prefixed_metrics(
                        "adjacent_j1_vs_physical",
                        r_j1,
                        r_physical,
                    ),
                    **prefixed_metrics(
                        "adjacent_exact_vs_physical",
                        r_local_exact,
                        r_physical,
                    ),
                    **prefixed_metrics(
                        "boundary_manual_vs_physical",
                        manual_physical_delta,
                        physical_delta,
                    ),
                    **prefixed_metrics(
                        "downstream_jvp_physical_vs_manual",
                        jvp_physical,
                        manual_physical_delta,
                    ),
                    **prefixed_metrics(
                        "downstream_jvp_j1_vs_manual",
                        jvp_j1,
                        manual_j1_delta,
                    ),
                    **prefixed_metrics(
                        "downstream_jvp_physical_vs_physical",
                        jvp_physical,
                        physical_delta,
                    ),
                    **prefixed_metrics(
                        "end_to_end_j1_vs_physical",
                        jvp_j1,
                        physical_delta,
                    ),
                }
                tables["response_rows"].append(row)
            fingerprint_after = downstream.cache_fingerprint()
            tables["unit_audit"].append(
                {
                    "sample_id": sample.sample_id,
                    "task": sample.task,
                    "split": "evaluation",
                    "anchor": anchor_step,
                    "layer": layer,
                    "boundary_layer": boundary,
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
                    "repeat_max_absolute_error": repeat_error,
                    "boundary_map_baseline_cosine": (
                        downstream_base_metrics["cosine"]
                    ),
                    "boundary_map_baseline_relative_l2": (
                        downstream_base_metrics["relative_l2"]
                    ),
                    "adjacent_map_baseline_cosine": (
                        adjacent_base_metrics["cosine"]
                    ),
                    "adjacent_map_baseline_relative_l2": (
                        adjacent_base_metrics["relative_l2"]
                    ),
                    "upstream_layer_output_max_abs": unit_upstream_max,
                    "position_maps_valid": unit_position_maps_valid,
                    "cache_fingerprint_invariant": (
                        downstream_fingerprint_before == fingerprint_after
                    ),
                    "jvp_methods_json": json.dumps(
                        sorted(unit_jvp_methods)
                    ),
                    "base_runtime_dtypes_json": json.dumps(
                        base_dtypes, sort_keys=True
                    ),
                    **anchor_checks,
                    "unit_wall_seconds": time.perf_counter()
                    - unit_started,
                }
            )
        print(
            json.dumps(
                {
                    "event": "anchor_complete",
                    "sample_id": sample.sample_id,
                    "anchor": anchor_step,
                    "rows": len(candidates) * len(layers),
                    "wall_seconds": time.perf_counter() - anchor_started,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    for name in TABLE_NAMES:
        atomic_frame(sequence_dir / f"{name}.parquet", pd.DataFrame(tables[name]))
    status.update(
        {
            "state": "complete",
            "wall_seconds": time.perf_counter() - started,
            "prompt_length": reference.prompt_length,
            "prompt_truncated": reference.prompt_truncated,
            "anchors": anchors,
            "layers": layers,
            "candidate_rows": len(tables["response_rows"]),
        }
    )
    atomic_json(sequence_dir / "status.json", status)
    return tables


def load_sequence_tables(sequence_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    return {
        name: pd.read_parquet(
            sequence_dir / f"{name}.parquet"
        ).to_dict("records")
        for name in TABLE_NAMES
    }


def git_state() -> Dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {
        "commit": commit,
        "worktree_dirty": bool(status.strip()),
    }


def evaluation_stage(
    protocol: Mapping[str, Any],
    config_path: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    started = time.perf_counter()
    split_audit = validate_split_isolation(protocol)
    calibration_path = output_dir / "calibration_summary.json"
    if not calibration_path.exists():
        raise RuntimeError("calibration summary is missing")
    calibration = json.loads(
        calibration_path.read_text(encoding="utf-8")
    )
    if not calibration.get("passed"):
        raise RuntimeError("calibration prerequisite did not pass")
    selected = protocol["numeric"]["fd_selected_relative_radius"]
    if selected is None:
        raise RuntimeError(
            "fd_selected_relative_radius must be frozen before evaluation"
        )
    if not np.isclose(
        float(selected),
        float(calibration["selected_relative_radius"]),
        rtol=0.0,
        atol=0.0,
    ):
        raise RuntimeError(
            "frozen radius differs from mechanical calibration selection"
        )
    config_hash_at_evaluation_start = sha256_file(config_path)
    freeze = {
        "evaluation_started_at_unix": time.time(),
        "config_sha256": config_hash_at_evaluation_start,
        "selected_relative_radius": float(selected),
        "calibration_summary_sha256": sha256_file(calibration_path),
        "split_audit": split_audit,
        "git": git_state(),
    }
    atomic_json(output_dir / "evaluation_freeze.json", freeze)
    model, model_info, samples, dataset_events = load_fp32_model(
        protocol, "evaluation"
    )
    status = {
        "stage": "formal_evaluation",
        "state": "running",
        "completed_sequences": [],
        "config_sha256": config_hash_at_evaluation_start,
        "selected_relative_radius": float(selected),
        "errors": [],
    }
    atomic_json(output_dir / "evaluation_status.json", status)
    all_tables: Dict[str, List[Dict[str, Any]]] = {
        name: [] for name in TABLE_NAMES
    }
    try:
        for sample in samples:
            sequence_dir = (
                output_dir
                / "checkpoints"
                / safe_id(sample.sample_id)
            )
            sequence_status_path = sequence_dir / "status.json"
            complete = False
            if (
                bool(protocol["runtime"]["resume"])
                and sequence_status_path.exists()
            ):
                previous = json.loads(
                    sequence_status_path.read_text(encoding="utf-8")
                )
                complete = (
                    previous.get("state") == "complete"
                    and all(
                        (sequence_dir / f"{name}.parquet").exists()
                        for name in TABLE_NAMES
                    )
                )
            if complete:
                tables = load_sequence_tables(sequence_dir)
                event = "sequence_resumed"
            else:
                tables = run_evaluation_sequence(
                    model, sample, protocol, sequence_dir
                )
                event = "sequence_complete"
            for name in TABLE_NAMES:
                all_tables[name].extend(tables[name])
            status["completed_sequences"].append(sample.sample_id)
            atomic_json(output_dir / "evaluation_status.json", status)
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
            atomic_frame(output_dir / f"{name}.parquet", frame)
        if sha256_file(config_path) != config_hash_at_evaluation_start:
            raise RuntimeError("config changed during formal evaluation")
        status.update(
            {
                "state": "complete",
                "wall_seconds": time.perf_counter() - started,
            }
        )
        atomic_json(output_dir / "evaluation_status.json", status)
        metadata = {
            "stage": "formal_evaluation",
            "completed": True,
            "row_counts": {
                name: len(frame) for name, frame in frames.items()
            },
            "sequence_ids": [
                sample.sample_id for sample in samples
            ],
            "model_info": model_info,
            "dataset_events": dataset_events,
            "split_audit": split_audit,
            "config_sha256": config_hash_at_evaluation_start,
            "selected_relative_radius": float(selected),
            "runtime": {
                "wall_seconds": time.perf_counter() - started,
                "peak_rss_bytes": int(
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                ),
            },
        }
        atomic_json(output_dir / "evaluation_metadata.json", metadata)
        return metadata
    except Exception as error:
        status["state"] = "failed"
        status["errors"].append(
            {
                "type": type(error).__name__,
                "message": str(error),
            }
        )
        atomic_json(output_dir / "evaluation_status.json", status)
        raise
    finally:
        model.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/frozen/p0_v2_config.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "experiments/p0_v2_fixed_boundary/results",
    )
    parser.add_argument(
        "--stage",
        choices=("dry-run", "smoke", "calibration", "evaluation"),
        required=True,
    )
    args = parser.parse_args()
    protocol = load_protocol(args.config.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "dry-run":
        result = {
            "config_sha256": sha256_file(args.config.resolve()),
            "split_audit": validate_split_isolation(protocol),
            "expected": {
                stage: expected_ids(protocol, stage)
                for stage in ("smoke", "calibration", "evaluation")
            },
        }
    elif args.stage == "smoke":
        result = smoke_stage(
            protocol, args.config.resolve(), args.output_dir.resolve()
        )
    elif args.stage == "calibration":
        result = calibration_stage(
            protocol, args.config.resolve(), args.output_dir.resolve()
        )
    else:
        result = evaluation_stage(
            protocol, args.config.resolve(), args.output_dir.resolve()
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

