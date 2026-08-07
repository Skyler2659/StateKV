#!/usr/bin/env python3
"""Run P1 state-conditioned fixed-boundary risk closure."""
from __future__ import annotations

import argparse
import gc
import json
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import yaml


ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
P0_DIR = ROOT / "experiments/p0_v2_fixed_boundary/scripts"
PREDICTIVE_DIR = ROOT / "experiments/predictive_closure/scripts"
for value in (ROOT, ROOT / "benchmarks/torch", P0_DIR, PREDICTIVE_DIR, SCRIPT_DIR):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from kvbench.temporal.config import DiscoveryConfig
from kvbench.temporal.tasks import load_discovery_tasks
from mlx_predictive_core import make_selector_candidates
from precision_diagnostic import (
    count_quantized_modules,
    dequantize_reference_model,
)
from p0_v2_core import (
    AdjacentBoundaryMap,
    FixedBoundaryReadoutMap,
    P0V2FP32TemporalModel,
    full_replay,
    physical_single_layer_replay,
)
from run_p0_v2 import (
    candidate_registry_rows,
    common_metadata,
    select_candidates,
    theoretical_pulse,
    verify_anchor_fp32,
)

from p1_core import (
    HistoryTrajectoryGenerator,
    atomic_frame,
    atomic_json,
    downstream_jvp_at,
    euclidean_cosine,
    exact_kl,
    fisher_cosine,
    history_state_key,
    prefixed_metrics,
    required_reference_anchors,
    select_fd_radius,
    sha256_file,
    stable_softmax,
    state_action_scores,
    validate_split_isolation,
    vector_metrics,
)


TABLE_NAMES = (
    "response_rows",
    "identity_rows",
    "candidate_registry",
    "state_registry",
    "unit_audit",
    "sequence_vector_metrics",
)


def load_protocol(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        protocol = yaml.safe_load(handle)
    required = {
        "schema_version",
        "experiment",
        "scope",
        "source_p0_v2",
        "model",
        "data",
        "cache",
        "history_conditions",
        "candidates",
        "numeric",
        "metrics",
        "gates",
        "outcomes",
        "diagnostics",
        "runtime",
    }
    missing = sorted(required - set(protocol))
    if missing:
        raise ValueError(f"P1 config missing fields: {missing}")
    expected_prohibited = {
        "predicted_history_state",
        "temporal_transition_fit",
        "E2",
        "E2_J1",
        "multi_step_horizon_risk",
        "future_query_prediction",
        "future_attention_oracle",
        "refresh_policy_evaluation",
        "controller",
        "free_generation",
        "joint_current_multilayer_mask",
        "multilayer_response_sum",
        "low_rank_Q",
        "online_deployment",
        "post_formal_threshold_tuning",
    }
    if set(protocol["scope"]["prohibited"]) != expected_prohibited:
        raise ValueError("frozen P1 prohibited-scope list changed")
    return protocol


def history_ids(protocol: Mapping[str, Any]) -> List[str]:
    return sorted(
        key
        for key, value in protocol["history_conditions"].items()
        if isinstance(value, Mapping) and "history_length" in value
    )


def stage_indices(
    protocol: Mapping[str, Any], stage: str
) -> Tuple[List[int], List[int]]:
    section = protocol["data"][stage]
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
    cfg.generation.max_new_tokens = int(protocol["data"]["max_new_tokens"])
    cfg.generation.temperature = 0.0
    cfg.generation.do_sample = False
    cfg.generation.stop_on_eos = bool(protocol["data"]["stop_on_eos"])
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
            "dataset_name": protocol["data"]["gov_report"]["dataset_name"],
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
    targets = [
        int(value)
        for value in protocol["data"][stage]["target_anchors"]
    ]
    cfg.anchor_steps = list(
        required_reference_anchors(
            targets, protocol["history_conditions"]
        )
    )
    cfg.horizons = [1]
    cfg.signal_lags = [1]
    cfg.strategies = []
    cfg.diagnostics.num_layers = 28
    cfg.diagnostics.heads_per_layer = 12
    cfg.diagnostics.layer_selection = "explicit"
    cfg.diagnostics.explicit_layers = list(range(28))
    cfg.diagnostics.explicit_heads = list(range(12))
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
            "p1_execution": "dequantized_float32",
            "p1_anchor_storage": "float32",
            "p1_quantized_kernel_reachable": False,
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


def p0_regression_stage(
    protocol: Mapping[str, Any], output_dir: Path
) -> Dict[str, Any]:
    started = time.perf_counter()
    source = protocol["source_p0_v2"]
    response_path = (
        ROOT / "experiments/p0_v2_fixed_boundary/results/response_rows.parquet"
    )
    manifest_path = (
        ROOT / "experiments/p0_v2_fixed_boundary/P0_V2_MANIFEST.yaml"
    )
    manifest = yaml.safe_load(
        manifest_path.read_text(encoding="utf-8")
    )
    response_rows = len(pd.read_parquet(response_path))
    checks = {
        "p0_outcome_is_A": str(manifest["outcome"]) == "A",
        "response_row_count_matches": response_rows
        == int(source["expected_response_rows"]),
    }
    result = {
        "stage": "p0_regression",
        "passed": all(checks.values()),
        "checks": checks,
        "response_row_count": response_rows,
        "wall_seconds": time.perf_counter() - started,
    }
    atomic_json(output_dir / "p0_regression_summary.json", result)
    if not result["passed"]:
        raise RuntimeError(f"P0 regression failed: {checks}")
    return result


def _state_delta(
    observation: Any, base_record: Any, boundary: int
) -> np.ndarray:
    return (
        observation.record.residual_inputs[int(boundary)]
        - base_record.residual_inputs[int(boundary)]
    ).double().numpy()


def _history_bundle(
    model: Any,
    reference: Any,
    protocol: Mapping[str, Any],
    target: int,
    base_logits: np.ndarray,
    base_record: Any,
    base_positions: Mapping[int, Any],
) -> Dict[str, Any]:
    generator = HistoryTrajectoryGenerator(model, reference, protocol)
    return {
        history_id: generator.generate(
            target,
            history_id,
            base_logits,
            base_record,
            base_positions,
        )
        for history_id in history_ids(protocol)
    }


def _candidate_context(
    model: Any,
    reference: Any,
    target: int,
    protocol: Mapping[str, Any],
    previous_attention_core: Sequence[int] | None,
) -> Tuple[List[Any], Mapping[str, Any]]:
    return make_selector_candidates(
        model,
        reference,
        target,
        model.cfg.cache,
        str(protocol["runtime"]["run_id"]),
        previous_attention_core=previous_attention_core,
    )


def smoke_stage(
    protocol: Mapping[str, Any],
    config_path: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    regression_path = output_dir / "p0_regression_summary.json"
    if not regression_path.exists():
        raise RuntimeError("run p0-regression before smoke")
    if not json.loads(regression_path.read_text())["passed"]:
        raise RuntimeError("P0 regression prerequisite failed")
    started = time.perf_counter()
    model, model_info, samples, dataset_events = load_fp32_model(
        protocol, "smoke"
    )
    rows: List[Dict[str, Any]] = []
    state_rows: List[Dict[str, Any]] = []
    try:
        section = protocol["data"]["smoke"]
        target = int(section["target_anchors"][0])
        layer = int(section["layers"][0])
        sources = list(section["candidate_sources"])
        for sample in samples:
            reference = model.generate_reference(
                sample.sample_id, sample.task, sample.prompt
            )
            base_logits, base_record, base_positions, _dtypes = full_replay(
                model, reference, target
            )
            anchor = reference.anchors[target]
            candidates_all, _registry = _candidate_context(
                model, reference, target, protocol, None
            )
            candidates = select_candidates(candidates_all, sources)
            histories = _history_bundle(
                model,
                reference,
                protocol,
                target,
                base_logits,
                base_record,
                base_positions,
            )
            boundary = layer + 1
            adjacent = AdjacentBoundaryMap(model, layer, base_record)
            downstream = FixedBoundaryReadoutMap(
                model, anchor, base_record, boundary
            )
            downstream_base = downstream.baseline()
            state_hashes: Dict[str, str] = {}
            state_vectors: Dict[str, np.ndarray] = {}
            state_jvps: Dict[str, np.ndarray] = {}
            for history_id, observation in histories.items():
                delta = _state_delta(observation, base_record, boundary)
                state_hash = history_state_key(
                    sample.sample_id,
                    target,
                    boundary,
                    history_id,
                    delta,
                )
                _output, state_jvp, _method = downstream.jvp(delta)
                state_hashes[history_id] = state_hash
                state_vectors[history_id] = delta
                state_jvps[history_id] = state_jvp
                state_rows.append(
                    {
                        "sample_id": sample.sample_id,
                        "task": sample.task,
                        "history_id": history_id,
                        "state_hash": state_hash,
                        "state_norm": float(np.linalg.norm(delta)),
                        "finite": bool(np.isfinite(delta).all()),
                        "query_position": int(observation.record.query_position),
                        "target_query_position": int(
                            reference.query_records[target].query_position
                        ),
                        "physical_history_kl": exact_kl(
                            base_logits, observation.logits
                        ),
                    }
                )
            for candidate in candidates:
                common = common_metadata(
                    sample, target, layer, candidate, "smoke"
                )
                action_u, _identity, _tensors = theoretical_pulse(
                    model,
                    anchor,
                    base_record,
                    candidate,
                    layer,
                    protocol["numeric"]["identity_norm_floors"],
                    common,
                )
                _out, action_r, _method = adjacent.jvp(action_u)
                _out, action_jvp, _method = downstream.jvp(action_r)
                for history_id in history_ids(protocol):
                    delta = state_vectors[history_id]
                    combined = delta + action_r
                    manual = downstream.evaluate(combined) - downstream_base
                    _out, predicted, _method = downstream.jvp(combined)
                    scores = state_action_scores(
                        stable_softmax(base_logits),
                        state_jvps[history_id],
                        action_jvp,
                    )
                    rows.append(
                        {
                            **common,
                            "history_id": history_id,
                            "state_hash": state_hashes[history_id],
                            "state_norm": float(np.linalg.norm(delta)),
                            "action_norm": float(np.linalg.norm(action_r)),
                            **scores,
                            **prefixed_metrics(
                                "combined_jvp_vs_manual",
                                predicted,
                                manual,
                            ),
                            "controlled_exact_kl": exact_kl(
                                base_logits, base_logits + manual
                            ),
                        }
                    )
            del reference
            gc.collect()
        frame = pd.DataFrame(rows)
        states = pd.DataFrame(state_rows)
        atomic_frame(output_dir / "smoke_rows.parquet", frame)
        atomic_frame(output_dir / "smoke_state_rows.parquet", states)
        checks = {
            "finite": bool(
                frame["combined_jvp_vs_manual_finite"].all()
                and states["finite"].all()
            ),
            "h0_state_exact_zero": float(
                states.loc[
                    states["history_id"] == "H0", "state_norm"
                ].max()
            )
            == 0.0,
            "nonzero_stale_states": float(
                states.loc[
                    states["history_id"].isin(["H1", "H2", "H3"]),
                    "state_norm",
                ].min()
            )
            > float(protocol["numeric"]["low_norm_threshold"]),
            "target_positions_match": bool(
                (
                    states["query_position"]
                    == states["target_query_position"]
                ).all()
            ),
            "candidate_shared_state_hash": bool(
                (
                    frame.groupby(
                        ["sample_id", "history_id"]
                    )["state_hash"].nunique()
                    == 1
                ).all()
            ),
            "h1_h2_differ": bool(
                all(
                    group.loc[
                        group["history_id"] == "H1", "state_hash"
                    ].iloc[0]
                    != group.loc[
                        group["history_id"] == "H2", "state_hash"
                    ].iloc[0]
                    for _sample, group in states.groupby("sample_id")
                )
            ),
            "h2_h3_differ": bool(
                all(
                    group.loc[
                        group["history_id"] == "H2", "state_hash"
                    ].iloc[0]
                    != group.loc[
                        group["history_id"] == "H3", "state_hash"
                    ].iloc[0]
                    for _sample, group in states.groupby("sample_id")
                )
            ),
        }
        summary = {
            "stage": "smoke",
            "passed": all(checks.values()),
            "checks": checks,
            "response_row_count": len(frame),
            "state_row_count": len(states),
            "state_norm_by_history": states.groupby("history_id")[
                "state_norm"
            ].describe().to_dict(),
            "model_info": model_info,
            "dataset_events": dataset_events,
            "config_sha256": sha256_file(config_path),
            "wall_seconds": time.perf_counter() - started,
        }
        atomic_json(output_dir / "smoke_summary.json", summary)
        if not summary["passed"]:
            raise RuntimeError(f"P1 smoke failed: {checks}")
        return summary
    finally:
        model.close()


def calibration_stage(
    protocol: Mapping[str, Any],
    config_path: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    smoke_path = output_dir / "smoke_summary.json"
    if not smoke_path.exists() or not json.loads(
        smoke_path.read_text()
    )["passed"]:
        raise RuntimeError("passing smoke prerequisite is missing")
    started = time.perf_counter()
    model, model_info, samples, dataset_events = load_fp32_model(
        protocol, "calibration"
    )
    rows: List[Dict[str, Any]] = []
    state_rows: List[Dict[str, Any]] = []
    try:
        section = protocol["data"]["calibration"]
        target = int(section["target_anchors"][0])
        layers = [int(value) for value in section["layers"]]
        sources = list(section["candidate_sources"])
        for sample in samples:
            reference = model.generate_reference(
                sample.sample_id, sample.task, sample.prompt
            )
            base_logits, base_record, base_positions, _dtypes = full_replay(
                model, reference, target
            )
            anchor = reference.anchors[target]
            all_candidates, _registry = _candidate_context(
                model, reference, target, protocol, None
            )
            candidates = select_candidates(all_candidates, sources)
            histories = _history_bundle(
                model,
                reference,
                protocol,
                target,
                base_logits,
                base_record,
                base_positions,
            )
            for layer in layers:
                boundary = layer + 1
                adjacent = AdjacentBoundaryMap(model, layer, base_record)
                downstream = FixedBoundaryReadoutMap(
                    model, anchor, base_record, boundary
                )
                deltas = {}
                for history_id, observation in histories.items():
                    delta = _state_delta(
                        observation, base_record, boundary
                    )
                    deltas[history_id] = delta
                    state_rows.append(
                        {
                            "sample_id": sample.sample_id,
                            "task": sample.task,
                            "layer": layer,
                            "boundary_layer": boundary,
                            "history_id": history_id,
                            "state_hash": history_state_key(
                                sample.sample_id,
                                target,
                                boundary,
                                history_id,
                                delta,
                            ),
                            "state_norm": float(np.linalg.norm(delta)),
                            "finite": bool(np.isfinite(delta).all()),
                        }
                    )
                for candidate in candidates:
                    common = common_metadata(
                        sample,
                        target,
                        layer,
                        candidate,
                        "calibration",
                    )
                    action_u, _identity, _tensors = theoretical_pulse(
                        model,
                        anchor,
                        base_record,
                        candidate,
                        layer,
                        protocol["numeric"]["identity_norm_floors"],
                        common,
                    )
                    _out, action_r, _method = adjacent.jvp(action_u)
                    for history_id, delta in deltas.items():
                        direction = delta + action_r
                        _base, predicted, jvp_method = downstream.jvp(
                            direction
                        )
                        for radius in protocol["numeric"][
                            "fd_relative_radius_grid"
                        ]:
                            fd = downstream.symmetric_fd(
                                direction,
                                float(radius),
                                norm_floor=float(
                                    protocol["numeric"][
                                        "vector_norm_floor"
                                    ]
                                ),
                            )
                            metrics = vector_metrics(
                                predicted,
                                fd["derivative"],
                                norm_floor=float(
                                    protocol["numeric"][
                                        "vector_norm_floor"
                                    ]
                                ),
                                low_norm_threshold=float(
                                    protocol["numeric"][
                                        "low_norm_threshold"
                                    ]
                                ),
                            )
                            rows.append(
                                {
                                    **common,
                                    "history_id": history_id,
                                    "state_norm": float(
                                        np.linalg.norm(delta)
                                    ),
                                    "action_norm": float(
                                        np.linalg.norm(action_r)
                                    ),
                                    "combined_norm": float(
                                        np.linalg.norm(direction)
                                    ),
                                    "jvp_method": jvp_method,
                                    "epsilon_relative": float(radius),
                                    "epsilon_absolute": fd[
                                        "epsilon_absolute"
                                    ],
                                    "fd_norm": fd["fd_norm"],
                                    **metrics,
                                }
                            )
            del reference
            gc.collect()
            print(
                json.dumps(
                    {
                        "event": "calibration_sequence_complete",
                        "sample_id": sample.sample_id,
                        "rows": len(rows),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        frame = pd.DataFrame(rows)
        states = pd.DataFrame(state_rows)
        selected, radius_summary = select_fd_radius(
            frame, protocol["numeric"]["fd_selection_rule"]
        )
        state_checks = {
            "finite": bool(states["finite"].all()),
            "h0_zero": float(
                states.loc[
                    states["history_id"] == "H0", "state_norm"
                ].max()
            )
            == 0.0,
            "stale_nonzero": float(
                states.loc[
                    states["history_id"] != "H0", "state_norm"
                ].min()
            )
            > float(protocol["numeric"]["low_norm_threshold"]),
        }
        passed = selected is not None and all(state_checks.values())
        atomic_frame(output_dir / "calibration_rows.parquet", frame)
        atomic_frame(
            output_dir / "calibration_state_rows.parquet", states
        )
        radius_summary.to_csv(
            output_dir / "calibration_radius_summary.csv", index=False
        )
        summary = {
            "stage": "calibration",
            "passed": passed,
            "selected_relative_radius": selected,
            "selection_rule": protocol["numeric"]["fd_selection_rule"],
            "state_checks": state_checks,
            "row_count": len(frame),
            "direction_count": int(
                frame[
                    [
                        "sample_id",
                        "anchor",
                        "layer",
                        "candidate_id",
                        "history_id",
                    ]
                ].drop_duplicates().shape[0]
            ),
            "radius_summary": radius_summary.to_dict("records"),
            "model_info": model_info,
            "dataset_events": dataset_events,
            "config_sha256_before_radius_freeze": sha256_file(
                config_path
            ),
            "wall_seconds": time.perf_counter() - started,
        }
        atomic_json(output_dir / "calibration_summary.json", summary)
        if not passed:
            raise RuntimeError(
                "state validity or frozen finite-difference rule failed"
            )
        return summary
    finally:
        model.close()


def _new_vector_accumulator() -> Dict[str, Any]:
    return {
        "dot": 0.0,
        "predicted_square": 0.0,
        "truth_square": 0.0,
        "error_square": 0.0,
        "maximum_absolute_error": 0.0,
        "finite": True,
        "row_count": 0,
    }


def _update_vector_accumulator(
    accumulator: Dict[str, Any],
    predicted: np.ndarray,
    truth: np.ndarray,
) -> None:
    left = np.asarray(predicted, dtype=np.float64).reshape(-1)
    right = np.asarray(truth, dtype=np.float64).reshape(-1)
    difference = left - right
    accumulator["dot"] += float(np.dot(left, right))
    accumulator["predicted_square"] += float(np.dot(left, left))
    accumulator["truth_square"] += float(np.dot(right, right))
    accumulator["error_square"] += float(
        np.dot(difference, difference)
    )
    accumulator["maximum_absolute_error"] = max(
        accumulator["maximum_absolute_error"],
        float(np.max(np.abs(difference), initial=0.0)),
    )
    accumulator["finite"] = bool(
        accumulator["finite"]
        and np.isfinite(left).all()
        and np.isfinite(right).all()
    )
    accumulator["row_count"] += 1


def _finalize_vector_accumulator(
    accumulator: Mapping[str, Any],
    norm_floor: float,
) -> Dict[str, Any]:
    predicted_norm = float(accumulator["predicted_square"]) ** 0.5
    truth_norm = float(accumulator["truth_square"]) ** 0.5
    error_norm = float(accumulator["error_square"]) ** 0.5
    return {
        "cosine": float(accumulator["dot"])
        / max(predicted_norm * truth_norm, norm_floor**2),
        "relative_l2": error_norm / max(truth_norm, norm_floor),
        "symmetric_norm_ratio": 2.0
        * min(predicted_norm, truth_norm)
        / max(predicted_norm + truth_norm, norm_floor),
        "maximum_absolute_error": float(
            accumulator["maximum_absolute_error"]
        ),
        "predicted_norm": predicted_norm,
        "truth_norm": truth_norm,
        "error_norm": error_norm,
        "finite": bool(accumulator["finite"]),
        "row_count": int(accumulator["row_count"]),
    }


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
        "sample_id": sample.sample_id,
        "task": sample.task,
        "split": "evaluation",
    }
    atomic_json(sequence_dir / "status.json", status)
    reference = model.generate_reference(
        sample.sample_id, sample.task, sample.prompt
    )
    section = protocol["data"]["evaluation"]
    targets = [int(value) for value in section["target_anchors"]]
    layers = [int(value) for value in section["layers"]]
    missing = [
        anchor
        for anchor in required_reference_anchors(
            targets, protocol["history_conditions"]
        )
        if anchor not in reference.anchors
    ]
    if missing:
        raise RuntimeError(f"missing P1 reference anchors: {missing}")
    vector_accumulators = {
        key: _new_vector_accumulator()
        for key in ["all", *history_ids(protocol)]
    }
    previous_attention_core = None
    for target in targets:
        target_started = time.perf_counter()
        candidates, registry = _candidate_context(
            model,
            reference,
            target,
            protocol,
            previous_attention_core,
        )
        previous_attention_core = registry["attention_core"]
        tables["candidate_registry"].extend(
            candidate_registry_rows(
                sample,
                target,
                candidates,
                registry,
                "evaluation",
            )
        )
        base_logits, base_record, base_positions, base_dtypes = full_replay(
            model, reference, target
        )
        repeated_logits, _record, _positions, _dtypes = full_replay(
            model, reference, target
        )
        repeat_error = float(
            np.max(np.abs(base_logits - repeated_logits))
        )
        anchor = reference.anchors[target]
        probability = stable_softmax(base_logits)
        histories = _history_bundle(
            model,
            reference,
            protocol,
            target,
            base_logits,
            base_record,
            base_positions,
        )
        for layer in layers:
            unit_started = time.perf_counter()
            boundary = layer + 1
            adjacent = AdjacentBoundaryMap(model, layer, base_record)
            downstream = FixedBoundaryReadoutMap(
                model, anchor, base_record, boundary
            )
            fingerprint_before = downstream.cache_fingerprint()
            downstream_base = downstream.baseline()
            baseline_metrics = vector_metrics(
                downstream_base, base_logits
            )
            adjacent_base = adjacent.baseline()
            adjacent_metrics = vector_metrics(
                adjacent_base,
                base_record.layer_outputs[layer].double().numpy(),
            )
            state_cache: Dict[str, Dict[str, Any]] = {}
            for history_id, observation in histories.items():
                delta = _state_delta(
                    observation, base_record, boundary
                )
                state_key = history_state_key(
                    sample.sample_id,
                    target,
                    boundary,
                    history_id,
                    delta,
                )
                _base, state_jvp, state_method = downstream.jvp(delta)
                manual_state = (
                    downstream.evaluate(delta) - downstream_base
                )
                state_cache[history_id] = {
                    "delta": delta,
                    "state_key": state_key,
                    "jvp": state_jvp,
                    "manual": manual_state,
                    "method": state_method,
                }
                tables["state_registry"].append(
                    {
                        "sample_id": sample.sample_id,
                        "task": sample.task,
                        "split": "evaluation",
                        "anchor": target,
                        "layer": layer,
                        "boundary_layer": boundary,
                        "history_id": history_id,
                        "state_hash": state_key,
                        "history_length": observation.history_length,
                        "start_anchor": observation.start_anchor,
                        "refresh_count": observation.refresh_count,
                        "initial_source": observation.initial_source,
                        "refresh_source": observation.refresh_source,
                        "state_norm": float(np.linalg.norm(delta)),
                        "state_finite": bool(np.isfinite(delta).all()),
                        "state_jvp_norm": float(
                            np.linalg.norm(state_jvp)
                        ),
                        "manual_state_delta_norm": float(
                            np.linalg.norm(manual_state)
                        ),
                        "physical_history_delta_norm": float(
                            np.linalg.norm(
                                observation.logits - base_logits
                            )
                        ),
                        "physical_history_kl": exact_kl(
                            base_logits, observation.logits
                        ),
                        "query_token_id": int(anchor.query_token_id),
                        "query_position": int(
                            observation.record.query_position
                        ),
                        "reference_query_position": int(
                            reference.query_records[
                                target
                            ].query_position
                        ),
                        "replay_trace_json": json.dumps(
                            observation.replay_trace,
                            sort_keys=True,
                        ),
                        **prefixed_metrics(
                            "state_jvp_vs_manual",
                            state_jvp,
                            manual_state,
                        ),
                    }
                )
            for candidate in candidates:
                common = common_metadata(
                    sample,
                    target,
                    layer,
                    candidate,
                    "evaluation",
                )
                action_u, identity_rows, _tensors = theoretical_pulse(
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
                    target,
                    candidate,
                    layer,
                )
                physical_u = (
                    physical_record.projected_attention_outputs[layer]
                    - base_record.projected_attention_outputs[layer]
                ).double().numpy()
                physical_r = (
                    physical_record.layer_outputs[layer]
                    - base_record.layer_outputs[layer]
                ).double().numpy()
                _base, action_r, adjacent_method = adjacent.jvp(action_u)
                _base, action_jvp, action_method = downstream.jvp(
                    action_r
                )
                direct_score = float(np.dot(action_u, action_u))
                local_score = float(np.dot(action_r, action_r))
                physical_action_delta = physical_logits - base_logits
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
                for history_id in history_ids(protocol):
                    state = state_cache[history_id]
                    combined = state["delta"] + action_r
                    manual_combined = (
                        downstream.evaluate(combined) - downstream_base
                    )
                    _base, predicted_combined, combined_method = (
                        downstream.jvp(combined)
                    )
                    scores = state_action_scores(
                        probability, state["jvp"], action_jvp
                    )
                    controlled_exact = exact_kl(
                        base_logits, base_logits + manual_combined
                    )
                    midpoint_probability = stable_softmax(
                        base_logits + 0.5 * manual_combined
                    )
                    midpoint_oracle = 0.5 * (
                        np.dot(
                            midpoint_probability,
                            (
                                manual_combined
                                - np.dot(
                                    midpoint_probability,
                                    manual_combined,
                                )
                            )
                            ** 2,
                        )
                    )
                    row_metrics = vector_metrics(
                        predicted_combined, manual_combined
                    )
                    for key in ("all", history_id):
                        _update_vector_accumulator(
                            vector_accumulators[key],
                            predicted_combined,
                            manual_combined,
                        )
                    cross_ratio = scores["cross_fisher_score"] / max(
                        abs(scores["action_fisher_score"]),
                        float(
                            protocol["numeric"]["cross_ratio_epsilon"]
                        ),
                    )
                    tables["response_rows"].append(
                        {
                            **common,
                            "history_id": history_id,
                            "history_length": histories[
                                history_id
                            ].history_length,
                            "state_hash": state["state_key"],
                            "query_token_id": int(
                                anchor.query_token_id
                            ),
                            "logical_position": int(
                                anchor.logical_length - 1
                            ),
                            "state_norm": float(
                                np.linalg.norm(state["delta"])
                            ),
                            "action_u_norm": float(
                                np.linalg.norm(action_u)
                            ),
                            "action_r_norm": float(
                                np.linalg.norm(action_r)
                            ),
                            "combined_boundary_norm": float(
                                np.linalg.norm(combined)
                            ),
                            "state_action_euclidean_cosine": (
                                euclidean_cosine(
                                    state["delta"], action_r
                                )
                            ),
                            "state_action_fisher_cosine": fisher_cosine(
                                probability,
                                state["jvp"],
                                action_jvp,
                            ),
                            "cross_to_action_ratio": float(cross_ratio),
                            "direct_score": direct_score,
                            "local_score": local_score,
                            **scores,
                            "midpoint_fisher_oracle": float(
                                midpoint_oracle
                            ),
                            "controlled_exact_kl": controlled_exact,
                            "history_only_controlled_kl": exact_kl(
                                base_logits,
                                base_logits + state["manual"],
                            ),
                            "physical_action_kl": exact_kl(
                                base_logits, physical_logits
                            ),
                            "physical_history_kl": exact_kl(
                                base_logits,
                                histories[history_id].logits,
                            ),
                            "physical_action_delta_norm": float(
                                np.linalg.norm(
                                    physical_action_delta
                                )
                            ),
                            "manual_combined_delta_norm": float(
                                np.linalg.norm(manual_combined)
                            ),
                            "predicted_combined_delta_norm": float(
                                np.linalg.norm(predicted_combined)
                            ),
                            "adjacent_method": adjacent_method,
                            "action_jvp_method": action_method,
                            "combined_jvp_method": combined_method,
                            "target_positions_match_candidate": (
                                target_positions_valid
                            ),
                            "non_target_position_maps_unchanged": (
                                non_target_maps_valid
                            ),
                            "physical_runtime_dtypes_json": json.dumps(
                                physical_dtypes, sort_keys=True
                            ),
                            **{
                                f"combined_jvp_vs_manual_{key}": value
                                for key, value in row_metrics.items()
                            },
                            **prefixed_metrics(
                                "pulse_theory_vs_physical",
                                action_u,
                                physical_u,
                            ),
                            **prefixed_metrics(
                                "adjacent_j1_vs_physical",
                                action_r,
                                physical_r,
                            ),
                        }
                    )
            state_hash_valid = all(
                len(
                    {
                        row["state_hash"]
                        for row in tables["response_rows"]
                        if row["sample_id"] == sample.sample_id
                        and row["anchor"] == target
                        and row["layer"] == layer
                        and row["history_id"] == history_id
                    }
                )
                == 1
                for history_id in history_ids(protocol)
            )
            tables["unit_audit"].append(
                {
                    "sample_id": sample.sample_id,
                    "task": sample.task,
                    "split": "evaluation",
                    "anchor": target,
                    "layer": layer,
                    "boundary_layer": boundary,
                    "candidate_count": len(candidates),
                    "history_count": len(history_ids(protocol)),
                    "response_row_count": len(candidates)
                    * len(history_ids(protocol)),
                    "candidate_distinct_count": len(
                        {candidate.mask_hash for candidate in candidates}
                    ),
                    "repeat_max_absolute_error": repeat_error,
                    "boundary_map_baseline_cosine": baseline_metrics[
                        "cosine"
                    ],
                    "boundary_map_baseline_relative_l2": (
                        baseline_metrics["relative_l2"]
                    ),
                    "adjacent_map_baseline_cosine": adjacent_metrics[
                        "cosine"
                    ],
                    "adjacent_map_baseline_relative_l2": (
                        adjacent_metrics["relative_l2"]
                    ),
                    "candidate_shared_state_hash": state_hash_valid,
                    "cache_fingerprint_invariant": (
                        fingerprint_before
                        == downstream.cache_fingerprint()
                    ),
                    "base_runtime_dtypes_json": json.dumps(
                        base_dtypes, sort_keys=True
                    ),
                    **verify_anchor_fp32(anchor),
                    "unit_wall_seconds": time.perf_counter()
                    - unit_started,
                }
            )
        print(
            json.dumps(
                {
                    "event": "target_complete",
                    "sample_id": sample.sample_id,
                    "target": target,
                    "response_rows": len(candidates)
                    * len(layers)
                    * len(history_ids(protocol)),
                    "wall_seconds": time.perf_counter()
                    - target_started,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    norm_floor = float(protocol["numeric"]["vector_norm_floor"])
    for history_id, accumulator in vector_accumulators.items():
        tables["sequence_vector_metrics"].append(
            {
                "sample_id": sample.sample_id,
                "task": sample.task,
                "split": "evaluation",
                "history_id": history_id,
                **_finalize_vector_accumulator(
                    accumulator, norm_floor
                ),
            }
        )
    for name in TABLE_NAMES:
        atomic_frame(
            sequence_dir / f"{name}.parquet",
            pd.DataFrame(tables[name]),
        )
    status.update(
        {
            "state": "complete",
            "wall_seconds": time.perf_counter() - started,
            "prompt_length": reference.prompt_length,
            "prompt_truncated": reference.prompt_truncated,
            "targets": targets,
            "layers": layers,
            "response_rows": len(tables["response_rows"]),
        }
    )
    atomic_json(sequence_dir / "status.json", status)
    return tables


def load_sequence_tables(
    sequence_dir: Path,
) -> Dict[str, List[Dict[str, Any]]]:
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
        text=True,
        capture_output=True,
        check=False,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--short"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    ).stdout.strip()
    return {"commit": commit, "worktree_dirty": bool(dirty)}


def evaluation_stage(
    protocol: Mapping[str, Any],
    config_path: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    started = time.perf_counter()
    regression = json.loads(
        (output_dir / "p0_regression_summary.json").read_text()
    )
    smoke = json.loads((output_dir / "smoke_summary.json").read_text())
    calibration_path = output_dir / "calibration_summary.json"
    calibration = json.loads(calibration_path.read_text())
    if not (
        regression["passed"]
        and smoke["passed"]
        and calibration["passed"]
    ):
        raise RuntimeError("formal evaluation prerequisite failed")
    selected = protocol["numeric"]["fd_selected_relative_radius"]
    if selected is None or not np.isclose(
        float(selected),
        float(calibration["selected_relative_radius"]),
        rtol=0.0,
        atol=0.0,
    ):
        raise RuntimeError("calibration radius is not mechanically frozen")
    split_audit = validate_split_isolation(protocol)
    config_hash = sha256_file(config_path)
    atomic_json(
        output_dir / "evaluation_freeze.json",
        {
            "evaluation_started_at_unix": time.time(),
            "config_sha256": config_hash,
            "selected_relative_radius": float(selected),
            "calibration_summary_sha256": sha256_file(calibration_path),
            "split_audit": split_audit,
            "git": git_state(),
        },
    )
    model, model_info, samples, dataset_events = load_fp32_model(
        protocol, "evaluation"
    )
    status = {
        "stage": "formal_evaluation",
        "state": "running",
        "completed_sequences": [],
        "errors": [],
        "config_sha256": config_hash,
    }
    atomic_json(output_dir / "evaluation_status.json", status)
    all_tables: Dict[str, List[Dict[str, Any]]] = {
        name: [] for name in TABLE_NAMES
    }
    try:
        for sample in samples:
            sequence_dir = (
                output_dir / "checkpoints" / safe_id(sample.sample_id)
            )
            status_path = sequence_dir / "status.json"
            complete = False
            if bool(protocol["runtime"]["resume"]) and status_path.exists():
                previous = json.loads(status_path.read_text())
                complete = previous.get("state") == "complete" and all(
                    (sequence_dir / f"{name}.parquet").exists()
                    for name in TABLE_NAMES
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
                        "completed": len(status["completed_sequences"]),
                        "total": len(samples),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        frames = {
            name: pd.DataFrame(rows)
            for name, rows in all_tables.items()
        }
        expected_rows = int(
            protocol["data"]["evaluation"][
                "expected_candidate_history_rows"
            ]
        )
        if len(frames["response_rows"]) != expected_rows:
            raise RuntimeError(
                f"formal response rows {len(frames['response_rows'])} "
                f"!= {expected_rows}"
            )
        for name, frame in frames.items():
            atomic_frame(output_dir / f"{name}.parquet", frame)
        if sha256_file(config_path) != config_hash:
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
            "sequence_ids": [sample.sample_id for sample in samples],
            "model_info": model_info,
            "dataset_events": dataset_events,
            "split_audit": split_audit,
            "config_sha256": config_hash,
            "selected_relative_radius": float(selected),
            "runtime": {
                "wall_seconds": time.perf_counter() - started,
                "peak_rss_bytes": int(
                    resource.getrusage(
                        resource.RUSAGE_SELF
                    ).ru_maxrss
                ),
            },
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "torch": torch.__version__,
            },
        }
        atomic_json(output_dir / "evaluation_metadata.json", metadata)
        return metadata
    except Exception as error:
        status["state"] = "failed"
        status["errors"].append(
            {"type": type(error).__name__, "message": str(error)}
        )
        atomic_json(output_dir / "evaluation_status.json", status)
        raise
    finally:
        model.close()


def state_operating_point_diagnostic(
    protocol: Mapping[str, Any],
    config_path: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    """Run the preregistered diagnostic only after a failed main readout."""
    evaluation = json.loads(
        (output_dir / "evaluation_metadata.json").read_text()
    )
    if not evaluation.get("completed"):
        raise RuntimeError("formal evaluation is incomplete")
    if sha256_file(config_path) != str(evaluation["config_sha256"]):
        raise RuntimeError(
            "frozen config changed after formal evaluation"
        )
    main_vectors = pd.read_parquet(
        output_dir / "sequence_vector_metrics.parquet"
    )
    main_all = main_vectors[main_vectors["history_id"] == "all"]
    gate = protocol["gates"]["combined_readout"]
    main_failed = bool(
        float(main_all["cosine"].median())
        < float(gate["overall_sequence_first_cosine_min"])
        or float(main_all["relative_l2"].median())
        > float(
            gate["overall_sequence_first_relative_l2_max"]
        )
        or float(
            pd.read_parquet(
                output_dir / "response_rows.parquet"
            )["combined_jvp_vs_manual_cosine"]
            .ge(float(gate["row_cosine_threshold"]))
            .mean()
        )
        < float(gate["row_pass_fraction_min"])
    )
    if not main_failed:
        raise RuntimeError(
            "state-operating-point diagnostic is forbidden when Gate 2 passes"
        )
    started = time.perf_counter()
    model, model_info, samples, dataset_events = load_fp32_model(
        protocol, "evaluation"
    )
    rows: List[Dict[str, Any]] = []
    sequence_rows: List[Dict[str, Any]] = []
    try:
        section = protocol["data"]["evaluation"]
        targets = [int(value) for value in section["target_anchors"]]
        layers = [int(value) for value in section["layers"]]
        previous_by_sample: Dict[str, Sequence[int] | None] = {}
        for sample in samples:
            reference = model.generate_reference(
                sample.sample_id, sample.task, sample.prompt
            )
            accumulators = {
                key: _new_vector_accumulator()
                for key in ["all", *history_ids(protocol)]
            }
            previous_attention_core = previous_by_sample.get(
                sample.sample_id
            )
            for target in targets:
                candidates, registry = _candidate_context(
                    model,
                    reference,
                    target,
                    protocol,
                    previous_attention_core,
                )
                previous_attention_core = registry["attention_core"]
                base_logits, base_record, base_positions, _dtypes = (
                    full_replay(model, reference, target)
                )
                anchor = reference.anchors[target]
                histories = _history_bundle(
                    model,
                    reference,
                    protocol,
                    target,
                    base_logits,
                    base_record,
                    base_positions,
                )
                for layer in layers:
                    boundary = layer + 1
                    adjacent = AdjacentBoundaryMap(
                        model, layer, base_record
                    )
                    downstream = FixedBoundaryReadoutMap(
                        model, anchor, base_record, boundary
                    )
                    downstream_base = downstream.baseline()
                    state_cache = {}
                    for history_id, observation in histories.items():
                        delta = _state_delta(
                            observation, base_record, boundary
                        )
                        state_cache[history_id] = {
                            "delta": delta,
                            "manual_logits": downstream.evaluate(delta),
                        }
                    for candidate in candidates:
                        common = common_metadata(
                            sample,
                            target,
                            layer,
                            candidate,
                            "diagnostic",
                        )
                        action_u, _identity, _tensors = theoretical_pulse(
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
                        _base, action_r, _method = adjacent.jvp(
                            action_u
                        )
                        for history_id, state in state_cache.items():
                            combined_manual = (
                                downstream.evaluate(
                                    state["delta"] + action_r
                                )
                                - downstream_base
                            )
                            (
                                operating_logits,
                                incremental,
                                method,
                            ) = downstream_jvp_at(
                                downstream,
                                state["delta"],
                                action_r,
                            )
                            operating_point_error = float(
                                np.max(
                                    np.abs(
                                        operating_logits
                                        - state["manual_logits"]
                                    )
                                )
                            )
                            predicted = (
                                state["manual_logits"]
                                - downstream_base
                                + incremental
                            )
                            metrics = vector_metrics(
                                predicted, combined_manual
                            )
                            for key in ("all", history_id):
                                _update_vector_accumulator(
                                    accumulators[key],
                                    predicted,
                                    combined_manual,
                                )
                            rows.append(
                                {
                                    **common,
                                    "history_id": history_id,
                                    "state_norm": float(
                                        np.linalg.norm(state["delta"])
                                    ),
                                    "action_norm": float(
                                        np.linalg.norm(action_r)
                                    ),
                                    "jvp_method": method,
                                    "operating_point_output_max_error": (
                                        operating_point_error
                                    ),
                                    **{
                                        f"state_local_vs_manual_{key}": value
                                        for key, value in metrics.items()
                                    },
                                }
                            )
            previous_by_sample[sample.sample_id] = previous_attention_core
            for history_id, accumulator in accumulators.items():
                sequence_rows.append(
                    {
                        "sample_id": sample.sample_id,
                        "task": sample.task,
                        "history_id": history_id,
                        **_finalize_vector_accumulator(
                            accumulator,
                            float(
                                protocol["numeric"][
                                    "vector_norm_floor"
                                ]
                            ),
                        ),
                    }
                )
            print(
                json.dumps(
                    {
                        "event": "state_operating_point_sequence_complete",
                        "sample_id": sample.sample_id,
                        "rows": len(rows),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            del reference
            gc.collect()
        frame = pd.DataFrame(rows)
        sequence = pd.DataFrame(sequence_rows)
        all_sequence = sequence[sequence["history_id"] == "all"]
        overall_cosine = float(all_sequence["cosine"].median())
        overall_relative = float(
            all_sequence["relative_l2"].median()
        )
        task_cosines = (
            all_sequence.groupby("task")["cosine"].median().to_dict()
        )
        row_fraction = float(
            frame["state_local_vs_manual_cosine"]
            .ge(float(gate["row_cosine_threshold"]))
            .mean()
        )
        checks = {
            "overall_sequence_first_cosine": overall_cosine
            >= float(gate["overall_sequence_first_cosine_min"]),
            "each_task_median_cosine": all(
                float(value)
                >= float(gate["each_task_median_cosine_min"])
                for value in task_cosines.values()
            ),
            "overall_sequence_first_relative_l2": overall_relative
            <= float(
                gate["overall_sequence_first_relative_l2_max"]
            ),
            "row_pass_fraction": row_fraction
            >= float(gate["row_pass_fraction_min"]),
            "operating_point_output_exact": float(
                frame["operating_point_output_max_error"].max()
            )
            <= 1.0e-6,
        }
        summary = {
            "stage": "state_operating_point_diagnostic",
            "main_gate2_failed": main_failed,
            "passed_same_readout_thresholds": all(checks.values()),
            "checks": checks,
            "metrics": {
                "overall_sequence_first_cosine": overall_cosine,
                "overall_sequence_first_relative_l2": overall_relative,
                "each_task_median_cosine": task_cosines,
                "row_pass_fraction": row_fraction,
                "row_count": len(frame),
            },
            "config_sha256": sha256_file(config_path),
            "model_info": model_info,
            "dataset_events": dataset_events,
            "wall_seconds": time.perf_counter() - started,
        }
        atomic_frame(
            output_dir / "state_operating_point_rows.parquet", frame
        )
        atomic_frame(
            output_dir
            / "state_operating_point_sequence_metrics.parquet",
            sequence,
        )
        atomic_json(
            output_dir / "state_operating_point_summary.json", summary
        )
        return summary
    finally:
        model.close()


def dry_run(
    protocol: Mapping[str, Any], config_path: Path
) -> Dict[str, Any]:
    return {
        "config_sha256": sha256_file(config_path),
        "split_audit": validate_split_isolation(protocol),
        "expected_ids": {
            stage: expected_ids(protocol, stage)
            for stage in ("smoke", "calibration", "evaluation")
        },
        "required_anchors": {
            stage: list(
                required_reference_anchors(
                    protocol["data"][stage]["target_anchors"],
                    protocol["history_conditions"],
                )
            )
            for stage in ("smoke", "calibration", "evaluation")
        },
        "expected_formal_rows": int(
            protocol["data"]["evaluation"][
                "expected_candidate_history_rows"
            ]
        ),
        "numeric_calibration_status": protocol[
            "numeric_calibration_status"
        ],
        "selected_relative_radius": protocol["numeric"][
            "fd_selected_relative_radius"
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/frozen/p1_state_conditioned_config.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments/p1_state_conditioned/results",
    )
    parser.add_argument(
        "--stage",
        choices=(
            "dry-run",
            "p0-regression",
            "smoke",
            "calibration",
            "evaluation",
            "state-diagnostic",
        ),
        required=True,
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = load_protocol(config_path)
    if args.stage == "dry-run":
        result = dry_run(protocol, config_path)
    elif args.stage == "p0-regression":
        result = p0_regression_stage(protocol, output_dir)
    elif args.stage == "smoke":
        result = smoke_stage(protocol, config_path, output_dir)
    elif args.stage == "calibration":
        result = calibration_stage(protocol, config_path, output_dir)
    elif args.stage == "state-diagnostic":
        result = state_operating_point_diagnostic(
            protocol, config_path, output_dir
        )
    else:
        result = evaluation_stage(protocol, config_path, output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
