#!/usr/bin/env python3
"""Run the frozen cross-model, cross-task P3PR generalization experiment."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import yaml


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "experiments/p3pr_generalization"
SCRIPT_DIR = Path(__file__).resolve().parent
IMPORT_DIRS = (
    ROOT,
    ROOT / "benchmarks/torch",
    ROOT / "experiments/predictive_closure/scripts",
    ROOT / "experiments/p0_v2_fixed_boundary/scripts",
    ROOT / "experiments/p1_state_conditioned/scripts",
    ROOT / "experiments/p3_physical_recovery/scripts",
    SCRIPT_DIR,
)
for value in IMPORT_DIRS:
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from kvbench.benchmarks.longbench import LongBenchBenchmark  # noqa: E402
from kvbench.benchmarks.ruler import _synthetic_vt  # noqa: E402
from kvbench.temporal.config import DiscoveryConfig  # noqa: E402
from mlx_predictive_core import PureMultiBoundaryMap  # noqa: E402
from p0_v2_core import FixedBoundaryReadoutMap, P0V2FP32TemporalModel  # noqa: E402
from p1_core import HistoryTrajectoryGenerator, clear_runtime_controls  # noqa: E402
from precision_diagnostic import (  # noqa: E402
    count_quantized_modules,
    dequantize_reference_model,
    layer_identity_and_injection,
)
from p3pr_core import (  # noqa: E402
    atomic_frame,
    atomic_json,
    clone_mlx_state,
    exact_kl,
    prune_shared_position,
    select_mechanism_disagreement,
    sha256_file,
    stable_softmax,
    state_to_anchor,
    unique_deletion_candidates,
)
from run_p3pr import (  # noqa: E402
    _path_delta,
    _state_fingerprint,
    disagreement_seed_scores,
    physical_candidate_scores,
    prequery_physical_state,
)


TABLES = ("candidate_rows", "unit_rows", "candidate_registry")


def load_config(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def safe_id(value: str) -> str:
    return "".join(
        character
        if character.isalnum() or character in "._-"
        else "_"
        for character in value
    )


def stage_ids(config: Mapping[str, Any], stage: str) -> List[int]:
    return [
        int(value)
        for value in config["data"]["role_isolation"][f"{stage}_ids"]
    ]


def build_backend_config(
    config: Mapping[str, Any], model_key: str
) -> DiscoveryConfig:
    model_spec = config["models"][model_key]
    physical = config["physical_protocol"]
    runtime = config["runtime"]
    cfg = DiscoveryConfig()
    cfg.experiment_name = f"{config['program']}_{model_key}"
    cfg.model.name = str(model_spec["source"])
    cfg.model.dtype = "4bit"
    cfg.model.backend = "mlx"
    cfg.model.quant_bits = 4
    cfg.model.deterministic = True
    cfg.model.temperature = 0.0
    cfg.model.do_sample = False
    cfg.model.revision = "main"
    cfg.model.trust_remote_code = False
    cfg.model.local_files_only = bool(model_spec["local_files_only"])
    cfg.model.attn_implementation = "eager"
    cfg.model.prompt_format = str(model_spec["prompt_format"])
    cfg.generation.max_new_tokens = int(config["data"]["max_new_tokens"])
    cfg.generation.temperature = 0.0
    cfg.generation.do_sample = False
    cfg.generation.stop_on_eos = bool(config["data"]["stop_on_eos"])
    cfg.cache.total_budget = int(physical["cache_total_budget"])
    cfg.cache.sink_size = int(physical["sink_size"])
    cfg.cache.recent_size = int(physical["recent_size"])
    cfg.cache.selected_core_budget = int(physical["selected_core_budget"])
    cfg.selectors.observation_window = 32
    cfg.selectors.snapkv_pooling_kernel = 63
    cfg.selectors.snapkv_pooling = "max"
    cfg.selectors.ridge_lambda = 1.0e-3
    cfg.selectors.ridge_lambda_mode = "relative"
    cfg.selectors.shared_token_selection = True
    cfg.runtime.seed = int(runtime["seed"])
    cfg.runtime.deterministic = bool(runtime["deterministic"])
    cfg.runtime.prefill_chunk_size = int(runtime["prefill_chunk_size"])
    cfg.runtime.resume = bool(runtime["resume"])
    cfg.runtime.fail_on_error = True
    cfg.runtime.max_prompt_tokens = int(config["data"]["max_prompt_tokens"])
    cfg.runtime.run_id = f"p3pr_generalization_{model_key}_v1"
    cfg.tasks = {}
    cfg.anchor_steps = [
        int(physical["history_start_anchor"]),
        int(physical["target_anchor"]),
    ]
    cfg.horizons = [1]
    cfg.signal_lags = [1]
    cfg.strategies = []
    # A request larger than the actual layer/head counts resolves to all
    # available layers/heads inside MLXTemporalModel.load().
    cfg.diagnostics.num_layers = 128
    cfg.diagnostics.heads_per_layer = 128
    cfg.diagnostics.layer_selection = "uniform"
    cfg.diagnostics.explicit_layers = []
    cfg.diagnostics.explicit_heads = []
    cfg.independent_fisher.enabled = True
    cfg.independent_fisher.anchors = list(cfg.anchor_steps)
    cfg.independent_fisher.segment_horizon = 1
    return cfg


def load_backend(
    config: Mapping[str, Any], model_key: str
) -> Tuple[Any, Dict[str, Any]]:
    cfg = build_backend_config(config, model_key)
    backend = P0V2FP32TemporalModel.create(cfg)
    model_info = backend.load()
    dequantization = dequantize_reference_model(backend.runner.model)
    remaining = count_quantized_modules(backend.runner.model)[
        "quantized_modules_total"
    ]
    model_info.update(
        {
            "experiment_model_key": model_key,
            "execution": "dequantized_float32",
            "dequantization": dequantization,
            "quantized_modules_remaining": int(remaining),
        }
    )
    if remaining != 0:
        raise RuntimeError(f"{model_key}: quantized module remains reachable")
    expected_family = str(config["models"][model_key]["family"])
    if str(model_info["model_family"]) != expected_family:
        raise RuntimeError(
            f"{model_key}: family {model_info['model_family']} != {expected_family}"
        )
    if not bool(model_info["attention_hook_installed"]):
        raise RuntimeError(f"{model_key}: attention hook not installed")
    if int(model_info["attention_hooked_layers"]) != int(
        model_info["num_layers"]
    ):
        raise RuntimeError(f"{model_key}: incomplete attention hook coverage")
    return backend, model_info


def _longbench_settings(
    task: str, task_spec: Mapping[str, Any], ids: Sequence[int]
) -> Any:
    return SimpleNamespace(
        task=task,
        dataset_name=str(task_spec["source"]),
        dataset_config=str(task_spec["dataset_config"]),
        split=str(task_spec["split"]),
        dataset_revision=None,
        use_official_prompt=bool(task_spec["official_prompt"]),
        max_words=int(task_spec["max_words"]),
        require_official=True,
        num_samples=len(ids),
        sample_strategy="first",
        sample_indices=list(ids),
        data_path=None,
    )


def load_samples(
    config: Mapping[str, Any], stage: str
) -> Tuple[List[Any], List[Dict[str, Any]]]:
    ids = stage_ids(config, stage)
    qmsum_spec = config["data"]["tasks"]["qmsum"]
    qmsum = LongBenchBenchmark(
        _longbench_settings("qmsum", qmsum_spec, ids),
        int(config["runtime"]["seed"]),
    ).load()
    maximum = max(ids)
    vt_spec = config["data"]["tasks"]["variable_tracking"]
    vt_all = _synthetic_vt(
        int(config["runtime"]["seed"]),
        maximum + 1,
        int(vt_spec["context_length"]),
    )
    variable_tracking = [vt_all[index] for index in ids]
    expected = [
        *[f"qmsum:{index}" for index in ids],
        *[f"synthetic_vt_{index}" for index in ids],
    ]
    samples = [*qmsum, *variable_tracking]
    actual = [str(sample.sample_id) for sample in samples]
    if actual != expected:
        raise RuntimeError(f"{stage}: sample isolation {actual} != {expected}")
    events = [
        {
            "task": "qmsum",
            "source": "official_LongBench_qmsum",
            "dataset_official": True,
            "ids": ids,
        },
        {
            "task": "vt",
            "source": "repository_deterministic_synthetic_vt",
            "dataset_official": False,
            "ids": ids,
        },
    ]
    return samples, events


def model_protocol(config: Mapping[str, Any]) -> Dict[str, Any]:
    protocol = yaml.safe_load(
        (ROOT / "configs/frozen/p2_state_local_config.yaml").read_text(encoding="utf-8")
    )
    protocol["numeric"]["seed"] = int(config["runtime"]["seed"])
    protocol["cache"]["total_budget"] = int(
        config["physical_protocol"]["cache_total_budget"]
    )
    protocol["cache"]["sink_size"] = int(
        config["physical_protocol"]["sink_size"]
    )
    protocol["cache"]["recent_size"] = int(
        config["physical_protocol"]["recent_size"]
    )
    protocol["cache"]["selected_core_budget"] = int(
        config["physical_protocol"]["selected_core_budget"]
    )
    return protocol


def relative_boundaries(
    num_layers: int, config: Mapping[str, Any]
) -> List[int]:
    """Return nonterminal residual boundaries b in [1, L-1]."""
    layers = int(num_layers)
    boundaries = {
        max(
            1,
            min(
                layers - 1,
                int(round(float(fraction) * layers)),
            ),
        )
        for fraction in config["boundaries"]["calibration_grid_fractions"]
    }
    boundaries.add(layers - 1)
    return sorted(boundaries)


def _candidate_branch(
    backend: Any,
    state: Any,
    deleted_position: int,
    token: int,
) -> Tuple[np.ndarray, Any, Any]:
    branch = clone_mlx_state(state)
    prune_shared_position(branch, int(deleted_position))
    clear_runtime_controls(backend)
    logits, record, _elapsed = backend.forward_one(
        branch, int(token), capture_attention=True
    )
    backend.validate_active_budget(branch, cache_config=backend.cfg.cache)
    return logits.double().numpy(), record, branch


def _actual_boundary_delta(
    candidate_record: Any, base_record: Any, layer: int
) -> np.ndarray:
    next_layer = int(layer) + 1
    if next_layer not in candidate_record.residual_inputs:
        raise RuntimeError("terminal boundary is excluded from this experiment")
    return (
        candidate_record.residual_inputs[next_layer]
        - base_record.residual_inputs[next_layer]
    ).double().numpy()


def run_unit(
    backend: Any,
    reference: Any,
    sample: Any,
    protocol: Mapping[str, Any],
    config: Mapping[str, Any],
    model_key: str,
    stage: str,
) -> Dict[str, List[Dict[str, Any]]]:
    import mlx.core as mx

    started = time.perf_counter()
    physical = config["physical_protocol"]
    start = int(physical["history_start_anchor"])
    target = int(physical["target_anchor"])
    state, _fixed, trace = prequery_physical_state(
        backend, reference, protocol, start, target
    )
    fingerprint = _state_fingerprint(state)
    token = int(reference.anchors[target].query_token_id)

    baseline_state = clone_mlx_state(state)
    clear_runtime_controls(backend)
    base_logits_tensor, base_record, _elapsed = backend.forward_one(
        baseline_state, token, capture_attention=True
    )
    base_logits = base_logits_tensor.double().numpy()
    baseline_anchor = state_to_anchor(
        backend, baseline_state, token, target
    )

    repeated_state = clone_mlx_state(state)
    clear_runtime_controls(backend)
    repeated_logits_tensor, _repeated_record, _elapsed = backend.forward_one(
        repeated_state, token, capture_attention=True
    )
    repeated_logits = repeated_logits_tensor.double().numpy()
    baseline_repeat_error = float(
        np.max(np.abs(base_logits - repeated_logits))
    )
    no_op_kl = exact_kl(base_logits, repeated_logits)
    if _state_fingerprint(state) != fingerprint:
        raise RuntimeError("baseline branches contaminated prequery state")

    num_layers = int(backend.model_info["num_layers"])
    hidden_size = int(backend.model_info["hidden_size"])
    boundaries = relative_boundaries(num_layers, config)
    primary_boundary = num_layers - 1
    probe_layers = [boundary - 1 for boundary in boundaries]
    positions = [
        int(value) for value in baseline_anchor.position_maps[0].tolist()
    ]
    if any(
        [
            int(value)
            for value in baseline_anchor.position_maps[layer].tolist()
        ]
        != positions
        for layer in range(num_layers)
    ):
        raise RuntimeError("candidate universe is not shared across layers")
    protected = set(
        positions[: int(physical["sink_size"])]
        + positions[-int(physical["recent_size"]) :]
    )
    eligible = [position for position in positions if position not in protected]

    seed_text = (
        f"{model_key}:{stage}:{sample.sample_id}:{target}:"
        f"{config['runtime']['seed']}"
    )
    candidate_seed = int.from_bytes(
        hashlib.sha256(seed_text.encode()).digest()[:8], "little"
    )
    scores, _per_position = physical_candidate_scores(
        backend,
        baseline_anchor,
        base_record,
        eligible,
        list(range(num_layers)),
        candidate_seed,
    )
    seed_scores, seed_order = disagreement_seed_scores(
        scores, config["candidates"]["source_order"]
    )
    seed_candidates, dedup_events = unique_deletion_candidates(
        eligible, seed_scores, seed_order
    )
    if len(seed_candidates) != int(config["candidates"]["seed_count"]):
        raise RuntimeError("candidate seed count mismatch")

    multi = PureMultiBoundaryMap(backend, baseline_anchor)
    zero_blocks = [
        np.zeros(hidden_size, dtype=np.float64) for _ in range(num_layers)
    ]
    multi_baseline = multi.evaluate(zero_blocks)
    multi_reconstruction_error = float(
        np.max(np.abs(multi_baseline - base_logits))
    )

    pulse_cache: Dict[str, Tuple[List[np.ndarray], List[List[Dict[str, Any]]]]] = {}
    generator_records = []
    for candidate in seed_candidates:
        retained = [
            position
            for position in positions
            if position != int(candidate.deleted_position)
        ]
        pulses: List[np.ndarray] = []
        identity_rows_by_layer: List[List[Dict[str, Any]]] = []
        for layer in range(num_layers):
            pulse, identity_rows, _tensors = layer_identity_and_injection(
                backend,
                baseline_anchor,
                base_record,
                retained,
                layer,
                torch.float64,
            )
            pulses.append(np.asarray(pulse, dtype=np.float64))
            identity_rows_by_layer.append(identity_rows)
        dense_output = multi.evaluate(pulses)
        dense_delta = dense_output - multi_baseline
        generator_records.append(
            {
                "candidate_id": candidate.candidate_id,
                "action_score": float(
                    sum(np.dot(pulse, pulse) for pulse in pulses)
                ),
                "dense_score": exact_kl(
                    base_logits, base_logits + dense_delta
                ),
            }
        )
        pulse_cache[candidate.candidate_id] = (
            pulses,
            identity_rows_by_layer,
        )
    selected_ids, generator_audit = select_mechanism_disagreement(
        generator_records, int(config["candidates"]["selected_count"])
    )
    candidate_by_id = {
        candidate.candidate_id: candidate for candidate in seed_candidates
    }
    selected = [candidate_by_id[candidate_id] for candidate_id in selected_ids]

    readouts = {
        boundary: FixedBoundaryReadoutMap(
            backend,
            baseline_anchor,
            base_record,
            boundary,
        )
        for boundary in boundaries
    }
    readout_baselines = {
        boundary: readout.baseline()
        for boundary, readout in readouts.items()
    }
    readout_reconstruction = {
        boundary: float(
            np.max(np.abs(readout_baselines[boundary] - base_logits))
        )
        for boundary in boundaries
    }
    path_midpoint_count = int(config["boundaries"]["path_midpoint_count"])
    path_radius = float(
        config["boundaries"]["finite_difference_relative_radius"]
    )

    candidate_rows: List[Dict[str, Any]] = []
    registry_rows: List[Dict[str, Any]] = []
    identity_errors: List[float] = []
    replay_errors: List[float] = []
    cache_lengths: List[int] = []
    for candidate_index, candidate in enumerate(selected):
        candidate_logits, candidate_record, candidate_state = _candidate_branch(
            backend, state, candidate.deleted_position, token
        )
        exact_value = exact_kl(base_logits, candidate_logits)
        if candidate_index == 0:
            replay_logits, _replay_record, replay_state = _candidate_branch(
                backend, state, candidate.deleted_position, token
            )
            replay_errors.append(
                float(np.max(np.abs(candidate_logits - replay_logits)))
            )
            backend.release(replay_state)
        if _state_fingerprint(state) != fingerprint:
            raise RuntimeError("candidate branch contaminated prequery state")
        candidate_positions = [
            int(value)
            for value in candidate_state.position_maps[0].tolist()
        ]
        if int(candidate.deleted_position) in candidate_positions:
            raise RuntimeError("physical deletion did not persist")
        cache_lengths.append(len(candidate_positions))

        pulses, identity_rows_by_layer = pulse_cache[candidate.candidate_id]
        for identity_rows in identity_rows_by_layer:
            identity_errors.extend(
                float(row["stable_relative_error_tau_1em08"])
                for row in identity_rows
                if np.isfinite(
                    float(row["stable_relative_error_tau_1em08"])
                )
            )
        dense_output = multi.evaluate(pulses)
        dense_delta = dense_output - multi_baseline
        row: Dict[str, Any] = {
            "model_key": model_key,
            "model_family": backend.model_info["model_family"],
            "model_source": backend.model_info["model_name"],
            "num_layers": num_layers,
            "hidden_size": hidden_size,
            "sample_id": sample.sample_id,
            "task": sample.task,
            "stage": stage,
            "target_anchor": target,
            "history_start_anchor": start,
            "history_length": target - start,
            "candidate_id": candidate.candidate_id,
            "candidate_source": candidate.source,
            "deleted_position": int(candidate.deleted_position),
            "exact_physical_kl": exact_value,
            "action_only_risk": float(
                sum(np.dot(pulse, pulse) for pulse in pulses)
            ),
            "dense_all_layer_mechanistic_risk": exact_kl(
                base_logits, base_logits + dense_delta
            ),
            "primary_boundary": primary_boundary,
            "primary_probe_layer": primary_boundary - 1,
            "eligible_count": len(eligible),
            "cache_length_baseline": len(positions),
            "cache_length_candidate": len(candidate_positions),
            "generator_action_argmin_candidate_id": generator_audit[
                "action_argmin_candidate_id"
            ],
            "generator_dense_argmin_candidate_id": generator_audit[
                "dense_argmin_candidate_id"
            ],
            "generator_predicted_normalized_regret": generator_audit[
                "predicted_normalized_regret"
            ],
        }
        for boundary in boundaries:
            layer = boundary - 1
            actual_delta = _actual_boundary_delta(
                candidate_record, base_record, layer
            )
            exact_map_delta = (
                readouts[boundary].evaluate(actual_delta)
                - readout_baselines[boundary]
            )
            path_delta = _path_delta(
                readouts[boundary],
                actual_delta,
                path_midpoint_count,
                path_radius,
            )
            row[f"b{boundary}_exact_map_risk"] = exact_kl(
                base_logits, base_logits + exact_map_delta
            )
            row[f"b{boundary}_path_k{path_midpoint_count}_risk"] = exact_kl(
                base_logits, base_logits + path_delta
            )
        row["relative_penultimate_exact_map_risk"] = row[
            f"b{primary_boundary}_exact_map_risk"
        ]
        row["relative_penultimate_path_k1_risk"] = row[
            f"b{primary_boundary}_path_k{path_midpoint_count}_risk"
        ]
        candidate_rows.append(row)
        registry_rows.append(
            {
                "model_key": model_key,
                "sample_id": sample.sample_id,
                "task": sample.task,
                "stage": stage,
                "candidate_id": candidate.candidate_id,
                "candidate_source": candidate.source,
                "deleted_position": int(candidate.deleted_position),
                "retained_positions_json": json.dumps(
                    [
                        position
                        for position in positions
                        if position != int(candidate.deleted_position)
                    ],
                    separators=(",", ":"),
                ),
            }
        )
        backend.release(candidate_state)
        mx.synchronize()
        gc.collect()
        mx.clear_cache()

    unit_rows = [
        {
            "model_key": model_key,
            "model_family": backend.model_info["model_family"],
            "model_source": backend.model_info["model_name"],
            "num_layers": num_layers,
            "hidden_size": hidden_size,
            "sample_id": sample.sample_id,
            "task": sample.task,
            "stage": stage,
            "target_anchor": target,
            "history_start_anchor": start,
            "history_length": target - start,
            "teacher_forcing": True,
            "future_token_as_feature": False,
            "future_attention_as_feature": False,
            "free_generation": False,
            "primary_boundary": primary_boundary,
            "primary_probe_layer": primary_boundary - 1,
            "diagnostic_boundaries_json": json.dumps(boundaries),
            "baseline_repeat_max_abs_error": baseline_repeat_error,
            "no_op_exact_kl": no_op_kl,
            "prequery_clone_isolated": _state_fingerprint(state) == fingerprint,
            "query_position": int(base_record.query_position),
            "expected_query_position": int(
                reference.query_records[target].query_position
            ),
            "token_id": token,
            "expected_token_id": int(reference.anchors[target].query_token_id),
            "candidate_count": len(candidate_rows),
            "finite_candidate_count": int(
                np.isfinite(
                    [row["exact_physical_kl"] for row in candidate_rows]
                ).sum()
            ),
            "exact_kl_range": float(
                np.ptp(
                    [row["exact_physical_kl"] for row in candidate_rows]
                )
            ),
            "candidate_replay_max_abs_error": max(
                replay_errors, default=0.0
            ),
            "identity_stable_relative_l2_tau_1e8_max": max(
                identity_errors, default=0.0
            ),
            "multi_reconstruction_max_abs_error": multi_reconstruction_error,
            "readout_reconstruction_max_abs_error": max(
                readout_reconstruction.values(), default=0.0
            ),
            "candidate_cache_length_min": min(cache_lengths),
            "candidate_cache_length_max": max(cache_lengths),
            "candidate_generator_seed_count": int(
                generator_audit["candidate_seed_count"]
            ),
            "candidate_generator_exact_kl_used": bool(
                generator_audit["exact_physical_kl_used"]
            ),
            "candidate_generator_endpoint_logits_used": bool(
                generator_audit["candidate_endpoint_logits_used"]
            ),
            "candidate_generator_task_id_used": bool(
                generator_audit["task_id_used"]
            ),
            "dedup_event_count": int(
                sum(bool(row["deduplicated"]) for row in dedup_events)
            ),
            "prompt_tokens": int(reference.prompt_length),
            "prompt_truncated": bool(reference.prompt_truncated),
            "reference_generation_seconds": float(reference.generation_time_s),
            "wall_seconds": time.perf_counter() - started,
            "trace_json": json.dumps(trace, separators=(",", ":")),
        }
    ]
    backend.release(baseline_state, repeated_state, state)
    return {
        "candidate_rows": candidate_rows,
        "unit_rows": unit_rows,
        "candidate_registry": registry_rows,
    }


def run_sequence(
    backend: Any,
    sample: Any,
    protocol: Mapping[str, Any],
    config: Mapping[str, Any],
    model_key: str,
    stage: str,
    checkpoint: Path,
) -> Dict[str, pd.DataFrame]:
    reference = backend.generate_reference(
        sample.sample_id, sample.task, sample.prompt
    )
    target = int(config["physical_protocol"]["target_anchor"])
    start = int(config["physical_protocol"]["history_start_anchor"])
    missing = [
        anchor for anchor in (start, target) if anchor not in reference.anchors
    ]
    if missing:
        raise RuntimeError(f"reference missing anchors: {missing}")
    output = run_unit(
        backend,
        reference,
        sample,
        protocol,
        config,
        model_key,
        stage,
    )
    frames = {
        name: pd.DataFrame(output[name]) for name in TABLES
    }
    for name, frame in frames.items():
        atomic_frame(checkpoint / f"{name}.parquet", frame)
    atomic_json(
        checkpoint / "status.json",
        {
            "state": "complete",
            "model_key": model_key,
            "stage": stage,
            "sample_id": sample.sample_id,
            "row_counts": {
                name: len(frame) for name, frame in frames.items()
            },
        },
    )
    return frames


def run_model_stage(
    backend: Any,
    model_info: Mapping[str, Any],
    config: Mapping[str, Any],
    config_path: Path,
    model_key: str,
    stage: str,
) -> Dict[str, Any]:
    samples, dataset_events = load_samples(config, stage)
    protocol = model_protocol(config)
    output_dir = EXPERIMENT / "results" / stage / model_key
    combined: Dict[str, List[pd.DataFrame]] = {
        name: [] for name in TABLES
    }
    started = time.perf_counter()
    for sample in samples:
        checkpoint = output_dir / "checkpoints" / safe_id(sample.sample_id)
        status_path = checkpoint / "status.json"
        if (
            bool(config["runtime"]["resume"])
            and status_path.exists()
            and json.loads(status_path.read_text()).get("state") == "complete"
        ):
            frames = {
                name: pd.read_parquet(checkpoint / f"{name}.parquet")
                for name in TABLES
            }
            event = "generalization_resume"
        else:
            frames = run_sequence(
                backend,
                sample,
                protocol,
                config,
                model_key,
                stage,
                checkpoint,
            )
            event = "generalization_unit_complete"
        for name in TABLES:
            combined[name].append(frames[name])
        print(
            json.dumps(
                {
                    "event": event,
                    "model_key": model_key,
                    "stage": stage,
                    "sample_id": sample.sample_id,
                }
            ),
            flush=True,
        )

    row_counts = {}
    for name, frames in combined.items():
        frame = pd.concat(frames, ignore_index=True)
        atomic_frame(output_dir / f"{name}.parquet", frame)
        frame.to_csv(output_dir / f"{name}.csv", index=False)
        row_counts[name] = len(frame)
    metadata = {
        "completed": True,
        "model_key": model_key,
        "stage": stage,
        "config_sha256": sha256_file(config_path),
        "sample_ids": [sample.sample_id for sample in samples],
        "row_counts": row_counts,
        "model_info": dict(model_info),
        "dataset_events": dataset_events,
        "wall_seconds": time.perf_counter() - started,
        "primary_boundary": int(model_info["num_layers"]) - 1,
        "target_semantics": config["physical_protocol"]["target"],
    }
    atomic_json(output_dir / "stage_metadata.json", metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path",
        default="experiments/p3pr_generalization/p3pr_generalization_config.yaml",
    )
    parser.add_argument(
        "--model",
        default="all",
        choices=["all", "qwen25_05b", "llama32_1b"],
    )
    parser.add_argument(
        "--stage",
        default="all",
        choices=["all", "calibration", "formal", "replication"],
    )
    args = parser.parse_args()
    config_path = (ROOT / args.config_path).resolve()
    config = load_config(config_path)
    models = (
        list(config["runtime"]["model_order"])
        if args.model == "all"
        else [args.model]
    )
    stages = (
        list(config["runtime"]["stage_order"])
        if args.stage == "all"
        else [args.stage]
    )
    all_metadata = []
    for model_key in models:
        backend, model_info = load_backend(config, model_key)
        print(
            json.dumps(
                {
                    "event": "generalization_model_ready",
                    "model_key": model_key,
                    "model_family": model_info["model_family"],
                    "num_layers": model_info["num_layers"],
                    "hidden_size": model_info["hidden_size"],
                    "attention_hooked_layers": model_info[
                        "attention_hooked_layers"
                    ],
                }
            ),
            flush=True,
        )
        try:
            for stage in stages:
                all_metadata.append(
                    run_model_stage(
                        backend,
                        model_info,
                        config,
                        config_path,
                        model_key,
                        stage,
                    )
                )
        finally:
            backend.close()
            gc.collect()
    atomic_json(
        EXPERIMENT / "results/run_summary.json",
        {
            "completed": True,
            "models": models,
            "stages": stages,
            "runs": all_metadata,
        },
    )


if __name__ == "__main__":
    main()
