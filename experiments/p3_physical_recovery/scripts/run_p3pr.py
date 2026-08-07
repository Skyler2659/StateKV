#!/usr/bin/env python3
"""Collect isolated current-physical-state candidate risk and descriptors."""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
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
EXPERIMENT = ROOT / "experiments/p3_physical_recovery"
SCRIPT_DIR = Path(__file__).resolve().parent
IMPORT_DIRS = (
    ROOT,
    ROOT / "benchmarks/torch",
    ROOT / "experiments/predictive_closure/scripts",
    ROOT / "experiments/p0_v2_fixed_boundary/scripts",
    ROOT / "experiments/p1_state_conditioned/scripts",
    ROOT / "experiments/p2_state_local_risk/scripts",
    ROOT / "experiments/p3_decision_validity/scripts",
    SCRIPT_DIR,
)
for value in IMPORT_DIRS:
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from mlx_predictive_core import (  # noqa: E402
    PureMultiBoundaryMap,
    joint_candidate_selection,
)
from p0_v2_core import AdjacentBoundaryMap, FixedBoundaryReadoutMap  # noqa: E402
from p1_core import HistoryTrajectoryGenerator, clear_runtime_controls  # noqa: E402
from precision_diagnostic import layer_identity_and_injection  # noqa: E402
from run_p1 import load_fp32_model, safe_id  # noqa: E402
from p3pr_core import (  # noqa: E402
    ROOT as CORE_ROOT,
    atomic_frame,
    atomic_json,
    clone_mlx_state,
    exact_kl,
    fisher_variance,
    prune_shared_position,
    sha256_file,
    select_mechanism_disagreement,
    source_integrity,
    stable_softmax,
    state_to_anchor,
    unique_deletion_candidates,
    vector_metrics,
)


TABLES = ("candidate_rows", "layer_rows", "unit_rows", "candidate_registry")


def load_config(path: Path | None = None) -> Dict[str, Any]:
    target = path or EXPERIMENT / "p3pr_config.yaml"
    return yaml.safe_load(target.read_text(encoding="utf-8"))


def stage_targets(config: Mapping[str, Any], stage: str) -> List[int]:
    if stage in {"diagnostic", "calibration"}:
        return [
            int(value)
            for value in config["physical_state"]["calibration_target_anchors"]
        ]
    if stage in {"formal", "recovery_formal", "disagreement_formal"}:
        return [
            int(value)
            for value in config["physical_state"]["formal_target_anchors"]
        ]
    if stage in {
        "replication",
        "recovery_replication",
        "disagreement_replication",
    }:
        return [
            int(value)
            for value in config["physical_state"]["replication_target_anchors"]
        ]
    if stage == "scope_history_anchor":
        return [int(config["physical_state"]["scope_target_anchor"])]
    return [int(config["physical_state"]["primary_target_anchor"])]


def model_protocol(
    config: Mapping[str, Any], stage: str
) -> Dict[str, Any]:
    protocol = yaml.safe_load(
        (ROOT / str(config["model"]["source_protocol"])).read_text(
            encoding="utf-8"
        )
    )
    section = copy.deepcopy(protocol["data"]["evaluation"])
    allocation = config["data"][stage]
    section["gov_report_indices"] = [
        int(value) for value in allocation["gov_report_indices"]
    ]
    section["niah_offsets"] = [
        int(value) for value in allocation["niah_offsets"]
    ]
    section["target_anchors"] = stage_targets(config, stage)
    section["layers"] = list(range(int(config["model"]["num_layers"])))
    protocol["data"]["evaluation"] = section
    protocol["runtime"]["run_id"] = str(config["runtime"]["run_id"])
    if stage == "scope_budget":
        protocol["cache"]["total_budget"] = int(
            config["cache"]["scope_total_budget"]
        )
        protocol["cache"]["selected_core_budget"] = int(
            config["cache"]["scope_selected_core_budget"]
        )
    return protocol


def _tensor(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().double().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def _record_query(record: Any, layer: int) -> torch.Tensor:
    values = [
        record.queries[key].detach().float().cpu()
        for key in sorted(record.queries)
        if str(key).startswith(f"{int(layer)}:")
    ]
    if not values:
        raise RuntimeError(f"physical query missing at layer {layer}")
    return torch.stack(values, dim=0)


def _normalize(values: Mapping[int, float]) -> Dict[int, float]:
    keys = sorted(values)
    array = np.asarray([float(values[key]) for key in keys], dtype=np.float64)
    scale = float(np.std(array))
    normalized = (
        (array - float(np.mean(array))) / max(scale, 1.0e-12)
    )
    return {key: float(value) for key, value in zip(keys, normalized)}


def _ridge_leverage(vectors: torch.Tensor) -> torch.Tensor:
    """Token leverage through the stable sample-space ridge hat matrix."""
    matrix = vectors.detach().double().cpu()
    gram = matrix @ matrix.T
    scale = max(float(torch.trace(gram).item()) / max(len(matrix), 1), 1.0e-12)
    regularized = gram + 1.0e-3 * scale * torch.eye(
        len(matrix), dtype=torch.float64
    )
    solved = torch.linalg.solve(regularized, gram)
    return torch.diagonal(solved).float()


def physical_candidate_scores(
    backend: Any,
    anchor: Any,
    record: Any,
    eligible_positions: Sequence[int],
    selected_layers: Sequence[int],
    seed: int,
) -> Tuple[Dict[str, Dict[int, float]], Dict[int, Dict[str, float]]]:
    positions = [
        int(value) for value in anchor.position_maps[0].tolist()
    ]
    if any(
        [
            int(value)
            for value in anchor.position_maps[int(layer)].tolist()
        ]
        != positions
        for layer in range(len(anchor.position_maps))
    ):
        raise RuntimeError("physical candidate universe is not layer-shared")
    row_by_position = {position: row for row, position in enumerate(positions)}
    accumulator = {
        name: {int(position): 0.0 for position in eligible_positions}
        for name in (
            "attention",
            "value_norm",
            "aov",
            "aor",
            "v_ridge",
            "key_query",
        )
    }
    layer_count = 0
    for layer in selected_layers:
        layer = int(layer)
        attention_heads = (
            record.all_head_attention_distributions[layer]
            .detach()
            .float()
            .cpu()
        )
        attention = attention_heads.mean(dim=0)
        keys = anchor.keys[layer][0].detach().float().cpu()
        values = anchor.values[layer][0].detach().float().cpu()
        query = _record_query(record, layer)
        repeats = int(query.shape[0] // keys.shape[0])
        repeated_keys = keys.repeat_interleave(repeats, dim=0)
        repeated_values = values.repeat_interleave(repeats, dim=0)
        value_norm = values.norm(dim=-1).mean(dim=0)
        key_query = (
            (repeated_keys * query[:, None, :]).sum(dim=-1)
            .abs()
            .mean(dim=0)
        )
        full_head = (
            record.all_head_attention_outputs[layer]
            .detach()
            .float()
            .cpu()
        )
        residual_distance = (
            repeated_values - full_head[:, None, :]
        ).norm(dim=-1).mean(dim=0)
        aov = attention * repeated_values.norm(dim=-1).mean(dim=0)
        aor = (
            attention / (1.0 - attention).clamp_min(1.0e-6)
        ) * residual_distance
        token_vectors = values.permute(1, 0, 2).reshape(len(positions), -1)
        leverage = _ridge_leverage(token_vectors)
        layer_values = {
            "attention": attention,
            "value_norm": value_norm,
            "aov": aov,
            "aor": aor,
            "v_ridge": leverage,
            "key_query": key_query,
        }
        for name, values_by_row in layer_values.items():
            current = {
                int(position): float(values_by_row[row_by_position[int(position)]])
                for position in eligible_positions
            }
            normalized = _normalize(current)
            for position, value in normalized.items():
                accumulator[name][position] += value
        layer_count += 1
    for name in accumulator:
        for position in accumulator[name]:
            accumulator[name][position] /= max(layer_count, 1)
    accumulator["age"] = {
        int(position): float(position) for position in eligible_positions
    }
    generator = np.random.default_rng(int(seed))
    accumulator["random"] = {
        int(position): float(value)
        for position, value in zip(
            eligible_positions, generator.uniform(size=len(eligible_positions))
        )
    }
    per_position = {
        int(position): {
            name: float(values[int(position)])
            for name, values in accumulator.items()
        }
        for position in eligible_positions
    }
    return accumulator, per_position


def disagreement_seed_scores(
    scores: Mapping[str, Mapping[int, float]],
    source_order: Sequence[str],
) -> Tuple[Dict[str, Dict[int, float]], List[str]]:
    """Create the frozen low/high/median 24-candidate seed universe."""
    output: Dict[str, Dict[int, float]] = {}
    order: List[str] = []
    for variant in ("low", "high", "median"):
        for source in source_order:
            name = f"{variant}_{source}"
            values = {
                int(position): float(value)
                for position, value in scores[str(source)].items()
            }
            array = np.asarray(list(values.values()), dtype=np.float64)
            median = float(np.median(array))
            if variant == "low":
                transformed = values
            elif variant == "high":
                transformed = {
                    position: -value for position, value in values.items()
                }
            else:
                transformed = {
                    position: abs(value - median)
                    for position, value in values.items()
                }
            output[name] = transformed
            order.append(name)
    return output, order


def prequery_physical_state(
    backend: Any,
    reference: Any,
    protocol: Mapping[str, Any],
    start_anchor: int,
    target_anchor: int,
) -> Tuple[Any, Mapping[int, set[int]], List[Dict[str, Any]]]:
    generator = HistoryTrajectoryGenerator(backend, reference, protocol)
    candidate = generator._candidate(int(start_anchor), "old_stale_core")
    selection = joint_candidate_selection(
        reference, int(start_anchor), candidate
    )
    state, fixed = backend.state_from_anchor(
        reference.anchors[int(start_anchor)],
        selection,
        cache_config=backend.cfg.cache,
    )
    trace = []
    for query_index in range(int(start_anchor), int(target_anchor)):
        if query_index > int(start_anchor):
            backend.prune_recent_before_query(
                state, fixed, cache_config=backend.cfg.cache
            )
        token = (
            int(reference.anchors[int(start_anchor)].query_token_id)
            if query_index == int(start_anchor)
            else int(reference.generated_token_ids[query_index - 1])
        )
        clear_runtime_controls(backend)
        _logits, record, _elapsed = backend.forward_one(
            state, token, capture_attention=True
        )
        backend.validate_active_budget(
            state, cache_config=backend.cfg.cache
        )
        trace.append(
            {
                "query_index": query_index,
                "token": token,
                "position": int(record.query_position),
                "cache_length": int(state.cache[0].offset),
            }
        )
    return state, fixed, trace


def _state_fingerprint(state: Any) -> str:
    digest = hashlib.sha256()
    for layer, cache in enumerate(state.cache):
        offset = int(cache.offset)
        digest.update(np.asarray(cache.keys[:, :, :offset, :]).tobytes())
        digest.update(np.asarray(cache.values[:, :, :offset, :]).tobytes())
        digest.update(
            np.asarray(state.position_maps[layer], dtype=np.int64).tobytes()
        )
    digest.update(str(int(state.logical_next_position)).encode())
    return digest.hexdigest()


def _path_delta(
    readout: Any,
    direction: np.ndarray,
    midpoint_count: int,
    relative_radius: float,
) -> np.ndarray:
    value = np.asarray(direction, dtype=np.float64).reshape(-1)
    base_norm = max(float(np.linalg.norm(readout.base_input)), 1.0e-12)
    direction_norm = max(float(np.linalg.norm(value)), 1.0e-12)
    epsilon = float(relative_radius) * base_norm / direction_norm
    derivatives = []
    for index in range(int(midpoint_count)):
        node = (index + 0.5) / float(midpoint_count)
        plus = readout.evaluate((node + epsilon) * value)
        minus = readout.evaluate((node - epsilon) * value)
        derivatives.append((plus - minus) / (2.0 * epsilon))
    return np.mean(np.stack(derivatives, axis=0), axis=0)


def _candidate_branch(
    backend: Any,
    prequery_state: Any,
    deleted_position: int,
    token: int,
) -> Tuple[np.ndarray, Any, Any]:
    branch = clone_mlx_state(prequery_state)
    prune_shared_position(branch, int(deleted_position))
    clear_runtime_controls(backend)
    logits, record, _elapsed = backend.forward_one(
        branch, int(token), capture_attention=True
    )
    backend.validate_active_budget(branch, cache_config=backend.cfg.cache)
    return logits.double().numpy(), record, branch


def run_unit(
    backend: Any,
    reference: Any,
    sample: Any,
    protocol: Mapping[str, Any],
    config: Mapping[str, Any],
    stage: str,
    target: int,
) -> Dict[str, List[Dict[str, Any]]]:
    import mlx.core as mx

    started = time.perf_counter()
    start = int(config["physical_state"]["history_start_anchor"])
    state, fixed, trace = prequery_physical_state(
        backend, reference, protocol, start, int(target)
    )
    fingerprint_before = _state_fingerprint(state)
    token = int(reference.anchors[int(target)].query_token_id)

    baseline_state = clone_mlx_state(state)
    clear_runtime_controls(backend)
    base_logits_tensor, base_record, _elapsed = backend.forward_one(
        baseline_state, token, capture_attention=True
    )
    base_logits = base_logits_tensor.double().numpy()
    baseline_anchor = state_to_anchor(
        backend, baseline_state, token, int(target)
    )

    repeated = clone_mlx_state(state)
    clear_runtime_controls(backend)
    repeat_logits_tensor, repeat_record, _elapsed = backend.forward_one(
        repeated, token, capture_attention=True
    )
    repeat_logits = repeat_logits_tensor.double().numpy()
    repeat_error = float(np.max(np.abs(base_logits - repeat_logits)))
    no_op_kl = exact_kl(base_logits, repeat_logits)
    if _state_fingerprint(state) != fingerprint_before:
        raise RuntimeError("baseline clone contaminated the prequery state")

    positions = [
        int(value) for value in baseline_anchor.position_maps[0].tolist()
    ]
    current_position = int(base_record.query_position)
    sink_size = int(config["cache"]["sink_size"])
    recent_size = int(config["cache"]["recent_size"])
    protected = set(positions[:sink_size] + positions[-recent_size:])
    eligible = [value for value in positions if value not in protected]
    diagnostic_layers = [
        int(value) for value in config["boundaries"]["diagnostic_uniform_layers"]
    ]
    seed_token = f"{sample.sample_id}:{target}:{config['runtime']['seed']}"
    candidate_seed = int.from_bytes(
        hashlib.sha256(seed_token.encode()).digest()[:8], "little"
    )
    scores, per_position = physical_candidate_scores(
        backend,
        baseline_anchor,
        base_record,
        eligible,
        diagnostic_layers,
        candidate_seed,
    )
    if stage == "scope_candidate_pool":
        # Frozen alternative composition: delete high-importance rather than
        # low-importance tokens for the six geometry selectors, choose the
        # newest eligible age token, and retain the independent random arm.
        for source in (
            "attention",
            "value_norm",
            "aov",
            "aor",
            "v_ridge",
            "key_query",
            "age",
        ):
            scores[source] = {
                int(position): -float(value)
                for position, value in scores[source].items()
            }
        per_position = {
            int(position): {
                name: float(values[int(position)])
                for name, values in scores.items()
            }
            for position in eligible
        }
    disagreement_stage = stage in {
        "disagreement_calibration",
        "disagreement_formal",
        "disagreement_replication",
    }
    if disagreement_stage:
        seed_scores, seed_order = disagreement_seed_scores(
            scores, config["candidates"]["physical_sources"]
        )
        candidates, dedup_events = unique_deletion_candidates(
            eligible, seed_scores, seed_order
        )
    else:
        candidates, dedup_events = unique_deletion_candidates(
            eligible,
            scores,
            config["candidates"]["physical_sources"],
        )
    if not disagreement_stage and len(candidates) != int(
        config["candidates"]["count"]
    ):
        raise RuntimeError("physical candidate count mismatch")

    frozen_path = (
        EXPERIMENT / "results/frozen_disagreement_model.json"
        if disagreement_stage
        else EXPERIMENT / "results/frozen_model.json"
    )
    frozen_payload = (
        json.loads(frozen_path.read_text(encoding="utf-8"))
        if frozen_path.exists()
        else {}
    )
    readout_layers = (
        list(range(int(config["model"]["num_layers"]) - 1))
        if stage == "calibration"
        else sorted(
            set(diagnostic_layers)
            | (
                {int(frozen_payload["probe_layer"])}
                if "probe_layer" in frozen_payload
                else set()
            )
        )
    )
    readouts = {}
    readout_baselines = {}
    readout_reconstruction = {}
    for layer in readout_layers:
        boundary = int(layer) + 1
        readout = FixedBoundaryReadoutMap(
            backend, baseline_anchor, base_record, boundary
        )
        baseline = readout.baseline()
        readouts[layer] = readout
        readout_baselines[layer] = baseline
        readout_reconstruction[layer] = float(
            np.max(np.abs(baseline - base_logits))
        )

    multi = PureMultiBoundaryMap(backend, baseline_anchor)
    multi_baseline = multi.evaluate(
        [
            np.zeros(int(config["model"]["hidden_size"]), dtype=np.float64)
            for _ in range(int(config["model"]["num_layers"]))
        ]
    )
    multi_reconstruction_error = float(
        np.max(np.abs(multi_baseline - base_logits))
    )
    base_probability = stable_softmax(base_logits)
    prefetched: Dict[str, Tuple[List[np.ndarray], List[List[Dict[str, Any]]]]] = {}
    generator_audit: Dict[str, Any] = {
        "candidate_seed_count": len(candidates),
        "selected_count": len(candidates),
        "exact_physical_kl_used": False,
        "candidate_endpoint_logits_used": False,
        "task_id_used": False,
    }
    if disagreement_stage:
        generator_records = []
        for seed_candidate in candidates:
            retained = [
                value
                for value in positions
                if value != int(seed_candidate.deleted_position)
            ]
            seed_pulses: List[np.ndarray] = []
            seed_identities: List[List[Dict[str, Any]]] = []
            for layer in range(int(config["model"]["num_layers"])):
                pulse, identity_rows, _tensors = layer_identity_and_injection(
                    backend,
                    baseline_anchor,
                    base_record,
                    retained,
                    layer,
                    torch.float64,
                )
                seed_pulses.append(np.asarray(pulse, dtype=np.float64))
                seed_identities.append(identity_rows)
            dense_output = multi.evaluate(seed_pulses)
            dense_delta = dense_output - multi_baseline
            generator_records.append(
                {
                    "candidate_id": seed_candidate.candidate_id,
                    "action_score": float(
                        sum(np.dot(value, value) for value in seed_pulses)
                    ),
                    "dense_score": exact_kl(
                        base_logits, base_logits + dense_delta
                    ),
                }
            )
            prefetched[seed_candidate.candidate_id] = (
                seed_pulses,
                seed_identities,
            )
        selected_ids, generator_audit = select_mechanism_disagreement(
            generator_records,
            int(config["candidates"]["disagreement_generator"][
                "selected_count"
            ]),
        )
        candidate_by_id = {
            candidate.candidate_id: candidate for candidate in candidates
        }
        candidates = [candidate_by_id[value] for value in selected_ids]
        if len(candidates) != int(config["candidates"]["count"]):
            raise RuntimeError("disagreement candidate count mismatch")
    candidate_rows: List[Dict[str, Any]] = []
    layer_rows: List[Dict[str, Any]] = []
    registry_rows: List[Dict[str, Any]] = []
    identity_errors: List[float] = []
    identity_absolute_errors: List[float] = []
    replay_errors: List[float] = []
    branch_position_lengths: List[int] = []

    for candidate_index, candidate in enumerate(candidates):
        candidate_logits, candidate_record, candidate_state = _candidate_branch(
            backend,
            state,
            candidate.deleted_position,
            token,
        )
        exact_value = exact_kl(base_logits, candidate_logits)
        if candidate_index == 0:
            repeat_candidate_logits, _repeat_candidate_record, repeat_branch = (
                _candidate_branch(
                    backend,
                    state,
                    candidate.deleted_position,
                    token,
                )
            )
            replay_errors.append(
                float(
                    np.max(
                        np.abs(candidate_logits - repeat_candidate_logits)
                    )
                )
            )
            backend.release(repeat_branch)
        if _state_fingerprint(state) != fingerprint_before:
            raise RuntimeError("candidate clone contaminated prequery state")
        candidate_positions = [
            int(value)
            for value in candidate_state.position_maps[0].tolist()
        ]
        branch_position_lengths.append(len(candidate_positions))
        if candidate.deleted_position in candidate_positions:
            raise RuntimeError("candidate deletion did not persist")

        retained = [
            value
            for value in positions
            if value != int(candidate.deleted_position)
        ]
        theoretical_pulses: List[np.ndarray] = []
        local_responses: Dict[int, np.ndarray] = {}
        actual_boundary: Dict[int, np.ndarray] = {}
        per_layer_features: Dict[int, Dict[str, float]] = {}
        for layer in range(int(config["model"]["num_layers"])):
            if candidate.candidate_id in prefetched:
                pulse = prefetched[candidate.candidate_id][0][layer]
                identity_rows = prefetched[candidate.candidate_id][1][layer]
            else:
                pulse, identity_rows, _tensors = layer_identity_and_injection(
                    backend,
                    baseline_anchor,
                    base_record,
                    retained,
                    layer,
                    torch.float64,
                )
                pulse = np.asarray(pulse, dtype=np.float64)
            theoretical_pulses.append(pulse)
            identity_errors.extend(
                float(row["stable_relative_error_tau_1em08"])
                for row in identity_rows
                if np.isfinite(
                    float(row["stable_relative_error_tau_1em08"])
                )
            )
            identity_absolute_errors.extend(
                float(row["absolute_l2_error"])
                for row in identity_rows
                if np.isfinite(float(row["absolute_l2_error"]))
            )
            actual_u = (
                candidate_record.projected_attention_outputs[layer]
                - base_record.projected_attention_outputs[layer]
            ).double().numpy()
            adjacent = AdjacentBoundaryMap(backend, layer, base_record)
            local = adjacent.evaluate(pulse) - adjacent.baseline()
            actual_r = (
                candidate_record.layer_outputs[layer]
                - base_record.layer_outputs[layer]
            ).double().numpy()
            local_responses[layer] = local
            if (layer + 1) in candidate_record.residual_inputs:
                actual_boundary[layer] = (
                    candidate_record.residual_inputs[layer + 1]
                    - base_record.residual_inputs[layer + 1]
                ).double().numpy()
                base_boundary_state = _tensor(
                    base_record.residual_inputs[layer + 1]
                )
            else:
                # The model exposes residual_inputs only for actual block
                # inputs.  After the final block, layer_outputs[27] is the
                # terminal residual boundary used for dense diagnostics.
                actual_boundary[layer] = actual_r
                base_boundary_state = _tensor(
                    base_record.layer_outputs[layer]
                )
            u_metrics = vector_metrics(pulse, actual_u)
            r_metrics = vector_metrics(local, actual_r)

            attention = (
                base_record.all_head_attention_distributions[layer]
                .detach()
                .float()
                .cpu()
            )
            row_index = positions.index(int(candidate.deleted_position))
            deleted_mass_heads = attention[:, row_index]
            keys = baseline_anchor.keys[layer][0].detach().float().cpu()
            values = baseline_anchor.values[layer][0].detach().float().cpu()
            key_norms = keys.norm(dim=-1)
            value_norms = values.norm(dim=-1)
            current = per_position[int(candidate.deleted_position)]
            features = {
                "sample_id": sample.sample_id,
                "task": sample.task,
                "stage": stage,
                "target_anchor": int(target),
                "history_length": int(target) - start,
                "candidate_id": candidate.candidate_id,
                "candidate_source": candidate.source,
                "deleted_position": int(candidate.deleted_position),
                "layer": layer,
                "boundary": layer + 1,
                "theory_u_norm": float(np.linalg.norm(pulse)),
                "actual_u_norm": float(np.linalg.norm(actual_u)),
                "injection_cosine": u_metrics["cosine"],
                "injection_relative_l2": u_metrics["relative_l2"],
                "injection_norm_ratio": u_metrics["norm_ratio"],
                "local_r_norm": float(np.linalg.norm(local)),
                "actual_r_norm": float(np.linalg.norm(actual_r)),
                "adjacent_cosine": r_metrics["cosine"],
                "adjacent_relative_l2": r_metrics["relative_l2"],
                "adjacent_norm_ratio": r_metrics["norm_ratio"],
                "actual_boundary_norm": float(
                    np.linalg.norm(actual_boundary[layer])
                ),
                "state_norm": float(np.linalg.norm(base_boundary_state)),
                "action_state_ratio": float(
                    np.linalg.norm(local)
                    / max(np.linalg.norm(base_boundary_state), 1.0e-30)
                ),
                "deleted_attention_mass_mean": float(
                    deleted_mass_heads.mean()
                ),
                "deleted_attention_mass_std": float(
                    deleted_mass_heads.std(unbiased=False)
                ),
                "attention_entropy": float(
                    -(
                        attention
                        * attention.clamp_min(1.0e-12).log()
                    ).sum(dim=1).mean()
                ),
                "attention_concentration": float(
                    attention.max(dim=1).values.mean()
                ),
                "head_disagreement": float(
                    attention[:, row_index].std(unbiased=False)
                ),
                "deleted_key_norm": float(
                    key_norms[:, row_index].mean()
                ),
                "key_norm_mean": float(key_norms.mean()),
                "key_norm_variance": float(
                    key_norms.var(unbiased=False)
                ),
                "deleted_value_norm": float(
                    value_norms[:, row_index].mean()
                ),
                "value_norm_mean": float(value_norms.mean()),
                "value_norm_variance": float(
                    value_norms.var(unbiased=False)
                ),
                "deleted_token_age": float(
                    current_position - int(candidate.deleted_position)
                ),
                **{
                    f"selector_{name}": float(value)
                    for name, value in current.items()
                },
            }
            if layer in readouts:
                theory_output = readouts[layer].evaluate(local)
                theory_delta = theory_output - readout_baselines[layer]
                probe_output = readouts[layer].evaluate(
                    actual_boundary[layer]
                )
                probe_delta = probe_output - readout_baselines[layer]
                features["single_boundary_theory_risk"] = exact_kl(
                    base_logits, base_logits + theory_delta
                )
                features["candidate_probe_risk"] = exact_kl(
                    base_logits, base_logits + probe_delta
                )
                features["single_boundary_fisher_risk"] = 0.5 * fisher_variance(
                    base_probability, theory_delta
                )
                features["probe_fisher_risk"] = 0.5 * fisher_variance(
                    base_probability, probe_delta
                )
            per_layer_features[layer] = features
            layer_rows.append(features)

        subset_definitions = {
            "multi_all_endpoint_risk": list(
                range(int(config["model"]["num_layers"]))
            ),
            "multi_uniform8_endpoint_risk": diagnostic_layers,
            "multi_inherited3_endpoint_risk": [
                int(value)
                for value in config["boundaries"]["inherited_layers"]
            ],
            "multi_pair_endpoint_risk": [7, 22],
            "multi_three_endpoint_risk": [4, 14, 22],
        }
        multi_scores = {}
        for name, active_layers in subset_definitions.items():
            active = set(active_layers)
            blocks = [
                theoretical_pulses[layer]
                if layer in active
                else np.zeros_like(theoretical_pulses[layer])
                for layer in range(int(config["model"]["num_layers"]))
            ]
            output = multi.evaluate(blocks)
            delta = output - multi_baseline
            multi_scores[name] = exact_kl(
                base_logits, base_logits + delta
            )

        path_scores = {}
        path_request = EXPERIMENT / "results/path_calibration_request.json"
        path_layer = (
            int(
                json.loads(path_request.read_text(encoding="utf-8"))[
                    "layer"
                ]
            )
            if stage == "calibration" and path_request.exists()
            else 26
            if stage == "disagreement_calibration"
            else int(frozen_payload.get("probe_layer", 26))
        )
        if stage in {
            "diagnostic",
            "calibration",
            "disagreement_calibration",
        }:
            midpoint_grid = [
                int(value)
                for value in config["representations"]["path_midpoint_grid"]
            ]
        else:
            midpoint_grid = (
                [
                    int(
                        frozen_payload["path_midpoint_count"]
                    )
                ]
                if frozen_path.exists()
                else [2]
            )
        for midpoint_count in midpoint_grid:
            predicted_delta = _path_delta(
                readouts[path_layer],
                actual_boundary[path_layer],
                midpoint_count,
                float(
                    config["representations"][
                        "finite_difference_relative_radius"
                    ]
                ),
            )
            path_scores[
                f"probe_b{path_layer + 1}_path_k{midpoint_count}_risk"
            ] = exact_kl(base_logits, base_logits + predicted_delta)

        row = {
            "sample_id": sample.sample_id,
            "task": sample.task,
            "stage": stage,
            "target_anchor": int(target),
            "candidate_pool_variant": (
                "mechanism_disagreement_pool_v1"
                if disagreement_stage
                else "high_importance_stress"
                if stage == "scope_candidate_pool"
                else "primary_low_importance"
            ),
            "history_start_anchor": start,
            "history_length": int(target) - start,
            "candidate_id": candidate.candidate_id,
            "candidate_source": candidate.source,
            "deleted_position": int(candidate.deleted_position),
            "candidate_raw_rank": int(candidate.raw_rank),
            "candidate_deduplicated": bool(candidate.deduplicated),
            "generator_action_argmin_candidate_id": generator_audit.get(
                "action_argmin_candidate_id"
            ),
            "generator_dense_argmin_candidate_id": generator_audit.get(
                "dense_argmin_candidate_id"
            ),
            "generator_predicted_normalized_regret": generator_audit.get(
                "predicted_normalized_regret"
            ),
            "exact_physical_kl": exact_value,
            "action_only_risk": float(
                sum(np.dot(value, value) for value in theoretical_pulses)
            ),
            "adjacent_only_risk": float(
                sum(np.dot(value, value) for value in local_responses.values())
            ),
            "dense_actual_boundary_energy": float(
                sum(np.dot(value, value) for value in actual_boundary.values())
            ),
            "eligible_count": len(eligible),
            "cache_length_prequery": int(state.cache[0].offset),
            "cache_length_baseline": len(positions),
            "cache_length_candidate": len(candidate_positions),
            "candidate_replay_max_abs_error": (
                replay_errors[0] if candidate_index == 0 else None
            ),
            **multi_scores,
            **path_scores,
        }
        for layer in readouts:
            row[f"theory_b{layer + 1}_risk"] = per_layer_features[layer][
                "single_boundary_theory_risk"
            ]
            row[f"probe_b{layer + 1}_risk"] = per_layer_features[layer][
                "candidate_probe_risk"
            ]
        for layer in diagnostic_layers:
            for feature in (
                "theory_u_norm",
                "local_r_norm",
                "actual_boundary_norm",
                "deleted_attention_mass_mean",
                "deleted_value_norm",
                "deleted_key_norm",
                "action_state_ratio",
            ):
                row[f"b{layer + 1}_{feature}"] = per_layer_features[layer][
                    feature
                ]
        candidate_rows.append(row)
        registry_rows.append(
            {
                "sample_id": sample.sample_id,
                "task": sample.task,
                "stage": stage,
                "target_anchor": int(target),
                "candidate_id": candidate.candidate_id,
                "candidate_source": candidate.source,
                "deleted_position": int(candidate.deleted_position),
                "raw_rank": int(candidate.raw_rank),
                "deduplicated": bool(candidate.deduplicated),
                "retained_positions_json": json.dumps(
                    retained, separators=(",", ":")
                ),
            }
        )
        backend.release(candidate_state)
        mx.synchronize()
        gc.collect()
        mx.clear_cache()

    unit_rows = [
        {
            "sample_id": sample.sample_id,
            "task": sample.task,
            "stage": stage,
            "target_anchor": int(target),
            "candidate_pool_variant": (
                "mechanism_disagreement_pool_v1"
                if disagreement_stage
                else "high_importance_stress"
                if stage == "scope_candidate_pool"
                else "primary_low_importance"
            ),
            "history_start_anchor": start,
            "history_length": int(target) - start,
            "teacher_forcing": True,
            "future_token_as_feature": False,
            "future_attention_as_feature": False,
            "baseline_repeat_max_abs_error": repeat_error,
            "no_op_exact_kl": no_op_kl,
            "prequery_clone_isolated": _state_fingerprint(state)
            == fingerprint_before,
            "query_position": current_position,
            "expected_query_position": int(
                reference.query_records[int(target)].query_position
            ),
            "token_id": token,
            "expected_token_id": int(
                reference.anchors[int(target)].query_token_id
            ),
            "position_maps_shared": True,
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
            "identity_absolute_l2_max": max(
                identity_absolute_errors, default=0.0
            ),
            "readout_reconstruction_max_abs_error": max(
                readout_reconstruction.values(), default=0.0
            ),
            "multi_reconstruction_max_abs_error": multi_reconstruction_error,
            "candidate_cache_length_min": min(branch_position_lengths),
            "candidate_cache_length_max": max(branch_position_lengths),
            "dedup_event_count": int(
                sum(bool(row["deduplicated"]) for row in dedup_events)
            ),
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
            "candidate_generator_action_argmin_candidate_id": (
                generator_audit.get("action_argmin_candidate_id")
            ),
            "candidate_generator_dense_argmin_candidate_id": (
                generator_audit.get("dense_argmin_candidate_id")
            ),
            "candidate_generator_predicted_normalized_regret": (
                generator_audit.get("predicted_normalized_regret")
            ),
            "wall_seconds": time.perf_counter() - started,
            "trace_json": json.dumps(trace, separators=(",", ":")),
        }
    ]
    backend.release(baseline_state, repeated, state)
    return {
        "candidate_rows": candidate_rows,
        "layer_rows": layer_rows,
        "unit_rows": unit_rows,
        "candidate_registry": registry_rows,
    }


def run_sequence(
    backend: Any,
    sample: Any,
    protocol: Mapping[str, Any],
    config: Mapping[str, Any],
    stage: str,
    checkpoint: Path,
) -> Dict[str, pd.DataFrame]:
    reference = backend.generate_reference(
        sample.sample_id, sample.task, sample.prompt
    )
    targets = stage_targets(config, stage)
    missing = [target for target in targets if target not in reference.anchors]
    if missing:
        raise RuntimeError(f"missing physical targets: {missing}")
    rows: Dict[str, List[Dict[str, Any]]] = {name: [] for name in TABLES}
    for target in targets:
        output = run_unit(
            backend,
            reference,
            sample,
            protocol,
            config,
            stage,
            int(target),
        )
        for name in TABLES:
            rows[name].extend(output[name])
        print(
            json.dumps(
                {
                    "event": "p3pr_unit_complete",
                    "stage": stage,
                    "sample_id": sample.sample_id,
                    "target": int(target),
                    "candidate_rows": len(output["candidate_rows"]),
                    "layer_rows": len(output["layer_rows"]),
                }
            ),
            flush=True,
        )
    frames = {name: pd.DataFrame(values) for name, values in rows.items()}
    for name, frame in frames.items():
        atomic_frame(checkpoint / f"{name}.parquet", frame)
    atomic_json(
        checkpoint / "status.json",
        {
            "state": "complete",
            "stage": stage,
            "sample_id": sample.sample_id,
            "targets": targets,
            "row_counts": {
                name: len(frame) for name, frame in frames.items()
            },
        },
    )
    return frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        required=True,
        choices=[
            "diagnostic",
            "calibration",
            "formal",
            "replication",
            "recovery_formal",
            "recovery_replication",
            "scope_budget",
            "scope_history_anchor",
            "scope_candidate_pool",
            "disagreement_calibration",
            "disagreement_formal",
            "disagreement_replication",
        ],
    )
    parser.add_argument(
        "--config-path",
        default="experiments/p3_physical_recovery/p3pr_config.yaml",
    )
    args = parser.parse_args()
    config_path = (ROOT / args.config_path).resolve()
    config = load_config(config_path)
    checks = source_integrity(config)
    scan_path = EXPERIMENT / "results/data_scan_all_allocated.json"
    if not scan_path.exists():
        scan_path = EXPERIMENT / "results/data_scan_110_129.json"
    if not scan_path.exists():
        raise RuntimeError("P3PR data scan must run before opening a role")
    scan = json.loads(scan_path.read_text(encoding="utf-8"))
    allocated = {
        f"gov_report:{value}"
        for value in config["data"][args.stage]["gov_report_indices"]
    } | {
        f"synthetic_niah_{value}"
        for value in config["data"][args.stage]["niah_offsets"]
    }
    passed = {
        str(row["sample_id"])
        for row in scan["rows"]
        if bool(row["success"])
    }
    if not allocated.issubset(passed):
        raise RuntimeError(
            f"stage contains unscanned IDs: {sorted(allocated - passed)}"
        )
    protocol = model_protocol(config, args.stage)
    backend, model_info, samples, dataset_events = load_fp32_model(
        protocol, "evaluation"
    )
    output_dir = EXPERIMENT / "results" / args.stage
    combined: Dict[str, List[pd.DataFrame]] = {
        name: [] for name in TABLES
    }
    started = time.perf_counter()
    try:
        for sample in samples:
            checkpoint = output_dir / "checkpoints" / safe_id(
                sample.sample_id
            )
            status_path = checkpoint / "status.json"
            if (
                bool(config["runtime"]["resume"])
                and status_path.exists()
                and json.loads(status_path.read_text()).get("state")
                == "complete"
            ):
                frames = {
                    name: pd.read_parquet(checkpoint / f"{name}.parquet")
                    for name in TABLES
                }
                print(
                    json.dumps(
                        {
                            "event": "p3pr_resume",
                            "stage": args.stage,
                            "sample_id": sample.sample_id,
                        }
                    ),
                    flush=True,
                )
            else:
                frames = run_sequence(
                    backend,
                    sample,
                    protocol,
                    config,
                    args.stage,
                    checkpoint,
                )
            for name in TABLES:
                combined[name].append(frames[name])
        row_counts = {}
        for name, frames in combined.items():
            frame = pd.concat(frames, ignore_index=True)
            atomic_frame(output_dir / f"{name}.parquet", frame)
            frame.to_csv(output_dir / f"{name}.csv", index=False)
            row_counts[name] = len(frame)
        metadata = {
            "completed": True,
            "stage": args.stage,
            "config_sha256": sha256_file(config_path),
            "source_checks": checks,
            "sample_ids": [sample.sample_id for sample in samples],
            "targets": stage_targets(config, args.stage),
            "row_counts": row_counts,
            "model_info": model_info,
            "dataset_events": dataset_events,
            "wall_seconds": time.perf_counter() - started,
            "target_semantics": config["physical_state"]["exact_target"],
            "full_vector_closure_claimed": False,
        }
        atomic_json(output_dir / "stage_metadata.json", metadata)
        print(json.dumps(metadata, indent=2, sort_keys=True))
    finally:
        backend.close()


if __name__ == "__main__":
    main()
