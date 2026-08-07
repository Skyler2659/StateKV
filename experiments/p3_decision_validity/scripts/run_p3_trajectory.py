#!/usr/bin/env python3
"""Generate controlled P3 trajectories and the independent physical sanity set."""
from __future__ import annotations

import argparse
import copy
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
P0_DIR = ROOT / "experiments/p0_v2_fixed_boundary/scripts"
P1_DIR = ROOT / "experiments/p1_state_conditioned/scripts"
P2_DIR = ROOT / "experiments/p2_state_local_risk/scripts"
PREDICTIVE_DIR = ROOT / "experiments/predictive_closure/scripts"
RECOVERY_DIR = ROOT / "experiments/p2_recovery/scripts"
for value in (
    ROOT,
    ROOT / "benchmarks/torch",
    P0_DIR,
    P1_DIR,
    P2_DIR,
    PREDICTIVE_DIR,
    RECOVERY_DIR,
    SCRIPT_DIR,
):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from mlx_predictive_core import joint_candidate_selection  # noqa: E402
from p0_v2_core import (  # noqa: E402
    AdjacentBoundaryMap,
    FixedBoundaryReadoutMap,
    exact_kl,
    full_replay,
)
from p2_core import state_local_symmetric_fd  # noqa: E402
from run_p0_v2 import common_metadata, theoretical_pulse  # noqa: E402
from run_p1 import _candidate_context, load_fp32_model, safe_id  # noqa: E402
from run_p2 import _state_geometry  # noqa: E402
from p1_core import (  # noqa: E402
    HistoryTrajectoryGenerator,
    clear_runtime_controls,
)
from run_r3 import disable_readout_recording  # noqa: E402
from p3_core import (  # noqa: E402
    atomic_frame,
    atomic_json,
    component_swap_scores,
    decision_event,
    mean_rank_disagreement,
    probability_entropy,
    projected_l2,
    ranking_spearman,
    retained_overlap,
    scalar_risk,
    sha256_file,
    token_age_statistics,
)


EXPERIMENT = ROOT / "experiments/p3_decision_validity"
TABLES = ("candidate_rows", "event_rows", "candidate_registry")


def load_config(path: Path | None = None) -> Dict[str, Any]:
    target = path or EXPERIMENT / "p3_config.yaml"
    return yaml.safe_load(target.read_text(encoding="utf-8"))


def source_integrity(config: Mapping[str, Any]) -> Dict[str, bool]:
    paths = {
        "p0_manifest": ROOT
        / "experiments/p0_v2_fixed_boundary/P0_V2_MANIFEST.yaml",
        "p1_manifest": ROOT
        / "experiments/p1_state_conditioned/"
        "P1_STATE_CONDITIONED_MANIFEST.yaml",
        "p2_manifest": ROOT
        / "experiments/p2_state_local_risk/P2_STATE_LOCAL_MANIFEST.yaml",
        "p2_recovery_manifest": ROOT
        / "experiments/p2_recovery/P2_RECOVERY_MANIFEST.yaml",
    }
    checks = {
        name: sha256_file(path)
        == str(config["source"][f"{name}_sha256"])
        for name, path in paths.items()
    }
    if not all(checks.values()):
        raise RuntimeError(f"P3 source integrity failed: {checks}")
    return checks


def model_protocol(
    config: Mapping[str, Any], stage: str
) -> Dict[str, Any]:
    protocol = yaml.safe_load(
        (ROOT / "configs/frozen/p2_state_local_config.yaml").read_text(
            encoding="utf-8"
        )
    )
    source = config["data"][stage]
    section = copy.deepcopy(protocol["data"]["evaluation"])
    section["gov_report_indices"] = list(
        source["gov_report_indices"]
    )
    section["niah_offsets"] = list(source["niah_offsets"])
    section["target_anchors"] = list(
        config["trajectory"]["target_anchors"]
    )
    section["layers"] = list(config["trajectory"]["layers"])
    protocol["data"]["evaluation"] = section
    protocol["runtime"]["run_id"] = config["runtime"]["run_id"]
    return protocol


def _tensor(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().float().numpy().astype(np.float64)
    return np.asarray(value, dtype=np.float64)


def _record_query(record: Any, layer: int) -> np.ndarray:
    values = [
        _tensor(value).reshape(-1)
        for key, value in sorted(record.queries.items())
        if str(key).startswith(f"{int(layer)}:")
    ]
    return (
        np.concatenate(values)
        if values
        else np.zeros(1, dtype=np.float64)
    )


def _record_attention(record: Any, layer: int) -> np.ndarray:
    value = _tensor(
        record.all_head_attention_distributions[int(layer)]
    )
    probability = value.mean(axis=0).reshape(-1)
    return probability / max(float(probability.sum()), 1.0e-30)


def _key_query_alignment(record: Any, layer: int) -> tuple[float, float]:
    queries = []
    for key, value in sorted(record.queries.items()):
        if str(key).startswith(f"{int(layer)}:"):
            head = int(str(key).split(":")[1])
            queries.append((head, _tensor(value).reshape(-1)))
    keys = {
        int(str(key).split(":")[1]): _tensor(value).reshape(-1)
        for key, value in record.new_keys.items()
        if str(key).startswith(f"{int(layer)}:")
    }
    if not queries or not keys:
        return 0.0, 0.0
    repeats = max(1, len(queries) // len(keys))
    cosine = []
    for head, query in queries:
        key = keys[min(head // repeats, max(keys))]
        denominator = max(
            float(np.linalg.norm(query) * np.linalg.norm(key)), 1.0e-30
        )
        cosine.append(float(np.dot(query, key) / denominator))
    return float(np.mean(cosine)), float(np.std(cosine))


def _position_attention(
    record: Any, position_maps: Mapping[int, Any], layer: int
) -> Dict[int, float]:
    positions = [
        int(value)
        for value in _tensor(position_maps[int(layer)]).reshape(-1)
    ]
    attention = _record_attention(record, layer)
    if len(positions) != len(attention):
        raise RuntimeError("compressed attention/position alignment failed")
    return {
        position: float(value)
        for position, value in zip(positions, attention)
    }


def compressed_observables(
    old_observation: Any,
    current_observation: Any,
    *,
    layer: int,
    boundary: int,
    config: Mapping[str, Any],
) -> Dict[str, float]:
    old_positions = [
        int(value)
        for value in _tensor(
            old_observation.position_maps[int(layer)]
        ).reshape(-1)
    ]
    current_positions = [
        int(value)
        for value in _tensor(
            current_observation.position_maps[int(layer)]
        ).reshape(-1)
    ]
    sink_size = 4
    recent_size = 32
    old_core = old_positions[sink_size:-recent_size]
    current_core = current_positions[sink_size:-recent_size]
    old_recent = set(old_positions[-recent_size:])
    current_recent = set(current_positions[-recent_size:])
    old_attention = _position_attention(
        old_observation.record, old_observation.position_maps, layer
    )
    current_attention = _position_attention(
        current_observation.record,
        current_observation.position_maps,
        layer,
    )
    current_probability = np.asarray(
        list(current_attention.values()), dtype=np.float64
    )
    shared = sorted(set(old_attention) & set(current_attention))
    attention_drift = float(
        sum(
            abs(current_attention[position] - old_attention[position])
            for position in shared
        )
        + sum(
            old_attention[position]
            for position in set(old_attention) - set(current_attention)
        )
        + sum(
            current_attention[position]
            for position in set(current_attention) - set(old_attention)
        )
    )
    old_residual = _tensor(
        old_observation.record.residual_inputs[int(boundary)]
    ).reshape(-1)
    current_residual = _tensor(
        current_observation.record.residual_inputs[int(boundary)]
    ).reshape(-1)
    old_query = _record_query(old_observation.record, layer)
    current_query = _record_query(current_observation.record, layer)
    alignment_mean, alignment_std = _key_query_alignment(
        current_observation.record, layer
    )
    sorted_attention = np.sort(current_probability)[::-1]
    current_position = int(
        current_observation.record.query_position
    )
    result = {
        "retained_overlap": retained_overlap(
            old_positions, current_positions
        ),
        "core_turnover": 1.0
        - retained_overlap(old_core, current_core),
        **token_age_statistics(current_positions, current_position),
        "recent_window_exits": float(
            len(old_recent - current_recent) / max(recent_size, 1)
        ),
        "retained_attention_mass": float(
            sum(
                current_attention[position]
                for position in set(old_positions)
                & set(current_positions)
            )
        ),
        "cache_occupancy": float(
            len(current_positions)
            / int(config["trajectory"]["cache_budget"])
        ),
        "selector_score_drift": attention_drift,
        "selected_core_score_margin": float(
            sorted_attention[0] - sorted_attention[1]
            if len(sorted_attention) > 1
            else 0.0
        ),
        "compressed_residual_norm_drift": float(
            abs(
                np.linalg.norm(current_residual)
                - np.linalg.norm(old_residual)
            )
            / max(np.linalg.norm(old_residual), 1.0e-12)
        ),
        "compressed_sketch_l2": projected_l2(
            old_residual,
            current_residual,
            output_dimension=int(
                config["observables"]["random_projection_dimension"]
            ),
            seed=int(config["observables"]["random_projection_seed"])
            + int(layer),
        ),
        "query_norm_drift": float(
            abs(np.linalg.norm(current_query) - np.linalg.norm(old_query))
            / max(np.linalg.norm(old_query), 1.0e-12)
        ),
        "key_query_alignment_mean": alignment_mean,
        "key_query_alignment_std": alignment_std,
        "attention_entropy": probability_entropy(current_probability),
        "attention_concentration": float(
            np.max(current_probability, initial=0.0)
        ),
        "sink_attention_mass": float(
            current_probability[:sink_size].sum()
        ),
        "recent_attention_mass": float(
            current_probability[-recent_size:].sum()
        ),
        "core_attention_mass": float(
            current_probability[sink_size:-recent_size].sum()
        ),
        "layer_attention_summary_drift": attention_drift,
    }
    return result


def fd_direction(
    downstream: Any,
    operating_point: np.ndarray,
    tangent: np.ndarray,
    radius: float,
) -> np.ndarray:
    return state_local_symmetric_fd(
        downstream,
        np.asarray(operating_point, dtype=np.float64),
        np.asarray(tangent, dtype=np.float64),
        float(radius),
    )["derivative"]


def physical_trajectories(
    model: Any,
    reference: Any,
    candidates: Sequence[Any],
    targets: Sequence[int],
    tau: int,
) -> Dict[tuple[str, int], Dict[str, Any]]:
    """Propagate each tau-selected all-layer mask once through all targets."""
    target_set = set(int(value) for value in targets)
    maximum = max(target_set)
    results: Dict[tuple[str, int], Dict[str, Any]] = {}
    for candidate in candidates:
        selection = joint_candidate_selection(
            reference, int(tau), candidate
        )
        state, fixed = model.state_from_anchor(
            reference.anchors[int(tau)],
            selection,
            cache_config=model.cfg.cache,
        )
        try:
            for offset in range(1, maximum - int(tau) + 2):
                if offset > 1:
                    model.prune_recent_before_query(
                        state, fixed, cache_config=model.cfg.cache
                    )
                target = int(tau) + offset - 1
                token = (
                    int(reference.anchors[int(tau)].query_token_id)
                    if offset == 1
                    else int(reference.generated_token_ids[target - 1])
                )
                expected = (
                    int(reference.anchors[target].query_token_id)
                    if target in reference.anchors
                    else token
                )
                if token != expected:
                    raise RuntimeError(
                        "physical teacher-forced token alignment failed"
                    )
                clear_runtime_controls(model)
                logits, record, _elapsed = model.forward_one(
                    state, token, capture_attention=True
                )
                model.validate_active_budget(
                    state, cache_config=model.cfg.cache
                )
                if target in target_set:
                    results[(candidate.candidate_id, target)] = {
                        "logits": logits.double().numpy(),
                        "record": record,
                        "position_maps": {
                            int(layer): value.detach().clone()
                            for layer, value in state.position_maps.items()
                        },
                    }
        finally:
            model.release(state)
    expected_count = len(candidates) * len(target_set)
    if len(results) != expected_count:
        raise RuntimeError(
            f"physical trajectory rows {len(results)} != {expected_count}"
        )
    return results


def run_sequence(
    model: Any,
    sample: Any,
    protocol: Mapping[str, Any],
    config: Mapping[str, Any],
    stage: str,
    checkpoint: Path,
) -> Dict[str, pd.DataFrame]:
    import mlx.core as mx

    started = time.perf_counter()
    tau = int(config["trajectory"]["calibration_anchor"])
    targets = [
        int(value) for value in config["trajectory"]["target_anchors"]
    ]
    layers = [int(value) for value in config["trajectory"]["layers"]]
    radius = float(config["predictor"]["fd_relative_radius"])
    reference = model.generate_reference(
        sample.sample_id, sample.task, sample.prompt
    )
    generator = HistoryTrajectoryGenerator(model, reference, protocol)

    candidate_by_target: Dict[int, Sequence[Any]] = {}
    registry_by_target: Dict[int, Mapping[str, Any]] = {}
    previous_attention_core = None
    registry_rows: List[Dict[str, Any]] = []
    for target in targets:
        candidates, registry = _candidate_context(
            model,
            reference,
            target,
            protocol,
            previous_attention_core,
        )
        previous_attention_core = registry["attention_core"]
        candidate_by_target[target] = candidates
        registry_by_target[target] = registry
        for candidate in candidates:
            registry_rows.append(
                {
                    "sample_id": sample.sample_id,
                    "task": sample.task,
                    "stage": stage,
                    "target_anchor": target,
                    "horizon": target - tau,
                    "candidate_id": candidate.candidate_id,
                    "candidate_source": candidate.source,
                    "mask_hash": candidate.mask_hash,
                    "retained_positions_json": json.dumps(
                        candidate.retained_positions,
                        separators=(",", ":"),
                    ),
                    "core_positions_json": json.dumps(
                        candidate.core_positions, separators=(",", ":")
                    ),
                    "candidate_seed": int(candidate.seed),
                    "dedup_event_count": len(
                        registry["dedup_events"]
                    ),
                }
            )

    physical = {}
    if stage in {"physical_evaluation", "physical_replication"}:
        physical = physical_trajectories(
            model,
            reference,
            candidate_by_target[tau],
            targets,
            tau,
        )

    tau_logits, tau_record, _tau_maps, _tau_dtypes = full_replay(
        model, reference, tau
    )
    tau_obs_values = generator._segment(
        tau, str(config["trajectory"]["history_source"]), 1
    )
    tau_observation = type(
        "Observation",
        (),
        {
            "logits": tau_obs_values[0],
            "record": tau_obs_values[1],
            "position_maps": tau_obs_values[2],
        },
    )()

    candidate_rows: List[Dict[str, Any]] = []
    event_rows: List[Dict[str, Any]] = []
    for layer in layers:
        boundary = layer + 1
        old_downstream = FixedBoundaryReadoutMap(
            model, reference.anchors[tau], tau_record, boundary
        )
        old_delta = (
            _tensor(tau_observation.record.residual_inputs[boundary])
            - _tensor(tau_record.residual_inputs[boundary])
        ).reshape(-1)
        old_state = _state_geometry(
            old_downstream, tau_logits, old_delta
        )
        for target in targets:
            horizon = target - tau
            base_logits, base_record, _base_maps, _dtypes = full_replay(
                model, reference, target
            )
            current_values = generator._segment(
                tau,
                str(config["trajectory"]["history_source"]),
                horizon + 1,
            )
            current_observation = type(
                "Observation",
                (),
                {
                    "logits": current_values[0],
                    "record": current_values[1],
                    "position_maps": current_values[2],
                },
            )()
            disable_readout_recording(model)
            current_downstream = FixedBoundaryReadoutMap(
                model,
                reference.anchors[target],
                base_record,
                boundary,
            )
            current_delta = (
                _tensor(
                    current_observation.record.residual_inputs[boundary]
                )
                - _tensor(base_record.residual_inputs[boundary])
            ).reshape(-1)
            current_state = _state_geometry(
                current_downstream, base_logits, current_delta
            )
            observables = compressed_observables(
                tau_observation,
                current_observation,
                layer=layer,
                boundary=boundary,
                config=config,
            )
            per_candidate: List[Dict[str, Any]] = []
            current_candidates = candidate_by_target[target]
            tau_by_source = {
                candidate.source: candidate
                for candidate in candidate_by_target[tau]
            }
            current_attention_map = _position_attention(
                current_observation.record,
                current_observation.position_maps,
                layer,
            )
            for candidate in current_candidates:
                common = common_metadata(
                    sample, target, layer, candidate, stage
                )
                action_u, _identity_rows, _tensors = theoretical_pulse(
                    model,
                    reference.anchors[target],
                    base_record,
                    candidate,
                    layer,
                    protocol["numeric"]["identity_norm_floors"],
                    common,
                )
                adjacent = AdjacentBoundaryMap(model, layer, base_record)
                _output, action_r, adjacent_method = adjacent.jvp(
                    action_u
                )
                action_r = np.asarray(action_r, dtype=np.float64)
                current_q0 = fd_direction(
                    current_downstream,
                    np.zeros_like(action_r),
                    action_r,
                    radius,
                )
                current_q1 = fd_direction(
                    current_downstream,
                    current_delta + 0.25 * action_r,
                    action_r,
                    radius,
                )
                current_q3 = fd_direction(
                    current_downstream,
                    current_delta + 0.75 * action_r,
                    action_r,
                    radius,
                )
                if physical:
                    fresh_value = scalar_risk(
                        current_state["gs"],
                        current_state["p_s"],
                        0.5 * (current_q1 + current_q3),
                    )
                    swaps = {
                        name: fresh_value
                        for name in (
                            "risk_all_old",
                            "risk_update_g",
                            "risk_update_f",
                            "risk_update_path",
                            "risk_update_gf",
                            "risk_update_gp",
                            "risk_update_fp",
                            "risk_full_fresh",
                            "risk_path_q1_only",
                            "risk_path_q3_only",
                            "risk_single_midpoint",
                        )
                    }
                else:
                    current_mid = fd_direction(
                        current_downstream,
                        current_delta + 0.50 * action_r,
                        action_r,
                        radius,
                    )
                    old_q1 = fd_direction(
                        old_downstream,
                        old_delta + 0.25 * action_r,
                        action_r,
                        radius,
                    )
                    old_q3 = fd_direction(
                        old_downstream,
                        old_delta + 0.75 * action_r,
                        action_r,
                        radius,
                    )
                    swaps = component_swap_scores(
                        old_state["gs"],
                        old_state["p_s"],
                        old_q1,
                        old_q3,
                        current_state["gs"],
                        current_state["p_s"],
                        current_q1,
                        current_q3,
                        current_midpoint=current_mid,
                    )
                endpoint = current_downstream.evaluate(
                    current_delta + action_r
                )
                controlled = exact_kl(base_logits, endpoint)
                action_only = float(
                    0.5
                    * np.dot(
                        current_state["p0"],
                        (
                            current_q0
                            - np.dot(current_state["p0"], current_q0)
                        )
                        ** 2,
                    )
                )
                retained_mass = float(
                    sum(
                        current_attention_map.get(position, 0.0)
                        for position in candidate.retained_positions
                    )
                )
                old_candidate = tau_by_source[candidate.source]
                row = {
                    "sample_id": sample.sample_id,
                    "task": sample.task,
                    "stage": stage,
                    "history_id": config["trajectory"]["history_id"],
                    "tau_anchor": tau,
                    "target_anchor": target,
                    "horizon": horizon,
                    "layer": layer,
                    "boundary": boundary,
                    "candidate_id": candidate.candidate_id,
                    "candidate_source": candidate.source,
                    "current_mask_hash": candidate.mask_hash,
                    "tau_mask_hash": old_candidate.mask_hash,
                    "mask_changed": (
                        candidate.mask_hash != old_candidate.mask_hash
                    ),
                    "controlled_exact_kl": controlled,
                    "action_only_risk": action_only,
                    "action_u_norm": float(np.linalg.norm(action_u)),
                    "action_r_norm": float(np.linalg.norm(action_r)),
                    "candidate_retained_attention_mass": retained_mass,
                    "adjacent_method": adjacent_method,
                    **swaps,
                }
                if physical:
                    physical_row = physical[
                        (candidate.candidate_id, target)
                    ]
                    row["physical_exact_kl"] = exact_kl(
                        base_logits, physical_row["logits"]
                    )
                    row["physical_boundary_effect_norm"] = float(
                        np.linalg.norm(
                            _tensor(
                                physical_row[
                                    "record"
                                ].residual_inputs[boundary]
                            )
                            - _tensor(
                                current_observation.record.residual_inputs[
                                    boundary
                                ]
                            )
                        )
                    )
                    row["physical_position_overlap"] = retained_overlap(
                        [
                            int(value)
                            for value in _tensor(
                                physical_row["position_maps"][layer]
                            ).reshape(-1)
                        ],
                        [
                            int(value)
                            for value in _tensor(
                                current_observation.position_maps[layer]
                            ).reshape(-1)
                        ],
                    )
                per_candidate.append(row)
                candidate_rows.append(row)
                mx.synchronize()
                gc.collect()
                mx.clear_cache()

            exact_values = [
                row["controlled_exact_kl"] for row in per_candidate
            ]
            fresh_values = [
                row["risk_full_fresh"] for row in per_candidate
            ]
            reused_values = [
                row["risk_all_old"] for row in per_candidate
            ]
            event = decision_event(
                exact_values, fresh_values, reused_values, 0.0
            )
            action_norm = np.asarray(
                [row["action_r_norm"] for row in per_candidate]
            )
            action_score = np.asarray(
                [row["action_only_risk"] for row in per_candidate]
            )
            retained_score = np.asarray(
                [
                    -row["candidate_retained_attention_mass"]
                    for row in per_candidate
                ]
            )
            reused_top = int(np.argmin(reused_values))
            one_midpoint_shift = abs(
                per_candidate[reused_top]["risk_single_midpoint"]
                - per_candidate[reused_top]["risk_all_old"]
            ) / max(
                abs(per_candidate[reused_top]["risk_all_old"]), 1.0e-12
            )
            event_row = {
                "sample_id": sample.sample_id,
                "task": sample.task,
                "stage": stage,
                "history_id": config["trajectory"]["history_id"],
                "tau_anchor": tau,
                "target_anchor": target,
                "horizon": horizon,
                "layer": layer,
                **event,
                **observables,
                "action_norm_median": float(np.median(action_norm)),
                "action_norm_spread": float(
                    np.max(action_norm) - np.min(action_norm)
                ),
                "action_to_compressed_state_ratio": float(
                    np.median(action_norm)
                    / max(
                        np.linalg.norm(
                            _tensor(
                                current_observation.record.residual_inputs[
                                    boundary
                                ]
                            )
                        ),
                        1.0e-12,
                    )
                ),
                "action_only_margin": float(
                    np.sort(action_score)[1] - np.sort(action_score)[0]
                ),
                "cheap_rank_disagreement": mean_rank_disagreement(
                    [action_score, reused_values, retained_score]
                ),
                "top_reused_one_midpoint_shift": float(
                    one_midpoint_shift
                ),
            }
            for epsilon in config["staleness"][
                "harmful_regret_epsilon_grid"
            ]:
                key = str(float(epsilon)).replace(".", "p")
                event_row[f"harmful_stale_eps_{key}"] = bool(
                    event["reuse_normalized_regret"] > float(epsilon)
                )
            if physical:
                physical_values = [
                    row["physical_exact_kl"] for row in per_candidate
                ]
                event_row.update(
                    {
                        "controlled_physical_spearman": (
                            ranking_spearman(
                                exact_values, physical_values
                            )
                        ),
                        "scalar_physical_spearman": (
                            ranking_spearman(
                                fresh_values, physical_values
                            )
                        ),
                        "action_only_physical_spearman": (
                            ranking_spearman(
                                action_score, physical_values
                            )
                        ),
                        "fresh_physical_normalized_regret": (
                            decision_event(
                                physical_values,
                                fresh_values,
                                reused_values,
                                0.0,
                            )["fresh_normalized_regret"]
                        ),
                        "action_only_physical_normalized_regret": (
                            decision_event(
                                physical_values,
                                action_score,
                                reused_values,
                                0.0,
                            )["fresh_normalized_regret"]
                        ),
                    }
                )
            event_rows.append(event_row)
            print(
                json.dumps(
                    {
                        "event": "p3_unit_complete",
                        "stage": stage,
                        "sample_id": sample.sample_id,
                        "target": target,
                        "layer": layer,
                        "candidate_rows": len(candidate_rows),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            del current_downstream
            gc.collect()
            mx.clear_cache()
        del old_downstream
        gc.collect()
        mx.clear_cache()

    frames = {
        "candidate_rows": pd.DataFrame(candidate_rows),
        "event_rows": pd.DataFrame(event_rows),
        "candidate_registry": pd.DataFrame(registry_rows),
    }
    for name, frame in frames.items():
        atomic_frame(checkpoint / f"{name}.parquet", frame)
    atomic_json(
        checkpoint / "status.json",
        {
            "state": "complete",
            "stage": stage,
            "sample_id": sample.sample_id,
            "candidate_row_count": len(candidate_rows),
            "event_row_count": len(event_rows),
            "wall_seconds": time.perf_counter() - started,
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
            "evaluation",
            "replication",
            "physical_evaluation",
            "physical_replication",
        ],
    )
    parser.add_argument(
        "--config-path",
        default="experiments/p3_decision_validity/p3_config.yaml",
    )
    args = parser.parse_args()
    stage = args.stage
    config_path = (ROOT / args.config_path).resolve()
    config = load_config(config_path)
    checks = source_integrity(config)
    scan_path = EXPERIMENT / "results/data_scan_98_109.json"
    if (
        not scan_path.exists()
        or not json.loads(scan_path.read_text())["all_pass"]
    ):
        raise RuntimeError("P3 data scan must pass before trajectories")
    protocol = model_protocol(config, stage)
    model, model_info, samples, dataset_events = load_fp32_model(
        protocol, "evaluation"
    )
    output = EXPERIMENT / "results" / stage
    all_frames: Dict[str, List[pd.DataFrame]] = {
        name: [] for name in TABLES
    }
    started = time.perf_counter()
    try:
        for sample in samples:
            checkpoint = output / "checkpoints" / safe_id(
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
                    name: pd.read_parquet(
                        checkpoint / f"{name}.parquet"
                    )
                    for name in TABLES
                }
                print(
                    json.dumps(
                        {
                            "event": "p3_resume",
                            "stage": stage,
                            "sample_id": sample.sample_id,
                        }
                    ),
                    flush=True,
                )
            else:
                frames = run_sequence(
                    model,
                    sample,
                    protocol,
                    config,
                    stage,
                    checkpoint,
                )
            for name in TABLES:
                all_frames[name].append(frames[name])
        counts = {}
        for name, frames in all_frames.items():
            combined = pd.concat(frames, ignore_index=True)
            atomic_frame(output / f"{name}.parquet", combined)
            combined.to_csv(output / f"{name}.csv", index=False)
            counts[name] = len(combined)
        metadata = {
            "completed": True,
            "stage": stage,
            "config_sha256": sha256_file(config_path),
            "source_checks": checks,
            "model_info": model_info,
            "dataset_events": dataset_events,
            "sample_ids": [sample.sample_id for sample in samples],
            "row_counts": counts,
            "wall_seconds": time.perf_counter() - started,
            "physical_history": stage.startswith("physical_"),
            "full_vector_closure_claimed": False,
        }
        atomic_json(output / "stage_metadata.json", metadata)
        print(json.dumps(metadata, indent=2, sort_keys=True))
    finally:
        model.close()


if __name__ == "__main__":
    main()
