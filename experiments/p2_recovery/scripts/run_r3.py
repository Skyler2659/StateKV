#!/usr/bin/env python3
"""Run R3 calibration, fresh formal evaluation, and replication."""
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
for value in (ROOT, P0_DIR, P1_DIR, P2_DIR, SCRIPT_DIR):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from p0_v2_core import (  # noqa: E402
    AdjacentBoundaryMap,
    FixedBoundaryReadoutMap,
    exact_kl,
    full_replay,
)
from run_p0_v2 import (  # noqa: E402
    candidate_registry_rows,
    common_metadata,
    select_candidates,
    theoretical_pulse,
)
from run_p1 import (  # noqa: E402
    _candidate_context,
    _finalize_vector_accumulator,
    _history_bundle,
    _new_vector_accumulator,
    _state_delta,
    _update_vector_accumulator,
    history_ids,
    load_fp32_model,
    safe_id,
)
from run_p2 import _state_geometry  # noqa: E402
from p2_core import (  # noqa: E402
    atomic_frame,
    atomic_json,
    downstream_jvp_at,
    fisher_variance,
    sha256_file,
    state_local_symmetric_fd,
)
from recovery_core import (  # noqa: E402
    finite_action_metrics,
    state_local_quadratic_risk,
)


TABLES = (
    "direction_rows",
    "path_response_rows",
    "identity_rows",
    "candidate_registry",
    "sequence_vector_metrics",
)


def load_config(
    path: Path | None = None,
) -> Dict[str, Any]:
    config_path = path or (
        ROOT
        / "experiments/p2_recovery/"
        "r3_path_integrated_readout/r3_config.yaml"
    )
    return yaml.safe_load(
        config_path.read_text(encoding="utf-8")
    )


def source_integrity(config: Mapping[str, Any]) -> Dict[str, bool]:
    paths = {
        "p0_manifest": ROOT
        / "experiments/p0_v2_fixed_boundary/P0_V2_MANIFEST.yaml",
        "p1_manifest": ROOT
        / "experiments/p1_state_conditioned/"
        "P1_STATE_CONDITIONED_MANIFEST.yaml",
        "p2_manifest": ROOT
        / "experiments/p2_state_local_risk/"
        "P2_STATE_LOCAL_MANIFEST.yaml",
        "p2_config": ROOT / "configs/frozen/p2_state_local_config.yaml",
        "r1_summary": ROOT
        / "experiments/p2_recovery/"
        "r1_amplitude_trust_region/results/r1_summary.json",
    }
    if "r3_config_sha256" in config["source"]:
        paths.update(
            {
                "r3_config": ROOT
                / "experiments/p2_recovery/"
                "r3_path_integrated_readout/r3_config.yaml",
                "r3_formal_summary": ROOT
                / "experiments/p2_recovery/"
                "r3_path_integrated_readout/results/evaluation/"
                "analysis_summary.json",
                "r3_replication_summary": ROOT
                / "experiments/p2_recovery/"
                "r3_path_integrated_readout/results/replication/"
                "analysis_summary.json",
            }
        )
    checks = {
        name: sha256_file(path)
        == str(config["source"][f"{name}_sha256"])
        for name, path in paths.items()
    }
    if not all(checks.values()):
        raise RuntimeError(f"R3 source integrity failed: {checks}")
    return checks


def model_protocol(
    config: Mapping[str, Any], stage: str
) -> tuple[Dict[str, Any], str]:
    base = yaml.safe_load(
        (ROOT / "configs/frozen/p2_state_local_config.yaml").read_text(
            encoding="utf-8"
        )
    )
    base["runtime"]["run_id"] = config["runtime"]["run_id"]
    source = config["data"][stage]
    if stage == "calibration":
        model_stage = "calibration"
        base["data"][model_stage] = copy.deepcopy(source)
    else:
        model_stage = "evaluation"
        base["data"][model_stage] = copy.deepcopy(source)
    return base, model_stage


def method_names(
    config: Mapping[str, Any], stage: str
) -> List[str]:
    if stage == "replication":
        selected = config["calibration_selection"][
            "selected_method"
        ]
        if selected is None:
            raise RuntimeError("R3 selected method is not frozen")
        return [str(selected)]
    return [
        str(name)
        for name, specification in config["methods"].items()
        if isinstance(specification, Mapping) and "nodes" in specification
    ]


def combine_method(
    node_values: Mapping[float, np.ndarray],
    method: Mapping[str, Any],
) -> np.ndarray:
    nodes = [float(value) for value in method["nodes"]]
    weights = [float(value) for value in method["weights"]]
    return sum(
        weight * node_values[node]
        for node, weight in zip(nodes, weights)
    )


def directional_readout(
    downstream: Any,
    operating_point: Any,
    tangent: Any,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, str]:
    backend = config["numeric_backend"]
    if backend["name"] == "symmetric_central_fd":
        estimate = state_local_symmetric_fd(
            downstream,
            operating_point,
            tangent,
            float(backend["relative_radius"]),
        )
        return (
            downstream.evaluate(operating_point),
            estimate["derivative"],
            "symmetric_central_fd",
        )
    if backend["name"] == "mlx_autodiff":
        return downstream_jvp_at(
            downstream, operating_point, tangent
        )
    raise ValueError(
        f"Unknown R3 numeric backend: {backend['name']}"
    )


def disable_readout_recording(model: Any) -> Dict[str, bool]:
    """Disable diagnostic tensor capture, not model attention itself."""
    state = model.runner.attention_state
    previous = {
        "enabled": bool(state.get("enabled", False)),
        "temporal_record_diagnostics": bool(
            state.get("temporal_record_diagnostics", False)
        ),
    }
    state["enabled"] = False
    state["temporal_record_diagnostics"] = False
    return previous


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
    section = config["data"][stage]
    methods = method_names(config, stage)
    tables: Dict[str, List[Dict[str, Any]]] = {
        name: [] for name in TABLES
    }
    accumulators = {
        method: _new_vector_accumulator() for method in methods
    }
    reference = model.generate_reference(
        sample.sample_id, sample.task, sample.prompt
    )
    previous_attention_core = None
    for target in [
        int(value) for value in section["target_anchors"]
    ]:
        candidates, registry = _candidate_context(
            model,
            reference,
            target,
            protocol,
            previous_attention_core,
        )
        previous_attention_core = registry["attention_core"]
        if stage == "calibration":
            candidates = select_candidates(
                candidates, section["candidate_sources"]
            )
        tables["candidate_registry"].extend(
            candidate_registry_rows(
                sample,
                target,
                candidates,
                registry,
                stage,
            )
        )
        base_logits, base_record, base_positions, _dtypes = full_replay(
            model, reference, target
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
        recording_state_before_readout = disable_readout_recording(
            model
        )
        for layer in [
            int(value) for value in section["layers"]
        ]:
            boundary = layer + 1
            adjacent = AdjacentBoundaryMap(model, layer, base_record)
            downstream = FixedBoundaryReadoutMap(
                model, anchor, base_record, boundary
            )
            state_cache = {}
            for history_id, observation in histories.items():
                delta = _state_delta(
                    observation, base_record, boundary
                )
                state_cache[history_id] = {
                    **_state_geometry(
                        downstream, base_logits, delta
                    ),
                    "delta": delta,
                    "physical_history_kl": exact_kl(
                        base_logits, observation.logits
                    ),
                }
            for candidate in candidates:
                common = common_metadata(
                    sample, target, layer, candidate, stage
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
                _out, action_r, adjacent_method = adjacent.jvp(action_u)
                _base, c0, reference_method = directional_readout(
                    downstream,
                    np.zeros_like(action_r),
                    action_r,
                    config,
                )
                action_baseline = 0.5 * fisher_variance(
                    state_cache["H0"]["p0"], c0
                )
                required_nodes = sorted(
                    {
                        float(node)
                        for method_name in methods
                        for node in config["methods"][method_name][
                            "nodes"
                        ]
                    }
                )
                for history_id, state in state_cache.items():
                    truth = (
                        downstream.evaluate(
                            state["delta"] + action_r
                        )
                        - state["z_s"]
                    )
                    exact_target = exact_kl(
                        base_logits, state["z_s"] + truth
                    )
                    node_values: Dict[float, np.ndarray] = {}
                    operating_max_error = 0.0
                    for alpha in required_nodes:
                        operating, value, _method = directional_readout(
                            downstream,
                            state["delta"] + alpha * action_r,
                            action_r,
                            config,
                        )
                        if (
                            config["numeric_backend"]["name"]
                            == "symmetric_central_fd"
                        ):
                            # directional_readout already evaluated this
                            # exact operating point; do not duplicate a
                            # forward pass merely to compare it to itself.
                            expected_operating = operating
                        else:
                            expected_operating = downstream.evaluate(
                                state["delta"] + alpha * action_r
                            )
                        operating_max_error = max(
                            operating_max_error,
                            float(
                                np.max(
                                    np.abs(
                                        operating
                                        - expected_operating
                                    )
                                )
                            ),
                        )
                        node_values[alpha] = value
                        # The fallback VJP-of-VJP path compiles a large
                        # Metal graph. Every returned value is already a
                        # materialized NumPy array, so the runtime graph
                        # cache can be released at each quadrature node.
                        if (
                            config["numeric_backend"]["name"]
                            == "mlx_autodiff"
                        ):
                            mx.synchronize()
                            gc.collect()
                            mx.clear_cache()
                    direction = {
                        **common,
                        "history_id": history_id,
                        "state_norm": float(
                            np.linalg.norm(state["delta"])
                        ),
                        "action_u_norm": float(
                            np.linalg.norm(action_u)
                        ),
                        "action_r_norm": float(
                            np.linalg.norm(action_r)
                        ),
                        "truth_norm": float(np.linalg.norm(truth)),
                        "controlled_exact_kl": exact_target,
                        "reference_action_fisher_score": float(
                            action_baseline
                        ),
                        "physical_history_kl": state[
                            "physical_history_kl"
                        ],
                        "operating_point_output_max_error": (
                            operating_max_error
                        ),
                        "adjacent_method": adjacent_method,
                        "reference_jvp_method": reference_method,
                        "readout_attention_recording_disabled": True,
                        "recording_enabled_before_disable": (
                            recording_state_before_readout["enabled"]
                        ),
                        "temporal_recording_before_disable": (
                            recording_state_before_readout[
                                "temporal_record_diagnostics"
                            ]
                        ),
                    }
                    tables["direction_rows"].append(direction)
                    for method_name in methods:
                        definition = config["methods"][method_name]
                        predicted = combine_method(
                            node_values, definition
                        )
                        metrics = finite_action_metrics(
                            predicted,
                            truth,
                            state["p_s"],
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
                        if history_id in protocol["metrics"][
                            "primary_histories"
                        ]:
                            _update_vector_accumulator(
                                accumulators[method_name],
                                predicted,
                                truth,
                            )
                        tables["path_response_rows"].append(
                            {
                                **common,
                                "history_id": history_id,
                                "method": method_name,
                                "jvp_cost": int(
                                    definition["jvp_cost"]
                                ),
                                "forward_probe_cost": int(
                                    definition["jvp_cost"]
                                    * int(
                                        config["numeric_backend"][
                                            "forward_probes_per_derivative"
                                        ]
                                    )
                                ),
                                "numeric_backend": config[
                                    "numeric_backend"
                                ]["name"],
                                "predicted_norm": float(
                                    np.linalg.norm(predicted)
                                ),
                                "truth_norm": float(
                                    np.linalg.norm(truth)
                                ),
                                "score": (
                                    state_local_quadratic_risk(
                                        state["gs"],
                                        predicted,
                                        state["p_s"],
                                    )
                                ),
                                "controlled_exact_kl": exact_target,
                                **metrics,
                            }
                        )
                # MLX retains compiled JVP/VJP graphs in its Metal cache.
                # Formal sequences have four times as many candidates as
                # calibration, so release only runtime caches after each
                # fully materialized candidate. All persisted values above
                # are NumPy scalars/arrays; this cannot change the estimator.
                mx.synchronize()
                gc.collect()
                mx.clear_cache()
        print(
            json.dumps(
                {
                    "event": "r3_target_complete",
                    "stage": stage,
                    "sample_id": sample.sample_id,
                    "target": target,
                    "directions": len(
                        tables["direction_rows"]
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    for method, accumulator in accumulators.items():
        tables["sequence_vector_metrics"].append(
            {
                "sample_id": sample.sample_id,
                "task": sample.task,
                "stage": stage,
                "method": method,
                "jvp_cost": int(
                    config["methods"][method]["jvp_cost"]
                ),
                **_finalize_vector_accumulator(
                    accumulator,
                    float(protocol["numeric"]["vector_norm_floor"]),
                ),
            }
        )
    frames = {
        name: pd.DataFrame(rows) for name, rows in tables.items()
    }
    for name, frame in frames.items():
        atomic_frame(checkpoint / f"{name}.parquet", frame)
    atomic_json(
        checkpoint / "status.json",
        {
            "state": "complete",
            "stage": stage,
            "sample_id": sample.sample_id,
            "direction_count": len(tables["direction_rows"]),
            "method_row_count": len(tables["path_response_rows"]),
            "wall_seconds": time.perf_counter() - started,
        },
    )
    return frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        required=True,
        choices=["calibration", "evaluation", "replication"],
    )
    parser.add_argument(
        "--config-path",
        default=(
            "experiments/p2_recovery/"
            "r3_path_integrated_readout/r3_config.yaml"
        ),
    )
    parser.add_argument(
        "--experiment-dir",
        default=(
            "experiments/p2_recovery/"
            "r3_path_integrated_readout"
        ),
    )
    args = parser.parse_args()
    stage = args.stage
    config_path = (ROOT / args.config_path).resolve()
    experiment_dir = (ROOT / args.experiment_dir).resolve()
    if ROOT.resolve() not in config_path.parents:
        raise ValueError("config path must stay inside repository")
    if ROOT.resolve() not in experiment_dir.parents:
        raise ValueError("experiment dir must stay inside repository")
    config = load_config(config_path)
    source_checks = source_integrity(config)
    if stage != "calibration":
        if (
            config["numeric_calibration_status"] != "frozen"
            or config["calibration_selection"][
                "selected_method"
            ]
            is None
        ):
            raise RuntimeError("R3 calibration selection not frozen")
    protocol, model_stage = model_protocol(config, stage)
    model, model_info, samples, dataset_events = load_fp32_model(
        protocol, model_stage
    )
    output = experiment_dir / "results" / stage
    config_hash = sha256_file(config_path)
    if stage in {"evaluation", "replication"}:
        atomic_json(
            output / "stage_freeze.json",
            {
                "stage": stage,
                "config_sha256": config_hash,
                "selected_method": config[
                    "calibration_selection"
                ]["selected_method"],
                "source_checks": source_checks,
                "started_at_unix": time.time(),
            },
        )
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
            complete = bool(
                config["runtime"]["resume"]
                and status_path.exists()
                and json.loads(status_path.read_text()).get("state")
                == "complete"
                and all(
                    (checkpoint / f"{name}.parquet").exists()
                    for name in TABLES
                )
            )
            if complete:
                frames = {
                    name: pd.read_parquet(
                        checkpoint / f"{name}.parquet"
                    )
                    for name in TABLES
                }
                event = "r3_sequence_resumed"
            else:
                frames = run_sequence(
                    model,
                    sample,
                    protocol,
                    config,
                    stage,
                    checkpoint,
                )
                event = "r3_sequence_complete"
            for name, frame in frames.items():
                all_frames[name].append(frame)
            print(
                json.dumps(
                    {
                        "event": event,
                        "stage": stage,
                        "sample_id": sample.sample_id,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            gc.collect()
        merged = {
            name: pd.concat(frames, ignore_index=True)
            for name, frames in all_frames.items()
        }
        section = config["data"][stage]
        expected_directions = int(section["expected_directions"])
        if len(merged["direction_rows"]) != expected_directions:
            raise RuntimeError("R3 direction count mismatch")
        if "expected_method_rows" in section:
            if len(merged["path_response_rows"]) != int(
                section["expected_method_rows"]
            ):
                raise RuntimeError("R3 method row count mismatch")
        for name, frame in merged.items():
            atomic_frame(output / f"{name}.parquet", frame)
        if sha256_file(config_path) != config_hash:
            raise RuntimeError("R3 config changed during stage")
        metadata = {
            "stage": stage,
            "completed": True,
            "config_sha256": config_hash,
            "selected_method": config["calibration_selection"][
                "selected_method"
            ],
            "row_counts": {
                name: len(frame)
                for name, frame in merged.items()
            },
            "sequence_ids": [
                sample.sample_id for sample in samples
            ],
            "source_checks": source_checks,
            "model_info": model_info,
            "dataset_events": dataset_events,
            "wall_seconds": time.perf_counter() - started,
        }
        atomic_json(output / "stage_metadata.json", metadata)
        print(json.dumps(metadata, indent=2, sort_keys=True))
    finally:
        model.close()


if __name__ == "__main__":
    main()
