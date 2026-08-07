#!/usr/bin/env python3
"""Replay P2 formal states for retrospective R0/R1 diagnostics."""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping

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
from run_p0_v2 import common_metadata, theoretical_pulse  # noqa: E402
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
    fisher_inner,
    sha256_file,
    vector_metrics,
)
from recovery_core import finite_action_metrics  # noqa: E402


GAMMA = [0.0625, 0.125, 0.25, 0.5, 0.75, 1.0]


def source_protocol() -> Dict[str, Any]:
    return yaml.safe_load(
        (ROOT / "configs/frozen/p2_state_local_config.yaml").read_text(
            encoding="utf-8"
        )
    )


def verify_sources(
    r0: Mapping[str, Any], r1: Mapping[str, Any]
) -> None:
    checks = {
        "p2_config_r0": sha256_file(
            ROOT / "configs/frozen/p2_state_local_config.yaml"
        )
        == r0["source"]["p2_config_sha256"],
        "p2_config_r1": sha256_file(
            ROOT / "configs/frozen/p2_state_local_config.yaml"
        )
        == r1["source"]["p2_config_sha256"],
        "p2_manifest_r0": sha256_file(
            ROOT / r0["source"]["p2_manifest"]
        )
        == r0["source"]["p2_manifest_sha256"],
        "p2_manifest_r1": sha256_file(
            ROOT
            / "experiments/p2_state_local_risk/"
            "P2_STATE_LOCAL_MANIFEST.yaml"
        )
        == r1["source"]["p2_manifest_sha256"],
    }
    if not all(checks.values()):
        raise RuntimeError(f"recovery source integrity failed: {checks}")


def run_sequence(
    model: Any,
    sample: Any,
    protocol: Mapping[str, Any],
    checkpoint: Path,
) -> Dict[str, pd.DataFrame]:
    started = time.perf_counter()
    rows: List[Dict[str, Any]] = []
    reference = model.generate_reference(
        sample.sample_id, sample.task, sample.prompt
    )
    section = protocol["data"]["evaluation"]
    vector_accumulators = {
        gamma: _new_vector_accumulator() for gamma in GAMMA
    }
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
        for layer in [
            int(value) for value in section["layers"]
        ]:
            boundary = layer + 1
            adjacent = AdjacentBoundaryMap(model, layer, base_record)
            downstream = FixedBoundaryReadoutMap(
                model, anchor, base_record, boundary
            )
            state_cache: Dict[str, Dict[str, Any]] = {}
            for history_id, observation in histories.items():
                delta = _state_delta(
                    observation, base_record, boundary
                )
                state = _state_geometry(
                    downstream, base_logits, delta
                )
                state.update(
                    {
                        "delta": delta,
                        "workpoint_norm": float(
                            np.linalg.norm(
                                np.asarray(
                                    downstream.base_input,
                                    dtype=np.float64,
                                )
                                + delta
                            )
                        ),
                        "physical_history_kl": exact_kl(
                            base_logits, observation.logits
                        ),
                    }
                )
                state_cache[history_id] = state
            for candidate in candidates:
                common = common_metadata(
                    sample,
                    target,
                    layer,
                    candidate,
                    "retrospective_r0_r1",
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
                retained_mass = [
                    float(row["retained_mass"])
                    for row in identity_rows
                ]
                _out, action_r, adjacent_method = adjacent.jvp(action_u)
                _base, c0, reference_method = downstream.jvp(action_r)
                for history_id, state in state_cache.items():
                    operating, cs, state_method = downstream_jvp_at(
                        downstream, state["delta"], action_r
                    )
                    operating_error = float(
                        np.max(np.abs(operating - state["z_s"]))
                    )
                    jacobian_drift = vector_metrics(c0, cs)
                    for gamma in GAMMA:
                        truth = (
                            downstream.evaluate(
                                state["delta"] + gamma * action_r
                            )
                            - state["z_s"]
                        )
                        predicted = gamma * cs
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
                                vector_accumulators[gamma],
                                predicted,
                                truth,
                            )
                        interaction = float(
                            np.dot(state["gs"], predicted)
                        )
                        rows.append(
                            {
                                **common,
                                "history_id": history_id,
                                "gamma": float(gamma),
                                "state_norm": float(
                                    np.linalg.norm(state["delta"])
                                ),
                                "state_workpoint_norm": state[
                                    "workpoint_norm"
                                ],
                                "action_u_norm": float(
                                    np.linalg.norm(action_u)
                                ),
                                "action_r_norm": float(
                                    np.linalg.norm(action_r)
                                ),
                                "scaled_action_norm": float(
                                    gamma * np.linalg.norm(action_r)
                                ),
                                "action_to_state_workpoint_ratio": float(
                                    np.linalg.norm(action_r)
                                    / max(
                                        state["workpoint_norm"],
                                        1.0e-12,
                                    )
                                ),
                                "nonlinear_increment_norm": float(
                                    np.linalg.norm(truth)
                                ),
                                "residual_norm": float(
                                    metrics["error_norm"]
                                ),
                                "physical_history_kl": state[
                                    "physical_history_kl"
                                ],
                                "controlled_exact_kl": exact_kl(
                                    base_logits,
                                    state["z_s"] + truth,
                                ),
                                "retained_mass_min": min(
                                    retained_mass
                                ),
                                "retained_mass_median": float(
                                    np.median(retained_mass)
                                ),
                                "retained_mass_max": max(
                                    retained_mass
                                ),
                                "state_action_interaction": interaction,
                                "state_action_interaction_sign": int(
                                    np.sign(interaction)
                                ),
                                "operating_point_output_max_error": (
                                    operating_error
                                ),
                                "adjacent_method": adjacent_method,
                                "reference_jvp_method": reference_method,
                                "state_jvp_method": state_method,
                                **{
                                    f"jacobian_drift_{key}": value
                                    for key, value in jacobian_drift.items()
                                },
                                **metrics,
                            }
                        )
        print(
            json.dumps(
                {
                    "event": "recovery_diagnostic_target_complete",
                    "sample_id": sample.sample_id,
                    "target": target,
                    "rows": len(rows),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    sequence_rows = [
        {
            "sample_id": sample.sample_id,
            "task": sample.task,
            "gamma": gamma,
            **_finalize_vector_accumulator(
                accumulator,
                float(protocol["numeric"]["vector_norm_floor"]),
            ),
        }
        for gamma, accumulator in vector_accumulators.items()
    ]
    frames = {
        "scaling_rows": pd.DataFrame(rows),
        "sequence_scaling_metrics": pd.DataFrame(sequence_rows),
    }
    for name, frame in frames.items():
        atomic_frame(checkpoint / f"{name}.parquet", frame)
    atomic_json(
        checkpoint / "status.json",
        {
            "state": "complete",
            "sample_id": sample.sample_id,
            "row_count": len(rows),
            "wall_seconds": time.perf_counter() - started,
        },
    )
    return frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    r0_path = (
        ROOT
        / "experiments/p2_recovery/r0_failure_map/r0_config.yaml"
    )
    r1_path = (
        ROOT
        / "experiments/p2_recovery/"
        "r1_amplitude_trust_region/r1_config.yaml"
    )
    r0 = yaml.safe_load(r0_path.read_text(encoding="utf-8"))
    r1 = yaml.safe_load(r1_path.read_text(encoding="utf-8"))
    verify_sources(r0, r1)
    protocol = source_protocol()
    model, model_info, samples, dataset_events = load_fp32_model(
        protocol, "evaluation"
    )
    output = (
        ROOT
        / "experiments/p2_recovery/"
        "r1_amplitude_trust_region/results"
    )
    all_frames = {
        "scaling_rows": [],
        "sequence_scaling_metrics": [],
    }
    started = time.perf_counter()
    try:
        for sample in samples:
            checkpoint = (
                output / "checkpoints" / safe_id(sample.sample_id)
            )
            status_path = checkpoint / "status.json"
            complete = bool(
                args.resume
                and status_path.exists()
                and json.loads(status_path.read_text()).get("state")
                == "complete"
                and all(
                    (checkpoint / f"{name}.parquet").exists()
                    for name in all_frames
                )
            )
            if complete:
                frames = {
                    name: pd.read_parquet(
                        checkpoint / f"{name}.parquet"
                    )
                    for name in all_frames
                }
                event = "recovery_diagnostic_sequence_resumed"
            else:
                frames = run_sequence(
                    model, sample, protocol, checkpoint
                )
                event = "recovery_diagnostic_sequence_complete"
            for name, frame in frames.items():
                all_frames[name].append(frame)
            print(
                json.dumps(
                    {"event": event, "sample_id": sample.sample_id},
                    sort_keys=True,
                ),
                flush=True,
            )
            gc.collect()
        merged = {
            name: pd.concat(frames, ignore_index=True)
            for name, frames in all_frames.items()
        }
        if len(merged["scaling_rows"]) != int(
            r1["data"]["expected_rows"]
        ):
            raise RuntimeError("R1 row count mismatch")
        for name, frame in merged.items():
            atomic_frame(output / f"{name}.parquet", frame)
        r0_output = (
            ROOT
            / "experiments/p2_recovery/r0_failure_map/results"
        )
        r0_rows = merged["scaling_rows"].query(
            "gamma == 1.0"
        ).reset_index(drop=True)
        if len(r0_rows) != int(r0["data"]["expected_rows"]):
            raise RuntimeError("R0 row count mismatch")
        atomic_frame(r0_output / "r0_rows.parquet", r0_rows)
        metadata = {
            "completed": True,
            "r0_row_count": len(r0_rows),
            "r1_row_count": len(merged["scaling_rows"]),
            "sequence_metric_count": len(
                merged["sequence_scaling_metrics"]
            ),
            "r0_config_sha256": sha256_file(r0_path),
            "r1_config_sha256": sha256_file(r1_path),
            "p2_config_sha256": sha256_file(
                ROOT / "configs/frozen/p2_state_local_config.yaml"
            ),
            "model_info": model_info,
            "dataset_events": dataset_events,
            "wall_seconds": time.perf_counter() - started,
        }
        atomic_json(output / "diagnostic_metadata.json", metadata)
        atomic_json(r0_output / "r0_metadata.json", metadata)
        print(json.dumps(metadata, indent=2, sort_keys=True))
    finally:
        model.close()


if __name__ == "__main__":
    main()
