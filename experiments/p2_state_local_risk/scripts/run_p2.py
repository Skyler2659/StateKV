#!/usr/bin/env python3
"""Run P2 state-local risk closure and geometry attribution."""
from __future__ import annotations

import argparse
import copy
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
P1_DIR = ROOT / "experiments/p1_state_conditioned/scripts"
for value in (ROOT, ROOT / "benchmarks/torch", P0_DIR, P1_DIR, SCRIPT_DIR):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from statekv.repository_layout import (  # noqa: E402
    verify_repository_checksum,
)

from p0_v2_core import (  # noqa: E402
    AdjacentBoundaryMap,
    FixedBoundaryReadoutMap,
    full_replay,
)
from run_p0_v2 import (  # noqa: E402
    candidate_registry_rows,
    common_metadata,
    select_candidates,
    theoretical_pulse,
    verify_anchor_fp32,
)
from run_p1 import (  # noqa: E402
    _candidate_context,
    _finalize_vector_accumulator,
    _history_bundle,
    _new_vector_accumulator,
    _state_delta,
    _update_vector_accumulator,
    git_state,
    history_ids,
    load_fp32_model,
    safe_id,
)

from p2_core import (  # noqa: E402
    FACTORIAL_REGISTRY,
    atomic_frame,
    atomic_json,
    downstream_jvp_at,
    exact_kl,
    exact_kl_gradient,
    fisher_inner,
    fisher_variance,
    fisher_vector_product,
    geometry_scores,
    history_state_key,
    prefixed_metrics,
    probability_drift,
    required_reference_anchors,
    score_registry_rows,
    select_fd_radius,
    sha256_array,
    sha256_file,
    stable_softmax,
    state_local_symmetric_fd,
    validate_split_isolation,
    vector_metrics,
)


TABLE_NAMES = (
    "response_rows",
    "geometry_score_rows",
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
        "source_integrity",
        "model",
        "data",
        "cache",
        "history_conditions",
        "candidates",
        "geometry",
        "numeric",
        "metrics",
        "gates",
        "outcomes",
        "runtime",
    }
    missing = sorted(required - set(protocol))
    if missing:
        raise ValueError(f"P2 config missing fields: {missing}")
    expected_prohibited = {
        "state_prediction",
        "temporal_transition",
        "horizon_risk",
        "future_query",
        "future_attention",
        "refresh_controller",
        "detector",
        "low_rank_update",
        "candidate_subspace_method",
        "free_generation",
        "joint_multilayer_action",
        "physical_history_transfer",
        "new_selector",
        "task_level_benchmark",
        "online_policy",
        "post_formal_threshold_tuning",
    }
    if set(protocol["scope"]["prohibited"]) != expected_prohibited:
        raise ValueError("frozen P2 prohibited-scope list changed")
    configured = {
        name: tuple(values)
        for name, values in protocol["geometry"][
            "factorial_scores"
        ].items()
    }
    if configured != FACTORIAL_REGISTRY:
        raise ValueError("P2 factorial registry differs from core")
    return protocol


def stage_indices(
    protocol: Mapping[str, Any], stage: str
) -> Tuple[List[int], List[int]]:
    section = protocol["data"][stage]
    return (
        [int(value) for value in section["gov_report_indices"]],
        [int(value) for value in section["niah_offsets"]],
    )


def expected_ids(
    protocol: Mapping[str, Any], stage: str
) -> List[str]:
    gov, niah = stage_indices(protocol, stage)
    return [
        *[f"gov_report:{value}" for value in gov],
        *[f"synthetic_niah_{value}" for value in niah],
    ]


def _manifest_checks(path: Path) -> Tuple[Mapping[str, Any], Dict[str, bool]]:
    manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
    checks = {}
    for relative, expected in manifest["checksums"].items():
        checks[str(relative)] = verify_repository_checksum(
            ROOT, relative, str(expected)
        )
    return manifest, checks


def integrity_stage(
    protocol: Mapping[str, Any],
    config_path: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    """Verify immutable P0/P1 evidence before any P2 model run."""
    started = time.perf_counter()
    source_paths = {
        "p0": {
            "config": ROOT / "configs/frozen/p0_v2_config.yaml",
            "results": ROOT / "experiments/p0_v2_fixed_boundary/docs/results.md",
            "core": P0_DIR / "p0_v2_core.py",
            "runner": P0_DIR / "run_p0_v2.py",
        },
        "p1": {
            "config": ROOT / "configs/frozen/p1_state_conditioned_config.yaml",
            "results": ROOT / "experiments/p1_state_conditioned/docs/results.md",
            "core": P1_DIR / "p1_core.py",
            "runner": P1_DIR / "run_p1.py",
            "analyzer": P1_DIR / "analyze_p1.py",
            "state_operating_rows": ROOT
            / "experiments/p1_state_conditioned/results/"
            "state_operating_point_rows.parquet",
            "state_operating_summary": ROOT
            / "experiments/p1_state_conditioned/results/"
            "state_operating_point_summary.json",
        },
    }
    source_hash_checks: Dict[str, bool] = {}
    manifest_details: Dict[str, Any] = {}
    for source_name in ("p0", "p1"):
        source = protocol["source_integrity"][source_name]
        manifest_path = ROOT / str(source["manifest_path"])
        manifest, artifact_checks = _manifest_checks(manifest_path)
        source_hash_checks[f"{source_name}_manifest"] = (
            sha256_file(manifest_path) == str(source["manifest_sha256"])
        )
        for label, path in source_paths[source_name].items():
            source_hash_checks[f"{source_name}_{label}"] = (
                path.exists()
                and sha256_file(path)
                == str(source[f"{label}_sha256"])
            )
        manifest_details[source_name] = {
            "outcome": manifest["outcome"],
            "outcome_matches": str(manifest["outcome"])
            == str(source["outcome"]),
            "entry_count": len(artifact_checks),
            "entry_count_matches": len(artifact_checks)
            == int(source["manifest_entry_count"]),
            "all_entries_match": all(artifact_checks.values()),
            "matched_entries": int(sum(artifact_checks.values())),
        }
    tests = subprocess.run(
        [
            str(ROOT / ".venv/bin/python"),
            "-m",
            "pytest",
            "-q",
            str(ROOT / "tests/test_p0_v2.py"),
            str(ROOT / "tests/test_p1_state_conditioned.py"),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    split_audit = validate_split_isolation(protocol)
    scan = json.loads(
        (
            output_dir / "data_scan_summary.json"
        ).read_text(encoding="utf-8")
    )
    scan_ids = sorted(str(row["sample_id"]) for row in scan["rows"])
    frozen_ids = sorted(
        expected_ids(protocol, "calibration")
        + expected_ids(protocol, "evaluation")
    )
    checks = {
        "source_hashes_match": all(source_hash_checks.values()),
        "p0_manifest_all_match": manifest_details["p0"][
            "all_entries_match"
        ],
        "p1_manifest_all_match": manifest_details["p1"][
            "all_entries_match"
        ],
        "manifest_entry_counts_match": all(
            value["entry_count_matches"]
            for value in manifest_details.values()
        ),
        "outcomes_unchanged": all(
            value["outcome_matches"]
            for value in manifest_details.values()
        ),
        "p0_p1_tests_pass": tests.returncode == 0,
        "split_isolation": all(split_audit["checks"].values()),
        "mechanical_scan_matches_frozen_ids": scan_ids == frozen_ids,
        "scan_all_constructed": bool(
            scan.get("all_constructed", scan.get("all_pass", False))
        ),
    }
    summary = {
        "stage": "integrity",
        "passed": all(checks.values()),
        "checks": checks,
        "source_hash_checks": source_hash_checks,
        "manifests": manifest_details,
        "split_audit": split_audit,
        "tests_stdout": tests.stdout,
        "tests_stderr": tests.stderr,
        "config_sha256": sha256_file(config_path),
        "wall_seconds": time.perf_counter() - started,
    }
    atomic_json(output_dir / "integrity_summary.json", summary)
    atomic_json(
        output_dir / "source_freeze.json",
        {
            "config_sha256_at_integrity": sha256_file(config_path),
            "source_hash_checks": source_hash_checks,
            "p0_manifest_sha256": protocol["source_integrity"]["p0"][
                "manifest_sha256"
            ],
            "p1_manifest_sha256": protocol["source_integrity"]["p1"][
                "manifest_sha256"
            ],
        },
    )
    if not summary["passed"]:
        raise RuntimeError(f"P2 integrity failed: {checks}")
    return summary


def _candidate_protocol(
    protocol: Mapping[str, Any], run_id: str | None = None
) -> Dict[str, Any]:
    candidate_protocol = copy.deepcopy(dict(protocol))
    if run_id is not None:
        candidate_protocol["runtime"]["run_id"] = str(run_id)
    return candidate_protocol


def _state_geometry(
    downstream: Any,
    base_logits: np.ndarray,
    delta: np.ndarray,
) -> Dict[str, Any]:
    z0 = np.asarray(base_logits, dtype=np.float64)
    p0 = stable_softmax(z0)
    z_s = downstream.evaluate(delta)
    p_s = stable_softmax(z_s)
    _base, a0, state_method = downstream.jvp(delta)
    g0 = fisher_vector_product(p0, a0)
    gs = exact_kl_gradient(p0, p_s)
    return {
        "z0": z0,
        "p0": p0,
        "z_s": z_s,
        "p_s": p_s,
        "a0": a0,
        "g0": g0,
        "gs": gs,
        "state_method": state_method,
    }


def _action_geometry(
    downstream: Any,
    delta: np.ndarray,
    action_r: np.ndarray,
) -> Dict[str, Any]:
    _base, c0, reference_method = downstream.jvp(action_r)
    operating_output, cs, state_method = downstream_jvp_at(
        downstream, delta, action_r
    )
    combined_output = downstream.evaluate(delta + action_r)
    nonlinear = combined_output - operating_output
    return {
        "c0": c0,
        "cs": cs,
        "operating_output": operating_output,
        "combined_output": combined_output,
        "nonlinear": nonlinear,
        "reference_method": reference_method,
        "state_method": state_method,
    }


def smoke_stage(
    protocol: Mapping[str, Any],
    config_path: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    integrity = json.loads(
        (output_dir / "integrity_summary.json").read_text()
    )
    if not integrity["passed"]:
        raise RuntimeError("passing P2 integrity prerequisite is missing")
    started = time.perf_counter()
    model, model_info, samples, dataset_events = load_fp32_model(
        protocol, "smoke"
    )
    section = protocol["data"]["smoke"]
    target = int(section["target_anchors"][0])
    layer = int(section["layers"][0])
    sources = list(section["candidate_sources"])
    p1_rows = pd.read_parquet(
        ROOT
        / "experiments/p1_state_conditioned/results/"
        "state_operating_point_rows.parquet"
    )
    rows: List[Dict[str, Any]] = []
    state_rows: List[Dict[str, Any]] = []
    try:
        for sample in samples:
            reference = model.generate_reference(
                sample.sample_id, sample.task, sample.prompt
            )
            base_logits, base_record, base_positions, _dtypes = full_replay(
                model, reference, target
            )
            anchor = reference.anchors[target]
            candidate_protocol = _candidate_protocol(
                protocol, section["candidate_run_id"]
            )
            all_candidates, _registry = _candidate_context(
                model, reference, target, candidate_protocol, None
            )
            candidates = select_candidates(all_candidates, sources)
            histories = _history_bundle(
                model,
                reference,
                candidate_protocol,
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
            for history_id, observation in histories.items():
                delta = _state_delta(observation, base_record, boundary)
                state = _state_geometry(downstream, base_logits, delta)
                state_rows.append(
                    {
                        "sample_id": sample.sample_id,
                        "task": sample.task,
                        "anchor": target,
                        "layer": layer,
                        "history_id": history_id,
                        "state_norm": float(np.linalg.norm(delta)),
                        "state_gradient_norm": float(
                            np.linalg.norm(state["gs"])
                        ),
                        **probability_drift(state["p0"], state["p_s"]),
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
                    action = _action_geometry(
                        downstream, delta, action_r
                    )
                    readout = vector_metrics(
                        action["cs"], action["nonlinear"]
                    )
                    total_predicted = (
                        state["z_s"] - base_logits + action["cs"]
                    )
                    total_truth = (
                        action["combined_output"] - base_logits
                    )
                    p1_total = vector_metrics(
                        total_predicted, total_truth
                    )
                    scores = geometry_scores(
                        reference_probability=state["p0"],
                        state_probability=state["p_s"],
                        reference_linear_gradient=state["g0"],
                        state_local_gradient=state["gs"],
                        reference_action_direction=action["c0"],
                        state_local_action_direction=action["cs"],
                        nonlinear_action_direction=action["nonlinear"],
                    )
                    p1_match = p1_rows[
                        (p1_rows["sample_id"] == sample.sample_id)
                        & (p1_rows["anchor"] == target)
                        & (p1_rows["layer"] == layer)
                        & (p1_rows["history_id"] == history_id)
                        & (
                            p1_rows["candidate_source"]
                            == candidate.source
                        )
                    ]
                    if len(p1_match) != 1:
                        raise RuntimeError(
                            "P1 smoke regression row is not unique"
                        )
                    old = p1_match.iloc[0]
                    comparisons = {
                        "state_norm": float(np.linalg.norm(delta)),
                        "action_norm": float(np.linalg.norm(action_r)),
                        "state_local_vs_manual_cosine": p1_total[
                            "cosine"
                        ],
                        "state_local_vs_manual_relative_l2": p1_total[
                            "relative_l2"
                        ],
                    }
                    differences = {
                        f"p1_regression_{name}_absolute_error": abs(
                            float(value) - float(old[name])
                        )
                        for name, value in comparisons.items()
                    }
                    rows.append(
                        {
                            **common,
                            "history_id": history_id,
                            "state_norm": comparisons["state_norm"],
                            "action_norm": comparisons["action_norm"],
                            "state_gradient_norm": float(
                                np.linalg.norm(state["gs"])
                            ),
                            "state_operating_point_output_max_error": float(
                                np.max(
                                    np.abs(
                                        action["operating_output"]
                                        - state["z_s"]
                                    )
                                )
                            ),
                            **prefixed_metrics(
                                "state_local_vs_nonlinear",
                                action["cs"],
                                action["nonlinear"],
                            ),
                            **prefixed_metrics(
                                "p1_total_state_local_vs_manual",
                                total_predicted,
                                total_truth,
                            ),
                            **differences,
                            **{
                                f"score_{name}": value
                                for name, value in scores.items()
                            },
                        }
                    )
            del reference
            gc.collect()
        frame = pd.DataFrame(rows)
        states = pd.DataFrame(state_rows)
        atomic_frame(output_dir / "smoke_rows.parquet", frame)
        atomic_frame(output_dir / "smoke_state_rows.parquet", states)
        tolerance = float(
            protocol["gates"]["integrity"][
                "p1_smoke_metric_absolute_tolerance"
            ]
        )
        regression_columns = [
            column
            for column in frame.columns
            if column.startswith("p1_regression_")
        ]
        h0 = frame[frame["history_id"] == "H0"]
        h0_states = states[states["history_id"] == "H0"]
        checks = {
            "finite": bool(
                np.isfinite(
                    frame.select_dtypes(include=[np.number]).to_numpy()
                ).all()
            ),
            "p1_state_local_diagnostic_reproduced": float(
                frame[regression_columns].max().max()
            )
            <= tolerance,
            "h0_state_zero": float(h0["state_norm"].max()) == 0.0,
            "h0_probability_identity": float(
                h0_states["probability_total_variation"].max()
            )
            == 0.0,
            "h0_gradient_zero": float(
                h0["state_gradient_norm"].max()
            )
            == 0.0,
            "h0_full_equals_action": float(
                (
                    h0["score_full_state_local"]
                    - h0["score_reference_action_fisher"]
                )
                .abs()
                .max()
            )
            <= float(
                protocol["gates"]["integrity"][
                    "h0_score_max_absolute_error"
                ]
            ),
            "operating_point_baseline_identity": float(
                frame["state_operating_point_output_max_error"].max()
            )
            <= float(
                protocol["numeric"][
                    "baseline_max_absolute_error_max"
                ]
            ),
        }
        summary = {
            "stage": "smoke",
            "passed": all(checks.values()),
            "checks": checks,
            "row_count": len(frame),
            "state_row_count": len(states),
            "maximum_p1_regression_absolute_error": float(
                frame[regression_columns].max().max()
            ),
            "stale_state_local_readout_median_cosine": float(
                frame.loc[
                    frame["history_id"] != "H0",
                    "state_local_vs_nonlinear_cosine",
                ].median()
            ),
            "stale_state_local_readout_median_relative_l2": float(
                frame.loc[
                    frame["history_id"] != "H0",
                    "state_local_vs_nonlinear_relative_l2",
                ].median()
            ),
            "model_info": model_info,
            "dataset_events": dataset_events,
            "config_sha256": sha256_file(config_path),
            "wall_seconds": time.perf_counter() - started,
        }
        atomic_json(output_dir / "smoke_summary.json", summary)
        if not summary["passed"]:
            raise RuntimeError(f"P2 smoke failed: {checks}")
        return summary
    finally:
        model.close()


def calibration_stage(
    protocol: Mapping[str, Any],
    config_path: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    smoke = json.loads((output_dir / "smoke_summary.json").read_text())
    if not smoke["passed"]:
        raise RuntimeError("passing P2 smoke prerequisite is missing")
    started = time.perf_counter()
    model, model_info, samples, dataset_events = load_fp32_model(
        protocol, "calibration"
    )
    section = protocol["data"]["calibration"]
    rows: List[Dict[str, Any]] = []
    state_rows: List[Dict[str, Any]] = []
    try:
        for sample in samples:
            reference = model.generate_reference(
                sample.sample_id, sample.task, sample.prompt
            )
            previous_attention_core = None
            for target in [
                int(value) for value in section["target_anchors"]
            ]:
                all_candidates, registry = _candidate_context(
                    model,
                    reference,
                    target,
                    protocol,
                    previous_attention_core,
                )
                previous_attention_core = registry["attention_core"]
                candidates = select_candidates(
                    all_candidates, section["candidate_sources"]
                )
                (
                    base_logits,
                    base_record,
                    base_positions,
                    _dtypes,
                ) = full_replay(model, reference, target)
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
                    adjacent = AdjacentBoundaryMap(
                        model, layer, base_record
                    )
                    downstream = FixedBoundaryReadoutMap(
                        model, anchor, base_record, boundary
                    )
                    deltas = {
                        history_id: _state_delta(
                            observation, base_record, boundary
                        )
                        for history_id, observation in histories.items()
                    }
                    for history_id, delta in deltas.items():
                        state_rows.append(
                            {
                                "sample_id": sample.sample_id,
                                "task": sample.task,
                                "anchor": target,
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
                                "state_norm": float(
                                    np.linalg.norm(delta)
                                ),
                                "finite": bool(
                                    np.isfinite(delta).all()
                                ),
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
                            protocol["numeric"][
                                "identity_norm_floors"
                            ],
                            common,
                        )
                        _out, action_r, _method = adjacent.jvp(action_u)
                        for history_id, delta in deltas.items():
                            operating_output, predicted, method = (
                                downstream_jvp_at(
                                    downstream, delta, action_r
                                )
                            )
                            state_output = downstream.evaluate(delta)
                            for radius in protocol["numeric"][
                                "fd_relative_radius_grid"
                            ]:
                                fd = state_local_symmetric_fd(
                                    downstream,
                                    delta,
                                    action_r,
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
                                        "jvp_method": method,
                                        "state_operating_point_output_max_error": float(
                                            np.max(
                                                np.abs(
                                                    operating_output
                                                    - state_output
                                                )
                                            )
                                        ),
                                        "epsilon_relative": float(
                                            radius
                                        ),
                                        "epsilon_absolute": fd[
                                            "epsilon_absolute"
                                        ],
                                        "state_workpoint_norm": fd[
                                            "state_workpoint_norm"
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
                        "event": "p2_calibration_sequence_complete",
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
        direction_count = int(
            frame[
                [
                    "sample_id",
                    "anchor",
                    "layer",
                    "candidate_id",
                    "history_id",
                ]
            ]
            .drop_duplicates()
            .shape[0]
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
            "operating_point_identity": float(
                frame["state_operating_point_output_max_error"].max()
            )
            <= float(
                protocol["numeric"][
                    "baseline_max_absolute_error_max"
                ]
            ),
            "direction_count": direction_count
            == int(section["expected_directions"]),
            "radius_row_count": len(frame)
            == int(section["expected_radius_rows"]),
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
            "direction_count": direction_count,
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
            raise RuntimeError("P2 state-local calibration failed")
        return summary
    finally:
        model.close()


def _fisher_geometry_drift(
    p0: np.ndarray,
    ps: np.ndarray,
    c0: np.ndarray,
    cs: np.ndarray,
) -> Dict[str, float]:
    f0_c0 = fisher_vector_product(p0, c0)
    fs_cs = fisher_vector_product(ps, cs)
    return {
        "reference_action_fisher_energy": 0.5
        * fisher_variance(p0, c0),
        "state_action_fisher_energy": 0.5
        * fisher_variance(ps, cs),
        "fisher_vector_drift_norm": float(
            np.linalg.norm(fs_cs - f0_c0)
        ),
        "jacobian_fisher_weighted_discrepancy_reference": (
            fisher_variance(p0, cs - c0) ** 0.5
        ),
        "jacobian_fisher_weighted_discrepancy_state": (
            fisher_variance(ps, cs - c0) ** 0.5
        ),
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
    atomic_json(
        sequence_dir / "status.json",
        {
            "state": "running",
            "sample_id": sample.sample_id,
            "task": sample.task,
            "split": "evaluation",
        },
    )
    reference = model.generate_reference(
        sample.sample_id, sample.task, sample.prompt
    )
    section = protocol["data"]["evaluation"]
    targets = [int(value) for value in section["target_anchors"]]
    layers = [int(value) for value in section["layers"]]
    missing = [
        value
        for value in required_reference_anchors(
            targets, protocol["history_conditions"]
        )
        if value not in reference.anchors
    ]
    if missing:
        raise RuntimeError(f"missing P2 reference anchors: {missing}")
    vector_accumulators = {
        key: _new_vector_accumulator()
        for key in ["all", "primary", *history_ids(protocol)]
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
        (
            base_logits,
            base_record,
            base_positions,
            base_dtypes,
        ) = full_replay(model, reference, target)
        repeated_logits, _record, _positions, _dtypes = full_replay(
            model, reference, target
        )
        repeat_error = float(
            np.max(np.abs(base_logits - repeated_logits))
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
            state_cache: Dict[str, Dict[str, Any]] = {}
            for history_id, observation in histories.items():
                delta = _state_delta(
                    observation, base_record, boundary
                )
                state = _state_geometry(
                    downstream, base_logits, delta
                )
                state_key = history_state_key(
                    sample.sample_id,
                    target,
                    boundary,
                    history_id,
                    delta,
                )
                gradient_metrics = vector_metrics(
                    state["g0"], state["gs"]
                )
                drift = probability_drift(
                    state["p0"], state["p_s"]
                )
                state.update(
                    {
                        "delta": delta,
                        "state_hash": state_key,
                        "gradient_metrics": gradient_metrics,
                        "probability_drift": drift,
                    }
                )
                state_cache[history_id] = state
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
                        "state_finite": bool(
                            np.isfinite(delta).all()
                        ),
                        "reference_linear_state_jvp_norm": float(
                            np.linalg.norm(state["a0"])
                        ),
                        "reference_linear_gradient_norm": float(
                            np.linalg.norm(state["g0"])
                        ),
                        "state_gradient_norm": float(
                            np.linalg.norm(state["gs"])
                        ),
                        "state_fisher_energy": 0.5
                        * fisher_variance(
                            state["p0"], state["a0"]
                        ),
                        "controlled_history_kl": exact_kl(
                            base_logits, state["z_s"]
                        ),
                        "physical_history_kl": exact_kl(
                            base_logits, observation.logits
                        ),
                        "state_output_norm": float(
                            np.linalg.norm(
                                state["z_s"] - base_logits
                            )
                        ),
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
                        **drift,
                        **{
                            f"gradient_reference_vs_state_{key}": value
                            for key, value in gradient_metrics.items()
                        },
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
                _out, action_r, adjacent_method = adjacent.jvp(action_u)
                for history_id in history_ids(protocol):
                    state = state_cache[history_id]
                    action = _action_geometry(
                        downstream, state["delta"], action_r
                    )
                    scores = geometry_scores(
                        reference_probability=state["p0"],
                        state_probability=state["p_s"],
                        reference_linear_gradient=state["g0"],
                        state_local_gradient=state["gs"],
                        reference_action_direction=action["c0"],
                        state_local_action_direction=action["cs"],
                        nonlinear_action_direction=action["nonlinear"],
                    )
                    controlled_exact = exact_kl(
                        base_logits, action["combined_output"]
                    )
                    readout = vector_metrics(
                        action["cs"], action["nonlinear"]
                    )
                    jacobian_drift = vector_metrics(
                        action["c0"], action["cs"]
                    )
                    fisher_drift = _fisher_geometry_drift(
                        state["p0"],
                        state["p_s"],
                        action["c0"],
                        action["cs"],
                    )
                    state_output_error = float(
                        np.max(
                            np.abs(
                                action["operating_output"]
                                - state["z_s"]
                            )
                        )
                    )
                    for key in ("all", history_id):
                        _update_vector_accumulator(
                            vector_accumulators[key],
                            action["cs"],
                            action["nonlinear"],
                        )
                    if history_id in protocol["metrics"][
                        "primary_histories"
                    ]:
                        _update_vector_accumulator(
                            vector_accumulators["primary"],
                            action["cs"],
                            action["nonlinear"],
                        )
                    h0_score_error = abs(
                        scores["full_state_local"]
                        - scores["reference_action_fisher"]
                    )
                    response_row = {
                        **common,
                        "history_id": history_id,
                        "history_length": histories[
                            history_id
                        ].history_length,
                        "state_hash": state["state_hash"],
                        "state_norm": float(
                            np.linalg.norm(state["delta"])
                        ),
                        "action_u_norm": float(
                            np.linalg.norm(action_u)
                        ),
                        "action_r_norm": float(
                            np.linalg.norm(action_r)
                        ),
                        "reference_action_logit_norm": float(
                            np.linalg.norm(action["c0"])
                        ),
                        "state_action_logit_norm": float(
                            np.linalg.norm(action["cs"])
                        ),
                        "nonlinear_action_logit_norm": float(
                            np.linalg.norm(action["nonlinear"])
                        ),
                        "reference_linear_gradient_norm": float(
                            np.linalg.norm(state["g0"])
                        ),
                        "state_gradient_norm": float(
                            np.linalg.norm(state["gs"])
                        ),
                        "state_fisher_energy": 0.5
                        * fisher_variance(
                            state["p0"], state["a0"]
                        ),
                        "controlled_history_kl": exact_kl(
                            base_logits, state["z_s"]
                        ),
                        "physical_history_kl": exact_kl(
                            base_logits,
                            histories[history_id].logits,
                        ),
                        "controlled_exact_kl": controlled_exact,
                        "state_operating_point_output_max_error": (
                            state_output_error
                        ),
                        "h0_full_action_score_absolute_error": (
                            h0_score_error
                        ),
                        "adjacent_method": adjacent_method,
                        "reference_jvp_method": action[
                            "reference_method"
                        ],
                        "state_jvp_method": action["state_method"],
                        **state["probability_drift"],
                        **{
                            f"gradient_reference_vs_state_{key}": value
                            for key, value in state[
                                "gradient_metrics"
                            ].items()
                        },
                        **{
                            f"state_local_readout_{key}": value
                            for key, value in readout.items()
                        },
                        **{
                            f"jacobian_reference_vs_state_{key}": value
                            for key, value in jacobian_drift.items()
                        },
                        **fisher_drift,
                        **{
                            f"score_{name}": value
                            for name, value in scores.items()
                        },
                    }
                    tables["response_rows"].append(response_row)
                    registry_rows = score_registry_rows()
                    for score_name, score_value in scores.items():
                        registry_row = registry_rows[score_name]
                        tables["geometry_score_rows"].append(
                            {
                                **common,
                                "history_id": history_id,
                                "state_hash": state["state_hash"],
                                "score_type": score_name,
                                "score": float(score_value),
                                "controlled_exact_kl": controlled_exact,
                                **registry_row,
                            }
                        )
            h0 = [
                row
                for row in tables["response_rows"]
                if row["sample_id"] == sample.sample_id
                and row["anchor"] == target
                and row["layer"] == layer
                and row["history_id"] == "H0"
            ]
            unit_rows = [
                row
                for row in tables["response_rows"]
                if row["sample_id"] == sample.sample_id
                and row["anchor"] == target
                and row["layer"] == layer
            ]
            tables["unit_audit"].append(
                {
                    "sample_id": sample.sample_id,
                    "task": sample.task,
                    "split": "evaluation",
                    "anchor": target,
                    "layer": layer,
                    "boundary_layer": boundary,
                    "candidate_count": len(candidates),
                    "candidate_distinct_count": len(
                        {candidate.mask_hash for candidate in candidates}
                    ),
                    "history_count": len(histories),
                    "response_row_count": len(unit_rows),
                    "repeat_max_absolute_error": repeat_error,
                    "boundary_map_baseline_cosine": baseline_metrics[
                        "cosine"
                    ],
                    "boundary_map_baseline_relative_l2": (
                        baseline_metrics["relative_l2"]
                    ),
                    "state_operating_point_output_max_error": max(
                        row[
                            "state_operating_point_output_max_error"
                        ]
                        for row in unit_rows
                    ),
                    "h0_state_norm_max": max(
                        row["state_norm"] for row in h0
                    ),
                    "h0_gradient_norm_max": max(
                        row["state_gradient_norm"] for row in h0
                    ),
                    "h0_probability_drift_max": max(
                        row["probability_total_variation"]
                        for row in h0
                    ),
                    "h0_score_identity_max_error": max(
                        row["h0_full_action_score_absolute_error"]
                        for row in h0
                    ),
                    "candidate_shared_state_hash": all(
                        len(
                            {
                                row["state_hash"]
                                for row in unit_rows
                                if row["history_id"] == history_id
                            }
                        )
                        == 1
                        for history_id in history_ids(protocol)
                    ),
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
                    "event": "p2_target_complete",
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
    atomic_json(
        sequence_dir / "status.json",
        {
            "state": "complete",
            "sample_id": sample.sample_id,
            "task": sample.task,
            "split": "evaluation",
            "wall_seconds": time.perf_counter() - started,
            "prompt_length": reference.prompt_length,
            "prompt_truncated": reference.prompt_truncated,
            "targets": targets,
            "layers": layers,
            "response_rows": len(tables["response_rows"]),
        },
    )
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


def evaluation_stage(
    protocol: Mapping[str, Any],
    config_path: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    started = time.perf_counter()
    integrity = json.loads(
        (output_dir / "integrity_summary.json").read_text()
    )
    smoke = json.loads((output_dir / "smoke_summary.json").read_text())
    calibration_path = output_dir / "calibration_summary.json"
    calibration = json.loads(calibration_path.read_text())
    if not (
        integrity["passed"]
        and smoke["passed"]
        and calibration["passed"]
    ):
        raise RuntimeError("formal P2 prerequisite failed")
    selected = protocol["numeric"]["fd_selected_relative_radius"]
    if selected is None or not np.isclose(
        float(selected),
        float(calibration["selected_relative_radius"]),
        rtol=0.0,
        atol=0.0,
    ):
        raise RuntimeError("P2 calibration radius is not frozen")
    if protocol["numeric_calibration_status"] != "frozen":
        raise RuntimeError("P2 numeric calibration status is not frozen")
    split_audit = validate_split_isolation(protocol)
    config_hash = sha256_file(config_path)
    atomic_json(
        output_dir / "evaluation_freeze.json",
        {
            "evaluation_started_at_unix": time.time(),
            "config_sha256": config_hash,
            "selected_relative_radius": float(selected),
            "calibration_summary_sha256": sha256_file(
                calibration_path
            ),
            "split_audit": split_audit,
            "source_freeze_sha256": sha256_file(
                output_dir / "source_freeze.json"
            ),
            "git": git_state(),
        },
    )
    model, model_info, samples, dataset_events = load_fp32_model(
        protocol, "evaluation"
    )
    status: Dict[str, Any] = {
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
                event = "p2_sequence_resumed"
            else:
                tables = run_evaluation_sequence(
                    model, sample, protocol, sequence_dir
                )
                event = "p2_sequence_complete"
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
                f"P2 response rows {len(frames['response_rows'])} "
                f"!= {expected_rows}"
            )
        expected_scores = expected_rows * len(score_registry_rows())
        if len(frames["geometry_score_rows"]) != expected_scores:
            raise RuntimeError(
                f"P2 geometry rows "
                f"{len(frames['geometry_score_rows'])} "
                f"!= {expected_scores}"
            )
        for name, frame in frames.items():
            atomic_frame(output_dir / f"{name}.parquet", frame)
        registry_frame = pd.DataFrame(
            [
                {"score_type": name, **values}
                for name, values in score_registry_rows().items()
            ]
        )
        atomic_frame(
            output_dir / "score_registry.parquet", registry_frame
        )
        if sha256_file(config_path) != config_hash:
            raise RuntimeError("P2 config changed during evaluation")
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
                **{
                    name: len(frame)
                    for name, frame in frames.items()
                },
                "score_registry": len(registry_frame),
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


def dry_run(
    protocol: Mapping[str, Any], config_path: Path
) -> Dict[str, Any]:
    split = validate_split_isolation(protocol)
    return {
        "experiment": protocol["experiment"],
        "config_sha256": sha256_file(config_path),
        "stage_ids": {
            stage: expected_ids(protocol, stage)
            for stage in ("smoke", "calibration", "evaluation")
        },
        "split_audit": split,
        "factorial_registry": FACTORIAL_REGISTRY,
        "score_registry": score_registry_rows(),
        "expected_calibration_rows": int(
            protocol["data"]["calibration"]["expected_radius_rows"]
        ),
        "expected_formal_rows": int(
            protocol["data"]["evaluation"][
                "expected_candidate_history_rows"
            ]
        ),
        "expected_geometry_score_rows": int(
            protocol["data"]["evaluation"][
                "expected_candidate_history_rows"
            ]
        )
        * len(score_registry_rows()),
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
        default=ROOT / "configs/frozen/p2_state_local_config.yaml",
    )
    parser.add_argument(
        "--stage",
        choices=[
            "dry-run",
            "integrity",
            "smoke",
            "calibration",
            "evaluation",
        ],
        required=True,
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    protocol = load_protocol(config_path)
    output_dir = ROOT / protocol["runtime"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "dry-run":
        result = dry_run(protocol, config_path)
    elif args.stage == "integrity":
        result = integrity_stage(protocol, config_path, output_dir)
    elif args.stage == "smoke":
        result = smoke_stage(protocol, config_path, output_dir)
    elif args.stage == "calibration":
        result = calibration_stage(protocol, config_path, output_dir)
    else:
        result = evaluation_stage(protocol, config_path, output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
