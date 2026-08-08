"""Post-attention multi-boundary shared-VJP development pilot."""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import yaml

from statekv.candidate_pullback import (
    CandidatePullbackRunner,
    PurePostAttentionTailMap,
)
from statekv.config import load_discovery_config
from statekv.fisher_pullback import fisher_output_random_direction
from statekv.functional_probe import _condition_cache
from statekv.shared_jvp_pilot import _source_frames, _vectors
from statekv.storage import atomic_frame, atomic_json, atomic_text
from statekv.tasks import load_discovery_tasks
from statekv.training_free_analysis import _decision_metrics
from statekv.training_free_routes import deletion_output, softmax, vjp_action_scores
from statekv.vjp_routes_pilot import _summarize


def _seed(*parts: Any) -> int:
    token = ":".join(str(value) for value in parts)
    return int.from_bytes(hashlib.sha256(token.encode("utf-8")).digest()[:8], "little")


def _inventory(
    source_root: Path,
    sample_id: str,
    anchor: int,
    candidate_ids: Sequence[str],
) -> Mapping[str, Mapping[str, List[int]]]:
    frame = pd.read_parquet(source_root / "independent_candidate_inventory.parquet")
    frame = frame[
        (frame["sample_id"].astype(str) == str(sample_id))
        & (frame["anchor"] == int(anchor))
        & frame["candidate_id"].astype(str).isin(set(candidate_ids))
    ]
    if frame["candidate_id"].nunique() != len(candidate_ids):
        raise RuntimeError("candidate inventory is incomplete")
    return {
        str(row["candidate_id"]): json.loads(str(row["selected_positions_json"]))
        for row in frame.to_dict("records")
    }


def _direct_actions(
    runner: CandidatePullbackRunner,
    record: Any,
    layer: int,
    values: np.ndarray,
    candidate_ids: Sequence[str],
    selections: Mapping[str, Mapping[str, List[int]]],
    sink_size: int,
    recent_size: int,
) -> np.ndarray:
    attention = (
        record.all_head_attention_distributions[int(layer)]
        .detach()
        .float()
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    full = (
        record.all_head_attention_outputs[int(layer)]
        .detach()
        .float()
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    if attention.shape[1] != values.shape[1]:
        raise RuntimeError("attention/value alignment failed")
    positions = list(range(int(values.shape[1])))
    mandatory = positions[: int(sink_size)] + (
        positions[-int(recent_size) :] if int(recent_size) else []
    )
    actions: List[np.ndarray] = []
    for candidate_id in candidate_ids:
        core = [int(value) for value in selections[candidate_id][str(layer)]]
        retained = sorted(set(core + mandatory))
        compressed = deletion_output(attention, values, retained)
        projected = runner.model.project_features(
            int(layer),
            torch.from_numpy(
                (compressed - full).astype(np.float32).reshape(1, -1)
            ),
        )[0]
        actions.append(projected.float().cpu().numpy().astype(np.float64))
    return np.stack(actions, axis=0)


def run_multiboundary_vjp_pilot(
    config_path: Path, repository_root: Path
) -> Path:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    cfg = load_discovery_config(str(repository_root / str(config["base_config"])))
    source_root = repository_root / str(config["source_run"])
    output_root = repository_root / str(config["output_run"])
    output_root.mkdir(parents=True, exist_ok=True)
    sample_ids = set(str(value) for value in config["sample_ids"])
    anchors = [int(value) for value in config["anchors"]]
    layers = [int(value) for value in config["layers"]]
    horizon = int(config["horizon"])
    sources = [str(value) for value in config["candidate_sources"]]
    widths = sorted(set(int(value) for value in config["widths"]))
    maximum_width = max(widths)
    intervals = sorted(set(int(value) for value in config["refresh_intervals"]))
    baseline = str(config["baseline"])
    sink_size = int(config["sink_size"])
    recent_size = int(config["recent_size"])

    runner = CandidatePullbackRunner(cfg, repository_root)
    samples, _ = load_discovery_tasks(cfg)
    selected_samples = [
        sample for sample in samples if str(sample.sample_id) in sample_ids
    ]
    if {str(sample.sample_id) for sample in selected_samples} != sample_ids:
        raise RuntimeError("configured multi-boundary samples were not loaded")

    score_rows: List[Dict[str, Any]] = []
    diagnostic_rows: List[Dict[str, Any]] = []
    reverse_passes = 0
    started = time.perf_counter()
    runner.model.load()
    try:
        for sample in selected_samples:
            reference = runner.model.generate_reference(
                sample.sample_id, sample.task, sample.prompt
            )
            geometry, index, arrays = _source_frames(
                source_root, str(sample.sample_id), sources
            )
            try:
                for anchor in anchors:
                    anchor_state = reference.anchors[int(anchor)]
                    full_selection = runner._all_history_selection(reference, anchor)
                    full_state, _ = runner.model.state_from_anchor(
                        anchor_state,
                        full_selection,
                        cache_config=_condition_cache(
                            cfg, int(anchor_state.logical_length) + horizon + 2, 1
                        ),
                    )
                    current_token = int(anchor_state.query_token_id)
                    values_by_layer = {
                        layer: (
                            anchor_state.values[layer][0]
                            .detach()
                            .float()
                            .cpu()
                            .numpy()
                            .astype(np.float64)
                        )
                        for layer in layers
                    }
                    steps: Dict[int, Dict[str, Any]] = {}
                    try:
                        for offset in range(1, horizon + 1):
                            target_index = int(anchor + offset - 1)
                            record = reference.query_records[target_index]
                            if offset > 1:
                                kv_heads = int(
                                    runner.model.model_info["num_key_value_heads"]
                                )
                                for layer in layers:
                                    appended = (
                                        runner._record_values(record, layer, kv_heads)
                                        .float()
                                        .cpu()
                                        .numpy()
                                        .astype(np.float64)[:, None, :]
                                    )
                                    values_by_layer[layer] = np.concatenate(
                                        [values_by_layer[layer], appended], axis=1
                                    )
                            current = geometry[
                                (geometry["anchor"] == anchor)
                                & (geometry["horizon_offset"] == offset)
                            ].sort_values("candidate_index", kind="stable")
                            candidate_ids = list(current["candidate_id"].astype(str))
                            if len(candidate_ids) != len(sources):
                                raise RuntimeError("multi-boundary candidate pool incomplete")
                            selections = _inventory(
                                source_root,
                                str(sample.sample_id),
                                anchor,
                                candidate_ids,
                            )
                            actions = {
                                layer: _direct_actions(
                                    runner,
                                    record,
                                    layer,
                                    values_by_layer[layer],
                                    candidate_ids,
                                    selections,
                                    sink_size,
                                    recent_size,
                                )
                                for layer in layers
                            }
                            _, stored_actions = _vectors(index, arrays, current)
                            reconstructed = actions[27]
                            reconstruction_error = float(
                                np.linalg.norm(reconstructed - stored_actions)
                                / max(np.linalg.norm(stored_actions), 1.0e-12)
                            )
                            gradients: Dict[int, np.ndarray] = {}
                            for layer in layers:
                                point = (
                                    record.post_attention_residuals[layer]
                                    .detach()
                                    .float()
                                    .cpu()
                                    .numpy()
                                    .reshape(-1)
                                    .astype(np.float64)
                                )
                                tail = PurePostAttentionTailMap(
                                    runner.model, full_state.cache, layer
                                )
                                logits = tail.evaluate(point)
                                probability = softmax(logits)
                                rng = np.random.default_rng(
                                    _seed(
                                        config["random_seed"],
                                        sample.sample_id,
                                        anchor,
                                        offset,
                                        layer,
                                    )
                                )
                                directions = []
                                probe_started = time.perf_counter()
                                for _ in range(maximum_width):
                                    raw = rng.choice(
                                        np.asarray([-1.0, 1.0]),
                                        size=probability.size,
                                    )
                                    cotangent = fisher_output_random_direction(
                                        probability, raw
                                    )
                                    _, gradient = tail.vjp(point, cotangent)
                                    directions.append(gradient)
                                    reverse_passes += 1
                                gradients[layer] = np.column_stack(directions)
                                reference_logits = (
                                    reference.probe_logits[target_index]
                                    .float()
                                    .numpy()
                                    .astype(np.float64)
                                )
                                adjoint_error = float("nan")
                                best_radius = float("nan")
                                if offset == 1:
                                    check_cotangent = fisher_output_random_direction(
                                        probability,
                                        rng.choice(
                                            np.asarray([-1.0, 1.0]),
                                            size=probability.size,
                                        ),
                                    )
                                    _, check_gradient = tail.vjp(
                                        point, check_cotangent
                                    )
                                    reverse_passes += 1
                                    left = float(
                                        np.dot(check_gradient, actions[layer][0])
                                    )
                                    checks = []
                                    for radius in config["adjoint_fd_radii"]:
                                        finite = tail.symmetric_fd(
                                            point,
                                            actions[layer][0],
                                            float(radius),
                                            center_output=logits,
                                        )
                                        right = float(
                                            np.dot(
                                                check_cotangent,
                                                finite["derivative"],
                                            )
                                        )
                                        checks.append(
                                            (
                                                abs(left - right)
                                                / max(abs(left), abs(right), 1.0e-12),
                                                float(radius),
                                            )
                                        )
                                    adjoint_error, best_radius = min(checks)
                                diagnostic_rows.append(
                                    {
                                        "sample_id": str(sample.sample_id),
                                        "task": str(sample.task),
                                        "anchor": anchor,
                                        "horizon_offset": offset,
                                        "layer": layer,
                                        "reference_relative_error": float(
                                            np.linalg.norm(logits - reference_logits)
                                            / max(
                                                np.linalg.norm(reference_logits),
                                                1.0e-12,
                                            )
                                        ),
                                        "adjoint_relative_error": adjoint_error,
                                        "adjoint_best_fd_radius": best_radius,
                                        "probe_elapsed_s": float(
                                            time.perf_counter() - probe_started
                                        ),
                                        "layer27_action_reconstruction_relative_error": (
                                            reconstruction_error
                                            if layer == 27
                                            else float("nan")
                                        ),
                                    }
                                )
                            steps[offset] = {
                                "metadata": current.reset_index(drop=True),
                                "candidate_ids": candidate_ids,
                                "actions": actions,
                                "gradients": gradients,
                            }
                            if offset < horizon:
                                runner._clear_controls()
                                _, replay_record, _ = runner.model.forward_one(
                                    full_state, current_token, capture_attention=True
                                )
                                if int(replay_record.query_position) != int(
                                    record.query_position
                                ):
                                    raise RuntimeError("full replay is position-misaligned")
                                current_token = int(
                                    reference.generated_token_ids[target_index]
                                )
                    finally:
                        runner.model.release(full_state)

                    for interval in intervals:
                        for offset in range(1, horizon + 1):
                            step = steps[offset]
                            refresh = 1 + ((offset - 1) // interval) * interval
                            probe = steps[refresh]
                            truth = step["metadata"]["exact_kl"].to_numpy(
                                dtype=np.float64
                            )
                            oracle = step["metadata"]["g3_midpoint_fisher"].to_numpy(
                                dtype=np.float64
                            )
                            for width in widths:
                                per_layer_scores = {
                                    layer: vjp_action_scores(
                                        step["actions"][layer],
                                        probe["gradients"][layer][:, :width],
                                    )
                                    for layer in layers
                                }
                                stacked = np.stack(
                                    [per_layer_scores[layer] for layer in layers],
                                    axis=1,
                                )
                                methods = {
                                    baseline: 0.5
                                    * np.square(step["actions"][27]).sum(axis=1),
                                    "multiboundary_hidden_l2": 0.5
                                    * sum(
                                        np.square(step["actions"][layer]).sum(axis=1)
                                        for layer in layers
                                    ),
                                    "output_fisher_oracle": oracle,
                                    "post_multiboundary_vjp_sum": stacked.sum(axis=1),
                                    "post_multiboundary_vjp_max": stacked.max(axis=1),
                                }
                                methods.update(
                                    {
                                        "post_l%d_vjp_action" % layer: score
                                        for layer, score in per_layer_scores.items()
                                    }
                                )
                                common = {
                                    "sample_id": str(sample.sample_id),
                                    "task": str(sample.task),
                                    "anchor": anchor,
                                    "horizon_offset": offset,
                                    "width": width,
                                    "refresh_interval": interval,
                                }
                                for method, scores in methods.items():
                                    score_rows.extend(
                                        {
                                            **common,
                                            "method": method,
                                            "candidate_id": candidate_id,
                                            "score": float(score),
                                            "exact_kl": float(target),
                                        }
                                        for candidate_id, score, target in zip(
                                            step["candidate_ids"], scores, truth
                                        )
                                    )
            finally:
                arrays.close()
                runner.model.release(reference)
    finally:
        runner.model.close()

    scores = pd.DataFrame(score_rows)
    decision_rows: List[Dict[str, Any]] = []
    decision_keys = [
        "sample_id",
        "task",
        "anchor",
        "horizon_offset",
        "width",
        "refresh_interval",
        "method",
    ]
    for values, current in scores.groupby(decision_keys, sort=True):
        if int(values[3]) < int(config["minimum_horizon"]):
            continue
        decision_rows.append(
            {
                **dict(zip(decision_keys, values)),
                **_decision_metrics(
                    current["score"].to_numpy(dtype=np.float64),
                    current["exact_kl"].to_numpy(dtype=np.float64),
                ),
            }
        )
    decisions = pd.DataFrame(decision_rows)
    summary = _summarize(decisions, baseline)
    primary = dict(config["primary"])
    selected = summary[
        (summary["task"] == "all")
        & (summary["method"] == str(primary["method"]))
        & (summary["width"] == int(primary["width"]))
        & (summary["refresh_interval"] == int(primary["refresh_interval"]))
    ]
    if len(selected) != 1:
        raise RuntimeError("primary multi-boundary summary missing")
    row = selected.iloc[0]
    values = {
        "median_spearman_gain": float(row["median_spearman_gain"]),
        "mean_pairwise_accuracy_gain": float(row["mean_pairwise_accuracy_gain"]),
        "mean_normalized_regret_gain": float(row["mean_normalized_regret_gain"]),
    }
    diagnostics = pd.DataFrame(diagnostic_rows)
    adjoint = diagnostics["adjoint_relative_error"].dropna()
    reconstruction = diagnostics[
        "layer27_action_reconstruction_relative_error"
    ].dropna()
    checks = {name + "_positive": bool(value > 0.0) for name, value in values.items()}
    checks["adjoint_identity"] = bool(
        float(adjoint.max()) <= float(config["adjoint_error_tolerance"])
    )
    checks["action_reconstruction"] = bool(
        float(reconstruction.max()) <= float(config["action_reconstruction_tolerance"])
    )
    costs = []
    for width in widths:
        for interval in intervals:
            refreshes = int(math.ceil(horizon / float(interval)))
            costs.append(
                {
                    "width": width,
                    "refresh_interval": interval,
                    "boundaries": len(layers),
                    "reverse_passes_per_horizon": width * len(layers) * refreshes,
                    "amortized_reverse_passes_per_token": (
                        width * len(layers) * refreshes / float(horizon)
                    ),
                    "candidate_dependent_reverse_passes": 0,
                }
            )
    gate = {
        "experiment": str(config["experiment_name"]),
        "status": "bounded_model_backed_development_pilot",
        "confirmatory_evidence": False,
        "samples": sorted(sample_ids),
        "anchors": anchors,
        "layers": layers,
        "horizon": horizon,
        "candidate_count": len(sources),
        "collection_reverse_passes": int(reverse_passes),
        "collection_elapsed_s": float(time.perf_counter() - started),
        "maximum_adjoint_relative_error": float(adjoint.max()),
        "maximum_action_reconstruction_relative_error": float(
            reconstruction.max()
        ),
        "primary": primary,
        "baseline": baseline,
        "values": values,
        "checks": checks,
        "passed": bool(all(checks.values())),
        "deployment_costs": costs,
        "scope": (
            "Post-attention boundaries at layers 0, 14, and 27; candidate actions "
            "are reconstructed from fixed core masks and full-reference local attention. "
            "This remains a candidate-ranking development pilot."
        ),
    }
    atomic_frame(scores, output_root / "candidate_scores.parquet")
    atomic_frame(decisions, output_root / "decision_metrics.parquet")
    atomic_frame(diagnostics, output_root / "boundary_diagnostics.parquet")
    atomic_frame(summary, output_root / "metrics.csv")
    atomic_json(output_root / "summary.json", gate)
    atomic_text(
        output_root / "config.yaml",
        yaml.safe_dump(config, sort_keys=True, allow_unicode=True),
    )
    return output_root


__all__ = ["run_multiboundary_vjp_pilot"]
