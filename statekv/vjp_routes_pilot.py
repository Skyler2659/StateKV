"""Model-backed pilot for shared output-side training-free VJP routes."""
from __future__ import annotations

import hashlib
import math
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yaml

from statekv.candidate_pullback import CandidatePullbackRunner, PureFinalBoundaryMap
from statekv.config import load_discovery_config
from statekv.fisher_pullback import fisher_output_random_direction
from statekv.functional_probe import _condition_cache
from statekv.shared_jvp_pilot import INDEX_KEYS, _source_frames, _vectors
from statekv.storage import atomic_frame, atomic_json, atomic_text
from statekv.tasks import load_discovery_tasks
from statekv.training_free_analysis import _decision_metrics
from statekv.training_free_routes import (
    entropy_cotangent,
    margin_cotangents,
    softmax,
    vjp_action_scores,
)


def _seed(*parts: Any) -> int:
    token = ":".join(str(value) for value in parts)
    return int.from_bytes(hashlib.sha256(token.encode("utf-8")).digest()[:8], "little")


def _summarize(decisions: pd.DataFrame, baseline: str) -> pd.DataFrame:
    joined = pd.concat(
        [decisions, decisions.assign(task="all")], ignore_index=True
    )
    keys = ["task", "method", "width", "refresh_interval"]
    match = ["sample_id", "anchor", "horizon_offset"]
    records: List[Dict[str, Any]] = []
    for values, current in joined.groupby(keys, sort=True):
        base = joined[
            (joined["task"] == values[0])
            & (joined["method"] == baseline)
            & (joined["width"] == values[2])
            & (joined["refresh_interval"] == values[3])
        ]
        if len(base) != len(current):
            raise RuntimeError("baseline decision units do not align")
        merged = current.merge(
            base[
                match
                + [
                    "spearman",
                    "pairwise_accuracy",
                    "top1_accuracy",
                    "normalized_regret",
                ]
            ],
            on=match,
            suffixes=("", "_baseline"),
            validate="one_to_one",
        )
        records.append(
            {
                **dict(zip(keys, values)),
                "decision_units": int(len(current)),
                "median_spearman": float(current["spearman"].median()),
                "mean_pairwise_accuracy": float(current["pairwise_accuracy"].mean()),
                "mean_top1_accuracy": float(current["top1_accuracy"].mean()),
                "mean_normalized_regret": float(current["normalized_regret"].mean()),
                "median_spearman_gain": float(
                    (merged["spearman"] - merged["spearman_baseline"]).median()
                ),
                "mean_pairwise_accuracy_gain": float(
                    (merged["pairwise_accuracy"] - merged["pairwise_accuracy_baseline"]).mean()
                ),
                "mean_top1_accuracy_gain": float(
                    (merged["top1_accuracy"] - merged["top1_accuracy_baseline"]).mean()
                ),
                "mean_normalized_regret_gain": float(
                    (merged["normalized_regret_baseline"] - merged["normalized_regret"]).mean()
                ),
            }
        )
    return pd.DataFrame(records)


def run_vjp_routes_pilot(config_path: Path, repository_root: Path) -> Path:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    cfg = load_discovery_config(str(repository_root / str(config["base_config"])))
    source_root = repository_root / str(config["source_run"])
    output_root = repository_root / str(config["output_run"])
    output_root.mkdir(parents=True, exist_ok=True)
    sample_ids = set(str(value) for value in config["sample_ids"])
    anchors = [int(value) for value in config["anchors"]]
    horizon = int(config["horizon"])
    sources = [str(value) for value in config["candidate_sources"]]
    widths = sorted(set(int(value) for value in config["widths"]))
    maximum_width = max(widths)
    fisher_families = [
        str(value)
        for value in config.get("fisher_probe_families", ["gaussian"])
    ]
    if not fisher_families or any(
        value not in {"gaussian", "rademacher"} for value in fisher_families
    ):
        raise ValueError("Fisher probe families must be gaussian or rademacher")
    intervals = sorted(set(int(value) for value in config["refresh_intervals"]))
    baseline = str(config["baseline"])
    decay = float(config["decay"])

    runner = CandidatePullbackRunner(cfg, repository_root)
    samples, _ = load_discovery_tasks(cfg)
    selected_samples = [
        sample for sample in samples if str(sample.sample_id) in sample_ids
    ]
    if {str(sample.sample_id) for sample in selected_samples} != sample_ids:
        raise RuntimeError("configured VJP samples were not loaded")

    probe_rows: List[Dict[str, Any]] = []
    score_rows: List[Dict[str, Any]] = []
    reverse_passes = 0
    tail_forward_evaluations = 0
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
                    full_cache = _condition_cache(
                        cfg, int(anchor_state.logical_length) + horizon + 2, 1
                    )
                    full_state, _ = runner.model.state_from_anchor(
                        anchor_state, full_selection, cache_config=full_cache
                    )
                    current_token = int(anchor_state.query_token_id)
                    steps: Dict[int, Dict[str, Any]] = {}
                    try:
                        for offset in range(1, horizon + 1):
                            target_index = int(anchor + offset - 1)
                            current = geometry[
                                (geometry["anchor"] == anchor)
                                & (geometry["horizon_offset"] == offset)
                            ].sort_values("candidate_index", kind="stable")
                            if current["candidate_id"].nunique() != len(sources):
                                raise RuntimeError(
                                    "VJP candidate pool is incomplete at offset=%d" % offset
                                )
                            point, actions = _vectors(index, arrays, current)
                            tail = PureFinalBoundaryMap(runner.model, full_state.cache[27])
                            base_logits = tail.evaluate(point)
                            tail_forward_evaluations += 1
                            probability = softmax(base_logits)
                            fisher_gradients: Dict[str, List[np.ndarray]] = {
                                family: [] for family in fisher_families
                            }
                            margin_gradients: List[np.ndarray] = []
                            probe_started = time.perf_counter()
                            rng = np.random.default_rng(
                                _seed(config["random_seed"], sample.sample_id, anchor, offset)
                            )
                            output_error = float("nan")
                            for family in fisher_families:
                                for probe in range(maximum_width):
                                    raw = (
                                        rng.standard_normal(probability.size)
                                        if family == "gaussian"
                                        else rng.choice(
                                            np.asarray([-1.0, 1.0]),
                                            size=probability.size,
                                        )
                                    )
                                    cotangent = fisher_output_random_direction(
                                        probability, raw
                                    )
                                    output, gradient = tail.vjp(point, cotangent)
                                    fisher_gradients[family].append(gradient)
                                    reverse_passes += 1
                                    if math.isnan(output_error):
                                        output_error = float(
                                            np.linalg.norm(output - base_logits)
                                            / max(np.linalg.norm(base_logits), 1.0e-12)
                                        )
                            margins, competitors = margin_cotangents(
                                base_logits, maximum_width
                            )
                            for cotangent in margins:
                                _, gradient = tail.vjp(point, cotangent)
                                margin_gradients.append(gradient)
                                reverse_passes += 1
                            _, entropy_gradient = tail.vjp(
                                point, entropy_cotangent(base_logits)
                            )
                            reverse_passes += 1
                            fisher_matrices = {
                                family: np.column_stack(gradients)
                                for family, gradients in fisher_gradients.items()
                            }
                            margin_matrix = np.column_stack(margin_gradients)
                            entropy_matrix = entropy_gradient[:, None]
                            reference_logits = (
                                reference.probe_logits[target_index]
                                .float()
                                .numpy()
                                .astype(np.float64)
                            )
                            reference_error = float(
                                np.linalg.norm(base_logits - reference_logits)
                                / max(np.linalg.norm(reference_logits), 1.0e-12)
                            )
                            adjoint_error = float("nan")
                            adjoint_radius = float("nan")
                            if offset == 1:
                                cotangent = fisher_output_random_direction(
                                    probability,
                                    np.random.default_rng(
                                        _seed(config["random_seed"], sample.sample_id, "adjoint")
                                    ).standard_normal(probability.size),
                                )
                                _, check_gradient = tail.vjp(point, cotangent)
                                reverse_passes += 1
                                left = float(np.dot(check_gradient, actions[0]))
                                checks = []
                                radii = config.get(
                                    "adjoint_fd_radii",
                                    [config.get("adjoint_fd_radius", 0.001)],
                                )
                                for radius in radii:
                                    finite = tail.symmetric_fd(
                                        point,
                                        actions[0],
                                        float(radius),
                                        center_output=base_logits,
                                    )
                                    tail_forward_evaluations += 2
                                    right = float(
                                        np.dot(cotangent, finite["derivative"])
                                    )
                                    checks.append(
                                        (
                                            float(
                                                abs(left - right)
                                                / max(abs(left), abs(right), 1.0e-12)
                                            ),
                                            float(radius),
                                        )
                                    )
                                adjoint_error, adjoint_radius = min(checks)
                            probe_rows.append(
                                {
                                    "sample_id": str(sample.sample_id),
                                    "task": str(sample.task),
                                    "anchor": anchor,
                                    "horizon_offset": offset,
                                    "vocab_size": int(base_logits.size),
                                    "maximum_width": maximum_width,
                                    "reference_relative_error": reference_error,
                                    "vjp_output_relative_error": output_error,
                                    "adjoint_relative_error": adjoint_error,
                                    "adjoint_best_fd_radius": adjoint_radius,
                                    "probe_elapsed_s": float(time.perf_counter() - probe_started),
                                    "top1_token": int(np.argmax(base_logits)),
                                    "margin_competitors": ",".join(str(int(x)) for x in competitors),
                                }
                            )
                            steps[offset] = {
                                "metadata": current.reset_index(drop=True),
                                "actions": actions,
                                "fisher": fisher_matrices,
                                "margin": margin_matrix,
                                "entropy": entropy_matrix,
                            }
                            if offset < horizon:
                                runner._clear_controls()
                                _, record, _ = runner.model.forward_one(
                                    full_state, current_token, capture_attention=True
                                )
                                expected_position = int(
                                    reference.query_records[target_index].query_position
                                )
                                if int(record.query_position) != expected_position:
                                    raise RuntimeError("full-reference replay position is misaligned")
                                current_token = int(reference.generated_token_ids[target_index])
                    finally:
                        runner.model.release(full_state)

                    candidate_ids = list(steps[1]["metadata"]["candidate_id"].astype(str))
                    hidden_width = int(steps[1]["actions"].shape[1])
                    for interval in intervals:
                        states = {
                            "repeated": np.zeros((len(candidate_ids), hidden_width)),
                            "innovation": np.zeros((len(candidate_ids), hidden_width)),
                        }
                        previous = np.zeros_like(states["innovation"])
                        for offset in range(1, horizon + 1):
                            step = steps[offset]
                            actions = step["actions"]
                            if list(step["metadata"]["candidate_id"].astype(str)) != candidate_ids:
                                raise RuntimeError("candidate identity changed within horizon")
                            refresh = 1 + ((offset - 1) // interval) * interval
                            probe = steps[refresh]
                            truth = step["metadata"]["exact_kl"].to_numpy(dtype=np.float64)
                            oracle = step["metadata"]["g3_midpoint_fisher"].to_numpy(
                                dtype=np.float64
                            )
                            hidden = 0.5 * np.square(actions).sum(axis=1)
                            for width in widths:
                                methods = {
                                    baseline: hidden,
                                    "output_fisher_oracle": oracle,
                                    "margin_vjp_action": vjp_action_scores(
                                        actions, probe["margin"][:, :width]
                                    ),
                                    "entropy_vjp_action": vjp_action_scores(
                                        actions, probe["entropy"]
                                    ),
                                }
                                for family in fisher_families:
                                    prefix = "fisher_%s_vjp" % family
                                    gradient = probe["fisher"][family][:, :width]
                                    methods[prefix + "_action"] = vjp_action_scores(
                                        actions, gradient
                                    )
                                    methods[prefix + "_repeated_state"] = vjp_action_scores(
                                        actions, gradient, states["repeated"]
                                    )
                                    methods[prefix + "_innovation_state"] = vjp_action_scores(
                                        actions, gradient, states["innovation"]
                                    )
                                    if len(fisher_families) == 1 and family == "gaussian":
                                        methods["fisher_vjp_action"] = methods[
                                            prefix + "_action"
                                        ]
                                        methods["fisher_vjp_repeated_state"] = methods[
                                            prefix + "_repeated_state"
                                        ]
                                        methods["fisher_vjp_innovation_state"] = methods[
                                            prefix + "_innovation_state"
                                        ]
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
                                            candidate_ids, scores, truth
                                        )
                                    )
                            delta = actions - previous
                            states["repeated"] = decay * states["repeated"] + actions
                            states["innovation"] = decay * states["innovation"] + delta
                            previous = actions.copy()
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
        raise RuntimeError("primary VJP summary is missing")
    row = selected.iloc[0]
    values = {
        "median_spearman_gain": float(row["median_spearman_gain"]),
        "mean_pairwise_accuracy_gain": float(row["mean_pairwise_accuracy_gain"]),
        "mean_normalized_regret_gain": float(row["mean_normalized_regret_gain"]),
    }
    checks = {name + "_positive": bool(value > 0.0) for name, value in values.items()}
    adjoint = pd.DataFrame(probe_rows)["adjoint_relative_error"].dropna()
    checks["adjoint_identity"] = bool(
        len(adjoint) == len(selected_samples) * len(anchors)
        and float(adjoint.max()) <= float(config["adjoint_error_tolerance"])
    )
    passed = bool(all(checks.values()))
    costs = []
    for width in widths:
        for interval in intervals:
            refreshes = int(math.ceil(horizon / float(interval)))
            costs.extend(
                [
                    {
                        "route": route,
                        "width": width,
                        "refresh_interval": interval,
                        "reverse_passes_per_horizon": width * refreshes,
                        "amortized_reverse_passes_per_token": width * refreshes / float(horizon),
                        "candidate_dependent_reverse_passes": 0,
                    }
                    for route in (
                        ["fisher_%s_vjp" % family for family in fisher_families]
                        + ["margin_vjp"]
                    )
                ]
            )
    gate = {
        "experiment": str(config["experiment_name"]),
        "status": "bounded_model_backed_development_pilot",
        "confirmatory_evidence": False,
        "samples": sorted(sample_ids),
        "anchors": anchors,
        "horizon": horizon,
        "candidate_count": len(sources),
        "collection_reverse_passes": int(reverse_passes),
        "collection_tail_forward_evaluations": int(tail_forward_evaluations),
        "collection_elapsed_s": float(time.perf_counter() - started),
        "maximum_adjoint_relative_error": float(adjoint.max()),
        "primary": primary,
        "baseline": baseline,
        "values": values,
        "checks": checks,
        "passed": passed,
        "outcome": (
            "shared_vjp_candidate_for_independent_replication"
            if passed
            else "shared_vjp_not_yet_a_low_cost_controller_metric"
        ),
        "deployment_costs": costs,
        "scope": (
            "Final-boundary layer-27 VJPs on development sequences. Reverse passes "
            "are shared across candidates; this tests ranking, not candidate generation "
            "or end-to-end latency."
        ),
    }
    atomic_frame(scores, output_root / "candidate_scores.parquet")
    atomic_frame(decisions, output_root / "decision_metrics.parquet")
    atomic_frame(pd.DataFrame(probe_rows), output_root / "probe_diagnostics.parquet")
    atomic_frame(summary, output_root / "metrics.csv")
    atomic_json(output_root / "summary.json", gate)
    atomic_text(
        output_root / "config.yaml",
        yaml.safe_dump(config, sort_keys=True, allow_unicode=True),
    )
    return output_root


__all__ = ["run_vjp_routes_pilot"]
