"""Minimal model-backed pilot for periodic shared-JVP Fisher sketches."""
from __future__ import annotations

import hashlib
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np
import pandas as pd
import yaml

from statekv.candidate_pullback import CandidatePullbackRunner, PureFinalBoundaryMap
from statekv.config import load_discovery_config
from statekv.functional_probe import _condition_cache
from statekv.shared_jvp import gram_variants, randomized_action_basis, state_action_scores
from statekv.storage import atomic_frame, atomic_json, atomic_text
from statekv.tasks import load_discovery_tasks
from statekv.training_free_analysis import _decision_metrics


INDEX_KEYS = [
    "sample_id",
    "candidate_id",
    "anchor",
    "horizon_offset",
    "target_index",
]


def _seed(*parts: Any) -> int:
    token = ":".join(str(value) for value in parts)
    return int.from_bytes(
        hashlib.sha256(token.encode("utf-8")).digest()[:8], "little"
    )


def _softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    weights = np.exp(values - float(np.max(values)))
    return weights / weights.sum()


def _source_frames(
    source_root: Path, sample_id: str, sources: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, Mapping[str, np.ndarray]]:
    geometry = pd.read_parquet(source_root / "independent_fisher_geometry_rows.parquet")
    geometry = geometry[
        (geometry["sample_id"].astype(str) == str(sample_id))
        & geometry["candidate_source"].astype(str).isin(set(sources))
    ].copy()
    index = pd.read_parquet(source_root / "independent_vector_index.parquet")
    index = index[index["sample_id"].astype(str) == str(sample_id)].copy()
    if geometry.empty or index.empty:
        raise RuntimeError("source artifacts missing sample=%s" % sample_id)
    recorded = Path(str(index.iloc[0]["vector_fragment"]))
    vector_path = recorded
    if not vector_path.exists():
        vector_path = (
            source_root
            / "fragments"
            / "independent_fisher"
            / "vectors"
            / recorded.name
        )
    arrays = np.load(vector_path, allow_pickle=False)
    return geometry, index, arrays


def _vectors(
    index: pd.DataFrame,
    arrays: Mapping[str, np.ndarray],
    current: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    merged = current[INDEX_KEYS].merge(
        index[INDEX_KEYS + ["vector_row"]],
        on=INDEX_KEYS,
        how="left",
        validate="one_to_one",
    )
    rows = merged["vector_row"].to_numpy(dtype=np.int64)
    actions = np.asarray(arrays["direct_projected_l27"][rows]).astype(
        np.float64
    )
    full = np.asarray(arrays["full_residual_l27"][rows]).astype(np.float64)
    if not np.allclose(full, full[0], atol=2.0e-3, rtol=2.0e-3):
        raise RuntimeError("reference residual differs across candidates")
    return full[0], actions


def _summarize(decisions: pd.DataFrame, baseline: str) -> pd.DataFrame:
    expanded = [decisions]
    overall = decisions.copy()
    overall["task"] = "all"
    joined = pd.concat(expanded + [overall], ignore_index=True)
    keys = ["task", "method", "rank", "refresh_interval"]
    records: List[Dict[str, Any]] = []
    for values, current in joined.groupby(keys, sort=True):
        base = joined[
            (joined["method"] == baseline)
            & (joined["task"] == values[0])
            & (joined["rank"] == values[2])
            & (joined["refresh_interval"] == values[3])
        ]
        match = ["sample_id", "anchor", "horizon_offset"]
        if values[1] == baseline:
            gains = current.assign(
                spearman_gain=0.0,
                pairwise_accuracy_gain=0.0,
                top1_accuracy_gain=0.0,
                normalized_regret_gain=0.0,
            )
        else:
            gains = current.merge(
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
            gains["spearman_gain"] = (
                gains["spearman"] - gains["spearman_baseline"]
            )
            gains["pairwise_accuracy_gain"] = (
                gains["pairwise_accuracy"]
                - gains["pairwise_accuracy_baseline"]
            )
            gains["top1_accuracy_gain"] = (
                gains["top1_accuracy"] - gains["top1_accuracy_baseline"]
            )
            gains["normalized_regret_gain"] = (
                gains["normalized_regret_baseline"]
                - gains["normalized_regret"]
            )
        records.append(
            {
                **dict(zip(keys, values)),
                "decision_units": int(len(current)),
                "median_spearman": float(current["spearman"].median()),
                "mean_pairwise_accuracy": float(
                    current["pairwise_accuracy"].mean()
                ),
                "mean_top1_accuracy": float(current["top1_accuracy"].mean()),
                "mean_normalized_regret": float(
                    current["normalized_regret"].mean()
                ),
                "median_spearman_gain": float(gains["spearman_gain"].median()),
                "mean_pairwise_accuracy_gain": float(
                    gains["pairwise_accuracy_gain"].mean()
                ),
                "mean_top1_accuracy_gain": float(
                    gains["top1_accuracy_gain"].mean()
                ),
                "mean_normalized_regret_gain": float(
                    gains["normalized_regret_gain"].mean()
                ),
            }
        )
    return pd.DataFrame(records)


def run_shared_jvp_pilot(config_path: Path, repository_root: Path) -> Path:
    """Run a bounded two-sequence development pilot and write real results."""

    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    base_config_path = repository_root / str(config["base_config"])
    cfg = load_discovery_config(str(base_config_path))
    source_root = repository_root / str(config["source_run"])
    output_root = repository_root / str(config["output_run"])
    output_root.mkdir(parents=True, exist_ok=True)
    sample_ids = set(str(value) for value in config["sample_ids"])
    anchors = [int(value) for value in config["anchors"]]
    horizon = int(config["horizon"])
    sources = [str(value) for value in config["candidate_sources"]]
    ranks = sorted(set(int(value) for value in config["ranks"]))
    maximum_rank = max(ranks)
    intervals = sorted(set(int(value) for value in config["refresh_intervals"]))
    sketch_widths = sorted(
        set(int(value) for value in config["fisher_sketch_widths"])
    )
    decay = float(config["decay"])
    baseline = str(config["baseline"])
    runner = CandidatePullbackRunner(cfg, repository_root)
    samples, _ = load_discovery_tasks(cfg)
    selected_samples = [
        sample for sample in samples if str(sample.sample_id) in sample_ids
    ]
    if {str(sample.sample_id) for sample in selected_samples} != sample_ids:
        raise RuntimeError("configured pilot samples were not loaded")

    probe_rows: List[Dict[str, Any]] = []
    score_rows: List[Dict[str, Any]] = []
    derivative_mode = str(config["directional_derivative"])
    finite_difference_radius = float(config["finite_difference_radius"])
    collection_directional_probes = 0
    collection_tail_forward_evaluations = 0
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
                        cfg,
                        int(anchor_state.logical_length) + horizon + 2,
                        1,
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
                                    "pilot candidate pool is incomplete at offset=%d"
                                    % offset
                                )
                            full_hidden, actions = _vectors(index, arrays, current)
                            basis_seed = _seed(
                                config["random_seed"], sample.sample_id, anchor, offset
                            )
                            basis = randomized_action_basis(
                                actions,
                                maximum_rank,
                                seed=basis_seed,
                                oversampling=int(config.get("basis_oversampling", 4)),
                            )
                            pure_map = PureFinalBoundaryMap(
                                runner.model, full_state.cache[27]
                            )
                            output_directions: List[np.ndarray] = []
                            asymmetries: List[float] = []
                            jvp_started = time.perf_counter()
                            base_logits = pure_map.evaluate(full_hidden)
                            collection_tail_forward_evaluations += 1
                            for column in range(maximum_rank):
                                if derivative_mode == "autodiff_jvp":
                                    _, derivative = pure_map.jvp(
                                        full_hidden, basis[:, column]
                                    )
                                    collection_tail_forward_evaluations += 1
                                elif derivative_mode == "symmetric_finite_difference":
                                    finite = pure_map.symmetric_fd(
                                        full_hidden,
                                        basis[:, column],
                                        finite_difference_radius,
                                        center_output=base_logits,
                                    )
                                    derivative = finite["derivative"]
                                    asymmetries.append(
                                        float(finite["plus_minus_asymmetry"])
                                    )
                                    collection_tail_forward_evaluations += 2
                                else:
                                    raise ValueError(
                                        "unknown directional_derivative=%s"
                                        % derivative_mode
                                    )
                                output_directions.append(derivative)
                                collection_directional_probes += 1
                            jvp_elapsed = time.perf_counter() - jvp_started
                            output_matrix = np.column_stack(output_directions)
                            probability = _softmax(base_logits)
                            grams = gram_variants(
                                probability,
                                output_matrix,
                                sketch_widths,
                                seed=_seed(basis_seed, "fisher"),
                            )
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
                            exact_gram = grams["full"]
                            for name, gram in grams.items():
                                probe_rows.append(
                                    {
                                        "sample_id": str(sample.sample_id),
                                        "task": str(sample.task),
                                        "anchor": anchor,
                                        "horizon_offset": offset,
                                        "gram_method": name,
                                        "rank": maximum_rank,
                                        "reference_relative_error": reference_error,
                                        "gram_relative_error": float(
                                            np.linalg.norm(gram - exact_gram)
                                            / max(
                                                np.linalg.norm(exact_gram), 1.0e-12
                                            )
                                        ),
                                        "derivative_mode": derivative_mode,
                                        "directional_probes": maximum_rank,
                                        "tail_forward_evaluations": (
                                            maximum_rank
                                            if derivative_mode == "autodiff_jvp"
                                            else 2 * maximum_rank + 1
                                        ),
                                        "probe_elapsed_s": float(jvp_elapsed),
                                        "median_fd_asymmetry": (
                                            float(np.median(asymmetries))
                                            if asymmetries
                                            else float("nan")
                                        ),
                                    }
                                )
                            steps[offset] = {
                                "metadata": current.reset_index(drop=True),
                                "actions": actions,
                                "basis": basis,
                                "grams": grams,
                            }
                            if offset < horizon:
                                runner._clear_controls()
                                _, record, _ = runner.model.forward_one(
                                    full_state,
                                    current_token,
                                    capture_attention=True,
                                )
                                if int(record.query_position) != int(
                                    reference.query_records[target_index].query_position
                                ):
                                    raise RuntimeError(
                                        "full-reference replay position is misaligned"
                                    )
                                current_token = int(
                                    reference.generated_token_ids[target_index]
                                )
                    finally:
                        runner.model.release(full_state)

                    candidate_ids = list(
                        steps[1]["metadata"]["candidate_id"].astype(str)
                    )
                    hidden_width = int(steps[1]["actions"].shape[1])
                    for interval in intervals:
                        states = {
                            name: np.zeros(
                                (len(candidate_ids), hidden_width), dtype=np.float64
                            )
                            for name in ("action", "repeated", "innovation", "ema")
                        }
                        previous = np.zeros_like(states["action"])
                        for offset in range(1, horizon + 1):
                            step = steps[offset]
                            current_ids = list(
                                step["metadata"]["candidate_id"].astype(str)
                            )
                            if current_ids != candidate_ids:
                                raise RuntimeError(
                                    "candidate identity changed within pilot horizon"
                                )
                            actions = step["actions"]
                            refresh = 1 + ((offset - 1) // interval) * interval
                            probe = steps[refresh]
                            truth = step["metadata"]["exact_kl"].to_numpy(
                                dtype=np.float64
                            )
                            oracle = step["metadata"][
                                "g3_midpoint_fisher"
                            ].to_numpy(dtype=np.float64)
                            for rank in ranks:
                                basis = probe["basis"][:, :rank]
                                hidden_energy = 0.5 * np.square(actions).sum(axis=1)
                                methods: Dict[str, np.ndarray] = {
                                    baseline: hidden_energy,
                                    "output_fisher_oracle": oracle,
                                }
                                for gram_name, maximum_gram in probe["grams"].items():
                                    gram = maximum_gram[:rank, :rank]
                                    for state_name, state in states.items():
                                        method = "pullback_%s_%s" % (
                                            gram_name,
                                            state_name,
                                        )
                                        methods[method] = state_action_scores(
                                            actions, state, basis, gram
                                        )
                                for method, scores in methods.items():
                                    common = {
                                        "sample_id": str(sample.sample_id),
                                        "task": str(sample.task),
                                        "anchor": anchor,
                                        "horizon_offset": offset,
                                        "rank": rank,
                                        "refresh_interval": interval,
                                        "method": method,
                                    }
                                    score_rows.extend(
                                        {
                                            **common,
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
                            states["innovation"] = (
                                decay * states["innovation"] + delta
                            )
                            states["ema"] = (
                                decay * states["ema"] + (1.0 - decay) * actions
                            )
                            previous = actions.copy()
            finally:
                arrays.close()
                runner.model.release(reference)
    finally:
        runner.model.close()

    # Decision metrics are recomputed from candidate rows to keep the raw score
    # table auditable and avoid embedding dictionaries in Parquet.
    scores = pd.DataFrame(score_rows)
    decision_records: List[Dict[str, Any]] = []
    decision_keys = [
        "sample_id",
        "task",
        "anchor",
        "horizon_offset",
        "rank",
        "refresh_interval",
        "method",
    ]
    for values, current in scores.groupby(decision_keys, sort=True):
        if int(values[3]) < int(config["minimum_horizon"]):
            continue
        decision_records.append(
            {
                **dict(zip(decision_keys, values)),
                **_decision_metrics(
                    current["score"].to_numpy(dtype=np.float64),
                    current["exact_kl"].to_numpy(dtype=np.float64),
                ),
            }
        )
    decisions = pd.DataFrame(decision_records)
    summary = _summarize(decisions, baseline)
    primary = config["primary"]
    primary_method = "pullback_randomized_%d_%s" % (
        int(primary["fisher_sketch_width"]), str(primary["state_mode"])
    )
    selected = summary[
        (summary["task"] == "all")
        & (summary["method"] == primary_method)
        & (summary["rank"] == int(primary["rank"]))
        & (summary["refresh_interval"] == int(primary["refresh_interval"]))
    ]
    if len(selected) != 1:
        raise RuntimeError("primary shared-JVP summary is missing")
    row = selected.iloc[0]
    values = {
        "median_spearman_gain": float(row["median_spearman_gain"]),
        "mean_pairwise_accuracy_gain": float(
            row["mean_pairwise_accuracy_gain"]
        ),
        "mean_normalized_regret_gain": float(
            row["mean_normalized_regret_gain"]
        ),
    }
    checks = {key + "_positive": bool(value > 0.0) for key, value in values.items()}
    passed = bool(checks and all(checks.values()))
    costs = []
    for rank in ranks:
        for interval in intervals:
            refreshes = int(math.ceil(horizon / float(interval)))
            probes = int(rank * refreshes)
            tail_forwards = int(
                (rank if derivative_mode == "autodiff_jvp" else 2 * rank + 1)
                * refreshes
            )
            costs.append(
                {
                    "rank": rank,
                    "refresh_interval": interval,
                    "directional_probes_per_horizon": probes,
                    "tail_forward_evaluations_per_horizon": tail_forwards,
                    "amortized_directional_probes_per_token": probes
                    / float(horizon),
                    "amortized_tail_forward_evaluations_per_token": tail_forwards
                    / float(horizon),
                }
            )
    gate = {
        "experiment": str(config["experiment_name"]),
        "status": "bounded_model_backed_development_pilot",
        "confirmatory_evidence": False,
        "samples": sorted(sample_ids),
        "anchors": anchors,
        "horizon": horizon,
        "candidate_count": len(sources),
        "autodiff_jvp_supported": derivative_mode == "autodiff_jvp",
        "derivative_mode": derivative_mode,
        "finite_difference_radius": finite_difference_radius,
        "collection_directional_probes": int(collection_directional_probes),
        "collection_tail_forward_evaluations": int(
            collection_tail_forward_evaluations
        ),
        "collection_elapsed_s": float(time.perf_counter() - started),
        "primary": {**dict(primary), "method": primary_method},
        "baseline": baseline,
        "values": values,
        "checks": checks,
        "passed": passed,
        "outcome": (
            "shared_jvp_candidate_for_independent_replication"
            if passed
            else "shared_jvp_not_yet_a_low_cost_controller_metric"
        ),
        "deployment_costs": costs,
        "scope": (
            "Final-boundary layer-27 pullback on two development sequences; "
            "not an end-to-end cache-controller or latency benchmark."
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


__all__ = ["run_shared_jvp_pilot"]
