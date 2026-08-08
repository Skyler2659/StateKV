"""Direct training-free cache-set, merge, and value-tier diagnostics."""
from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import yaml

from statekv.candidate_pullback import CandidatePullbackRunner
from statekv.config import load_discovery_config
from statekv.selectors import mandatory_and_eligible
from statekv.storage import atomic_frame, atomic_json, atomic_text
from statekv.tasks import load_discovery_tasks
from statekv.training_free_routes import (
    attention_output,
    deletion_output,
    merge_output_with_assignments,
    nearest_value_assignments,
    scenario_token_scores,
    select_top_with_mandatory,
    symmetric_quantize,
)


def _seed(*parts: Any) -> int:
    token = ":".join(str(value) for value in parts)
    return int.from_bytes(hashlib.sha256(token.encode("utf-8")).digest()[:8], "little")


def _record_attention(record: Any, layer: int, token_count: int) -> np.ndarray:
    source = (
        record.all_head_attention_distributions[int(layer)]
        .detach()
        .float()
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    if source.shape[1] > int(token_count):
        raise RuntimeError("past attention is longer than the anchor value bank")
    output = np.zeros((int(source.shape[0]), int(token_count)), dtype=np.float64)
    output[:, : source.shape[1]] = source
    return output


def _projected_norm(runner: CandidatePullbackRunner, layer: int, heads: np.ndarray) -> float:
    vector = torch.from_numpy(np.asarray(heads, dtype=np.float32).reshape(1, -1))
    projected = runner.model.project_features(int(layer), vector)[0]
    return float(projected.float().norm().item())


def _candidate_methods(
    scenario_bank: np.ndarray,
    values: np.ndarray,
    windows: Sequence[int],
    random_seed: int,
) -> Mapping[str, np.ndarray]:
    methods: Dict[str, np.ndarray] = {
        "value_norm": np.linalg.norm(values, axis=-1).mean(axis=0),
        "random": np.random.default_rng(int(random_seed)).random(values.shape[1]),
    }
    for window in windows:
        current = scenario_bank[-int(window) :]
        methods["attention_mean_w%d" % window] = scenario_token_scores(
            current, values, "mean", contribution_weighted=False
        )
        methods["attention_max_w%d" % window] = scenario_token_scores(
            current, values, "max", contribution_weighted=False
        )
        methods["contribution_mean_w%d" % window] = scenario_token_scores(
            current, values, "mean", contribution_weighted=True
        )
        methods["contribution_max_w%d" % window] = scenario_token_scores(
            current, values, "max", contribution_weighted=True
        )
        methods["contribution_mean_plus_std_w%d" % window] = scenario_token_scores(
            current, values, "mean_plus_std", contribution_weighted=True
        )
        methods["attention_ema_w%d" % window] = scenario_token_scores(
            current, values, "ema", contribution_weighted=False
        )
        methods["contribution_ema_w%d" % window] = scenario_token_scores(
            current, values, "ema", contribution_weighted=True
        )
        methods["contribution_q75_w%d" % window] = scenario_token_scores(
            current, values, "q75", contribution_weighted=True
        )
    return methods


def _summaries(
    rows: pd.DataFrame, baseline: str
) -> pd.DataFrame:
    keys = ["method", "layer"]
    records: List[Dict[str, Any]] = []
    tagged = rows.assign(unit_layer=rows["layer"].astype(int))
    expanded = pd.concat([tagged, tagged.assign(layer=-1)], ignore_index=True)
    for values, current in expanded.groupby(keys, sort=True):
        base = expanded[
            (expanded["method"] == baseline) & (expanded["layer"] == values[1])
        ]
        match = [
            "sample_id",
            "anchor",
            "horizon_offset",
            "layer",
            "unit_layer",
        ]
        merged = current.merge(
            base[match + ["projected_relative_error"]],
            on=match,
            suffixes=("", "_baseline"),
            validate="one_to_one",
        )
        records.append(
            {
                **dict(zip(keys, values)),
                "evaluation_units": int(len(current)),
                "mean_projected_relative_error": float(
                    current["projected_relative_error"].mean()
                ),
                "median_projected_relative_error": float(
                    current["projected_relative_error"].median()
                ),
                "p95_projected_relative_error": float(
                    current["projected_relative_error"].quantile(0.95)
                ),
                "maximum_projected_relative_error": float(
                    current["projected_relative_error"].max()
                ),
                "mean_deleted_attention_mass": float(
                    current["deleted_attention_mass"].mean()
                ),
                "mean_relative_error_reduction": float(
                    (
                        merged["projected_relative_error_baseline"]
                        - merged["projected_relative_error"]
                    ).mean()
                ),
                "win_rate_vs_baseline": float(
                    (
                        merged["projected_relative_error"]
                        < merged["projected_relative_error_baseline"]
                    ).mean()
                ),
            }
        )
    return pd.DataFrame(records)


def run_direct_coreset_pilot(config_path: Path, repository_root: Path) -> Path:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    cfg = load_discovery_config(str(repository_root / str(config["base_config"])))
    output_root = repository_root / str(config["output_run"])
    output_root.mkdir(parents=True, exist_ok=True)
    sample_ids = set(str(value) for value in config["sample_ids"])
    anchors = [int(value) for value in config["anchors"]]
    layers = [int(value) for value in config["layers"]]
    horizon = int(config["horizon"])
    windows = sorted(set(int(value) for value in config["scenario_windows"]))
    total_budget = int(config["total_budget"])
    sink_size = int(config["sink_size"])
    recent_size = int(config["recent_size"])
    baseline = str(config["baseline"])
    primary = str(config["primary"])
    merge_methods = set(str(value) for value in config["merge_methods"])
    tier_methods = set(str(value) for value in config["tier_methods"])
    bits = [int(value) for value in config["cold_value_bits"]]

    runner = CandidatePullbackRunner(cfg, repository_root)
    samples, _ = load_discovery_tasks(cfg)
    selected_samples = [
        sample for sample in samples if str(sample.sample_id) in sample_ids
    ]
    if {str(sample.sample_id) for sample in selected_samples} != sample_ids:
        raise RuntimeError("configured coreset samples were not loaded")

    selection_rows: List[Dict[str, Any]] = []
    evaluation_rows: List[Dict[str, Any]] = []
    merge_rows: List[Dict[str, Any]] = []
    tier_rows: List[Dict[str, Any]] = []
    started = time.perf_counter()
    runner.model.load()
    try:
        for sample in selected_samples:
            reference = runner.model.generate_reference(
                sample.sample_id, sample.task, sample.prompt
            )
            try:
                for anchor in anchors:
                    state = reference.anchors[int(anchor)]
                    selections: Dict[int, Dict[str, np.ndarray]] = {}
                    assignments: Dict[int, Dict[str, np.ndarray]] = {}
                    values_by_layer: Dict[int, np.ndarray] = {}
                    base_lengths: Dict[int, int] = {}
                    for layer in layers:
                        values = (
                            state.values[int(layer)][0]
                            .detach()
                            .float()
                            .cpu()
                            .numpy()
                            .astype(np.float64)
                        )
                        values_by_layer[layer] = values
                        base_lengths[layer] = int(values.shape[1])
                        positions = [
                            int(value)
                            for value in state.position_maps[int(layer)].tolist()
                        ]
                        if len(positions) != int(values.shape[1]):
                            raise RuntimeError("position/value alignment failed")
                        sink, recent, _ = mandatory_and_eligible(
                            positions, sink_size, recent_size
                        )
                        row_by_position = {
                            position: row for row, position in enumerate(positions)
                        }
                        mandatory = [
                            row_by_position[position]
                            for position in sorted(set(sink + recent))
                        ]
                        past_records = reference.query_records[
                            max(0, int(anchor) - max(windows)) : int(anchor)
                        ]
                        scenario_bank = np.stack(
                            [
                                _record_attention(record, layer, values.shape[1])
                                for record in past_records
                            ],
                            axis=0,
                        )
                        score_methods = _candidate_methods(
                            scenario_bank,
                            values,
                            windows,
                            _seed(config["random_seed"], sample.sample_id, anchor, layer),
                        )
                        selections[layer] = {}
                        assignments[layer] = {}
                        for method, scores in score_methods.items():
                            chosen = select_top_with_mandatory(
                                scores, total_budget, mandatory
                            )
                            selections[layer][method] = chosen
                            if method in merge_methods:
                                assignments[layer][method] = nearest_value_assignments(
                                    values, chosen
                                )
                            selection_rows.append(
                                {
                                    "sample_id": str(sample.sample_id),
                                    "task": str(sample.task),
                                    "anchor": anchor,
                                    "layer": layer,
                                    "method": method,
                                    "scenario_window": int(
                                        method.rsplit("_w", 1)[1]
                                        if "_w" in method
                                        else 0
                                    ),
                                    "selected_tokens": int(chosen.size),
                                    "mandatory_tokens": int(len(mandatory)),
                                    "candidate_algorithms_run": 0,
                                    "score_vector_length": int(scores.size),
                                    "selection_hash": hashlib.sha256(
                                        chosen.tobytes()
                                    ).hexdigest(),
                                }
                            )
                    for offset in range(1, horizon + 1):
                        target_index = int(anchor + offset - 1)
                        record = reference.query_records[target_index]
                        if offset > 1:
                            kv_heads = int(runner.model.model_info["num_key_value_heads"])
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
                        for layer in layers:
                            values = values_by_layer[layer]
                            attention = (
                                record.all_head_attention_distributions[layer]
                                .detach()
                                .float()
                                .cpu()
                                .numpy()
                                .astype(np.float64)
                            )
                            if attention.shape[1] != values.shape[1]:
                                raise RuntimeError(
                                    "future attention/value alignment failed at layer=%d offset=%d"
                                    % (layer, offset)
                                )
                            stored_full = (
                                record.all_head_attention_outputs[layer]
                                .detach()
                                .float()
                                .cpu()
                                .numpy()
                                .astype(np.float64)
                            )
                            computed_full = attention_output(attention, values)
                            full_error = float(
                                np.linalg.norm(computed_full - stored_full)
                                / max(np.linalg.norm(stored_full), 1.0e-12)
                            )
                            full_projected_norm = _projected_norm(
                                runner, layer, stored_full
                            )
                            new_rows = np.arange(
                                base_lengths[layer], values.shape[1], dtype=np.int64
                            )
                            for method, base_selection in selections[layer].items():
                                retained = np.concatenate([base_selection, new_rows])
                                deleted_output = deletion_output(
                                    attention, values, retained
                                )
                                difference = deleted_output - stored_full
                                raw_error = float(
                                    np.linalg.norm(difference, axis=1).mean()
                                )
                                projected_error = _projected_norm(
                                    runner, layer, difference
                                )
                                deleted_mask = np.ones(values.shape[1], dtype=bool)
                                deleted_mask[retained] = False
                                deleted_mass = float(
                                    attention[:, deleted_mask].sum(axis=1).mean()
                                )
                                common = {
                                    "sample_id": str(sample.sample_id),
                                    "task": str(sample.task),
                                    "anchor": anchor,
                                    "horizon_offset": offset,
                                    "layer": layer,
                                    "method": method,
                                }
                                evaluation_rows.append(
                                    {
                                        **common,
                                        "raw_head_mean_error": raw_error,
                                        "projected_error": projected_error,
                                        "projected_relative_error": float(
                                            projected_error
                                            / max(full_projected_norm, 1.0e-12)
                                        ),
                                        "deleted_attention_mass": deleted_mass,
                                        "full_reconstruction_relative_error": full_error,
                                        "retained_tokens": int(retained.size),
                                    }
                                )
                                if method in merge_methods:
                                    base_mapping = assignments[layer][method]
                                    mapping = np.concatenate(
                                        [
                                            base_mapping,
                                            np.tile(
                                                new_rows[None, :],
                                                (base_mapping.shape[0], 1),
                                            ),
                                        ],
                                        axis=1,
                                    )
                                    merged = merge_output_with_assignments(
                                        attention, values, mapping
                                    )
                                    merge_difference = merged["output"] - stored_full
                                    bound_difference = (
                                        merged["output"] - computed_full
                                    )
                                    per_head_error = np.linalg.norm(
                                        merge_difference, axis=1
                                    )
                                    bound_head_error = np.linalg.norm(
                                        bound_difference, axis=1
                                    )
                                    merge_projected = _projected_norm(
                                        runner, layer, merge_difference
                                    )
                                    merge_rows.append(
                                        {
                                            **common,
                                            "deletion_raw_head_mean_error": raw_error,
                                            "merge_raw_head_mean_error": float(
                                                per_head_error.mean()
                                            ),
                                            "merge_projected_relative_error": float(
                                                merge_projected
                                                / max(full_projected_norm, 1.0e-12)
                                            ),
                                            "mean_triangle_bound": float(
                                                merged["bound"].mean()
                                            ),
                                            "maximum_bound_violation": float(
                                                np.max(
                                                    bound_head_error
                                                    - merged["bound"]
                                                )
                                            ),
                                            "merge_beats_deletion": bool(
                                                float(per_head_error.mean()) < raw_error
                                            ),
                                        }
                                    )
                                if method in tier_methods:
                                    for bit_width in bits:
                                        tiered = symmetric_quantize(values, bit_width)
                                        tiered[:, retained, :] = values[:, retained, :]
                                        tier_output = attention_output(attention, tiered)
                                        tier_difference = tier_output - stored_full
                                        tier_projected = _projected_norm(
                                            runner, layer, tier_difference
                                        )
                                        hot = int(retained.size)
                                        total = int(values.shape[1])
                                        tier_rows.append(
                                            {
                                                **common,
                                                "cold_value_bits": bit_width,
                                                "projected_relative_error": float(
                                                    tier_projected
                                                    / max(full_projected_norm, 1.0e-12)
                                                ),
                                                "raw_head_mean_error": float(
                                                    np.linalg.norm(
                                                        tier_difference, axis=1
                                                    ).mean()
                                                ),
                                                "value_storage_ratio_vs_fp16": float(
                                                    (
                                                        hot * 16
                                                        + (total - hot) * bit_width
                                                    )
                                                    / max(total * 16, 1)
                                                ),
                                                "hot_tokens": hot,
                                                "cold_tokens": total - hot,
                                            }
                                        )
            finally:
                runner.model.release(reference)
    finally:
        runner.model.close()

    evaluations = pd.DataFrame(evaluation_rows)
    summary = _summaries(evaluations, baseline)
    overall = summary[summary["layer"] == -1]
    selected = overall[overall["method"] == primary]
    base = overall[overall["method"] == baseline]
    if len(selected) != 1 or len(base) != 1:
        raise RuntimeError("primary or baseline coreset summary is missing")
    selected_row = selected.iloc[0]
    base_row = base.iloc[0]
    merge_frame = pd.DataFrame(merge_rows)
    tier_frame = pd.DataFrame(tier_rows)
    checks = {
        "mean_local_error_improves": bool(
            selected_row["mean_projected_relative_error"]
            < base_row["mean_projected_relative_error"]
        ),
        "p95_local_error_improves": bool(
            selected_row["p95_projected_relative_error"]
            < base_row["p95_projected_relative_error"]
        ),
        "majority_local_win_rate": bool(selected_row["win_rate_vs_baseline"] > 0.5),
        "merge_bound_holds": bool(
            not merge_frame.empty
            and float(merge_frame["maximum_bound_violation"].max()) <= 1.0e-5
        ),
    }
    gate = {
        "experiment": str(config["experiment_name"]),
        "status": "bounded_model_backed_local_mechanism_pilot",
        "confirmatory_evidence": False,
        "samples": sorted(sample_ids),
        "anchors": anchors,
        "layers": layers,
        "horizon": horizon,
        "total_budget": total_budget,
        "candidate_algorithms_run_per_decision": 0,
        "method_count": int(evaluations["method"].nunique()),
        "primary": primary,
        "baseline": baseline,
        "primary_values": {
            "mean_projected_relative_error": float(
                selected_row["mean_projected_relative_error"]
            ),
            "p95_projected_relative_error": float(
                selected_row["p95_projected_relative_error"]
            ),
            "win_rate_vs_baseline": float(selected_row["win_rate_vs_baseline"]),
        },
        "baseline_values": {
            "mean_projected_relative_error": float(
                base_row["mean_projected_relative_error"]
            ),
            "p95_projected_relative_error": float(
                base_row["p95_projected_relative_error"]
            ),
        },
        "merge": {
            "rows": int(len(merge_frame)),
            "win_rate_vs_deletion": float(
                merge_frame["merge_beats_deletion"].mean()
            ),
            "maximum_bound_violation": float(
                merge_frame["maximum_bound_violation"].max()
            ),
        },
        "cold_value_tier": {
            "rows": int(len(tier_frame)),
            "bits": bits,
            "note": "Only V is quantized; K/routing and transfer latency are not tested.",
        },
        "checks": checks,
        "passed": bool(all(checks.values())),
        "collection_elapsed_s": float(time.perf_counter() - started),
        "scope": (
            "Selections are generated directly from past attention/value scenarios at "
            "three diagnostic layers and evaluated on future local attention-output error. "
            "No future query enters selection, and no end-to-end KL or latency claim is made."
        ),
    }
    atomic_frame(pd.DataFrame(selection_rows), output_root / "selection_inventory.parquet")
    atomic_frame(evaluations, output_root / "local_error_rows.parquet")
    atomic_frame(merge_frame, output_root / "merge_rows.parquet")
    atomic_frame(tier_frame, output_root / "cold_value_tier_rows.parquet")
    atomic_frame(summary, output_root / "metrics.csv")
    atomic_json(output_root / "summary.json", gate)
    atomic_text(
        output_root / "config.yaml",
        yaml.safe_dump(config, sort_keys=True, allow_unicode=True),
    )
    return output_root


__all__ = ["run_direct_coreset_pilot"]
