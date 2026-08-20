"""Matched-granularity data collection for the dynamic-horizon oracle.

This module records full per-KV-head attention trajectories on a small,
preregistered layer panel while replaying the existing recoverable qk_pool
baseline.  It does not implement an adaptive policy.  Future information is
used only by the separate offline oracle analysis.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
import yaml

from statekv.candidate_pullback import CandidatePullbackRunner
from statekv.config import CacheDiscoveryConfig, apply_named_overrides, load_discovery_config
from statekv.oracle_closed_loop import KVBackingStore
from statekv.oracle_policy_comparison import _selection_from_scores
from statekv.oracle_policy_freegen import (
    _advance_full_state,
    _check_prompt_truncation,
    _free_rollout,
    _metric_row,
)
from statekv.qkv_decomposition import _scoring_forward_per_head, rank_and_margin
from statekv.selectors import mandatory_and_eligible
from statekv.storage import atomic_frame, atomic_json, atomic_text
from statekv.tasks import load_discovery_tasks


def _safe_sample_id(sample_id: str) -> str:
    return str(sample_id).replace(":", "__").replace("/", "_")


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    """Write one compressed trajectory without exposing a partial artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=path.name, suffix=".npz", dir=path.parent)
    os.close(handle)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _pack_trajectory(
    attention_steps: Sequence[np.ndarray],
    position_steps: Sequence[Sequence[int]],
) -> Dict[str, np.ndarray]:
    cycles = len(attention_steps)
    width = max(len(values) for values in position_steps)
    layers, heads = attention_steps[0].shape[:2]
    attention = np.full((cycles, layers, heads, width), np.nan, dtype=np.float32)
    position_ids = np.full((cycles, width), -1, dtype=np.int32)
    for cycle, (values, positions) in enumerate(zip(attention_steps, position_steps)):
        count = len(positions)
        attention[cycle, :, :, :count] = values.astype(np.float32, copy=False)
        position_ids[cycle, :count] = np.asarray(positions, dtype=np.int32)
    return {"attention": attention, "position_ids": position_ids}


def collect_dynamic_horizon_trajectories(config_path: Path, repository_root: Path) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cfg = load_discovery_config(str(repository_root / str(config["base_config"])))
    cfg.experiment_name = str(config["experiment_name"])
    for key, value in dict(config.get("model_overrides") or {}).items():
        setattr(cfg.model, str(key), value)
    apply_named_overrides(cfg.runtime, config.get("runtime_overrides"), "runtime")
    cfg.tasks = dict(config["task_overrides"])
    cfg.runtime.seed = int(config["data_seed"])
    cfg.runtime.run_id = str(config["runtime_run_id"])
    cfg.anchor_steps = [0]

    sample_ids = [str(value) for value in config["sample_ids"]]
    output_root = repository_root / str(config["output_run"])
    trajectory_root = output_root / "trajectories"
    output_root.mkdir(parents=True, exist_ok=True)
    layers = [int(value) for value in config["diagnostic_layers"]]
    cycles = int(config["control_cycles"])
    total_budget = int(config["total_budget"])
    sink_size = int(config["sink_size"])
    recent_size = int(config["recent_size"])
    core_budget = int(config["core_budget"])

    runner = CandidatePullbackRunner(cfg, repository_root)
    samples, _ = load_discovery_tasks(cfg)
    by_id = {str(sample.sample_id): sample for sample in samples}
    if set(by_id).intersection(sample_ids) != set(sample_ids):
        missing = sorted(set(sample_ids) - set(by_id))
        raise RuntimeError(f"configured oracle samples were not loaded: {missing}")
    selected = [by_id[value] for value in sample_ids]

    step_rows: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    started = time.perf_counter()
    model_info = runner.model.load()
    kv_heads = int(model_info["num_key_value_heads"])
    model_layers = int(model_info["num_layers"])
    if any(layer < 0 or layer >= model_layers for layer in layers):
        raise ValueError("diagnostic layer outside model")

    try:
        for sample_index, sample in enumerate(selected, start=1):
            artifact = trajectory_root / f"{_safe_sample_id(sample.sample_id)}.npz"
            if artifact.exists():
                print(f"[dynamic-oracle] reuse {sample.sample_id}", flush=True)
                continue
            reference = runner.model.generate_reference(sample.sample_id, sample.task, sample.prompt)
            _check_prompt_truncation(reference, str(sample.sample_id), False)
            try:
                anchor_state = reference.anchors[0]
                full_selection = runner._all_history_selection(reference, 0)
                full_cache = CacheDiscoveryConfig(
                    total_budget=int(anchor_state.logical_length + cycles + 2),
                    sink_size=0,
                    recent_size=1,
                    selected_core_budget=int(anchor_state.logical_length + 1),
                )
                compressed_state, _ = runner.model.state_from_anchor(
                    anchor_state, full_selection, cache_config=full_cache
                )
                full_state, _ = runner.model.state_from_anchor(
                    anchor_state, full_selection, cache_config=full_cache
                )
                backing = KVBackingStore()
                full_backing = KVBackingStore()
                initial_cache = CacheDiscoveryConfig(
                    total_budget=total_budget,
                    sink_size=sink_size,
                    recent_size=max(1, recent_size - 1),
                    selected_core_budget=core_budget,
                )
                rolling_cache = CacheDiscoveryConfig(
                    total_budget=total_budget,
                    sink_size=sink_size,
                    recent_size=recent_size,
                    selected_core_budget=core_budget,
                )
                current_token = int(anchor_state.query_token_id)
                generated: List[int] = []
                attention_steps: List[np.ndarray] = []
                position_steps: List[List[int]] = []
                sample_step_rows: List[Dict[str, Any]] = []
                for cycle in range(cycles):
                    backing.update(runner, compressed_state)
                    full_backing.update(runner, full_state)
                    per_head, positions, scoring_s = _scoring_forward_per_head(
                        runner, compressed_state, backing, current_token
                    )
                    observed_heads = int(per_head[layers[0]].shape[0])
                    if observed_heads != kv_heads:
                        raise RuntimeError(
                            f"attention hook returned {observed_heads} heads, expected {kv_heads}"
                        )
                    attention_steps.append(
                        np.stack([per_head[layer] for layer in layers], axis=0)
                    )
                    position_steps.append([int(value) for value in positions])

                    _, _, eligible = mandatory_and_eligible(positions, sink_size, recent_size)
                    cores_by_layer = {}
                    scores_by_layer = {}
                    for layer in range(model_layers):
                        mean_attention = np.asarray(per_head[layer], dtype=np.float64).mean(axis=0)
                        _, _, core = rank_and_margin(
                            mean_attention, positions, eligible, core_budget
                        )
                        cores_by_layer[layer] = core
                        scores_by_layer[layer] = mean_attention
                    selection = _selection_from_scores(
                        "qk_pool", positions, eligible, cores_by_layer, scores_by_layer
                    )
                    rollout, new_tokens = _free_rollout(
                        runner,
                        compressed_state,
                        backing,
                        current_token,
                        selection,
                        1,
                        initial_cache,
                        rolling_cache,
                    )
                    trajectory_rows = _advance_full_state(
                        runner, full_state, current_token, new_tokens, rollout.logits
                    )
                    row = {
                        "sample_id": str(sample.sample_id),
                        "task": str(sample.task),
                        "cycle": int(cycle),
                        "generated_token_id": int(new_tokens[0]),
                        "pool_scoring_forward_time_s": float(scoring_s),
                        **trajectory_rows[0],
                    }
                    step_rows.append(row)
                    sample_step_rows.append(row)
                    compressed_state = rollout.state
                    current_token = int(new_tokens[-1])
                    generated.extend(int(value) for value in new_tokens)

                packed = _pack_trajectory(attention_steps, position_steps)
                _atomic_npz(
                    artifact,
                    **packed,
                    layers=np.asarray(layers, dtype=np.int16),
                    sample_id=np.asarray(str(sample.sample_id)),
                    task=np.asarray(str(sample.task)),
                )
                mean_kl = float(np.mean([row["exact_kl"] for row in sample_step_rows]))
                summaries.append(
                    {
                        "sample_id": str(sample.sample_id),
                        "task": str(sample.task),
                        "mean_trajectory_exact_kl": mean_kl,
                        **_metric_row(runner, sample, "qk_pool", generated, mean_kl),
                    }
                )
                atomic_frame(pd.DataFrame(step_rows), output_root / "partial_step_rows.parquet")
                atomic_frame(pd.DataFrame(summaries), output_root / "partial_sample_summary.csv")
                atomic_json(
                    output_root / "progress.json",
                    {
                        "completed_samples": sample_index,
                        "expected_samples": len(selected),
                        "elapsed_s": float(time.perf_counter() - started),
                    },
                )
                print(
                    "[dynamic-oracle] sample %d/%d %s kl=%.6f"
                    % (sample_index, len(selected), sample.sample_id, mean_kl),
                    flush=True,
                )
            finally:
                runner.model.release(reference)
    finally:
        runner.model.close()

    # A resumed run can reuse trajectories, but the small evaluation tables
    # must be complete before the collection is declared finished.
    if len(summaries) != len(selected):
        baseline = repository_root / str(config["source_baseline_run"])
        baseline_steps = pd.read_parquet(baseline / "step_rows.parquet")
        baseline_summary = pd.read_csv(baseline / "sample_summary.csv")
        step_rows = baseline_steps[baseline_steps["sample_id"].astype(str).isin(sample_ids)].to_dict("records")
        summaries = baseline_summary[baseline_summary["sample_id"].astype(str).isin(sample_ids)].to_dict("records")
    atomic_frame(pd.DataFrame(step_rows), output_root / "step_rows.parquet")
    atomic_frame(pd.DataFrame(summaries), output_root / "sample_summary.csv")
    atomic_json(
        output_root / "collection_summary.json",
        {
            "experiment": str(config["experiment_name"]),
            "samples": sample_ids,
            "control_cycles": cycles,
            "diagnostic_layers": layers,
            "kv_heads": kv_heads,
            "attention_granularity": "kv_head",
            "policy": "qk_pool",
            "total_budget": total_budget,
            "core_budget": core_budget,
            "collection_elapsed_s": float(time.perf_counter() - started),
        },
    )
    atomic_text(output_root / "config.yaml", yaml.safe_dump(config, sort_keys=False))
    return output_root

