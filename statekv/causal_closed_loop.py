"""Strict pure-eviction closed loop for the causal rollout existence proof."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

from statekv.candidate_pullback import CandidatePullbackRunner
from statekv.causal_existence import (
    _scoring_forward,
    causal_prefix_reference,
    expand_split_ids,
    sample_id_for,
    task_overrides,
)
from statekv.causal_predictors import _rho_key
from statekv.causal_rollout import _causal_self_rollout, _prefix_recompute_state
from statekv.causal_student import (
    RuntimeStudentScorer,
    load_student_checkpoint,
    runtime_observation_from_record,
)
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
from statekv.storage import atomic_frame, atomic_json
from statekv.tasks import load_discovery_tasks
from statekv.trajectory_analysis import cluster_bootstrap_interval


def _paired_comparison_frame(
    summary: pd.DataFrame,
    config: Mapping[str, Any],
    policies: Sequence[str],
) -> pd.DataFrame:
    comparisons: List[Dict[str, Any]] = []
    for budget, group in summary.groupby("budget"):
        pivot = group.pivot(
            index="sample_id", columns="policy", values="mean_trajectory_exact_kl"
        )
        causal_policy = "STRICT_CAUSAL_ROLLOUT_R2"
        if causal_policy not in pivot:
            continue
        for baseline_policy in policies:
            if baseline_policy == causal_policy:
                continue
            delta = (pivot[baseline_policy] - pivot[causal_policy]).rename(
                "kl_improvement"
            ).reset_index()
            ci = cluster_bootstrap_interval(
                delta,
                "kl_improvement",
                samples=int(config["gate_a"]["bootstrap_repetitions"]),
                seed=(
                    int(config["gate_a"]["random_seed"])
                    + int(budget)
                    + list(policies).index(baseline_policy)
                ),
                statistic="mean",
            )
            comparisons.append(
                {
                    "budget": int(budget),
                    "baseline_policy": baseline_policy,
                    "causal_policy": causal_policy,
                    "primary_comparison": baseline_policy
                    == str(config["closed_loop"]["primary_baseline"]),
                    "mean_kl_improvement": float(ci["estimate"]),
                    "ci_low": float(ci["ci_low"]),
                    "ci_high": float(ci["ci_high"]),
                    "sequence_win_rate": float(
                        (delta["kl_improvement"] > 0).mean()
                    ),
                    "sequences": int(len(delta)),
                }
            )
    return pd.DataFrame(comparisons)


def _closed_loop_task_overrides(config: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    indices = [int(value) for value in config["closed_loop_test_indices"]]
    settings_by_task = dict(config.get("task_settings") or {})
    output: Dict[str, Dict[str, Any]] = {}
    for family in config["task_families"]:
        family = str(family)
        settings = dict(settings_by_task.get(family) or {})
        settings["num_samples"] = len(indices)
        if family in {"ruler_niah", "ruler_niah_multikey"}:
            settings["sample_offset"] = min(indices)
        else:
            settings["sample_indices"] = indices
        output[family] = settings
    return output


def _closed_loop_ids(config: Mapping[str, Any], split: str) -> List[str]:
    if str(split) == "closed_loop_test":
        return [
            sample_id_for(str(family), int(index))
            for family in config["task_families"]
            for index in config["closed_loop_test_indices"]
        ]
    return expand_split_ids(config)[str(split)]


def _mean_attention_array(value: Any) -> np.ndarray:
    array = value.detach().float().cpu().numpy().astype(np.float64)
    return array.reshape(-1, array.shape[-1]).mean(axis=0)


def _prompt_attention_memory(
    anchor: Any, score_layers: Sequence[int], window: int
) -> Tuple[Dict[int, float], List[Dict[int, float]]]:
    layer_cumulative: List[Dict[int, float]] = []
    for layer in score_layers:
        positions = [
            int(value) for value in anchor.position_maps[int(layer)].tolist()
        ]
        values = _mean_attention_array(
            anchor.attention.accumulated_by_layer[int(layer)]
        )
        layer_cumulative.append(
            {
                position: float(values[index])
                for index, position in enumerate(positions)
                if index < len(values)
            }
        )
    universe = sorted({position for row in layer_cumulative for position in row})
    cumulative = {
        position: float(
            np.mean([row.get(position, 0.0) for row in layer_cumulative])
        )
        for position in universe
    }

    observations = anchor.attention_observation_rows or {}
    available = min(
        [len(observations.get(int(layer), [])) for layer in score_layers]
        or [0]
    )
    # The anchor query is replayed as cycle 0, so exclude the final prompt row.
    begin = max(0, available - 1 - int(window))
    rows: List[Dict[int, float]] = []
    for observation_index in range(begin, max(begin, available - 1)):
        layer_rows = []
        for layer in score_layers:
            positions = [
                int(value) for value in anchor.position_maps[int(layer)].tolist()
            ]
            values = _mean_attention_array(
                observations[int(layer)][observation_index]
            )
            layer_rows.append(
                {
                    position: float(values[index])
                    for index, position in enumerate(positions)
                    if index < len(values)
                }
            )
        row_universe = sorted(
            {position for layer_row in layer_rows for position in layer_row}
        )
        rows.append(
            {
                position: float(
                    np.mean(
                        [layer_row.get(position, 0.0) for layer_row in layer_rows]
                    )
                )
                for position in row_universe
            }
        )
    return cumulative, rows


def hybrid_trigger(
    mode: str,
    eligible_scores: Sequence[float],
    k: int,
    cycle: int,
    cfg: Mapping[str, Any],
) -> Tuple[bool, float]:
    """Decide whether the hybrid policy pays for an R2 rollout this cycle.

    Returns ``(triggered, trigger_stat)`` where ``trigger_stat`` is the
    decision statistic (normalized margin or entropy) for post-hoc analysis.
    """

    scores = np.sort(np.asarray(eligible_scores, dtype=np.float64))[::-1]
    k = int(k)
    cycle = int(cycle)
    margin_threshold = float(cfg.get("margin_threshold", 0.1))
    entropy_threshold = float(cfg.get("entropy_threshold", 0.95))
    base_refresh = int(cfg.get("base_refresh", 8))
    # Without a position beyond the budget there is no boundary to be
    # uncertain about, so the margin is treated as infinite (never triggers).
    margin = float("inf")
    if 0 < k < len(scores):
        margin = float(
            (scores[k - 1] - scores[k]) / (float(np.std(scores)) + 1e-12)
        )
    mode = str(mode)
    if mode in {"margin", "periodic_margin"}:
        triggered = margin < margin_threshold
        if mode == "periodic_margin":
            triggered = bool(triggered or (cycle % max(1, base_refresh) == 0))
        return bool(triggered), margin
    if mode == "entropy":
        top = scores[: max(1, min(k + 32, len(scores)))]
        shifted = top - float(top.max())
        p = np.exp(shifted)
        p = p / float(p.sum())
        entropy = float(-np.sum(p * np.log(p + 1e-12)))
        normalized = float(entropy / max(float(np.log(k + 32)), 1e-12))
        return bool(normalized > entropy_threshold), normalized
    raise ValueError(f"unknown hybrid trigger mode: {mode}")


def _strict_policy_run(
    runner: CandidatePullbackRunner,
    reference: Any,
    sample: Any,
    policy: str,
    budget: int,
    cycles: int,
    sink_size: int,
    recent_size: int,
    rollout_horizon: int,
    score_layers: Sequence[int],
    fixed_baseline_rhos: Mapping[str, float],
    snapkv_window: int,
    snapkv_pooling_kernel: int,
    refresh_frequency: int,
    student: Optional[Any] = None,
    hybrid: Optional[Mapping[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if policy == "STRICT_STATEKV_STUDENT":
        if student is None:
            raise RuntimeError(
                "STRICT_STATEKV_STUDENT requires a loaded student checkpoint"
            )
        student.reset(int(cycles))
    anchor = reference.anchors[0]
    full_selection = runner._all_history_selection(reference, 0)
    full_cache = CacheDiscoveryConfig(
        total_budget=int(anchor.logical_length + cycles + rollout_horizon + 4),
        sink_size=0,
        recent_size=1,
        selected_core_budget=int(anchor.logical_length + 1),
    )
    state, _ = runner.model.state_from_anchor(
        anchor, full_selection, cache_config=full_cache
    )
    full_state, _ = runner.model.state_from_anchor(
        anchor, full_selection, cache_config=full_cache
    )
    core_budget = int(budget) - int(sink_size) - int(recent_size)
    initial_cache = CacheDiscoveryConfig(
        total_budget=int(budget),
        sink_size=int(sink_size),
        recent_size=max(1, int(recent_size) - 1),
        selected_core_budget=core_budget,
    )
    rolling_cache = CacheDiscoveryConfig(
        total_budget=int(budget),
        sink_size=int(sink_size),
        recent_size=int(recent_size),
        selected_core_budget=core_budget,
    )
    processed_tokens = [int(value) for value in reference.prompt_token_ids[:-1]]
    current_token = int(anchor.query_token_id)
    generated: List[int] = []
    rows: List[Dict[str, Any]] = []
    model_layers = len(state.cache)
    peak_active = 0
    total_teacher_s = 0.0
    cumulative_scores, snapkv_rows = _prompt_attention_memory(
        anchor, score_layers, snapkv_window
    )
    fixed_memory: Dict[Tuple[int, int, int], float] = {}
    cached_rollout_scores: Dict[int, float] = {}
    teacher_refreshes = 0
    started = time.perf_counter()
    for cycle in range(int(cycles)):
        # A new one-cycle backing store contains only physically active KV.
        # Dropped positions never survive to the next decision boundary.
        active_backing = KVBackingStore()
        active_backing.update(runner, state)
        record = None
        logits = None
        if policy == "STRICT_STATEKV_STUDENT":
            # The student rebuilds artifact_boundary features at runtime and
            # needs the diagnostic record (queries, hidden states) and logits
            # in addition to the pooled per-head attention.
            per_head_raw, positions, record, logits, scoring_s = _scoring_forward(
                runner, state, active_backing, current_token
            )
            per_head = {
                int(layer): np.asarray(values, dtype=np.float64)
                for layer, values in per_head_raw.items()
            }
        else:
            per_head, positions, scoring_s = _scoring_forward_per_head(
                runner, state, active_backing, current_token
            )
        _, _, eligible = mandatory_and_eligible(
            positions, int(sink_size), max(0, int(recent_size) - 1)
        )
        if len(eligible) < core_budget:
            raise RuntimeError("strict active pool cannot fill the core budget")
        selected_head_scores = [
            np.asarray(per_head[int(layer)], dtype=np.float64)
            for layer in score_layers
        ]
        current_shared = np.mean(np.stack(selected_head_scores), axis=(0, 1))
        if cycle == 0:
            # Prefill accumulation contains the anchor query. Remove it before
            # the ordinary online update below so cycle 0 is counted once.
            for position, value in zip(positions, current_shared):
                cumulative_scores[int(position)] = max(
                    0.0,
                    float(cumulative_scores.get(int(position), 0.0))
                    - float(value),
                )
        for position, value in zip(positions, current_shared):
            cumulative_scores[int(position)] = (
                float(cumulative_scores.get(int(position), 0.0)) + float(value)
            )
        snapkv_rows.append(
            {
                int(position): float(value)
                for position, value in zip(positions, current_shared)
            }
        )
        if len(snapkv_rows) > int(snapkv_window):
            del snapkv_rows[: len(snapkv_rows) - int(snapkv_window)]

        fixed_rows: List[np.ndarray] = []
        for layer_index, layer_scores in enumerate(selected_head_scores):
            for head, values in enumerate(layer_scores):
                rho = float(
                    fixed_baseline_rhos[
                        _rho_key(int(rollout_horizon), layer_index, head)
                    ]
                )
                row = np.zeros(len(positions), dtype=np.float64)
                for index, (position, value) in enumerate(zip(positions, values)):
                    key = (int(layer_index), int(head), int(position))
                    previous = fixed_memory.get(key)
                    updated = (
                        float(value)
                        if previous is None
                        else rho * float(previous) + (1.0 - rho) * float(value)
                    )
                    fixed_memory[key] = updated
                    row[index] = updated
                fixed_rows.append(row)
        fixed_shared = np.mean(np.stack(fixed_rows), axis=0)

        hybrid_r2_fired = False
        trigger_stat = float("nan")
        if policy == "STRICT_QK_CURRENT":
            shared_scores = current_shared
            teacher_s = 0.0
        elif policy == "STRICT_H2O_CUMULATIVE":
            shared_scores = np.asarray(
                [cumulative_scores[int(position)] for position in positions],
                dtype=np.float64,
            )
            teacher_s = 0.0
        elif policy == "STRICT_SNAPKV_OBSWIN":
            raw = np.asarray(
                [
                    sum(row.get(int(position), 0.0) for row in snapkv_rows)
                    for position in positions
                ],
                dtype=np.float64,
            )
            from src.runners.mlx_runner import snapkv_pool_scores_numpy

            shared_scores = np.asarray(
                snapkv_pool_scores_numpy(
                    raw, int(snapkv_pooling_kernel), "max"
                ),
                dtype=np.float64,
            )
            teacher_s = 0.0
        elif policy == "STRICT_BEST_PER_HEAD_FIXED_EMA":
            shared_scores = fixed_shared
            teacher_s = 0.0
        elif policy == "STRICT_CAUSAL_ROLLOUT_R2":
            refresh = cycle % int(refresh_frequency) == 0 or not cached_rollout_scores
            if refresh:
                recompute_started = time.perf_counter()
                branch = _prefix_recompute_state(
                    runner, processed_tokens, int(rollout_horizon) + 2
                )
                recompute_s = time.perf_counter() - recompute_started
                rollout = _causal_self_rollout(
                    runner,
                    branch,
                    current_token,
                    eligible,
                    [int(layer) for layer in score_layers],
                    [int(rollout_horizon)],
                )
                predicted_shared = np.asarray(
                    rollout["scores"][int(rollout_horizon)], dtype=np.float64
                ).mean(axis=(0, 1))
                cached_rollout_scores = {
                    int(position): float(value)
                    for position, value in zip(eligible, predicted_shared)
                }
                teacher_s = float(recompute_s + rollout["wall_time_s"])
                teacher_refreshes += 1
            else:
                teacher_s = 0.0
            shared_scores = np.full(len(positions), -np.inf, dtype=np.float64)
            for index, position in enumerate(positions):
                if int(position) in set(eligible):
                    shared_scores[index] = float(
                        cached_rollout_scores.get(int(position), current_shared[index])
                    )
        elif policy == "STRICT_HYBRID_QK_R2":
            hybrid_cfg = dict(hybrid or {})
            eligible_set = {int(value) for value in eligible}
            eligible_scores = np.asarray(
                [
                    float(current_shared[index])
                    for index, position in enumerate(positions)
                    if int(position) in eligible_set
                ],
                dtype=np.float64,
            )
            hybrid_r2_fired, trigger_stat = hybrid_trigger(
                str(hybrid_cfg.get("trigger", "margin")),
                eligible_scores,
                core_budget,
                int(cycle),
                hybrid_cfg,
            )
            if hybrid_r2_fired:
                # Same rollout mechanics as STRICT_CAUSAL_ROLLOUT_R2, but the
                # scores are used only for this cycle (no cross-cycle cache).
                recompute_started = time.perf_counter()
                branch = _prefix_recompute_state(
                    runner, processed_tokens, int(rollout_horizon) + 2
                )
                recompute_s = time.perf_counter() - recompute_started
                rollout = _causal_self_rollout(
                    runner,
                    branch,
                    current_token,
                    eligible,
                    [int(layer) for layer in score_layers],
                    [int(rollout_horizon)],
                )
                predicted_shared = np.asarray(
                    rollout["scores"][int(rollout_horizon)], dtype=np.float64
                ).mean(axis=(0, 1))
                rollout_scores = {
                    int(position): float(value)
                    for position, value in zip(eligible, predicted_shared)
                }
                teacher_s = float(recompute_s + rollout["wall_time_s"])
                teacher_refreshes += 1
                shared_scores = np.full(len(positions), -np.inf, dtype=np.float64)
                for index, position in enumerate(positions):
                    if int(position) in eligible_set:
                        shared_scores[index] = float(
                            rollout_scores.get(int(position), current_shared[index])
                        )
            else:
                shared_scores = current_shared
                teacher_s = 0.0
        elif policy == "STRICT_STATEKV_STUDENT":
            student_started = time.perf_counter()
            observation = runtime_observation_from_record(
                record,
                logits,
                active_backing,
                score_layers,
                int(student.query_heads),
            )
            predicted = student.observe_and_score(
                cycle=int(cycle),
                positions=positions,
                per_head_attention={
                    int(layer): per_head[int(layer)] for layer in score_layers
                },
                **observation,
            )
            teacher_s = float(time.perf_counter() - student_started)
            shared_scores = np.full(len(positions), -np.inf, dtype=np.float64)
            eligible_set = {int(value) for value in eligible}
            for index, position in enumerate(positions):
                if int(position) in eligible_set:
                    shared_scores[index] = float(
                        predicted.get(int(position), current_shared[index])
                    )
        else:
            raise ValueError(f"unknown strict closed-loop policy: {policy}")
        total_teacher_s += teacher_s
        _, _, shared_core = rank_and_margin(
            shared_scores, positions, eligible, core_budget
        )
        cores_by_layer: Dict[int, Tuple[int, ...]] = {
            layer: shared_core for layer in range(model_layers)
        }
        scores_by_layer = {
            layer: shared_scores for layer in range(model_layers)
        }
        selection = _selection_from_scores(
            policy, positions, eligible, cores_by_layer, scores_by_layer
        )
        rollout_state, new_tokens = _free_rollout(
            runner,
            state,
            active_backing,
            current_token,
            selection,
            1,
            initial_cache,
            rolling_cache,
        )
        metrics = _advance_full_state(
            runner, full_state, current_token, new_tokens, rollout_state.logits
        )[0]
        active = int(runner.model.active_cache_tokens(rollout_state.state))
        peak_active = max(peak_active, active)
        rows.append(
            {
                "sample_id": str(sample.sample_id),
                "task": str(sample.task),
                "policy": policy,
                "budget": int(budget),
                "cycle": int(cycle),
                "generated_token_id": int(new_tokens[0]),
                "active_cache_tokens": active,
                "pool_scoring_time_s": float(scoring_s),
                "causal_teacher_time_s": teacher_s,
                "causal_teacher_refreshed": bool(
                    policy
                    in {"STRICT_CAUSAL_ROLLOUT_R2", "STRICT_HYBRID_QK_R2"}
                    and teacher_s > 0.0
                ),
                "hybrid_r2_fired": bool(hybrid_r2_fired),
                "trigger_stat": float(trigger_stat),
                "recoverable_cold_tokens": 0,
                **metrics,
            }
        )
        processed_tokens.append(int(current_token))
        runner.model.release(state)
        state = rollout_state.state
        current_token = int(new_tokens[-1])
        generated.extend(int(value) for value in new_tokens)
    runner.model.release(state)
    runner.model.release(full_state)
    mean_kl = float(np.mean([row["exact_kl"] for row in rows]))
    summary = {
        "sample_id": str(sample.sample_id),
        "task": str(sample.task),
        "policy": policy,
        "budget": int(budget),
        "mean_trajectory_exact_kl": mean_kl,
        "peak_active_cache_tokens": peak_active,
        "causal_teacher_time_s": total_teacher_s,
        "causal_teacher_refreshes": int(teacher_refreshes),
        "r2_invocation_rate": float(teacher_refreshes) / max(1, int(cycles)),
        "refresh_frequency": int(refresh_frequency),
        "wall_time_s": float(time.perf_counter() - started),
        "strict_pure_eviction": True,
        "recoverable_cold_tokens": 0,
        **_metric_row(runner, sample, policy, generated, mean_kl),
    }
    return rows, summary


def _full_cache_reference_run(
    runner: CandidatePullbackRunner,
    reference: Any,
    sample: Any,
    cycles: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    anchor = reference.anchors[0]
    selection = runner._all_history_selection(reference, 0)
    cache = CacheDiscoveryConfig(
        total_budget=int(anchor.logical_length + int(cycles) + 2),
        sink_size=0,
        recent_size=1,
        selected_core_budget=int(anchor.logical_length + 1),
    )
    state, _ = runner.model.state_from_anchor(anchor, selection, cache_config=cache)
    current_token = int(anchor.query_token_id)
    generated: List[int] = []
    rows: List[Dict[str, Any]] = []
    peak_active = 0
    started = time.perf_counter()
    try:
        for cycle in range(int(cycles)):
            logits, _, forward_s = runner.model.forward_one(
                state, current_token, capture_attention=True
            )
            next_token = int(torch.argmax(logits.float()).item())
            active = int(runner.model.active_cache_tokens(state))
            peak_active = max(peak_active, active)
            rows.append(
                {
                    "sample_id": str(sample.sample_id),
                    "task": str(sample.task),
                    "policy": "FULL_CACHE_REFERENCE",
                    "budget": 0,
                    "cycle": int(cycle),
                    "generated_token_id": next_token,
                    "active_cache_tokens": active,
                    "pool_scoring_time_s": float(forward_s),
                    "causal_teacher_time_s": 0.0,
                    "causal_teacher_refreshed": False,
                    "recoverable_cold_tokens": 0,
                    "exact_kl": 0.0,
                    "exact_js": 0.0,
                    "delta_nll": 0.0,
                }
            )
            generated.append(next_token)
            current_token = next_token
    finally:
        runner.model.release(state)
    summary = {
        "sample_id": str(sample.sample_id),
        "task": str(sample.task),
        "policy": "FULL_CACHE_REFERENCE",
        "budget": 0,
        "mean_trajectory_exact_kl": 0.0,
        "peak_active_cache_tokens": peak_active,
        "causal_teacher_time_s": 0.0,
        "causal_teacher_refreshes": 0,
        "refresh_frequency": 0,
        "wall_time_s": float(time.perf_counter() - started),
        "strict_pure_eviction": False,
        "recoverable_cold_tokens": 0,
        **_metric_row(
            runner,
            sample,
            "FULL_CACHE_REFERENCE",
            generated,
            0.0,
        ),
    }
    return rows, summary


def run_strict_causal_closed_loop(
    config_path: Path,
    repository_root: Path,
    split: str = "closed_loop_test",
    max_samples: Optional[int] = None,
    budgets: Optional[Sequence[int]] = None,
    cycle_limit: Optional[int] = None,
    refresh_frequency: Optional[int] = None,
    sample_ids: Optional[Sequence[str]] = None,
    policies: Optional[Sequence[str]] = None,
    output_tag: Optional[str] = None,
) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cfg = load_discovery_config(str(repository_root / str(config["base_config"])))
    cfg.experiment_name = str(config["experiment_name"] + "_strict_closed_loop")
    for key, value in dict(config.get("model_overrides") or {}).items():
        setattr(cfg.model, str(key), value)
    apply_named_overrides(cfg.runtime, config.get("runtime_overrides"), "runtime")
    cfg.tasks = (
        _closed_loop_task_overrides(config)
        if str(split) == "closed_loop_test"
        else task_overrides(config)
    )
    cfg.runtime.seed = int(config["data_seed"])
    cfg.runtime.run_id = str(
        config["runtime_run_id"] + f"_strict_closed_loop_{split}"
    )
    cfg.anchor_steps = [0]

    selected_sample_ids = _closed_loop_ids(config, split)
    if sample_ids:
        requested_ids = {str(value) for value in sample_ids}
        unknown = requested_ids - set(selected_sample_ids)
        if unknown:
            raise ValueError(
                f"closed-loop sample IDs are outside split {split}: {sorted(unknown)}"
            )
        selected_sample_ids = [
            value for value in selected_sample_ids if value in requested_ids
        ]
    if max_samples is not None:
        selected_sample_ids = selected_sample_ids[: int(max_samples)]
    run_budgets = [
        int(value)
        for value in (budgets or config["closed_loop"]["budgets"])
    ]
    configured_policies = [str(value) for value in config["closed_loop"]["policies"]]
    policies = [
        str(value) for value in (policies or configured_policies)
    ]
    unknown_policies = set(policies) - set(configured_policies)
    if not policies or unknown_policies:
        raise ValueError(f"unknown or empty closed-loop policies: {sorted(unknown_policies)}")
    if cycle_limit is None and policies != configured_policies:
        raise RuntimeError("publication closed loop must run every configured policy")
    run_refresh_frequency = int(
        config["closed_loop"]["refresh_frequency"]
        if refresh_frequency is None
        else refresh_frequency
    )
    if run_refresh_frequency not in {
        int(value)
        for value in config["closed_loop"]["validation_refresh_frequencies"]
    }:
        raise ValueError("refresh frequency is outside the preregistered sweep")
    score_layers = [int(value) for value in config["diagnostic_layers"]]
    student_checkpoint: Optional[Dict[str, Any]] = None
    if "STRICT_STATEKV_STUDENT" in policies:
        student_path = str(config["closed_loop"].get("student_model_path") or "")
        if not student_path:
            raise RuntimeError(
                "STRICT_STATEKV_STUDENT requires closed_loop.student_model_path"
            )
        # Fails loudly when the checkpoint is missing or its feature contract
        # (width, horizons) mismatches the runtime feature construction.
        student_checkpoint = load_student_checkpoint(
            repository_root / student_path
        )
        # The student rebuilds artifact_boundary features at runtime and needs
        # per-head post-RoPE queries plus hidden states on every score layer.
        cfg.diagnostics.explicit_layers = [
            int(value) for value in config["diagnostic_layers"]
        ]
        cfg.diagnostics.explicit_heads = [
            int(value) for value in config["diagnostic_query_heads"]
        ]
    fixed_baseline_path = (
        repository_root
        / str(config["output_run"])
        / "models"
        / "fixed_baseline_tuning.json"
    )
    if fixed_baseline_path.exists():
        fixed_baseline_rhos = json.loads(
            fixed_baseline_path.read_text(encoding="utf-8")
        )["per_head"]
    elif cycle_limit is not None:
        fixed_baseline_rhos = {
            _rho_key(int(config["closed_loop"]["rollout_horizon"]), layer_index, head): 0.0
            for layer_index in range(len(score_layers))
            for head in range(8)
        }
    else:
        raise RuntimeError("strict closed loop requires train-tuned fixed baselines")
    output_root = repository_root / str(config["output_run"]) / "closed_loop" / str(split)
    if cycle_limit is not None:
        output_root = output_root / "_smoke"
    if refresh_frequency is not None:
        if str(split) == "closed_loop_test" and run_refresh_frequency != int(
            config["closed_loop"]["refresh_frequency"]
        ):
            raise RuntimeError("closed-loop test frequency must be frozen in config")
        output_root = output_root / f"_freq{run_refresh_frequency}"
    if output_tag is not None:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", str(output_tag)):
            raise ValueError("closed-loop output tag must be a simple name")
        output_root = output_root / "_shards" / str(output_tag)
    output_root.mkdir(parents=True, exist_ok=True)
    if str(split) == "closed_loop_test":
        if not (
            repository_root
            / str(config["output_run"])
            / "gate_b_passed.json"
        ).exists():
            raise RuntimeError("closed-loop test is sealed until Gate B passes")
        refresh_selection_path = (
            repository_root
            / str(config["output_run"])
            / "closed_loop"
            / "validation_refresh_selection.json"
        )
        if not refresh_selection_path.exists():
            raise RuntimeError(
                "closed-loop test is sealed until refresh frequency is frozen on validation"
            )
        refresh_selection = json.loads(
            refresh_selection_path.read_text(encoding="utf-8")
        )
        if run_refresh_frequency != int(
            refresh_selection["selected_refresh_frequency"]
        ):
            raise RuntimeError(
                "closed-loop config does not match the frozen validation refresh frequency"
            )

    runner = CandidatePullbackRunner(cfg, repository_root)
    samples, task_events = load_discovery_tasks(cfg)
    by_id = {str(sample.sample_id): sample for sample in samples}
    missing = sorted(set(selected_sample_ids) - set(by_id))
    if missing:
        raise RuntimeError(f"closed-loop samples were not loaded: {missing}")
    all_rows: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    runner.model.load()
    student_scorer: Optional[Any] = None
    if student_checkpoint is not None:
        if str(student_checkpoint.get("kind")) == "structured_mlp":
            from statekv.structured_student import RuntimeStructuredScorer

            student_cls: Any = RuntimeStructuredScorer
        else:
            student_cls = RuntimeStudentScorer
        student_scorer = student_cls(
            student_checkpoint,
            score_layers=score_layers,
            kv_heads=int(runner.model.model_info["num_key_value_heads"]),
            query_heads=int(runner.model.model_info["num_attention_heads"]),
            sink_size=int(config["sink_size"]),
            recent_size=int(config["recent_size"]),
            horizon=int(config["closed_loop"]["rollout_horizon"]),
        )
    try:
        total = len(selected_sample_ids) * len(run_budgets) * len(policies)
        ordinal = 0
        for sample_id in selected_sample_ids:
            sample = by_id[sample_id]
            reference = causal_prefix_reference(runner, sample)
            _check_prompt_truncation(reference, sample_id, False)
            try:
                full_rows, full_summary = _full_cache_reference_run(
                    runner,
                    reference,
                    sample,
                    (
                        int(config["control_cycles"])
                        if cycle_limit is None
                        else min(int(config["control_cycles"]), int(cycle_limit))
                    ),
                )
                all_rows.extend(full_rows)
                summaries.append(full_summary)
                for budget in run_budgets:
                    for policy in policies:
                        ordinal += 1
                        rows, summary = _strict_policy_run(
                            runner,
                            reference,
                            sample,
                            policy,
                            int(budget),
                            (
                                int(config["control_cycles"])
                                if cycle_limit is None
                                else min(
                                    int(config["control_cycles"]),
                                    int(cycle_limit),
                                )
                            ),
                            int(config["sink_size"]),
                            int(config["recent_size"]),
                            int(config["closed_loop"]["rollout_horizon"]),
                            score_layers,
                            fixed_baseline_rhos,
                            int(config["closed_loop"]["snapkv_window"]),
                            int(config["closed_loop"]["snapkv_pooling_kernel"]),
                            run_refresh_frequency,
                            student=student_scorer,
                            hybrid=config["closed_loop"].get("hybrid"),
                        )
                        all_rows.extend(rows)
                        summaries.append(summary)
                        atomic_frame(
                            pd.DataFrame(all_rows),
                            output_root / "partial_step_rows.parquet",
                        )
                        atomic_frame(
                            pd.DataFrame(summaries),
                            output_root / "partial_sample_summary.csv",
                        )
                        print(
                            f"[strict-closed-loop] {ordinal}/{total} {sample_id} budget={budget} policy={policy}",
                            flush=True,
                        )
            finally:
                runner.model.release(reference)
    finally:
        runner.model.close()
    steps = pd.DataFrame(all_rows)
    summary = pd.DataFrame(summaries)
    atomic_frame(steps, output_root / "step_rows.parquet")
    atomic_frame(summary, output_root / "sample_summary.csv")
    atomic_frame(
        _paired_comparison_frame(summary, config, policies),
        output_root / "paired_comparison.csv",
    )
    atomic_json(
        output_root / "protocol_summary.json",
        {
            "strict_pure_eviction": True,
            "cold_token_recovery": False,
            "temporary_prefix_recomputation": True,
            "persistent_full_shadow": False,
            "shared_token_core_across_layers": True,
            "task_events": task_events,
            "budgets": run_budgets,
            "policies": policies,
            "score_layers": score_layers,
            "primary_baseline": str(config["closed_loop"]["primary_baseline"]),
            "refresh_frequency": run_refresh_frequency,
            "full_cache_reference_exact_kl": 0.0,
            "publication_artifact": cycle_limit is None,
        },
    )
    return output_root


def select_validation_refresh_frequency(
    config_path: Path, repository_root: Path
) -> Path:
    """Freeze the preregistered validation-only refresh-frequency choice."""

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_root = repository_root / str(config["output_run"])
    tuning = dict(config["closed_loop"]["validation_refresh_tuning"])
    budget = int(tuning["budget"])
    rows: List[Dict[str, Any]] = []
    for frequency in config["closed_loop"]["validation_refresh_frequencies"]:
        frequency = int(frequency)
        path = (
            output_root
            / "closed_loop"
            / "validation"
            / "_smoke"
            / f"_freq{frequency}"
            / "paired_comparison.csv"
        )
        if not path.exists():
            raise RuntimeError(f"missing validation refresh sweep result: {path}")
        comparison = pd.read_csv(path)
        selected = comparison[
            (comparison["budget"].astype(int) == budget)
            & (comparison["primary_comparison"].astype(str).str.lower() == "true")
        ]
        if len(selected) != 1:
            raise RuntimeError(
                f"refresh frequency {frequency} lacks one primary comparison"
            )
        row = selected.iloc[0]
        rows.append(
            {
                "refresh_frequency": frequency,
                "mean_kl_improvement": float(row["mean_kl_improvement"]),
                "ci_low": float(row["ci_low"]),
                "ci_high": float(row["ci_high"]),
                "sequence_win_rate": float(row["sequence_win_rate"]),
            }
        )
    # Refreshing less often is cheaper, hence the larger interval wins only an
    # exact metric tie. The existence-quality metric remains the primary key.
    selected = sorted(
        rows,
        key=lambda row: (
            float(row["mean_kl_improvement"]),
            int(row["refresh_frequency"]),
        ),
        reverse=True,
    )[0]
    path = output_root / "closed_loop" / "validation_refresh_selection.json"
    atomic_json(
        path,
        {
            "selection_split": "validation",
            "sample_indices": [int(value) for value in tuning["sample_indices"]],
            "budget": budget,
            "cycle_limit": int(tuning["cycle_limit"]),
            "primary_metric": str(tuning["primary_metric"]),
            "selection_rule": str(tuning["selection_rule"]),
            "candidates": rows,
            "selected_refresh_frequency": int(selected["refresh_frequency"]),
            "closed_loop_test_seen": False,
        },
    )
    return path


def merge_closed_loop_shards(
    config_path: Path,
    repository_root: Path,
    split: str = "closed_loop_test",
) -> Path:
    """Merge disjoint formal shards and recompute paired sequence statistics."""

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_root = repository_root / str(config["output_run"]) / "closed_loop" / str(split)
    shard_root = output_root / "_shards"
    shard_paths = sorted(
        path for path in shard_root.iterdir() if (path / "sample_summary.csv").exists()
    ) if shard_root.exists() else []
    if not shard_paths:
        raise RuntimeError("no completed closed-loop shards are available")
    steps = pd.concat(
        [pd.read_parquet(path / "step_rows.parquet") for path in shard_paths],
        ignore_index=True,
    )
    summary = pd.concat(
        [pd.read_csv(path / "sample_summary.csv") for path in shard_paths],
        ignore_index=True,
    )
    expected_ids = set(_closed_loop_ids(config, split))
    observed_ids = set(summary["sample_id"].astype(str))
    if observed_ids != expected_ids:
        raise RuntimeError(
            "closed-loop shards do not cover exactly the frozen sample set"
        )
    if summary.duplicated(["sample_id", "policy", "budget"]).any():
        raise RuntimeError("closed-loop shards overlap")
    policies = [str(value) for value in config["closed_loop"]["policies"]]
    budgets = [int(value) for value in config["closed_loop"]["budgets"]]
    for policy in policies:
        for budget in budgets:
            count = len(
                summary[
                    (summary["policy"] == policy)
                    & (summary["budget"].astype(int) == budget)
                ]
            )
            if count != len(expected_ids):
                raise RuntimeError(
                    f"closed-loop shard grid is incomplete for {policy}, budget {budget}"
                )
    full = summary[summary["policy"] == "FULL_CACHE_REFERENCE"]
    if len(full) != len(expected_ids):
        raise RuntimeError("closed-loop shards lack one full-cache reference per sample")
    atomic_frame(steps, output_root / "step_rows.parquet")
    atomic_frame(summary, output_root / "sample_summary.csv")
    atomic_frame(
        _paired_comparison_frame(summary, config, policies),
        output_root / "paired_comparison.csv",
    )
    atomic_json(
        output_root / "protocol_summary.json",
        {
            "strict_pure_eviction": True,
            "cold_token_recovery": False,
            "temporary_prefix_recomputation": True,
            "persistent_full_shadow": False,
            "shared_token_core_across_layers": True,
            "budgets": budgets,
            "policies": policies,
            "score_layers": [int(value) for value in config["diagnostic_layers"]],
            "primary_baseline": str(config["closed_loop"]["primary_baseline"]),
            "refresh_frequency": int(config["closed_loop"]["refresh_frequency"]),
            "full_cache_reference_exact_kl": 0.0,
            "publication_artifact": True,
            "merged_shards": [path.name for path in shard_paths],
            "sequences": len(expected_ids),
        },
    )
    return output_root


__all__ = [
    "hybrid_trigger",
    "merge_closed_loop_shards",
    "run_strict_causal_closed_loop",
    "select_validation_refresh_frequency",
]
