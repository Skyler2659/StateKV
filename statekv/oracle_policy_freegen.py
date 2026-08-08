"""Greedy closed-loop generation for StateKV's exact-risk teacher and baselines."""
from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

from src.evaluation.official_metrics import (
    longbench_score,
    normalize_answer,
    rouge_l_score,
    ruler_score,
)
from statekv.candidate_pullback import CandidatePullbackRunner
from statekv.config import CacheDiscoveryConfig, load_discovery_config
from statekv.core.decision import select_lowest_risk
from statekv.oracle_closed_loop import (
    CandidateRollout,
    KVBackingStore,
    _rollout_candidate,
)
from statekv.oracle_policy_comparison import (
    AttentionPolicyMemory,
    _core_map,
    _physical_candidate_panel,
)
from statekv.output_sensitivity_freegen import _ngram_f1, _repetition_rate
from statekv.selectors import CoreSelection, LayerSelection
from statekv.storage import atomic_frame, atomic_json, atomic_text
from statekv.tasks import load_discovery_tasks
from statekv.trajectory_model import exact_distribution_metrics


def _all_backing_selection(
    state: Any, backing: KVBackingStore
) -> CoreSelection:
    positions = backing.positions()
    return CoreSelection(
        strategy="full_backing",
        horizon_condition=None,
        by_layer={
            int(layer): LayerSelection(
                layer=int(layer),
                selected_positions=list(positions),
                eligible_positions=list(positions),
                aggregate_scores=[1.0] * len(positions),
            )
            for layer in range(len(state.cache))
        },
    )


def _clone_full_state(
    runner: CandidatePullbackRunner,
    state: Any,
    backing: KVBackingStore,
    current_token: int,
    reserve: int,
) -> Any:
    positions = backing.positions()
    cache = CacheDiscoveryConfig(
        total_budget=len(positions) + int(reserve) + 2,
        sink_size=0,
        recent_size=1,
        selected_core_budget=len(positions),
    )
    anchor = backing.anchor(
        int(state.logical_next_position), int(current_token)
    )
    cloned, _ = runner.model.state_from_anchor(
        anchor, _all_backing_selection(state, backing), cache_config=cache
    )
    return cloned


def _full_reference_segment(
    runner: CandidatePullbackRunner,
    state: Any,
    backing: KVBackingStore,
    current_token: int,
    horizon: int,
) -> Any:
    cloned = _clone_full_state(
        runner, state, backing, current_token, int(horizon)
    )
    token = int(current_token)
    targets: List[int] = []
    logits_by_step: Dict[int, torch.Tensor] = {}
    try:
        for offset in range(int(horizon)):
            logits, _, _ = runner.model.forward_one(
                cloned, token, capture_attention=True
            )
            logits_by_step[int(offset)] = logits.detach().float().cpu()
            token = int(torch.argmax(logits.float()).item())
            targets.append(token)
    finally:
        runner.model.release(cloned)
    return SimpleNamespace(
        generated_token_ids=targets,
        probe_logits=logits_by_step,
    )


def _free_rollout(
    runner: CandidatePullbackRunner,
    base_state: Any,
    backing: KVBackingStore,
    current_token: int,
    selection: CoreSelection,
    horizon: int,
    initial_cache: CacheDiscoveryConfig,
    rolling_cache: CacheDiscoveryConfig,
) -> Tuple[CandidateRollout, List[int]]:
    anchor = backing.anchor(
        int(base_state.logical_next_position), int(current_token)
    )
    state, fixed = runner.model.state_from_anchor(
        anchor, selection, cache_config=initial_cache
    )
    token = int(current_token)
    outputs: List[int] = []
    logits_rows: List[torch.Tensor] = []
    records: List[Any] = []
    maps_by_step: List[Dict[int, Tuple[int, ...]]] = []
    step_rows: List[Dict[str, Any]] = []
    maximum_active = 0
    last_record = None
    for offset in range(int(horizon)):
        if offset > 0:
            runner.model.prune_recent_before_query(
                state, fixed, cache_config=rolling_cache
            )
        runner._clear_controls()
        logits, record, forward_s = runner.model.forward_one(
            state, token, capture_attention=True
        )
        runner.model.validate_active_budget(
            state, cache_config=rolling_cache
        )
        active = int(runner.model.active_cache_tokens(state))
        maximum_active = max(maximum_active, active)
        output = int(torch.argmax(logits.float()).item())
        outputs.append(output)
        logits_rows.append(logits.detach().float().cpu())
        records.append(record)
        maps_by_step.append(
            {
                int(layer): tuple(
                    int(value) for value in positions.tolist()
                )
                for layer, positions in state.position_maps.items()
            }
        )
        step_rows.append(
            {
                "horizon_offset": int(offset + 1),
                "active_cache_tokens": active,
                "forward_time_s": float(forward_s),
            }
        )
        token = output
        last_record = record
    if last_record is None:
        raise RuntimeError("free rollout produced no tokens")
    return (
        CandidateRollout(
            name=str(selection.strategy),
            core=tuple(selection.by_layer[0].selected_positions),
            state=state,
            last_record=last_record,
            next_token=int(outputs[-1]),
            logits=logits_rows,
            step_rows=step_rows,
            records=records,
            position_maps_by_step=maps_by_step,
            maximum_active_tokens=maximum_active,
        ),
        outputs,
    )


def _advance_full_state(
    runner: CandidatePullbackRunner,
    state: Any,
    current_token: int,
    generated: Sequence[int],
    compressed_logits: Sequence[torch.Tensor],
) -> List[Dict[str, float]]:
    token = int(current_token)
    rows = []
    for output, candidate_logits in zip(generated, compressed_logits):
        full_logits, _, _ = runner.model.forward_one(
            state, token, capture_attention=True
        )
        rows.append(
            exact_distribution_metrics(
                full_logits, candidate_logits, int(output)
            )
        )
        token = int(output)
    return rows


def _metric_row(
    runner: CandidatePullbackRunner,
    sample: Any,
    policy: str,
    token_ids: Sequence[int],
    mean_trajectory_kl: float,
) -> Dict[str, Any]:
    text = runner.model.tokenizer.decode(
        [int(value) for value in token_ids], skip_special_tokens=True
    )
    references = [str(value) for value in sample.references]
    result: Dict[str, Any] = {
        "sample_id": str(sample.sample_id),
        "task": str(sample.task),
        "task_bucket": "GovReport" if "gov" in sample.task.lower() else "NIAH",
        "policy": str(policy),
        "generation_text": text,
        "generation_length_tokens": len(token_ids),
        "repetition_4gram_rate": _repetition_rate(text),
        "mean_trajectory_exact_kl": float(mean_trajectory_kl),
    }
    if "gov" in sample.task.lower():
        rouge_l = max(rouge_l_score(text, reference) for reference in references)
        result.update(
            {
                "rouge_l": float(rouge_l),
                "rouge_1": float(
                    max(_ngram_f1(text, reference, 1) for reference in references)
                ),
                "rouge_2": float(
                    max(_ngram_f1(text, reference, 2) for reference in references)
                ),
                "official_score": float(
                    longbench_score("gov_report", text, references) or 0.0
                ),
                "needle_retrieval_accuracy": None,
            }
        )
    else:
        normalized = normalize_answer(text)
        retrieval = any(
            normalize_answer(reference) in normalized for reference in references
        )
        result.update(
            {
                "rouge_l": None,
                "rouge_1": None,
                "rouge_2": None,
                "official_score": float(
                    ruler_score(sample.task, text, references) or 0.0
                ),
                "needle_retrieval_accuracy": float(retrieval),
            }
        )
    return result


def _run_free_policy(
    runner: CandidatePullbackRunner,
    reference: Any,
    sample: Any,
    policy: str,
    candidate_names: Sequence[str],
    start_anchor: int,
    cycles: int,
    horizon: int,
    total_budget: int,
    sink_size: int,
    recent_size: int,
    core_budget: int,
    observation_window: int,
    pooling_kernel: int,
    pooling_method: str,
    cheap_policy_context: Optional[Any] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    anchor_state = reference.anchors[int(start_anchor)]
    full_selection = runner._all_history_selection(reference, int(start_anchor))
    full_cache = CacheDiscoveryConfig(
        total_budget=int(anchor_state.logical_length + cycles * horizon + 2),
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
    backing.update(runner, compressed_state)
    full_backing.update(runner, full_state)
    memory = AttentionPolicyMemory.initialize(
        reference,
        int(start_anchor),
        range(len(compressed_state.cache)),
        int(observation_window),
        backing.positions(),
    )
    initial_cache = CacheDiscoveryConfig(
        total_budget=int(total_budget),
        sink_size=int(sink_size),
        recent_size=max(1, int(recent_size) - 1),
        selected_core_budget=int(core_budget),
    )
    rolling_cache = CacheDiscoveryConfig(
        total_budget=int(total_budget),
        sink_size=int(sink_size),
        recent_size=int(recent_size),
        selected_core_budget=int(core_budget),
    )
    previous_cores = None
    current_token = int(anchor_state.query_token_id)
    generated = [
        int(value)
        for value in reference.generated_token_ids[: int(start_anchor)]
    ]
    cycle_rows: List[Dict[str, Any]] = []
    step_rows: List[Dict[str, Any]] = []
    for cycle in range(int(cycles)):
        backing.update(runner, compressed_state)
        full_backing.update(runner, full_state)
        policy_decision_started = time.perf_counter()
        requires_panel = bool(
            cheap_policy_context is None
            or cheap_policy_context.requires_candidate_panel(policy)
        )
        panel = (
            _physical_candidate_panel(
                runner,
                compressed_state,
                backing,
                memory,
                previous_cores,
                candidate_names,
                sink_size,
                recent_size,
                core_budget,
                pooling_kernel,
                pooling_method,
            )
            if requires_panel
            else {}
        )
        teacher_scores: Dict[str, float] = {}
        if policy == "statekv_exact_mean":
            segment = _full_reference_segment(
                runner,
                full_state,
                full_backing,
                current_token,
                horizon,
            )
            teacher_outcomes = {
                name: _rollout_candidate(
                    runner,
                    segment,
                    compressed_state,
                    backing,
                    current_token,
                    selection,
                    0,
                    horizon,
                    initial_cache,
                    rolling_cache,
                )
                for name, selection in panel.items()
            }
            teacher_scores = {
                name: float(
                    np.mean(
                        [float(row["exact_kl"]) for row in outcome.step_rows]
                    )
                )
                for name, outcome in teacher_outcomes.items()
            }
            selected_name = str(
                select_lowest_risk(teacher_scores).candidate_id
            )
            for outcome in teacher_outcomes.values():
                outcome.state = None
            runner.model.release()
            selection_diagnostics: Dict[str, Any] = {}
        elif cheap_policy_context is not None:
            (
                selected_selection,
                selected_name,
                selection_diagnostics,
            ) = cheap_policy_context.select(
                policy,
                panel,
                memory,
                backing,
                previous_cores,
                cycle,
                cycles,
                str(sample.task),
            )
            selection_diagnostics = {
                **selection_diagnostics,
                "selection_time_s": float(
                    time.perf_counter() - policy_decision_started
                ),
                "candidate_model_rollouts": 0,
                "candidate_screening_rules": int(len(panel)),
            }
        else:
            selected_name = str(policy)
            selected_selection = panel[selected_name]
            selection_diagnostics = {}
        if cheap_policy_context is None or policy == "statekv_exact_mean":
            selected_selection = panel[selected_name]
        selected_core_counts = {
            int(layer): len(current.selected_positions)
            for layer, current in selected_selection.by_layer.items()
        }
        maximum_core = max(selected_core_counts.values(), default=core_budget)
        adaptive = policy == "b3_layer_adaptive_budget"
        current_initial_cache = initial_cache
        current_rolling_cache = rolling_cache
        if adaptive:
            current_initial_cache = CacheDiscoveryConfig(
                total_budget=int(sink_size + recent_size + maximum_core),
                sink_size=int(sink_size),
                recent_size=max(1, int(recent_size) - 1),
                selected_core_budget=int(maximum_core),
            )
            current_rolling_cache = CacheDiscoveryConfig(
                total_budget=int(sink_size + recent_size + maximum_core),
                sink_size=int(sink_size),
                recent_size=int(recent_size),
                selected_core_budget=int(maximum_core),
            )
        rollout, new_tokens = _free_rollout(
            runner,
            compressed_state,
            backing,
            current_token,
            selected_selection,
            horizon,
            current_initial_cache,
            current_rolling_cache,
        )
        trajectory_rows = _advance_full_state(
            runner,
            full_state,
            current_token,
            new_tokens,
            rollout.logits,
        )
        for offset, metrics in enumerate(trajectory_rows):
            step_rows.append(
                {
                    "policy": str(policy),
                    "cycle": int(cycle),
                    "horizon_offset": int(offset + 1),
                    "generated_token_id": int(new_tokens[offset]),
                    **metrics,
                }
            )
        selected_cores = _core_map(selected_selection)
        stale_cores = (
            _core_map(panel["stale"])
            if "stale" in panel
            else previous_cores
            if previous_cores is not None
            else selected_cores
        )
        input_maps = {
            int(layer): set(int(value) for value in positions.tolist())
            for layer, positions in compressed_state.position_maps.items()
        }
        active_lengths = [
            len(rollout.state.position_maps[int(layer)])
            for layer in sorted(rollout.state.position_maps)
        ]
        layer_count = len(active_lengths)
        selected_core_total = int(sum(selected_core_counts.values()))
        nominal_core_total = int(layer_count * core_budget)
        if adaptive and selected_core_total != nominal_core_total:
            raise RuntimeError(
                "adaptive policy changed the global retained-core budget"
            )
        budget_respected = (
            sum(active_lengths) <= int(layer_count * total_budget)
            if adaptive
            else rollout.maximum_active_tokens <= int(total_budget)
        )
        cycle_rows.append(
            {
                "policy": str(policy),
                "cycle": int(cycle),
                "selected_candidate": selected_name,
                "refresh": selected_cores != stale_cores,
                "selected_recovered_layer_tokens": int(
                    sum(
                        len(set(core) - input_maps[int(layer)])
                        for layer, core in selected_cores.items()
                    )
                ),
                "mean_trajectory_exact_kl": float(
                    np.mean([row["exact_kl"] for row in trajectory_rows])
                ),
                "mean_trajectory_delta_nll": float(
                    np.mean([row["delta_nll"] for row in trajectory_rows])
                ),
                "teacher_selected_risk": (
                    float(teacher_scores[selected_name])
                    if teacher_scores
                    else None
                ),
                "maximum_active_cache_tokens": int(
                    rollout.maximum_active_tokens
                ),
                "mean_active_cache_tokens": float(np.mean(active_lengths)),
                "total_active_layer_tokens": int(sum(active_lengths)),
                "selected_core_tokens_total": selected_core_total,
                "maximum_configured_layer_budget": int(
                    current_rolling_cache.total_budget
                ),
                "budget_respected": bool(budget_respected),
                **selection_diagnostics,
            }
        )
        memory.update_rollout(rollout)
        compressed_state = rollout.state
        backing.update(runner, compressed_state)
        full_backing.update(runner, full_state)
        previous_cores = selected_cores
        current_token = int(new_tokens[-1])
        generated.extend(int(value) for value in new_tokens)
    mean_kl = float(np.mean([row["exact_kl"] for row in step_rows]))
    metric = _metric_row(runner, sample, policy, generated, mean_kl)
    summary = {
        "policy": str(policy),
        "cycles_completed": len(cycle_rows),
        "generated_tokens": len(generated),
        "mean_trajectory_exact_kl": mean_kl,
        "refresh_events": int(sum(row["refresh"] for row in cycle_rows)),
        "recovery_events": int(
            sum(row["selected_recovered_layer_tokens"] > 0 for row in cycle_rows)
        ),
        "all_budgets_respected": bool(
            all(row["budget_respected"] for row in cycle_rows)
        ),
        **metric,
    }
    runner.model.release(compressed_state, full_state)
    return cycle_rows, step_rows, summary


def _paired_bootstrap_interval(
    values: Sequence[float], seed: int, samples: int
) -> Tuple[float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return float("nan"), float("nan")
    if array.size == 1:
        value = float(array[0])
        return value, value
    rng = np.random.default_rng(int(seed))
    draws = rng.choice(
        array,
        size=(max(1, int(samples)), int(array.size)),
        replace=True,
    ).mean(axis=1)
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return float(lower), float(upper)


def _aggregate_free_results(
    frame: pd.DataFrame,
    bootstrap_seed: int = 0,
    bootstrap_samples: int = 20000,
) -> Dict[str, Any]:
    rows = []
    for policy, current in frame.groupby("policy", sort=True):
        rows.append(
            {
                "policy": str(policy),
                "samples": int(len(current)),
                "mean_trajectory_exact_kl": float(
                    current["mean_trajectory_exact_kl"].mean()
                ),
                "mean_official_score": float(current["official_score"].mean()),
                "mean_govreport_rouge_l": (
                    float(
                        current.loc[
                            current["task_bucket"] == "GovReport", "rouge_l"
                        ].mean()
                    )
                    if (current["task_bucket"] == "GovReport").any()
                    else None
                ),
                "mean_niah_retrieval": (
                    float(
                        current.loc[
                            current["task_bucket"] == "NIAH",
                            "needle_retrieval_accuracy",
                        ].mean()
                    )
                    if (current["task_bucket"] == "NIAH").any()
                    else None
                ),
            }
        )
    by_policy = {row["policy"]: row for row in rows}
    statekv = by_policy["statekv_exact_mean"]
    comparisons = []
    fixed_policies = ("attention", "snapkv", "h2o")
    for baseline_index, baseline in enumerate(fixed_policies):
        current = by_policy[baseline]
        statekv_samples = frame.loc[
            frame["policy"] == "statekv_exact_mean",
            ["sample_id", "official_score", "mean_trajectory_exact_kl"],
        ].set_index("sample_id")
        baseline_samples = frame.loc[
            frame["policy"] == baseline,
            ["sample_id", "official_score", "mean_trajectory_exact_kl"],
        ].set_index("sample_id")
        paired = statekv_samples.join(
            baseline_samples,
            how="inner",
            lsuffix="_statekv",
            rsuffix="_baseline",
        )
        quality_delta = (
            paired["official_score_statekv"]
            - paired["official_score_baseline"]
        ).to_numpy(dtype=np.float64)
        quality_ci = _paired_bootstrap_interval(
            quality_delta,
            int(bootstrap_seed) + baseline_index,
            int(bootstrap_samples),
        )
        tolerance = 1.0e-12
        comparisons.append(
            {
                "baseline": baseline,
                "trajectory_kl_baseline_minus_statekv": (
                    current["mean_trajectory_exact_kl"]
                    - statekv["mean_trajectory_exact_kl"]
                ),
                "official_score_statekv_minus_baseline": (
                    statekv["mean_official_score"]
                    - current["mean_official_score"]
                ),
                "govreport_rouge_l_statekv_minus_baseline": (
                    statekv["mean_govreport_rouge_l"]
                    - current["mean_govreport_rouge_l"]
                ),
                "niah_retrieval_statekv_minus_baseline": (
                    statekv["mean_niah_retrieval"]
                    - current["mean_niah_retrieval"]
                ),
                "paired_samples": int(len(paired)),
                "official_score_delta_ci95": list(quality_ci),
                "official_score_sample_wins": int(
                    np.sum(quality_delta > tolerance)
                ),
                "official_score_sample_ties": int(
                    np.sum(np.abs(quality_delta) <= tolerance)
                ),
                "official_score_sample_losses": int(
                    np.sum(quality_delta < -tolerance)
                ),
            }
        )
    best_fixed_quality = max(
        (by_policy[policy] for policy in fixed_policies),
        key=lambda row: row["mean_official_score"],
    )
    best_fixed_kl = min(
        (by_policy[policy] for policy in fixed_policies),
        key=lambda row: row["mean_trajectory_exact_kl"],
    )
    return {
        "policy_aggregates": rows,
        "paired_comparisons": comparisons,
        "overall_comparisons": {
            "best_fixed_quality_policy": best_fixed_quality["policy"],
            "mean_official_score_statekv_minus_best_fixed": float(
                statekv["mean_official_score"]
                - best_fixed_quality["mean_official_score"]
            ),
            "best_fixed_kl_policy": best_fixed_kl["policy"],
            "mean_trajectory_kl_best_fixed_minus_statekv": float(
                best_fixed_kl["mean_trajectory_exact_kl"]
                - statekv["mean_trajectory_exact_kl"]
            ),
        },
    }


def run_oracle_policy_freegen(config_path: Path, repository_root: Path) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cfg = load_discovery_config(str(repository_root / str(config["base_config"])))
    cfg.experiment_name = str(config["experiment_name"])
    for key, value in dict(config.get("model_overrides") or {}).items():
        if not hasattr(cfg.model, str(key)):
            raise ValueError("unknown model override: %s" % key)
        setattr(cfg.model, str(key), value)
    cfg.tasks = dict(config["task_overrides"])
    cfg.runtime.seed = int(config["data_seed"])
    cfg.runtime.run_id = str(config["runtime_run_id"])
    sample_ids = set(str(value) for value in config["sample_ids"])
    expected_sample_count = int(
        config.get("expected_sample_count", len(sample_ids))
    )
    output_root = repository_root / str(config["output_run"])
    output_root.mkdir(parents=True, exist_ok=True)
    policies = [str(value) for value in config["policies"]]
    candidates = [str(value) for value in config["candidate_panel"]]
    start_anchor = int(config["start_anchor"])
    # Free-generation starts from one anchor. Retaining unrelated later anchors
    # duplicates the full prompt KV tensors without changing any policy action.
    cfg.anchor_steps = [start_anchor]
    cycles = int(config["control_cycles"])
    horizon = int(config["control_horizon"])
    total_budget = int(config["total_budget"])
    sink_size = int(config["sink_size"])
    recent_size = int(config["recent_size"])
    core_budget = int(config["core_budget"])
    cfg.cache.total_budget = total_budget
    cfg.cache.sink_size = sink_size
    cfg.cache.recent_size = recent_size
    cfg.cache.selected_core_budget = core_budget
    cfg.validate()
    runner = CandidatePullbackRunner(cfg, repository_root)
    samples, _ = load_discovery_tasks(cfg)
    selected_samples = [
        sample for sample in samples if str(sample.sample_id) in sample_ids
    ]
    if {str(sample.sample_id) for sample in selected_samples} != sample_ids:
        raise RuntimeError("configured free-generation samples were not loaded")
    if len(selected_samples) != expected_sample_count:
        raise RuntimeError(
            "expected %d free-generation samples, loaded %d"
            % (expected_sample_count, len(selected_samples))
        )
    cycle_rows: List[Dict[str, Any]] = []
    step_rows: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    started = time.perf_counter()
    model_info = runner.model.load()
    print(
        "[freegen] loaded %s with %d/%d attention hooks"
        % (
            model_info.get("model_name"),
            int(model_info.get("attention_hooked_layers") or 0),
            int(model_info.get("attention_hook_expected_layers") or 0),
        ),
        flush=True,
    )
    try:
        for sample_index, sample in enumerate(selected_samples, start=1):
            reference = runner.model.generate_reference(
                sample.sample_id, sample.task, sample.prompt
            )
            try:
                prompt_tokens = int(len(reference.prompt_token_ids))
                print(
                    "[freegen] sample %d/%d %s prompt_tokens=%d"
                    % (
                        sample_index,
                        len(selected_samples),
                        sample.sample_id,
                        prompt_tokens,
                    ),
                    flush=True,
                )
                reference_tokens = [
                    int(value)
                    for value in reference.generated_token_ids[
                        : start_anchor + cycles * horizon
                    ]
                ]
                summaries.append(
                    {
                        "policy": "full_cache",
                        "cycles_completed": cycles,
                        "generated_tokens": len(reference_tokens),
                        "mean_trajectory_exact_kl": 0.0,
                        "refresh_events": 0,
                        "recovery_events": 0,
                        "all_budgets_respected": True,
                        "prompt_tokens": prompt_tokens,
                        "cache_budget": None,
                        "retained_prompt_fraction": 1.0,
                        **_metric_row(
                            runner,
                            sample,
                            "full_cache",
                            reference_tokens,
                            0.0,
                        ),
                    }
                )
                for policy in policies:
                    cycles_current, steps_current, summary = _run_free_policy(
                        runner,
                        reference,
                        sample,
                        policy,
                        candidates,
                        start_anchor,
                        cycles,
                        horizon,
                        total_budget,
                        sink_size,
                        recent_size,
                        core_budget,
                        int(config["snapkv_observation_window"]),
                        int(config["snapkv_pooling_kernel"]),
                        str(config["snapkv_pooling"]),
                    )
                    summary["prompt_tokens"] = prompt_tokens
                    summary["cache_budget"] = total_budget
                    summary["retained_prompt_fraction"] = float(
                        min(1.0, total_budget / max(1, prompt_tokens))
                    )
                    base = {
                        "sample_id": str(sample.sample_id),
                        "task": str(sample.task),
                    }
                    cycle_rows.extend({**base, **row} for row in cycles_current)
                    step_rows.extend({**base, **row} for row in steps_current)
                    summaries.append(summary)
                    print(
                        "[freegen] sample %d/%d policy=%s kl=%.6f score=%.4f"
                        % (
                            sample_index,
                            len(selected_samples),
                            policy,
                            float(summary["mean_trajectory_exact_kl"]),
                            float(summary["official_score"]),
                        ),
                        flush=True,
                    )
                atomic_frame(
                    pd.DataFrame(summaries),
                    output_root / "partial_sample_results.csv",
                )
                atomic_frame(
                    pd.DataFrame(cycle_rows),
                    output_root / "partial_cycle_rows.parquet",
                )
                atomic_frame(
                    pd.DataFrame(step_rows),
                    output_root / "partial_step_rows.parquet",
                )
                atomic_json(
                    output_root / "progress.json",
                    {
                        "completed_samples": sample_index,
                        "expected_samples": expected_sample_count,
                        "last_sample_id": str(sample.sample_id),
                        "elapsed_s": float(time.perf_counter() - started),
                    },
                )
            finally:
                runner.model.release(reference)
    finally:
        runner.model.close()
    summary_frame = pd.DataFrame(summaries)
    aggregates = _aggregate_free_results(
        summary_frame,
        bootstrap_seed=int(config["data_seed"]),
        bootstrap_samples=int(config.get("bootstrap_samples", 20000)),
    )
    comparisons = aggregates["paired_comparisons"]
    overall = aggregates["overall_comparisons"]
    result = {
        "experiment": str(config["experiment_name"]),
        "status": "greedy_policy_conditioned_physical_closed_loop_generation",
        "samples": sorted(sample_ids),
        "policies": policies,
        "control_cycles": cycles,
        "control_horizon": horizon,
        "total_budget": total_budget,
        "model_info": model_info,
        "sample_results": summaries,
        **aggregates,
        "all_budgets_respected": bool(
            summary_frame["all_budgets_respected"].all()
        ),
        "statekv_lower_trajectory_kl_than_each_fixed_policy": bool(
            all(
                row["trajectory_kl_baseline_minus_statekv"] > 0.0
                for row in comparisons
            )
        ),
        "statekv_task_metrics_nonworse_than_each_fixed_policy": bool(
            all(
                row["govreport_rouge_l_statekv_minus_baseline"] >= 0.0
                and row["niah_retrieval_statekv_minus_baseline"] >= 0.0
                for row in comparisons
            )
        ),
        "strict_pareto_diagnostic": bool(
            all(
                row["trajectory_kl_baseline_minus_statekv"] > 0.0
                and row["govreport_rouge_l_statekv_minus_baseline"] >= 0.0
                and row["niah_retrieval_statekv_minus_baseline"] >= 0.0
                for row in comparisons
            )
        ),
        "statekv_highest_mean_official_score_among_compressed": bool(
            overall["mean_official_score_statekv_minus_best_fixed"] >= 0.0
        ),
        "statekv_lowest_mean_kl_among_compressed": bool(
            overall["mean_trajectory_kl_best_fixed_minus_statekv"] >= 0.0
        ),
        "collection_elapsed_s": float(time.perf_counter() - started),
    }
    result["scientific_outcome"] = (
        "joint_quality_and_fidelity_support"
        if result["statekv_highest_mean_official_score_among_compressed"]
        and result["statekv_lowest_mean_kl_among_compressed"]
        else "quality_support_only"
        if result["statekv_highest_mean_official_score_among_compressed"]
        else "fidelity_support_only"
        if result["statekv_lowest_mean_kl_among_compressed"]
        else "no_overall_support"
    )
    result["execution_valid"] = bool(
        result["all_budgets_respected"]
        and len(selected_samples) == expected_sample_count
    )
    result["passed"] = result["execution_valid"]
    atomic_frame(pd.DataFrame(cycle_rows), output_root / "cycle_rows.parquet")
    atomic_frame(pd.DataFrame(step_rows), output_root / "step_rows.parquet")
    atomic_frame(summary_frame, output_root / "sample_results.csv")
    atomic_json(output_root / "summary.json", result)
    atomic_text(
        output_root / "config.yaml",
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
    )
    return output_root


__all__ = ["run_oracle_policy_freegen"]
