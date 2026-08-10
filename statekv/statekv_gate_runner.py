"""Execution paths for StateKV's dynamic-budget, pure-eviction, and tail gates."""
from __future__ import annotations

import gc
import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import psutil
import torch
import yaml

from statekv.budget_dynamics import (
    DirectBudgetController,
    FrozenRanking,
    MECHANISM_POLICIES,
    PURE_EVICTION_POLICIES,
    active_cache_view,
    average_static_budgets,
    backing_cache_view,
    boundary_margin_by_layer,
    core_churn_by_layer,
    coverage_mass_by_layer,
    mask_overlap,
    score_tv_by_layer,
)
from statekv.selectors import CoreSelection, LayerSelection, mandatory_and_eligible
from statekv.candidate_pullback import CandidatePullbackRunner
from statekv.config import CacheDiscoveryConfig, load_discovery_config
from statekv.oracle_closed_loop import KVBackingStore
from statekv.oracle_policy_comparison import AttentionPolicyMemory
from statekv.oracle_policy_freegen import _free_rollout, _metric_row
from statekv.refresh_trigger import (
    CHEAP_TRIGGER_FEATURES,
    TriggerRule,
    decide_trigger_refresh,
    load_trigger_rule,
)
from statekv.storage import atomic_frame, atomic_json, atomic_text
from statekv.tasks import load_discovery_tasks
from statekv.trajectory_model import exact_distribution_metrics


def _json_vector(values: Mapping[int, Any]) -> str:
    return json.dumps([values[layer] for layer in sorted(values)])


class _MemorySampler:
    def __init__(self, interval_s: float = 0.02) -> None:
        self.interval_s = float(interval_s)
        self.process = psutil.Process(os.getpid())
        self.baseline = int(self.process.memory_info().rss)
        self.peak = self.baseline
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            try:
                self.peak = max(self.peak, int(self.process.memory_info().rss))
            except psutil.Error:
                return

    def __enter__(self) -> "_MemorySampler":
        self._thread.start()
        return self

    def __exit__(self, *_args: Any) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        try:
            self.peak = max(self.peak, int(self.process.memory_info().rss))
        except psutil.Error:
            pass


def _softmax_row(
    full_logits: torch.Tensor,
    compressed_logits: torch.Tensor,
    monitored_ids: Mapping[str, Sequence[int]],
) -> Dict[str, Any]:
    full_probability = torch.softmax(full_logits.detach().double().cpu(), dim=-1)
    compressed_probability = torch.softmax(
        compressed_logits.detach().double().cpu(), dim=-1
    )

    def distribution_stats(probability: torch.Tensor) -> Tuple[float, float, int]:
        top = torch.topk(probability, k=2)
        margin = float((top.values[0] - top.values[1]).item())
        entropy = float(
            (-(probability * torch.log(torch.clamp(probability, min=1.0e-15))).sum()).item()
        )
        return margin, entropy, int(top.indices[0].item())

    full_margin, full_entropy, full_argmax = distribution_stats(full_probability)
    compressed_margin, compressed_entropy, compressed_argmax = distribution_stats(
        compressed_probability
    )
    result: Dict[str, Any] = {
        "full_margin": full_margin,
        "compressed_margin": compressed_margin,
        "full_entropy": full_entropy,
        "compressed_entropy": compressed_entropy,
        "full_argmax_token_id": full_argmax,
        "compressed_argmax_token_id": compressed_argmax,
        "argmax_diverged": full_argmax != compressed_argmax,
    }
    for label, token_ids in monitored_ids.items():
        ids = sorted(set(int(value) for value in token_ids))
        result[f"full_probability_{label}"] = float(
            full_probability[ids].sum().item()
        ) if ids else 0.0
        result[f"compressed_probability_{label}"] = float(
            compressed_probability[ids].sum().item()
        ) if ids else 0.0
    return result


def _advance_full_state(
    runner: CandidatePullbackRunner,
    full_state: Any,
    current_token: int,
    outputs: Sequence[int],
    compressed_logits: Sequence[torch.Tensor],
    monitored_ids: Mapping[str, Sequence[int]],
) -> List[Dict[str, Any]]:
    token = int(current_token)
    rows: List[Dict[str, Any]] = []
    for output, candidate_logits in zip(outputs, compressed_logits):
        full_logits, _, forward_s = runner.model.forward_one(
            full_state, token, capture_attention=True
        )
        rows.append(
            {
                **exact_distribution_metrics(
                    full_logits, candidate_logits, int(output)
                ),
                **_softmax_row(full_logits, candidate_logits, monitored_ids),
                "full_forward_time_s": float(forward_s),
            }
        )
        token = int(output)
    return rows


def _find_subsequence(sequence: Sequence[int], query: Sequence[int]) -> List[int]:
    if not query or len(query) > len(sequence):
        return []
    matches: List[int] = []
    width = len(query)
    for start in range(len(sequence) - width + 1):
        if list(sequence[start : start + width]) == list(query):
            matches.extend(range(start, start + width))
    return sorted(set(matches))


def _monitor_spec(
    tokenizer: Any,
    prompt_ids: Sequence[int],
    probability_labels: Mapping[str, Sequence[str]],
    evidence_phrases: Sequence[str],
) -> Tuple[Dict[str, List[int]], List[int], Dict[str, List[int]]]:
    monitored: Dict[str, List[int]] = {}
    encodings: Dict[str, List[int]] = {}
    for label, variants in probability_labels.items():
        ids = set()
        for text in variants:
            encoded = [
                int(value)
                for value in tokenizer.encode(
                    str(text), add_special_tokens=False
                )
            ]
            encodings[f"probability:{label}:{text}"] = encoded
            if encoded:
                # Qwen tokenizes numbers digit by digit.  At the factual branch
                # after the shared leading digit, the final token is the exact
                # conditional choice between strings such as "17" and "14".
                ids.add(encoded[-1])
        monitored[str(label)] = sorted(ids)
    evidence = set()
    for phrase in evidence_phrases:
        encoded = [
            int(value)
            for value in tokenizer.encode(
                str(phrase), add_special_tokens=False
            )
        ]
        encodings[f"evidence:{phrase}"] = encoded
        evidence.update(_find_subsequence(prompt_ids, encoded))
    return monitored, sorted(evidence), encodings


def _evidence_survival(
    position_maps: Mapping[int, Any], evidence_positions: Sequence[int]
) -> Dict[str, Any]:
    evidence = set(int(value) for value in evidence_positions)
    fractions = []
    counts = []
    for layer in sorted(position_maps):
        active = set(int(value) for value in position_maps[layer].tolist())
        count = len(evidence & active)
        counts.append(count)
        fractions.append(count / max(1, len(evidence)))
    return {
        "evidence_token_count": int(len(evidence)),
        "evidence_survival_by_layer_json": json.dumps(fractions),
        "evidence_surviving_tokens_by_layer_json": json.dumps(counts),
        "evidence_layers_with_full_survival": int(
            sum(value == len(evidence) for value in counts)
        ) if evidence else 0,
    }


def _counterfactual_overlap(
    controller: DirectBudgetController,
    decision: Any,
    memory: AttentionPolicyMemory,
    view: Any,
    cycle: int,
    sample_id: str,
) -> Dict[str, Any]:
    kwargs = {
        "core_budget": controller.core_budget,
        "sink_size": controller.sink_size,
        "recent_size": controller.recent_size,
        "pooling_kernel": controller.pooling_kernel,
        "pooling_method": controller.pooling_method,
        "maximum_delta": controller.maximum_delta,
    }
    a2 = DirectBudgetController(**kwargs).select(
        "a2_temporal_volatility", memory, view, cycle, sample_id
    )
    attention = DirectBudgetController(**kwargs).select(
        "attention", memory, view, cycle, sample_id
    )
    a2_overlap = mask_overlap(decision.selection, a2.selection)
    attention_overlap = mask_overlap(decision.selection, attention.selection)
    return {
        "a2_mask_mean_jaccard": a2_overlap["mean_jaccard"],
        "a2_mask_minimum_jaccard": a2_overlap["minimum_jaccard"],
        "a2_mask_jaccard_by_layer_json": a2_overlap["jaccard_by_layer_json"],
        "attention_mask_mean_jaccard": attention_overlap["mean_jaccard"],
        "attention_mask_minimum_jaccard": attention_overlap["minimum_jaccard"],
        "attention_mask_jaccard_by_layer_json": attention_overlap[
            "jaccard_by_layer_json"
        ],
    }


def _cache_configs(
    sink_size: int,
    recent_size: int,
    maximum_core: int,
) -> Tuple[CacheDiscoveryConfig, CacheDiscoveryConfig]:
    total = int(sink_size + recent_size + maximum_core)
    return (
        CacheDiscoveryConfig(
            total_budget=total,
            sink_size=int(sink_size),
            recent_size=max(1, int(recent_size) - 1),
            selected_core_budget=int(maximum_core),
        ),
        CacheDiscoveryConfig(
            total_budget=total,
            sink_size=int(sink_size),
            recent_size=int(recent_size),
            selected_core_budget=int(maximum_core),
        ),
    )


def run_backed_policy(
    runner: CandidatePullbackRunner,
    reference: Any,
    sample: Any,
    policy: str,
    controller: DirectBudgetController,
    cycles: int,
    horizon: int,
    monitor_labels: Mapping[str, Sequence[int]],
    evidence_positions: Sequence[int],
    evaluate_exact_kl: bool = True,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    anchor_state = reference.anchors[0]
    full_selection = runner._all_history_selection(reference, 0)
    full_cache = CacheDiscoveryConfig(
        total_budget=int(anchor_state.logical_length + cycles * horizon + 2),
        sink_size=0,
        recent_size=1,
        selected_core_budget=int(anchor_state.logical_length + 1),
    )
    compressed_state, _ = runner.model.state_from_anchor(
        anchor_state, full_selection, cache_config=full_cache
    )
    full_state = None
    if evaluate_exact_kl:
        full_state, _ = runner.model.state_from_anchor(
            anchor_state, full_selection, cache_config=full_cache
        )
    backing = KVBackingStore()
    backing.update(runner, compressed_state)
    memory = AttentionPolicyMemory.initialize(
        reference,
        0,
        range(len(compressed_state.cache)),
        int(controller.recent_size),
        backing.positions(),
    )
    current_token = int(anchor_state.query_token_id)
    generated: List[int] = []
    cycle_rows: List[Dict[str, Any]] = []
    step_rows: List[Dict[str, Any]] = []
    previous_budget: Optional[Mapping[int, int]] = None
    started = time.perf_counter()
    selection_total = 0.0
    forward_total = 0.0
    try:
        for cycle in range(int(cycles)):
            backing.update(runner, compressed_state)
            view = backing_cache_view(backing, range(len(compressed_state.cache)))
            selection_started = time.perf_counter()
            decision = controller.select(
                policy, memory, view, cycle, str(sample.sample_id)
            )
            selection_time = float(time.perf_counter() - selection_started)
            telemetry_started = time.perf_counter()
            overlaps = _counterfactual_overlap(
                controller,
                decision,
                memory,
                view,
                cycle,
                str(sample.sample_id),
            )
            telemetry_time = float(time.perf_counter() - telemetry_started)
            selection_total += selection_time
            maximum_core = max(decision.requested_budgets.values())
            initial_cache, rolling_cache = _cache_configs(
                controller.sink_size, controller.recent_size, maximum_core
            )
            input_maps = {
                int(layer): set(int(value) for value in positions.tolist())
                for layer, positions in compressed_state.position_maps.items()
            }
            rollout, outputs = _free_rollout(
                runner,
                compressed_state,
                backing,
                current_token,
                decision.selection,
                horizon,
                initial_cache,
                rolling_cache,
            )
            forward_total += float(
                sum(float(row["forward_time_s"]) for row in rollout.step_rows)
            )
            metrics = (
                _advance_full_state(
                    runner,
                    full_state,
                    current_token,
                    outputs,
                    rollout.logits,
                    monitor_labels,
                )
                if full_state is not None
                else [{} for _ in outputs]
            )
            for offset, (output, row) in enumerate(zip(outputs, metrics)):
                step_rows.append(
                    {
                        "policy": str(policy),
                        "cycle": int(cycle),
                        "horizon_offset": int(offset + 1),
                        "input_token_id": int(
                            current_token if offset == 0 else outputs[offset - 1]
                        ),
                        "generated_token_id": int(output),
                        **row,
                    }
                )
            output_maps = rollout.state.position_maps
            selected_cores = {
                int(layer): tuple(
                    int(value)
                    for value in current.selected_positions
                )
                for layer, current in decision.selection.by_layer.items()
            }
            recovery = int(
                sum(
                    len(set(core) - input_maps[layer])
                    for layer, core in selected_cores.items()
                )
            )
            budget_change = (
                0
                if previous_budget is None
                else int(
                    sum(
                        abs(
                            int(decision.requested_budgets[layer])
                            - int(previous_budget[layer])
                        )
                        for layer in decision.requested_budgets
                    )
                )
            )
            active_lengths = {
                int(layer): len(positions)
                for layer, positions in output_maps.items()
            }
            current_metrics = metrics[-1] if metrics else {}
            cycle_rows.append(
                {
                    "policy": str(policy),
                    "cycle": int(cycle),
                    "selection_time_s": selection_time,
                    "telemetry_time_s": telemetry_time,
                    "budget_l1_change": int(budget_change),
                    "selected_recovered_layer_tokens": recovery,
                    "maximum_active_cache_tokens": int(max(active_lengths.values())),
                    "total_active_layer_tokens": int(sum(active_lengths.values())),
                    "active_tokens_by_layer_json": _json_vector(active_lengths),
                    "mean_trajectory_exact_kl": current_metrics.get("exact_kl"),
                    **decision.diagnostics,
                    **overlaps,
                    **_evidence_survival(output_maps, evidence_positions),
                }
            )
            memory.update_rollout(rollout)
            compressed_state = rollout.state
            current_token = int(outputs[-1])
            generated.extend(int(value) for value in outputs)
            previous_budget = dict(decision.requested_budgets)
        mean_kl = (
            float(np.mean([float(row["exact_kl"]) for row in step_rows]))
            if evaluate_exact_kl
            else float("nan")
        )
        metric = _metric_row(runner, sample, policy, generated, mean_kl)
        summary = {
            "policy": str(policy),
            "cycles_completed": len(cycle_rows),
            "mean_trajectory_exact_kl": mean_kl,
            "official_score": float(metric["official_score"]),
            "wall_time_s": float(time.perf_counter() - started),
            "selection_time_s_total": float(selection_total),
            "decode_forward_time_s_total": float(forward_total),
            "controller_overhead_fraction": float(
                selection_total / max(selection_total + forward_total, 1.0e-12)
            ),
            "decode_tokens_per_s": float(
                len(generated) / max(forward_total, 1.0e-12)
            ),
            "controller_decode_tokens_per_s": float(
                len(generated) / max(forward_total + selection_total, 1.0e-12)
            ),
            "end_to_end_tokens_per_s": float(
                len(generated) / max(time.perf_counter() - started, 1.0e-12)
            ),
            "fixed_global_requested_core_budget": bool(
                all(
                    int(row["requested_core_tokens_total"])
                    == len(compressed_state.cache) * int(controller.core_budget)
                    for row in cycle_rows
                )
            ),
            **metric,
        }
        return cycle_rows, step_rows, summary
    finally:
        runner.model.release(compressed_state, full_state)


def _scheduled_refresh(
    refresh_mode: str, cycle: int, refresh_k: int, label_mode: bool = False
) -> bool:
    """Calendar-based refresh decision (every/never/fixed_k label collection)."""
    return bool(
        label_mode
        or refresh_mode == "every"
        or int(cycle) == 0
        or (refresh_mode == "fixed_k" and int(cycle) % int(refresh_k) == 0)
    )


def _trigger_features(
    candidate: Any,
    previous_fresh: Optional[
        Tuple[Mapping[int, Tuple[int, ...]], Mapping[int, np.ndarray], Any]
    ],
    positions_by_layer: Mapping[int, Any],
    core_budget: int,
) -> Dict[str, float]:
    """Online-computable cheap features of a candidate fresh decision.

    Only uses controller-side scores/selections; no teacher quantities.
    """
    features: Dict[str, float] = {}
    if previous_fresh is not None:
        churn = core_churn_by_layer(previous_fresh[2], candidate.selection)
        television = score_tv_by_layer(
            previous_fresh[1],
            previous_fresh[0],
            candidate.scores_by_layer,
            positions_by_layer,
        )
        features["churn_jaccard_mean"] = float(np.mean(list(churn.values())))
        features["score_tv_mean"] = float(np.mean(list(television.values())))
    else:
        features["churn_jaccard_mean"] = float("nan")
        features["score_tv_mean"] = float("nan")
    margins = boundary_margin_by_layer(
        candidate.scores_by_layer,
        positions_by_layer,
        candidate.eligible_by_layer,
        int(core_budget),
    )
    coverage = coverage_mass_by_layer(
        candidate.scores_by_layer,
        positions_by_layer,
        candidate.selection,
    )
    features["boundary_margin_mean"] = float(np.mean(list(margins.values())))
    features["coverage_mass_mean"] = float(np.mean(list(coverage.values())))
    return features


def _ladder_rollout(
    runner: CandidatePullbackRunner,
    compressed_state: Any,
    current_token: int,
    reference: Any,
    candidate: Any,
    rolling_cache: CacheDiscoveryConfig,
    sink_size: int,
    horizons: Sequence[int],
    probe_offset: int,
) -> List[Dict[str, Any]]:
    """Teacher-forced multi-horizon risk ladder for one candidate action.

    Step 1 consumes ``current_token`` (the actual compressed query); steps
    >= 2 feed reference tokens, mirroring the P31 re-anchor rollout
    semantics.  Each step's risk is the exact KL against the reference
    (full-cache) logits ``reference.probe_logits[probe_offset + step - 1]``.
    The state is a clone of the *surviving* cache pruned to the candidate
    selection (pure eviction: no deleted history is read).  Returns one row
    per horizon with the stepwise KLs.
    """

    clone = runner.model.shallow_clone_state(compressed_state)
    try:
        runner.model.apply_selection_in_place(
            clone, candidate.selection, cache_config=rolling_cache
        )
        sink_positions = {
            int(layer): set(
                int(value)
                for value in clone.position_maps[layer].tolist()[
                    : min(int(sink_size), len(clone.position_maps[layer]))
                ]
            )
            for layer in clone.position_maps
        }
        fixed = {
            int(layer): sink_positions[int(layer)]
            | set(
                int(value)
                for value in candidate.selection.by_layer[int(layer)].selected_positions
            )
            for layer in clone.position_maps
        }
        token = int(current_token)
        step_kls: List[float] = []
        rows: List[Dict[str, Any]] = []
        for step in range(1, int(max(horizons)) + 1):
            if step > 1:
                runner.model.prune_recent_before_query(
                    clone, fixed, cache_config=rolling_cache
                )
            logits, _, _ = runner.model.forward_one(
                clone, token, capture_attention=False
            )
            target_index = int(probe_offset + step - 1)
            reference_logits = reference.probe_logits[target_index]
            deltas = exact_distribution_metrics(
                reference_logits,
                logits,
                int(reference.generated_token_ids[target_index]),
            )
            step_kls.append(float(deltas["exact_kl"]))
            if step in horizons:
                rows.append(
                    {
                        "horizon": int(step),
                        "exact_kl": float(deltas["exact_kl"]),
                        "js": float(deltas["js"]),
                        "logit_l2_sq": float(deltas["logit_l2_sq"]),
                        "cumulative_kl": float(np.mean(step_kls)),
                    }
                )
            token = int(reference.generated_token_ids[target_index])
        return rows
    finally:
        del clone


def _swap_selection(
    decision: Any,
    layer: int,
    remove_positions: Sequence[int],
    add_positions: Sequence[int],
) -> CoreSelection:
    """Return a decision selection with layer-0 core positions swapped.

    The swap respects the per-layer budget: remove and add the same count.
    """
    by_layer = dict(decision.selection.by_layer)
    original = dict(by_layer)
    new_selection = CoreSelection(
        strategy="swap",
        horizon_condition=None,
        by_layer=original,
        metadata={"direct_action": True, "swap": True},
    )
    for target_layer, layer_selection in decision.selection.by_layer.items():
        if target_layer != int(layer):
            continue
        current = set(
            int(value) for value in layer_selection.selected_positions
        )
        swapped = sorted((current - set(int(p) for p in remove_positions)) | set(int(p) for p in add_positions))
        replaced = LayerSelection(
            layer=target_layer,
            selected_positions=swapped,
            eligible_positions=list(layer_selection.eligible_positions),
            aggregate_scores=list(layer_selection.aggregate_scores),
            metadata=dict(layer_selection.metadata),
        )
        new_selection.by_layer[target_layer] = replaced
    return new_selection


def _marginal_measurement(
    runner: CandidatePullbackRunner,
    compressed_state: Any,
    reference: Any,
    current_token: int,
    decision: Any,
    controller: DirectBudgetController,
    memory: AttentionPolicyMemory,
    view: LayerCacheView,
    cycle: int,
    rolling_cache: CacheDiscoveryConfig,
    boundary_tokens: int,
    pair_tokens: int,
    control_tokens: int,
) -> List[Dict[str, Any]]:
    """2C: single-swap marginal deletion risks and pair interactions.

    On layer 0 only, boundary tokens are the highest-scoring eligible tokens
    outside the core (just-below) and the lowest-scoring core tokens
    (just-above).  Each marginal replaces one core token with a boundary
    token and measures the one-step exact-KL delta; pairs replace two core
    tokens with two just-below tokens and test additivity
    (I = joint - sum of singles).  Controls: mid-core and far-out tokens.
    """

    layer = 0
    positions = tuple(int(value) for value in view.positions_by_layer[layer])
    _, _, eligible = mandatory_and_eligible(
        positions, int(controller.sink_size), int(controller.recent_size)
    )
    scores = memory.score(
        layer,
        positions,
        "attention",
        int(controller.pooling_kernel),
        str(controller.pooling_method),
    )
    row_by_position = {
        int(position): row for row, position in enumerate(positions)
    }
    core = set(
        int(value)
        for value in decision.selection.by_layer[layer].selected_positions
    )
    ranked = sorted(
        (int(position) for position in eligible),
        key=lambda position: -float(scores[row_by_position[position]]),
    )
    in_core = [position for position in ranked if position in core]
    out_core = [position for position in ranked if position not in core]
    lowest_in = in_core[-int(boundary_tokens):]
    highest_out = out_core[: int(boundary_tokens)]
    mid_core = in_core[
        len(in_core) // 2 - int(control_tokens) // 2 : len(in_core) // 2 + int(control_tokens) // 2
    ]
    far_out = out_core[-int(control_tokens):]

    def evaluate(selection: CoreSelection) -> float:
        clone = runner.model.shallow_clone_state(compressed_state)
        try:
            runner.model.apply_selection_in_place(
                clone, selection, cache_config=rolling_cache
            )
            logits, _, _ = runner.model.forward_one(
                clone, current_token, capture_attention=False
            )
            return float(
                exact_distribution_metrics(
                    reference.probe_logits[int(cycle)], logits, int(current_token)
                )["exact_kl"]
            )
        finally:
            del clone

    base_risk = evaluate(decision.selection)
    rows: List[Dict[str, Any]] = []
    marginals: Dict[int, float] = {}

    def record(token: int, kind: str, risk: float) -> None:
        marginals[int(token)] = float(risk - base_risk)
        rows.append(
            {
                "cycle": int(cycle),
                "token": int(token),
                "kind": str(kind),
                "risk": float(risk),
                "base_risk": base_risk,
                "marginal_delta": float(risk - base_risk),
            }
        )

    # just-below tokens: swap in for the lowest core token
    for token in highest_out:
        selection = _swap_selection(decision, layer, lowest_in[:1], [token])
        record(token, "just_below", evaluate(selection))
    # just-above tokens: swap out, replaced by the weakest outside token
    for token in lowest_in:
        selection = _swap_selection(decision, layer, [token], highest_out[:1])
        record(token, "just_above", evaluate(selection))
    # controls
    for token in mid_core:
        selection = _swap_selection(decision, layer, [token], highest_out[:1])
        record(token, "mid_core", evaluate(selection))
    for token in far_out:
        selection = _swap_selection(decision, layer, lowest_in[:1], [token])
        record(token, "far_out", evaluate(selection))
    # pairs of just-below tokens: joint swap of the two lowest core tokens
    for index in range(0, int(pair_tokens) * 2, 2):
        pair = highest_out[index : index + 2]
        if len(pair) < 2:
            continue
        joint_selection = _swap_selection(
            decision, layer, lowest_in[:2], pair
        )
        joint = evaluate(joint_selection)
        single_sum = marginals.get(int(pair[0]), 0.0) + marginals.get(int(pair[1]), 0.0)
        rows.append(
            {
                "cycle": int(cycle),
                "token": int(pair[0]),
                "token_j": int(pair[1]),
                "kind": "pair",
                "risk": float(joint),
                "base_risk": base_risk,
                "marginal_delta": float(joint - base_risk),
                "single_sum_delta": float(single_sum),
                "interaction": float(joint - base_risk - single_sum),
            }
        )
    return rows


def _run_attention_ladder(
    runner: CandidatePullbackRunner,
    reference: Any,
    sample: Any,
    controller: DirectBudgetController,
    cycles: int,
    panel_policies: Sequence[str],
    horizons: Sequence[int],
    ladder_step: int,
    monitor_labels: Mapping[str, Sequence[int]],
    evidence_positions: Sequence[int],
    marginal_cfg: Optional[Mapping[str, Any]] = None,
    marginal_rows_out: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Attention pure-eviction trajectory with ladder measurements.

    Committed policy is attention refreshed every cycle (the P35 healthy
    trajectory).  At every ``ladder_step``-th cycle the panel candidates are
    rolled out teacher-forced at multiple horizons on clones of the surviving
    cache; risks are exact KL against the reference probe logits.
    """

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
    initial_positions = tuple(
        int(value) for value in compressed_state.position_maps[0].tolist()
    )
    memory = AttentionPolicyMemory.initialize(
        reference,
        0,
        range(len(compressed_state.cache)),
        int(controller.recent_size),
        initial_positions,
    )
    current_token = int(anchor_state.query_token_id)
    generated: List[int] = []
    cycle_rows: List[Dict[str, Any]] = []
    step_rows: List[Dict[str, Any]] = []
    ladder_rows: List[Dict[str, Any]] = []
    previous_budget: Optional[Mapping[int, int]] = None
    started = time.perf_counter()
    try:
        for cycle in range(int(cycles)):
            view = active_cache_view(runner, compressed_state)
            decision = controller.refresh_scores(
                "attention", memory, view, cycle, str(sample.sample_id)
            )
            maximum_core = max(int(value) for value in decision.requested_budgets.values())
            _, rolling_cache = _cache_configs(
                controller.sink_size, controller.recent_size + 1, maximum_core
            )
            committed_core: Dict[int, Set[int]] = {}
            for layer, layer_selection in decision.selection.by_layer.items():
                committed_core[int(layer)] = set(
                    int(value) for value in layer_selection.selected_positions
                )
            if cycle % int(ladder_step) == 0:
                panel_candidates = {
                    str(name): controller.select(
                        name, memory, view, cycle, str(sample.sample_id)
                    )
                    for name in panel_policies
                }
                for name, candidate in panel_candidates.items():
                    rollout_rows = _ladder_rollout(
                        runner,
                        compressed_state,
                        current_token,
                        reference,
                        candidate,
                        rolling_cache,
                        int(controller.sink_size),
                        horizons,
                        probe_offset=int(cycle),
                    )
                    candidate_l1 = sum(
                        len(
                            committed_core[int(layer)]
                            ^ set(
                                int(value)
                                for value in candidate.selection.by_layer[
                                    int(layer)
                                ].selected_positions
                            )
                        )
                        for layer in committed_core
                    )
                    for row in rollout_rows:
                        ladder_rows.append(
                            {
                                "candidate": str(name),
                                "cycle": int(cycle),
                                "core_l1_vs_committed": int(candidate_l1),
                                **row,
                            }
                        )
            if marginal_cfg and cycle % int(marginal_cfg.get("cycle_step", 8)) == 0:
                marginal_rows = _marginal_measurement(
                    runner,
                    compressed_state,
                    reference,
                    current_token,
                    decision,
                    controller,
                    memory,
                    view,
                    cycle,
                    rolling_cache,
                    int(marginal_cfg.get("boundary_tokens", 8)),
                    int(marginal_cfg.get("pair_tokens", 4)),
                    int(marginal_cfg.get("control_tokens", 4)),
                )
                if marginal_rows_out is not None:
                    marginal_rows_out.extend(marginal_rows)
            query_position = int(compressed_state.logical_next_position)
            runner.model.apply_selection_in_place(
                compressed_state,
                decision.selection,
                cache_config=rolling_cache,
            )
            runner._clear_controls()
            compressed_logits, record, forward_s = runner.model.forward_one(
                compressed_state, current_token, capture_attention=True
            )
            runner.model.validate_active_budget(
                compressed_state, cache_config=rolling_cache
            )
            output = int(torch.argmax(compressed_logits.float()).item())
            reference_logits = reference.probe_logits[int(cycle)]
            metrics = {
                **exact_distribution_metrics(
                    reference_logits,
                    compressed_logits,
                    int(reference.generated_token_ids[int(cycle)]),
                ),
                **_softmax_row(
                    reference_logits, compressed_logits, monitor_labels
                ),
            }
            step_rows.append(
                {
                    "policy": "attention",
                    "cycle": int(cycle),
                    "horizon_offset": 1,
                    "input_token_id": int(current_token),
                    "generated_token_id": output,
                    **metrics,
                }
            )
            input_maps = {
                int(layer): set(
                    int(value) for value in view.positions_by_layer[layer]
                )
                for layer in view.positions_by_layer
            }
            output_maps = compressed_state.position_maps
            irreversible = all(
                set(int(value) for value in output_maps[layer].tolist())
                <= input_maps[layer] | {query_position}
                for layer in output_maps
            )
            if not irreversible:
                raise RuntimeError("pure eviction set inclusion was violated")
            active_lengths = {
                int(layer): len(positions)
                for layer, positions in output_maps.items()
            }
            nominal_global_kv = int(
                len(active_lengths)
                * (
                    int(controller.sink_size)
                    + int(controller.recent_size + 1)
                    + int(controller.core_budget)
                )
            )
            budget_change = (
                0
                if previous_budget is None
                else int(
                    sum(
                        abs(
                            int(decision.requested_budgets[layer])
                            - int(previous_budget[layer])
                        )
                        for layer in decision.requested_budgets
                    )
                )
            )
            cycle_rows.append(
                {
                    "policy": "attention",
                    "cycle": int(cycle),
                    "selection_time_s": 0.0,
                    "telemetry_time_s": 0.0,
                    "forward_time_s": float(forward_s),
                    "budget_l1_change": int(budget_change),
                    "ranking_refreshed": True,
                    "trigger_fired": False,
                    "refresh_count": 1,
                    "irreversible_set_inclusion": True,
                    "persistent_cpu_kv_backing": False,
                    "maximum_active_cache_tokens": int(max(active_lengths.values())),
                    "total_active_layer_tokens": int(sum(active_lengths.values())),
                    "nominal_global_kv_budget": nominal_global_kv,
                    "global_kv_budget_respected": bool(
                        sum(active_lengths.values()) <= nominal_global_kv
                    ),
                    "active_tokens_by_layer_json": _json_vector(active_lengths),
                    "mean_trajectory_exact_kl": metrics.get("exact_kl"),
                    **decision.diagnostics,
                    **_evidence_survival(output_maps, evidence_positions),
                }
            )
            maps = {
                int(layer): tuple(int(value) for value in positions.tolist())
                for layer, positions in output_maps.items()
            }
            memory.update_record(record, maps)
            current_token = output
            generated.append(output)
            previous_budget = dict(decision.requested_budgets)
        mean_kl = float(np.mean([float(row["exact_kl"]) for row in step_rows]))
        metric = _metric_row(runner, sample, "attention", generated, mean_kl)
        wall = float(time.perf_counter() - started)
        summary = {
            "policy": "attention",
            "cycles_completed": len(cycle_rows),
            "mean_trajectory_exact_kl": mean_kl,
            "refresh_mode": "every",
            "label_mode": False,
            "refresh_count": len(cycle_rows),
            "trigger_fired_count": 0,
            "wall_time_s": wall,
            "selection_time_s_total": 0.0,
            "decode_forward_time_s_total": float(
                sum(float(row["forward_time_s"]) for row in cycle_rows)
            ),
            "controller_overhead_fraction": 0.0,
            "decode_tokens_per_s": float(
                len(generated) / max(
                    sum(float(row["forward_time_s"]) for row in cycle_rows),
                    1.0e-12,
                )
            ),
            "controller_decode_tokens_per_s": float(
                len(generated) / max(
                    sum(float(row["forward_time_s"]) for row in cycle_rows),
                    1.0e-12,
                )
            ),
            "end_to_end_tokens_per_s": float(
                len(generated) / max(wall, 1.0e-12)
            ),
            "peak_accelerator_bytes": 0,
            "active_accelerator_bytes_end": 0,
            "cpu_rss_baseline_bytes": 0,
            "peak_cpu_rss_bytes": 0,
            "peak_cpu_rss_delta_bytes": 0,
            "persistent_cpu_kv_backing": False,
            "irreversible_set_inclusion_all_cycles": bool(
                all(row["irreversible_set_inclusion"] for row in cycle_rows)
            ),
            "global_kv_budget_respected_all_cycles": bool(
                all(row["global_kv_budget_respected"] for row in cycle_rows)
            ),
            "maximum_layer_capacity": int(
                max(row["maximum_active_cache_tokens"] for row in cycle_rows)
            ),
            "maximum_actual_global_kv": int(
                max(row["total_active_layer_tokens"] for row in cycle_rows)
            ),
            **metric,
        }
        return cycle_rows, step_rows, ladder_rows, summary
    finally:
        runner.model.release(compressed_state)


def _teacher_panel_decision(
    runner: CandidatePullbackRunner,
    compressed_state: Any,
    current_token: int,
    full_logits: torch.Tensor,
    panel_candidates: Mapping[str, Any],
    rolling_cache: CacheDiscoveryConfig,
    previous_decision: Optional[Any] = None,
) -> Tuple[Any, List[Dict[str, Any]]]:
    """Evaluate a fixed panel of legal cheap actions on the surviving cache.

    Every panel action is a subset of the current compressed state's
    position maps (built from ``view.positions_by_layer``), so applying it
    to a shallow clone respects strict pure eviction: no deleted history is
    read and no counterfactual clone is ever committed.  The returned
    decision is the minimum exact-KL panel action; the rows record each
    candidate's physical one-step risk for the fixed-action-space regret
    decomposition (Gate 1).
    """

    candidates = dict(panel_candidates)
    if previous_decision is not None:
        candidates["stale_prev"] = previous_decision
    rows: List[Dict[str, Any]] = []
    risks: Dict[str, float] = {}
    for name, candidate in candidates.items():
        clone = runner.model.shallow_clone_state(compressed_state)
        try:
            runner.model.apply_selection_in_place(
                clone, candidate.selection, cache_config=rolling_cache
            )
            logits, _, _ = runner.model.forward_one(
                clone, current_token, capture_attention=False
            )
            deltas = exact_distribution_metrics(
                full_logits, logits, int(current_token)
            )
            risks[str(name)] = float(deltas["exact_kl"])
            rows.append(
                {
                    "candidate": str(name),
                    "exact_kl": float(deltas["exact_kl"]),
                    "js": float(deltas["js"]),
                    "logit_l2_sq": float(deltas["logit_l2_sq"]),
                    "fisher_quadratic": float(deltas["fisher_quadratic"]),
                }
            )
        finally:
            del clone
    ordered = sorted(risks.items(), key=lambda item: (item[1], item[0]))
    for rank, (name, _) in enumerate(ordered):
        for row in rows:
            if row["candidate"] == name:
                row["risk_rank"] = int(rank)
                row["selected"] = name == ordered[0][0]
    best_name = ordered[0][0]
    return candidates[best_name], rows


def run_pure_eviction_policy(
    runner: CandidatePullbackRunner,
    reference: Any,
    sample: Any,
    policy: str,
    controller: DirectBudgetController,
    cycles: int,
    monitor_labels: Mapping[str, Sequence[int]],
    evidence_positions: Sequence[int],
    evaluate_exact_kl: bool = True,
    drop_reference_payload_after_initialization: bool = False,
    collect_diagnostic_telemetry: bool = True,
    refresh_mode: str = "every",
    refresh_k: int = 4,
    label_mode: bool = False,
    label_stale_lags: Sequence[int] = (4, 16),
    refresh_event_sink: Optional[List[Dict[str, Any]]] = None,
    trigger_rule: Optional[TriggerRule] = None,
    panel_policies: Optional[Sequence[str]] = None,
    panel_event_sink: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    import mlx.core as mx

    if drop_reference_payload_after_initialization and evaluate_exact_kl:
        raise ValueError("controller-only profiling cannot retain the Full-KV evaluator")
    if refresh_mode not in ("every", "never", "fixed_k", "trigger", "teacher"):
        raise ValueError(f"unknown refresh_mode={refresh_mode}")
    if label_mode and not evaluate_exact_kl:
        raise ValueError("refresh label collection requires the Full-KV evaluator")
    if refresh_mode == "trigger":
        if trigger_rule is None:
            raise ValueError("refresh_mode='trigger' requires a frozen trigger_rule")
        if label_mode:
            raise ValueError("refresh_mode='trigger' is incompatible with label_mode")
    if refresh_mode == "teacher":
        if panel_policies is None:
            raise ValueError("refresh_mode='teacher' requires a panel_policies panel")
        if not evaluate_exact_kl:
            raise ValueError("refresh_mode='teacher' requires the Full-KV evaluator")
        if label_mode or collect_diagnostic_telemetry:
            raise ValueError(
                "refresh_mode='teacher' is incompatible with label_mode and "
                "diagnostic telemetry"
            )

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
    full_state = None
    if evaluate_exact_kl:
        full_state, _ = runner.model.state_from_anchor(
            anchor_state, full_selection, cache_config=full_cache
        )
    initial_positions = tuple(
        int(value) for value in compressed_state.position_maps[0].tolist()
    )
    memory = AttentionPolicyMemory.initialize(
        reference,
        0,
        range(len(compressed_state.cache)),
        int(controller.recent_size),
        initial_positions,
    )
    current_token = int(anchor_state.query_token_id)
    if drop_reference_payload_after_initialization:
        reference.anchors.clear()
        reference.query_records.clear()
        reference.score_states.clear()
        reference.probe_logits.clear()
        anchor_state = None
        full_selection = None
        gc.collect()
        mx.clear_cache()
    generated: List[int] = []
    cycle_rows: List[Dict[str, Any]] = []
    step_rows: List[Dict[str, Any]] = []
    previous_budget: Optional[Mapping[int, int]] = None
    previous_fresh: Optional[
        Tuple[Mapping[int, Tuple[int, ...]], Mapping[int, np.ndarray], Any]
    ] = None
    teacher_previous: Optional[Any] = None
    stale_rankings: Dict[int, FrozenRanking] = {}
    label_lags = [int(value) for value in label_stale_lags]
    selection_total = 0.0
    forward_total = 0.0
    refresh_count = 0
    started = time.perf_counter()
    mx.reset_peak_memory()
    with _MemorySampler() as memory_sampler:
        try:
            for cycle in range(int(cycles)):
                input_maps = {
                    int(layer): set(int(value) for value in positions.tolist())
                    for layer, positions in compressed_state.position_maps.items()
                }
                view = active_cache_view(runner, compressed_state)
                selection_started = time.perf_counter()
                trigger_fired = False
                trigger_feature_row: Dict[str, float] = {
                    name: float("nan") for name in CHEAP_TRIGGER_FEATURES
                }
                if refresh_mode == "trigger":
                    # Cheap-feature gated refresh: compute the candidate fresh
                    # ranking every step, fire the frozen rule on cheap
                    # features only, and keep the stale ranking otherwise.
                    candidate = controller.select(
                        policy, memory, view, cycle, str(sample.sample_id)
                    )
                    trigger_feature_row = _trigger_features(
                        candidate,
                        previous_fresh,
                        view.positions_by_layer,
                        int(controller.core_budget),
                    )
                    trigger_fired, refresh = decide_trigger_refresh(
                        trigger_rule, trigger_feature_row, cycle
                    )
                    if refresh:
                        controller.freeze(candidate, view, cycle)
                        decision = candidate
                    else:
                        decision = controller.stale_selection(
                            view, cycle, str(sample.sample_id), memory=memory
                        )
                    previous_fresh = (
                        view.positions_by_layer,
                        candidate.scores_by_layer,
                        candidate.selection,
                    )
                    del candidate
                elif refresh_mode == "teacher":
                    # Strict-pure-eviction teacher (Gate 0): forward the full
                    # reference for this query, build a panel of cheap legal
                    # actions from the *surviving* cache, evaluate each action
                    # as a counterfactual clone forward, and commit the
                    # minimum exact-KL action.  No deleted history is read:
                    # every panel action is a subset of the current view.
                    panel_candidates = {
                        str(name): controller.select(
                            name, memory, view, cycle, str(sample.sample_id)
                        )
                        for name in panel_policies
                    }
                    maximum_core = max(
                        int(value)
                        for candidate in panel_candidates.values()
                        for value in candidate.requested_budgets.values()
                    )
                    _, rolling_cache = _cache_configs(
                        controller.sink_size,
                        controller.recent_size + 1,
                        maximum_core,
                    )
                    full_logits, _, full_forward_s = runner.model.forward_one(
                        full_state, current_token, capture_attention=True
                    )
                    decision, panel_rows = _teacher_panel_decision(
                        runner,
                        compressed_state,
                        current_token,
                        full_logits,
                        panel_candidates,
                        rolling_cache,
                        previous_decision=teacher_previous,
                    )
                    if panel_event_sink is not None:
                        for row in panel_rows:
                            panel_event_sink.append(
                                {
                                    "cycle": int(cycle),
                                    **row,
                                }
                            )
                    refresh = True
                else:
                    refresh = _scheduled_refresh(
                        refresh_mode, cycle, refresh_k, label_mode
                    )
                    if refresh:
                        decision = controller.refresh_scores(
                            policy, memory, view, cycle, str(sample.sample_id)
                        )
                        if label_mode:
                            stale_rankings[int(cycle)] = controller.frozen
                    else:
                        decision = controller.stale_selection(
                            view, cycle, str(sample.sample_id), memory=memory
                        )
                refresh_count += int(refresh)
                selection_time = float(time.perf_counter() - selection_started)
                telemetry_started = time.perf_counter()
                overlaps = (
                    _counterfactual_overlap(
                        controller,
                        decision,
                        memory,
                        view,
                        cycle,
                        str(sample.sample_id),
                    )
                    if collect_diagnostic_telemetry
                    else {}
                )
                telemetry_time = float(time.perf_counter() - telemetry_started)
                selection_total += selection_time
                stale_decisions: Dict[int, Any] = {}
                if label_mode:
                    for lag in label_lags:
                        source_cycle = max(0, int(cycle) - int(lag))
                        stale_decisions[int(lag)] = controller.stale_selection(
                            view,
                            cycle,
                            str(sample.sample_id),
                            frozen=stale_rankings[source_cycle],
                        )
                maximum_core = max(
                    [int(value) for value in decision.requested_budgets.values()]
                    + [
                        int(value)
                        for stale in stale_decisions.values()
                        for value in stale.requested_budgets.values()
                    ]
                )
                _, rolling_cache = _cache_configs(
                    controller.sink_size,
                    controller.recent_size + 1,
                    maximum_core,
                )
                stale_logits: Dict[int, torch.Tensor] = {}
                for lag, stale in stale_decisions.items():
                    clone = runner.model.shallow_clone_state(compressed_state)
                    runner.model.apply_selection_in_place(
                        clone, stale.selection, cache_config=rolling_cache
                    )
                    logits, _, _ = runner.model.forward_one(
                        clone, current_token, capture_attention=True
                    )
                    stale_logits[int(lag)] = logits
                    del clone
                query_position = int(compressed_state.logical_next_position)
                runner.model.apply_selection_in_place(
                    compressed_state,
                    decision.selection,
                    cache_config=rolling_cache,
                )
                runner._clear_controls()
                compressed_logits, record, forward_s = runner.model.forward_one(
                    compressed_state, current_token, capture_attention=True
                )
                forward_total += float(forward_s)
                runner.model.validate_active_budget(
                    compressed_state, cache_config=rolling_cache
                )
                output = int(torch.argmax(compressed_logits.float()).item())
                metrics: Dict[str, Any] = {}
                if refresh_mode == "teacher":
                    # The full reference was already advanced for this
                    # query by the teacher branch; reuse its logits.
                    full_forward_time = float(full_forward_s)
                else:
                    full_logits = None
                    full_forward_time = float("nan")
                    if full_state is not None:
                        full_logits, _, full_forward_s = runner.model.forward_one(
                            full_state, current_token, capture_attention=True
                        )
                        full_forward_time = float(full_forward_s)
                if full_state is not None:
                    metrics = {
                        **exact_distribution_metrics(
                            full_logits, compressed_logits, output
                        ),
                        **_softmax_row(
                            full_logits, compressed_logits, monitor_labels
                        ),
                        "full_forward_time_s": full_forward_time,
                    }
                if label_mode and refresh_event_sink is not None:
                    event: Dict[str, Any] = {
                        "policy": str(policy),
                        "step": int(cycle),
                    }
                    if previous_fresh is not None:
                        churn = core_churn_by_layer(
                            previous_fresh[2], decision.selection
                        )
                        television = score_tv_by_layer(
                            previous_fresh[1],
                            previous_fresh[0],
                            decision.scores_by_layer,
                            view.positions_by_layer,
                        )
                        event.update(
                            {
                                "churn_jaccard_mean": float(
                                    np.mean(list(churn.values()))
                                ),
                                "churn_jaccard_min": float(
                                    np.min(list(churn.values()))
                                ),
                                "churn_jaccard_max": float(
                                    np.max(list(churn.values()))
                                ),
                                "score_tv_mean": float(
                                    np.mean(list(television.values()))
                                ),
                            }
                        )
                    else:
                        event.update(
                            {
                                "churn_jaccard_mean": float("nan"),
                                "churn_jaccard_min": float("nan"),
                                "churn_jaccard_max": float("nan"),
                                "score_tv_mean": float("nan"),
                            }
                        )
                    margins = boundary_margin_by_layer(
                        decision.scores_by_layer,
                        view.positions_by_layer,
                        decision.eligible_by_layer,
                        int(controller.core_budget),
                    )
                    coverage = coverage_mass_by_layer(
                        decision.scores_by_layer,
                        view.positions_by_layer,
                        decision.selection,
                    )
                    event["boundary_margin_mean"] = float(
                        np.mean(list(margins.values()))
                    )
                    event["coverage_mass_mean"] = float(
                        np.mean(list(coverage.values()))
                    )
                    fresh_kl = float(metrics.get("exact_kl", float("nan")))
                    for lag, stale in stale_decisions.items():
                        stale_kl = float(
                            exact_distribution_metrics(
                                full_logits, stale_logits[int(lag)], output
                            )["exact_kl"]
                        )
                        event[f"stale_exact_kl_lag{int(lag)}"] = stale_kl
                        event[f"refresh_benefit_lag{int(lag)}"] = float(
                            stale_kl - fresh_kl
                        )
                        event[f"stale_action_l1_lag{int(lag)}"] = int(
                            sum(
                                len(
                                    set(
                                        int(value)
                                        for value in stale.selection.by_layer[
                                            layer
                                        ].selected_positions
                                    )
                                    ^ set(
                                        int(value)
                                        for value in decision.selection.by_layer[
                                            layer
                                        ].selected_positions
                                    )
                                )
                                for layer in decision.selection.by_layer
                            )
                        )
                    event.update(
                        {
                            key: value
                            for key, value in metrics.items()
                            if key != "full_forward_time_s"
                        }
                    )
                    refresh_event_sink.append(event)
                step_rows.append(
                    {
                        "policy": str(policy),
                        "cycle": int(cycle),
                        "horizon_offset": 1,
                        "input_token_id": int(current_token),
                        "generated_token_id": output,
                        **metrics,
                    }
                )
                output_maps = compressed_state.position_maps
                irreversible = all(
                    set(int(value) for value in output_maps[layer].tolist())
                    <= input_maps[layer] | {query_position}
                    for layer in output_maps
                )
                if not irreversible:
                    raise RuntimeError("pure eviction set inclusion was violated")
                active_lengths = {
                    int(layer): len(positions)
                    for layer, positions in output_maps.items()
                }
                nominal_global_kv = int(
                    len(active_lengths)
                    * (
                        int(controller.sink_size)
                        + int(controller.recent_size + 1)
                        + int(controller.core_budget)
                    )
                )
                budget_change = (
                    0
                    if previous_budget is None
                    else int(
                        sum(
                            abs(
                                int(decision.requested_budgets[layer])
                                - int(previous_budget[layer])
                            )
                            for layer in decision.requested_budgets
                        )
                    )
                )
                cycle_rows.append(
                    {
                        "policy": str(policy),
                        "cycle": int(cycle),
                        "selection_time_s": selection_time,
                        "telemetry_time_s": telemetry_time,
                        "forward_time_s": float(forward_s),
                        "budget_l1_change": int(budget_change),
                        "ranking_refreshed": bool(refresh),
                        "trigger_fired": bool(trigger_fired),
                        "refresh_count": int(refresh_count),
                        **{
                            f"trigger_feature_{name}": float(value)
                            for name, value in trigger_feature_row.items()
                        },
                        "irreversible_set_inclusion": True,
                        "persistent_cpu_kv_backing": False,
                        "maximum_active_cache_tokens": int(max(active_lengths.values())),
                        "total_active_layer_tokens": int(sum(active_lengths.values())),
                        "nominal_global_kv_budget": nominal_global_kv,
                        "global_kv_budget_respected": bool(
                            sum(active_lengths.values()) <= nominal_global_kv
                        ),
                        "active_tokens_by_layer_json": _json_vector(active_lengths),
                        "mean_trajectory_exact_kl": metrics.get("exact_kl"),
                        **decision.diagnostics,
                        **overlaps,
                        **_evidence_survival(output_maps, evidence_positions),
                    }
                )
                maps = {
                    int(layer): tuple(int(value) for value in positions.tolist())
                    for layer, positions in output_maps.items()
                }
                memory.update_record(record, maps)
                current_token = output
                generated.append(output)
                previous_budget = dict(decision.requested_budgets)
                if refresh_mode == "teacher":
                    teacher_previous = decision
                if label_mode:
                    previous_fresh = (
                        view.positions_by_layer,
                        decision.scores_by_layer,
                        decision.selection,
                    )
            mean_kl = (
                float(np.mean([float(row["exact_kl"]) for row in step_rows]))
                if evaluate_exact_kl
                else float("nan")
            )
            metric = _metric_row(runner, sample, policy, generated, mean_kl)
            wall = float(time.perf_counter() - started)
            summary = {
                "policy": str(policy),
                "cycles_completed": len(cycle_rows),
                "mean_trajectory_exact_kl": mean_kl,
                "refresh_mode": str(refresh_mode if not label_mode else "every"),
                "label_mode": bool(label_mode),
                "refresh_count": int(refresh_count),
                "trigger_fired_count": int(
                    sum(1 for row in cycle_rows if row["trigger_fired"])
                ),
                "wall_time_s": wall,
                "selection_time_s_total": float(selection_total),
                "decode_forward_time_s_total": float(forward_total),
                "controller_overhead_fraction": float(
                    selection_total / max(selection_total + forward_total, 1.0e-12)
                ),
                "decode_tokens_per_s": float(
                    len(generated) / max(forward_total, 1.0e-12)
                ),
                "controller_decode_tokens_per_s": float(
                    len(generated) / max(forward_total + selection_total, 1.0e-12)
                ),
                "end_to_end_tokens_per_s": float(
                    len(generated) / max(wall, 1.0e-12)
                ),
                "peak_accelerator_bytes": int(mx.get_peak_memory()),
                "active_accelerator_bytes_end": int(mx.get_active_memory()),
                "cpu_rss_baseline_bytes": int(memory_sampler.baseline),
                "peak_cpu_rss_bytes": int(memory_sampler.peak),
                "peak_cpu_rss_delta_bytes": int(
                    max(0, memory_sampler.peak - memory_sampler.baseline)
                ),
                "persistent_cpu_kv_backing": False,
                "memory_measurement_scope": (
                    "controller_only_after_reference_payload_release"
                    if drop_reference_payload_after_initialization
                    else "quality_run_including_full_kv_evaluator"
                ),
                "irreversible_set_inclusion_all_cycles": bool(
                    all(row["irreversible_set_inclusion"] for row in cycle_rows)
                ),
                "global_kv_budget_respected_all_cycles": bool(
                    all(row["global_kv_budget_respected"] for row in cycle_rows)
                ),
                "maximum_layer_capacity": int(
                    max(row["maximum_active_cache_tokens"] for row in cycle_rows)
                ),
                "maximum_actual_global_kv": int(
                    max(row["total_active_layer_tokens"] for row in cycle_rows)
                ),
                **metric,
            }
            return cycle_rows, step_rows, summary
        finally:
            runner.model.release(compressed_state, full_state)


def _tail_statistics(values: Sequence[float], level: float = 0.95) -> Dict[str, float]:
    array = np.asarray([float(value) for value in values], dtype=np.float64)
    if array.size == 0:
        return {"p95_exact_kl": float("nan"), "cvar95_exact_kl": float("nan")}
    threshold = float(np.quantile(array, level))
    tail = array[array >= threshold]
    return {
        "p95_exact_kl": threshold,
        "cvar95_exact_kl": float(np.mean(tail)),
    }


def _aggregate(
    samples: pd.DataFrame, steps: pd.DataFrame, group_columns: Sequence[str]
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for keys, current in samples.groupby(list(group_columns), sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        record = {
            column: key.item() if isinstance(key, np.generic) else key
            for column, key in zip(group_columns, keys)
        }
        step_current = steps
        for column, key in record.items():
            if column in step_current.columns:
                step_current = step_current[step_current[column] == key]
        exact = (
            step_current["exact_kl"].dropna().astype(float).tolist()
            if "exact_kl" in step_current
            else []
        )
        gov = current["task_bucket"] == "GovReport"
        niah = current["task_bucket"] == "NIAH"
        rows.append(
            {
                **record,
                "samples": int(len(current)),
                "mean_exact_kl": float(np.mean(exact)) if exact else None,
                **_tail_statistics(exact),
                "mean_official_score": float(current["official_score"].mean()),
                "mean_govreport_rouge_l": (
                    float(current.loc[gov, "rouge_l"].mean()) if gov.any() else None
                ),
                "mean_niah_retrieval": (
                    float(current.loc[niah, "needle_retrieval_accuracy"].mean())
                    if niah.any()
                    else None
                ),
                "mean_decode_tokens_per_s": float(
                    current["decode_tokens_per_s"].mean()
                ),
                "mean_end_to_end_tokens_per_s": float(
                    current["end_to_end_tokens_per_s"].mean()
                ),
                "mean_controller_decode_tokens_per_s": float(
                    current["controller_decode_tokens_per_s"].mean()
                ),
                "mean_controller_overhead_fraction": (
                    float(current["controller_overhead_fraction"].mean())
                    if "controller_overhead_fraction" in current
                    else None
                ),
                "peak_accelerator_bytes": (
                    int(current["peak_accelerator_bytes"].max())
                    if "peak_accelerator_bytes" in current
                    else None
                ),
                "peak_cpu_rss_bytes": (
                    int(current["peak_cpu_rss_bytes"].max())
                    if "peak_cpu_rss_bytes" in current
                    else None
                ),
                "maximum_layer_capacity": (
                    int(current["maximum_layer_capacity"].max())
                    if "maximum_layer_capacity" in current
                    else None
                ),
                "maximum_actual_global_kv": (
                    int(current["maximum_actual_global_kv"].max())
                    if "maximum_actual_global_kv" in current
                    else None
                ),
            }
        )
    return rows


def _load_partial(output: Path, stem: str, parquet: bool = False) -> pd.DataFrame:
    suffix = ".parquet" if parquet else ".csv"
    path = output / ("partial_" + stem + suffix)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path) if parquet else pd.read_csv(path)


def _load_runner(
    config: Mapping[str, Any], repository_root: Path, stage_name: str
) -> Tuple[CandidatePullbackRunner, List[Any]]:
    cfg = load_discovery_config(str(repository_root / str(config["base_config"])))
    cfg.experiment_name = str(config["experiment_name"]) + "_" + stage_name
    for key, value in dict(config.get("model_overrides") or {}).items():
        setattr(cfg.model, str(key), value)
    stage = dict(config[stage_name])
    cfg.tasks = dict(stage["task_overrides"])
    cfg.runtime.seed = int(config["data_seed"])
    cfg.runtime.run_id = str(config["runtime_run_id"]) + "_" + stage_name
    cfg.anchor_steps = [0]
    cfg.generation.max_new_tokens = max(
        int(stage.get("control_cycles", 64)),
        max((int(value) for value in cfg.horizons), default=1),
        max((int(value) for value in cfg.anchor_steps), default=0),
    )
    cfg.validate()
    runner = CandidatePullbackRunner(cfg, repository_root)
    samples, _ = load_discovery_tasks(cfg)
    expected = set(str(value) for value in stage["sample_ids"])
    selected = [sample for sample in samples if str(sample.sample_id) in expected]
    if {str(sample.sample_id) for sample in selected} != expected:
        raise RuntimeError(f"{stage_name} did not load its declared sample split")
    return runner, selected


def _controller(
    config: Mapping[str, Any],
    core_budget: int,
    maximum_delta: int,
    static_budgets: Optional[Mapping[int, int]] = None,
    pure: bool = False,
) -> DirectBudgetController:
    recent = int(config["recent_size"])
    return DirectBudgetController(
        core_budget=int(core_budget),
        sink_size=int(config["sink_size"]),
        recent_size=max(0, recent - 1) if pure else recent,
        pooling_kernel=int(config["snapkv_pooling_kernel"]),
        pooling_method=str(config["snapkv_pooling"]),
        maximum_delta=int(maximum_delta),
        static_budgets=static_budgets,
        shuffle_seed=int(config["data_seed"]),
        stale_lag=int(dict(config.get("p1") or {}).get("stale_lag", 4)),
    )


def _run_backed_stage(
    config: Mapping[str, Any],
    repository_root: Path,
    stage_name: str,
    policies: Sequence[str],
    static_budgets: Optional[Mapping[int, int]] = None,
) -> Path:
    stage = dict(config[stage_name])
    output = repository_root / str(stage["output_run"])
    output.mkdir(parents=True, exist_ok=True)
    runner, samples = _load_runner(config, repository_root, stage_name)
    cycles = int(stage["control_cycles"])
    monitor_cfg = dict(config.get("tail_telemetry") or {})
    previous_samples = _load_partial(output, "sample_results")
    previous_cycles = _load_partial(output, "cycle_rows", parquet=True)
    previous_steps = _load_partial(output, "step_rows", parquet=True)
    cycle_rows: List[Dict[str, Any]] = previous_cycles.to_dict("records")
    step_rows: List[Dict[str, Any]] = previous_steps.to_dict("records")
    summaries: List[Dict[str, Any]] = previous_samples.to_dict("records")
    completed = set()
    if not previous_samples.empty:
        completed = set(
            zip(
                previous_samples["sample_id"].astype(str),
                previous_samples["policy"].astype(str),
            )
        )
    model_info = runner.model.load()
    started = time.perf_counter()
    try:
        for sample_index, sample in enumerate(samples, start=1):
            if all(
                (str(sample.sample_id), str(policy)) in completed
                for policy in policies
            ):
                print(
                    f"[{stage_name}] resume skip {sample.sample_id}", flush=True
                )
                continue
            reference = runner.model.generate_reference(
                sample.sample_id, sample.task, sample.prompt
            )
            monitored, evidence, encodings = _monitor_spec(
                runner.model.tokenizer,
                reference.prompt_token_ids,
                monitor_cfg.get("probability_labels", {}),
                monitor_cfg.get("evidence_phrases", []),
            )
            try:
                for policy in policies:
                    controller = _controller(
                        config,
                        int(stage["core_budget"]),
                        int(stage["maximum_delta"]),
                        static_budgets,
                    )
                    current_cycles, current_steps, summary = run_backed_policy(
                        runner,
                        reference,
                        sample,
                        policy,
                        controller,
                        cycles,
                        int(stage.get("control_horizon", 1)),
                        monitored,
                        evidence,
                        bool(stage.get("evaluate_exact_kl", True)),
                    )
                    base = {
                        "sample_id": str(sample.sample_id),
                        "task": str(sample.task),
                    }
                    cycle_rows.extend({**base, **row} for row in current_cycles)
                    step_rows.extend({**base, **row} for row in current_steps)
                    summaries.append(
                        {
                            **summary,
                            "sample_id": str(sample.sample_id),
                            "task": str(sample.task),
                            "prompt_tokens": len(reference.prompt_token_ids),
                            "evidence_positions_json": json.dumps(evidence),
                            "monitor_encodings_json": json.dumps(encodings),
                        }
                    )
                    print(
                        f"[{stage_name}] {sample_index}/{len(samples)} "
                        f"{sample.sample_id} {policy} "
                        f"kl={summary['mean_trajectory_exact_kl']:.6f} "
                        f"score={summary['official_score']:.4f}",
                        flush=True,
                    )
                atomic_frame(pd.DataFrame(summaries), output / "partial_sample_results.csv")
                atomic_frame(pd.DataFrame(cycle_rows), output / "partial_cycle_rows.parquet")
                atomic_frame(pd.DataFrame(step_rows), output / "partial_step_rows.parquet")
                atomic_json(
                    output / "progress.json",
                    {
                        "completed_samples": sample_index,
                        "expected_samples": len(samples),
                        "last_sample_id": str(sample.sample_id),
                    },
                )
            finally:
                runner.model.release(reference)
    finally:
        runner.model.close()
    sample_frame = pd.DataFrame(summaries)
    cycle_frame = pd.DataFrame(cycle_rows)
    step_frame = pd.DataFrame(step_rows)
    aggregates = _aggregate(sample_frame, step_frame, ["policy"])
    result: Dict[str, Any] = {
        "experiment": str(config["experiment_name"]),
        "stage": stage_name,
        "model_info": model_info,
        "samples": sorted(str(sample.sample_id) for sample in samples),
        "policies": list(policies),
        "policy_aggregates": aggregates,
        "collection_elapsed_s": float(time.perf_counter() - started),
        "fixed_global_requested_core_budget": bool(
            sample_frame["fixed_global_requested_core_budget"].all()
        ),
        "execution_valid": bool(
            len(sample_frame) == len(samples) * len(policies)
        ),
    }
    if stage_name == "calibration":
        vectors = [
            {
                layer: int(value)
                for layer, value in enumerate(json.loads(raw))
            }
            for raw in cycle_frame["dynamic_budget_by_layer_json"]
        ]
        static = average_static_budgets(
            vectors,
            int(stage["core_budget"]),
            int(stage["maximum_delta"]),
        )
        result["static_budget_by_layer"] = {
            str(layer): int(value) for layer, value in static.items()
        }
    atomic_frame(sample_frame, output / "sample_results.csv")
    atomic_frame(cycle_frame, output / "cycle_rows.parquet")
    atomic_frame(step_frame, output / "step_rows.parquet")
    atomic_frame(pd.DataFrame(aggregates), output / "aggregate_results.csv")
    atomic_json(output / "summary.json", result)
    atomic_text(output / "config.yaml", yaml.safe_dump(dict(config), sort_keys=False))
    return output


def run_calibration(config_path: Path, repository_root: Path) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return _run_backed_stage(
        config, repository_root, "calibration", ["dynamic_b3"]
    )


def run_p1(config_path: Path, repository_root: Path) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    calibration = repository_root / str(config["calibration"]["output_run"])
    summary = json.loads((calibration / "summary.json").read_text(encoding="utf-8"))
    static = {
        int(layer): int(value)
        for layer, value in summary["static_budget_by_layer"].items()
    }
    return _run_backed_stage(
        config, repository_root, "p1", MECHANISM_POLICIES, static
    )


def run_p3(config_path: Path, repository_root: Path) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return _run_backed_stage(config, repository_root, "p3", ["dynamic_b3"])


def run_p2(config_path: Path, repository_root: Path) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    stage = dict(config["p2"])
    output = repository_root / str(stage["output_run"])
    output.mkdir(parents=True, exist_ok=True)
    runner, samples = _load_runner(config, repository_root, "p2")
    monitor_cfg = dict(config.get("tail_telemetry") or {})
    previous_samples = _load_partial(output, "sample_results")
    previous_cycles = _load_partial(output, "cycle_rows", parquet=True)
    previous_steps = _load_partial(output, "step_rows", parquet=True)
    previous_full = _load_partial(output, "full_cache_reference")
    cycle_rows: List[Dict[str, Any]] = previous_cycles.to_dict("records")
    step_rows: List[Dict[str, Any]] = previous_steps.to_dict("records")
    summaries: List[Dict[str, Any]] = previous_samples.to_dict("records")
    full_rows: List[Dict[str, Any]] = previous_full.to_dict("records")
    completed = set()
    if not previous_samples.empty:
        completed = set(
            zip(
                previous_samples["sample_id"].astype(str),
                previous_samples["total_budget"].astype(int),
                previous_samples["policy"].astype(str),
            )
        )
    model_info = runner.model.load()
    started = time.perf_counter()
    try:
        for sample_index, sample in enumerate(samples, start=1):
            expected = {
                (str(sample.sample_id), int(budget["total_budget"]), str(policy))
                for budget in stage["budgets"]
                for policy in PURE_EVICTION_POLICIES
            }
            if expected <= completed:
                print(f"[p2] resume skip {sample.sample_id}", flush=True)
                continue
            reference = runner.model.generate_reference(
                sample.sample_id, sample.task, sample.prompt
            )
            monitored, evidence, encodings = _monitor_spec(
                runner.model.tokenizer,
                reference.prompt_token_ids,
                monitor_cfg.get("probability_labels", {}),
                monitor_cfg.get("evidence_phrases", []),
            )
            try:
                full_tokens = [
                    int(value)
                    for value in reference.generated_token_ids[
                        : int(stage["control_cycles"])
                    ]
                ]
                full_rows.append(
                    {
                        **_metric_row(
                            runner, sample, "full_cache", full_tokens, 0.0
                        ),
                        "sample_id": str(sample.sample_id),
                        "task": str(sample.task),
                    }
                )
                for budget in stage["budgets"]:
                    total_budget = int(budget["total_budget"])
                    core_budget = int(budget["core_budget"])
                    maximum_delta = int(budget["maximum_delta"])
                    if total_budget != int(config["sink_size"]) + int(config["recent_size"]) + core_budget:
                        raise ValueError("P2 total/core budget decomposition is inconsistent")
                    for policy in PURE_EVICTION_POLICIES:
                        controller = _controller(
                            config,
                            core_budget,
                            maximum_delta,
                            pure=True,
                        )
                        current_cycles, current_steps, summary = run_pure_eviction_policy(
                            runner,
                            reference,
                            sample,
                            policy,
                            controller,
                            int(stage["control_cycles"]),
                            monitored,
                            evidence,
                            True,
                        )
                        base = {
                            "sample_id": str(sample.sample_id),
                            "task": str(sample.task),
                            "total_budget": total_budget,
                            "core_budget": core_budget,
                        }
                        cycle_rows.extend({**base, **row} for row in current_cycles)
                        step_rows.extend({**base, **row} for row in current_steps)
                        summaries.append(
                            {
                                **base,
                                **summary,
                                "prompt_tokens": len(reference.prompt_token_ids),
                                "monitor_encodings_json": json.dumps(encodings),
                            }
                        )
                        print(
                            f"[p2] {sample_index}/{len(samples)} {sample.sample_id} "
                            f"budget={total_budget} {policy} "
                            f"kl={summary['mean_trajectory_exact_kl']:.6f} "
                            f"score={summary['official_score']:.4f}",
                            flush=True,
                        )
                atomic_frame(pd.DataFrame(summaries), output / "partial_sample_results.csv")
                atomic_frame(pd.DataFrame(cycle_rows), output / "partial_cycle_rows.parquet")
                atomic_frame(pd.DataFrame(step_rows), output / "partial_step_rows.parquet")
                atomic_frame(
                    pd.DataFrame(full_rows),
                    output / "partial_full_cache_reference.csv",
                )
                atomic_json(
                    output / "progress.json",
                    {
                        "completed_samples": sample_index,
                        "expected_samples": len(samples),
                        "last_sample_id": str(sample.sample_id),
                    },
                )
            finally:
                runner.model.release(reference)
    finally:
        runner.model.close()
    sample_frame = pd.DataFrame(summaries)
    cycle_frame = pd.DataFrame(cycle_rows)
    step_frame = pd.DataFrame(step_rows)
    aggregate = _aggregate(sample_frame, step_frame, ["total_budget", "policy"])
    result = {
        "experiment": str(config["experiment_name"]),
        "stage": "p2",
        "model_info": model_info,
        "samples": sorted(str(sample.sample_id) for sample in samples),
        "budgets": list(stage["budgets"]),
        "policies": list(PURE_EVICTION_POLICIES),
        "policy_aggregates": aggregate,
        "full_cache_quality_reference": full_rows,
        "all_irreversible_set_inclusions_hold": bool(
            sample_frame["irreversible_set_inclusion_all_cycles"].all()
        ),
        "persistent_cpu_kv_backing": False,
        "collection_elapsed_s": float(time.perf_counter() - started),
        "execution_valid": bool(
            len(sample_frame)
            == len(samples) * len(stage["budgets"]) * len(PURE_EVICTION_POLICIES)
        ),
    }
    atomic_frame(sample_frame, output / "sample_results.csv")
    atomic_frame(cycle_frame, output / "cycle_rows.parquet")
    atomic_frame(step_frame, output / "step_rows.parquet")
    atomic_frame(pd.DataFrame(aggregate), output / "aggregate_results.csv")
    atomic_frame(pd.DataFrame(full_rows), output / "full_cache_reference.csv")
    atomic_json(output / "summary.json", result)
    atomic_text(output / "config.yaml", yaml.safe_dump(dict(config), sort_keys=False))
    return output


def run_p2_profile(config_path: Path, repository_root: Path) -> Path:
    """Profile controller-only pure eviction after releasing reference payloads."""
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    stage = dict(config["p2"])
    output = repository_root / str(stage["output_run"])
    output.mkdir(parents=True, exist_ok=True)
    runner, loaded = _load_runner(config, repository_root, "p2")
    profile_ids = set(str(value) for value in stage["profile_sample_ids"])
    samples = [sample for sample in loaded if str(sample.sample_id) in profile_ids]
    if {str(sample.sample_id) for sample in samples} != profile_ids:
        raise RuntimeError("P2 profile split was not loaded")
    monitor_cfg = dict(config.get("tail_telemetry") or {})
    profile_path = output / "runtime_profile.csv"
    cycle_path = output / "runtime_profile_cycles.parquet"
    previous = pd.read_csv(profile_path) if profile_path.exists() else pd.DataFrame()
    previous_cycles = pd.read_parquet(cycle_path) if cycle_path.exists() else pd.DataFrame()
    rows: List[Dict[str, Any]] = previous.to_dict("records")
    cycle_rows: List[Dict[str, Any]] = previous_cycles.to_dict("records")
    completed = set()
    if not previous.empty:
        completed = set(
            zip(
                previous["sample_id"].astype(str),
                previous["total_budget"].astype(int),
                previous["policy"].astype(str),
            )
        )
    model_info = runner.model.load()
    started = time.perf_counter()
    try:
        for sample in samples:
            for budget in stage["budgets"]:
                total_budget = int(budget["total_budget"])
                for policy in PURE_EVICTION_POLICIES:
                    key = (str(sample.sample_id), total_budget, str(policy))
                    if key in completed:
                        print(f"[p2-profile] resume skip {key}", flush=True)
                        continue
                    reference = runner.model.generate_reference(
                        sample.sample_id, sample.task, sample.prompt
                    )
                    prompt_tokens = len(reference.prompt_token_ids)
                    monitored, evidence, _ = _monitor_spec(
                        runner.model.tokenizer,
                        reference.prompt_token_ids,
                        monitor_cfg.get("probability_labels", {}),
                        monitor_cfg.get("evidence_phrases", []),
                    )
                    try:
                        controller = _controller(
                            config,
                            int(budget["core_budget"]),
                            int(budget["maximum_delta"]),
                            pure=True,
                        )
                        current_cycles, _, summary = run_pure_eviction_policy(
                            runner,
                            reference,
                            sample,
                            policy,
                            controller,
                            int(stage["control_cycles"]),
                            monitored,
                            evidence,
                            evaluate_exact_kl=False,
                            drop_reference_payload_after_initialization=True,
                            collect_diagnostic_telemetry=False,
                        )
                        base = {
                            "sample_id": str(sample.sample_id),
                            "task": str(sample.task),
                            "total_budget": total_budget,
                            "core_budget": int(budget["core_budget"]),
                            "policy": str(policy),
                        }
                        rows.append(
                            {
                                **base,
                                **summary,
                                "prompt_tokens": int(prompt_tokens),
                            }
                        )
                        cycle_rows.extend({**base, **row} for row in current_cycles)
                        atomic_frame(pd.DataFrame(rows), profile_path)
                        atomic_frame(pd.DataFrame(cycle_rows), cycle_path)
                        print(
                            f"[p2-profile] {sample.sample_id} budget={total_budget} "
                            f"{policy} e2e_tok_s={summary['end_to_end_tokens_per_s']:.3f}",
                            flush=True,
                        )
                    finally:
                        runner.model.release(reference)
    finally:
        runner.model.close()
    frame = pd.DataFrame(rows)
    aggregates = []
    for (budget, policy), current in frame.groupby(
        ["total_budget", "policy"], sort=True
    ):
        aggregates.append(
            {
                "total_budget": int(budget),
                "policy": str(policy),
                "profile_samples": int(len(current)),
                "mean_decode_tokens_per_s": float(
                    current["decode_tokens_per_s"].mean()
                ),
                "mean_end_to_end_tokens_per_s": float(
                    current["end_to_end_tokens_per_s"].mean()
                ),
                "mean_controller_decode_tokens_per_s": float(
                    current["controller_decode_tokens_per_s"].mean()
                ),
                "mean_controller_overhead_fraction": float(
                    current["controller_overhead_fraction"].mean()
                ),
                "peak_accelerator_bytes": int(
                    current["peak_accelerator_bytes"].max()
                ),
                "peak_cpu_rss_bytes": int(current["peak_cpu_rss_bytes"].max()),
                "maximum_layer_capacity": int(
                    current["maximum_layer_capacity"].max()
                ),
                "maximum_actual_global_kv": int(
                    current["maximum_actual_global_kv"].max()
                ),
            }
        )
    result = {
        "stage": "p2_controller_only_profile",
        "model_info": model_info,
        "samples": sorted(profile_ids),
        "measurement_scope": "controller_only_after_reference_payload_release",
        "persistent_cpu_kv_backing": False,
        "policy_aggregates": aggregates,
        "collection_elapsed_s": float(time.perf_counter() - started),
        "execution_valid": bool(
            len(frame)
            == len(samples) * len(stage["budgets"]) * len(PURE_EVICTION_POLICIES)
            and frame["irreversible_set_inclusion_all_cycles"].all()
            and frame["global_kv_budget_respected_all_cycles"].all()
        ),
    }
    atomic_frame(pd.DataFrame(aggregates), output / "runtime_profile_aggregate.csv")
    atomic_json(output / "runtime_profile_summary.json", result)
    return output


def run_ladder(config_path: Path, repository_root: Path) -> Path:
    """2B propagation ladder: at what risk depth does candidate risk appear.

    Runs the standard attention pure-eviction trajectory (P35 substrate) and,
    at every ``ladder_cycle_step``-th cycle, measures the teacher-forced
    multi-horizon risk of each panel candidate on a clone of the surviving
    cache (horizons {1,2,4,8} default).  The ladder answers: does one-step
    exact KL rank candidates correctly, or does the risk of dropping a token
    appear only at depth >= 2 (e.g., when the needle is queried)?
    """
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    stage = dict(config["ladder"])
    output = repository_root / str(stage["output_run"])
    output.mkdir(parents=True, exist_ok=True)
    runner, samples = _load_runner(config, repository_root, "ladder")
    panel_policies = [str(value) for value in stage["panel_policies"]]
    horizons = [int(value) for value in stage["horizons"]]
    ladder_step = int(stage.get("ladder_cycle_step", 4))
    budgets = list(stage["budgets"])
    cycles = int(stage["control_cycles"])
    probe_indices = range(0, int(cycles) + max(horizons) - 1)
    monitor_cfg = dict(config.get("tail_telemetry") or {})
    previous_samples = _load_partial(output, "sample_results")
    previous_cycles = _load_partial(output, "cycle_rows", parquet=True)
    previous_steps = _load_partial(output, "step_rows", parquet=True)
    previous_ladder = _load_partial(output, "ladder_rows", parquet=True)
    previous_marginal = _load_partial(output, "marginal_rows", parquet=True)
    cycle_rows: List[Dict[str, Any]] = previous_cycles.to_dict("records")
    step_rows: List[Dict[str, Any]] = previous_steps.to_dict("records")
    ladder_rows: List[Dict[str, Any]] = previous_ladder.to_dict("records")
    marginal_rows: List[Dict[str, Any]] = previous_marginal.to_dict("records")
    summaries: List[Dict[str, Any]] = previous_samples.to_dict("records")
    completed = set()
    if not previous_samples.empty:
        completed = set(
            zip(
                previous_samples["sample_id"].astype(str),
                previous_samples["total_budget"].astype(int),
            )
        )
    model_info = runner.model.load()
    started = time.perf_counter()
    try:
        for sample_index, sample in enumerate(samples, start=1):
            expected = {
                (str(sample.sample_id), int(budget["total_budget"]))
                for budget in budgets
            }
            if expected <= completed:
                print(f"[ladder] resume skip {sample.sample_id}", flush=True)
                continue
            reference = runner.model.generate_reference(
                sample.sample_id,
                sample.task,
                sample.prompt,
                extra_probe_target_indices=probe_indices,
            )
            monitored, evidence, encodings = _monitor_spec(
                runner.model.tokenizer,
                reference.prompt_token_ids,
                monitor_cfg.get("probability_labels", {}),
                monitor_cfg.get("evidence_phrases", []),
            )
            try:
                for budget in budgets:
                    total_budget = int(budget["total_budget"])
                    core_budget = int(budget["core_budget"])
                    maximum_delta = int(budget["maximum_delta"])
                    controller = _controller(
                        config, core_budget, maximum_delta, pure=True
                    )
                    marginal_sink: List[Dict[str, Any]] = []
                    (sample_cycles, sample_steps, sample_ladder,
                     summary) = _run_attention_ladder(
                        runner,
                        reference,
                        sample,
                        controller,
                        cycles,
                        panel_policies,
                        horizons,
                        ladder_step,
                        monitored,
                        evidence,
                        marginal_cfg=dict(stage.get("marginal") or {}),
                        marginal_rows_out=marginal_sink,
                    )
                    base = {
                        "sample_id": str(sample.sample_id),
                        "task": str(sample.task),
                        "total_budget": total_budget,
                        "core_budget": core_budget,
                    }
                    cycle_rows.extend({**base, **row} for row in sample_cycles)
                    step_rows.extend({**base, **row} for row in sample_steps)
                    ladder_rows.extend({**base, **row} for row in sample_ladder)
                    marginal_rows.extend({**base, **row} for row in marginal_sink)
                    summaries.append(
                        {
                            **base,
                            **summary,
                            "prompt_tokens": len(reference.prompt_token_ids),
                            "monitor_encodings_json": json.dumps(encodings),
                        }
                    )
                    print(
                        f"[ladder] {sample_index}/{len(samples)} "
                        f"{sample.sample_id} budget={total_budget} "
                        f"kl={summary['mean_trajectory_exact_kl']:.6f} "
                        f"ladder_rows={len(sample_ladder)}",
                        flush=True,
                    )
                atomic_frame(pd.DataFrame(summaries), output / "partial_sample_results.csv")
                atomic_frame(pd.DataFrame(cycle_rows), output / "partial_cycle_rows.parquet")
                atomic_frame(pd.DataFrame(step_rows), output / "partial_step_rows.parquet")
                atomic_frame(pd.DataFrame(ladder_rows), output / "partial_ladder_rows.parquet")
                atomic_frame(pd.DataFrame(marginal_rows), output / "partial_marginal_rows.parquet")
                atomic_json(
                    output / "progress.json",
                    {
                        "completed_samples": sample_index,
                        "expected_samples": len(samples),
                        "last_sample_id": str(sample.sample_id),
                    },
                )
            finally:
                runner.model.release(reference)
    finally:
        runner.model.close()
    sample_frame = pd.DataFrame(summaries)
    cycle_frame = pd.DataFrame(cycle_rows)
    step_frame = pd.DataFrame(step_rows)
    ladder_frame = pd.DataFrame(ladder_rows)
    aggregate = _aggregate(sample_frame, step_frame, ["total_budget", "policy"])
    ladder_aggregates: List[Dict[str, Any]] = []
    if not ladder_frame.empty:
        for (candidate, horizon), group in ladder_frame.groupby(
            ["candidate", "horizon"]
        ):
            ladder_aggregates.append(
                {
                    "candidate": str(candidate),
                    "horizon": int(horizon),
                    "rows": int(len(group)),
                    "mean_exact_kl": float(group["exact_kl"].mean()),
                    "p95_exact_kl": float(group["exact_kl"].quantile(0.95)),
                    "mean_cumulative_kl": float(group["cumulative_kl"].mean()),
                }
            )
    result = {
        "experiment": str(config["experiment_name"]),
        "stage": "ladder",
        "model_info": model_info,
        "samples": sorted(str(sample.sample_id) for sample in samples),
        "budgets": list(budgets),
        "panel_policies": list(panel_policies),
        "horizons": list(horizons),
        "ladder_cycle_step": int(ladder_step),
        "policy_aggregates": aggregate,
        "ladder_aggregates": ladder_aggregates,
        "all_irreversible_set_inclusions_hold": bool(
            sample_frame["irreversible_set_inclusion_all_cycles"].all()
        ),
        "persistent_cpu_kv_backing": False,
        "collection_elapsed_s": float(time.perf_counter() - started),
        "execution_valid": bool(
            len(sample_frame) == len(samples) * len(budgets)
        ),
    }
    atomic_frame(sample_frame, output / "sample_results.csv")
    atomic_frame(cycle_frame, output / "cycle_rows.parquet")
    atomic_frame(step_frame, output / "step_rows.parquet")
    atomic_frame(ladder_frame, output / "ladder_rows.parquet")
    atomic_frame(pd.DataFrame(marginal_rows), output / "marginal_rows.parquet")
    atomic_frame(pd.DataFrame(aggregate), output / "aggregate_results.csv")
    atomic_frame(pd.DataFrame(ladder_aggregates), output / "ladder_aggregates.csv")
    atomic_json(output / "summary.json", result)
    atomic_text(output / "config.yaml", yaml.safe_dump(dict(config), sort_keys=False))
    return output


def run_teacher_gate(config_path: Path, repository_root: Path) -> Path:
    """Gate 0/1: strict-pure-eviction teacher headroom and fixed-action-space
    regret on the P35 substrate.

    The teacher arm commits, at every cycle, the minimum exact-KL action from
    a fixed panel of cheap legal actions (attention / b2_uniform /
    a2_temporal_volatility / uniform / snapkv / stale_prev), evaluated as
    counterfactual clones of the *surviving* cache.  The per-candidate
    one-step KL rows are the fixed-action-space regret table; the committed
    trajectory is the teacher's strict-pure-eviction closed loop.
    """
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    stage = dict(config["teacher_gate"])
    output = repository_root / str(stage["output_run"])
    output.mkdir(parents=True, exist_ok=True)
    runner, samples = _load_runner(config, repository_root, "teacher_gate")
    panel_policies = [str(value) for value in stage["panel_policies"]]
    budgets = list(stage["budgets"])
    cycles = int(stage["control_cycles"])
    monitor_cfg = dict(config.get("tail_telemetry") or {})
    previous_samples = _load_partial(output, "sample_results")
    previous_cycles = _load_partial(output, "cycle_rows", parquet=True)
    previous_steps = _load_partial(output, "step_rows", parquet=True)
    previous_panel = _load_partial(output, "panel_rows", parquet=True)
    previous_full = _load_partial(output, "full_cache_reference")
    cycle_rows: List[Dict[str, Any]] = previous_cycles.to_dict("records")
    step_rows: List[Dict[str, Any]] = previous_steps.to_dict("records")
    panel_rows: List[Dict[str, Any]] = previous_panel.to_dict("records")
    summaries: List[Dict[str, Any]] = previous_samples.to_dict("records")
    full_rows: List[Dict[str, Any]] = previous_full.to_dict("records")
    completed = set()
    if not previous_samples.empty:
        completed = set(
            zip(
                previous_samples["sample_id"].astype(str),
                previous_samples["total_budget"].astype(int),
            )
        )
    model_info = runner.model.load()
    started = time.perf_counter()
    try:
        for sample_index, sample in enumerate(samples, start=1):
            expected = {
                (str(sample.sample_id), int(budget["total_budget"]))
                for budget in budgets
            }
            if expected <= completed:
                print(f"[teacher-gate] resume skip {sample.sample_id}", flush=True)
                continue
            reference = runner.model.generate_reference(
                sample.sample_id, sample.task, sample.prompt
            )
            monitored, evidence, encodings = _monitor_spec(
                runner.model.tokenizer,
                reference.prompt_token_ids,
                monitor_cfg.get("probability_labels", {}),
                monitor_cfg.get("evidence_phrases", []),
            )
            try:
                full_tokens = [
                    int(value)
                    for value in reference.generated_token_ids[: int(cycles)]
                ]
                full_rows.append(
                    {
                        **_metric_row(
                            runner, sample, "full_cache", full_tokens, 0.0
                        ),
                        "sample_id": str(sample.sample_id),
                        "task": str(sample.task),
                    }
                )
                for budget in budgets:
                    total_budget = int(budget["total_budget"])
                    core_budget = int(budget["core_budget"])
                    maximum_delta = int(budget["maximum_delta"])
                    if total_budget != int(config["sink_size"]) + int(config["recent_size"]) + core_budget:
                        raise ValueError(
                            "teacher gate total/core budget decomposition is inconsistent"
                        )
                    controller = _controller(
                        config,
                        core_budget,
                        maximum_delta,
                        pure=True,
                    )
                    event_sink: List[Dict[str, Any]] = []
                    current_cycles, current_steps, summary = run_pure_eviction_policy(
                        runner,
                        reference,
                        sample,
                        "teacher_panel",
                        controller,
                        cycles,
                        monitored,
                        evidence,
                        True,
                        collect_diagnostic_telemetry=False,
                        refresh_mode="teacher",
                        panel_policies=panel_policies,
                        panel_event_sink=event_sink,
                    )
                    base = {
                        "sample_id": str(sample.sample_id),
                        "task": str(sample.task),
                        "total_budget": total_budget,
                        "core_budget": core_budget,
                    }
                    cycle_rows.extend({**base, **row} for row in current_cycles)
                    step_rows.extend({**base, **row} for row in current_steps)
                    panel_rows.extend({**base, **row} for row in event_sink)
                    summaries.append(
                        {
                            **base,
                            **summary,
                            "prompt_tokens": len(reference.prompt_token_ids),
                            "monitor_encodings_json": json.dumps(encodings),
                        }
                    )
                    print(
                        f"[teacher-gate] {sample_index}/{len(samples)} "
                        f"{sample.sample_id} budget={total_budget} "
                        f"teacher kl={summary['mean_trajectory_exact_kl']:.6f} "
                        f"score={summary['official_score']:.4f}",
                        flush=True,
                    )
                atomic_frame(pd.DataFrame(summaries), output / "partial_sample_results.csv")
                atomic_frame(pd.DataFrame(cycle_rows), output / "partial_cycle_rows.parquet")
                atomic_frame(pd.DataFrame(step_rows), output / "partial_step_rows.parquet")
                atomic_frame(pd.DataFrame(panel_rows), output / "partial_panel_rows.parquet")
                atomic_frame(
                    pd.DataFrame(full_rows),
                    output / "partial_full_cache_reference.csv",
                )
                atomic_json(
                    output / "progress.json",
                    {
                        "completed_samples": sample_index,
                        "expected_samples": len(samples),
                        "last_sample_id": str(sample.sample_id),
                    },
                )
            finally:
                runner.model.release(reference)
    finally:
        runner.model.close()
    sample_frame = pd.DataFrame(summaries)
    cycle_frame = pd.DataFrame(cycle_rows)
    step_frame = pd.DataFrame(step_rows)
    panel_frame = pd.DataFrame(panel_rows)
    aggregate = _aggregate(sample_frame, step_frame, ["total_budget", "policy"])
    panel_aggregates: List[Dict[str, Any]] = []
    if not panel_frame.empty:
        for candidate, group in panel_frame.groupby("candidate"):
            panel_aggregates.append(
                {
                    "candidate": str(candidate),
                    "rows": int(len(group)),
                    "mean_exact_kl": float(group["exact_kl"].mean()),
                    "std_exact_kl": float(group["exact_kl"].std()),
                    "median_exact_kl": float(group["exact_kl"].median()),
                    "p95_exact_kl": float(group["exact_kl"].quantile(0.95)),
                    "win_fraction": float(
                        group["selected"].sum() / max(1, len(group))
                    ),
                    "mean_risk_rank": float(group["risk_rank"].mean()),
                }
            )
    result = {
        "experiment": str(config["experiment_name"]),
        "stage": "teacher_gate",
        "model_info": model_info,
        "samples": sorted(str(sample.sample_id) for sample in samples),
        "budgets": list(budgets),
        "panel_policies": list(panel_policies),
        "policy_aggregates": aggregate,
        "panel_aggregates": panel_aggregates,
        "full_cache_quality_reference": full_rows,
        "all_irreversible_set_inclusions_hold": bool(
            sample_frame["irreversible_set_inclusion_all_cycles"].all()
        ),
        "persistent_cpu_kv_backing": False,
        "collection_elapsed_s": float(time.perf_counter() - started),
        "execution_valid": bool(
            not panel_frame.empty
            and len(sample_frame) == len(samples) * len(budgets)
            and len(panel_frame)
            == len(samples)
            * len(budgets)
            * (cycles * len(panel_policies) + (cycles - 1))
        ),
    }
    atomic_frame(sample_frame, output / "sample_results.csv")
    atomic_frame(cycle_frame, output / "cycle_rows.parquet")
    atomic_frame(step_frame, output / "step_rows.parquet")
    atomic_frame(panel_frame, output / "panel_rows.parquet")
    atomic_frame(pd.DataFrame(aggregate), output / "aggregate_results.csv")
    atomic_frame(pd.DataFrame(panel_aggregates), output / "panel_aggregates.csv")
    atomic_frame(pd.DataFrame(full_rows), output / "full_cache_reference.csv")
    atomic_json(output / "summary.json", result)
    atomic_text(output / "config.yaml", yaml.safe_dump(dict(config), sort_keys=False))
    return output


def run_r2a(config_path: Path, repository_root: Path) -> Path:
    """Collect the R2a refresh-event table with stale-counterfactual labels."""
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    stage = dict(config["r2a_labels"])
    output = repository_root / str(stage["output_run"])
    output.mkdir(parents=True, exist_ok=True)
    runner, samples = _load_runner(config, repository_root, "r2a_labels")
    monitor_cfg = dict(config.get("tail_telemetry") or {})
    policies = [str(value) for value in stage["policies"]]
    total_budget = int(stage["total_budget"])
    core_budget = int(stage["core_budget"])
    maximum_delta = int(stage["maximum_delta"])
    if total_budget != int(config["sink_size"]) + int(config["recent_size"]) + core_budget:
        raise ValueError("R2a total/core budget decomposition is inconsistent")
    cycles = int(stage["control_cycles"])
    label_lags = [int(value) for value in stage.get("label_stale_lags", [4, 16])]
    previous_samples = _load_partial(output, "sample_results")
    previous_cycles = _load_partial(output, "cycle_rows", parquet=True)
    previous_steps = _load_partial(output, "step_rows", parquet=True)
    previous_events = _load_partial(output, "refresh_event_rows", parquet=True)
    cycle_rows: List[Dict[str, Any]] = previous_cycles.to_dict("records")
    step_rows: List[Dict[str, Any]] = previous_steps.to_dict("records")
    event_rows: List[Dict[str, Any]] = previous_events.to_dict("records")
    summaries: List[Dict[str, Any]] = previous_samples.to_dict("records")
    completed = set()
    if not previous_samples.empty:
        completed = set(
            zip(
                previous_samples["sample_id"].astype(str),
                previous_samples["policy"].astype(str),
            )
        )
    model_info = runner.model.load()
    started = time.perf_counter()
    try:
        for sample_index, sample in enumerate(samples, start=1):
            if all(
                (str(sample.sample_id), str(policy)) in completed
                for policy in policies
            ):
                print(f"[r2a-labels] resume skip {sample.sample_id}", flush=True)
                continue
            reference = runner.model.generate_reference(
                sample.sample_id, sample.task, sample.prompt
            )
            monitored, evidence, encodings = _monitor_spec(
                runner.model.tokenizer,
                reference.prompt_token_ids,
                monitor_cfg.get("probability_labels", {}),
                monitor_cfg.get("evidence_phrases", []),
            )
            try:
                for policy in policies:
                    controller = _controller(
                        config, core_budget, maximum_delta, pure=True
                    )
                    event_sink: List[Dict[str, Any]] = []
                    current_cycles, current_steps, summary = run_pure_eviction_policy(
                        runner,
                        reference,
                        sample,
                        policy,
                        controller,
                        cycles,
                        monitored,
                        evidence,
                        True,
                        label_mode=bool(stage.get("label_mode", True)),
                        label_stale_lags=label_lags,
                        refresh_event_sink=event_sink,
                    )
                    base = {
                        "sample_id": str(sample.sample_id),
                        "task": str(sample.task),
                    }
                    cycle_rows.extend({**base, **row} for row in current_cycles)
                    step_rows.extend({**base, **row} for row in current_steps)
                    event_rows.extend({**base, **row} for row in event_sink)
                    summaries.append(
                        {
                            **summary,
                            "sample_id": str(sample.sample_id),
                            "task": str(sample.task),
                            "prompt_tokens": len(reference.prompt_token_ids),
                            "monitor_encodings_json": json.dumps(encodings),
                        }
                    )
                    print(
                        f"[r2a-labels] {sample_index}/{len(samples)} "
                        f"{sample.sample_id} {policy} "
                        f"kl={summary['mean_trajectory_exact_kl']:.6f} "
                        f"events={len(event_sink)}",
                        flush=True,
                    )
                atomic_frame(pd.DataFrame(summaries), output / "partial_sample_results.csv")
                atomic_frame(pd.DataFrame(cycle_rows), output / "partial_cycle_rows.parquet")
                atomic_frame(pd.DataFrame(step_rows), output / "partial_step_rows.parquet")
                atomic_frame(
                    pd.DataFrame(event_rows),
                    output / "partial_refresh_event_rows.parquet",
                )
                atomic_json(
                    output / "progress.json",
                    {
                        "completed_samples": sample_index,
                        "expected_samples": len(samples),
                        "last_sample_id": str(sample.sample_id),
                    },
                )
            finally:
                runner.model.release(reference)
    finally:
        runner.model.close()
    sample_frame = pd.DataFrame(summaries)
    cycle_frame = pd.DataFrame(cycle_rows)
    step_frame = pd.DataFrame(step_rows)
    event_frame = pd.DataFrame(event_rows)
    aggregates = _aggregate(sample_frame, step_frame, ["policy"])
    result: Dict[str, Any] = {
        "experiment": str(config["experiment_name"]),
        "stage": "r2a_labels",
        "model_info": model_info,
        "samples": sorted(str(sample.sample_id) for sample in samples),
        "policies": list(policies),
        "label_stale_lags": list(label_lags),
        "policy_aggregates": aggregates,
        "refresh_event_rows": int(len(event_frame)),
        "all_irreversible_set_inclusions_hold": bool(
            sample_frame["irreversible_set_inclusion_all_cycles"].all()
        ),
        "persistent_cpu_kv_backing": False,
        "collection_elapsed_s": float(time.perf_counter() - started),
        "execution_valid": bool(
            len(sample_frame) == len(samples) * len(policies)
            and len(event_frame) == len(samples) * len(policies) * cycles
        ),
    }
    atomic_frame(sample_frame, output / "sample_results.csv")
    atomic_frame(cycle_frame, output / "cycle_rows.parquet")
    atomic_frame(step_frame, output / "step_rows.parquet")
    atomic_frame(event_frame, output / "refresh_event_rows.parquet")
    atomic_frame(pd.DataFrame(aggregates), output / "aggregate_results.csv")
    atomic_json(output / "summary.json", result)
    atomic_text(output / "config.yaml", yaml.safe_dump(dict(config), sort_keys=False))
    return output


def _r2b_arms(
    stage: Mapping[str, Any], repository_root: Path
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Build the 4-arm schedule; the trigger arm needs a frozen rule file."""
    arms: List[Dict[str, Any]] = [
        {"arm": "every", "refresh_mode": "every"},
        {"arm": "never", "refresh_mode": "never"},
    ]
    for k in stage.get("refresh_k", [4, 8, 16]):
        arms.append(
            {
                "arm": f"fixed_k{int(k)}",
                "refresh_mode": "fixed_k",
                "refresh_k": int(k),
            }
        )
    trigger_skip: Optional[Dict[str, Any]] = None
    rule_path_raw = stage.get("trigger_rule")
    rule_path = repository_root / str(rule_path_raw) if rule_path_raw else None
    if rule_path is not None and rule_path.exists():
        arms.append(
            {
                "arm": "trigger",
                "refresh_mode": "trigger",
                "trigger_rule_path": str(rule_path),
            }
        )
    else:
        trigger_skip = {
            "arm": "trigger",
            "skipped": True,
            "reason": (
                "no trigger_rule configured"
                if rule_path is None
                else f"frozen trigger rule file missing: {rule_path}"
            ),
        }
    return arms, trigger_skip


def run_r2b(config_path: Path, repository_root: Path) -> Path:
    """Run the R2b 4-arm selective-refresh gate on the held-out test split."""
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    stage = dict(config["r2b_gate"])
    output = repository_root / str(stage["output_run"])
    output.mkdir(parents=True, exist_ok=True)
    runner, samples = _load_runner(config, repository_root, "r2b_gate")
    monitor_cfg = dict(config.get("tail_telemetry") or {})
    policies = [str(value) for value in stage["policies"]]
    total_budget = int(stage["total_budget"])
    core_budget = int(stage["core_budget"])
    maximum_delta = int(stage["maximum_delta"])
    if total_budget != int(config["sink_size"]) + int(config["recent_size"]) + core_budget:
        raise ValueError("R2b total/core budget decomposition is inconsistent")
    cycles = int(stage["control_cycles"])
    arms, trigger_skip = _r2b_arms(stage, repository_root)
    trigger_rules: Dict[str, Any] = {}
    for arm in arms:
        if arm["refresh_mode"] == "trigger":
            trigger_rules[arm["arm"]] = load_trigger_rule(arm["trigger_rule_path"])
    previous_samples = _load_partial(output, "sample_results")
    previous_cycles = _load_partial(output, "cycle_rows", parquet=True)
    previous_steps = _load_partial(output, "step_rows", parquet=True)
    cycle_rows: List[Dict[str, Any]] = previous_cycles.to_dict("records")
    step_rows: List[Dict[str, Any]] = previous_steps.to_dict("records")
    summaries: List[Dict[str, Any]] = previous_samples.to_dict("records")
    completed = set()
    if not previous_samples.empty:
        completed = set(
            zip(
                previous_samples["sample_id"].astype(str),
                previous_samples["policy"].astype(str),
                previous_samples["arm"].astype(str),
            )
        )
    model_info = runner.model.load()
    started = time.perf_counter()
    try:
        for sample_index, sample in enumerate(samples, start=1):
            if all(
                (str(sample.sample_id), str(policy), str(arm["arm"])) in completed
                for policy in policies
                for arm in arms
            ):
                print(f"[r2b-gate] resume skip {sample.sample_id}", flush=True)
                continue
            reference = runner.model.generate_reference(
                sample.sample_id, sample.task, sample.prompt
            )
            monitored, evidence, encodings = _monitor_spec(
                runner.model.tokenizer,
                reference.prompt_token_ids,
                monitor_cfg.get("probability_labels", {}),
                monitor_cfg.get("evidence_phrases", []),
            )
            try:
                for policy in policies:
                    for arm in arms:
                        key = (str(sample.sample_id), str(policy), str(arm["arm"]))
                        if key in completed:
                            print(f"[r2b-gate] resume skip {key}", flush=True)
                            continue
                        controller = _controller(
                            config, core_budget, maximum_delta, pure=True
                        )
                        current_cycles, current_steps, summary = run_pure_eviction_policy(
                            runner,
                            reference,
                            sample,
                            policy,
                            controller,
                            cycles,
                            monitored,
                            evidence,
                            bool(stage.get("evaluate_exact_kl", True)),
                            refresh_mode=str(arm["refresh_mode"]),
                            refresh_k=int(arm.get("refresh_k", 4)),
                            trigger_rule=trigger_rules.get(arm["arm"]),
                        )
                        base = {
                            "sample_id": str(sample.sample_id),
                            "task": str(sample.task),
                            "arm": str(arm["arm"]),
                        }
                        cycle_rows.extend({**base, **row} for row in current_cycles)
                        step_rows.extend({**base, **row} for row in current_steps)
                        summaries.append(
                            {
                                **summary,
                                **base,
                                "total_budget": total_budget,
                                "core_budget": core_budget,
                                "prompt_tokens": len(reference.prompt_token_ids),
                                "monitor_encodings_json": json.dumps(encodings),
                            }
                        )
                        print(
                            f"[r2b-gate] {sample_index}/{len(samples)} "
                            f"{sample.sample_id} {policy} {arm['arm']} "
                            f"kl={summary['mean_trajectory_exact_kl']:.6f} "
                            f"refreshes={summary['refresh_count']} "
                            f"score={summary['official_score']:.4f}",
                            flush=True,
                        )
                        atomic_frame(
                            pd.DataFrame(summaries),
                            output / "partial_sample_results.csv",
                        )
                        atomic_frame(
                            pd.DataFrame(cycle_rows),
                            output / "partial_cycle_rows.parquet",
                        )
                        atomic_frame(
                            pd.DataFrame(step_rows),
                            output / "partial_step_rows.parquet",
                        )
                atomic_json(
                    output / "progress.json",
                    {
                        "completed_samples": sample_index,
                        "expected_samples": len(samples),
                        "last_sample_id": str(sample.sample_id),
                    },
                )
            finally:
                runner.model.release(reference)
    finally:
        runner.model.close()
    sample_frame = pd.DataFrame(summaries)
    cycle_frame = pd.DataFrame(cycle_rows)
    step_frame = pd.DataFrame(step_rows)
    aggregates = _aggregate(sample_frame, step_frame, ["arm", "policy"])
    result: Dict[str, Any] = {
        "experiment": str(config["experiment_name"]),
        "stage": "r2b_gate",
        "model_info": model_info,
        "samples": sorted(str(sample.sample_id) for sample in samples),
        "policies": list(policies),
        "arms": [
            {key: value for key, value in arm.items() if key != "trigger_rule_path"}
            for arm in arms
        ],
        "trigger_arm_skipped": trigger_skip,
        "trigger_rule": (
            {
                "path": str(arms[-1]["trigger_rule_path"]),
                "name": trigger_rules["trigger"].name,
                "clauses": [
                    {"feature": c.feature, "op": c.op, "threshold": c.threshold}
                    for c in trigger_rules["trigger"].clauses
                ],
                "provenance": dict(trigger_rules["trigger"].provenance or {}),
            }
            if "trigger" in trigger_rules
            else None
        ),
        "policy_aggregates": aggregates,
        "all_irreversible_set_inclusions_hold": bool(
            sample_frame["irreversible_set_inclusion_all_cycles"].all()
        ),
        "global_kv_budget_respected_all_cycles": bool(
            sample_frame["global_kv_budget_respected_all_cycles"].all()
        ),
        "persistent_cpu_kv_backing": False,
        "collection_elapsed_s": float(time.perf_counter() - started),
        "execution_valid": bool(
            len(sample_frame) == len(samples) * len(policies) * len(arms)
        ),
    }
    atomic_frame(sample_frame, output / "sample_results.csv")
    atomic_frame(cycle_frame, output / "cycle_rows.parquet")
    atomic_frame(step_frame, output / "step_rows.parquet")
    atomic_frame(pd.DataFrame(aggregates), output / "aggregate_results.csv")
    atomic_json(output / "summary.json", result)
    atomic_text(output / "config.yaml", yaml.safe_dump(dict(config), sort_keys=False))
    return output


__all__ = [
    "run_backed_policy",
    "run_calibration",
    "run_p1",
    "run_p2",
    "run_p2_profile",
    "run_p3",
    "run_pure_eviction_policy",
    "run_r2a",
    "run_r2b",
]
