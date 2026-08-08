"""Expensive teacher-forced closed-loop KV-cache controller.

The runner keeps one physical compressed state alive across control periods.
At every boundary it creates a fixed panel of legal retained sets, rolls each
one forward on the same reference tokens, scores the panel, commits the chosen
physical state, and repeats.  It is an oracle experiment, not a deployment
path: full-reference logits and several candidate rollouts are required.
"""
from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

from statekv.candidate_pullback import CandidatePullbackRunner
from statekv.config import CacheDiscoveryConfig, load_discovery_config
from statekv.core.decision import select_lowest_risk
from statekv.core.risk import state_conditioned_quadratic_risk
from statekv.selectors import CoreSelection, LayerSelection, mandatory_and_eligible
from statekv.storage import atomic_frame, atomic_json, atomic_text
from statekv.tasks import load_discovery_tasks
from statekv.trajectory_model import exact_distribution_metrics


@dataclass
class CandidateRollout:
    name: str
    core: Tuple[int, ...]
    state: Any
    last_record: Any
    next_token: int
    logits: List[torch.Tensor]
    step_rows: List[Dict[str, Any]]
    records: List[Any]
    position_maps_by_step: List[Dict[int, Tuple[int, ...]]]
    maximum_active_tokens: int


class KVBackingStore:
    """CPU-resident exact K/V rows available for later cold-token recovery."""

    def __init__(self) -> None:
        self.keys: Dict[int, Dict[int, torch.Tensor]] = {}
        self.values: Dict[int, Dict[int, torch.Tensor]] = {}

    def update(self, runner: CandidatePullbackRunner, state: Any) -> None:
        for layer, layer_cache in enumerate(state.cache):
            positions = [
                int(value) for value in state.position_maps[int(layer)].tolist()
            ]
            offset = int(layer_cache.offset)
            keys = runner.model._torch(
                layer_cache.keys[:, :, :offset, :], torch.float16
            )
            values = runner.model._torch(
                layer_cache.values[:, :, :offset, :], torch.float16
            )
            if int(keys.shape[2]) != len(positions):
                raise RuntimeError("backing-store key positions are misaligned")
            self.keys.setdefault(int(layer), {})
            self.values.setdefault(int(layer), {})
            for row, position in enumerate(positions):
                self.keys[int(layer)][position] = keys[:, :, row : row + 1, :].clone()
                self.values[int(layer)][position] = (
                    values[:, :, row : row + 1, :].clone()
                )

    def positions(self) -> List[int]:
        if not self.keys:
            return []
        layers = sorted(self.keys)
        reference = set(self.keys[layers[0]])
        if any(set(self.keys[layer]) != reference for layer in layers[1:]):
            raise RuntimeError("backing-store layer position universes diverged")
        return sorted(reference)

    def layer_arrays(self, layer: int) -> Tuple[torch.Tensor, torch.Tensor]:
        positions = self.positions()
        return (
            torch.cat([self.keys[int(layer)][value] for value in positions], dim=2),
            torch.cat([self.values[int(layer)][value] for value in positions], dim=2),
        )

    def anchor(self, logical_next_position: int, query_token_id: int) -> Any:
        from statekv.backend import AnchorState
        from kvbench.types import AttentionSignals

        positions = self.positions()
        layers = sorted(self.keys)
        return AnchorState(
            anchor_step=int(logical_next_position),
            logical_length=int(logical_next_position) + 1,
            query_token_id=int(query_token_id),
            keys=[self.layer_arrays(layer)[0] for layer in layers],
            values=[self.layer_arrays(layer)[1] for layer in layers],
            position_maps={
                int(layer): torch.tensor(positions, dtype=torch.long)
                for layer in layers
            },
            attention=AttentionSignals({}, {}, {}, {}),
        )


def _finite_mean(values: Iterable[float]) -> float:
    result = np.asarray(tuple(float(value) for value in values), dtype=np.float64)
    if result.size == 0 or not np.isfinite(result).all():
        raise ValueError("teacher scores must be non-empty and finite")
    return float(np.mean(result))


def aggregate_teacher_scores(
    strategy: str,
    exact_kl_by_candidate: Mapping[str, Sequence[float]],
    dense_risk_by_candidate: Mapping[str, Sequence[float]],
) -> Dict[str, float]:
    """Aggregate one fixed candidate panel under a declared teacher objective."""

    names = set(exact_kl_by_candidate)
    if not names or names != set(dense_risk_by_candidate):
        raise ValueError("exact and dense candidate panels must match")
    if strategy == "exact_mean":
        return {
            name: _finite_mean(exact_kl_by_candidate[name]) for name in names
        }
    if strategy == "exact_max":
        return {
            name: float(max(float(value) for value in exact_kl_by_candidate[name]))
            for name in names
        }
    if strategy == "dense_quadratic_h1":
        return {
            name: float(tuple(dense_risk_by_candidate[name])[0]) for name in names
        }
    if strategy == "dense_quadratic_mean":
        return {
            name: _finite_mean(dense_risk_by_candidate[name]) for name in names
        }
    raise ValueError("unknown teacher strategy=%s" % strategy)


def deterministic_uniform_core(
    eligible_positions: Sequence[int], budget: int
) -> Tuple[int, ...]:
    """Select a deterministic position-coverage core without duplicate slots."""

    eligible = tuple(int(value) for value in eligible_positions)
    take = min(int(budget), len(eligible))
    if take < 0:
        raise ValueError("budget must be non-negative")
    if take == 0:
        return ()
    anchors = np.linspace(0, len(eligible) - 1, num=take)
    chosen = []
    for index in np.rint(anchors).astype(np.int64).tolist():
        value = eligible[int(index)]
        if value not in chosen:
            chosen.append(value)
    if len(chosen) < take:
        chosen.extend(value for value in eligible if value not in set(chosen))
    return tuple(sorted(chosen[:take]))


def _normalize_on_eligible(
    values: np.ndarray, eligible_rows: np.ndarray
) -> np.ndarray:
    score = np.maximum(np.asarray(values, dtype=np.float64).reshape(-1), 0.0)
    output = np.zeros_like(score)
    mass = float(score[eligible_rows].sum())
    if mass > 0.0:
        output[eligible_rows] = score[eligible_rows] / mass
    return output


def _top_core(
    positions: Sequence[int], eligible: Sequence[int], score: np.ndarray, budget: int
) -> Tuple[int, ...]:
    row_by_position = {
        int(position): row for row, position in enumerate(positions)
    }
    ranked = sorted(
        (int(position) for position in eligible),
        key=lambda position: (-float(score[row_by_position[position]]), position),
    )
    return tuple(sorted(ranked[: min(int(budget), len(ranked))]))


def _stale_core(
    previous_core: Optional[Sequence[int]],
    eligible: Sequence[int],
    attention_score: np.ndarray,
    positions: Sequence[int],
    budget: int,
) -> Tuple[int, ...]:
    if previous_core is None:
        return _top_core(positions, eligible, attention_score, budget)
    eligible_set = set(int(value) for value in eligible)
    preserved = sorted(
        set(int(value) for value in previous_core) & eligible_set
    )
    if len(preserved) > int(budget):
        preserved = list(
            _top_core(positions, preserved, attention_score, int(budget))
        )
    fill = _top_core(
        positions,
        [value for value in eligible if value not in set(preserved)],
        attention_score,
        max(0, int(budget) - len(preserved)),
    )
    return tuple(sorted(preserved + list(fill)))


def _selection(
    name: str,
    positions_by_layer: Mapping[int, Sequence[int]],
    core: Sequence[int],
    eligible: Sequence[int],
    score: np.ndarray,
) -> CoreSelection:
    normalized_core = tuple(sorted(int(value) for value in core))
    by_layer = {
        int(layer): LayerSelection(
            layer=int(layer),
            selected_positions=list(normalized_core),
            eligible_positions=[int(value) for value in eligible],
            aggregate_scores=[float(value) for value in score],
            metadata={
                "source": name,
                "physical_shared_mask": True,
                "oracle_candidate": True,
            },
        )
        for layer in positions_by_layer
    }
    digest = hashlib.sha256(
        np.asarray(normalized_core, dtype=np.int64).tobytes()
    ).hexdigest()
    return CoreSelection(
        strategy=name,
        horizon_condition=None,
        by_layer=by_layer,
        metadata={"selection_hash": digest, "physical_shared_mask": True},
    )


def _live_scores(
    runner: CandidatePullbackRunner,
    state: Any,
    last_record: Any,
    backing: KVBackingStore,
    attention_memory: Dict[int, float],
    diagnostic_layers: Sequence[int],
    sink_size: int,
    recent_size: int,
) -> Tuple[List[int], List[int], Mapping[str, np.ndarray]]:
    active_positions = [int(value) for value in state.position_maps[0].tolist()]
    if any(
        [int(value) for value in current.tolist()] != active_positions
        for current in state.position_maps.values()
    ):
        raise RuntimeError("closed-loop shared candidates require aligned maps")
    positions = backing.positions()
    if not set(active_positions) <= set(positions):
        raise RuntimeError("active cache is not contained in the backing store")
    _, _, eligible = mandatory_and_eligible(positions, sink_size, recent_size)
    row_by_position = {position: row for row, position in enumerate(positions)}
    active_row_by_position = {
        position: row for row, position in enumerate(active_positions)
    }
    eligible_rows = np.asarray(
        [row_by_position[position] for position in eligible], dtype=np.int64
    )
    attention_layers = []
    key_layers = []
    value_layers = []
    for layer in diagnostic_layers:
        distribution = (
            last_record.all_head_attention_distributions[int(layer)]
            .detach()
            .float()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        if int(distribution.shape[-1]) != len(active_positions):
            raise RuntimeError(
                "attention/cache support mismatch at layer=%d: %d != %d"
                % (
                    int(layer),
                    int(distribution.shape[-1]),
                    len(active_positions),
                )
            )
        active_attention = distribution.mean(axis=0)
        expanded_attention = np.zeros(len(positions), dtype=np.float64)
        for position in active_positions:
            expanded_attention[row_by_position[position]] = float(
                active_attention[active_row_by_position[position]]
            )
        attention_layers.append(
            _normalize_on_eligible(expanded_attention, eligible_rows)
        )
        keys_t, values_t = backing.layer_arrays(int(layer))
        keys = keys_t.float().numpy()
        values = values_t.float().numpy()
        key_layers.append(
            _normalize_on_eligible(
                np.linalg.norm(keys, axis=-1).mean(axis=(0, 1)), eligible_rows
            )
        )
        value_layers.append(
            _normalize_on_eligible(
                np.linalg.norm(values, axis=-1).mean(axis=(0, 1)), eligible_rows
            )
        )
    attention = np.mean(np.stack(attention_layers, axis=0), axis=0)
    key_norm = np.mean(np.stack(key_layers, axis=0), axis=0)
    value_norm = np.mean(np.stack(value_layers, axis=0), axis=0)
    for position in active_positions:
        row = row_by_position[position]
        attention_memory[position] = max(
            float(attention_memory.get(position, 0.0)), float(attention[row])
        )
    historical_attention = np.asarray(
        [float(attention_memory.get(position, 0.0)) for position in positions],
        dtype=np.float64,
    )
    historical_attention = _normalize_on_eligible(
        historical_attention, eligible_rows
    )
    return positions, eligible, {
        "attention": attention,
        "attention_history": historical_attention,
        "key_norm": key_norm,
        "value_norm": value_norm,
        "attention_value": attention * np.sqrt(np.maximum(value_norm, 0.0)),
    }


def _candidate_panel(
    runner: CandidatePullbackRunner,
    state: Any,
    last_record: Any,
    backing: KVBackingStore,
    attention_memory: Dict[int, float],
    previous_core: Optional[Sequence[int]],
    candidate_names: Sequence[str],
    diagnostic_layers: Sequence[int],
    sink_size: int,
    recent_size: int,
    core_budget: int,
) -> Mapping[str, CoreSelection]:
    positions, eligible, scores = _live_scores(
        runner,
        state,
        last_record,
        backing,
        attention_memory,
        diagnostic_layers,
        sink_size,
        recent_size,
    )
    positions_by_layer = {
        int(layer): list(positions) for layer in state.position_maps
    }
    result: Dict[str, CoreSelection] = {}
    for name in candidate_names:
        if name == "stale":
            core = _stale_core(
                previous_core,
                eligible,
                scores["attention"],
                positions,
                core_budget,
            )
            score = scores["attention"]
        elif name == "uniform":
            core = deterministic_uniform_core(eligible, core_budget)
            score = np.zeros(len(positions), dtype=np.float64)
            row_by_position = {
                position: row for row, position in enumerate(positions)
            }
            for position in core:
                score[row_by_position[position]] = 1.0
        elif name in scores:
            score = scores[name]
            core = _top_core(positions, eligible, score, core_budget)
        else:
            raise ValueError("unknown oracle candidate=%s" % name)
        if len(core) != min(int(core_budget), len(eligible)):
            raise RuntimeError("candidate core does not fill its legal budget")
        result[name] = _selection(
            name, positions_by_layer, core, eligible, score
        )
    return result


def _rollout_candidate(
    runner: CandidatePullbackRunner,
    reference: Any,
    base_state: Any,
    backing: KVBackingStore,
    current_token: int,
    selection: CoreSelection,
    target_start: int,
    horizon: int,
    initial_cache: CacheDiscoveryConfig,
    rolling_cache: CacheDiscoveryConfig,
) -> CandidateRollout:
    anchor = backing.anchor(
        int(base_state.logical_next_position), int(current_token)
    )
    state, fixed = runner.model.state_from_anchor(
        anchor, selection, cache_config=initial_cache
    )
    token = int(current_token)
    logits_rows: List[torch.Tensor] = []
    metric_rows: List[Dict[str, Any]] = []
    records: List[Any] = []
    position_maps_by_step: List[Dict[int, Tuple[int, ...]]] = []
    last_record = None
    maximum_active = 0
    for offset in range(int(horizon)):
        target_index = int(target_start + offset)
        if offset > 0:
            runner.model.prune_recent_before_query(
                state, fixed, cache_config=rolling_cache
            )
        runner._clear_controls()
        logits, record, forward_s = runner.model.forward_one(
            state, token, capture_attention=True
        )
        runner.model.validate_active_budget(state, cache_config=rolling_cache)
        active = int(runner.model.active_cache_tokens(state))
        maximum_active = max(maximum_active, active)
        target_token = int(reference.generated_token_ids[target_index])
        metrics = exact_distribution_metrics(
            reference.probe_logits[target_index], logits, target_token
        )
        metric_rows.append(
            {
                "horizon_offset": int(offset + 1),
                "target_index": target_index,
                "active_cache_tokens": active,
                "forward_time_s": float(forward_s),
                **metrics,
            }
        )
        logits_rows.append(logits.detach().float().cpu())
        records.append(record)
        position_maps_by_step.append(
            {
                int(layer): tuple(
                    int(value) for value in positions.tolist()
                )
                for layer, positions in state.position_maps.items()
            }
        )
        last_record = record
        token = target_token
    if last_record is None:
        raise RuntimeError("candidate rollout produced no steps")
    return CandidateRollout(
        name=str(selection.strategy),
        core=tuple(selection.by_layer[0].selected_positions),
        state=state,
        last_record=last_record,
        next_token=token,
        logits=logits_rows,
        step_rows=metric_rows,
        records=records,
        position_maps_by_step=position_maps_by_step,
        maximum_active_tokens=maximum_active,
    )


def _dense_scores(
    reference: Any,
    target_start: int,
    outcomes: Mapping[str, CandidateRollout],
) -> Mapping[str, Sequence[float]]:
    stale = outcomes["stale"]
    result: Dict[str, List[float]] = {}
    for name, outcome in outcomes.items():
        values = []
        for offset, (candidate_logits, stale_logits) in enumerate(
            zip(outcome.logits, stale.logits)
        ):
            reference_logits = reference.probe_logits[
                int(target_start + offset)
            ].detach().double().cpu()
            state_logits = stale_logits.double()
            delta_logits = candidate_logits.double() - state_logits
            value = state_conditioned_quadratic_risk(
                reference_logits, state_logits, delta_logits
            )
            values.append(float(value.item()))
        result[name] = values
    return result


def _run_strategy(
    runner: CandidatePullbackRunner,
    reference: Any,
    strategy: str,
    candidate_names: Sequence[str],
    start_anchor: int,
    cycles: int,
    horizon: int,
    diagnostic_layers: Sequence[int],
    total_budget: int,
    sink_size: int,
    recent_size: int,
    core_budget: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    full_selection = runner._all_history_selection(reference, int(start_anchor))
    anchor_state = reference.anchors[int(start_anchor)]
    full_cache = CacheDiscoveryConfig(
        total_budget=int(anchor_state.logical_length + 2),
        sink_size=0,
        recent_size=1,
        selected_core_budget=int(anchor_state.logical_length + 1),
    )
    base_state, _ = runner.model.state_from_anchor(
        anchor_state, full_selection, cache_config=full_cache
    )
    backing = KVBackingStore()
    backing.update(runner, base_state)
    attention_memory: Dict[int, float] = {}
    current_token = int(anchor_state.query_token_id)
    last_record = reference.query_records[int(start_anchor) - 1]
    previous_core: Optional[Tuple[int, ...]] = None
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
    cycle_rows: List[Dict[str, Any]] = []
    candidate_rows: List[Dict[str, Any]] = []
    previous_output_position: Optional[int] = None
    for cycle in range(int(cycles)):
        target_start = int(start_anchor + cycle * horizon)
        input_position = int(base_state.logical_next_position)
        continuity = (
            previous_output_position is None
            or input_position == previous_output_position
        )
        input_positions = set(
            int(value) for value in base_state.position_maps[0].tolist()
        )
        backing.update(runner, base_state)
        carry_fraction = (
            1.0
            if previous_core is None
            else float(
                len(set(previous_core) & input_positions)
                / max(1, len(set(previous_core)))
            )
        )
        panel = _candidate_panel(
            runner,
            base_state,
            last_record,
            backing,
            attention_memory,
            previous_core,
            candidate_names,
            diagnostic_layers,
            sink_size,
            recent_size,
            core_budget,
        )
        outcomes = {
            name: _rollout_candidate(
                runner,
                reference,
                base_state,
                backing,
                current_token,
                selection,
                target_start,
                horizon,
                initial_cache,
                rolling_cache,
            )
            for name, selection in panel.items()
        }
        exact = {
            name: [float(row["exact_kl"]) for row in outcome.step_rows]
            for name, outcome in outcomes.items()
        }
        dense = _dense_scores(reference, target_start, outcomes)
        scores = aggregate_teacher_scores(strategy, exact, dense)
        decision = select_lowest_risk(scores)
        selected_name = str(decision.candidate_id)
        selected = outcomes[selected_name]
        stale = outcomes["stale"]
        regret = float(scores["stale"] - scores[selected_name])
        if regret < -1.0e-10:
            raise RuntimeError("teacher regret is negative after minimization")
        stale_core = tuple(stale.core)
        refresh = tuple(selected.core) != stale_core
        unique_cores = len({tuple(value.core) for value in outcomes.values()})
        selected_recovered = len(set(selected.core) - input_positions)
        for name, outcome in outcomes.items():
            for offset, row in enumerate(outcome.step_rows):
                candidate_rows.append(
                    {
                        "strategy": strategy,
                        "cycle": int(cycle),
                        "candidate": name,
                        "selected": name == selected_name,
                        "teacher_score": float(scores[name]),
                        "dense_quadratic_risk": float(dense[name][offset]),
                        **row,
                    }
                )
        output_position = int(selected.state.logical_next_position)
        selected_core_survives = float(
            len(set(selected.core) & set(selected.state.position_maps[0].tolist()))
            / max(1, len(set(selected.core)))
        )
        cycle_rows.append(
            {
                "strategy": strategy,
                "cycle": int(cycle),
                "target_start": target_start,
                "input_logical_position": input_position,
                "output_logical_position": output_position,
                "state_advanced_by": int(output_position - input_position),
                "state_continuity": bool(continuity),
                "previous_core_carry_fraction": carry_fraction,
                "selected_core_survival_fraction": selected_core_survives,
                "candidate_count": int(len(outcomes)),
                "unique_candidate_cores": int(unique_cores),
                "selected_candidate": selected_name,
                "selected_teacher_score": float(scores[selected_name]),
                "stale_teacher_score": float(scores["stale"]),
                "teacher_regret": max(regret, 0.0),
                "refresh": bool(refresh),
                "selected_recovered_core_tokens": int(selected_recovered),
                "selected_exact_kl_mean": _finite_mean(exact[selected_name]),
                "stale_exact_kl_mean": _finite_mean(exact["stale"]),
                "selected_exact_kl_max": float(max(exact[selected_name])),
                "maximum_active_cache_tokens": int(
                    max(value.maximum_active_tokens for value in outcomes.values())
                ),
                "budget_respected": bool(
                    all(
                        value.maximum_active_tokens <= int(total_budget)
                        for value in outcomes.values()
                    )
                ),
            }
        )
        base_state = selected.state
        backing.update(runner, base_state)
        current_token = int(selected.next_token)
        last_record = selected.last_record
        previous_core = tuple(selected.core)
        previous_output_position = output_position
        for name, outcome in outcomes.items():
            if name != selected_name:
                outcome.state = None
    runner.model.release(base_state)
    summary = {
        "strategy": strategy,
        "cycles_completed": int(len(cycle_rows)),
        "refresh_events": int(sum(bool(row["refresh"]) for row in cycle_rows)),
        "post_initial_refresh_events": int(
            sum(bool(row["refresh"]) for row in cycle_rows[1:])
        ),
        "recovery_events": int(
            sum(
                int(row["selected_recovered_core_tokens"]) > 0
                for row in cycle_rows[1:]
            )
        ),
        "selected_candidates": [row["selected_candidate"] for row in cycle_rows],
        "all_state_continuity": bool(
            all(bool(row["state_continuity"]) for row in cycle_rows)
        ),
        "all_periods_advanced": bool(
            all(int(row["state_advanced_by"]) == int(horizon) for row in cycle_rows)
        ),
        "minimum_previous_core_carry_fraction": float(
            min(row["previous_core_carry_fraction"] for row in cycle_rows)
        ),
        "minimum_selected_core_survival_fraction": float(
            min(row["selected_core_survival_fraction"] for row in cycle_rows)
        ),
        "maximum_active_cache_tokens": int(
            max(row["maximum_active_cache_tokens"] for row in cycle_rows)
        ),
        "all_budgets_respected": bool(
            all(bool(row["budget_respected"]) for row in cycle_rows)
        ),
        "all_regrets_nonnegative": bool(
            all(float(row["teacher_regret"]) >= 0.0 for row in cycle_rows)
        ),
        "minimum_unique_candidate_cores": int(
            min(row["unique_candidate_cores"] for row in cycle_rows)
        ),
        "mean_selected_exact_kl": float(
            np.mean([row["selected_exact_kl_mean"] for row in cycle_rows])
        ),
        "mean_stale_exact_kl": float(
            np.mean([row["stale_exact_kl_mean"] for row in cycle_rows])
        ),
    }
    summary["closed_loop_passed"] = bool(
        summary["cycles_completed"] == int(cycles)
        and summary["all_state_continuity"]
        and summary["all_periods_advanced"]
        and summary["minimum_previous_core_carry_fraction"] >= 1.0
        and summary["minimum_selected_core_survival_fraction"] >= 1.0
        and summary["all_budgets_respected"]
        and summary["all_regrets_nonnegative"]
        and summary["minimum_unique_candidate_cores"] >= 2
    )
    return cycle_rows, candidate_rows, summary


def run_oracle_closed_loop(config_path: Path, repository_root: Path) -> Path:
    """Execute all declared oracle strategies on continuous physical states."""

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    cfg = load_discovery_config(str(repository_root / str(config["base_config"])))
    cfg.tasks = dict(config["task_overrides"])
    cfg.runtime.seed = int(config["data_seed"])
    cfg.runtime.run_id = str(config["runtime_run_id"])
    sample_ids = set(str(value) for value in config["sample_ids"])
    output_root = repository_root / str(config["output_run"])
    output_root.mkdir(parents=True, exist_ok=True)
    strategies = [str(value) for value in config["teacher_strategies"]]
    candidates = [str(value) for value in config["candidate_panel"]]
    if "stale" not in candidates:
        raise ValueError("candidate panel must contain stale")
    start_anchor = int(config["start_anchor"])
    cycles = int(config["control_cycles"])
    horizon = int(config["control_horizon"])
    total_budget = int(config["total_budget"])
    sink_size = int(config["sink_size"])
    recent_size = int(config["recent_size"])
    core_budget = int(config["core_budget"])
    if core_budget != total_budget - sink_size - recent_size:
        raise ValueError("core budget must fill total minus sink and recent")
    diagnostic_layers = [int(value) for value in config["diagnostic_layers"]]
    runner = CandidatePullbackRunner(cfg, repository_root)
    samples, _ = load_discovery_tasks(cfg)
    selected_samples = [
        sample for sample in samples if str(sample.sample_id) in sample_ids
    ]
    if {str(sample.sample_id) for sample in selected_samples} != sample_ids:
        raise RuntimeError("configured closed-loop samples were not loaded")
    cycle_rows: List[Dict[str, Any]] = []
    candidate_rows: List[Dict[str, Any]] = []
    strategy_summaries: List[Dict[str, Any]] = []
    started = time.perf_counter()
    runner.model.load()
    try:
        for sample in selected_samples:
            target_indices = list(
                range(start_anchor, start_anchor + cycles * horizon)
            )
            reference = runner.model.generate_reference(
                sample.sample_id,
                sample.task,
                sample.prompt,
                extra_probe_target_indices=target_indices,
            )
            try:
                for strategy in strategies:
                    current_cycles, current_candidates, current_summary = (
                        _run_strategy(
                            runner,
                            reference,
                            strategy,
                            candidates,
                            start_anchor,
                            cycles,
                            horizon,
                            diagnostic_layers,
                            total_budget,
                            sink_size,
                            recent_size,
                            core_budget,
                        )
                    )
                    base = {
                        "sample_id": str(sample.sample_id),
                        "task": str(sample.task),
                    }
                    cycle_rows.extend({**base, **row} for row in current_cycles)
                    candidate_rows.extend(
                        {**base, **row} for row in current_candidates
                    )
                    strategy_summaries.append({**base, **current_summary})
            finally:
                runner.model.release(reference)
    finally:
        runner.model.close()
    summaries = pd.DataFrame(strategy_summaries)
    refresh_events = int(summaries["refresh_events"].sum())
    post_initial_refresh_events = int(
        summaries["post_initial_refresh_events"].sum()
    )
    recovery_events = int(summaries["recovery_events"].sum())
    result = {
        "experiment": str(config["experiment_name"]),
        "status": "teacher_forced_physical_oracle_closed_loop",
        "samples": sorted(sample_ids),
        "teacher_strategies": strategies,
        "candidate_panel": candidates,
        "control_cycles": cycles,
        "control_horizon": horizon,
        "total_budget": total_budget,
        "all_strategy_sample_loops_passed": bool(
            summaries["closed_loop_passed"].all()
        ),
        "refresh_branch_executed": bool(refresh_events > 0),
        "refresh_events": refresh_events,
        "post_initial_refresh_events": post_initial_refresh_events,
        "recovery_branch_executed": bool(recovery_events > 0),
        "recovery_events": recovery_events,
        "passed": bool(
            summaries["closed_loop_passed"].all()
            and post_initial_refresh_events > 0
            and recovery_events > 0
        ),
        "strategy_sample_summaries": strategy_summaries,
        "collection_elapsed_s": float(time.perf_counter() - started),
        "scope": (
            "Teacher-forced physical closed loop with full-reference logits and "
            "candidate rollouts. This validates state continuity and oracle "
            "control mechanics, not deployment efficiency or free generation."
        ),
    }
    atomic_frame(pd.DataFrame(cycle_rows), output_root / "cycle_rows.parquet")
    atomic_frame(
        pd.DataFrame(candidate_rows), output_root / "candidate_step_rows.parquet"
    )
    atomic_frame(summaries, output_root / "strategy_sample_summary.csv")
    atomic_json(output_root / "summary.json", result)
    atomic_text(
        output_root / "config.yaml",
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
    )
    return output_root


__all__ = [
    "aggregate_teacher_scores",
    "deterministic_uniform_core",
    "run_oracle_closed_loop",
]
