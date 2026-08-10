"""Matched closed-loop comparison of StateKV's risk teacher and fixed policies.

Each policy owns an independent physical compressed trajectory.  All policies
consume the same full-cache tokens in this module, so full-vocabulary KL and NLL
remain exactly paired.  The StateKV policy chooses the lowest exact-mean-risk
candidate at every boundary; fixed policies repeatedly commit their own action.
"""
from __future__ import annotations

import hashlib
import math
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

from statekv.candidate_pullback import CandidatePullbackRunner
from statekv.config import CacheDiscoveryConfig, load_discovery_config
from statekv.core.decision import select_lowest_risk
from statekv.oracle_closed_loop import (
    KVBackingStore,
    _rollout_candidate,
    _stale_core,
    _top_core,
    deterministic_uniform_core,
    quest_like_core,
    recency_core,
)
from statekv.selectors import (
    CoreSelection,
    LayerSelection,
    mandatory_and_eligible,
)
from statekv.storage import atomic_frame, atomic_json, atomic_text
from statekv.tasks import load_discovery_tasks


def _mean_attention(value: torch.Tensor) -> np.ndarray:
    array = value.detach().float().cpu().numpy().astype(np.float64)
    if array.ndim == 1:
        return array
    return array.reshape(-1, array.shape[-1]).mean(axis=0)


def _score_on_universe(
    score_by_position: Mapping[int, float], positions: Sequence[int]
) -> np.ndarray:
    return np.asarray(
        [float(score_by_position.get(int(position), 0.0)) for position in positions],
        dtype=np.float64,
    )


def token_rarity_scores(
    stream_token_ids: Sequence[int], positions: Sequence[int]
) -> np.ndarray:
    """Local span-smoothed inverse-frequency score of the observed stream.

    Mirrors the frozen P20/P21 ``token_rarity_shared`` policy
    (``benchmarks/mlx/src/runners/mlx_runner.py``): the score at position p
    is the mean of ``log((t + 1) / (f(x_j) + 1))`` over the clipped local
    span ``{p-2, ..., p+2}``, where ``f`` counts token ids in the observed
    stream of length ``t``.  Positions outside the stream score 0.  The
    score is model-free and identical across layers (one shared core).
    """
    ids = [int(value) for value in stream_token_ids]
    counts = Counter(ids)
    total = max(1, len(ids))
    values: List[float] = []
    for position in positions:
        index = int(position)
        if index < 0 or index >= len(ids):
            values.append(0.0)
            continue
        start = max(0, index - 2)
        end = min(len(ids), index + 3)
        local = [
            math.log((total + 1.0) / (max(1, counts[int(neighbor)]) + 1.0))
            for neighbor in ids[start:end]
        ]
        values.append(float(sum(local) / max(1, len(local))))
    return np.asarray(values, dtype=np.float64)


@dataclass
class AttentionPolicyMemory:
    """Attention state needed by latest-attention, SnapKV, and H2O."""

    layers: Tuple[int, ...]
    window_size: int
    latest: Dict[int, Dict[int, float]] = field(default_factory=dict)
    latest_by_head: Dict[int, Dict[int, np.ndarray]] = field(default_factory=dict)
    cumulative: Dict[int, Dict[int, float]] = field(default_factory=dict)
    window: Dict[int, List[Dict[int, float]]] = field(default_factory=dict)

    @classmethod
    def initialize(
        cls,
        reference: Any,
        start_anchor: int,
        layers: Sequence[int],
        window_size: int,
        universe: Sequence[int],
    ) -> "AttentionPolicyMemory":
        result = cls(tuple(int(value) for value in layers), int(window_size))
        anchor = reference.anchors[int(start_anchor)]
        current_record = reference.query_records[int(start_anchor)]
        universe_set = set(int(value) for value in universe)
        for layer in result.layers:
            accumulated = _mean_attention(
                anchor.attention.accumulated_by_layer[int(layer)]
            )
            current = _mean_attention(
                current_record.oracle_attention_by_layer[int(layer)]
            )
            positions = [
                int(value)
                for value in anchor.position_maps[int(layer)].tolist()
            ]
            result.cumulative[int(layer)] = {
                position: max(
                    0.0,
                    float(accumulated[row])
                    - (float(current[row]) if row < len(current) else 0.0),
                )
                for row, position in enumerate(positions)
                if position in universe_set
            }
        if int(start_anchor) == 0 and anchor.attention_observation_rows:
            for layer in result.layers:
                rows = anchor.attention_observation_rows.get(int(layer), [])
                # The anchor is rewound by one token before selection, so the
                # final prompt query is excluded from the policy memory.
                for value in rows[:-1]:
                    scores = _mean_attention(value)
                    head_scores = (
                        value.detach()
                        .float()
                        .cpu()
                        .numpy()
                        .astype(np.float64)
                        .reshape(-1, int(value.shape[-1]))
                    )
                    mapping = {
                        int(position): float(scores[index])
                        for index, position in enumerate(universe)
                        if index < len(scores)
                    }
                    result.latest_by_head[int(layer)] = {
                        int(position): head_scores[:, index].copy()
                        for index, position in enumerate(universe)
                        if index < int(head_scores.shape[-1])
                    }
                    result.latest[int(layer)] = mapping
                    result.window.setdefault(int(layer), []).append(mapping)
                history = result.window.setdefault(int(layer), [])
                if len(history) > int(window_size):
                    del history[: len(history) - int(window_size)]
        else:
            begin = max(0, int(start_anchor) - int(window_size))
            for index in range(begin, int(start_anchor)):
                record = reference.query_records[int(index)]
                maps = {
                    int(layer): tuple(
                        range(
                            int(
                                record.oracle_attention_by_layer[
                                    int(layer)
                                ].shape[-1]
                            )
                        )
                    )
                    for layer in result.layers
                }
                result.update_record(record, maps, update_cumulative=False)
        return result

    def update_record(
        self,
        record: Any,
        position_maps: Mapping[int, Sequence[int]],
        *,
        update_cumulative: bool = True,
    ) -> None:
        for layer in self.layers:
            positions = tuple(int(value) for value in position_maps[int(layer)])
            values = _mean_attention(
                record.oracle_attention_by_layer[int(layer)]
            )
            raw_values = (
                record.oracle_attention_by_layer[int(layer)]
                .detach()
                .float()
                .cpu()
                .numpy()
                .astype(np.float64)
                .reshape(-1, len(positions))
            )
            if len(values) != len(positions):
                raise RuntimeError(
                    "policy attention and physical position map are misaligned"
                )
            row = {
                int(position): float(values[index])
                for index, position in enumerate(positions)
            }
            self.latest[int(layer)] = row
            self.latest_by_head[int(layer)] = {
                int(position): raw_values[:, index].copy()
                for index, position in enumerate(positions)
            }
            history = self.window.setdefault(int(layer), [])
            history.append(row)
            if len(history) > int(self.window_size):
                del history[: len(history) - int(self.window_size)]
            if update_cumulative:
                cumulative = self.cumulative.setdefault(int(layer), {})
                for position, value in row.items():
                    cumulative[position] = float(cumulative.get(position, 0.0)) + value

    def update_rollout(self, rollout: Any) -> None:
        for record, maps in zip(
            rollout.records, rollout.position_maps_by_step
        ):
            self.update_record(record, maps)

    def score(
        self,
        layer: int,
        positions: Sequence[int],
        source: str,
        pooling_kernel: int,
        pooling_method: str,
    ) -> np.ndarray:
        if source == "attention":
            return _score_on_universe(self.latest.get(int(layer), {}), positions)
        if source == "h2o":
            return _score_on_universe(
                self.cumulative.get(int(layer), {}), positions
            )
        if source != "snapkv":
            raise ValueError("unknown attention memory source=%s" % source)
        combined: Dict[int, float] = {}
        for row in self.window.get(int(layer), []):
            for position, value in row.items():
                combined[position] = float(combined.get(position, 0.0)) + value
        raw = _score_on_universe(combined, positions)
        from src.runners.mlx_runner import snapkv_pool_scores_numpy

        return np.asarray(
            snapkv_pool_scores_numpy(
                raw, int(pooling_kernel), str(pooling_method)
            ),
            dtype=np.float64,
        )

    def head_score(
        self, layer: int, positions: Sequence[int]
    ) -> np.ndarray:
        """Return the latest query-head attention on a requested universe."""

        rows = self.latest_by_head.get(int(layer), {})
        if not rows:
            return np.empty((0, len(positions)), dtype=np.float64)
        head_count = int(next(iter(rows.values())).shape[0])
        output = np.zeros((head_count, len(positions)), dtype=np.float64)
        for index, position in enumerate(positions):
            value = rows.get(int(position))
            if value is not None:
                output[:, index] = value
        return output

    def volatility_score(
        self, layer: int, positions: Sequence[int]
    ) -> np.ndarray:
        """Recent temporal variation of head-averaged token attention."""

        history = self.window.get(int(layer), [])
        if len(history) < 2:
            return _score_on_universe(
                self.latest.get(int(layer), {}), positions
            )
        matrix = np.stack(
            [_score_on_universe(row, positions) for row in history], axis=0
        )
        return np.std(matrix, axis=0, dtype=np.float64)


def _selection_digest(cores: Mapping[int, Sequence[int]]) -> str:
    digest = hashlib.sha256()
    for layer in sorted(cores):
        digest.update(np.asarray([int(layer)], dtype=np.int64).tobytes())
        digest.update(
            np.asarray(tuple(cores[layer]), dtype=np.int64).tobytes()
        )
    return digest.hexdigest()


def _selection_from_scores(
    name: str,
    positions: Sequence[int],
    eligible: Sequence[int],
    cores: Mapping[int, Sequence[int]],
    scores: Mapping[int, np.ndarray],
) -> CoreSelection:
    by_layer = {
        int(layer): LayerSelection(
            layer=int(layer),
            selected_positions=sorted(int(value) for value in cores[int(layer)]),
            eligible_positions=[int(value) for value in eligible],
            aggregate_scores=[float(value) for value in scores[int(layer)]],
            metadata={
                "source": str(name),
                "physical_shared_mask": False,
                "per_layer_shared_across_kv_heads": True,
            },
        )
        for layer in sorted(cores)
    }
    return CoreSelection(
        strategy=str(name),
        horizon_condition=None,
        by_layer=by_layer,
        metadata={
            "selection_hash": _selection_digest(cores),
            "per_layer_shared_across_kv_heads": True,
        },
    )


def _physical_candidate_panel(
    runner: CandidatePullbackRunner,
    state: Any,
    backing: KVBackingStore,
    memory: AttentionPolicyMemory,
    previous_cores: Optional[Mapping[int, Sequence[int]]],
    candidate_names: Sequence[str],
    sink_size: int,
    recent_size: int,
    core_budget: int,
    pooling_kernel: int,
    pooling_method: str,
    pool_scores: Optional[Mapping[int, Mapping[int, float]]] = None,
    quest_page_size: int = 16,
    stream_token_ids: Optional[Sequence[int]] = None,
) -> Mapping[str, CoreSelection]:
    positions = backing.positions()
    _, _, eligible = mandatory_and_eligible(
        positions, int(sink_size), int(recent_size)
    )
    layer_ids = tuple(range(len(state.cache)))
    score_bundle: Dict[str, Dict[int, np.ndarray]] = {
        name: {} for name in candidate_names
    }
    core_bundle: Dict[str, Dict[int, Tuple[int, ...]]] = {
        name: {} for name in candidate_names
    }
    for layer in layer_ids:
        attention = memory.score(
            layer,
            positions,
            "attention",
            pooling_kernel,
            pooling_method,
        )
        source_scores = {
            "attention": attention,
            "snapkv": memory.score(
                layer,
                positions,
                "snapkv",
                pooling_kernel,
                pooling_method,
            ),
            "h2o": memory.score(
                layer,
                positions,
                "h2o",
                pooling_kernel,
                pooling_method,
            ),
        }
        keys, values = backing.layer_arrays(int(layer))
        source_scores["key_norm"] = (
            torch.linalg.vector_norm(keys.float(), dim=-1)
            .mean(dim=(0, 1))
            .numpy()
            .astype(np.float64)
        )
        source_scores["value_norm"] = (
            torch.linalg.vector_norm(values.float(), dim=-1)
            .mean(dim=(0, 1))
            .numpy()
            .astype(np.float64)
        )
        for name in candidate_names:
            if name == "stale":
                core = _stale_core(
                    None if previous_cores is None else previous_cores[int(layer)],
                    eligible,
                    attention,
                    positions,
                    core_budget,
                )
                score = attention
            elif name == "uniform":
                core = deterministic_uniform_core(eligible, core_budget)
                score = np.zeros(len(positions), dtype=np.float64)
                row_by_position = {
                    int(position): row
                    for row, position in enumerate(positions)
                }
                for position in core:
                    score[row_by_position[int(position)]] = 1.0
            elif name == "recency":
                core = recency_core(eligible, core_budget)
                score = np.zeros(len(positions), dtype=np.float64)
                row_by_position = {
                    int(position): row
                    for row, position in enumerate(positions)
                }
                for position in core:
                    score[row_by_position[int(position)]] = 1.0
            elif name in ("qk_pool", "quest_like", "qk_obswin"):
                if pool_scores is None:
                    raise ValueError(
                        "%s candidate requires full-pool scores" % name
                    )
                score = _score_on_universe(
                    pool_scores.get(int(layer), {}), positions
                )
                if name in ("qk_pool", "qk_obswin"):
                    core = _top_core(
                        positions, eligible, score, int(core_budget)
                    )
                else:
                    score_by_position = {
                        int(position): float(score[row])
                        for row, position in enumerate(positions)
                    }
                    core = quest_like_core(
                        eligible,
                        score_by_position,
                        int(quest_page_size),
                        int(core_budget),
                    )
            elif name == "token_rarity":
                if stream_token_ids is None:
                    raise ValueError(
                        "token_rarity candidate requires the observed stream "
                        "token ids"
                    )
                score = token_rarity_scores(stream_token_ids, positions)
                core = _top_core(
                    positions, eligible, score, int(core_budget)
                )
            elif name in source_scores:
                score = source_scores[name]
                core = _top_core(
                    positions, eligible, score, int(core_budget)
                )
            else:
                raise ValueError("unknown comparison candidate=%s" % name)
            if len(core) != min(int(core_budget), len(eligible)):
                raise RuntimeError("comparison candidate does not fill budget")
            score_bundle[name][int(layer)] = np.asarray(
                score, dtype=np.float64
            )
            core_bundle[name][int(layer)] = tuple(core)
    return {
        name: _selection_from_scores(
            name,
            positions,
            eligible,
            core_bundle[name],
            score_bundle[name],
        )
        for name in candidate_names
    }


def _core_map(selection: CoreSelection) -> Dict[int, Tuple[int, ...]]:
    return {
        int(layer): tuple(int(value) for value in current.selected_positions)
        for layer, current in selection.by_layer.items()
    }


def _minimum_fraction(
    cores: Mapping[int, Sequence[int]],
    positions: Mapping[int, Sequence[int]],
) -> float:
    values = []
    for layer, core in cores.items():
        current = set(int(value) for value in positions[int(layer)])
        expected = set(int(value) for value in core)
        values.append(len(expected & current) / max(1, len(expected)))
    return float(min(values)) if values else 1.0


def _run_policy(
    runner: CandidatePullbackRunner,
    reference: Any,
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
    memory = AttentionPolicyMemory.initialize(
        reference,
        int(start_anchor),
        range(len(base_state.cache)),
        int(observation_window),
        backing.positions(),
    )
    current_token = int(anchor_state.query_token_id)
    previous_cores: Optional[Dict[int, Tuple[int, ...]]] = None
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
        input_maps = {
            int(layer): tuple(int(value) for value in positions.tolist())
            for layer, positions in base_state.position_maps.items()
        }
        backing.update(runner, base_state)
        panel = _physical_candidate_panel(
            runner,
            base_state,
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
        exact_scores = {
            name: float(
                np.mean([float(row["exact_kl"]) for row in outcome.step_rows])
            )
            for name, outcome in outcomes.items()
        }
        teacher_decision = select_lowest_risk(exact_scores)
        teacher_best = str(teacher_decision.candidate_id)
        selected_name = teacher_best if policy == "statekv_exact_mean" else policy
        if selected_name not in outcomes:
            raise ValueError("fixed policy is absent from candidate panel")
        selected = outcomes[selected_name]
        selected_selection = panel[selected_name]
        selected_cores = _core_map(selected_selection)
        stale_cores = _core_map(panel["stale"])
        refresh = selected_cores != stale_cores
        recovered = sum(
            len(set(core) - set(input_maps[int(layer)]))
            for layer, core in selected_cores.items()
        )
        for name, outcome in outcomes.items():
            for offset, row in enumerate(outcome.step_rows):
                candidate_rows.append(
                    {
                        "policy": str(policy),
                        "cycle": int(cycle),
                        "candidate": str(name),
                        "selected": name == selected_name,
                        "teacher_best": name == teacher_best,
                        "candidate_exact_kl_mean": exact_scores[name],
                        **row,
                    }
                )
        output_position = int(selected.state.logical_next_position)
        output_maps = {
            int(layer): tuple(int(value) for value in positions.tolist())
            for layer, positions in selected.state.position_maps.items()
        }
        cycle_rows.append(
            {
                "policy": str(policy),
                "cycle": int(cycle),
                "target_start": target_start,
                "input_logical_position": input_position,
                "output_logical_position": output_position,
                "state_advanced_by": output_position - input_position,
                "state_continuity": bool(
                    previous_output_position is None
                    or input_position == previous_output_position
                ),
                "selected_candidate": selected_name,
                "teacher_best_candidate": teacher_best,
                "teacher_top1_agreement": selected_name == teacher_best,
                "selected_exact_kl_mean": exact_scores[selected_name],
                "teacher_best_exact_kl_mean": exact_scores[teacher_best],
                "stale_exact_kl_mean": exact_scores["stale"],
                "policy_regret_to_teacher": max(
                    0.0, exact_scores[selected_name] - exact_scores[teacher_best]
                ),
                "refresh": bool(refresh),
                "selected_recovered_layer_tokens": int(recovered),
                "previous_core_carry_fraction": (
                    1.0
                    if previous_cores is None
                    else _minimum_fraction(previous_cores, input_maps)
                ),
                "selected_core_survival_fraction": _minimum_fraction(
                    selected_cores, output_maps
                ),
                "maximum_active_cache_tokens": int(
                    max(
                        outcome.maximum_active_tokens
                        for outcome in outcomes.values()
                    )
                ),
                "budget_respected": bool(
                    all(
                        outcome.maximum_active_tokens <= int(total_budget)
                        for outcome in outcomes.values()
                    )
                ),
            }
        )
        memory.update_rollout(selected)
        base_state = selected.state
        current_token = int(selected.next_token)
        previous_cores = selected_cores
        previous_output_position = output_position
        backing.update(runner, base_state)
        for name, outcome in outcomes.items():
            if name != selected_name:
                outcome.state = None
        runner.model.release()
    runner.model.release(base_state)
    summary = {
        "policy": str(policy),
        "cycles_completed": len(cycle_rows),
        "mean_exact_kl": float(
            np.mean([row["selected_exact_kl_mean"] for row in cycle_rows])
        ),
        "mean_policy_regret_to_teacher": float(
            np.mean([row["policy_regret_to_teacher"] for row in cycle_rows])
        ),
        "teacher_top1_agreement": float(
            np.mean([row["teacher_top1_agreement"] for row in cycle_rows])
        ),
        "refresh_events": int(sum(row["refresh"] for row in cycle_rows)),
        "recovery_events": int(
            sum(row["selected_recovered_layer_tokens"] > 0 for row in cycle_rows)
        ),
        "closed_loop_passed": bool(
            len(cycle_rows) == int(cycles)
            and all(row["state_continuity"] for row in cycle_rows)
            and all(row["state_advanced_by"] == int(horizon) for row in cycle_rows)
            and all(row["budget_respected"] for row in cycle_rows)
            and min(row["previous_core_carry_fraction"] for row in cycle_rows)
            >= 1.0
            and min(row["selected_core_survival_fraction"] for row in cycle_rows)
            >= 1.0
        ),
    }
    return cycle_rows, candidate_rows, summary


def _comparison_summary(
    cycles: pd.DataFrame, policy_summaries: pd.DataFrame
) -> Dict[str, Any]:
    aggregates = []
    for policy, current in cycles.groupby("policy", sort=True):
        sample_means = current.groupby("sample_id")[
            "selected_exact_kl_mean"
        ].mean()
        aggregates.append(
            {
                "policy": str(policy),
                "sample_loops": int(current["sample_id"].nunique()),
                "control_cycles": int(len(current)),
                "mean_exact_kl": float(current["selected_exact_kl_mean"].mean()),
                "median_exact_kl": float(
                    current["selected_exact_kl_mean"].median()
                ),
                "mean_policy_regret_to_teacher": float(
                    current["policy_regret_to_teacher"].mean()
                ),
                "sample_mean_exact_kl": {
                    str(key): float(value) for key, value in sample_means.items()
                },
            }
        )
    by_policy = {row["policy"]: row for row in aggregates}
    statekv = by_policy["statekv_exact_mean"]
    comparisons = []
    for baseline in ("attention", "snapkv", "h2o"):
        current = by_policy[baseline]
        shared = sorted(
            set(statekv["sample_mean_exact_kl"])
            & set(current["sample_mean_exact_kl"])
        )
        deltas = [
            current["sample_mean_exact_kl"][sample]
            - statekv["sample_mean_exact_kl"][sample]
            for sample in shared
        ]
        comparisons.append(
            {
                "baseline": baseline,
                "statekv_mean_exact_kl": statekv["mean_exact_kl"],
                "baseline_mean_exact_kl": current["mean_exact_kl"],
                "baseline_minus_statekv": (
                    current["mean_exact_kl"] - statekv["mean_exact_kl"]
                ),
                "statekv_sample_wins": int(sum(value > 0.0 for value in deltas)),
                "ties": int(sum(abs(value) <= 1.0e-12 for value in deltas)),
                "sample_count": len(shared),
            }
        )
    return {
        "policy_aggregates": aggregates,
        "paired_comparisons": comparisons,
        "all_physical_loops_passed": bool(
            policy_summaries["closed_loop_passed"].all()
        ),
        "statekv_lower_overall_mean_than_each_fixed_policy": bool(
            all(row["baseline_minus_statekv"] > 0.0 for row in comparisons)
        ),
    }


def run_oracle_policy_comparison(
    config_path: Path, repository_root: Path
) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cfg = load_discovery_config(str(repository_root / str(config["base_config"])))
    cfg.tasks = dict(config["task_overrides"])
    cfg.runtime.seed = int(config["data_seed"])
    cfg.runtime.run_id = str(config["runtime_run_id"])
    sample_ids = set(str(value) for value in config["sample_ids"])
    output_root = repository_root / str(config["output_run"])
    output_root.mkdir(parents=True, exist_ok=True)
    policies = [str(value) for value in config["policies"]]
    candidates = [str(value) for value in config["candidate_panel"]]
    required = {"stale", "attention", "snapkv", "h2o"}
    if not required <= set(candidates):
        raise ValueError("candidate panel is missing a required fixed policy")
    if policies != ["statekv_exact_mean", "attention", "snapkv", "h2o"]:
        raise ValueError("comparison policies must retain the frozen order")
    start_anchor = int(config["start_anchor"])
    cycles = int(config["control_cycles"])
    horizon = int(config["control_horizon"])
    total_budget = int(config["total_budget"])
    sink_size = int(config["sink_size"])
    recent_size = int(config["recent_size"])
    core_budget = int(config["core_budget"])
    if core_budget != total_budget - sink_size - recent_size:
        raise ValueError("core budget must fill total minus sink and recent")
    runner = CandidatePullbackRunner(cfg, repository_root)
    samples, _ = load_discovery_tasks(cfg)
    selected_samples = [
        sample for sample in samples if str(sample.sample_id) in sample_ids
    ]
    if {str(sample.sample_id) for sample in selected_samples} != sample_ids:
        raise RuntimeError("configured comparison samples were not loaded")
    cycle_rows: List[Dict[str, Any]] = []
    candidate_rows: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    started = time.perf_counter()
    runner.model.load()
    try:
        for sample in selected_samples:
            targets = range(start_anchor, start_anchor + cycles * horizon)
            reference = runner.model.generate_reference(
                sample.sample_id,
                sample.task,
                sample.prompt,
                extra_probe_target_indices=list(targets),
            )
            try:
                for policy in policies:
                    rows, candidates_rows, summary = _run_policy(
                        runner,
                        reference,
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
                    base = {
                        "sample_id": str(sample.sample_id),
                        "task": str(sample.task),
                    }
                    cycle_rows.extend({**base, **row} for row in rows)
                    candidate_rows.extend(
                        {**base, **row} for row in candidates_rows
                    )
                    summaries.append({**base, **summary})
            finally:
                runner.model.release(reference)
    finally:
        runner.model.close()
    cycle_frame = pd.DataFrame(cycle_rows)
    summary_frame = pd.DataFrame(summaries)
    comparison = _comparison_summary(cycle_frame, summary_frame)
    result = {
        "experiment": str(config["experiment_name"]),
        "status": "teacher_forced_matched_physical_closed_loop_comparison",
        "samples": sorted(sample_ids),
        "policies": policies,
        "candidate_panel": candidates,
        "control_cycles": cycles,
        "control_horizon": horizon,
        "total_budget": total_budget,
        "policy_sample_summaries": summaries,
        **comparison,
        "passed": bool(
            comparison["all_physical_loops_passed"]
            and comparison["statekv_lower_overall_mean_than_each_fixed_policy"]
        ),
        "collection_elapsed_s": float(time.perf_counter() - started),
        "scope": (
            "Matched teacher-forced generation with independent compressed "
            "states and a common full-cache token trajectory. Cost is ignored; "
            "free-generation task quality is evaluated separately."
        ),
    }
    atomic_frame(cycle_frame, output_root / "cycle_rows.parquet")
    atomic_frame(
        pd.DataFrame(candidate_rows), output_root / "candidate_step_rows.parquet"
    )
    atomic_frame(summary_frame, output_root / "policy_sample_summary.csv")
    atomic_json(output_root / "summary.json", result)
    atomic_text(
        output_root / "config.yaml",
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
    )
    return output_root


__all__ = [
    "AttentionPolicyMemory",
    "run_oracle_policy_comparison",
]
