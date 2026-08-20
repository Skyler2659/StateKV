"""Layer-budget controls used by the StateKV mechanism and eviction gates.

The five mechanism variants in this module share the same direct token utility.
Only the mapping from state to per-layer core budgets changes.  The same
controller can operate on a full cold-token view or on the currently active KV
rows, which lets the pure-eviction runner enforce irreversible deletion.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from statekv.direct_policy_runtime import contribution_token_score
from statekv.oracle_closed_loop import _top_core, deterministic_uniform_core
from statekv.oracle_policy_comparison import (
    AttentionPolicyMemory,
    _selection_from_scores,
)
from statekv.selectors import CoreSelection, LayerSelection, mandatory_and_eligible


MECHANISM_POLICIES = (
    "b2_uniform",
    "static_adaptive",
    "dynamic_b3",
    "layer_shuffled_b3",
    "stale_b3",
)

PURE_EVICTION_POLICIES = (
    "dynamic_b3",
    "b2_uniform",
    "a2_temporal_volatility",
    "attention",
    "snapkv",
    "uniform",
    "shared_attention",
)

TEMPORAL_POLICIES = (
    "fixed_ema",
    "adaptive_dual",
)


@dataclass
class OnlineTemporalScoreMemory:
    """Causal per-layer/token score state for live pure-eviction runs."""

    fixed_rho: float
    fast_rho: float
    slow_rho: float
    variance_rho: float
    threshold: float
    smooth_alpha: float
    epsilon: float

    def __post_init__(self) -> None:
        self.last_cycle = -1
        self.fixed: Dict[int, Dict[int, float]] = {}
        self.fast: Dict[int, Dict[int, float]] = {}
        self.slow: Dict[int, Dict[int, float]] = {}
        self.variance: Dict[int, Dict[int, float]] = {}
        self.dual: Dict[int, Dict[int, float]] = {}
        self.rho: Dict[int, Dict[int, float]] = {}

    def update(
        self,
        memory: AttentionPolicyMemory,
        view: "LayerCacheView",
        cycle: int,
    ) -> None:
        if int(cycle) == int(self.last_cycle):
            return
        if int(cycle) < int(self.last_cycle):
            raise RuntimeError("temporal score memory received a decreasing cycle")
        for layer in view.layers:
            fixed = self.fixed.setdefault(int(layer), {})
            fast = self.fast.setdefault(int(layer), {})
            slow = self.slow.setdefault(int(layer), {})
            variance = self.variance.setdefault(int(layer), {})
            dual = self.dual.setdefault(int(layer), {})
            rho = self.rho.setdefault(int(layer), {})
            latest = memory.latest.get(int(layer), {})
            for position in view.positions_by_layer[int(layer)]:
                token = int(position)
                if token not in latest:
                    continue
                observation = max(0.0, float(latest[token]))
                if token not in fixed:
                    fixed[token] = observation
                    fast[token] = observation
                    slow[token] = observation
                    variance[token] = 0.0
                    dual[token] = observation
                    rho[token] = 1.0
                    continue
                previous_slow = slow[token]
                residual = observation - previous_slow
                variance[token] = (
                    float(self.variance_rho) * variance[token]
                    + (1.0 - float(self.variance_rho)) * residual * residual
                )
                fast[token] = (
                    float(self.fast_rho) * fast[token]
                    + (1.0 - float(self.fast_rho)) * observation
                )
                slow[token] = (
                    float(self.slow_rho) * slow[token]
                    + (1.0 - float(self.slow_rho)) * observation
                )
                drift = abs(fast[token] - slow[token]) / (
                    math.sqrt(max(variance[token], 0.0)) + float(self.epsilon)
                )
                stable_gate = 1.0 / (
                    1.0
                    + math.exp(
                        max(
                            -60.0,
                            min(
                                60.0,
                                float(self.smooth_alpha)
                                * (drift - float(self.threshold)),
                            ),
                        )
                    )
                )
                dual[token] = stable_gate * slow[token] + (1.0 - stable_gate) * fast[token]
                rho[token] = stable_gate
                fixed[token] = (
                    float(self.fixed_rho) * fixed[token]
                    + (1.0 - float(self.fixed_rho)) * observation
                )
        self.last_cycle = int(cycle)

    @staticmethod
    def _aligned(
        state: Mapping[int, float], positions: Sequence[int]
    ) -> np.ndarray:
        return np.asarray(
            [float(state.get(int(position), 0.0)) for position in positions],
            dtype=np.float64,
        )

    def score(self, policy: str, layer: int, positions: Sequence[int]) -> np.ndarray:
        if policy == "fixed_ema":
            return self._aligned(self.fixed.get(int(layer), {}), positions)
        if policy == "adaptive_dual":
            return self._aligned(self.dual.get(int(layer), {}), positions)
        raise ValueError(f"unknown temporal policy={policy}")

    def gate_values(self, layer: int, positions: Sequence[int]) -> np.ndarray:
        return self._aligned(self.rho.get(int(layer), {}), positions)


def _normalize(values: np.ndarray, rows: np.ndarray) -> np.ndarray:
    raw = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    result = np.zeros_like(raw)
    mass = float(raw[rows].sum()) if rows.size else 0.0
    if mass > 0.0:
        result[rows] = raw[rows] / mass
    return result


def _effective_support(probability: np.ndarray) -> float:
    values = np.maximum(np.asarray(probability, dtype=np.float64), 0.0)
    mass = float(values.sum())
    if mass <= 0.0:
        return 1.0
    values = values / mass
    entropy = -float(np.sum(values * np.log(np.maximum(values, 1.0e-12))))
    return float(np.exp(entropy))


def allocate_layer_budgets(
    difficulty_by_layer: Mapping[int, float],
    core_budget: int,
    maximum_delta: int,
) -> Dict[int, int]:
    """Project positive layer difficulty to an integer fixed-global budget."""

    layers = tuple(sorted(int(layer) for layer in difficulty_by_layer))
    if not layers:
        raise ValueError("layer budget allocation requires at least one layer")
    difficulty = np.asarray(
        [max(float(difficulty_by_layer[layer]), 0.0) for layer in layers],
        dtype=np.float64,
    )
    if float(difficulty.sum()) <= 0.0:
        difficulty = np.ones(len(layers), dtype=np.float64)
    target = len(layers) * int(core_budget)
    raw = target * difficulty / float(difficulty.sum())
    minimum = max(0, int(core_budget) - int(maximum_delta))
    maximum = int(core_budget) + int(maximum_delta)
    budget = np.clip(np.floor(raw).astype(np.int64), minimum, maximum)
    while int(budget.sum()) < target:
        choices = [index for index in range(len(layers)) if budget[index] < maximum]
        if not choices:
            raise RuntimeError("layer budgets cannot fill their global target")
        index = max(choices, key=lambda item: (raw[item] - budget[item], -item))
        budget[index] += 1
    while int(budget.sum()) > target:
        choices = [index for index in range(len(layers)) if budget[index] > minimum]
        if not choices:
            raise RuntimeError("layer budgets cannot meet their global target")
        index = max(choices, key=lambda item: (budget[item] - raw[item], item))
        budget[index] -= 1
    return {
        int(layer): int(value) for layer, value in zip(layers, budget.tolist())
    }


def average_static_budgets(
    dynamic_vectors: Sequence[Mapping[int, int]],
    core_budget: int,
    maximum_delta: int,
) -> Dict[int, int]:
    """Create a fixed layer prior from disjoint calibration trajectories."""

    if not dynamic_vectors:
        raise ValueError("static calibration requires dynamic budget vectors")
    layers = tuple(sorted(int(layer) for layer in dynamic_vectors[0]))
    if any(tuple(sorted(int(layer) for layer in row)) != layers for row in dynamic_vectors):
        raise ValueError("calibration budget vectors have inconsistent layers")
    mean = {
        layer: float(np.mean([int(row[layer]) for row in dynamic_vectors]))
        for layer in layers
    }
    return allocate_layer_budgets(mean, core_budget, maximum_delta)


@dataclass(frozen=True)
class LayerCacheView:
    positions_by_layer: Mapping[int, Tuple[int, ...]]
    values_by_layer: Mapping[int, torch.Tensor]

    @property
    def layers(self) -> Tuple[int, ...]:
        return tuple(sorted(int(layer) for layer in self.positions_by_layer))


def backing_cache_view(backing: Any, layers: Sequence[int]) -> LayerCacheView:
    positions = tuple(int(value) for value in backing.positions())
    return LayerCacheView(
        positions_by_layer={int(layer): positions for layer in layers},
        values_by_layer={
            int(layer): backing.layer_arrays(int(layer))[1] for layer in layers
        },
    )


def active_cache_view(runner: Any, state: Any) -> LayerCacheView:
    positions: Dict[int, Tuple[int, ...]] = {}
    values: Dict[int, torch.Tensor] = {}
    for layer, layer_cache in enumerate(state.cache):
        offset = int(layer_cache.offset)
        positions[int(layer)] = tuple(
            int(value) for value in state.position_maps[int(layer)].tolist()
        )
        values[int(layer)] = runner.model._torch(
            layer_cache.values[:, :, :offset, :], torch.float16
        )
        if int(values[int(layer)].shape[2]) != len(positions[int(layer)]):
            raise RuntimeError("active KV values and position map diverged")
    return LayerCacheView(positions, values)


@dataclass(frozen=True)
class BudgetDecision:
    selection: CoreSelection
    requested_budgets: Mapping[int, int]
    realized_budgets: Mapping[int, int]
    dynamic_budgets: Mapping[int, int]
    difficulty_by_layer: Mapping[int, float]
    scores_by_layer: Mapping[int, np.ndarray]
    eligible_by_layer: Mapping[int, Tuple[int, ...]]
    diagnostics: Mapping[str, Any]


@dataclass(frozen=True)
class FrozenRanking:
    """Per-layer score vectors frozen at a refresh step.

    ``scores_by_layer[layer]`` is aligned with ``positions_by_layer[layer]`` at
    freeze time.  A stale selection re-ranks the currently active positions
    with these frozen scores; positions created after the freeze carry no
    score and can only survive through the mandatory sink/recent window.
    """

    policy: str
    cycle: int
    positions_by_layer: Mapping[int, Tuple[int, ...]]
    scores_by_layer: Mapping[int, np.ndarray]
    requested_budgets: Mapping[int, int]
    dynamic_budgets: Mapping[int, int]
    difficulty_by_layer: Mapping[int, float]


@dataclass
class DirectBudgetController:
    core_budget: int
    sink_size: int
    recent_size: int
    pooling_kernel: int
    pooling_method: str
    maximum_delta: int
    static_budgets: Optional[Mapping[int, int]] = None
    shuffle_seed: int = 0
    stale_lag: int = 4
    temporal_fixed_rho: float = 0.5
    temporal_fast_rho: float = 0.5
    temporal_slow_rho: float = 0.95
    temporal_variance_rho: float = 0.5
    temporal_threshold: float = 0.25
    temporal_smooth_alpha: float = 4.0
    temporal_epsilon: float = 1.0e-8

    def __post_init__(self) -> None:
        self._dynamic_history: list[Dict[int, int]] = []
        self._frozen: Optional[FrozenRanking] = None
        self._temporal = OnlineTemporalScoreMemory(
            fixed_rho=float(self.temporal_fixed_rho),
            fast_rho=float(self.temporal_fast_rho),
            slow_rho=float(self.temporal_slow_rho),
            variance_rho=float(self.temporal_variance_rho),
            threshold=float(self.temporal_threshold),
            smooth_alpha=float(self.temporal_smooth_alpha),
            epsilon=float(self.temporal_epsilon),
        )

    @staticmethod
    def requires_candidate_panel(_policy: str) -> bool:
        return False

    def _signals(
        self,
        memory: AttentionPolicyMemory,
        view: LayerCacheView,
        cycle: int,
    ) -> Tuple[
        Dict[int, Tuple[int, ...]],
        Dict[int, np.ndarray],
        Dict[int, Dict[str, np.ndarray]],
    ]:
        self._temporal.update(memory, view, cycle)
        eligible: Dict[int, Tuple[int, ...]] = {}
        eligible_rows: Dict[int, np.ndarray] = {}
        signals: Dict[int, Dict[str, np.ndarray]] = {}
        for layer in view.layers:
            positions = view.positions_by_layer[layer]
            _, _, current_eligible = mandatory_and_eligible(
                positions, int(self.sink_size), int(self.recent_size)
            )
            eligible[layer] = tuple(int(value) for value in current_eligible)
            row_by_position = {
                int(position): row for row, position in enumerate(positions)
            }
            rows = np.asarray(
                [row_by_position[int(position)] for position in current_eligible],
                dtype=np.int64,
            )
            eligible_rows[layer] = rows
            latest = memory.score(
                layer,
                positions,
                "attention",
                self.pooling_kernel,
                self.pooling_method,
            )
            volatility = memory.volatility_score(layer, positions)
            snapkv = memory.score(
                layer,
                positions,
                "snapkv",
                self.pooling_kernel,
                self.pooling_method,
            )
            signals[layer] = {
                "attention": _normalize(latest, rows),
                "volatility": _normalize(volatility, rows),
                "volatility_raw": np.asarray(volatility, dtype=np.float64),
                "snapkv": _normalize(snapkv, rows),
                "fixed_ema": _normalize(
                    self._temporal.score("fixed_ema", layer, positions), rows
                ),
                "adaptive_dual": _normalize(
                    self._temporal.score("adaptive_dual", layer, positions), rows
                ),
                "adaptive_gate": self._temporal.gate_values(layer, positions),
            }
        return eligible, eligible_rows, signals

    def _direct_scores(
        self,
        memory: AttentionPolicyMemory,
        view: LayerCacheView,
        eligible_rows: Mapping[int, np.ndarray],
        signals: Mapping[int, Mapping[str, np.ndarray]],
    ) -> Dict[int, np.ndarray]:
        result: Dict[int, np.ndarray] = {}
        for layer in view.layers:
            positions = view.positions_by_layer[layer]
            contribution = np.zeros(len(positions), dtype=np.float64)
            head_attention = memory.head_score(layer, positions)
            if head_attention.size:
                values = view.values_by_layer[layer].detach().float().cpu().numpy()[0]
                contribution = contribution_token_score(
                    head_attention, values, dtype=np.dtype(np.float32)
                ).astype(np.float64)
            contribution = _normalize(contribution, eligible_rows[layer])
            result[layer] = (
                0.50 * np.asarray(signals[layer]["attention"], dtype=np.float64)
                + 0.30 * np.asarray(signals[layer]["volatility"], dtype=np.float64)
                + 0.20 * contribution
            )
        return result

    def _shuffled(
        self,
        budgets: Mapping[int, int],
        sample_id: str,
        cycle: int,
    ) -> Dict[int, int]:
        layers = tuple(sorted(int(layer) for layer in budgets))
        digest = hashlib.sha256(
            (f"{self.shuffle_seed}|{sample_id}|{cycle}").encode("utf-8")
        ).digest()
        seed = int.from_bytes(digest[:8], byteorder="little", signed=False)
        permutation = np.random.default_rng(seed).permutation(len(layers))
        values = [int(budgets[layer]) for layer in layers]
        return {
            layer: values[int(permutation[index])]
            for index, layer in enumerate(layers)
        }

    def select(
        self,
        policy: str,
        memory: AttentionPolicyMemory,
        view: LayerCacheView,
        cycle: int,
        sample_id: str,
    ) -> BudgetDecision:
        if policy not in (
            set(MECHANISM_POLICIES)
            | set(PURE_EVICTION_POLICIES)
            | set(TEMPORAL_POLICIES)
        ):
            raise ValueError(f"unknown direct budget policy={policy}")
        eligible, eligible_rows, signals = self._signals(memory, view, cycle)
        if policy == "a2_temporal_volatility":
            scores = {
                layer: np.asarray(signals[layer]["volatility"], dtype=np.float64)
                for layer in view.layers
            }
        elif policy == "attention":
            scores = {
                layer: np.asarray(signals[layer]["attention"], dtype=np.float64)
                for layer in view.layers
            }
        elif policy == "snapkv":
            scores = {
                layer: np.asarray(signals[layer]["snapkv"], dtype=np.float64)
                for layer in view.layers
            }
        elif policy in TEMPORAL_POLICIES:
            scores = {
                layer: np.asarray(signals[layer][policy], dtype=np.float64)
                for layer in view.layers
            }
        elif policy == "shared_attention":
            # P23b-substrate shared-mask attention: one score per position =
            # the mean of the per-layer normalized attention scores, and one
            # core applied to every layer (shared token selection).
            shared = np.zeros(
                len(view.positions_by_layer[next(iter(view.layers))]),
                dtype=np.float64,
            )
            for layer in view.layers:
                shared = shared + np.asarray(
                    signals[layer]["attention"], dtype=np.float64
                )
            shared = shared / float(len(view.layers))
            scores = {layer: shared.copy() for layer in view.layers}
        else:
            scores = self._direct_scores(memory, view, eligible_rows, signals)
        difficulty = {
            layer: _effective_support(scores[layer][eligible_rows[layer]])
            for layer in view.layers
        }
        dynamic = allocate_layer_budgets(
            difficulty, self.core_budget, self.maximum_delta
        )
        budget_age = 0
        if policy == "static_adaptive":
            if self.static_budgets is None:
                raise RuntimeError("static adaptive policy lacks calibration budgets")
            requested = {
                layer: int(self.static_budgets[layer]) for layer in view.layers
            }
        elif policy == "layer_shuffled_b3":
            requested = self._shuffled(dynamic, sample_id, cycle)
        elif policy == "stale_b3":
            self._dynamic_history.append(dict(dynamic))
            source = max(0, len(self._dynamic_history) - 1 - int(self.stale_lag))
            requested = dict(self._dynamic_history[source])
            budget_age = int(cycle - source)
        else:
            requested = (
                dict(dynamic)
                if policy == "dynamic_b3"
                else {layer: int(self.core_budget) for layer in view.layers}
            )
        if sum(requested.values()) != len(view.layers) * int(self.core_budget):
            raise RuntimeError("requested layer budgets changed the global core total")
        cores: Dict[int, Tuple[int, ...]] = {}
        realized: Dict[int, int] = {}
        if policy == "uniform":
            cores = {
                layer: tuple(
                    deterministic_uniform_core(
                        eligible[layer], int(requested[layer])
                    )
                )
                for layer in view.layers
            }
            realized = {layer: len(cores[layer]) for layer in view.layers}
            scores = {
                layer: np.zeros(
                    len(view.positions_by_layer[layer]), dtype=np.float64
                )
                for layer in view.layers
            }
        else:
            for layer in view.layers:
                core = _top_core(
                    view.positions_by_layer[layer],
                    eligible[layer],
                    scores[layer],
                    requested[layer],
                )
                cores[layer] = tuple(int(value) for value in core)
                realized[layer] = len(core)
        selection = CoreSelection(
            strategy=str(policy),
            horizon_condition=None,
            by_layer={
                layer: LayerSelection(
                    layer=layer,
                    selected_positions=list(cores[layer]),
                    eligible_positions=list(eligible[layer]),
                    aggregate_scores=[float(value) for value in scores[layer]],
                    metadata={
                        "source": str(policy),
                        "requested_core_budget": int(requested[layer]),
                        "realized_core_budget": int(realized[layer]),
                        "per_layer_shared_across_kv_heads": True,
                    },
                )
                for layer in view.layers
            },
            metadata={
                "direct_action": True,
                "fixed_global_requested_core_budget": True,
            },
        )
        diagnostics = {
            "budget_by_layer_json": json.dumps(
                [int(requested[layer]) for layer in view.layers]
            ),
            "dynamic_budget_by_layer_json": json.dumps(
                [int(dynamic[layer]) for layer in view.layers]
            ),
            "realized_core_by_layer_json": json.dumps(
                [int(realized[layer]) for layer in view.layers]
            ),
            "difficulty_by_layer_json": json.dumps(
                [float(difficulty[layer]) for layer in view.layers]
            ),
            "budget_age": int(budget_age),
            "requested_core_tokens_total": int(sum(requested.values())),
            "realized_core_tokens_total": int(sum(realized.values())),
            "core_budget_shortfall": int(sum(requested.values()) - sum(realized.values())),
            "minimum_layer_core_budget": int(min(requested.values())),
            "maximum_layer_core_budget": int(max(requested.values())),
            "mean_layer_core_budget": float(np.mean(list(requested.values()))),
            "mean_layer_volatility": float(
                np.mean(
                    [
                        float(
                            np.mean(
                                signals[layer]["volatility_raw"][eligible_rows[layer]]
                            )
                        )
                        if eligible_rows[layer].size
                        else 0.0
                        for layer in view.layers
                    ]
                )
            ),
            "maximum_layer_volatility": float(
                max(
                    (
                        float(
                            np.max(
                                signals[layer]["volatility_raw"][eligible_rows[layer]]
                            )
                        )
                        if eligible_rows[layer].size
                        else 0.0
                    )
                    for layer in view.layers
                )
            ),
            "mean_layer_effective_support": float(
                np.mean(list(difficulty.values()))
            ),
            "mean_temporal_stable_gate": float(
                np.mean(
                    [
                        np.mean(signals[layer]["adaptive_gate"][eligible_rows[layer]])
                        if eligible_rows[layer].size
                        else 0.0
                        for layer in view.layers
                    ]
                )
            ),
        }
        return BudgetDecision(
            selection=selection,
            requested_budgets=requested,
            realized_budgets=realized,
            dynamic_budgets=dynamic,
            difficulty_by_layer=difficulty,
            scores_by_layer=scores,
            eligible_by_layer=eligible,
            diagnostics=diagnostics,
        )

    @property
    def frozen(self) -> Optional[FrozenRanking]:
        return self._frozen

    def refresh_scores(
        self,
        policy: str,
        memory: AttentionPolicyMemory,
        view: LayerCacheView,
        cycle: int,
        sample_id: str,
    ) -> BudgetDecision:
        """Recompute the ranking and freeze it for later stale selections."""
        decision = self.select(policy, memory, view, cycle, sample_id)
        self.freeze(decision, view, cycle)
        return decision

    def freeze(
        self,
        decision: BudgetDecision,
        view: LayerCacheView,
        cycle: int,
    ) -> None:
        """Freeze an already-computed decision as the current ranking."""
        self._frozen = FrozenRanking(
            policy=str(decision.selection.strategy),
            cycle=int(cycle),
            positions_by_layer={
                int(layer): tuple(
                    int(position) for position in view.positions_by_layer[layer]
                )
                for layer in view.layers
            },
            scores_by_layer={
                int(layer): np.asarray(
                    decision.scores_by_layer[layer], dtype=np.float64
                )
                for layer in view.layers
            },
            requested_budgets={
                int(layer): int(value)
                for layer, value in decision.requested_budgets.items()
            },
            dynamic_budgets={
                int(layer): int(value)
                for layer, value in decision.dynamic_budgets.items()
            },
            difficulty_by_layer={
                int(layer): float(value)
                for layer, value in decision.difficulty_by_layer.items()
            },
        )

    def stale_selection(
        self,
        view: LayerCacheView,
        cycle: int,
        sample_id: str,
        frozen: Optional[FrozenRanking] = None,
        memory: Optional[AttentionPolicyMemory] = None,
    ) -> BudgetDecision:
        """Emit the selection a frozen ranking implies on the active view.

        The per-layer core is the top ``core_budget`` currently active
        positions by the frozen scores.  Positions created after the freeze
        carry no score and are never cored; positions that slid out of the
        mandatory window keep their frozen scores and stay eligible.
        """
        ranking = frozen if frozen is not None else self._frozen
        if ranking is None:
            raise RuntimeError("stale selection requires a refreshed ranking")
        requested = {
            int(layer): int(ranking.requested_budgets[layer]) for layer in view.layers
        }
        if sum(requested.values()) != len(view.layers) * int(self.core_budget):
            raise RuntimeError("frozen layer budgets changed the global core total")
        cores: Dict[int, Tuple[int, ...]] = {}
        realized: Dict[int, int] = {}
        eligible: Dict[int, Tuple[int, ...]] = {}
        scores: Dict[int, np.ndarray] = {}
        for layer in view.layers:
            positions = view.positions_by_layer[layer]
            _, _, current_eligible = mandatory_and_eligible(
                positions, int(self.sink_size), int(self.recent_size)
            )
            eligible[layer] = tuple(int(value) for value in current_eligible)
            frozen_positions = ranking.positions_by_layer[layer]
            frozen_scores = np.asarray(
                ranking.scores_by_layer[layer], dtype=np.float64
            )
            frozen_row = {
                int(position): row for row, position in enumerate(frozen_positions)
            }
            aligned = np.zeros(len(positions), dtype=np.float64)
            for row, position in enumerate(positions):
                source = frozen_row.get(int(position))
                if source is not None:
                    aligned[row] = float(frozen_scores[source])
            covered = [
                int(position)
                for position in current_eligible
                if int(position) in frozen_row
            ]
            core = _top_core(positions, covered, aligned, requested[layer])
            cores[layer] = tuple(int(value) for value in core)
            realized[layer] = len(core)
            scores[layer] = aligned
        selection = CoreSelection(
            strategy=str(ranking.policy),
            horizon_condition=None,
            by_layer={
                layer: LayerSelection(
                    layer=layer,
                    selected_positions=list(cores[layer]),
                    eligible_positions=list(eligible[layer]),
                    aggregate_scores=[float(value) for value in scores[layer]],
                    metadata={
                        "source": f"stale:{ranking.policy}",
                        "ranking_refresh_cycle": int(ranking.cycle),
                        "requested_core_budget": int(requested[layer]),
                        "realized_core_budget": int(realized[layer]),
                        "per_layer_shared_across_kv_heads": True,
                    },
                )
                for layer in view.layers
            },
            metadata={
                "direct_action": True,
                "fixed_global_requested_core_budget": True,
                "stale_ranking": True,
            },
        )
        volatility_by_layer: Dict[int, np.ndarray] = {}
        if memory is not None:
            for layer in view.layers:
                positions = view.positions_by_layer[layer]
                row_by_position = {
                    int(position): row for row, position in enumerate(positions)
                }
                rows = np.asarray(
                    [row_by_position[int(position)] for position in eligible[layer]],
                    dtype=np.int64,
                )
                raw = np.asarray(
                    memory.volatility_score(layer, positions), dtype=np.float64
                )
                volatility_by_layer[layer] = (
                    raw[rows] if rows.size else np.asarray([], dtype=np.float64)
                )
        diagnostics = {
            "budget_by_layer_json": json.dumps(
                [int(requested[layer]) for layer in view.layers]
            ),
            "dynamic_budget_by_layer_json": json.dumps(
                [int(ranking.dynamic_budgets[layer]) for layer in view.layers]
            ),
            "realized_core_by_layer_json": json.dumps(
                [int(realized[layer]) for layer in view.layers]
            ),
            "difficulty_by_layer_json": json.dumps(
                [float(ranking.difficulty_by_layer[layer]) for layer in view.layers]
            ),
            "budget_age": int(cycle - ranking.cycle),
            "ranking_refresh_cycle": int(ranking.cycle),
            "requested_core_tokens_total": int(sum(requested.values())),
            "realized_core_tokens_total": int(sum(realized.values())),
            "core_budget_shortfall": int(sum(requested.values()) - sum(realized.values())),
            "minimum_layer_core_budget": int(min(requested.values())),
            "maximum_layer_core_budget": int(max(requested.values())),
            "mean_layer_core_budget": float(np.mean(list(requested.values()))),
            "mean_layer_volatility": (
                float(
                    np.mean(
                        [
                            float(np.mean(volatility_by_layer[layer]))
                            if volatility_by_layer[layer].size
                            else 0.0
                            for layer in view.layers
                        ]
                    )
                )
                if memory is not None
                else float("nan")
            ),
            "maximum_layer_volatility": (
                float(
                    max(
                        (
                            float(np.max(volatility_by_layer[layer]))
                            if volatility_by_layer[layer].size
                            else 0.0
                        )
                        for layer in view.layers
                    )
                )
                if memory is not None
                else float("nan")
            ),
            "mean_layer_effective_support": float(
                np.mean(list(ranking.difficulty_by_layer.values()))
            ),
        }
        return BudgetDecision(
            selection=selection,
            requested_budgets=requested,
            realized_budgets=realized,
            dynamic_budgets=dict(ranking.dynamic_budgets),
            difficulty_by_layer=dict(ranking.difficulty_by_layer),
            scores_by_layer=scores,
            eligible_by_layer=eligible,
            diagnostics=diagnostics,
        )


def core_churn_by_layer(
    previous: CoreSelection, current: CoreSelection
) -> Dict[int, float]:
    """One minus the Jaccard overlap of consecutive cores, per layer."""
    result: Dict[int, float] = {}
    for layer in sorted(set(previous.by_layer) & set(current.by_layer)):
        left = set(int(value) for value in previous.by_layer[layer].selected_positions)
        right = set(int(value) for value in current.by_layer[layer].selected_positions)
        union = left | right
        result[int(layer)] = 1.0 - len(left & right) / max(1, len(union))
    return result


def boundary_margin_by_layer(
    scores_by_layer: Mapping[int, np.ndarray],
    positions_by_layer: Mapping[int, Sequence[int]],
    eligible_by_layer: Mapping[int, Sequence[int]],
    core_budget: int,
) -> Dict[int, float]:
    """Gap between the core_budget-th and (core_budget+1)-th eligible score."""
    result: Dict[int, float] = {}
    for layer in sorted(scores_by_layer):
        scores = np.asarray(scores_by_layer[layer], dtype=np.float64)
        row_by_position = {
            int(position): row
            for row, position in enumerate(positions_by_layer[layer])
        }
        rows = np.asarray(
            [row_by_position[int(position)] for position in eligible_by_layer[layer]],
            dtype=np.int64,
        )
        ranked = np.sort(scores[rows])[::-1] if rows.size else np.asarray([])
        take = min(int(core_budget), int(ranked.size))
        result[int(layer)] = (
            float(ranked[take - 1] - ranked[take]) if 0 < take < ranked.size else 0.0
        )
    return result


def score_tv_by_layer(
    previous_scores: Mapping[int, np.ndarray],
    previous_positions: Mapping[int, Sequence[int]],
    current_scores: Mapping[int, np.ndarray],
    current_positions: Mapping[int, Sequence[int]],
) -> Dict[int, float]:
    """Half the L1 distance between normalized score vectors aligned by position."""
    result: Dict[int, float] = {}
    for layer in sorted(current_scores):
        previous_row = {
            int(position): row
            for row, position in enumerate(previous_positions.get(layer, ()))
        }
        current_row = {
            int(position): row
            for row, position in enumerate(current_positions[layer])
        }
        previous = np.asarray(
            previous_scores.get(layer, np.zeros(0)), dtype=np.float64
        )
        current = np.asarray(current_scores[layer], dtype=np.float64)
        total = 0.0
        for position in set(previous_row) | set(current_row):
            before = (
                float(previous[previous_row[position]])
                if position in previous_row
                else 0.0
            )
            after = (
                float(current[current_row[position]])
                if position in current_row
                else 0.0
            )
            total += abs(after - before)
        result[int(layer)] = 0.5 * total
    return result


def coverage_mass_by_layer(
    scores_by_layer: Mapping[int, np.ndarray],
    positions_by_layer: Mapping[int, Sequence[int]],
    selection: CoreSelection,
) -> Dict[int, float]:
    """Fraction of the eligible score mass retained by the applied core."""
    result: Dict[int, float] = {}
    for layer in sorted(scores_by_layer):
        scores = np.asarray(scores_by_layer[layer], dtype=np.float64)
        row_by_position = {
            int(position): row
            for row, position in enumerate(positions_by_layer[layer])
        }
        layer_selection = selection.by_layer[layer]
        eligible_rows = np.asarray(
            [
                row_by_position[int(position)]
                for position in layer_selection.eligible_positions
                if int(position) in row_by_position
            ],
            dtype=np.int64,
        )
        core_rows = np.asarray(
            [
                row_by_position[int(position)]
                for position in layer_selection.selected_positions
                if int(position) in row_by_position
            ],
            dtype=np.int64,
        )
        mass = float(scores[eligible_rows].sum()) if eligible_rows.size else 0.0
        kept = float(scores[core_rows].sum()) if core_rows.size else 0.0
        result[int(layer)] = kept / mass if mass > 0.0 else 1.0
    return result


def mask_overlap(
    left: CoreSelection, right: CoreSelection
) -> Dict[str, Any]:
    layers = tuple(sorted(set(left.by_layer) & set(right.by_layer)))
    jaccard = []
    containment = []
    per_layer = []
    for layer in layers:
        left_set = set(int(value) for value in left.by_layer[layer].selected_positions)
        right_set = set(int(value) for value in right.by_layer[layer].selected_positions)
        intersection = len(left_set & right_set)
        union = len(left_set | right_set)
        current_jaccard = intersection / max(1, union)
        current_containment = intersection / max(1, len(left_set))
        jaccard.append(current_jaccard)
        containment.append(current_containment)
        per_layer.append(current_jaccard)
    return {
        "mean_jaccard": float(np.mean(jaccard)) if jaccard else 1.0,
        "minimum_jaccard": float(np.min(jaccard)) if jaccard else 1.0,
        "mean_left_retained_by_right": (
            float(np.mean(containment)) if containment else 1.0
        ),
        "jaccard_by_layer_json": json.dumps(per_layer),
    }


__all__ = [
    "BudgetDecision",
    "DirectBudgetController",
    "FrozenRanking",
    "LayerCacheView",
    "MECHANISM_POLICIES",
    "PURE_EVICTION_POLICIES",
    "active_cache_view",
    "allocate_layer_budgets",
    "average_static_budgets",
    "backing_cache_view",
    "boundary_margin_by_layer",
    "core_churn_by_layer",
    "coverage_mass_by_layer",
    "mask_overlap",
    "score_tv_by_layer",
]
