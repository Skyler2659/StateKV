"""Training-free and tiny-ranker controllers for physical KV-cache actions.

The controllers in this module never run candidate model trajectories.  A1,
A3, and A4 rank the same legal action panel used by the exact-risk teacher;
A2, B2, and B3 emit a retained set directly.  B1 fits a small CPU ridge model
only from a declared historical oracle artifact.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from statekv.direct_policy_runtime import contribution_token_score
from statekv.oracle_closed_loop import (
    KVBackingStore,
    _normalize_on_eligible,
    _top_core,
)
from statekv.oracle_policy_comparison import (
    AttentionPolicyMemory,
    _core_map,
    _selection_from_scores,
)
from statekv.selectors import CoreSelection, mandatory_and_eligible


CHEAP_POLICIES = (
    "a1_candidate_proxy",
    "a2_temporal_volatility",
    "a3_set_output_perturbation",
    "a4_uncertainty_cascade",
    "b1_historical_tiny_ranker",
    "b2_direct_action_generator",
    "b3_layer_adaptive_budget",
)


def _eligible_rows(
    positions: Sequence[int], eligible: Sequence[int]
) -> np.ndarray:
    row_by_position = {
        int(position): index for index, position in enumerate(positions)
    }
    return np.asarray(
        [row_by_position[int(position)] for position in eligible],
        dtype=np.int64,
    )


def _direct_selection(
    name: str,
    positions: Sequence[int],
    eligible: Sequence[int],
    scores: Mapping[int, np.ndarray],
    budgets: Mapping[int, int],
) -> CoreSelection:
    cores = {
        int(layer): _top_core(
            positions,
            eligible,
            np.asarray(score, dtype=np.float64),
            int(budgets[int(layer)]),
        )
        for layer, score in scores.items()
    }
    result = _selection_from_scores(name, positions, eligible, cores, scores)
    result.metadata.update(
        {
            "direct_action": True,
            "core_budget_by_layer": {
                str(layer): int(budgets[int(layer)])
                for layer in sorted(budgets)
            },
        }
    )
    return result


@dataclass(frozen=True)
class HistoricalCandidateRanker:
    """Small ridge ranker fitted from a prior, disjoint oracle trace."""

    candidates: Tuple[str, ...]
    coefficients: np.ndarray
    feature_names: Tuple[str, ...]
    source_path: str
    training_rows: int

    @staticmethod
    def _bucket(task: str) -> float:
        return 0.0 if "gov" in str(task).lower() else 1.0

    @classmethod
    def _features(
        cls,
        candidate: str,
        cycle_fraction: float,
        task: str,
        candidates: Sequence[str],
    ) -> np.ndarray:
        niah = cls._bucket(task)
        values = [1.0, float(cycle_fraction), niah]
        for name in candidates:
            flag = float(str(candidate) == str(name))
            values.extend(
                [flag, flag * float(cycle_fraction), flag * niah]
            )
        return np.asarray(values, dtype=np.float64)

    @classmethod
    def fit(cls, path: Path, ridge: float = 1.0e-3) -> "HistoricalCandidateRanker":
        frame = pd.read_parquet(path)
        required = {"task", "cycle", "candidate", "exact_kl"}
        if not required.issubset(frame.columns):
            raise ValueError("historical ranker artifact lacks required columns")
        grouped = (
            frame.groupby(
                ["sample_id", "task", "policy", "cycle", "candidate"],
                as_index=False,
            )["exact_kl"]
            .mean()
            .replace([np.inf, -np.inf], np.nan)
            .dropna(subset=["exact_kl"])
        )
        candidates = tuple(sorted(str(value) for value in grouped["candidate"].unique()))
        if not candidates:
            raise ValueError("historical ranker artifact has no candidate rows")
        maximum_cycle = max(1, int(grouped["cycle"].max()))
        design = np.stack(
            [
                cls._features(
                    str(row.candidate),
                    float(row.cycle) / maximum_cycle,
                    str(row.task),
                    candidates,
                )
                for row in grouped.itertuples(index=False)
            ],
            axis=0,
        )
        target = np.log(np.maximum(grouped["exact_kl"].to_numpy(np.float64), 0.0) + 1.0e-8)
        penalty = np.eye(design.shape[1], dtype=np.float64) * float(ridge)
        penalty[0, 0] = 0.0
        coefficients = np.linalg.solve(
            design.T @ design + penalty, design.T @ target
        )
        feature_names = ["intercept", "cycle_fraction", "niah"]
        for name in candidates:
            feature_names.extend(
                [f"candidate[{name}]", f"candidate[{name}]*cycle", f"candidate[{name}]*niah"]
            )
        return cls(
            candidates=candidates,
            coefficients=coefficients,
            feature_names=tuple(feature_names),
            source_path=str(path),
            training_rows=int(len(grouped)),
        )

    def predict(self, candidate: str, cycle_fraction: float, task: str) -> float:
        if str(candidate) not in self.candidates:
            return float("inf")
        features = self._features(
            str(candidate), float(cycle_fraction), str(task), self.candidates
        )
        return float(features @ self.coefficients)

    def metadata(self) -> Dict[str, Any]:
        return {
            "source_path": self.source_path,
            "training_rows": self.training_rows,
            "target": "log(mean_exact_kl + 1e-8)",
            "feature_names": list(self.feature_names),
            "coefficients": [float(value) for value in self.coefficients],
        }


@dataclass
class CheapPolicyContext:
    core_budget: int
    sink_size: int
    recent_size: int
    pooling_kernel: int
    pooling_method: str
    cascade_margin: float = 0.15
    adaptive_budget_delta: int = 44
    output_diagnostic_layers: Optional[Tuple[int, ...]] = None
    ranker: Optional[HistoricalCandidateRanker] = None

    @staticmethod
    def requires_candidate_panel(policy: str) -> bool:
        return str(policy) in {
            "a1_candidate_proxy",
            "a3_set_output_perturbation",
            "a4_uncertainty_cascade",
            "b1_historical_tiny_ranker",
        }

    def _signals(
        self,
        memory: AttentionPolicyMemory,
        backing: KVBackingStore,
    ) -> Tuple[
        Sequence[int], Sequence[int], np.ndarray, Dict[int, Dict[str, np.ndarray]]
    ]:
        positions = backing.positions()
        _, _, eligible = mandatory_and_eligible(
            positions, int(self.sink_size), int(self.recent_size)
        )
        rows = _eligible_rows(positions, eligible)
        signals: Dict[int, Dict[str, np.ndarray]] = {}
        for layer in memory.layers:
            latest = memory.score(
                layer,
                positions,
                "attention",
                self.pooling_kernel,
                self.pooling_method,
            )
            volatility = memory.volatility_score(layer, positions)
            cumulative = memory.score(
                layer,
                positions,
                "h2o",
                self.pooling_kernel,
                self.pooling_method,
            )
            signals[int(layer)] = {
                "attention": _normalize_on_eligible(latest, rows),
                "volatility": _normalize_on_eligible(volatility, rows),
                "cumulative": _normalize_on_eligible(cumulative, rows),
            }
        return positions, eligible, rows, signals

    def _candidate_proxy_scores(
        self,
        panel: Mapping[str, CoreSelection],
        positions: Sequence[int],
        eligible: Sequence[int],
        signals: Mapping[int, Mapping[str, np.ndarray]],
        previous_cores: Optional[Mapping[int, Sequence[int]]],
    ) -> Dict[str, float]:
        row_by_position = {
            int(position): index for index, position in enumerate(positions)
        }
        scores: Dict[str, float] = {}
        for name, selection in panel.items():
            layer_risks = []
            for layer, current in selection.by_layer.items():
                core = set(int(value) for value in current.selected_positions)
                deleted_rows = np.asarray(
                    [
                        row_by_position[int(position)]
                        for position in eligible
                        if int(position) not in core
                    ],
                    dtype=np.int64,
                )
                current_signals = signals[int(layer)]
                attention_loss = float(
                    current_signals["attention"][deleted_rows].sum()
                )
                volatility_loss = float(
                    current_signals["volatility"][deleted_rows].sum()
                )
                cumulative_loss = float(
                    current_signals["cumulative"][deleted_rows].sum()
                )
                churn = 0.0
                if previous_cores is not None:
                    previous = set(
                        int(value) for value in previous_cores[int(layer)]
                    )
                    churn = 1.0 - len(core & previous) / max(1, len(core | previous))
                layer_risks.append(
                    0.60 * attention_loss
                    + 0.25 * volatility_loss
                    + 0.15 * cumulative_loss
                    + 0.02 * churn
                )
            scores[str(name)] = float(np.mean(layer_risks))
        return scores

    def _set_output_scores(
        self,
        panel: Mapping[str, CoreSelection],
        memory: AttentionPolicyMemory,
        backing: KVBackingStore,
        positions: Sequence[int],
        eligible: Sequence[int],
    ) -> Dict[str, float]:
        layer_ids = tuple(
            int(value)
            for value in (
                self.output_diagnostic_layers
                if self.output_diagnostic_layers is not None
                else memory.layers
            )
            if int(value) in memory.layers
        )
        row_by_position = {
            int(position): index for index, position in enumerate(positions)
        }
        result = {str(name): [] for name in panel}
        for layer in layer_ids:
            weights = memory.head_score(layer, positions)
            if weights.size == 0:
                continue
            _, value_tensor = backing.layer_arrays(layer)
            values = value_tensor.detach().float().cpu().numpy()[0]
            kv_heads, token_count, _ = values.shape
            query_heads = int(weights.shape[0])
            if token_count != len(positions) or query_heads % kv_heads:
                raise RuntimeError("output proxy attention/value shapes diverged")
            grouped = weights.reshape(kv_heads, query_heads // kv_heads, token_count)
            output = np.einsum("kgt,ktd->kgd", grouped, values, optimize=True)
            output_energy = np.square(output).sum(axis=-1) + 1.0e-8
            for name, selection in panel.items():
                core = set(
                    int(value)
                    for value in selection.by_layer[layer].selected_positions
                )
                deleted = np.asarray(
                    [
                        row_by_position[int(position)]
                        for position in eligible
                        if int(position) not in core
                    ],
                    dtype=np.int64,
                )
                if deleted.size == 0:
                    result[str(name)].append(0.0)
                    continue
                removed_mass = grouped[:, :, deleted].sum(axis=-1)
                removed_value = np.einsum(
                    "kgt,ktd->kgd",
                    grouped[:, :, deleted],
                    values[:, deleted, :],
                    optimize=True,
                )
                denominator = np.maximum(1.0 - removed_mass, 1.0e-5)
                delta = (
                    removed_mass[:, :, None] * output - removed_value
                ) / denominator[:, :, None]
                relative_energy = np.square(delta).sum(axis=-1) / output_energy
                result[str(name)].append(float(np.mean(relative_energy)))
        return {
            name: float(np.mean(values)) if values else float("inf")
            for name, values in result.items()
        }

    def _direct_scores(
        self,
        memory: AttentionPolicyMemory,
        backing: KVBackingStore,
        positions: Sequence[int],
        eligible_rows: np.ndarray,
        signals: Mapping[int, Mapping[str, np.ndarray]],
    ) -> Dict[int, np.ndarray]:
        result: Dict[int, np.ndarray] = {}
        for layer in memory.layers:
            attention = np.asarray(signals[int(layer)]["attention"])
            volatility = np.asarray(signals[int(layer)]["volatility"])
            head_attention = memory.head_score(layer, positions)
            contribution = np.zeros(len(positions), dtype=np.float64)
            if head_attention.size:
                _, value_tensor = backing.layer_arrays(int(layer))
                values = value_tensor.detach().float().cpu().numpy()[0]
                contribution = contribution_token_score(
                    head_attention, values, dtype=np.dtype(np.float32)
                ).astype(np.float64)
            contribution = _normalize_on_eligible(
                contribution, eligible_rows
            )
            result[int(layer)] = (
                0.50 * attention
                + 0.30 * volatility
                + 0.20 * contribution
            )
        return result

    @staticmethod
    def _adaptive_budgets(
        scores: Mapping[int, np.ndarray],
        eligible_rows: np.ndarray,
        core_budget: int,
        maximum_delta: int,
        eligible_count: int,
    ) -> Dict[int, int]:
        layers = tuple(sorted(int(layer) for layer in scores))
        difficulty = []
        for layer in layers:
            probability = np.asarray(scores[layer], dtype=np.float64)[eligible_rows]
            mass = float(probability.sum())
            if mass <= 0.0:
                difficulty.append(1.0)
                continue
            probability = probability / mass
            entropy = -float(
                np.sum(probability * np.log(np.maximum(probability, 1.0e-12)))
            )
            difficulty.append(float(np.exp(entropy)))
        difficulty_array = np.asarray(difficulty, dtype=np.float64)
        raw = (
            len(layers)
            * int(core_budget)
            * difficulty_array
            / max(float(difficulty_array.sum()), 1.0e-12)
        )
        minimum = max(0, int(core_budget) - int(maximum_delta))
        maximum = min(
            int(eligible_count), int(core_budget) + int(maximum_delta)
        )
        budget = np.clip(np.floor(raw).astype(np.int64), minimum, maximum)
        target = len(layers) * int(core_budget)
        while int(budget.sum()) < target:
            choices = [index for index in range(len(layers)) if budget[index] < maximum]
            if not choices:
                raise RuntimeError("adaptive layer budgets cannot fill target")
            index = max(choices, key=lambda item: (raw[item] - budget[item], -item))
            budget[index] += 1
        while int(budget.sum()) > target:
            choices = [index for index in range(len(layers)) if budget[index] > minimum]
            if not choices:
                raise RuntimeError("adaptive layer budgets cannot meet target")
            index = max(choices, key=lambda item: (budget[item] - raw[item], item))
            budget[index] -= 1
        return {
            int(layer): int(value) for layer, value in zip(layers, budget.tolist())
        }

    def select(
        self,
        policy: str,
        panel: Mapping[str, CoreSelection],
        memory: AttentionPolicyMemory,
        backing: KVBackingStore,
        previous_cores: Optional[Mapping[int, Sequence[int]]],
        cycle: int,
        cycles: int,
        task: str,
    ) -> Tuple[CoreSelection, str, Dict[str, Any]]:
        if str(policy) not in CHEAP_POLICIES:
            raise ValueError("unknown cheap policy=%s" % policy)
        positions, eligible, eligible_rows, signals = self._signals(memory, backing)
        diagnostics: Dict[str, Any] = {}
        if policy in {"a1_candidate_proxy", "a4_uncertainty_cascade"}:
            proxy_scores = self._candidate_proxy_scores(
                panel, positions, eligible, signals, previous_cores
            )
            ranked = sorted(proxy_scores, key=lambda name: (proxy_scores[name], name))
            selected_name = ranked[0]
            denominator = max(abs(proxy_scores[ranked[1]]), 1.0e-12)
            margin = float(
                (proxy_scores[ranked[1]] - proxy_scores[ranked[0]])
                / denominator
            )
            diagnostics.update(
                {
                    "proxy_selected_candidate": selected_name,
                    "proxy_score": float(proxy_scores[selected_name]),
                    "proxy_relative_margin": margin,
                }
            )
            if policy == "a1_candidate_proxy" or margin >= float(self.cascade_margin):
                diagnostics["cascade_branch"] = "a1_proxy"
                return panel[selected_name], selected_name, diagnostics
        if policy in {"a3_set_output_perturbation", "a4_uncertainty_cascade"}:
            output_scores = self._set_output_scores(
                panel, memory, backing, positions, eligible
            )
            selected_name = min(
                output_scores, key=lambda name: (output_scores[name], name)
            )
            diagnostics.update(
                {
                    "set_output_score": float(output_scores[selected_name]),
                    "cascade_branch": "a3_set_output",
                }
            )
            return panel[selected_name], selected_name, diagnostics
        if policy == "b1_historical_tiny_ranker":
            if self.ranker is None:
                raise RuntimeError("B1 requires a fitted historical ranker")
            fraction = float(cycle) / max(1, int(cycles) - 1)
            predicted = {
                name: self.ranker.predict(name, fraction, task) for name in panel
            }
            selected_name = min(
                predicted, key=lambda name: (predicted[name], name)
            )
            diagnostics["ranker_predicted_log_risk"] = float(
                predicted[selected_name]
            )
            return panel[selected_name], selected_name, diagnostics
        if policy == "a2_temporal_volatility":
            direct_scores = {
                int(layer): np.asarray(values["volatility"], dtype=np.float64)
                for layer, values in signals.items()
            }
            budgets = {int(layer): int(self.core_budget) for layer in direct_scores}
        else:
            direct_scores = self._direct_scores(
                memory, backing, positions, eligible_rows, signals
            )
            if policy == "b3_layer_adaptive_budget":
                budgets = self._adaptive_budgets(
                    direct_scores,
                    eligible_rows,
                    self.core_budget,
                    self.adaptive_budget_delta,
                    len(eligible),
                )
                diagnostics.update(
                    {
                        "minimum_layer_core_budget": int(min(budgets.values())),
                        "maximum_layer_core_budget": int(max(budgets.values())),
                        "mean_layer_core_budget": float(np.mean(list(budgets.values()))),
                    }
                )
            else:
                budgets = {
                    int(layer): int(self.core_budget) for layer in direct_scores
                }
        selection = _direct_selection(
            str(policy), positions, eligible, direct_scores, budgets
        )
        diagnostics["direct_action"] = True
        return selection, str(policy), diagnostics


__all__ = ["CHEAP_POLICIES", "CheapPolicyContext", "HistoricalCandidateRanker"]
