"""Stage-1 same-selector functional-staleness experiment."""
from __future__ import annotations

import json
import math
import time
import traceback
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import torch

from statekv.artifacts import json_text
from statekv.backend import QueryRecord, ReferenceTrajectory
from statekv.config import CacheDiscoveryConfig, DiscoveryConfig
from statekv.functional_features import (
    build_layer_features,
    functional_measurement,
)
from statekv.runner import TemporalDiscoveryRunner, _sample_slug
from statekv.storage import atomic_frame as _atomic_frame
from statekv.selectors import CoreSelection, CoreSelector
from statekv.tasks import load_discovery_tasks


FUNCTIONAL_TABLES = (
    "reference_inventory",
    "probe_index",
    "set_metrics",
    "functional_features",
    "attention_labels",
    "downstream_labels",
    "identity_checks",
    "runtime_costs",
)


@dataclass
class ProbeStep:
    logits: torch.Tensor
    diagnostic: QueryRecord
    position_maps: Dict[int, torch.Tensor]
    target_index: int
    target_token_id: int
    target_token_position: int
    active_cache_tokens: int
    forward_time_s: float


def _jaccard(left: Iterable[int], right: Iterable[int]) -> float:
    a, b = set(int(value) for value in left), set(
        int(value) for value in right
    )
    union = a | b
    return float(len(a & b) / len(union)) if union else 1.0


def _score_correlation(
    base: Any,
    fresh: Any,
    base_positions: Sequence[int],
    fresh_positions: Sequence[int],
) -> Tuple[float, float, int]:
    left = dict(zip(base_positions, base.aggregate_scores))
    right = dict(zip(fresh_positions, fresh.aggregate_scores))
    common = [
        position
        for position in base.eligible_positions
        if position in set(fresh.eligible_positions)
        and math.isfinite(float(left.get(position, float("nan"))))
        and math.isfinite(float(right.get(position, float("nan"))))
    ]
    if len(common) < 2:
        return float("nan"), float("nan"), len(common)
    left_series = pd.Series([left[position] for position in common])
    right_series = pd.Series([right[position] for position in common])
    return (
        float(left_series.corr(right_series, method="spearman")),
        float(left_series.corr(right_series, method="pearson")),
        len(common),
    )


def _squared_distance(left: torch.Tensor, right: torch.Tensor) -> float:
    difference = left.detach().double().cpu() - right.detach().double().cpu()
    return float((difference * difference).sum().item())


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    a = left.detach().double().cpu().flatten()
    b = right.detach().double().cpu().flatten()
    denominator = float(torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b))
    if denominator <= 0.0:
        return float("nan")
    return float(torch.dot(a, b).item() / denominator)


def _distribution_metrics(
    full_logits: torch.Tensor,
    old_logits: torch.Tensor,
    fresh_logits: torch.Tensor,
    target_token: int,
    floor: float,
) -> Dict[str, float]:
    full_log = torch.log_softmax(full_logits.detach().double().flatten(), dim=0)
    old_log = torch.log_softmax(old_logits.detach().double().flatten(), dim=0)
    fresh_log = torch.log_softmax(
        fresh_logits.detach().double().flatten(), dim=0
    )
    full_probability = torch.exp(full_log)

    def kl(candidate_log: torch.Tensor) -> float:
        value = (full_probability * (full_log - candidate_log)).sum()
        return float(value.item())

    def js(candidate_log: torch.Tensor) -> float:
        candidate = torch.exp(candidate_log)
        mixture = 0.5 * (full_probability + candidate)
        log_mixture = torch.log(mixture.clamp_min(float(floor)))
        value = 0.5 * (
            (full_probability * (full_log - log_mixture)).sum()
            + (candidate * (candidate_log - log_mixture)).sum()
        )
        return float(value.item())

    full_nll = -float(full_log[int(target_token)].item())
    old_nll = -float(old_log[int(target_token)].item())
    fresh_nll = -float(fresh_log[int(target_token)].item())
    old_kl = kl(old_log)
    fresh_kl = kl(fresh_log)
    old_js = js(old_log)
    fresh_js = js(fresh_log)
    return {
        "full_nll": full_nll,
        "old_nll": old_nll,
        "fresh_nll": fresh_nll,
        "old_delta_nll": old_nll - full_nll,
        "fresh_delta_nll": fresh_nll - full_nll,
        "refresh_benefit_nll": old_nll - fresh_nll,
        "old_exact_kl": old_kl,
        "fresh_exact_kl": fresh_kl,
        "refresh_benefit_exact_kl": old_kl - fresh_kl,
        "old_js": old_js,
        "fresh_js": fresh_js,
        "refresh_benefit_js": old_js - fresh_js,
        "old_fresh_logit_l2_sq": _squared_distance(
            old_logits, fresh_logits
        ),
        "full_old_logit_l2_sq": _squared_distance(
            full_logits, old_logits
        ),
        "full_fresh_logit_l2_sq": _squared_distance(
            full_logits, fresh_logits
        ),
    }


def _condition_cache(
    cfg: DiscoveryConfig, total_budget: int, protected_recent: int
) -> CacheDiscoveryConfig:
    # One transient slot is required for the current query. protected_recent=0
    # therefore maps to an effective replay recent_size of one.
    effective_recent = max(1, int(protected_recent))
    core = int(total_budget) - int(cfg.cache.sink_size) - effective_recent
    if core <= 0:
        raise ValueError("functional cache condition leaves no core budget")
    return CacheDiscoveryConfig(
        total_budget=int(total_budget),
        sink_size=int(cfg.cache.sink_size),
        recent_size=effective_recent,
        selected_core_budget=core,
    )


class FunctionalProbeRunner(TemporalDiscoveryRunner):
    """Execute the pre-registered small functional-staleness matrix."""

    def run(self) -> Path:
        if not self.cfg.functional_probe.enabled:
            raise ValueError("functional_probe.enabled must be true")
        self.store.status["state"] = "running"
        self.store.status["protocol"] = "functional_staleness_stage1_v1"
        self.store.save_status()
        samples, task_events = load_discovery_tasks(self.cfg)
        model_info = self.model.load()
        if not hasattr(self.model, "project_features"):
            raise RuntimeError(
                "functional probing requires backend output projection access"
            )
        self.metadata = self.store.write_metadata(model_info, task_events)
        for table in FUNCTIONAL_TABLES:
            (
                self.store.run_dir
                / "fragments"
                / "functional_probe"
                / table
            ).mkdir(parents=True, exist_ok=True)
        try:
            for sample in samples:
                self._run_functional_sample(sample)
            outputs = self._consolidate()
            self.store.status["state"] = "complete"
            self.store.status["functional_outputs"] = {
                key: str(value) for key, value in outputs.items()
            }
            self.store.save_status()
        finally:
            self.model.close()
        return self.store.run_dir

    def _fragment_path(self, table: str, sample_id: str) -> Path:
        return (
            self.store.run_dir
            / "fragments"
            / "functional_probe"
            / table
            / ("%s.parquet" % _sample_slug(sample_id))
        )

    def _write_tables(
        self, sample_id: str, tables: Mapping[str, pd.DataFrame]
    ) -> None:
        for table in FUNCTIONAL_TABLES:
            _atomic_frame(
                tables.get(table, pd.DataFrame()),
                self._fragment_path(table, sample_id),
            )

    def _consolidate(self) -> Dict[str, Path]:
        outputs: Dict[str, Path] = {}
        for table in FUNCTIONAL_TABLES:
            fragments = sorted(
                (
                    self.store.run_dir
                    / "fragments"
                    / "functional_probe"
                    / table
                ).glob("*.parquet")
            )
            frames = [pd.read_parquet(path) for path in fragments]
            frame = (
                pd.concat(frames, ignore_index=True, sort=False)
                if frames
                else pd.DataFrame()
            )
            parquet = self.store.run_dir / ("%s.parquet" % table)
            csv = self.store.run_dir / ("%s.csv" % table)
            _atomic_frame(frame, parquet)
            _atomic_frame(frame, csv)
            outputs[table] = parquet
        return outputs

    def _condition_base(
        self,
        sample: Any,
        base_anchor: int,
        lag: int,
        strategy: str,
        total_budget: int,
        protected_recent: int,
        cache_cfg: CacheDiscoveryConfig,
    ) -> Dict[str, Any]:
        return {
            **self._base(sample),
            "base_anchor": int(base_anchor),
            "probe_lag": int(lag),
            "refresh_anchor": int(base_anchor + lag),
            "strategy": strategy,
            "total_budget": int(total_budget),
            "sink_size": int(cache_cfg.sink_size),
            "protected_recent_size": int(protected_recent),
            "effective_replay_recent_size": int(cache_cfg.recent_size),
            "selected_core_budget": int(cache_cfg.selected_core_budget),
        }

    def _selection(
        self,
        reference: ReferenceTrajectory,
        step: int,
        strategy: str,
        cache_cfg: CacheDiscoveryConfig,
    ) -> CoreSelection:
        condition_cfg = replace(self.cfg, cache=cache_cfg)
        selector = CoreSelector(condition_cfg)
        return selector.select(
            reference.anchors[int(step)].snapshot(reference.sample_id),
            strategy,
        )

    def _replay_probes(
        self,
        reference: ReferenceTrajectory,
        anchor_step: int,
        selection: CoreSelection,
        cache_cfg: CacheDiscoveryConfig,
        capture_steps: Set[int],
    ) -> Dict[int, ProbeStep]:
        state, fixed = self.model.state_from_anchor(
            reference.anchors[int(anchor_step)],
            selection,
            cache_config=cache_cfg,
        )
        current_token = int(
            reference.anchors[int(anchor_step)].query_token_id
        )
        output: Dict[int, ProbeStep] = {}
        try:
            for future_step in range(1, max(capture_steps) + 1):
                if future_step > 1:
                    self.model.prune_recent_before_query(
                        state, fixed, cache_config=cache_cfg
                    )
                logits, diagnostic, forward_s = self.model.forward_one(
                    state, current_token, capture_attention=True
                )
                self.model.validate_active_budget(
                    state, cache_config=cache_cfg
                )
                target_index = int(anchor_step + future_step - 1)
                target_token = int(
                    reference.generated_token_ids[target_index]
                )
                if future_step in capture_steps:
                    output[future_step] = ProbeStep(
                        logits=logits.detach().float().cpu().clone(),
                        diagnostic=diagnostic,
                        position_maps={
                            int(layer): positions.detach().cpu().clone()
                            for layer, positions in state.position_maps.items()
                        },
                        target_index=target_index,
                        target_token_id=target_token,
                        target_token_position=int(
                            reference.prompt_length + target_index
                        ),
                        active_cache_tokens=int(
                            self.model.active_cache_tokens(state)
                        ),
                        forward_time_s=float(forward_s),
                    )
                current_token = target_token
        finally:
            self.model.release(state)
        return output

    @staticmethod
    def _mandatory_positions(
        positions: Sequence[int],
        cache_cfg: CacheDiscoveryConfig,
        include_recent: bool,
    ) -> Set[int]:
        sink = set(positions[: int(cache_cfg.sink_size)])
        if not include_recent:
            return sink
        recent = (
            set(positions[-int(cache_cfg.recent_size) :])
            if int(cache_cfg.recent_size) > 0
            else set()
        )
        return sink | recent

    def _set_rows(
        self,
        common: Dict[str, Any],
        base_anchor: Any,
        current_anchor: Any,
        old_selection: CoreSelection,
        fresh_selection: CoreSelection,
    ) -> List[Dict[str, Any]]:
        rows = []
        for layer in sorted(old_selection.by_layer):
            old = old_selection.by_layer[layer]
            fresh = fresh_selection.by_layer[layer]
            base_positions = [
                int(value)
                for value in base_anchor.position_maps[layer].tolist()
            ]
            current_positions = [
                int(value)
                for value in current_anchor.position_maps[layer].tolist()
            ]
            spearman, pearson, comparable = _score_correlation(
                old, fresh, base_positions, current_positions
            )
            old_set = set(old.selected_positions)
            fresh_set = set(fresh.selected_positions)
            new_positions = {
                position
                for position in fresh_set
                if position > max(base_positions)
            }
            rows.append(
                {
                    **common,
                    "layer": int(layer),
                    "old_selected_count": len(old_set),
                    "fresh_selected_count": len(fresh_set),
                    "intersection_count": len(old_set & fresh_set),
                    "symmetric_difference_count": len(old_set ^ fresh_set),
                    "selected_core_jaccard": _jaccard(old_set, fresh_set),
                    "selected_core_turnover": float(
                        1.0
                        - len(old_set & fresh_set) / max(1, len(old_set))
                    ),
                    "new_token_entry_count": len(new_positions),
                    "new_token_entry_fraction": float(
                        len(new_positions) / max(1, len(fresh_set))
                    ),
                    "old_token_score_spearman": spearman,
                    "old_token_score_pearson": pearson,
                    "common_old_token_count": comparable,
                    "old_boundary_margin": old.boundary_margin,
                    "fresh_boundary_margin": fresh.boundary_margin,
                    "old_eligible_count": len(old.eligible_positions),
                    "fresh_eligible_count": len(fresh.eligible_positions),
                }
            )
        return rows

    def _feature_rows(
        self,
        common: Dict[str, Any],
        base_anchor: Any,
        current_anchor: Any,
        old_selection: CoreSelection,
        fresh_selection: CoreSelection,
        old_probe: ProbeStep,
        fresh_probe: ProbeStep,
        cache_cfg: CacheDiscoveryConfig,
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        variants = self.cfg.functional_probe.feature_variants
        cache_key = (id(base_anchor), id(current_anchor))
        if getattr(self, "_feature_pair_cache_key", None) != cache_key:
            if hasattr(self, "_feature_pair_cache"):
                del self._feature_pair_cache
                self.model.release()
            self._feature_pair_cache_key = cache_key
            self._feature_pair_cache = {}
            for cached_layer in self.model.selected_layers:
                cached_layer = int(cached_layer)
                cached_heads = self.model.selected_heads[cached_layer]
                self._feature_pair_cache[cached_layer] = (
                    build_layer_features(
                        self.model,
                        base_anchor,
                        cached_layer,
                        cached_heads,
                        variants,
                    ),
                    build_layer_features(
                        self.model,
                        current_anchor,
                        cached_layer,
                        cached_heads,
                        variants,
                    ),
                )
        for layer in self.model.selected_layers:
            layer = int(layer)
            base_features, current_features = self._feature_pair_cache[layer]
            base_positions = base_features.positions
            current_positions = current_features.positions
            base_fixed = (
                self._mandatory_positions(
                    base_positions, cache_cfg, include_recent=False
                )
                | set(old_selection.by_layer[layer].selected_positions)
            )
            current_sink = self._mandatory_positions(
                current_positions, cache_cfg, include_recent=False
            )
            old_fixed = current_sink | set(
                old_selection.by_layer[layer].selected_positions
            )
            fresh_fixed = current_sink | set(
                fresh_selection.by_layer[layer].selected_positions
            )
            for scope in (
                "historical_core_only",
                "active_cache_with_recent",
            ):
                if scope == "historical_core_only":
                    old_history = old_fixed
                    fresh_history = fresh_fixed
                    base_history = base_fixed
                else:
                    old_history = set(
                        int(value)
                        for value in old_probe.position_maps[layer].tolist()
                    )
                    fresh_history = set(
                        int(value)
                        for value in fresh_probe.position_maps[layer].tolist()
                    )
                    base_history = base_fixed | self._mandatory_positions(
                        base_positions, cache_cfg, include_recent=True
                    )
                for key in sorted(
                    current_features.matrices,
                    key=lambda value: (
                        value[0],
                        value[1],
                        -1 if value[2] is None else value[2],
                    ),
                ):
                    measurement = functional_measurement(
                        base_features=base_features,
                        current_features=current_features,
                        key=key,
                        old_history_positions=old_history,
                        fresh_history_positions=fresh_history,
                        base_old_history_positions=base_history,
                        epsilon=self.cfg.functional_probe.identity_epsilon,
                        ridge_coefficient=self.cfg.selectors.ridge_lambda,
                        ridge_mode=self.cfg.selectors.ridge_lambda_mode,
                    )
                    variant, granularity, head = key
                    rows.append(
                        {
                            **common,
                            "layer": layer,
                            "head": head,
                            "feature_variant": variant,
                            "feature_granularity": granularity,
                            "coverage_scope": scope,
                            "evaluation_universe": "full_reference_history",
                            "feature_dimension": int(
                                current_features.matrices[key].shape[1]
                            ),
                            "feature_dtype": "float32_input_float64_factor",
                            "observation_window_queries": (
                                current_features.observation_window_queries
                            ),
                            "observation_weight_source": (
                                current_features.observation_weight_source
                            ),
                            "access_class": "offline_oracle",
                            "uses_fresh_selection": True,
                            "uses_full_history": True,
                            "uses_current_query": False,
                            "estimated_online_cost": (
                                "not_deployable_full_history_counterfactual"
                            ),
                            "deployable_approx_access_class": (
                                "online_deployable"
                                if variant in {"raw_v", "projected_v"}
                                else (
                                    "requires_backing_store_and_current_"
                                    "attention_for_new_tokens"
                                )
                            ),
                            "old_coverage_raw_sum": measurement[
                                "old_coverage"
                            ]["raw_sum"],
                            "old_coverage_normalized_sum": measurement[
                                "old_coverage"
                            ]["normalized_sum"],
                            "old_coverage_energy_normalized_sum": measurement[
                                "old_coverage"
                            ]["energy_normalized_sum"],
                            "fresh_coverage_raw_sum": measurement[
                                "fresh_coverage"
                            ]["raw_sum"],
                            "fresh_coverage_normalized_sum": measurement[
                                "fresh_coverage"
                            ]["normalized_sum"],
                            "fresh_coverage_energy_normalized_sum": measurement[
                                "fresh_coverage"
                            ]["energy_normalized_sum"],
                            "delta_e_raw_sum": measurement["delta_e"][
                                "raw_sum"
                            ],
                            "delta_e_normalized_sum": measurement["delta_e"][
                                "normalized_sum"
                            ],
                            "delta_e_energy_normalized_sum": measurement[
                                "delta_e"
                            ]["energy_normalized_sum"],
                            "d_new_raw_sum": measurement["d_new"]["raw_sum"],
                            "d_new_normalized_sum": measurement["d_new"][
                                "normalized_sum"
                            ],
                            "d_new_token_count": measurement["d_new"][
                                "token_count"
                            ],
                            "d_rew_raw_sum": measurement["d_rew"]["raw_sum"],
                            "d_rew_normalized_sum": measurement["d_rew"][
                                "normalized_sum"
                            ],
                            "d_rew_token_count": measurement["d_rew"][
                                "token_count"
                            ],
                            "d_func_raw_sum": measurement["d_func_raw_sum"],
                            "d_func_normalized_sum": measurement[
                                "d_func_normalized_sum"
                            ],
                            "arrival_residual_raw_sum": measurement[
                                "arrival_residual"
                            ]["raw_sum"],
                            "arrival_residual_normalized_sum": measurement[
                                "arrival_residual"
                            ]["normalized_sum"],
                            "arrival_residual_token_count": measurement[
                                "arrival_residual"
                            ]["token_count"],
                            "retained_reweighting_raw_sum": measurement[
                                "retained_reweighting"
                            ]["raw_sum"],
                            "retained_reweighting_normalized_sum": measurement[
                                "retained_reweighting"
                            ]["normalized_sum"],
                            "retained_reweighting_token_count": measurement[
                                "retained_reweighting"
                            ]["token_count"],
                            "deployable_approx_raw_sum": measurement[
                                "deployable_approx_raw_sum"
                            ],
                            "deployable_approx_normalized_sum": measurement[
                                "deployable_approx_normalized_sum"
                            ],
                            "old_history_rows": measurement["old_factor"][
                                "ridge_history_rows"
                            ],
                            "fresh_history_rows": measurement[
                                "fresh_factor"
                            ]["ridge_history_rows"],
                            "old_ridge": measurement["old_factor"]["ridge"],
                            "fresh_ridge": measurement["fresh_factor"][
                                "ridge"
                            ],
                            "old_ridge_calculation": measurement[
                                "old_factor"
                            ]["ridge_calculation"],
                            "fresh_ridge_calculation": measurement[
                                "fresh_factor"
                            ]["ridge_calculation"],
                            "old_regularized_condition_number": measurement[
                                "old_factor"
                            ]["regularized_condition_number"],
                            "fresh_regularized_condition_number": measurement[
                                "fresh_factor"
                            ]["regularized_condition_number"],
                        }
                    )
        return rows

    def _attention_and_identity_rows(
        self,
        common: Dict[str, Any],
        reference: ReferenceTrajectory,
        current_anchor: Any,
        old_probe: ProbeStep,
        fresh_probe: ProbeStep,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        labels: List[Dict[str, Any]] = []
        identities: List[Dict[str, Any]] = []
        full_record = reference.query_records[old_probe.target_index]
        epsilon = float(self.cfg.functional_probe.identity_epsilon)
        denominator_floor = float(
            self.cfg.functional_probe.identity_denominator_floor
        )
        for layer in self.model.selected_layers:
            layer = int(layer)
            full_projected = full_record.projected_attention_outputs[layer]
            old_projected = (
                old_probe.diagnostic.projected_attention_outputs[layer]
            )
            fresh_projected = (
                fresh_probe.diagnostic.projected_attention_outputs[layer]
            )
            old_projected_error = _squared_distance(
                old_projected, full_projected
            )
            fresh_projected_error = _squared_distance(
                fresh_projected, full_projected
            )
            labels.append(
                {
                    **common,
                    "layer": layer,
                    "head": None,
                    "label_granularity": "layer_projected",
                    "old_full_error_sq": old_projected_error,
                    "fresh_full_error_sq": fresh_projected_error,
                    "refresh_benefit_output": (
                        old_projected_error - fresh_projected_error
                    ),
                    "old_fresh_distance_sq": _squared_distance(
                        old_projected, fresh_projected
                    ),
                    "old_full_cosine": _cosine(
                        old_projected, full_projected
                    ),
                    "fresh_full_cosine": _cosine(
                        fresh_projected, full_projected
                    ),
                }
            )
            full_positions = [
                int(value)
                for value in current_anchor.position_maps[layer].tolist()
            ]
            row_by_position = {
                position: row
                for row, position in enumerate(full_positions)
            }
            values = current_anchor.values[layer].detach()[0].float().cpu()
            group = int(
                self.model.model_info["gqa_query_heads_per_kv_head"]
            )
            for head in self.model.selected_heads[layer]:
                head = int(head)
                full = full_record.all_head_attention_outputs[layer][head]
                old = old_probe.diagnostic.all_head_attention_outputs[layer][
                    head
                ]
                fresh = (
                    fresh_probe.diagnostic.all_head_attention_outputs[layer][
                        head
                    ]
                )
                old_error = _squared_distance(old, full)
                fresh_error = _squared_distance(fresh, full)
                labels.append(
                    {
                        **common,
                        "layer": layer,
                        "head": head,
                        "label_granularity": "head_pre_projection",
                        "old_full_error_sq": old_error,
                        "fresh_full_error_sq": fresh_error,
                        "refresh_benefit_output": old_error - fresh_error,
                        "old_fresh_distance_sq": _squared_distance(
                            old, fresh
                        ),
                        "old_full_cosine": _cosine(old, full),
                        "fresh_full_cosine": _cosine(fresh, full),
                    }
                )
                old_difference = old - full
                fresh_difference = fresh - full
                old_projected_head = self.model.project_features(
                    layer, old_difference.reshape(1, -1), head=head
                )[0]
                fresh_projected_head = self.model.project_features(
                    layer, fresh_difference.reshape(1, -1), head=head
                )[0]
                old_projected_head_error = float(
                    (old_projected_head.double() ** 2).sum().item()
                )
                fresh_projected_head_error = float(
                    (fresh_projected_head.double() ** 2).sum().item()
                )
                labels.append(
                    {
                        **common,
                        "layer": layer,
                        "head": head,
                        "label_granularity": "head_projected",
                        "old_full_error_sq": old_projected_head_error,
                        "fresh_full_error_sq": fresh_projected_head_error,
                        "refresh_benefit_output": (
                            old_projected_head_error
                            - fresh_projected_head_error
                        ),
                        "old_fresh_distance_sq": _squared_distance(
                            old_projected_head, fresh_projected_head
                        ),
                        "old_full_cosine": float("nan"),
                        "fresh_full_cosine": float("nan"),
                    }
                )
                full_attention = (
                    full_record.all_head_attention_distributions[layer][
                        head
                    ]
                    .detach()
                    .double()
                    .cpu()
                )
                kv_head = head // group
                full_values = values[kv_head].double()
                a_full = full.double()
                for arm, probe in (
                    ("old", old_probe),
                    ("fresh", fresh_probe),
                ):
                    retained = set(
                        int(value)
                        for value in probe.position_maps[layer].tolist()
                    )
                    deleted_rows = [
                        row
                        for position, row in row_by_position.items()
                        if position not in retained
                    ]
                    if deleted_rows:
                        deleted_index = torch.tensor(
                            deleted_rows, dtype=torch.long
                        )
                        deleted_attention = full_attention.index_select(
                            0, deleted_index
                        )
                        deleted_values = full_values.index_select(
                            0, deleted_index
                        )
                        deleted_mass = float(deleted_attention.sum().item())
                        deleted_value = (
                            deleted_attention.unsqueeze(-1)
                            * deleted_values
                        ).sum(dim=0)
                    else:
                        deleted_mass = 0.0
                        deleted_value = torch.zeros_like(a_full)
                    denominator = 1.0 - deleted_mass
                    stable = denominator >= denominator_floor
                    if stable:
                        masked = (a_full - deleted_value) / denominator
                        identity = (
                            deleted_mass * a_full - deleted_value
                        ) / denominator
                        actual_delta = masked - a_full
                        identity_relative_error = float(
                            torch.linalg.vector_norm(
                                actual_delta - identity
                            ).item()
                            / (
                                torch.linalg.vector_norm(actual_delta).item()
                                + epsilon
                            )
                        )
                        identity_delta_norm_sq = float(
                            (identity * identity).sum().item()
                        )
                    else:
                        identity_relative_error = float("nan")
                        identity_delta_norm_sq = float("nan")
                    mass_term = deleted_mass * a_full
                    mass_norm_sq = float((mass_term * mass_term).sum().item())
                    deleted_value_norm_sq = float(
                        (deleted_value * deleted_value).sum().item()
                    )
                    cross_term = float(
                        -2.0
                        * deleted_mass
                        * torch.dot(a_full, deleted_value).item()
                    )
                    additive = mass_norm_sq + deleted_value_norm_sq
                    identities.append(
                        {
                            **common,
                            "layer": layer,
                            "head": head,
                            "arm": arm,
                            "deleted_token_count": len(deleted_rows),
                            "deleted_attention_mass": deleted_mass,
                            "retained_attention_mass": denominator,
                            "stable_denominator": stable,
                            "identity_relative_error": (
                                identity_relative_error
                            ),
                            "identity_delta_norm_sq": (
                                identity_delta_norm_sq
                            ),
                            "mass_term_norm_sq": mass_norm_sq,
                            "deleted_value_norm_sq": deleted_value_norm_sq,
                            "cross_term": cross_term,
                            "additive_terms": additive,
                            "cross_to_additive_ratio": float(
                                cross_term / max(additive, epsilon)
                            ),
                            "fixed_qkv_only": True,
                        }
                    )
        return labels, identities

    def _run_functional_sample(self, sample: Any) -> None:
        slug = _sample_slug(sample.sample_id)
        completion_key = "functional:%s" % slug
        if self.cfg.runtime.resume and self.store.is_complete(completion_key):
            if all(
                self._fragment_path(table, sample.sample_id).exists()
                for table in FUNCTIONAL_TABLES
            ):
                return
        reference: Optional[ReferenceTrajectory] = None
        sample_started = time.perf_counter()
        try:
            reference = self.model.generate_reference(
                sample.sample_id, sample.task, sample.prompt
            )
            generated = len(reference.generated_token_ids)
            required_anchors = set(self.cfg.captured_anchor_steps())
            missing = sorted(
                step
                for step in required_anchors
                if step <= generated and step not in reference.anchors
            )
            if missing:
                raise RuntimeError(
                    "required functional anchors are missing: %s" % missing
                )
            selections: Dict[
                Tuple[int, int, int, str], CoreSelection
            ] = {}

            def select(
                total_budget: int,
                protected_recent: int,
                step: int,
                strategy: str,
            ) -> CoreSelection:
                key = (
                    int(total_budget),
                    int(protected_recent),
                    int(step),
                    strategy,
                )
                if key not in selections:
                    cache_cfg = _condition_cache(
                        self.cfg, total_budget, protected_recent
                    )
                    selections[key] = self._selection(
                        reference, step, strategy, cache_cfg
                    )
                return selections[key]

            stale_cache: Dict[
                Tuple[int, int, int, str], Dict[int, ProbeStep]
            ] = {}
            fresh_cache: Dict[
                Tuple[int, int, int, str], ProbeStep
            ] = {}
            lags = [int(value) for value in self.cfg.functional_probe.probe_lags]
            capture_steps = {lag + 1 for lag in lags}
            runtime_rows: List[Dict[str, Any]] = []
            for total_budget in self.cfg.functional_probe.total_budgets:
                for protected_recent in (
                    self.cfg.functional_probe.protected_recent_sizes
                ):
                    cache_cfg = _condition_cache(
                        self.cfg, total_budget, protected_recent
                    )
                    for base_anchor in (
                        self.cfg.functional_probe.base_anchor_steps
                    ):
                        for strategy in self.cfg.functional_probe.selectors:
                            replay_started = time.perf_counter()
                            stale = self._replay_probes(
                                reference,
                                int(base_anchor),
                                select(
                                    total_budget,
                                    protected_recent,
                                    int(base_anchor),
                                    strategy,
                                ),
                                cache_cfg,
                                capture_steps,
                            )
                            stale_cache[
                                (
                                    int(total_budget),
                                    int(protected_recent),
                                    int(base_anchor),
                                    strategy,
                                )
                            ] = stale
                            runtime_rows.append(
                                {
                                    **self._base(sample),
                                    "arm": "stale_trajectory",
                                    "base_anchor": int(base_anchor),
                                    "strategy": strategy,
                                    "total_budget": int(total_budget),
                                    "protected_recent_size": int(
                                        protected_recent
                                    ),
                                    "replay_wall_time_s": float(
                                        time.perf_counter() - replay_started
                                    ),
                                    "forward_time_s_sum": float(
                                        sum(
                                            probe.forward_time_s
                                            for probe in stale.values()
                                        )
                                    ),
                                    "captured_probe_count": len(stale),
                                }
                            )
                            for lag in lags:
                                refresh_anchor = int(base_anchor + lag)
                                fresh_key = (
                                    int(total_budget),
                                    int(protected_recent),
                                    refresh_anchor,
                                    strategy,
                                )
                                if fresh_key in fresh_cache:
                                    continue
                                fresh_cache[fresh_key] = self._replay_probes(
                                    reference,
                                    refresh_anchor,
                                    select(
                                        total_budget,
                                        protected_recent,
                                        refresh_anchor,
                                        strategy,
                                    ),
                                    cache_cfg,
                                    {1},
                                )[1]

            probe_rows: List[Dict[str, Any]] = []
            set_rows: List[Dict[str, Any]] = []
            feature_rows: List[Dict[str, Any]] = []
            attention_rows: List[Dict[str, Any]] = []
            downstream_rows: List[Dict[str, Any]] = []
            identity_rows: List[Dict[str, Any]] = []
            for base_anchor in self.cfg.functional_probe.base_anchor_steps:
                base_anchor = int(base_anchor)
                for lag in lags:
                    refresh_anchor = base_anchor + lag
                    current_anchor = reference.anchors[refresh_anchor]
                    for total_budget in (
                        self.cfg.functional_probe.total_budgets
                    ):
                        for protected_recent in (
                            self.cfg.functional_probe.protected_recent_sizes
                        ):
                            cache_cfg = _condition_cache(
                                self.cfg, total_budget, protected_recent
                            )
                            for strategy in (
                                self.cfg.functional_probe.selectors
                            ):
                                common = self._condition_base(
                                    sample,
                                    base_anchor,
                                    lag,
                                    strategy,
                                    total_budget,
                                    protected_recent,
                                    cache_cfg,
                                )
                                old_selection = select(
                                    total_budget,
                                    protected_recent,
                                    base_anchor,
                                    strategy,
                                )
                                fresh_selection = select(
                                    total_budget,
                                    protected_recent,
                                    refresh_anchor,
                                    strategy,
                                )
                                old_probe = stale_cache[
                                    (
                                        int(total_budget),
                                        int(protected_recent),
                                        base_anchor,
                                        strategy,
                                    )
                                ][lag + 1]
                                fresh_probe = fresh_cache[
                                    (
                                        int(total_budget),
                                        int(protected_recent),
                                        refresh_anchor,
                                        strategy,
                                    )
                                ]
                                if (
                                    old_probe.target_index
                                    != fresh_probe.target_index
                                    or old_probe.target_token_id
                                    != fresh_probe.target_token_id
                                ):
                                    raise RuntimeError(
                                        "old/fresh target alignment failed"
                                    )
                                probe_rows.append(
                                    {
                                        **common,
                                        "target_index": old_probe.target_index,
                                        "target_token_id": (
                                            old_probe.target_token_id
                                        ),
                                        "target_token_position": (
                                            old_probe.target_token_position
                                        ),
                                        "same_reference_token_verified": True,
                                        "old_active_cache_tokens": (
                                            old_probe.active_cache_tokens
                                        ),
                                        "fresh_active_cache_tokens": (
                                            fresh_probe.active_cache_tokens
                                        ),
                                    }
                                )
                                set_rows.extend(
                                    self._set_rows(
                                        common,
                                        reference.anchors[base_anchor],
                                        current_anchor,
                                        old_selection,
                                        fresh_selection,
                                    )
                                )
                                feature_rows.extend(
                                    self._feature_rows(
                                        common,
                                        reference.anchors[base_anchor],
                                        current_anchor,
                                        old_selection,
                                        fresh_selection,
                                        old_probe,
                                        fresh_probe,
                                        cache_cfg,
                                    )
                                )
                                labels, identities = (
                                    self._attention_and_identity_rows(
                                        common,
                                        reference,
                                        current_anchor,
                                        old_probe,
                                        fresh_probe,
                                    )
                                )
                                attention_rows.extend(labels)
                                identity_rows.extend(identities)
                                full_logits = reference.probe_logits.get(
                                    old_probe.target_index
                                )
                                if full_logits is None:
                                    raise RuntimeError(
                                        "full reference logits are missing"
                                    )
                                downstream_rows.append(
                                    {
                                        **common,
                                        "target_index": (
                                            old_probe.target_index
                                        ),
                                        "target_token_id": (
                                            old_probe.target_token_id
                                        ),
                                        **_distribution_metrics(
                                            full_logits,
                                            old_probe.logits,
                                            fresh_probe.logits,
                                            old_probe.target_token_id,
                                            self.cfg.metrics.probability_floor,
                                        ),
                                    }
                                )
            if hasattr(self, "_feature_pair_cache"):
                del self._feature_pair_cache
                del self._feature_pair_cache_key
                self.model.release()
            inventory = pd.DataFrame(
                [
                    {
                        **self._base(sample),
                        "prompt_length": int(reference.prompt_length),
                        "generated_length": generated,
                        "captured_anchor_count": len(reference.anchors),
                        "captured_anchor_steps": json_text(
                            sorted(reference.anchors)
                        ),
                        "probe_logit_count": len(reference.probe_logits),
                        "selected_diagnostic_layers": json_text(
                            reference.selected_layers
                        ),
                        "selected_diagnostic_heads": json_text(
                            reference.selected_heads
                        ),
                        "prompt_truncated": bool(
                            reference.prompt_truncated
                        ),
                        "generation_stopped_on_eos": bool(
                            reference.generation_stopped_on_eos
                        ),
                        "generation_time_s": float(
                            reference.generation_time_s
                        ),
                        "functional_time_s": float(
                            time.perf_counter() - sample_started
                        ),
                        "sample_metadata": json_text(sample.metadata),
                    }
                ]
            )
            tables = {
                "reference_inventory": inventory,
                "probe_index": pd.DataFrame(probe_rows),
                "set_metrics": pd.DataFrame(set_rows),
                "functional_features": pd.DataFrame(feature_rows),
                "attention_labels": pd.DataFrame(attention_rows),
                "downstream_labels": pd.DataFrame(downstream_rows),
                "identity_checks": pd.DataFrame(identity_rows),
                "runtime_costs": pd.DataFrame(runtime_rows),
            }
            self._write_tables(sample.sample_id, tables)
            self.store.mark_complete(
                completion_key,
                {
                    "valid": True,
                    "comparisons": len(probe_rows),
                    "feature_rows": len(feature_rows),
                    "elapsed_s": float(
                        time.perf_counter() - sample_started
                    ),
                },
            )
        except Exception as exc:
            self.store.append_error(
                {
                    **self._base(sample),
                    "key": completion_key,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "time": time.time(),
                }
            )
            self.store.mark_failed(
                completion_key,
                "%s: %s" % (type(exc).__name__, exc),
            )
            if self.cfg.runtime.fail_on_error:
                raise
        finally:
            if reference is not None:
                self.model.release(reference)
