"""Physical action collection for output-sensitivity closure experiments."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import time
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import torch

from statekv.backend import QueryRecord, ReferenceTrajectory
from statekv.config import DiscoveryConfig
from statekv.functional_probe import _condition_cache
from statekv.robust_envelope import (
    RobustEnvelopeRunner,
    physical_shared_mask,
)
from statekv.runner import _sample_slug
from statekv.selectors import (
    CoreSelection,
    CoreSelector,
    LayerSelection,
    mandatory_and_eligible,
)
from statekv.tasks import load_discovery_tasks
from statekv.theory_closing import _atomic_frame
from statekv.trajectory_model import exact_distribution_metrics


OUTPUT_RAW_TABLES = (
    "output_candidate_rows",
    "output_candidate_inventory",
    "output_jacobian_probe_rows",
)


def candidate_budget_equal(selections: Sequence[CoreSelection]) -> bool:
    if not selections:
        return True
    signatures = [
        tuple(
            sorted(
                (int(layer), len(value.selected_positions))
                for layer, value in selection.by_layer.items()
            )
        )
        for selection in selections
    ]
    return len(set(signatures)) == 1


def no_task_feature(columns: Sequence[str]) -> bool:
    forbidden = {"task", "task_id", "sequence_id", "sample_id"}
    return not bool(forbidden & {str(value) for value in columns})


class OutputSensitivityRunner(RobustEnvelopeRunner):
    """Collect expanded physical actions and residual-to-logit probes."""

    @property
    def output_cfg(self) -> Any:
        return self.cfg.output_sensitivity

    def run(self) -> Path:
        if not self.output_cfg.enabled:
            raise ValueError("output_sensitivity.enabled must be true")
        self.store.status["state"] = "running"
        self.store.status["protocol"] = "output_sensitivity_closure_v1"
        self.store.save_status()
        samples, task_events = load_discovery_tasks(self.cfg)
        model_info = self.model.load()
        self.metadata = self.store.write_metadata(model_info, task_events)
        for table in OUTPUT_RAW_TABLES:
            (
                self.store.run_dir
                / "fragments"
                / "output_sensitivity"
                / table
            ).mkdir(parents=True, exist_ok=True)
        try:
            for sample_index, sample in enumerate(samples):
                self._run_output_sample(sample, sample_index)
            self._consolidate_output()
            self.store.status["state"] = "data_complete_analysis_pending"
            self.store.save_status()
        finally:
            self.model.close()
        return self.store.run_dir

    def _fragment(self, table: str, sample_id: str) -> Path:
        return (
            self.store.run_dir
            / "fragments"
            / "output_sensitivity"
            / table
            / ("%s.parquet" % _sample_slug(sample_id))
        )

    def _consolidate_output(self) -> None:
        for table in OUTPUT_RAW_TABLES:
            paths = sorted(
                (
                    self.store.run_dir
                    / "fragments"
                    / "output_sensitivity"
                    / table
                ).glob("*.parquet")
            )
            frames = [pd.read_parquet(path) for path in paths]
            frame = (
                pd.concat(frames, ignore_index=True, sort=False)
                if frames
                else pd.DataFrame()
            )
            _atomic_frame(frame, self.store.run_dir / ("%s.parquet" % table))

    def _completed_fragment_valid(self, sample_id: str) -> bool:
        paths = {
            table: self._fragment(table, sample_id)
            for table in OUTPUT_RAW_TABLES
        }
        if not all(path.exists() for path in paths.values()):
            return False
        try:
            inventory = pd.read_parquet(
                paths["output_candidate_inventory"]
            )
        except Exception:
            return False
        expected = int(self.output_cfg.candidate_count)
        counts = inventory.groupby("anchor")["candidate_id"].nunique()
        unique_masks = inventory.groupby("anchor")["mask_hash"].nunique()
        return bool(
            len(counts) == len(self.output_cfg.anchors)
            and counts.eq(expected).all()
            and unique_masks.eq(expected).all()
        )

    def _selector_at(self, reference: ReferenceTrajectory, anchor: int) -> CoreSelector:
        cache_cfg = _condition_cache(
            self.cfg,
            int(self.output_cfg.total_budget),
            int(self.output_cfg.protected_recent),
        )
        condition = replace(self.cfg, cache=cache_cfg)
        return CoreSelector(condition)

    @staticmethod
    def _seed(*parts: Any) -> int:
        token = ":".join(str(value) for value in parts)
        return int.from_bytes(
            hashlib.sha256(token.encode("utf-8")).digest()[:8],
            "little",
        )

    def _custom_selection(
        self,
        reference: ReferenceTrajectory,
        anchor: int,
        baseline: CoreSelection,
        source: str,
        variant: int,
        sample_index: int,
    ) -> CoreSelection:
        result = copy.deepcopy(baseline)
        state = reference.anchors[int(anchor)]
        diagnostic = set(self.model.selected_layers)
        for layer, base_layer in result.by_layer.items():
            positions = [
                int(value)
                for value in state.position_maps[int(layer)].tolist()
            ]
            sink, recent, eligible = mandatory_and_eligible(
                positions,
                int(self.cfg.cache.sink_size),
                int(self.output_cfg.protected_recent),
            )
            core_budget = int(self.output_cfg.total_budget) - len(
                set(sink + recent)
            )
            if int(layer) in diagnostic:
                bundle = self._score_bundle(reference, int(anchor), int(layer))
                row = {
                    int(position): index
                    for index, position in enumerate(positions)
                }
                attention = np.asarray(
                    [float(bundle["attention"][row[value]]) for value in eligible]
                )
                aov = np.asarray(
                    [float(bundle["aov"][row[value]]) for value in eligible]
                )
                aor = np.asarray(
                    [float(bundle["aor"][row[value]]) for value in eligible]
                )
                ridge = np.asarray(
                    [float(bundle["v_ridge"][row[value]]) for value in eligible]
                )
            else:
                attention = np.ones(len(eligible), dtype=np.float64)
                aov = np.ones(len(eligible), dtype=np.float64)
                aor = np.ones(len(eligible), dtype=np.float64)
                aggregate = np.asarray(
                    base_layer.aggregate_scores, dtype=np.float64
                )
                ridge = (
                    aggregate[: len(eligible)]
                    if len(aggregate) >= len(eligible)
                    else np.ones(len(eligible), dtype=np.float64)
                )
                ridge = np.nan_to_num(ridge, nan=1.0)
            rng = np.random.default_rng(
                self._seed(
                    reference.sample_id,
                    anchor,
                    layer,
                    source,
                    variant,
                    sample_index,
                    self.output_cfg.random_seed,
                )
            )
            selected: List[int]
            score: np.ndarray
            if source == "aov":
                score = aov
                selected = [
                    eligible[index]
                    for index in np.argsort(-score, kind="stable")[:core_budget]
                ]
            elif source == "aor":
                score = aor
                selected = [
                    eligible[index]
                    for index in np.argsort(-score, kind="stable")[:core_budget]
                ]
            elif source == "direct_energy_greedy":
                score = aov + aor
                selected = [
                    eligible[index]
                    for index in np.argsort(-score, kind="stable")[:core_budget]
                ]
            elif source == "recent_aware":
                age = np.linspace(0.0, 1.0, max(1, len(eligible)))
                score = ridge * (1.0 + age)
                selected = [
                    eligible[index]
                    for index in np.argsort(-score, kind="stable")[:core_budget]
                ]
            elif source == "stratified_random":
                bins = np.array_split(np.arange(len(eligible)), 4)
                chosen: List[int] = []
                target = int(math.ceil(core_budget / 4.0))
                for current in bins:
                    if len(current):
                        picked = rng.choice(
                            current,
                            size=min(target, len(current)),
                            replace=False,
                        )
                        chosen.extend(int(value) for value in picked.tolist())
                if len(chosen) < core_budget:
                    remaining = sorted(set(range(len(eligible))) - set(chosen))
                    extra = rng.choice(
                        np.asarray(remaining),
                        size=min(core_budget - len(chosen), len(remaining)),
                        replace=False,
                    )
                    chosen.extend(int(value) for value in extra.tolist())
                selected = [eligible[index] for index in chosen[:core_budget]]
                score = np.ones(len(eligible))
            elif source == "high_attention_random":
                score = np.maximum(attention, 0.0) + 1e-12
                probability = score / score.sum()
                chosen = rng.choice(
                    np.arange(len(eligible)),
                    size=min(core_budget, len(eligible)),
                    replace=False,
                    p=probability,
                )
                selected = [eligible[int(index)] for index in chosen.tolist()]
            elif source == "low_attention_high_value":
                score = (ridge + np.sqrt(np.maximum(aov + aor, 0.0))) / (
                    attention + 1e-4
                )
                probability = np.maximum(score, 0.0) + 1e-12
                probability = probability / probability.sum()
                chosen = rng.choice(
                    np.arange(len(eligible)),
                    size=min(core_budget, len(eligible)),
                    replace=False,
                    p=probability,
                )
                selected = [eligible[int(index)] for index in chosen.tolist()]
            else:
                raise ValueError("unknown custom candidate source: %s" % source)
            result.by_layer[int(layer)] = LayerSelection(
                layer=int(layer),
                selected_positions=sorted(int(value) for value in selected),
                eligible_positions=list(eligible),
                aggregate_scores=[float(value) for value in score.tolist()],
                metadata={
                    "physical_shared_mask": True,
                    "per_query_head_selection": False,
                    "source": source,
                    "variant": int(variant),
                },
            )
        result.strategy = "%s_%02d" % (source, int(variant))
        return result

    def _old_core_selection(
        self,
        reference: ReferenceTrajectory,
        anchor: int,
        baseline: CoreSelection,
    ) -> CoreSelection:
        previous_candidates = [
            value
            for value in [0] + list(self.output_cfg.anchors)
            if int(value) < int(anchor)
        ]
        previous = max(previous_candidates)
        old = self._selector_at(reference, previous).select(
            reference.anchors[previous].snapshot(reference.sample_id),
            "v_ridge_leverage",
        )
        result = copy.deepcopy(baseline)
        current_state = reference.anchors[int(anchor)]
        for layer, base in result.by_layer.items():
            positions = [
                int(value)
                for value in current_state.position_maps[int(layer)].tolist()
            ]
            sink, recent, eligible = mandatory_and_eligible(
                positions,
                int(self.cfg.cache.sink_size),
                int(self.output_cfg.protected_recent),
            )
            core_budget = int(self.output_cfg.total_budget) - len(
                set(sink + recent)
            )
            inherited = [
                int(value)
                for value in old.by_layer[int(layer)].selected_positions
                if int(value) in set(eligible)
            ]
            fill = [
                int(value)
                for value in base.selected_positions
                if int(value) in set(eligible)
                and int(value) not in set(inherited)
            ]
            selected = (inherited + fill)[:core_budget]
            result.by_layer[int(layer)] = LayerSelection(
                layer=int(layer),
                selected_positions=sorted(selected),
                eligible_positions=list(eligible),
                aggregate_scores=list(base.aggregate_scores),
                metadata={
                    "physical_shared_mask": True,
                    "per_query_head_selection": False,
                    "source": "preceding_anchor_old_core",
                    "preceding_anchor": int(previous),
                },
            )
        result.strategy = "preceding_anchor_old_core"
        return result

    def _candidate_selections(
        self,
        reference: ReferenceTrajectory,
        anchor: int,
        sample_index: int,
    ) -> List[Tuple[str, CoreSelection]]:
        selector = self._selector_at(reference, anchor)
        snapshot = reference.anchors[int(anchor)].snapshot(reference.sample_id)
        v_ridge = selector.select(snapshot, "v_ridge_leverage")
        attention = selector.select(snapshot, "snapkv")
        attention.strategy = "attention"
        candidates: List[Tuple[str, CoreSelection]] = [
            ("attention", attention),
            ("v_ridge", v_ridge),
            (
                "aov",
                self._custom_selection(
                    reference, anchor, v_ridge, "aov", 0, sample_index
                ),
            ),
            (
                "aor",
                self._custom_selection(
                    reference, anchor, v_ridge, "aor", 0, sample_index
                ),
            ),
            (
                "direct_energy_greedy",
                self._custom_selection(
                    reference,
                    anchor,
                    v_ridge,
                    "direct_energy_greedy",
                    0,
                    sample_index,
                ),
            ),
            (
                "preceding_anchor_old_core",
                self._old_core_selection(reference, anchor, v_ridge),
            ),
            (
                "recent_aware",
                self._custom_selection(
                    reference,
                    anchor,
                    v_ridge,
                    "recent_aware",
                    0,
                    sample_index,
                ),
            ),
        ]
        for source, count in (
            ("stratified_random", 8),
            ("high_attention_random", 5),
            ("low_attention_high_value", 4),
        ):
            for variant in range(count):
                candidates.append(
                    (
                        "%s_%02d" % (source, variant),
                        self._custom_selection(
                            reference,
                            anchor,
                            v_ridge,
                            source,
                            variant,
                            sample_index,
                        ),
                    )
                )
        # Required selectors can occasionally coincide physically (for
        # example, AOR and direct-energy may induce the same ranking).  Keep
        # both named mechanisms but apply a deterministic one-token
        # diversity tie-break to the later action so the registered pool
        # still contains 24 distinct physical subsets.
        seen: Set[str] = set()
        distinct: List[Tuple[str, CoreSelection]] = []
        for source, selection in candidates:
            masks = {
                str(layer): sorted(
                    int(value)
                    for value in layer_selection.selected_positions
                )
                for layer, layer_selection in selection.by_layer.items()
            }
            signature = json.dumps(masks, sort_keys=True)
            if signature in seen:
                repaired = copy.deepcopy(selection)
                repaired_ok = False
                for layer in reversed(sorted(repaired.by_layer)):
                    current = repaired.by_layer[int(layer)]
                    selected = sorted(
                        int(value) for value in current.selected_positions
                    )
                    outside = sorted(
                        set(int(value) for value in current.eligible_positions)
                        - set(selected)
                    )
                    for removed in selected:
                        for added in outside:
                            proposal = sorted(
                                (set(selected) - {removed}) | {added}
                            )
                            trial_masks = dict(masks)
                            trial_masks[str(layer)] = proposal
                            trial_signature = json.dumps(
                                trial_masks, sort_keys=True
                            )
                            if trial_signature not in seen:
                                current.selected_positions = proposal
                                current.metadata[
                                    "candidate_diversity_tiebreak"
                                ] = True
                                current.metadata[
                                    "diversity_removed_position"
                                ] = int(removed)
                                current.metadata[
                                    "diversity_added_position"
                                ] = int(added)
                                masks = trial_masks
                                signature = trial_signature
                                repaired_ok = True
                                break
                        if repaired_ok:
                            break
                    if repaired_ok:
                        break
                if not repaired_ok:
                    raise RuntimeError(
                        "could not make candidate pool physically distinct"
                    )
                selection = repaired
            seen.add(signature)
            distinct.append((source, selection))
        candidates = distinct
        if len(candidates) != int(self.output_cfg.candidate_count):
            raise RuntimeError("candidate registry does not match pre-registration")
        selections = [value for _, value in candidates]
        if not candidate_budget_equal(selections):
            raise RuntimeError("candidate core budgets differ")
        if not all(physical_shared_mask(value) for value in selections):
            raise RuntimeError("candidate pool contains a non-physical mask")
        return candidates

    def _operating_features(
        self,
        record: QueryRecord,
        full_logits: torch.Tensor,
    ) -> Dict[str, float]:
        probability = torch.softmax(full_logits.float(), dim=-1)
        log_probability = torch.log_softmax(full_logits.float(), dim=-1)
        entropy = float(
            -(probability * log_probability).sum().item()
        )
        top = torch.topk(full_logits.float(), k=2).values
        margin = float((top[0] - top[1]).item())
        head_entropies = []
        maxima = []
        top8_mass = []
        for layer in self.model.selected_layers:
            attention = record.all_head_attention_distributions[
                int(layer)
            ].float()
            head_entropies.append(
                float(
                    -(
                        attention
                        * torch.log(attention.clamp_min(1e-30))
                    )
                    .sum(dim=1)
                    .mean()
                    .item()
                )
            )
            maxima.append(float(attention.max(dim=1).values.mean().item()))
            top8_mass.append(
                float(
                    torch.topk(
                        attention,
                        k=min(8, int(attention.shape[1])),
                        dim=1,
                    ).values.sum(dim=1).mean().item()
                )
            )
        return {
            "output_entropy": entropy,
            "top1_top2_margin": margin,
            "inverse_logit_margin": 1.0 / max(margin, 1e-6),
            "attention_entropy": float(np.mean(head_entropies)),
            "attention_concentration": float(np.mean(maxima)),
            "attention_top8_mass": float(np.mean(top8_mass)),
            "prefix_length": float(record.query_position + 1),
            "current_hidden_norm": float(
                record.residual_inputs[27].float().norm().item()
            ),
            "current_projected_output_norm": float(
                record.projected_attention_outputs[27].float().norm().item()
            ),
        }

    def _inventory_row(
        self,
        sample: Any,
        reference: ReferenceTrajectory,
        anchor: int,
        candidate_index: int,
        source: str,
        selection: CoreSelection,
    ) -> Dict[str, Any]:
        state = reference.anchors[int(anchor)]
        masks = {
            str(layer): sorted(
                int(value)
                for value in selection.by_layer[int(layer)].selected_positions
            )
            for layer in selection.by_layer
        }
        ages = []
        for layer in self.model.selected_layers:
            ages.extend(
                int(state.logical_length - 1 - value)
                for value in masks[str(layer)]
            )
        mask_text = json.dumps(masks, sort_keys=True)
        return {
            **self._base(sample),
            "anchor": int(anchor),
            "candidate_index": int(candidate_index),
            "candidate_id": "%s:a%d:c%02d" % (
                reference.sample_id,
                int(anchor),
                int(candidate_index),
            ),
            "candidate_source": source,
            "physical_layer_shared_mask": True,
            "gqa_shared": True,
            "total_budget": int(self.output_cfg.total_budget),
            "protected_recent": int(self.output_cfg.protected_recent),
            "selected_core_budget": int(
                self.output_cfg.total_budget
                - self.cfg.cache.sink_size
                - self.output_cfg.protected_recent
            ),
            "selected_positions_json": mask_text,
            "mask_hash": hashlib.sha256(mask_text.encode("utf-8")).hexdigest(),
            "mean_selected_age": float(np.mean(ages)),
            "maximum_selected_age": int(max(ages)),
            "uses_future_compressed_truth": False,
            "uses_task_feature": False,
        }

    def _initial_values(
        self, reference: ReferenceTrajectory, anchor: int
    ) -> Dict[int, torch.Tensor]:
        return {
            int(layer): reference.anchors[int(anchor)].values[int(layer)][
                0
            ].detach().float().cpu().clone()
            for layer in self.model.selected_layers
        }

    def _replay_candidate(
        self,
        sample: Any,
        reference: ReferenceTrajectory,
        anchor: int,
        selection: CoreSelection,
        candidate_id: str,
        source: str,
        candidate_index: int,
        horizon: int,
        trajectory_kind: str = "candidate",
    ) -> List[Dict[str, Any]]:
        cache_cfg = _condition_cache(
            self.cfg,
            int(self.output_cfg.total_budget),
            int(self.output_cfg.protected_recent),
        )
        state, fixed = self.model.state_from_anchor(
            reference.anchors[int(anchor)],
            selection,
            cache_config=cache_cfg,
        )
        full_values = self._initial_values(reference, anchor)
        current_token = int(reference.anchors[int(anchor)].query_token_id)
        rows: List[Dict[str, Any]] = []
        try:
            for offset in range(1, int(horizon) + 1):
                target_index = int(anchor + offset - 1)
                reference_record = reference.query_records[target_index]
                if offset > 1:
                    self.model.prune_recent_before_query(
                        state, fixed, cache_config=cache_cfg
                    )
                    self._append_reference_value(full_values, reference_record)
                current_position = int(reference_record.query_position)
                direct_by_layer: Dict[int, Dict[str, Any]] = {}
                for layer in self.model.selected_layers:
                    retained = set(
                        int(value)
                        for value in state.position_maps[int(layer)].tolist()
                    )
                    retained.add(current_position)
                    direct_by_layer[int(layer)] = self._direct_at_step(
                        reference_record,
                        int(layer),
                        full_values[int(layer)],
                        sorted(retained),
                    )
                self._clear_controls()
                logits, record, forward_s = self.model.forward_one(
                    state, current_token, capture_attention=True
                )
                self.model.validate_active_budget(
                    state, cache_config=cache_cfg
                )
                target_token = int(reference.generated_token_ids[target_index])
                metrics = exact_distribution_metrics(
                    reference.probe_logits[target_index],
                    logits,
                    target_token,
                )
                features = self._operating_features(
                    reference_record,
                    reference.probe_logits[target_index],
                )
                kv_heads = int(self.model.model_info["num_key_value_heads"])
                for layer in self.model.selected_layers:
                    query_error = (
                        self._record_queries(
                            record, layer, self.model.selected_heads[layer]
                        )
                        - self._record_queries(
                            reference_record,
                            layer,
                            self.model.selected_heads[layer],
                        )
                    ).norm()
                    key_error = (
                        self._record_keys(record, layer, kv_heads)
                        - self._record_keys(
                            reference_record, layer, kv_heads
                        )
                    ).norm()
                    value_error = (
                        self._record_values(record, layer, kv_heads)
                        - self._record_values(
                            reference_record, layer, kv_heads
                        )
                    ).norm()
                    rows.append(
                        {
                            **self._base(sample),
                            "trajectory_id": candidate_id,
                            "trajectory_kind": trajectory_kind,
                            "candidate_id": candidate_id,
                            "candidate_index": int(candidate_index),
                            "candidate_source": source,
                            "anchor": int(anchor),
                            "horizon_offset": int(offset),
                            "target_index": int(target_index),
                            "layer": int(layer),
                            "residual_error": float(
                                (
                                    record.residual_inputs[int(layer)].float()
                                    - reference_record.residual_inputs[
                                        int(layer)
                                    ].float()
                                ).norm().item()
                            ),
                            "layer_output_error": float(
                                (
                                    record.layer_outputs[int(layer)].float()
                                    - reference_record.layer_outputs[
                                        int(layer)
                                    ].float()
                                ).norm().item()
                            ),
                            "query_error": float(query_error.item()),
                            "new_key_error": float(key_error.item()),
                            "new_value_error": float(value_error.item()),
                            "direct_coordinate": float(
                                direct_by_layer[int(layer)]["coordinate"]
                            ),
                            "deleted_attention_mass": float(
                                direct_by_layer[int(layer)][
                                    "deleted_attention_mass"
                                ]
                            ),
                            "exact_kl": float(metrics["exact_kl"]),
                            "js": float(metrics["js"]),
                            "full_nll": float(metrics["full_nll"]),
                            "perturbed_nll": float(
                                metrics["perturbed_nll"]
                            ),
                            "delta_nll": float(metrics["delta_nll"]),
                            "logit_l2_sq": float(metrics["logit_l2_sq"]),
                            "fisher_quadratic": float(
                                metrics["fisher_quadratic"]
                            ),
                            "projected_output_error": float(
                                (
                                    record.projected_attention_outputs[
                                        int(layer)
                                    ].float()
                                    - reference_record.projected_attention_outputs[
                                        int(layer)
                                    ].float()
                                ).square().sum().item()
                            ),
                            "active_cache_tokens": int(
                                self.model.active_cache_tokens(state)
                            ),
                            "total_budget": int(
                                self.output_cfg.total_budget
                            ),
                            "protected_recent": int(
                                self.output_cfg.protected_recent
                            ),
                            "refresh_event": False,
                            "uses_future_compressed_truth": False,
                            "token_position_aligned": bool(
                                int(record.query_position)
                                == int(reference_record.query_position)
                            ),
                            "forward_time_s": float(forward_s),
                            **features,
                        }
                    )
                current_token = target_token
        finally:
            self._clear_controls()
            self.model.release(state)
        return rows

    def _jacobian_probes(
        self,
        sample: Any,
        reference: ReferenceTrajectory,
    ) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []
        for anchor in self.output_cfg.anchors:
            anchor_state = reference.anchors[int(anchor)]
            full_selection = self._all_history_selection(reference, int(anchor))
            full_cache = _condition_cache(
                self.cfg,
                int(anchor_state.logical_length + 2),
                1,
            )
            current_token = int(anchor_state.query_token_id)
            full_logits = reference.probe_logits[int(anchor)].float()
            reference_record = reference.query_records[int(anchor)]
            for layer in self.model.selected_layers:
                residual = reference_record.residual_inputs[int(layer)].float()
                residual_norm = max(float(residual.norm().item()), 1e-12)
                for direction_index in range(
                    int(self.output_cfg.jacobian_directions)
                ):
                    rng = np.random.default_rng(
                        self._seed(
                            reference.sample_id,
                            anchor,
                            layer,
                            direction_index,
                            self.output_cfg.random_seed,
                        )
                    )
                    direction = torch.from_numpy(
                        rng.choice(
                            np.asarray([-1.0, 1.0], dtype=np.float32),
                            size=int(residual.numel()),
                        )
                    ).reshape_as(residual)
                    direction = direction / direction.norm().clamp_min(1e-12)
                    for radius in self.output_cfg.jacobian_radii:
                        step = float(radius) * residual_norm
                        logits_by_sign: Dict[int, torch.Tensor] = {}
                        delta_by_sign: Dict[int, float] = {}
                        for sign in (-1, 1):
                            state, _ = self.model.state_from_anchor(
                                anchor_state,
                                full_selection,
                                cache_config=full_cache,
                            )
                            try:
                                self._clear_controls()
                                self.model.runner.attention_state[
                                    "temporal_layer_input_overrides"
                                ] = {
                                    int(layer): (
                                        residual + sign * step * direction
                                    ).numpy()
                                }
                                logits, _, _ = self.model.forward_one(
                                    state,
                                    current_token,
                                    capture_attention=True,
                                )
                                logits_by_sign[int(sign)] = logits.float()
                                delta_by_sign[int(sign)] = float(
                                    (logits.float() - full_logits).norm().item()
                                )
                            finally:
                                self._clear_controls()
                                self.model.release(state)
                        symmetric = float(
                            (
                                logits_by_sign[1] - logits_by_sign[-1]
                            ).norm().item()
                            / max(2.0 * step, 1e-12)
                        )
                        result.append(
                            {
                                **self._base(sample),
                                "probe_id": (
                                    "%s:a%d:l%d:d%02d:r%.6f"
                                    % (
                                        reference.sample_id,
                                        int(anchor),
                                        int(layer),
                                        int(direction_index),
                                        float(radius),
                                    )
                                ),
                                "anchor": int(anchor),
                                "layer": int(layer),
                                "direction_index": int(direction_index),
                                "relative_radius": float(radius),
                                "absolute_step_l2": float(step),
                                "residual_input_norm": residual_norm,
                                "symmetric_directional_gain": symmetric,
                                "plus_logit_l2": delta_by_sign[1],
                                "minus_logit_l2": delta_by_sign[-1],
                                "radius_symmetry_error": float(
                                    abs(
                                        delta_by_sign[1]
                                        - delta_by_sign[-1]
                                    )
                                    / max(
                                        delta_by_sign[1]
                                        + delta_by_sign[-1],
                                        1e-12,
                                    )
                                ),
                                "directions_per_operating_point": int(
                                    self.output_cfg.jacobian_directions
                                ),
                                "finite_difference_symmetric": True,
                                "claimed_operator_norm": False,
                                "uses_compressed_future_truth": False,
                            }
                        )
        return result

    def _run_output_sample(self, sample: Any, sample_index: int) -> None:
        key = "output_sensitivity:%s" % _sample_slug(sample.sample_id)
        if (
            self.cfg.runtime.resume
            and self.store.is_complete(key)
            and self._completed_fragment_valid(sample.sample_id)
        ):
            return
        started = time.perf_counter()
        try:
            reference = self.model.generate_reference(
                sample.sample_id, sample.task, sample.prompt
            )
            rows: List[Dict[str, Any]] = []
            inventory: List[Dict[str, Any]] = []
            for anchor in self.output_cfg.anchors:
                candidates = self._candidate_selections(
                    reference, int(anchor), sample_index
                )
                for candidate_index, (source, selection) in enumerate(
                    candidates
                ):
                    inventory_row = self._inventory_row(
                        sample,
                        reference,
                        int(anchor),
                        candidate_index,
                        source,
                        selection,
                    )
                    inventory.append(inventory_row)
                    rows.extend(
                        self._replay_candidate(
                            sample,
                            reference,
                            int(anchor),
                            selection,
                            str(inventory_row["candidate_id"]),
                            source,
                            candidate_index,
                            int(self.output_cfg.segment_horizon),
                        )
                    )
            reference_anchor = int(
                self.output_cfg.state_reference_anchor
            )
            selector = self._selector_at(reference, reference_anchor)
            state_selection = selector.select(
                reference.anchors[reference_anchor].snapshot(
                    reference.sample_id
                ),
                "v_ridge_leverage",
            )
            rows.extend(
                self._replay_candidate(
                    sample,
                    reference,
                    reference_anchor,
                    state_selection,
                    "%s:state_reference" % reference.sample_id,
                    "v_ridge_state_reference",
                    -1,
                    int(self.output_cfg.state_reference_horizon),
                    trajectory_kind="state_reference",
                )
            )
            probes = self._jacobian_probes(sample, reference)
            _atomic_frame(
                pd.DataFrame(rows),
                self._fragment("output_candidate_rows", sample.sample_id),
            )
            _atomic_frame(
                pd.DataFrame(inventory),
                self._fragment(
                    "output_candidate_inventory", sample.sample_id
                ),
            )
            _atomic_frame(
                pd.DataFrame(probes),
                self._fragment(
                    "output_jacobian_probe_rows", sample.sample_id
                ),
            )
            self.store.mark_complete(
                key,
                {
                    "elapsed_s": float(time.perf_counter() - started),
                    "candidate_rows": len(rows),
                    "inventory_rows": len(inventory),
                    "jacobian_probe_rows": len(probes),
                },
            )
            self.model.release(reference)
        except Exception as exc:
            self.store.append_error(
                {
                    "key": key,
                    "sample_id": sample.sample_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            self.store.mark_failed(
                key, "%s: %s" % (type(exc).__name__, exc)
            )
            if self.cfg.runtime.fail_on_error:
                raise
