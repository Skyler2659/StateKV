"""Physical compressed-cache trajectories for robust envelope analysis."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import time
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from statekv.backend import QueryRecord, ReferenceTrajectory
from statekv.config import DiscoveryConfig
from statekv.functional_probe import _condition_cache
from statekv.runner import _sample_slug
from statekv.selectors import (
    CoreSelection,
    CoreSelector,
    LayerSelection,
    mandatory_and_eligible,
)
from statekv.tasks import load_discovery_tasks
from statekv.theory_closing import _atomic_frame
from statekv.trajectory_model import (
    TrajectoryModelRunner,
    exact_distribution_metrics,
)


ROBUST_TABLES = (
    "robust_trajectory_rows",
    "envelope_subset_inventory",
    "architecture_gain_probe_rows",
)


def block_triangular_mask(
    output_layers: Sequence[int],
    input_layers: Sequence[int],
) -> np.ndarray:
    """Architecture-order mask: an output sees only same/earlier layers."""

    return np.asarray(
        [
            [int(source) <= int(target) for source in input_layers]
            for target in output_layers
        ],
        dtype=bool,
    )


def refresh_preserves_error(
    current_error: np.ndarray, refreshed_direct_input: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Refresh changes future input, never the accumulated state error."""

    return (
        np.asarray(current_error, dtype=np.float64).copy(),
        np.asarray(refreshed_direct_input, dtype=np.float64).copy(),
    )


def physical_shared_mask(selection: CoreSelection) -> bool:
    return all(
        bool(layer.metadata.get("per_query_head_selection", False)) is False
        for layer in selection.by_layer.values()
    )


class RobustEnvelopeRunner(TrajectoryModelRunner):
    """Collect static, subset, and directional-gain trajectories."""

    @property
    def envelope_cfg(self) -> Any:
        return self.cfg.robust_envelope

    def run(self) -> Path:
        if not self.cfg.robust_envelope.enabled:
            raise ValueError("robust_envelope.enabled must be true")
        self.store.status["state"] = "running"
        self.store.status["protocol"] = "robust_trajectory_envelope_v1"
        self.store.save_status()
        samples, task_events = load_discovery_tasks(self.cfg)
        model_info = self.model.load()
        self.metadata = self.store.write_metadata(model_info, task_events)
        for table in ROBUST_TABLES:
            (
                self.store.run_dir
                / "fragments"
                / "robust_envelope"
                / table
            ).mkdir(parents=True, exist_ok=True)
        try:
            for sample_index, sample in enumerate(samples):
                self._run_envelope_sample(sample, sample_index)
            self._consolidate()
            self.store.status["state"] = "data_complete_analysis_pending"
            self.store.save_status()
        finally:
            self.model.close()
        return self.store.run_dir

    def _robust_fragment_path(self, table: str, sample_id: str) -> Path:
        return (
            self.store.run_dir
            / "fragments"
            / "robust_envelope"
            / table
            / ("%s.parquet" % _sample_slug(sample_id))
        )

    def _consolidate(self) -> None:
        for table in ROBUST_TABLES:
            paths = sorted(
                (
                    self.store.run_dir
                    / "fragments"
                    / "robust_envelope"
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

    def _selection(
        self,
        reference: ReferenceTrajectory,
        strategy: str,
        sample_index: int,
        subset_index: int = 0,
    ) -> CoreSelection:
        cfg = self.envelope_cfg
        anchor = int(cfg.anchor)
        cache_cfg = _condition_cache(
            self.cfg, int(cfg.total_budget), int(cfg.protected_recent)
        )
        condition = replace(self.cfg, cache=cache_cfg)
        selector = CoreSelector(condition)
        snapshot = reference.anchors[anchor].snapshot(reference.sample_id)
        if strategy == "attention":
            return selector.select(snapshot, "snapkv")
        if strategy == "v_ridge":
            return selector.select(snapshot, "v_ridge_leverage")
        baseline = selector.select(snapshot, "v_ridge_leverage")
        if strategy in {"aov", "aor"}:
            result = copy.deepcopy(baseline)
            for layer in self.model.selected_layers:
                state = reference.anchors[anchor]
                positions = [
                    int(value)
                    for value in state.position_maps[layer].tolist()
                ]
                sink, recent, eligible = mandatory_and_eligible(
                    positions,
                    int(cache_cfg.sink_size),
                    int(cache_cfg.recent_size),
                )
                core_budget = (
                    int(cache_cfg.total_budget)
                    - len(set(sink + recent))
                )
                scores = self._score_bundle(
                    reference, anchor, int(layer)
                )[strategy]
                row = {
                    position: index
                    for index, position in enumerate(positions)
                }
                ordered = sorted(
                    eligible,
                    key=lambda position: (
                        -float(scores[row[position]].item()),
                        int(position),
                    ),
                )
                result.by_layer[layer] = LayerSelection(
                    layer=int(layer),
                    selected_positions=ordered[:core_budget],
                    eligible_positions=eligible,
                    aggregate_scores=[
                        float(value) for value in scores.tolist()
                    ],
                    metadata={
                        "physical_shared_mask": True,
                        "per_query_head_selection": False,
                        "score": strategy,
                    },
                )
            result.strategy = strategy
            return result
        if strategy in {"random", "candidate_subset"}:
            result = copy.deepcopy(baseline)
            for layer, layer_selection in result.by_layer.items():
                state = reference.anchors[anchor]
                positions = [
                    int(value)
                    for value in state.position_maps[layer].tolist()
                ]
                sink, recent, eligible = mandatory_and_eligible(
                    positions,
                    int(cache_cfg.sink_size),
                    int(cache_cfg.recent_size),
                )
                core_budget = (
                    int(cache_cfg.total_budget)
                    - len(set(sink + recent))
                )
                token = "%s:%d:%d:%d:%d" % (
                    reference.sample_id,
                    int(layer),
                    int(sample_index),
                    int(subset_index),
                    int(cfg.random_seed),
                )
                seed = int.from_bytes(
                    hashlib.sha256(token.encode("utf-8")).digest()[:8],
                    "little",
                )
                rng = np.random.default_rng(seed)
                selected = sorted(
                    int(value)
                    for value in rng.choice(
                        np.asarray(eligible, dtype=np.int64),
                        size=min(core_budget, len(eligible)),
                        replace=False,
                    ).tolist()
                )
                result.by_layer[layer] = LayerSelection(
                    layer=int(layer),
                    selected_positions=selected,
                    eligible_positions=eligible,
                    aggregate_scores=[float("nan")] * len(positions),
                    metadata={
                        "physical_shared_mask": True,
                        "per_query_head_selection": False,
                        "seed": int(seed),
                    },
                )
            result.strategy = (
                "candidate_subset_%02d" % subset_index
                if strategy == "candidate_subset"
                else "random"
            )
            return result
        raise ValueError("unknown robust strategy: %s" % strategy)

    def _initial_full_values(
        self, reference: ReferenceTrajectory, anchor: int
    ) -> Dict[int, torch.Tensor]:
        return {
            int(layer): value[0].detach().float().cpu().clone()
            for layer, value in enumerate(reference.anchors[anchor].values)
            if int(layer) in set(self.model.selected_layers)
        }

    def _append_reference_value(
        self,
        full_values: Dict[int, torch.Tensor],
        record: QueryRecord,
    ) -> None:
        kv_heads = int(self.model.model_info["num_key_value_heads"])
        for layer in full_values:
            new_value = self._record_values(
                record, int(layer), kv_heads
            ).float()[:, None, :]
            full_values[layer] = torch.cat(
                [full_values[layer], new_value], dim=1
            )

    def _direct_at_step(
        self,
        record: QueryRecord,
        layer: int,
        values: torch.Tensor,
        retained_rows: Sequence[int],
    ) -> Dict[str, Any]:
        attention = record.all_head_attention_distributions[layer].float()
        if int(attention.shape[1]) != int(values.shape[1]):
            raise RuntimeError(
                "reference value/attention alignment failed at layer=%d" % layer
            )
        rows = torch.as_tensor(list(retained_rows), dtype=torch.long)
        query_heads = int(attention.shape[0])
        kv_heads = int(values.shape[0])
        repeated = values.repeat_interleave(query_heads // kv_heads, dim=0)
        kept_attention = attention.index_select(1, rows)
        denominator = kept_attention.sum(dim=1).clamp_min(1e-12)
        masked = (
            kept_attention[:, :, None] * repeated.index_select(1, rows)
        ).sum(dim=1) / denominator[:, None]
        full = record.all_head_attention_outputs[layer].float()
        projected = self.model.project_features(
            int(layer), (masked - full).reshape(1, -1)
        )[0]
        deleted = torch.ones(int(attention.shape[1]), dtype=torch.bool)
        deleted[rows] = False
        return {
            "coordinate": float(projected.norm().item()),
            "projected": projected,
            "deleted_attention_mass": float(
                attention[:, deleted].sum(dim=1).mean().item()
            ),
        }

    def _candidate_anchor_scores(
        self,
        reference: ReferenceTrajectory,
        selection: CoreSelection,
    ) -> Dict[str, float]:
        anchor = int(self.envelope_cfg.anchor)
        state = reference.anchors[anchor]
        output = defaultdict_float()
        for layer in self.model.selected_layers:
            positions = [
                int(value)
                for value in state.position_maps[layer].tolist()
            ]
            selected = set(
                selection.by_layer[layer].selected_positions
            )
            sink = set(positions[: int(self.cfg.cache.sink_size)])
            recent = set(
                positions[-int(self.envelope_cfg.protected_recent) :]
            )
            retained = sink | recent | selected
            deleted_rows = [
                index
                for index, position in enumerate(positions)
                if position not in retained
            ]
            bundle = self._score_bundle(reference, anchor, int(layer))
            for name in ("attention", "aov", "aor"):
                output[name] += float(
                    bundle[name][deleted_rows].sum().item()
                )
            direct = self._direct_at_step(
                reference.query_records[anchor],
                int(layer),
                state.values[layer][0].float(),
                [
                    index
                    for index, position in enumerate(positions)
                    if position in retained
                ],
            )
            output["direct_only"] += direct["coordinate"] ** 2
        return dict(output)

    def _replay_compressed(
        self,
        sample: Any,
        reference: ReferenceTrajectory,
        selection: CoreSelection,
        trajectory_id: str,
        kind: str,
        horizon: int,
        subset_index: int = -1,
    ) -> List[Dict[str, Any]]:
        cfg = self.envelope_cfg
        anchor = int(cfg.anchor)
        cache_cfg = _condition_cache(
            self.cfg, int(cfg.total_budget), int(cfg.protected_recent)
        )
        state, fixed = self.model.state_from_anchor(
            reference.anchors[anchor],
            selection,
            cache_config=cache_cfg,
        )
        full_values = self._initial_full_values(reference, anchor)
        current_token = int(reference.anchors[anchor].query_token_id)
        rows: List[Dict[str, Any]] = []
        try:
            for offset in range(1, int(horizon) + 1):
                target_index = anchor + offset - 1
                reference_record = reference.query_records[target_index]
                if offset > 1:
                    self.model.prune_recent_before_query(
                        state, fixed, cache_config=cache_cfg
                    )
                    self._append_reference_value(
                        full_values, reference_record
                    )
                current_position = int(reference_record.query_position)
                direct_by_layer: Dict[int, Dict[str, Any]] = {}
                for layer in self.model.selected_layers:
                    full_length = int(full_values[layer].shape[1])
                    retained_positions = set(
                        int(value)
                        for value in state.position_maps[layer].tolist()
                    )
                    retained_positions.add(current_position)
                    retained_rows = [
                        position
                        for position in sorted(retained_positions)
                        if 0 <= position < full_length
                    ]
                    direct_by_layer[layer] = self._direct_at_step(
                        reference_record,
                        int(layer),
                        full_values[layer],
                        retained_rows,
                    )
                self._clear_controls()
                logits, record, forward_s = self.model.forward_one(
                    state, current_token, capture_attention=True
                )
                self.model.validate_active_budget(
                    state, cache_config=cache_cfg
                )
                target_token = int(
                    reference.generated_token_ids[target_index]
                )
                metrics = exact_distribution_metrics(
                    reference.probe_logits[target_index],
                    logits,
                    target_token,
                )
                kv_heads = int(self.model.model_info["num_key_value_heads"])
                for layer in self.model.selected_layers:
                    query = (
                        self._record_queries(
                            record,
                            layer,
                            self.model.selected_heads[layer],
                        )
                        - self._record_queries(
                            reference_record,
                            layer,
                            self.model.selected_heads[layer],
                        )
                    )
                    key = (
                        self._record_keys(record, layer, kv_heads)
                        - self._record_keys(
                            reference_record, layer, kv_heads
                        )
                    )
                    value = (
                        self._record_values(record, layer, kv_heads)
                        - self._record_values(
                            reference_record, layer, kv_heads
                        )
                    )
                    rows.append(
                        {
                            **self._base(sample),
                            "trajectory_id": trajectory_id,
                            "trajectory_kind": kind,
                            "selector": selection.strategy,
                            "subset_index": int(subset_index),
                            "anchor": anchor,
                            "horizon_offset": int(offset),
                            "target_index": int(target_index),
                            "layer": int(layer),
                            "layer_order": int(
                                self.model.selected_layers.index(layer)
                            ),
                            "residual_error": float(
                                (
                                    record.residual_inputs[layer].float()
                                    - reference_record.residual_inputs[
                                        layer
                                    ].float()
                                )
                                .norm()
                                .item()
                            ),
                            "layer_output_error": float(
                                (
                                    record.layer_outputs[layer].float()
                                    - reference_record.layer_outputs[
                                        layer
                                    ].float()
                                )
                                .norm()
                                .item()
                            ),
                            "query_error": float(query.norm().item()),
                            "new_key_error": float(key.norm().item()),
                            "new_value_error": float(value.norm().item()),
                            "direct_coordinate": direct_by_layer[layer][
                                "coordinate"
                            ],
                            "deleted_attention_mass": direct_by_layer[layer][
                                "deleted_attention_mass"
                            ],
                            "exact_kl": metrics["exact_kl"],
                            "js": metrics["js"],
                            "delta_nll": metrics["delta_nll"],
                            "projected_output_error": float(
                                (
                                    record.projected_attention_outputs[
                                        layer
                                    ].float()
                                    - reference_record.projected_attention_outputs[
                                        layer
                                    ].float()
                                )
                                .square()
                                .sum()
                                .item()
                            ),
                            "active_cache_tokens": int(
                                self.model.active_cache_tokens(state)
                            ),
                            "protected_recent": int(
                                cfg.protected_recent
                            ),
                            "total_budget": int(cfg.total_budget),
                            "refresh_event": False,
                            "token_position_aligned": bool(
                                record.query_position
                                == reference_record.query_position
                            ),
                            "physical_layer_shared_mask": True,
                            "forward_time_s": float(forward_s),
                        }
                    )
                current_token = target_token
        finally:
            self._clear_controls()
            self.model.release(state)
        return rows

    def _architecture_probes(
        self,
        sample: Any,
        reference: ReferenceTrajectory,
        base_selection: CoreSelection,
    ) -> List[Dict[str, Any]]:
        cfg = self.envelope_cfg
        anchor = int(cfg.anchor)
        anchor_state = reference.anchors[anchor]
        cache_cfg = replace(
            self.cfg.cache,
            total_budget=int(anchor_state.logical_length + 8),
            sink_size=0,
            recent_size=1,
            selected_core_budget=int(anchor_state.logical_length + 7),
        )
        full_selection = self._all_history_selection(reference, anchor)
        result: List[Dict[str, Any]] = []
        injection_layers = [0, 7, 14, 21, 27]
        for injection_layer in injection_layers:
            positions = [
                int(value)
                for value in anchor_state.position_maps[
                    injection_layer
                ].tolist()
            ]
            retained = set(
                base_selection.by_layer[
                    injection_layer
                ].selected_positions
            )
            retained.update(
                positions[: int(self.cfg.cache.sink_size)]
            )
            retained.update(
                positions[-int(cfg.protected_recent) :]
            )
            direct = self._direct_at_step(
                reference.query_records[anchor],
                injection_layer,
                anchor_state.values[injection_layer][0].float(),
                [
                    index
                    for index, position in enumerate(positions)
                    if position in retained
                ],
            )
            injection = float(cfg.jacobian_beta) * direct["projected"]
            state, _ = self.model.state_from_anchor(
                anchor_state,
                full_selection,
                cache_config=cache_cfg,
            )
            current_token = int(anchor_state.query_token_id)
            try:
                for offset in (1, 2):
                    target_index = anchor + offset - 1
                    self._clear_controls()
                    if offset == 1:
                        self.model.runner.attention_state[
                            "temporal_projected_injections"
                        ] = {injection_layer: injection.numpy()}
                    _, record, _ = self.model.forward_one(
                        state, current_token, capture_attention=True
                    )
                    reference_record = reference.query_records[
                        target_index
                    ]
                    for response_layer in self.model.selected_layers:
                        result.append(
                            {
                                **self._base(sample),
                                "probe_id": "%s:l%d" % (
                                    reference.sample_id,
                                    injection_layer,
                                ),
                                "injection_layer": int(injection_layer),
                                "response_layer": int(response_layer),
                                "horizon_offset": int(offset),
                                "jacobian_beta": float(cfg.jacobian_beta),
                                "direct_input_coordinate": float(
                                    injection.norm().item()
                                ),
                                "residual_response": float(
                                    (
                                        record.residual_inputs[
                                            response_layer
                                        ].float()
                                        - reference_record.residual_inputs[
                                            response_layer
                                        ].float()
                                    )
                                    .norm()
                                    .item()
                                ),
                                "uses_compressed_future_truth": False,
                            }
                        )
                    current_token = int(
                        reference.generated_token_ids[target_index]
                    )
            finally:
                self._clear_controls()
                self.model.release(state)
        return result

    def _run_envelope_sample(self, sample: Any, sample_index: int) -> None:
        key = "robust_envelope:%s" % _sample_slug(sample.sample_id)
        if self.cfg.runtime.resume and self.store.is_complete(key):
            return
        started = time.perf_counter()
        try:
            reference = self.model.generate_reference(
                sample.sample_id, sample.task, sample.prompt
            )
            trajectory_rows: List[Dict[str, Any]] = []
            inventory_rows: List[Dict[str, Any]] = []
            selections: Dict[str, CoreSelection] = {}
            for strategy in self.envelope_cfg.static_strategies:
                selection = self._selection(
                    reference, strategy, sample_index
                )
                if not physical_shared_mask(selection):
                    raise RuntimeError("non-physical selection generated")
                selections[strategy] = selection
                trajectory_id = "%s:static:%s" % (
                    sample.sample_id,
                    strategy,
                )
                trajectory_rows.extend(
                    self._replay_compressed(
                        sample,
                        reference,
                        selection,
                        trajectory_id,
                        "static",
                        int(self.envelope_cfg.horizon),
                    )
                )
            for subset_index in range(
                int(self.envelope_cfg.subset_count)
            ):
                selection = self._selection(
                    reference,
                    "candidate_subset",
                    sample_index,
                    subset_index=subset_index,
                )
                trajectory_id = "%s:subset:%02d" % (
                    sample.sample_id,
                    subset_index,
                )
                scores = self._candidate_anchor_scores(
                    reference, selection
                )
                inventory_rows.append(
                    {
                        **self._base(sample),
                        "trajectory_id": trajectory_id,
                        "subset_index": int(subset_index),
                        "selector": selection.strategy,
                        "physical_layer_shared_mask": True,
                        "gqa_shared": True,
                        "budget": int(
                            self.envelope_cfg.total_budget
                        ),
                        "protected_recent": int(
                            self.envelope_cfg.protected_recent
                        ),
                        **{
                            "%s_objective" % name: float(value)
                            for name, value in scores.items()
                        },
                    }
                )
                trajectory_rows.extend(
                    self._replay_compressed(
                        sample,
                        reference,
                        selection,
                        trajectory_id,
                        "subset",
                        int(self.envelope_cfg.subset_horizon),
                        subset_index=subset_index,
                    )
                )
            probe_rows = self._architecture_probes(
                sample, reference, selections["v_ridge"]
            )
            _atomic_frame(
                pd.DataFrame(trajectory_rows),
                self._robust_fragment_path(
                    "robust_trajectory_rows", sample.sample_id
                ),
            )
            _atomic_frame(
                pd.DataFrame(inventory_rows),
                self._robust_fragment_path(
                    "envelope_subset_inventory", sample.sample_id
                ),
            )
            _atomic_frame(
                pd.DataFrame(probe_rows),
                self._robust_fragment_path(
                    "architecture_gain_probe_rows", sample.sample_id
                ),
            )
            self.store.mark_complete(
                key,
                {
                    "elapsed_s": float(
                        time.perf_counter() - started
                    ),
                    "trajectory_rows": len(trajectory_rows),
                    "subset_count": len(inventory_rows),
                    "probe_rows": len(probe_rows),
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


def defaultdict_float() -> Dict[str, float]:
    return {"attention": 0.0, "aov": 0.0, "aor": 0.0, "direct_only": 0.0}
