"""End-to-end temporal cache discovery runner."""
from __future__ import annotations

import hashlib
import json
import math
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch

from statekv.artifacts import ArtifactStore, json_text
from statekv.backend import (
    ReferenceTrajectory,
    TemporalModel,
    peak_process_rss_bytes,
)
from statekv.config import DiscoveryConfig
from statekv.metrics import (
    approximate_kl,
    attention_output_relative_errors,
    loss_shape,
    validity_observations,
)
from statekv.selectors import CoreSelection, CoreSelector, selection_overlap
from statekv.signals import (
    candidate_layer_records,
    geometry_rows,
    query_attention_rows,
    score_drift_rows,
)
from statekv.tasks import load_discovery_tasks


def _sample_slug(sample_id: str) -> str:
    digest = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:10]
    readable = "".join(
        character if character.isalnum() else "_" for character in sample_id
    )[:48]
    return "%s_%s" % (readable, digest)


class TemporalDiscoveryRunner:
    def __init__(
        self,
        cfg: DiscoveryConfig,
        repository_root: Path,
    ):
        self.cfg = cfg
        self.repository_root = repository_root.resolve()
        self.store = ArtifactStore(cfg, self.repository_root)
        if cfg.model.backend == "mlx":
            from statekv.backend_mlx import MLXTemporalModel

            self.model = MLXTemporalModel(cfg)
        else:
            self.model = TemporalModel(cfg)
        self.selector = CoreSelector(cfg)
        self.metadata: Dict[str, Any] = {}

    def run(self) -> Path:
        self.store.status["state"] = "running"
        self.store.save_status()
        samples, task_events = load_discovery_tasks(self.cfg)
        model_info = self.model.load()
        self.metadata = self.store.write_metadata(model_info, task_events)
        try:
            for sample in samples:
                self._run_sample(sample)
            outputs = self.store.consolidate()
            self.store.status["state"] = "complete"
            self.store.status["parquet_outputs"] = {
                key: str(value) for key, value in outputs.items()
            }
            self.store.save_status()
        finally:
            self.model.close()
        return self.store.run_dir

    def _base(self, sample: Any) -> Dict[str, Any]:
        return {
            "run_id": self.store.run_dir.name,
            "model": self.cfg.model.name,
            "task": sample.task,
            "sample_id": sample.sample_id,
            "seed": int(self.cfg.runtime.seed),
            "config_hash": self.cfg.config_hash,
            "git_commit": self.metadata.get("git_commit"),
        }

    def _expected_combo_keys(self, sample_id: str) -> List[str]:
        slug = _sample_slug(sample_id)
        return [
            "combo:%s:a%d:%s:h%d" % (slug, anchor, strategy, horizon)
            for anchor in self.cfg.anchor_steps
            for strategy in self.cfg.strategies
            for horizon in self.cfg.horizons
        ]

    def _run_sample(self, sample: Any) -> None:
        combo_keys = self._expected_combo_keys(sample.sample_id)
        if self.cfg.runtime.resume and all(
            self.store.is_complete(key) for key in combo_keys
        ):
            return
        slug = _sample_slug(sample.sample_id)
        reference_key = "reference:%s" % slug
        try:
            reference = self.model.generate_reference(
                sample.sample_id, sample.task, sample.prompt
            )
            reference_path = self.store.save_reference_npz(reference)
            row = {
                **self._base(sample),
                "prompt_length": int(reference.prompt_length),
                "generated_length": len(reference.generated_token_ids),
                "reference_npz_path": str(reference_path),
                "selected_diagnostic_layers": json_text(reference.selected_layers),
                "selected_diagnostic_heads": json_text(reference.selected_heads),
                "prompt_truncated": bool(reference.prompt_truncated),
                "generation_stopped_on_eos": bool(
                    reference.generation_stopped_on_eos
                ),
                "generation_time_s": float(reference.generation_time_s),
                "peak_rss_bytes": int(reference.peak_rss_bytes),
                "peak_accelerator_bytes": reference.peak_accelerator_bytes,
                "sample_metadata": json_text(sample.metadata),
            }
            self.store.write_fragment("reference", reference_key, [row])
            self.store.mark_complete(
                reference_key,
                {"generated_length": len(reference.generated_token_ids)},
            )
        except Exception as exc:
            self._record_failure(reference_key, sample, exc)
            if self.cfg.runtime.fail_on_error:
                raise
            for key in combo_keys:
                self.store.mark_failed(key, "reference_failed: %s" % exc)
            return

        for anchor_step in self.cfg.anchor_steps:
            self._run_anchor(sample, reference, int(anchor_step))
        self.model.release(reference)

    def _run_anchor(
        self,
        sample: Any,
        reference: ReferenceTrajectory,
        anchor_step: int,
    ) -> None:
        slug = _sample_slug(sample.sample_id)
        base = self._base(sample)
        anchor = reference.anchors.get(anchor_step)
        remaining = len(reference.generated_token_ids) - anchor_step
        if anchor is None:
            reason = "reference_trajectory_does_not_reach_anchor"
            for strategy in self.cfg.strategies:
                for horizon in self.cfg.horizons:
                    key = "combo:%s:a%d:%s:h%d" % (
                        slug,
                        anchor_step,
                        strategy,
                        horizon,
                    )
                    self._write_invalid_combo(
                        key, base, anchor_step, strategy, horizon, reason
                    )
            return
        snapshot = anchor.snapshot(sample.sample_id)
        deployable: Dict[str, CoreSelection] = {}
        try:
            for strategy in self.cfg.strategies:
                if strategy == "future_attention_oracle":
                    continue
                deployable[strategy] = self.selector.select(snapshot, strategy)
        except Exception as exc:
            key = "selection:%s:a%d" % (slug, anchor_step)
            self._record_failure(key, sample, exc)
            if self.cfg.runtime.fail_on_error:
                raise
            for strategy in self.cfg.strategies:
                for horizon in self.cfg.horizons:
                    combo = "combo:%s:a%d:%s:h%d" % (
                        slug,
                        anchor_step,
                        strategy,
                        horizon,
                    )
                    self.store.mark_failed(combo, "selection_failed: %s" % exc)
            return

        oracle: Dict[int, CoreSelection] = {}
        for horizon in self.cfg.horizons:
            if remaining < horizon:
                continue
            try:
                future = self.model.future_attention(reference, anchor_step, horizon)
                oracle[horizon] = self.selector.select(
                    snapshot,
                    "future_attention_oracle",
                    future_attention=future,
                    horizon=horizon,
                )
            except Exception as exc:
                key = "oracle-selection:%s:a%d:h%d" % (
                    slug,
                    anchor_step,
                    horizon,
                )
                self._record_failure(key, sample, exc)
                if self.cfg.runtime.fail_on_error:
                    raise

        all_named: Dict[str, CoreSelection] = dict(deployable)
        all_named.update(
            {
                "future_attention_oracle@%d" % horizon: selection
                for horizon, selection in oracle.items()
            }
        )
        self._write_candidates(
            sample, reference, anchor_step, all_named
        )

        try:
            temporal_rows = score_drift_rows(
                reference,
                deployable,
                anchor_step,
                self.cfg.signal_lags,
                self.cfg,
                base,
            )
            temporal_rows += query_attention_rows(
                reference,
                deployable,
                anchor_step,
                self.cfg.signal_lags,
                base,
                self.cfg.cache.selected_core_budget,
            )
            geometry, residual_means = geometry_rows(
                reference,
                deployable,
                anchor_step,
                base,
                self.cfg.cache.sink_size,
                self.cfg.cache.recent_size,
            )
            temporal_rows += geometry
            temporal_key = "signals:%s:a%d" % (slug, anchor_step)
            self.store.write_fragment("temporal", temporal_key, temporal_rows)
            self.store.mark_complete(temporal_key, {"rows": len(temporal_rows)})
        except Exception as exc:
            residual_means = {}
            key = "signals:%s:a%d" % (slug, anchor_step)
            self._record_failure(key, sample, exc)
            if self.cfg.runtime.fail_on_error:
                raise

        for strategy in self.cfg.strategies:
            for horizon in self.cfg.horizons:
                key = "combo:%s:a%d:%s:h%d" % (
                    slug,
                    anchor_step,
                    strategy,
                    horizon,
                )
                if self.cfg.runtime.resume and self.store.is_complete(key):
                    continue
                if remaining < horizon:
                    self._write_invalid_combo(
                        key,
                        base,
                        anchor_step,
                        strategy,
                        horizon,
                        "insufficient_reference_tokens_after_anchor",
                    )
                    continue
                selection = (
                    oracle.get(horizon)
                    if strategy == "future_attention_oracle"
                    else deployable.get(strategy)
                )
                if selection is None:
                    self.store.mark_failed(key, "candidate_selection_unavailable")
                    continue
                try:
                    step_rows, horizon_row, loss_signal = self._replay(
                        sample,
                        reference,
                        anchor_step,
                        strategy,
                        horizon,
                        selection,
                        residual_means,
                    )
                    self.store.write_fragment("step", key, step_rows)
                    self.store.write_fragment("horizon", key, [horizon_row])
                    self.store.write_fragment(
                        "temporal", "loss-shape:" + key, [loss_signal]
                    )
                    self.store.mark_complete(
                        key,
                        {
                            "valid": True,
                            "step_rows": len(step_rows),
                            "max_active_cache": horizon_row[
                                "max_active_cache_tokens"
                            ],
                        },
                    )
                except Exception as exc:
                    self._record_failure(key, sample, exc)
                    if self.cfg.runtime.fail_on_error:
                        raise
                    self.model.release()

    def _write_candidates(
        self,
        sample: Any,
        reference: ReferenceTrajectory,
        anchor_step: int,
        selections: Mapping[str, CoreSelection],
    ) -> None:
        anchor = reference.anchors[anchor_step]
        slug = _sample_slug(sample.sample_id)
        token_ids = (
            reference.prompt_token_ids
            + reference.generated_token_ids[:anchor_step]
        )
        for name, selection in selections.items():
            overlaps = {
                other_name: selection_overlap(selection, other)
                for other_name, other in selections.items()
                if other_name != name
            }
            core_sizes = [
                len(layer.selected_positions)
                for layer in selection.by_layer.values()
            ]
            all_ages = [
                int(anchor.logical_length - 1 - position)
                for layer in selection.by_layer.values()
                for position in layer.selected_positions
            ]
            row = {
                **self._base(sample),
                "anchor": int(anchor_step),
                "strategy": selection.strategy,
                "horizon_condition": selection.horizon_condition,
                "valid": True,
                "invalid_reason": None,
                "total_budget": int(self.cfg.cache.total_budget),
                "sink_size": int(self.cfg.cache.sink_size),
                "recent_size": int(self.cfg.cache.recent_size),
                "selected_core_budget": int(
                    self.cfg.cache.selected_core_budget
                ),
                "effective_core_size_min": min(core_sizes) if core_sizes else 0,
                "effective_core_size_max": max(core_sizes) if core_sizes else 0,
                "selected_token_age_mean": (
                    float(np.mean(all_ages)) if all_ages else None
                ),
                "selected_token_age_median": (
                    float(np.median(all_ages)) if all_ages else None
                ),
                "selected_token_age_min": min(all_ages) if all_ages else None,
                "selected_token_age_max": max(all_ages) if all_ages else None,
                "layers": json_text(
                    candidate_layer_records(
                        selection,
                        anchor,
                        token_ids,
                        self.cfg.cache.sink_size,
                        self.cfg.cache.recent_size,
                    )
                ),
                "overlaps": json_text(overlaps),
                "selection_metadata": json_text(selection.metadata),
            }
            condition = (
                "none"
                if selection.horizon_condition is None
                else str(selection.horizon_condition)
            )
            key = "candidate:%s:a%d:%s:h%s" % (
                slug,
                anchor_step,
                name,
                condition,
            )
            self.store.write_fragment("candidate", key, [row])
            self.store.mark_complete(key, {"valid": True})

    def _replay(
        self,
        sample: Any,
        reference: ReferenceTrajectory,
        anchor_step: int,
        strategy: str,
        horizon: int,
        selection: CoreSelection,
        residual_means: Mapping[Tuple[str, int], float],
        compute_oracle_overlap: bool = True,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
        anchor = reference.anchors[anchor_step]
        state, fixed = self.model.state_from_anchor(anchor, selection)
        current_token = int(anchor.query_token_id)
        rows = []
        max_active = 0
        replay_started = time.perf_counter()
        for future_step in range(1, int(horizon) + 1):
            if future_step > 1:
                self.model.prune_recent_before_query(state, fixed)
            logits, diagnostic, forward_s = self.model.forward_one(
                state, current_token, capture_attention=True
            )
            self.model.validate_active_budget(state)
            max_active = max(max_active, self.model.active_cache_tokens(state))
            target_index = int(anchor_step + future_step - 1)
            target_token = int(reference.generated_token_ids[target_index])
            compressed_log_probs = torch.log_softmax(logits.detach().float(), dim=-1)
            compressed_log_probability = float(
                compressed_log_probs[target_token].item()
            )
            reference_log_probability = float(
                reference.reference_log_probabilities[target_index]
            )
            delta_nll = (
                -compressed_log_probability + reference_log_probability
            )
            kl = approximate_kl(
                reference.top_ids[target_index],
                reference.top_probabilities[target_index],
                logits,
                floor=self.cfg.metrics.probability_floor,
            )
            full_record = reference.query_records[target_index]
            attention_errors = attention_output_relative_errors(
                full_record.attention_outputs,
                diagnostic.attention_outputs,
                epsilon=self.cfg.metrics.attention_error_epsilon,
            )
            attention_error_values = [
                record["relative_error"] for record in attention_errors
            ]
            if not math.isfinite(delta_nll):
                raise FloatingPointError("delta NLL is NaN/Inf")
            row = {
                **self._base(sample),
                "anchor": int(anchor_step),
                "strategy": strategy,
                "target_horizon": int(horizon),
                "future_step": int(future_step),
                "valid": True,
                "invalid_reason": None,
                "reference_token_id": target_token,
                "reference_token_position": int(
                    reference.prompt_length + target_index
                ),
                "reference_log_probability": reference_log_probability,
                "compressed_log_probability": compressed_log_probability,
                "delta_nll": float(delta_nll),
                **kl,
                "attention_output_error_mean": (
                    float(np.mean(attention_error_values))
                    if attention_error_values
                    else None
                ),
                "attention_output_error_max": (
                    float(np.max(attention_error_values))
                    if attention_error_values
                    else None
                ),
                "attention_output_errors": json_text(attention_errors),
                "active_cache_tokens": self.model.active_cache_tokens(state),
                "forward_time_s": float(forward_s),
                "new_token_value_residual_mean": residual_means.get(
                    (strategy, future_step)
                ),
            }
            rows.append(row)
            current_token = target_token
        replay_s = time.perf_counter() - replay_started
        delta = np.asarray([row["delta_nll"] for row in rows], dtype=np.float64)
        kl_values = np.asarray([row["approx_kl"] for row in rows], dtype=np.float64)
        attention_values = np.asarray(
            [
                row["attention_output_error_mean"]
                for row in rows
                if row["attention_output_error_mean"] is not None
            ],
            dtype=np.float64,
        )
        shape = loss_shape(
            delta.tolist(), self.cfg.metrics.large_loss_spike_threshold
        )
        for index, row in enumerate(rows):
            row.update(
                {
                    "cumulative_delta_nll": shape["cumulative"][index],
                    "average_delta_nll": shape["running_average"][index],
                    "running_max_delta_nll": shape["running_max"][index],
                    "delta_nll_slope": shape["slope"][index],
                    "delta_nll_curvature": shape["curvature"][index],
                    "first_large_loss_spike": shape[
                        "first_large_loss_spike"
                    ],
                    "change_point": shape["change_point"],
                }
            )
        thresholds = {
            "avg_delta_nll": self.cfg.validity_thresholds.avg_delta_nll,
            "max_delta_nll": self.cfg.validity_thresholds.max_delta_nll,
            "avg_approx_kl": self.cfg.validity_thresholds.avg_approx_kl,
        }
        validity = validity_observations(
            rows, thresholds, max_measured_horizon=horizon
        )
        oracle_overlap = None
        if compute_oracle_overlap and strategy != "future_attention_oracle":
            try:
                future = self.model.future_attention(
                    reference, anchor_step, horizon
                )
                oracle = self.selector.select(
                    anchor.snapshot(sample.sample_id),
                    "future_attention_oracle",
                    future_attention=future,
                    horizon=horizon,
                )
                oracle_overlap = selection_overlap(selection, oracle)[
                    "mean_jaccard"
                ]
            except Exception:
                oracle_overlap = None
        margins = [
            layer.boundary_margin
            for layer in selection.by_layer.values()
            if layer.boundary_margin is not None
        ]
        horizon_row = {
            **self._base(sample),
            "anchor": int(anchor_step),
            "strategy": strategy,
            "horizon": int(horizon),
            "valid": True,
            "invalid_reason": None,
            "sum_delta_nll": float(delta.sum()),
            "avg_delta_nll": float(delta.mean()),
            "max_delta_nll": float(delta.max()),
            "sum_approx_kl": float(kl_values.sum()),
            "avg_approx_kl": float(kl_values.mean()),
            "max_approx_kl": float(kl_values.max()),
            "attention_output_error_mean": (
                float(attention_values.mean()) if attention_values.size else None
            ),
            "attention_output_error_max": (
                float(attention_values.max()) if attention_values.size else None
            ),
            "validity_horizons": json_text(validity),
            "oracle_overlap": oracle_overlap,
            "selected_core_size_mean": float(
                np.mean(
                    [
                        len(layer.selected_positions)
                        for layer in selection.by_layer.values()
                    ]
                )
            ),
            "selection_boundary_margin_mean": (
                float(np.mean(margins)) if margins else None
            ),
            "max_active_cache_tokens": int(max_active),
            "replay_time_s": float(replay_s),
            "peak_rss_bytes": int(peak_process_rss_bytes()),
            "peak_accelerator_bytes": self.model._peak_accelerator_memory(),
        }
        loss_signal = {
            **self._base(sample),
            "anchor": int(anchor_step),
            "strategy": strategy,
            "signal_kind": "loss_temporal_shape",
            "layer": None,
            "head": None,
            "lag": int(horizon),
            "target_horizon": int(horizon),
            "cumulative_delta_nll": json_text(shape["cumulative"]),
            "average_delta_nll": json_text(shape["running_average"]),
            "running_max_delta_nll": json_text(shape["running_max"]),
            "delta_nll_slope": json_text(shape["slope"]),
            "delta_nll_curvature": json_text(shape["curvature"]),
            "first_large_loss_spike": shape["first_large_loss_spike"],
            "change_point": shape["change_point"],
            "change_point_method": shape["change_point_method"],
        }
        self.model.release(state)
        return rows, horizon_row, loss_signal

    def _write_invalid_combo(
        self,
        key: str,
        base: Dict[str, Any],
        anchor: int,
        strategy: str,
        horizon: int,
        reason: str,
    ) -> None:
        step_row = {
            **base,
            "anchor": int(anchor),
            "strategy": strategy,
            "target_horizon": int(horizon),
            "future_step": None,
            "valid": False,
            "invalid_reason": reason,
        }
        horizon_row = {
            **base,
            "anchor": int(anchor),
            "strategy": strategy,
            "horizon": int(horizon),
            "valid": False,
            "invalid_reason": reason,
        }
        self.store.write_fragment("step", key, [step_row])
        self.store.write_fragment("horizon", key, [horizon_row])
        self.store.mark_complete(key, {"valid": False, "reason": reason})

    def _record_failure(self, key: str, sample: Any, exc: Exception) -> None:
        record = {
            **self._base(sample),
            "key": key,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "time": time.time(),
        }
        self.store.append_error(record)
        self.store.mark_failed(key, "%s: %s" % (type(exc).__name__, exc))
