"""Minimal targeted experiment for online leverage and cache refresh."""
from __future__ import annotations

import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from statekv.artifacts import json_text
from statekv.backend import ReferenceTrajectory
from statekv.runner import TemporalDiscoveryRunner, _sample_slug
from statekv.selectors import (
    CoreSelection,
    fit_online_ridge_factor,
    mandatory_and_eligible,
)
from statekv.tasks import load_discovery_tasks


DEPLOYABLE = (
    "snapkv",
    "v_ridge_leverage",
    "attention_weighted_v_ridge_leverage",
)
MECHANISM_TABLES = (
    "reference_inventory",
    "online_leverage",
    "online_leverage_core_entry",
    "dense_refresh_counterfactuals",
    "refresh_set_rank_changes",
    "recent_window_exit_events",
)


def _atomic_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    os.close(fd)
    temporary_path = Path(temporary)
    try:
        if path.suffix == ".parquet":
            frame.to_parquet(temporary_path, index=False)
        elif path.suffix == ".csv":
            frame.to_csv(temporary_path, index=False)
        else:
            raise ValueError("unsupported frame output: %s" % path)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _jaccard(left: Iterable[int], right: Iterable[int]) -> float:
    a, b = set(left), set(right)
    union = a | b
    return float(len(a & b) / len(union)) if union else 1.0


def _rank(values: List[float], target_index: int) -> Optional[int]:
    array = np.asarray(values, dtype=np.float64)
    if target_index >= len(array) or not np.isfinite(array[target_index]):
        return None
    finite = np.flatnonzero(np.isfinite(array))
    order = finite[np.argsort(-array[finite], kind="stable")]
    location = np.flatnonzero(order == int(target_index))
    return int(location[0] + 1) if len(location) else None


def _selection_transition(
    base: CoreSelection,
    refreshed: CoreSelection,
    base_anchor: Any,
    refreshed_anchor: Any,
    base_step: int,
    refresh_lag: int,
    common: Dict[str, Any],
    probe_type: str,
) -> List[Dict[str, Any]]:
    rows = []
    for layer in sorted(set(base.by_layer) & set(refreshed.by_layer)):
        left = base.by_layer[layer]
        right = refreshed.by_layer[layer]
        left_positions = [
            int(value) for value in base_anchor.position_maps[layer].tolist()
        ]
        right_positions = [
            int(value) for value in refreshed_anchor.position_maps[layer].tolist()
        ]
        left_map = dict(zip(left_positions, left.aggregate_scores))
        right_map = dict(zip(right_positions, right.aggregate_scores))
        comparable = [
            position
            for position in left.eligible_positions
            if position in set(right.eligible_positions)
            and math.isfinite(float(left_map.get(position, float("nan"))))
            and math.isfinite(float(right_map.get(position, float("nan"))))
        ]
        if len(comparable) >= 2:
            left_score = pd.Series([left_map[position] for position in comparable])
            right_score = pd.Series([right_map[position] for position in comparable])
            spearman = float(left_score.corr(right_score, method="spearman"))
            pearson = float(left_score.corr(right_score, method="pearson"))
        else:
            spearman = np.nan
            pearson = np.nan
        left_selected = set(left.selected_positions)
        right_selected = set(right.selected_positions)
        intersection = len(left_selected & right_selected)
        new_selected = [
            position
            for position in right_selected
            if position >= int(base_anchor.logical_length)
        ]
        rows.append(
            {
                **common,
                "base_anchor": int(base_step),
                "refresh_lag": int(refresh_lag),
                "refresh_anchor": int(base_step + refresh_lag),
                "strategy": base.strategy,
                "layer": int(layer),
                "probe_type": probe_type,
                "old_token_score_spearman": spearman,
                "old_token_score_pearson": pearson,
                "selected_core_jaccard": _jaccard(
                    left.selected_positions, right.selected_positions
                ),
                "selected_core_retention": float(
                    intersection / max(1, len(left_selected))
                ),
                "selected_core_turnover": float(
                    1.0 - intersection / max(1, len(left_selected))
                ),
                "new_token_selected_count": int(len(new_selected)),
                "new_token_selected_fraction": float(
                    len(new_selected) / max(1, len(right_selected))
                ),
                "base_boundary_margin": left.boundary_margin,
                "refreshed_boundary_margin": right.boundary_margin,
                "common_old_token_count": int(len(comparable)),
            }
        )
    return rows


class MechanismTargetedRunner(TemporalDiscoveryRunner):
    """One mechanism-directed run; no future oracle or benchmark sweep."""

    def run(self) -> Path:
        if not self.cfg.mechanism.enabled:
            raise ValueError("mechanism.enabled must be true")
        self.store.status["state"] = "running"
        self.store.status["protocol"] = "mechanism_targeted_v1"
        self.store.save_status()
        samples, task_events = load_discovery_tasks(self.cfg)
        model_info = self.model.load()
        self.metadata = self.store.write_metadata(model_info, task_events)
        for table in MECHANISM_TABLES:
            (
                self.store.run_dir / "fragments" / "mechanism_targeted" / table
            ).mkdir(parents=True, exist_ok=True)
        try:
            for sample in samples:
                self._run_mechanism_sample(sample)
            outputs = self._consolidate_mechanism()
            self.store.status["state"] = "complete"
            self.store.status["mechanism_outputs"] = {
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
            / "mechanism_targeted"
            / table
            / ("%s.parquet" % _sample_slug(sample_id))
        )

    def _write_sample_tables(
        self, sample_id: str, tables: Mapping[str, pd.DataFrame]
    ) -> None:
        for table in MECHANISM_TABLES:
            frame = tables.get(table, pd.DataFrame())
            _atomic_frame(frame, self._fragment_path(table, sample_id))

    def _consolidate_mechanism(self) -> Dict[str, Path]:
        outputs = {}
        for table in MECHANISM_TABLES:
            fragments = sorted(
                (
                    self.store.run_dir
                    / "fragments"
                    / "mechanism_targeted"
                    / table
                ).glob("*.parquet")
            )
            frames = [pd.read_parquet(path) for path in fragments]
            frame = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
            parquet = self.store.run_dir / ("%s.parquet" % table)
            csv = self.store.run_dir / ("%s.csv" % table)
            _atomic_frame(frame, parquet)
            _atomic_frame(frame, csv)
            outputs[table] = parquet
        return outputs

    def _decode_token(self, token_id: int) -> Optional[str]:
        tokenizer = getattr(getattr(self.model, "runner", None), "tokenizer", None)
        if tokenizer is None:
            tokenizer = getattr(getattr(self.model, "backend", None), "tokenizer", None)
        if tokenizer is None:
            return None
        try:
            return str(tokenizer.decode([int(token_id)]))
        except Exception:
            return None

    def _online_leverage(
        self,
        sample: Any,
        reference: ReferenceTrajectory,
        base_step: int,
    ) -> pd.DataFrame:
        anchor = reference.anchors[base_step]
        maximum = min(
            max(self.cfg.mechanism.refresh_lags),
            len(reference.generated_token_ids) - base_step,
        )
        base = self._base(sample)
        rows = []
        for layer in reference.selected_layers:
            positions = [
                int(value) for value in anchor.position_maps[layer].tolist()
            ]
            _, _, eligible = mandatory_and_eligible(
                positions,
                self.cfg.cache.sink_size,
                self.cfg.cache.recent_size,
            )
            position_to_row = {
                int(position): index for index, position in enumerate(positions)
            }
            eligible_rows = torch.tensor(
                [position_to_row[position] for position in eligible],
                dtype=torch.long,
            )
            layer_values = anchor.values[layer].detach()[0].float().cpu()
            for kv_head in range(int(layer_values.shape[0])):
                histories = {
                    "full_history": layer_values[kv_head],
                    "selector_candidate_history": layer_values[
                        kv_head
                    ].index_select(0, eligible_rows),
                }
                factors = {
                    scope: fit_online_ridge_factor(
                        histories[scope],
                        self.cfg.selectors.ridge_lambda,
                        self.cfg.selectors.ridge_lambda_mode,
                    )
                    for scope in self.cfg.mechanism.online_leverage_scopes
                }
                vectors = []
                offsets = []
                for token_offset in range(1, maximum + 1):
                    record_index = base_step + token_offset
                    key = "%d:%d" % (layer, kv_head)
                    vector = reference.query_records[record_index].new_values.get(key)
                    if vector is None:
                        continue
                    vectors.append(vector)
                    offsets.append(token_offset)
                if not vectors:
                    continue
                stacked = torch.stack(vectors)
                scores = {
                    scope: factor.score(stacked).detach().cpu().numpy()
                    for scope, factor in factors.items()
                }
                for local, token_offset in enumerate(offsets):
                    target_index = base_step + token_offset - 1
                    token_id = int(reference.generated_token_ids[target_index])
                    position = int(anchor.logical_length + token_offset - 1)
                    for scope, factor in factors.items():
                        rows.append(
                            {
                                **base,
                                "base_anchor": int(base_step),
                                "token_offset": int(token_offset),
                                "token_position": position,
                                "token_id": token_id,
                                "token_text": self._decode_token(token_id),
                                "layer": int(layer),
                                "kv_head": int(kv_head),
                                "history_scope": scope,
                                "online_leverage": float(scores[scope][local]),
                                **factor.diagnostics,
                                "formula": (
                                    "v^T (V_anchor^T V_anchor + lambda I)^-1 v"
                                ),
                            }
                        )
        return pd.DataFrame(rows)

    def _select_deployable(
        self, reference: ReferenceTrajectory, step: int
    ) -> Dict[str, CoreSelection]:
        anchor = reference.anchors.get(int(step))
        if anchor is None:
            raise ValueError("captured reference anchor is missing at step=%d" % step)
        snapshot = anchor.snapshot(reference.sample_id)
        return {
            strategy: self.selector.select(snapshot, strategy)
            for strategy in DEPLOYABLE
        }

    def _run_mechanism_sample(self, sample: Any) -> None:
        slug = _sample_slug(sample.sample_id)
        key = "mechanism:%s" % slug
        if self.cfg.runtime.resume and self.store.is_complete(key):
            if all(
                self._fragment_path(table, sample.sample_id).exists()
                for table in MECHANISM_TABLES
            ):
                return
        reference: Optional[ReferenceTrajectory] = None
        started = time.perf_counter()
        try:
            reference = self.model.generate_reference(
                sample.sample_id, sample.task, sample.prompt
            )
            base = self._base(sample)
            required = set(self.cfg.captured_anchor_steps())
            available = set(reference.anchors)
            missing = sorted(
                step for step in required if step <= len(reference.generated_token_ids) and step not in available
            )
            if missing:
                raise RuntimeError("required mechanism anchors missing: %s" % missing)

            online_frames = [
                self._online_leverage(sample, reference, int(base_step))
                for base_step in self.cfg.mechanism.base_anchor_steps
            ]
            online = pd.concat(online_frames, ignore_index=True)
            online_aggregate = (
                online[
                    online["history_scope"].eq("selector_candidate_history")
                ]
                .groupby(
                    [
                        "sample_id",
                        "task",
                        "base_anchor",
                        "token_offset",
                        "token_position",
                        "token_id",
                        "token_text",
                        "layer",
                    ],
                    dropna=False,
                    as_index=False,
                )
                .agg(
                    online_leverage_mean=("online_leverage", "mean"),
                    online_leverage_max=("online_leverage", "max"),
                )
            )
            full_aggregate = (
                online[online["history_scope"].eq("full_history")]
                .groupby(
                    [
                        "sample_id",
                        "base_anchor",
                        "token_offset",
                        "layer",
                    ],
                    as_index=False,
                )
                .agg(
                    full_history_online_leverage_mean=("online_leverage", "mean"),
                    full_history_online_leverage_max=("online_leverage", "max"),
                )
            )
            online_aggregate = online_aggregate.merge(
                full_aggregate,
                on=["sample_id", "base_anchor", "token_offset", "layer"],
                how="left",
            )

            selection_cache: Dict[int, Dict[str, CoreSelection]] = {}

            def selections(step: int) -> Dict[str, CoreSelection]:
                if step not in selection_cache:
                    selection_cache[step] = self._select_deployable(reference, step)
                return selection_cache[step]

            stale_cache: Dict[Tuple[int, str], Dict[int, Dict[str, Any]]] = {}
            maximum_lag = max(self.cfg.mechanism.refresh_lags)
            for base_step in self.cfg.mechanism.base_anchor_steps:
                base_selections = selections(int(base_step))
                remaining = len(reference.generated_token_ids) - int(base_step)
                replay_horizon = min(int(maximum_lag + 1), int(remaining))
                for strategy in DEPLOYABLE:
                    step_rows, _, _ = self._replay(
                        sample,
                        reference,
                        int(base_step),
                        strategy,
                        replay_horizon,
                        base_selections[strategy],
                        {},
                        compute_oracle_overlap=False,
                    )
                    stale_cache[(int(base_step), strategy)] = {
                        int(row["future_step"]): row for row in step_rows
                    }

            fresh_cache: Dict[Tuple[int, str], Dict[str, Any]] = {}

            def fresh(step: int, strategy: str) -> Dict[str, Any]:
                cache_key = (int(step), strategy)
                if cache_key not in fresh_cache:
                    step_rows, _, _ = self._replay(
                        sample,
                        reference,
                        int(step),
                        strategy,
                        1,
                        selections(int(step))[strategy],
                        {},
                        compute_oracle_overlap=False,
                    )
                    fresh_cache[cache_key] = step_rows[0]
                return fresh_cache[cache_key]

            dense_rows: List[Dict[str, Any]] = []
            rank_rows: List[Dict[str, Any]] = []
            entry_rows: List[Dict[str, Any]] = []
            for base_step in self.cfg.mechanism.base_anchor_steps:
                base_step = int(base_step)
                anchor = reference.anchors[base_step]
                base_selections = selections(base_step)
                for refresh_lag in self.cfg.mechanism.refresh_lags:
                    refresh_lag = int(refresh_lag)
                    refresh_step = base_step + refresh_lag
                    if refresh_step not in reference.anchors:
                        continue
                    refreshed = selections(refresh_step)
                    transition_by_strategy: Dict[str, pd.DataFrame] = {}
                    for strategy in DEPLOYABLE:
                        stale = stale_cache[(base_step, strategy)].get(
                            refresh_lag + 1
                        )
                        if stale is None:
                            continue
                        refreshed_row = fresh(refresh_step, strategy)
                        same_token = (
                            stale["reference_token_id"]
                            == refreshed_row["reference_token_id"]
                            and stale["reference_token_position"]
                            == refreshed_row["reference_token_position"]
                        )
                        if not same_token:
                            raise RuntimeError(
                                "dense refresh comparison failed token alignment"
                            )
                        transition = _selection_transition(
                            base_selections[strategy],
                            refreshed[strategy],
                            anchor,
                            reference.anchors[refresh_step],
                            base_step,
                            refresh_lag,
                            base,
                            "dense_refresh_grid",
                        )
                        rank_rows.extend(transition)
                        transition_frame = pd.DataFrame(transition)
                        transition_by_strategy[strategy] = transition_frame
                        online_window = online[
                            online["base_anchor"].eq(base_step)
                            & online["token_offset"].le(refresh_lag)
                            & online["history_scope"].eq(
                                "selector_candidate_history"
                            )
                        ]
                        exited = online_window[
                            online_window["token_offset"].le(
                                refresh_lag - int(self.cfg.cache.recent_size)
                            )
                        ]
                        dense_rows.append(
                            {
                                **base,
                                "base_anchor": base_step,
                                "refresh_lag": refresh_lag,
                                "refresh_anchor": refresh_step,
                                "strategy": strategy,
                                "same_reference_token_verified": True,
                                "reference_token_id": stale["reference_token_id"],
                                "reference_token_position": stale[
                                    "reference_token_position"
                                ],
                                "stale_delta_nll": stale["delta_nll"],
                                "refreshed_delta_nll": refreshed_row["delta_nll"],
                                "refresh_benefit_delta_nll": (
                                    stale["delta_nll"]
                                    - refreshed_row["delta_nll"]
                                ),
                                "stale_approx_kl": stale["approx_kl"],
                                "refreshed_approx_kl": refreshed_row["approx_kl"],
                                "refresh_benefit_approx_kl": (
                                    stale["approx_kl"]
                                    - refreshed_row["approx_kl"]
                                ),
                                "stale_attention_output_error": stale[
                                    "attention_output_error_mean"
                                ],
                                "refreshed_attention_output_error": refreshed_row[
                                    "attention_output_error_mean"
                                ],
                                "new_token_online_leverage_mean": (
                                    float(online_window["online_leverage"].mean())
                                    if len(online_window)
                                    else np.nan
                                ),
                                "new_token_online_leverage_max": (
                                    float(online_window["online_leverage"].max())
                                    if len(online_window)
                                    else np.nan
                                ),
                                "exited_recent_online_leverage_max": (
                                    float(exited["online_leverage"].max())
                                    if len(exited)
                                    else np.nan
                                ),
                                "mean_old_token_score_spearman": float(
                                    transition_frame[
                                        "old_token_score_spearman"
                                    ].mean()
                                ),
                                "mean_selected_core_turnover": float(
                                    transition_frame[
                                        "selected_core_turnover"
                                    ].mean()
                                ),
                                "mean_new_token_selected_fraction": float(
                                    transition_frame[
                                        "new_token_selected_fraction"
                                    ].mean()
                                ),
                            }
                        )

                    refresh_anchor = reference.anchors[refresh_step]
                    online_at_base = online_aggregate[
                        online_aggregate["base_anchor"].eq(base_step)
                        & online_aggregate["token_offset"].le(refresh_lag)
                    ]
                    for row in online_at_base.itertuples(index=False):
                        for strategy in DEPLOYABLE:
                            layer_selection = refreshed[strategy].by_layer[int(row.layer)]
                            positions = [
                                int(value)
                                for value in refresh_anchor.position_maps[
                                    int(row.layer)
                                ].tolist()
                            ]
                            position_to_row = {
                                position: index
                                for index, position in enumerate(positions)
                            }
                            token_row = position_to_row.get(int(row.token_position))
                            eligible = int(row.token_position) in set(
                                layer_selection.eligible_positions
                            )
                            selected = int(row.token_position) in set(
                                layer_selection.selected_positions
                            )
                            rank = (
                                _rank(layer_selection.aggregate_scores, token_row)
                                if token_row is not None and eligible
                                else None
                            )
                            score = (
                                layer_selection.aggregate_scores[token_row]
                                if token_row is not None
                                and math.isfinite(
                                    float(
                                        layer_selection.aggregate_scores[
                                            token_row
                                        ]
                                    )
                                )
                                else None
                            )
                            entry_rows.append(
                                {
                                    **base,
                                    "base_anchor": base_step,
                                    "refresh_lag": refresh_lag,
                                    "refresh_anchor": refresh_step,
                                    "strategy": strategy,
                                    "layer": int(row.layer),
                                    "token_offset": int(row.token_offset),
                                    "token_position": int(row.token_position),
                                    "token_id": int(row.token_id),
                                    "token_text": row.token_text,
                                    "online_leverage_mean": float(
                                        row.online_leverage_mean
                                    ),
                                    "online_leverage_max": float(
                                        row.online_leverage_max
                                    ),
                                    "full_history_online_leverage_mean": float(
                                        row.full_history_online_leverage_mean
                                    ),
                                    "full_history_online_leverage_max": float(
                                        row.full_history_online_leverage_max
                                    ),
                                    "token_age_at_refresh": int(
                                        refresh_lag - int(row.token_offset)
                                    ),
                                    "protected_by_recent_window": bool(
                                        not eligible
                                    ),
                                    "eligible_for_refreshed_core": bool(eligible),
                                    "selected_in_refreshed_core": bool(selected),
                                    "refreshed_selector_rank": rank,
                                    "refreshed_selector_score": score,
                                }
                            )

            entries = pd.DataFrame(entry_rows)
            if len(entries):
                entries = entries.sort_values(
                    [
                        "sample_id",
                        "base_anchor",
                        "strategy",
                        "layer",
                        "token_offset",
                        "refresh_lag",
                    ]
                )
                entries["entered_refreshed_core_at_this_observation"] = (
                    entries.groupby(
                        [
                            "sample_id",
                            "base_anchor",
                            "strategy",
                            "layer",
                            "token_offset",
                        ]
                    )["selected_in_refreshed_core"]
                    .transform(lambda values: values & ~values.shift(fill_value=False))
                    .astype(bool)
                )

            exit_rows: List[Dict[str, Any]] = []
            if self.cfg.mechanism.recent_exit_enabled:
                exit_base = int(self.cfg.mechanism.recent_exit_base_anchor)
                search_max = int(
                    self.cfg.mechanism.recent_exit_search_max_offset
                )
                candidates = online[
                    online["base_anchor"].eq(exit_base)
                    & online["history_scope"].eq(
                        "selector_candidate_history"
                    )
                    & online["token_offset"].le(search_max)
                ]
                token_scores = candidates.groupby(
                    ["token_offset", "token_position", "token_id", "token_text"],
                    dropna=False,
                    as_index=False,
                )["online_leverage"].mean()
                chosen = token_scores.sort_values(
                    "online_leverage", ascending=False
                ).iloc[0]
                token_offset = int(chosen["token_offset"])
                exit_lag = token_offset + int(self.cfg.cache.recent_size)
                for relative in self.cfg.mechanism.recent_exit_relative_lags:
                    probe_lag = exit_lag + int(relative)
                    probe_step = exit_base + probe_lag
                    if probe_step not in reference.anchors:
                        continue
                    probe_anchor = reference.anchors[probe_step]
                    probe_selections = selections(probe_step)
                    for strategy in DEPLOYABLE:
                        stale = stale_cache[(exit_base, strategy)].get(
                            probe_lag + 1
                        )
                        if stale is None:
                            continue
                        refreshed_row = fresh(probe_step, strategy)
                        selected_fraction = float(
                            np.mean(
                                [
                                    int(chosen["token_position"])
                                    in set(layer.selected_positions)
                                    for layer in probe_selections[
                                        strategy
                                    ].by_layer.values()
                                ]
                            )
                        )
                        first_layer = reference.selected_layers[0]
                        positions = [
                            int(value)
                            for value in probe_anchor.position_maps[
                                first_layer
                            ].tolist()
                        ]
                        _, recent, eligible = mandatory_and_eligible(
                            positions,
                            self.cfg.cache.sink_size,
                            self.cfg.cache.recent_size,
                        )
                        exit_rows.append(
                            {
                                **base,
                                "base_anchor": exit_base,
                                "strategy": strategy,
                                "chosen_by": (
                                    "max_mean_candidate_history_online_leverage_"
                                    "across_diagnostic_layers_and_kv_heads"
                                ),
                                "token_offset": token_offset,
                                "token_position": int(chosen["token_position"]),
                                "token_id": int(chosen["token_id"]),
                                "token_text": chosen["token_text"],
                                "chosen_online_leverage_mean": float(
                                    chosen["online_leverage"]
                                ),
                                "recent_window_size": int(
                                    self.cfg.cache.recent_size
                                ),
                                "exit_lag": int(exit_lag),
                                "relative_to_exit": int(relative),
                                "probe_lag": int(probe_lag),
                                "probe_anchor": int(probe_step),
                                "token_in_recent_window": bool(
                                    int(chosen["token_position"]) in set(recent)
                                ),
                                "token_eligible_for_core": bool(
                                    int(chosen["token_position"]) in set(eligible)
                                ),
                                "selected_in_refreshed_core_layer_fraction": (
                                    selected_fraction
                                ),
                                "same_reference_token_verified": bool(
                                    stale["reference_token_id"]
                                    == refreshed_row["reference_token_id"]
                                    and stale["reference_token_position"]
                                    == refreshed_row["reference_token_position"]
                                ),
                                "stale_delta_nll": stale["delta_nll"],
                                "refreshed_delta_nll": refreshed_row["delta_nll"],
                                "refresh_benefit_delta_nll": (
                                    stale["delta_nll"]
                                    - refreshed_row["delta_nll"]
                                ),
                                "stale_approx_kl": stale["approx_kl"],
                                "refreshed_approx_kl": refreshed_row["approx_kl"],
                            }
                        )

            reference_inventory = pd.DataFrame(
                [
                    {
                        **base,
                        "prompt_length": int(reference.prompt_length),
                        "generated_length": int(
                            len(reference.generated_token_ids)
                        ),
                        "captured_anchor_count": int(len(reference.anchors)),
                        "captured_anchor_steps": json_text(
                            sorted(reference.anchors)
                        ),
                        "selected_diagnostic_layers": json_text(
                            reference.selected_layers
                        ),
                        "selected_diagnostic_heads": json_text(
                            reference.selected_heads
                        ),
                        "prompt_truncated": bool(reference.prompt_truncated),
                        "generation_stopped_on_eos": bool(
                            reference.generation_stopped_on_eos
                        ),
                        "generation_time_s": float(
                            reference.generation_time_s
                        ),
                        "mechanism_time_s": float(time.perf_counter() - started),
                        "sample_metadata": json_text(sample.metadata),
                    }
                ]
            )
            tables = {
                "reference_inventory": reference_inventory,
                "online_leverage": online,
                "online_leverage_core_entry": entries,
                "dense_refresh_counterfactuals": pd.DataFrame(dense_rows),
                "refresh_set_rank_changes": pd.DataFrame(rank_rows),
                "recent_window_exit_events": pd.DataFrame(exit_rows),
            }
            self._write_sample_tables(sample.sample_id, tables)
            self.store.mark_complete(
                key,
                {
                    "valid": True,
                    "online_rows": int(len(online)),
                    "dense_refresh_rows": int(len(dense_rows)),
                    "recent_exit_rows": int(len(exit_rows)),
                    "elapsed_s": float(time.perf_counter() - started),
                },
            )
        except Exception as exc:
            self._record_failure(key, sample, exc)
            if self.cfg.runtime.fail_on_error:
                raise
        finally:
            if reference is not None:
                self.model.release(reference)
