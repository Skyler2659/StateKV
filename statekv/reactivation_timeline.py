"""Full-causal-timeline reactivation collection and timeline RI.

The decode-only reactivation index (statekv/reactivation_index.py) misses
needles whose reactivation happens while the question is read during late
prefill. This module records every historical token's received-attention
trajectory over the WHOLE causal timeline:

- prefill query positions, aggregated into blocks of `prefill_block_size`
  consecutive query rows (mean received attention over the block's rows);
- decode query positions, one row per greedy full-cache decoding cycle.

Timeline rows are therefore mixed-granularity BY DESIGN: one prefill row
summarizes `prefill_block_size` query positions while one decode row is a
single query position. Dormancy windows (`dormant_window_rows`) count
timeline rows across both phases in the same units. This is a deliberate,
documented choice so a needle that goes dormant mid-prefill and spikes at
the late-prefill question accumulates dormancy in the same currency as a
decode-time reactivation.

Event taxonomy:
- Type I: dormant -> reactivation entry whose entry row is a prefill block.
- Type II: dormant -> reactivation entry whose entry row is a decode cycle.
- Type III persistent: positions in the top-`top_k` at every timeline row
  since their first active row (never dormant); counted separately, not
  events.

RI is computed ONLY from full-cache attention trajectories; no
compressed-policy output is read.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

from statekv.candidate_pullback import CandidatePullbackRunner
from statekv.causal_existence import (
    SPLIT_ORDER,
    expand_split_ids,
    task_overrides,
)
from statekv.config import apply_named_overrides, load_discovery_config
from statekv.storage import atomic_frame, atomic_json, atomic_npz, safe_path_component
from statekv.tasks import load_discovery_tasks


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------


def _harvest_observed_rows(
    state: Mapping[str, Any],
    layers: Sequence[int],
    expected_rows: int,
    expected_k_len: int,
) -> np.ndarray:
    """Copy this forward call's per-KV-head attention rows and check shapes.

    Returns an array of shape (n_layers, expected_rows, n_kv_heads,
    expected_k_len). Fails loudly when the hook recorded anything else.
    """

    observe = state.get("observe_heads", {})
    per_layer: List[np.ndarray] = []
    kv_heads: Optional[int] = None
    for layer in layers:
        entries = list(observe.get(int(layer), []))
        if len(entries) != int(expected_rows):
            raise RuntimeError(
                "attention hook recorded %d rows for layer %d, expected %d "
                "(check record_all_queries and max_observe)"
                % (len(entries), int(layer), int(expected_rows))
            )
        rows = np.stack(
            [np.asarray(entry, dtype=np.float32) for entry in entries], axis=0
        )
        if rows.ndim != 3 or int(rows.shape[-1]) != int(expected_k_len):
            raise RuntimeError(
                "attention hook row shape %s for layer %d, expected (*, %d)"
                % (rows.shape, int(layer), int(expected_k_len))
            )
        if kv_heads is None:
            kv_heads = int(rows.shape[1])
        elif int(rows.shape[1]) != kv_heads:
            raise RuntimeError("inconsistent KV-head count across layers")
        per_layer.append(rows)
    return np.stack(per_layer, axis=0)


def _clear_observed_rows(state: Dict[str, Any], layers: Sequence[int]) -> None:
    observe = state.get("observe_heads", {})
    pooled = state.get("observe", {})
    for layer in layers:
        if int(layer) in observe:
            observe[int(layer)] = []
        if int(layer) in pooled:
            pooled[int(layer)] = []


def _check_hook_health(state: Mapping[str, Any]) -> None:
    errors = int(state.get("hook_errors", 0))
    if errors:
        raise RuntimeError(
            "attention hook reported %d errors: %s"
            % (errors, state.get("hook_error_events", [])[:3])
        )


def _needle_token_spans(
    tokenizer: Any,
    prompt_ids: Sequence[int],
    evidence_texts: Sequence[str],
) -> np.ndarray:
    """Locate each evidence text inside the tokenized prompt.

    Mid-prompt texts may tokenize differently at the leading boundary, so a
    leading-space variant is tried as well. Missing evidence is not an error
    (gov_report has none); it yields zero spans.
    """

    spans: List[Tuple[int, int]] = []
    for text in evidence_texts:
        found: Optional[Tuple[int, int]] = None
        for variant in (str(text), " " + str(text)):
            ids = [
                int(value)
                for value in tokenizer.encode(variant, add_special_tokens=False)
            ]
            if not ids:
                continue
            limit = len(prompt_ids) - len(ids) + 1
            for start in range(max(0, limit)):
                if list(prompt_ids[start : start + len(ids)]) == ids:
                    found = (start, start + len(ids))
                    break
            if found is not None:
                break
        if found is not None:
            spans.append(found)
    if not spans:
        return np.zeros((0, 2), dtype=np.int32)
    return np.asarray(spans, dtype=np.int32)


def collect_sample_timeline(
    model: Any,
    sample: Any,
    layers: Sequence[int],
    cycles: int,
    prefill_block_size: int,
) -> Dict[str, np.ndarray]:
    """Record one sample's full-causal-timeline attention trajectory.

    Prefill runs in chunks of `prefill_block_size` query positions; after
    each chunk the hook's per-KV-head query rows are harvested and cleared,
    so every block maps to a known query-position span. Decode then feeds
    greedy tokens one at a time and harvests the single query row per cycle.
    Decode row t is query position prompt_length + t (the forward of the
    t-th generated token), so prefill blocks and decode cycles tile the
    query timeline without overlap.
    """

    import mlx.core as mx

    layers = [int(value) for value in layers]
    cycles = int(cycles)
    block_size = int(prefill_block_size)
    if cycles <= 0 or block_size <= 0:
        raise ValueError("cycles and prefill_block_size must be positive")

    runner = model.runner
    runner.reset_attention_state()
    model._configure_attention("prefill")
    state = runner.attention_state
    state["record_all_queries"] = True
    state["max_observe"] = block_size
    cache = runner.make_cache("full", model.cfg.cache.total_budget)
    prompt_ids, prompt_truncated = model.encode_prompt(sample.prompt)
    if prompt_truncated:
        raise RuntimeError(
            "timeline collection requires the untruncated prompt: %s"
            % str(sample.sample_id)
        )
    prompt_length = len(prompt_ids)
    if prompt_length < 2:
        raise ValueError("timeline collection requires at least two prompt tokens")

    block_arrays: List[np.ndarray] = []
    spans: List[Tuple[int, int]] = []
    logits = None
    for offset in range(0, prompt_length, block_size):
        chunk = prompt_ids[offset : offset + block_size]
        logits = runner.model(mx.array([chunk]), cache=cache)
        mx.eval(logits)
        harvested = _harvest_observed_rows(
            state, layers, len(chunk), offset + len(chunk)
        )
        block_arrays.append(harvested.mean(axis=1))
        spans.append((offset, offset + len(chunk)))
        _clear_observed_rows(state, layers)
    _check_hook_health(state)

    state["phase"] = "decode"
    decode_arrays: List[np.ndarray] = []
    generated: List[int] = []
    token = int(mx.argmax(logits[0, -1, :]).item())
    for cycle in range(cycles):
        logits = runner.model(mx.array([[token]]), cache=cache)
        mx.eval(logits)
        harvested = _harvest_observed_rows(
            state, layers, 1, prompt_length + cycle + 1
        )
        decode_arrays.append(harvested[:, 0, :, :])
        generated.append(int(token))
        token = int(mx.argmax(logits[0, -1, :]).item())
        _clear_observed_rows(state, layers)
    _check_hook_health(state)

    n_blocks = len(block_arrays)
    n_layers = len(layers)
    kv_heads = int(block_arrays[0].shape[1])
    width = prompt_length + cycles

    prefill_attention = np.zeros(
        (n_blocks, n_layers, kv_heads, prompt_length), dtype=np.float32
    )
    for block, (array, (_, end)) in enumerate(zip(block_arrays, spans)):
        if int(array.shape[-1]) != int(end):
            raise RuntimeError("prefill block width does not match its span")
        prefill_attention[block, :, :, :end] = array
    decode_attention = np.zeros(
        (cycles, n_layers, kv_heads, width), dtype=np.float32
    )
    for cycle, array in enumerate(decode_arrays):
        expected = prompt_length + cycle + 1
        if int(array.shape[-1]) != expected:
            raise RuntimeError("decode row width does not match the timeline")
        decode_attention[cycle, :, :, :expected] = array

    position_lengths = np.asarray(
        [end for _, end in spans]
        + [prompt_length + cycle + 1 for cycle in range(cycles)],
        dtype=np.int32,
    )
    evidence = sample.metadata.get("evidence_texts") or []
    needle_spans = _needle_token_spans(runner.hf_tokenizer, prompt_ids, evidence)
    return {
        "prefill_block_attention": prefill_attention,
        "prefill_block_spans": np.asarray(spans, dtype=np.int32),
        "decode_attention": decode_attention,
        "position_lengths": position_lengths,
        "prompt_length": np.asarray(prompt_length, dtype=np.int32),
        "prefill_block_size": np.asarray(block_size, dtype=np.int32),
        "layers": np.asarray(layers, dtype=np.int16),
        "prompt_token_ids": np.asarray(prompt_ids, dtype=np.int32),
        "generated_token_ids": np.asarray(generated, dtype=np.int32),
        "needle_token_spans": needle_spans,
        "sample_id": np.asarray(str(sample.sample_id)),
        "task": np.asarray(str(sample.task)),
    }


def collect_reactivation_timeline_dataset(
    config_path: Path,
    repository_root: Path,
    splits: Optional[Sequence[str]] = None,
    sample_ids: Optional[Sequence[str]] = None,
    cycle_limit: Optional[int] = None,
) -> Path:
    """Collect timeline trajectories for the configured splits/samples."""

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    split_ids = expand_split_ids(config)
    requested_splits = tuple(str(value) for value in (splits or SPLIT_ORDER))
    if set(requested_splits) - set(SPLIT_ORDER):
        raise ValueError("requested an unknown split")

    cfg = load_discovery_config(str(repository_root / str(config["base_config"])))
    cfg.experiment_name = str(config["experiment_name"])
    for key, value in dict(config.get("model_overrides") or {}).items():
        setattr(cfg.model, str(key), value)
    apply_named_overrides(cfg.runtime, config.get("runtime_overrides"), "runtime")
    cfg.tasks = task_overrides(config)
    cfg.runtime.seed = int(config["data_seed"])
    cfg.runtime.run_id = str(config["runtime_run_id"])
    cfg.anchor_steps = [0]
    cfg.diagnostics.explicit_layers = [
        int(value) for value in config["diagnostic_layers"]
    ]

    output_root = repository_root / str(config["output_run"])
    artifact_root = output_root / "timeline"
    output_root.mkdir(parents=True, exist_ok=True)

    selected_ids = [
        sample_id for split in requested_splits for sample_id in split_ids[split]
    ]
    if sample_ids:
        requested = {str(value) for value in sample_ids}
        unknown = requested - set(selected_ids)
        if unknown:
            raise ValueError(
                "explicit sample IDs are outside the requested splits: %s"
                % sorted(unknown)
            )
        selected_ids = [value for value in selected_ids if value in requested]
    split_by_id = {
        sample_id: split for split, values in split_ids.items() for sample_id in values
    }

    cycles = int(config["cycles"])
    if cycle_limit is not None:
        cycles = min(cycles, int(cycle_limit))
    if cycles <= 0:
        raise ValueError("cycle count must be positive")
    block_size = int(config.get("prefill_block_size", 64))
    layers = [int(value) for value in config["diagnostic_layers"]]

    runner = CandidatePullbackRunner(cfg, repository_root)
    samples, task_events = load_discovery_tasks(cfg)
    by_id = {str(sample.sample_id): sample for sample in samples}
    missing = sorted(set(selected_ids) - set(by_id))
    if missing:
        raise RuntimeError(f"configured timeline samples were not loaded: {missing}")

    started = time.perf_counter()
    model_info = runner.model.load()
    model_layers = int(model_info["num_layers"])
    if any(layer < 0 or layer >= model_layers for layer in layers):
        raise ValueError("diagnostic layer outside model")

    timings: List[Dict[str, Any]] = []
    try:
        for ordinal, sample_id in enumerate(selected_ids, start=1):
            split = split_by_id[sample_id]
            artifact = artifact_root / split / f"{safe_path_component(sample_id)}.npz"
            if artifact.exists():
                print(f"[reactivation-timeline] reuse {sample_id}", flush=True)
                continue
            sample = by_id[sample_id]
            sample_started = time.perf_counter()
            arrays = collect_sample_timeline(
                runner.model, sample, layers, cycles, block_size
            )
            elapsed = time.perf_counter() - sample_started
            atomic_npz(artifact, **arrays, split=np.asarray(split))
            timings.append(
                {
                    "sample_id": str(sample_id),
                    "split": split,
                    "collection_time_s": float(elapsed),
                    "prompt_length": int(arrays["prompt_length"]),
                    "prefill_blocks": int(arrays["prefill_block_attention"].shape[0]),
                    "cycles": int(arrays["decode_attention"].shape[0]),
                }
            )
            print(
                "[reactivation-timeline] sample %d/%d %s split=%s "
                "prompt=%d blocks=%d cycles=%d time=%.1fs"
                % (
                    ordinal,
                    len(selected_ids),
                    sample_id,
                    split,
                    int(arrays["prompt_length"]),
                    int(arrays["prefill_block_attention"].shape[0]),
                    int(arrays["decode_attention"].shape[0]),
                    elapsed,
                ),
                flush=True,
            )
    finally:
        runner.model.close()

    if timings:
        atomic_frame(pd.DataFrame(timings), output_root / "collection_timings.csv")
    atomic_json(
        output_root / "collection_summary.json",
        {
            "experiment": str(config["experiment_name"]),
            "requested_splits": list(requested_splits),
            "requested_samples": len(selected_ids),
            "cycles_this_invocation": cycles,
            "prefill_block_size": block_size,
            "diagnostic_layers": layers,
            "elapsed_s": float(time.perf_counter() - started),
            "sample_timings": timings,
            "task_events": task_events,
            "timeline_semantics": (
                "prefill rows are block_size query-position means; decode rows "
                "are single query positions starting at prompt_length"
            ),
        },
    )
    return output_root


# ---------------------------------------------------------------------------
# Timeline RI
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TimelineReactivationParams:
    """Frozen timeline-RI parameters (selected on train/validation only).

    `dormant_window_rows` counts unified timeline rows: prefill blocks and
    decode cycles share the same row currency (see module docstring).
    """

    top_k: int = 10
    dormant_window_rows: int = 8
    dormant_rank_quantile: float = 0.1
    min_row: int = 1


@dataclass
class TimelineReactivationEvent:
    position: int
    row: int
    phase: str  # "prefill" or "decode"
    event_type: str  # "I" or "II"
    dormancy_duration: int
    reactivation_distance: int
    dormant_mean_importance: float
    reactivation_importance: float

    @property
    def amplitude(self) -> float:
        return self.reactivation_importance - self.dormant_mean_importance


@dataclass
class SequenceTimelineReactivation:
    sample_id: str
    task: str
    n_rows: int
    n_prefill_rows: int
    n_positions: int
    n_entry_events: int
    n_reactivation_events: int
    n_type_i_events: int
    n_type_ii_events: int
    n_type_iii_persistent: int
    events: List[TimelineReactivationEvent] = field(default_factory=list)

    @property
    def ri_fraction(self) -> float:
        if self.n_entry_events == 0:
            return 0.0
        return self.n_reactivation_events / self.n_entry_events

    def summary(self) -> Dict[str, Any]:
        durations = [event.dormancy_duration for event in self.events]
        distances = [event.reactivation_distance for event in self.events]
        amplitudes = [event.amplitude for event in self.events]
        return {
            "sample_id": self.sample_id,
            "task": self.task,
            "n_rows": self.n_rows,
            "n_prefill_rows": self.n_prefill_rows,
            "n_positions": self.n_positions,
            "ri_count": self.n_reactivation_events,
            "ri_fraction": self.ri_fraction,
            "entry_events": self.n_entry_events,
            "type_i_events": self.n_type_i_events,
            "type_ii_events": self.n_type_ii_events,
            "type_iii_persistent": self.n_type_iii_persistent,
            "mean_dormancy_duration": (
                float(np.mean(durations)) if durations else 0.0
            ),
            "mean_reactivation_distance": (
                float(np.mean(distances)) if distances else 0.0
            ),
            "mean_reactivation_amplitude": (
                float(np.mean(amplitudes)) if amplitudes else 0.0
            ),
        }


def timeline_importance(
    artifact: Mapping[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Unified per-row per-position importance over the whole timeline.

    Returns (importance, position_lengths, n_prefill_rows) with importance
    of shape (n_prefill_rows + n_decode_rows, prompt_length + n_decode_rows),
    averaged over layers and KV heads and zeroed beyond each row's active
    prefix. Fails loudly on shape inconsistencies.
    """

    prefill = np.asarray(artifact["prefill_block_attention"], dtype=np.float64)
    decode = np.asarray(artifact["decode_attention"], dtype=np.float64)
    lengths = np.asarray(artifact["position_lengths"], dtype=np.int64)
    prompt_length = int(np.asarray(artifact["prompt_length"]))
    if prefill.ndim != 4 or decode.ndim != 4:
        raise ValueError("timeline attention arrays must be 4-dimensional")
    n_blocks = int(prefill.shape[0])
    n_cycles = int(decode.shape[0])
    if n_blocks == 0 or n_cycles == 0:
        raise ValueError("timeline requires prefill blocks and decode cycles")
    if int(prefill.shape[3]) != prompt_length:
        raise ValueError("prefill width does not match prompt_length")
    if int(decode.shape[3]) != prompt_length + n_cycles:
        raise ValueError("decode width does not match prompt_length + cycles")
    if tuple(prefill.shape[1:3]) != tuple(decode.shape[1:3]):
        raise ValueError("layer/head dimensions differ between phases")
    if lengths.shape != (n_blocks + n_cycles,):
        raise ValueError("position_lengths must cover every timeline row")
    if (np.diff(lengths) < 0).any():
        raise ValueError("position_lengths must be non-decreasing")
    if int(lengths[-1]) != prompt_length + n_cycles:
        raise ValueError("final timeline row must see the full sequence")

    width = prompt_length + n_cycles
    importance = np.zeros((n_blocks + n_cycles, width), dtype=np.float64)
    importance[:n_blocks, :prompt_length] = prefill.mean(axis=(1, 2))
    importance[n_blocks:, :] = decode.mean(axis=(1, 2))
    for row in range(n_blocks + n_cycles):
        importance[row, int(lengths[row]) :] = 0.0
    return importance, lengths, n_blocks


def compute_timeline_reactivation(
    artifact: Mapping[str, np.ndarray],
    params: TimelineReactivationParams,
) -> SequenceTimelineReactivation:
    """Compute dormant->reactivation events on the full causal timeline."""

    importance, lengths, n_blocks = timeline_importance(artifact)
    n_rows, n_positions = importance.shape

    ranks = np.full((n_rows, n_positions), np.inf)
    in_top_k = np.zeros((n_rows, n_positions), dtype=bool)
    for row in range(n_rows):
        active = int(lengths[row])
        if active == 0:
            continue
        order = np.argsort(-importance[row, :active], kind="stable")
        rank_of = np.empty(active, dtype=np.float64)
        rank_of[order] = np.arange(active, dtype=np.float64) / active
        ranks[row, :active] = rank_of
        top = order[: min(params.top_k, active)]
        in_top_k[row, top] = True

    dormant_cut = float(params.dormant_rank_quantile)
    n_entries = 0
    events: List[TimelineReactivationEvent] = []
    n_type_i = 0
    n_type_ii = 0
    n_type_iii = 0
    for position in range(n_positions):
        first_row = int(np.searchsorted(lengths, position + 1))
        if first_row >= n_rows:
            continue
        if bool(in_top_k[first_row:, position].all()):
            n_type_iii += 1
        for row in range(max(first_row, int(params.min_row)), n_rows):
            if not in_top_k[row, position]:
                continue
            if row > 0 and in_top_k[row - 1, position]:
                continue  # continuation, not an entry
            n_entries += 1
            streak = 0
            lookback = row - 1
            while lookback >= first_row:
                rank = ranks[lookback, position]
                if np.isinf(rank) or rank < dormant_cut:
                    break
                streak += 1
                lookback -= 1
            if streak < int(params.dormant_window_rows):
                continue
            previous = np.flatnonzero(in_top_k[:row, position])
            last_important = int(previous[-1]) if len(previous) else first_row
            dormant_span = importance[row - streak : row, position]
            phase = "prefill" if row < n_blocks else "decode"
            event_type = "I" if row < n_blocks else "II"
            if event_type == "I":
                n_type_i += 1
            else:
                n_type_ii += 1
            events.append(
                TimelineReactivationEvent(
                    position=position,
                    row=row,
                    phase=phase,
                    event_type=event_type,
                    dormancy_duration=streak,
                    reactivation_distance=row - last_important,
                    dormant_mean_importance=float(dormant_span.mean()),
                    reactivation_importance=float(importance[row, position]),
                )
            )

    return SequenceTimelineReactivation(
        sample_id=str(np.asarray(artifact["sample_id"])),
        task=str(np.asarray(artifact["task"])),
        n_rows=n_rows,
        n_prefill_rows=n_blocks,
        n_positions=n_positions,
        n_entry_events=n_entries,
        n_reactivation_events=len(events),
        n_type_i_events=n_type_i,
        n_type_ii_events=n_type_ii,
        n_type_iii_persistent=n_type_iii,
        events=events,
    )


def load_artifact(path: str) -> Dict[str, Any]:
    """Load a collected .npz artifact into a plain mapping."""
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


__all__ = [
    "SequenceTimelineReactivation",
    "TimelineReactivationEvent",
    "TimelineReactivationParams",
    "collect_reactivation_timeline_dataset",
    "collect_sample_timeline",
    "compute_timeline_reactivation",
    "load_artifact",
    "timeline_importance",
]
