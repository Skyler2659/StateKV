"""Causal-state dataset collection for the StateKV existence study.

Future attention is written only as an offline label trajectory. Every feature
in the same artifact is copied from the current prefix before the next token is
generated. Fresh-test artifacts can be collected separately after model and
analysis choices are frozen.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

from statekv.candidate_pullback import CandidatePullbackRunner
from statekv.config import CacheDiscoveryConfig, apply_named_overrides, load_discovery_config
from statekv.oracle_closed_loop import KVBackingStore
from statekv.oracle_policy_comparison import _selection_from_scores
from statekv.oracle_policy_freegen import (
    _advance_full_state,
    _check_prompt_truncation,
    _clone_full_state,
    _free_rollout,
    _metric_row,
)
from statekv.qkv_decomposition import rank_and_margin
from statekv.selectors import mandatory_and_eligible
from statekv.storage import atomic_frame, atomic_json
from statekv.tasks import load_discovery_tasks


SPLIT_ORDER = ("debug", "train", "validation", "fresh_test")


def causal_prefix_reference(runner: CandidatePullbackRunner, sample: Any) -> Any:
    """Prefill-only anchor; no real or saved future is ever generated."""

    import mlx.core as mx

    model = runner.model
    model.runner.reset_attention_state()
    model._configure_attention("prefill")
    model.runner.attention_state["record_all_queries"] = False
    cache = model.runner.make_cache("full", model.cfg.cache.total_budget)
    prompt_ids, prompt_truncated = model.encode_prompt(sample.prompt)
    if len(prompt_ids) < 2:
        raise ValueError("causal prefix requires at least two prompt tokens")
    chunk_size = max(1, int(model.cfg.runtime.prefill_chunk_size))
    for offset in range(0, len(prompt_ids), chunk_size):
        logits = model.runner.model(
            mx.array([prompt_ids[offset : offset + chunk_size]]), cache=cache
        )
        mx.eval(logits)
    logical_length = len(prompt_ids)
    position_maps = model._cache_position_maps(cache, logical_length)
    anchor = model._anchor_state(
        cache,
        position_maps,
        logical_length,
        0,
        int(prompt_ids[-1]),
    )
    return SimpleNamespace(
        sample_id=str(sample.sample_id),
        task=str(sample.task),
        prompt_token_ids=[int(value) for value in prompt_ids],
        prompt_length=int(logical_length),
        prompt_truncated=bool(prompt_truncated),
        anchors={0: anchor},
        generated_token_ids=[],
        query_records=[],
        probe_logits={},
    )


def _safe_sample_id(sample_id: str) -> str:
    return str(sample_id).replace(":", "__").replace("/", "_")


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(handle, "wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def sample_id_for(task_family: str, index: int) -> str:
    prefixes = {
        "ruler_niah": "synthetic_niah_",
        "govreport_or_qmsum": "gov_report:",
        "ruler_niah_multikey": "synthetic_niah_multikey_",
        "ruler_niah_multiquery": "synthetic_niah_multiquery_",
        "ruler_variable_tracking": "synthetic_vt_",
        "passage_retrieval_en": "passage_retrieval_en:",
        "hotpotqa": "hotpotqa:",
    }
    if task_family not in prefixes:
        raise ValueError(f"unsupported existence-study task family: {task_family}")
    return f"{prefixes[task_family]}{int(index)}"


def expand_split_ids(config: Mapping[str, Any]) -> Dict[str, List[str]]:
    families = [str(value) for value in config["task_families"]]
    expanded = {
        str(split): [
            sample_id_for(family, int(index))
            for family in families
            for index in indices
        ]
        for split, indices in dict(config["split_indices"]).items()
    }
    unknown = set(expanded) - set(SPLIT_ORDER)
    if unknown:
        raise ValueError(f"unknown split names: {sorted(unknown)}")
    flat = [sample_id for split in SPLIT_ORDER for sample_id in expanded.get(split, [])]
    if len(flat) != len(set(flat)):
        raise ValueError("existence-study sample IDs overlap across splits")
    expected = dict(config.get("expected_split_sizes") or {})
    for split, size in expected.items():
        if len(expanded.get(str(split), [])) != int(size):
            raise ValueError(f"split size mismatch for {split}")
    return expanded


def task_overrides(config: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    all_indices = sorted(
        {
            int(index)
            for indices in dict(config["split_indices"]).values()
            for index in indices
        }
    )
    if not all_indices:
        raise ValueError("empty existence-study index set")
    if all_indices != list(range(all_indices[0], all_indices[-1] + 1)):
        raise ValueError("existence-study indices must form one contiguous range")
    settings_by_task = dict(config.get("task_settings") or {})
    output: Dict[str, Dict[str, Any]] = {}
    for family in config["task_families"]:
        family = str(family)
        settings = dict(settings_by_task.get(family) or {})
        settings["num_samples"] = len(all_indices)
        if family in {
            "ruler_niah",
            "ruler_niah_multikey",
            "ruler_niah_multiquery",
            "ruler_variable_tracking",
        }:
            settings["sample_offset"] = int(all_indices[0])
        elif family in {"govreport_or_qmsum", "passage_retrieval_en", "hotpotqa"}:
            # LongBench rows are addressed by dataset index; the local
            # THUDM/LongBench snapshot has 200 rows per task, so indices
            # beyond 199 fail loudly at selection time.
            settings["sample_indices"] = list(all_indices)
        output[family] = settings
    return output


def _scoring_forward(
    runner: CandidatePullbackRunner,
    state: Any,
    backing: KVBackingStore,
    current_token: int,
) -> Tuple[Dict[int, np.ndarray], List[int], Any, torch.Tensor, float]:
    """Run one full-prefix causal forward and copy its diagnostic record."""

    scoring_state = _clone_full_state(runner, state, backing, int(current_token), 1)
    positions = backing.positions()
    try:
        logits, record, forward_s = runner.model.forward_one(
            scoring_state, int(current_token), capture_attention=True
        )
        per_head: Dict[int, np.ndarray] = {}
        for layer in range(len(scoring_state.cache)):
            maps = [
                int(value)
                for value in scoring_state.position_maps[int(layer)].tolist()
            ]
            row_by_position = {position: row for row, position in enumerate(maps)}
            raw = (
                record.oracle_attention_by_layer[int(layer)]
                .detach()
                .float()
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
                .reshape(-1, len(maps))
            )
            keep = [row_by_position[int(position)] for position in positions]
            per_head[int(layer)] = raw[:, keep].copy()
    finally:
        runner.model.release(scoring_state)
    return per_head, positions, record, logits.detach().float().cpu(), float(forward_s)


def _record_queries(
    record: Any,
    layers: Sequence[int],
    heads: Sequence[int],
    attribute: str,
) -> np.ndarray:
    values = getattr(record, attribute)
    return np.stack(
        [
            np.stack(
                [
                    values[f"{int(layer)}:{int(head)}"]
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                    for head in heads
                ],
                axis=0,
            )
            for layer in layers
        ],
        axis=0,
    ).astype(np.float16)


def _record_hidden(record: Any, layers: Sequence[int], attribute: str) -> np.ndarray:
    values = getattr(record, attribute)
    return np.stack(
        [
            values[int(layer)].detach().float().cpu().numpy()
            for layer in layers
        ],
        axis=0,
    ).astype(np.float16)


def _global_logit_features(logits: torch.Tensor) -> np.ndarray:
    probabilities = torch.softmax(logits.double(), dim=-1)
    top = torch.topk(probabilities, k=2).values
    entropy = -(probabilities * probabilities.clamp_min(1.0e-300).log()).sum()
    return np.asarray(
        [
            float(entropy.item()),
            float(top[0].item()),
            float((top[0] - top[1]).item()),
            float(logits.float().norm().item()),
        ],
        dtype=np.float32,
    )


def _pack_attention(
    attention_steps: Sequence[np.ndarray],
    position_steps: Sequence[Sequence[int]],
) -> Dict[str, np.ndarray]:
    cycles = len(attention_steps)
    width = max(len(values) for values in position_steps)
    layers, heads = attention_steps[0].shape[:2]
    attention = np.full((cycles, layers, heads, width), np.nan, dtype=np.float32)
    positions = np.full((cycles, width), -1, dtype=np.int32)
    lengths = np.zeros(cycles, dtype=np.int32)
    for cycle, (values, step_positions) in enumerate(
        zip(attention_steps, position_steps)
    ):
        count = len(step_positions)
        attention[cycle, :, :, :count] = values
        positions[cycle, :count] = np.asarray(step_positions, dtype=np.int32)
        lengths[cycle] = count
    return {
        "attention": attention,
        "position_ids": positions,
        "position_lengths": lengths,
    }


def _read_artifact_metadata(path: Path, repository_root: Path) -> Dict[str, Any]:
    with np.load(path, allow_pickle=False) as artifact:
        return {
            "sample_id": str(artifact["sample_id"].item()),
            "task": str(artifact["task"].item()),
            "split": str(artifact["split"].item()),
            "cycles": int(artifact["attention"].shape[0]),
            "path": str(path.relative_to(repository_root)),
        }


def collect_causal_existence_dataset(
    config_path: Path,
    repository_root: Path,
    splits: Optional[Sequence[str]] = None,
    max_samples: Optional[int] = None,
    cycle_limit: Optional[int] = None,
    sample_prefixes: Optional[Sequence[str]] = None,
    sample_ids: Optional[Sequence[str]] = None,
) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    split_ids = expand_split_ids(config)
    requested_splits = tuple(str(value) for value in (splits or SPLIT_ORDER))
    if set(requested_splits) - set(SPLIT_ORDER):
        raise ValueError("requested an unknown existence-study split")

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
    cfg.diagnostics.explicit_heads = [
        int(value) for value in config["diagnostic_query_heads"]
    ]

    publication_output_root = repository_root / str(config["output_run"])
    smoke_mode = max_samples is not None or cycle_limit is not None
    if (
        "fresh_test" in requested_splits
        and not smoke_mode
        and not (publication_output_root / "frozen_validation_selection.json").exists()
    ):
        raise RuntimeError(
            "fresh-test collection is sealed until validation selection is frozen"
        )
    output_root = (
        publication_output_root / "_smoke"
        if smoke_mode
        else publication_output_root
    )
    artifact_root = output_root / "artifacts"
    output_root.mkdir(parents=True, exist_ok=True)
    sample_manifest = [
        {"sample_id": sample_id, "split": split}
        for split in SPLIT_ORDER
        for sample_id in split_ids[split]
    ]
    atomic_frame(pd.DataFrame(sample_manifest), output_root / "split_manifest.csv")
    ledger_path = output_root / "test_open_ledger.json"
    if not ledger_path.exists():
        atomic_json(
            ledger_path,
            {
                "fresh_test_open_limit": int(config["fresh_test_open_limit"]),
                "fresh_test_evaluations": 0,
                "events": [],
            },
        )

    samples, task_events = load_discovery_tasks(cfg)
    by_id = {str(sample.sample_id): sample for sample in samples}
    selected_ids = [
        sample_id
        for split in requested_splits
        for sample_id in split_ids[split]
    ]
    if sample_prefixes:
        prefixes = tuple(str(value) for value in sample_prefixes)
        selected_ids = [
            sample_id
            for sample_id in selected_ids
            if sample_id.startswith(prefixes)
        ]
    if sample_ids:
        requested_ids = {str(value) for value in sample_ids}
        unknown_ids = requested_ids - set(selected_ids)
        if unknown_ids:
            raise ValueError(
                f"explicit sample IDs are outside the requested splits: {sorted(unknown_ids)}"
            )
        selected_ids = [
            sample_id for sample_id in selected_ids if sample_id in requested_ids
        ]
    missing = sorted(set(selected_ids) - set(by_id))
    if missing:
        raise RuntimeError(f"configured existence samples were not loaded: {missing}")
    if max_samples is not None:
        selected_ids = selected_ids[: int(max_samples)]
    split_by_id = {
        sample_id: split for split, values in split_ids.items() for sample_id in values
    }

    configured_cycles = int(config["control_cycles"])
    cycles = (
        configured_cycles
        if cycle_limit is None
        else min(configured_cycles, int(cycle_limit))
    )
    if cycles <= 0:
        raise ValueError("cycle count must be positive")
    layers = [int(value) for value in config["diagnostic_layers"]]
    query_heads = [int(value) for value in config["diagnostic_query_heads"]]
    total_budget = int(config["total_budget"])
    sink_size = int(config["sink_size"])
    recent_size = int(config["recent_size"])
    core_budget = int(config["core_budget"])

    runner = CandidatePullbackRunner(cfg, repository_root)
    started = time.perf_counter()
    model_info = runner.model.load()
    kv_heads = int(model_info["num_key_value_heads"])
    if query_heads != list(range(int(model_info["num_attention_heads"]))):
        raise ValueError("causal existence collection requires every query head")

    try:
        for ordinal, sample_id in enumerate(selected_ids, start=1):
            split = split_by_id[sample_id]
            artifact = artifact_root / split / f"{_safe_sample_id(sample_id)}.npz"
            if artifact.exists():
                print(f"[causal-existence] reuse {sample_id}", flush=True)
                continue
            sample = by_id[sample_id]
            reference = causal_prefix_reference(runner, sample)
            _check_prompt_truncation(reference, str(sample.sample_id), False)
            try:
                anchor_state = reference.anchors[0]
                full_selection = runner._all_history_selection(reference, 0)
                full_cache = CacheDiscoveryConfig(
                    total_budget=int(anchor_state.logical_length + cycles + 2),
                    sink_size=0,
                    recent_size=1,
                    selected_core_budget=int(anchor_state.logical_length + 1),
                )
                compressed_state, _ = runner.model.state_from_anchor(
                    anchor_state, full_selection, cache_config=full_cache
                )
                full_state, _ = runner.model.state_from_anchor(
                    anchor_state, full_selection, cache_config=full_cache
                )
                backing = KVBackingStore()
                initial_cache = CacheDiscoveryConfig(
                    total_budget=total_budget,
                    sink_size=sink_size,
                    recent_size=max(1, recent_size - 1),
                    selected_core_budget=core_budget,
                )
                rolling_cache = CacheDiscoveryConfig(
                    total_budget=total_budget,
                    sink_size=sink_size,
                    recent_size=recent_size,
                    selected_core_budget=core_budget,
                )
                current_token = int(anchor_state.query_token_id)
                generated: List[int] = []
                attention_steps: List[np.ndarray] = []
                position_steps: List[List[int]] = []
                query_pre_steps: List[np.ndarray] = []
                query_post_steps: List[np.ndarray] = []
                residual_steps: List[np.ndarray] = []
                attention_input_steps: List[np.ndarray] = []
                global_steps: List[np.ndarray] = []
                current_tokens: List[int] = []
                step_rows: List[Dict[str, Any]] = []

                for cycle in range(cycles):
                    backing.update(runner, compressed_state)
                    per_head, positions, record, logits, scoring_s = _scoring_forward(
                        runner, compressed_state, backing, current_token
                    )
                    selected_attention = np.stack(
                        [per_head[layer] for layer in layers], axis=0
                    )
                    if int(selected_attention.shape[1]) != kv_heads:
                        raise RuntimeError("unexpected pooled KV-head count")
                    attention_steps.append(selected_attention)
                    position_steps.append([int(value) for value in positions])
                    query_pre_steps.append(
                        _record_queries(record, layers, query_heads, "queries")
                    )
                    query_post_steps.append(
                        _record_queries(
                            record, layers, query_heads, "post_rope_queries"
                        )
                    )
                    residual_steps.append(
                        _record_hidden(record, layers, "residual_inputs")
                    )
                    attention_input_steps.append(
                        _record_hidden(record, layers, "attention_inputs")
                    )
                    global_steps.append(_global_logit_features(logits))
                    current_tokens.append(int(current_token))

                    _, _, eligible = mandatory_and_eligible(
                        positions, sink_size, recent_size
                    )
                    cores_by_layer: Dict[int, Tuple[int, ...]] = {}
                    scores_by_layer: Dict[int, np.ndarray] = {}
                    for layer in range(len(compressed_state.cache)):
                        mean_attention = np.asarray(
                            per_head[layer], dtype=np.float64
                        ).mean(axis=0)
                        _, _, core = rank_and_margin(
                            mean_attention, positions, eligible, core_budget
                        )
                        cores_by_layer[layer] = core
                        scores_by_layer[layer] = mean_attention
                    selection = _selection_from_scores(
                        "qk_pool",
                        positions,
                        eligible,
                        cores_by_layer,
                        scores_by_layer,
                    )
                    rollout, new_tokens = _free_rollout(
                        runner,
                        compressed_state,
                        backing,
                        current_token,
                        selection,
                        1,
                        initial_cache,
                        rolling_cache,
                    )
                    metric = _advance_full_state(
                        runner, full_state, current_token, new_tokens, rollout.logits
                    )[0]
                    step_rows.append(
                        {
                            "sample_id": str(sample.sample_id),
                            "task": str(sample.task),
                            "split": split,
                            "cycle": int(cycle),
                            "generated_token_id": int(new_tokens[0]),
                            "scoring_forward_time_s": float(scoring_s),
                            **metric,
                        }
                    )
                    compressed_state = rollout.state
                    current_token = int(new_tokens[-1])
                    generated.extend(int(value) for value in new_tokens)

                packed = _pack_attention(attention_steps, position_steps)
                universe = backing.positions()
                keys = []
                values = []
                for layer in layers:
                    layer_keys, layer_values = backing.layer_arrays(layer)
                    keys.append(layer_keys[0].detach().cpu().numpy())
                    values.append(layer_values[0].detach().cpu().numpy())
                _atomic_npz(
                    artifact,
                    **packed,
                    query_pre=np.stack(query_pre_steps).astype(np.float16),
                    query_post=np.stack(query_post_steps).astype(np.float16),
                    residual=np.stack(residual_steps).astype(np.float16),
                    attention_input=np.stack(attention_input_steps).astype(np.float16),
                    global_features=np.stack(global_steps).astype(np.float32),
                    current_token_ids=np.asarray(current_tokens, dtype=np.int32),
                    generated_token_ids=np.asarray(generated, dtype=np.int32),
                    kv_position_ids=np.asarray(universe, dtype=np.int32),
                    keys=np.stack(keys).astype(np.float16),
                    values=np.stack(values).astype(np.float16),
                    layers=np.asarray(layers, dtype=np.int16),
                    query_heads=np.asarray(query_heads, dtype=np.int16),
                    sample_id=np.asarray(str(sample.sample_id)),
                    task=np.asarray(str(sample.task)),
                    split=np.asarray(split),
                    feature_cutoff=np.asarray("current_prefix_before_generation"),
                    label_source=np.asarray("offline_future_attention_only"),
                )
                sample_frame = pd.DataFrame(step_rows)
                atomic_frame(sample_frame, artifact.with_suffix(".steps.parquet"))
                print(
                    "[causal-existence] sample %d/%d %s split=%s kl=%.6f"
                    % (
                        ordinal,
                        len(selected_ids),
                        sample.sample_id,
                        split,
                        float(sample_frame["exact_kl"].mean()),
                    ),
                    flush=True,
                )
            finally:
                runner.model.release(reference)
    finally:
        runner.model.close()

    artifacts = sorted(artifact_root.glob("*/*.npz"))
    metadata = [
        _read_artifact_metadata(path, repository_root) for path in artifacts
    ]
    atomic_frame(pd.DataFrame(metadata), output_root / "collection_manifest.csv")
    atomic_json(
        output_root / "collection_summary.json",
        {
            "experiment": str(config["experiment_name"]),
            "requested_splits": list(requested_splits),
            "requested_samples": len(selected_ids),
            "available_artifacts": len(metadata),
            "cycles_this_invocation": cycles,
            "elapsed_s": float(time.perf_counter() - started),
            "task_events": task_events,
            "feature_time_boundary": "current prefix before next-token generation",
            "future_information_in_features": False,
            "future_information_in_labels": True,
            "publication_artifact": not smoke_mode,
        },
    )
    return output_root


__all__ = [
    "SPLIT_ORDER",
    "collect_causal_existence_dataset",
    "causal_prefix_reference",
    "expand_split_ids",
    "sample_id_for",
    "task_overrides",
]
