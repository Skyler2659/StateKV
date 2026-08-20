"""Strictly causal self-rollout and counterfactual teachers for StateKV."""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import spearmanr

from statekv.candidate_pullback import CandidatePullbackRunner
from statekv.causal_existence import (
    _atomic_npz,
    _safe_sample_id,
    causal_prefix_reference,
    expand_split_ids,
    task_overrides,
)
from statekv.causal_existence_analysis import (
    aggregate_sequence_metrics,
    boundary_metrics,
)
from statekv.causal_predictors import _rho_key, ema_score
from statekv.config import CacheDiscoveryConfig, apply_named_overrides, load_discovery_config
from statekv.oracle_closed_loop import KVBackingStore
from statekv.oracle_policy_comparison import _selection_from_scores
from statekv.oracle_policy_freegen import (
    _advance_full_state,
    _check_prompt_truncation,
    _free_rollout,
)
from statekv.qkv_decomposition import _scoring_forward_per_head, rank_and_margin
from statekv.robust_envelope_policy import _clone_state
from statekv.selectors import mandatory_and_eligible
from statekv.storage import atomic_frame, atomic_json
from statekv.tasks import load_discovery_tasks
from statekv.trajectory_model import exact_distribution_metrics


def _load_artifact(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def _state_kv_bytes(state: Any) -> int:
    total = 0
    for cache in state.cache:
        for value in (cache.keys, cache.values):
            nbytes = getattr(value, "nbytes", None)
            if nbytes is not None:
                total += int(nbytes)
            else:
                total += int(np.prod(value.shape)) * 2
    return int(total)


def _prefix_recompute_state(
    runner: CandidatePullbackRunner,
    processed_tokens: Sequence[int],
    reserve: int,
) -> Any:
    """Recompute temporary full KV from tokens; no persistent shadow KV."""

    import mlx.core as mx
    from statekv.backend_mlx import MLXReplayState

    tokens = [int(value) for value in processed_tokens]
    if not tokens:
        raise ValueError("prefix recomputation requires at least one processed token")
    runner.model.runner.reset_attention_state()
    runner.model._configure_attention("prefill")
    # Prefix recomputation needs final KV only; retaining every prefill query's
    # attention would add quadratic memory without changing the causal teacher.
    runner.model.runner.attention_state["record_all_queries"] = False
    cache = runner.model.runner.make_cache("full", len(tokens) + int(reserve) + 2)
    chunk_size = max(1, int(runner.cfg.runtime.prefill_chunk_size))
    for offset in range(0, len(tokens), chunk_size):
        logits = runner.model.runner.model(
            mx.array([tokens[offset : offset + chunk_size]]), cache=cache
        )
        mx.eval(logits)
    runner.model.runner.attention_state["phase"] = "decode"
    return MLXReplayState(
        cache=cache,
        position_maps={
            layer: torch.arange(len(tokens), dtype=torch.long)
            for layer in range(len(cache))
        },
        logical_next_position=len(tokens),
    )


def _record_pool_attention(
    record: Any,
    state: Any,
    candidate_positions: Sequence[int],
    layers: Sequence[int],
) -> np.ndarray:
    rows: List[np.ndarray] = []
    for layer in layers:
        positions = [
            int(value) for value in state.position_maps[int(layer)].tolist()
        ]
        row_by_position = {position: row for row, position in enumerate(positions)}
        columns = [row_by_position[int(position)] for position in candidate_positions]
        raw = (
            record.oracle_attention_by_layer[int(layer)]
            .detach()
            .float()
            .cpu()
            .numpy()
            .reshape(-1, len(positions))
        )
        rows.append(raw[:, columns])
    return np.stack(rows, axis=0).astype(np.float32)


def _causal_self_rollout(
    runner: CandidatePullbackRunner,
    state: Any,
    current_token: int,
    candidate_positions: Sequence[int],
    layers: Sequence[int],
    horizons: Sequence[int],
) -> Dict[str, Any]:
    """Generate the model's own future and never read a saved continuation."""

    import mlx.core as mx

    mx.reset_peak_memory()
    started = time.perf_counter()
    maximum = max(int(value) for value in horizons)
    logits_rows: List[torch.Tensor] = []
    generated: List[int] = []
    attention_rows: List[np.ndarray] = []
    cumulative_wall_time: List[float] = []
    cumulative_peak_memory: List[int] = []
    token = int(current_token)
    try:
        # The current query creates the first simulated future token; its
        # attention is current utility and is deliberately excluded.
        logits, _, _ = runner.model.forward_one(state, token, capture_attention=True)
        logits_rows.append(logits.detach().float().cpu())
        token = int(torch.argmax(logits.float()).item())
        generated.append(token)
        for _ in range(maximum):
            logits, record, _ = runner.model.forward_one(
                state, token, capture_attention=True
            )
            attention_rows.append(
                _record_pool_attention(
                    record, state, candidate_positions, layers
                )
            )
            logits_rows.append(logits.detach().float().cpu())
            token = int(torch.argmax(logits.float()).item())
            generated.append(token)
            cumulative_wall_time.append(float(time.perf_counter() - started))
            cumulative_peak_memory.append(int(mx.get_peak_memory()))
        stacked = np.stack(attention_rows, axis=0)
        scores = {
            int(horizon): stacked[: int(horizon)].sum(axis=0)
            for horizon in horizons
        }
        return {
            "scores": scores,
            "generated": generated,
            "logits": logits_rows,
            "wall_time_s": float(time.perf_counter() - started),
            "wall_time_by_horizon": {
                int(horizon): cumulative_wall_time[int(horizon) - 1]
                for horizon in horizons
            },
            "peak_memory_by_horizon": {
                int(horizon): cumulative_peak_memory[int(horizon) - 1]
                for horizon in horizons
            },
            "peak_memory_bytes": int(mx.get_peak_memory()),
            "forwards": int(maximum + 1),
        }
    finally:
        runner.model.release(state)


def _delete_positions(state: Any, positions_to_delete: Sequence[int]) -> None:
    import mlx.core as mx

    remove = {int(value) for value in positions_to_delete}
    for layer, cache in enumerate(state.cache):
        positions = [int(value) for value in state.position_maps[layer].tolist()]
        keep_rows = [row for row, position in enumerate(positions) if position not in remove]
        rows = mx.array(keep_rows)
        offset = int(cache.offset)
        cache.keys = mx.take(cache.keys[:, :, :offset, :], rows, axis=2)
        cache.values = mx.take(cache.values[:, :, :offset, :], rows, axis=2)
        cache.offset = len(keep_rows)
        state.position_maps[layer] = torch.tensor(
            [positions[row] for row in keep_rows], dtype=torch.long
        )


def _counterfactual_group_scores(
    runner: CandidatePullbackRunner,
    full_state: Any,
    input_tokens: Sequence[int],
    reference_logits: Sequence[torch.Tensor],
    groups: Sequence[Sequence[int]],
    horizon: int,
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for group_id, group in enumerate(groups):
        branch = _clone_state(full_state)
        _delete_positions(branch, group)
        kl = 0.0
        delta_nll = 0.0
        logit_l2 = 0.0
        started = time.perf_counter()
        try:
            for offset, token in enumerate(input_tokens[: int(horizon) + 1]):
                logits, _, _ = runner.model.forward_one(
                    branch, int(token), capture_attention=True
                )
                baseline = reference_logits[offset]
                target = int(torch.argmax(baseline.float()).item())
                metrics = exact_distribution_metrics(baseline, logits, target)
                kl += float(metrics["exact_kl"])
                delta_nll += float(metrics["delta_nll"])
                logit_l2 += float(
                    torch.mean((baseline.float() - logits.float()) ** 2).sqrt().item()
                )
        finally:
            runner.model.release(branch)
        rows.append(
            {
                "group_id": int(group_id),
                "causal_kl": kl,
                "causal_delta_nll": delta_nll,
                "causal_logit_l2": logit_l2,
                "wall_time_s": float(time.perf_counter() - started),
            }
        )
    return rows


def _near_cutoff_groups(
    positions: Sequence[int],
    eligible: Sequence[int],
    scores: np.ndarray,
    core_budget: int,
    group_size: int = 4,
    group_count: int = 8,
) -> List[List[int]]:
    row_by_position = {int(position): row for row, position in enumerate(positions)}
    ordered = sorted(
        (int(value) for value in eligible),
        key=lambda position: (-float(scores[row_by_position[position]]), position),
    )
    center = min(max(0, int(core_budget)), len(ordered))
    width = int(group_size) * int(group_count)
    start = min(max(0, center - width // 2), max(0, len(ordered) - width))
    panel = ordered[start : start + width]
    return [
        panel[offset : offset + int(group_size)]
        for offset in range(0, len(panel), int(group_size))
        if len(panel[offset : offset + int(group_size)]) == int(group_size)
    ]


def run_causal_rollout_study(
    config_path: Path,
    repository_root: Path,
    split: str = "validation",
    max_samples: Optional[int] = None,
    cycles: Sequence[int] = (0, 8, 16, 24),
    counterfactual: bool = False,
    implementations: Optional[Sequence[str]] = None,
    sample_ids: Optional[Sequence[str]] = None,
) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cfg = load_discovery_config(str(repository_root / str(config["base_config"])))
    cfg.experiment_name = str(config["experiment_name"] + "_causal_rollout")
    for key, value in dict(config.get("model_overrides") or {}).items():
        setattr(cfg.model, str(key), value)
    apply_named_overrides(cfg.runtime, config.get("runtime_overrides"), "runtime")
    cfg.tasks = task_overrides(config)
    cfg.runtime.seed = int(config["data_seed"])
    cfg.runtime.run_id = str(config["runtime_run_id"] + "_causal_rollout")
    cfg.anchor_steps = [0]
    cfg.diagnostics.explicit_layers = [int(value) for value in config["diagnostic_layers"]]
    cfg.diagnostics.explicit_heads = [int(value) for value in config["diagnostic_query_heads"]]

    split_ids = expand_split_ids(config)[str(split)]
    if sample_ids:
        requested_ids = {str(value) for value in sample_ids}
        unknown = requested_ids - set(split_ids)
        if unknown:
            raise ValueError(
                f"rollout sample IDs are outside split {split}: {sorted(unknown)}"
            )
        split_ids = [value for value in split_ids if value in requested_ids]
    if max_samples is not None:
        split_ids = split_ids[: int(max_samples)]
    output_root = repository_root / str(config["output_run"])
    artifact_root = output_root / "artifacts" / str(split)
    result_root = output_root / "rollout" / str(split)
    result_root.mkdir(parents=True, exist_ok=True)
    artifacts = {
        str(_load_artifact(path)["sample_id"].item()): path
        for path in artifact_root.glob("*.npz")
    }
    missing = sorted(set(split_ids) - set(artifacts))
    if missing:
        raise RuntimeError(f"rollout study is missing causal label artifacts: {missing}")
    if str(split) == "fresh_test":
        from statekv.existence_reporting import register_fresh_test_component

        register_fresh_test_component(
            output_root,
            "causal_rollout_and_counterfactual"
            if counterfactual
            else "causal_rollout",
        )

    runner = CandidatePullbackRunner(cfg, repository_root)
    samples, _ = load_discovery_tasks(cfg)
    by_id = {str(sample.sample_id): sample for sample in samples}
    horizons = [int(value) for value in config["causal_rollout"]["horizons"]]
    run_implementations = {
        str(value)
        for value in (
            implementations or config["causal_rollout"]["implementations"]
        )
    }
    allowed_implementations = {"full_shadow", "prefix_recomputation"}
    if not run_implementations or not run_implementations <= allowed_implementations:
        raise ValueError("unknown or empty causal rollout implementation set")
    if "prefix_recomputation" not in run_implementations:
        raise ValueError("prefix recomputation is required for causal-teacher artifacts")
    requested_cycles = sorted({int(value) for value in cycles})
    requested_cycle_set = set(requested_cycles)
    if not requested_cycles or requested_cycles[0] < 0:
        raise ValueError("rollout study requires non-negative decision cycles")
    rollout_cycles = max(requested_cycles) + 1
    if rollout_cycles > int(config["control_cycles"]):
        raise ValueError("rollout decision cycle exceeds the collected trajectory")
    layers = [int(value) for value in config["diagnostic_layers"]]
    sink_size = int(config["sink_size"])
    recent_size = int(config["recent_size"])
    core_budget = int(config["core_budget"])
    total_budget = int(config["total_budget"])
    metric_rows: List[Dict[str, Any]] = []
    cost_rows: List[Dict[str, Any]] = []
    counterfactual_rows: List[Dict[str, Any]] = []
    model_info = runner.model.load()
    baseline_path = output_root / "models" / "fixed_baseline_tuning.json"
    if baseline_path.exists():
        import json

        fixed_baseline = json.loads(
            baseline_path.read_text(encoding="utf-8")
        )["per_head"]
    elif str(split) == "debug":
        fixed_baseline = {
            _rho_key(horizon, layer_index, head): 0.0
            for horizon in horizons
            for layer_index in range(len(layers))
            for head in range(int(model_info["num_key_value_heads"]))
        }
    else:
        raise RuntimeError("rollout evaluation requires train-tuned fixed baseline")
    try:
        for ordinal, sample_id in enumerate(split_ids, start=1):
            sample = by_id[sample_id]
            artifact = _load_artifact(artifacts[sample_id])
            teacher_cycles: List[int] = []
            teacher_positions: List[List[int]] = []
            teacher_scores: List[np.ndarray] = []
            reference = causal_prefix_reference(runner, sample)
            _check_prompt_truncation(reference, sample_id, False)
            try:
                anchor = reference.anchors[0]
                full_selection = runner._all_history_selection(reference, 0)
                full_cache = CacheDiscoveryConfig(
                    total_budget=int(anchor.logical_length + int(config["control_cycles"]) + 40),
                    sink_size=0,
                    recent_size=1,
                    selected_core_budget=int(anchor.logical_length + 1),
                )
                compressed_state, _ = runner.model.state_from_anchor(
                    anchor, full_selection, cache_config=full_cache
                )
                full_state, _ = runner.model.state_from_anchor(
                    anchor, full_selection, cache_config=full_cache
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
                current_token = int(anchor.query_token_id)
                processed_tokens = [int(value) for value in reference.prompt_token_ids[:-1]]
                for cycle in range(rollout_cycles):
                    backing.update(runner, compressed_state)
                    per_head, positions, scoring_s = _scoring_forward_per_head(
                        runner, compressed_state, backing, current_token
                    )
                    _, _, eligible = mandatory_and_eligible(
                        positions, sink_size, recent_size
                    )
                    selected_rows = [positions.index(int(value)) for value in eligible]
                    if cycle in requested_cycle_set:
                        rollout_results: List[Tuple[str, Dict[str, Any], float, int, int]] = []
                        if "full_shadow" in run_implementations:
                            r1_state = _clone_state(full_state)
                            r1_kv_bytes = _state_kv_bytes(r1_state)
                            r1 = _causal_self_rollout(
                                runner,
                                r1_state,
                                current_token,
                                eligible,
                                layers,
                                horizons,
                            )
                            rollout_results.append(
                                (
                                    "CAUSAL_EXPENSIVE_ROLLOUT_R1_FULL_SHADOW",
                                    r1,
                                    0.0,
                                    0,
                                    r1_kv_bytes,
                                )
                            )
                        import mlx.core as mx

                        mx.reset_peak_memory()
                        recompute_started = time.perf_counter()
                        r2_state = _prefix_recompute_state(
                            runner, processed_tokens, max(horizons) + 2
                        )
                        recompute_time = time.perf_counter() - recompute_started
                        recompute_peak_memory = int(mx.get_peak_memory())
                        r2_kv_bytes = _state_kv_bytes(r2_state)
                        r2 = _causal_self_rollout(
                            runner,
                            r2_state,
                            current_token,
                            eligible,
                            layers,
                            horizons,
                        )
                        rollout_results.append(
                            (
                                "CAUSAL_EXPENSIVE_ROLLOUT_R2_PREFIX_RECOMPUTE",
                                r2,
                                recompute_time,
                                recompute_peak_memory,
                                r2_kv_bytes,
                            )
                        )
                        teacher_cycles.append(int(cycle))
                        teacher_positions.append([int(value) for value in eligible])
                        teacher_scores.append(
                            np.stack(
                                [r2["scores"][int(horizon)] for horizon in horizons],
                                axis=0,
                            ).astype(np.float32)
                        )
                        for (
                            method,
                            result,
                            extra_recompute,
                            extra_peak,
                            temporary_kv_bytes,
                        ) in rollout_results:
                            for horizon in horizons:
                                horizon_wall_time = float(
                                    result["wall_time_by_horizon"][int(horizon)]
                                )
                                cost_rows.append(
                                    {
                                        "sample_id": sample_id,
                                        "task": str(sample.task),
                                        "split": split,
                                        "cycle": cycle,
                                        "method": method,
                                        "future_horizon": int(horizon),
                                        "rollout_wall_time_s": horizon_wall_time,
                                        "prefix_recompute_time_s": extra_recompute,
                                        "runtime_multiplier": (
                                            1.0
                                            + (horizon_wall_time + extra_recompute)
                                            / max(float(scoring_s), 1.0e-9)
                                        ),
                                        "peak_memory_bytes": max(
                                            int(
                                                result["peak_memory_by_horizon"][
                                                    int(horizon)
                                                ]
                                            ),
                                            int(temporary_kv_bytes),
                                            int(extra_peak),
                                        ),
                                        "temporary_kv_bytes": int(temporary_kv_bytes),
                                        "persistent_full_kv": method.endswith("FULL_SHADOW"),
                                        "forwards": int(horizon) + 1,
                                    }
                                )
                                truth = np.take(
                                    artifact["attention"][
                                        cycle + 1 : cycle + horizon + 1
                                    ],
                                    selected_rows,
                                    axis=-1,
                                ).sum(axis=0)
                                predicted = result["scores"][horizon]
                                for layer_index, layer in enumerate(layers):
                                    for head in range(predicted.shape[1]):
                                        history = np.take(
                                            artifact["attention"][
                                                : cycle + 1,
                                                layer_index,
                                                head,
                                            ],
                                            selected_rows,
                                            axis=-1,
                                        )
                                        baseline = ema_score(
                                            history,
                                            float(
                                                fixed_baseline[
                                                    _rho_key(
                                                        horizon,
                                                        layer_index,
                                                        head,
                                                    )
                                                ]
                                            ),
                                        )
                                        metrics = boundary_metrics(
                                            truth[layer_index, head],
                                            predicted[layer_index, head],
                                            baseline,
                                            core_budget,
                                        )
                                        metric_rows.append(
                                            {
                                                "sample_id": sample_id,
                                                "task": str(sample.task),
                                                "split": split,
                                                "cycle": cycle,
                                                "layer": layer,
                                                "head": head,
                                                "method": method,
                                                "future_horizon": horizon,
                                                **metrics,
                                            }
                                        )
                        if counterfactual:
                            if "full_shadow" not in run_implementations:
                                raise RuntimeError(
                                    "counterfactual diagnostic requires full-shadow rollout"
                                )
                            mean_scores = np.mean(
                                np.stack([per_head[layer] for layer in layers]),
                                axis=(0, 1),
                            )
                            groups = _near_cutoff_groups(
                                positions, eligible, mean_scores, core_budget
                            )
                            group_scores = _counterfactual_group_scores(
                                runner,
                                full_state,
                                [current_token] + r1["generated"],
                                r1["logits"],
                                groups,
                                horizon=16,
                            )
                            truth_h16 = artifact["attention"][
                                cycle + 1 : cycle + 17
                            ].sum(axis=(0, 1, 2))
                            row_by_position = {
                                int(position): row for row, position in enumerate(positions)
                            }
                            for group, values in zip(groups, group_scores):
                                noncausal = float(
                                    sum(truth_h16[row_by_position[int(position)]] for position in group)
                                )
                                counterfactual_rows.append(
                                    {
                                        "sample_id": sample_id,
                                        "task": str(sample.task),
                                        "split": split,
                                        "cycle": cycle,
                                        "positions": ",".join(str(value) for value in group),
                                        "noncausal_future_attention": noncausal,
                                        **values,
                                    }
                                )

                    cores_by_layer: Dict[int, Tuple[int, ...]] = {}
                    scores_by_layer: Dict[int, np.ndarray] = {}
                    for layer in range(len(compressed_state.cache)):
                        mean_attention = np.asarray(per_head[layer], dtype=np.float64).mean(axis=0)
                        _, _, core = rank_and_margin(
                            mean_attention, positions, eligible, core_budget
                        )
                        cores_by_layer[layer] = core
                        scores_by_layer[layer] = mean_attention
                    selection = _selection_from_scores(
                        "qk_pool", positions, eligible, cores_by_layer, scores_by_layer
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
                    _advance_full_state(
                        runner, full_state, current_token, new_tokens, rollout.logits
                    )
                    processed_tokens.append(int(current_token))
                    compressed_state = rollout.state
                    current_token = int(new_tokens[-1])
                teacher_width = max(len(values) for values in teacher_positions)
                packed_positions = np.full(
                    (len(teacher_positions), teacher_width), -1, dtype=np.int32
                )
                packed_scores = np.full(
                    (
                        len(teacher_scores),
                        len(horizons),
                        len(layers),
                        int(teacher_scores[0].shape[2]),
                        teacher_width,
                    ),
                    np.nan,
                    dtype=np.float32,
                )
                for index, (positions_row, scores_row) in enumerate(
                    zip(teacher_positions, teacher_scores)
                ):
                    count = len(positions_row)
                    packed_positions[index, :count] = np.asarray(
                        positions_row, dtype=np.int32
                    )
                    packed_scores[index, :, :, :, :count] = scores_row
                _atomic_npz(
                    result_root
                    / "teacher_scores"
                    / f"{_safe_sample_id(sample_id)}.npz",
                    cycles=np.asarray(teacher_cycles, dtype=np.int16),
                    horizons=np.asarray(horizons, dtype=np.int16),
                    position_ids=packed_positions,
                    position_lengths=np.asarray(
                        [len(values) for values in teacher_positions], dtype=np.int32
                    ),
                    scores=packed_scores,
                    sample_id=np.asarray(sample_id),
                    task=np.asarray(str(sample.task)),
                    split=np.asarray(str(split)),
                    teacher=np.asarray("CAUSAL_EXPENSIVE_ROLLOUT_R2_PREFIX_RECOMPUTE"),
                    runtime_future_access=np.asarray(False),
                )
                print(
                    f"[causal-rollout] sample {ordinal}/{len(split_ids)} {sample_id}",
                    flush=True,
                )
            finally:
                runner.model.release(reference)
    finally:
        runner.model.close()

    metrics = pd.DataFrame(metric_rows)
    costs = pd.DataFrame(cost_rows)
    atomic_frame(metrics, result_root / "boundary_metrics.parquet")
    atomic_frame(costs, result_root / "costs.csv")
    sequence = aggregate_sequence_metrics(metrics)
    atomic_frame(sequence, result_root / "sequence_metrics.csv")
    if counterfactual_rows:
        counterfactual_frame = pd.DataFrame(counterfactual_rows)
        atomic_frame(counterfactual_frame, result_root / "counterfactual_groups.parquet")
        correlations = []
        for key, group in counterfactual_frame.groupby(["sample_id", "cycle"]):
            for score in ("causal_kl", "causal_delta_nll", "causal_logit_l2"):
                rho = spearmanr(group["noncausal_future_attention"], group[score]).statistic
                correlations.append(
                    {
                        "sample_id": key[0],
                        "cycle": key[1],
                        "score": score,
                        "spearman": float(0.0 if not np.isfinite(rho) else rho),
                    }
                )
        atomic_frame(pd.DataFrame(correlations), result_root / "counterfactual_summary.csv")
    atomic_json(
        result_root / "protocol_summary.json",
        {
            "causal": True,
            "saved_real_future_access_at_runtime": False,
            "rollout_token_source": "model greedy continuation from current prefix",
            "r1": "temporary full-shadow KV; not memory efficient",
            "r2": "prefix-token recomputation; no persistent full-shadow KV",
            "horizons": horizons,
            "implementations": sorted(run_implementations),
            "counterfactual": bool(counterfactual),
        },
    )
    return result_root


__all__ = ["run_causal_rollout_study"]
