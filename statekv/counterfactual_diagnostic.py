"""Counterfactual MVP diagnostic: which utility predicts realized eviction damage?

Dev-only diagnostic (train panel, not a formal gate).  For stratified
candidate token groups at decoding boundaries, four runtime-causal utilities
are ranked against REALIZED physical removal damage:

- (A) current QK attention at the boundary,
- (B) fixed per-head EMA of attention history (train-tuned rhos),
- (C) R2 future-attention utility (dumped per-token teacher scores),
- (D) first-order counterfactual utility predicted from the baseline rollout:
  the norm of the attention-output contribution the group injects over the
  continuation, sum_t || sum_{i in group} a_{t,i} * v_i ||, summed over
  layers/heads.  No removal branch is needed for (D).

Realized damage physically deletes the group from a temporary branch of the
prefix-recomputed state (strict physical eviction semantics, no shadow KV)
and force-feeds the baseline greedy continuation, measuring summed exact KL
and delta-NLL against the unmodified continuation logits.  (D) is a
rollout predictor, not a removal measurement, so its ground truth never
comes from its own metric.
"""
from __future__ import annotations

import json
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
    causal_prefix_reference,
    sample_id_for,
    task_overrides,
)
from statekv.causal_existence_analysis import (
    deterministic_pairwise_accuracy,
    topk_indices,
)
from statekv.causal_predictors import _load_npz, _rho_key, ema_score
from statekv.causal_rollout import (
    _delete_positions,
    _prefix_recompute_state,
    _record_pool_attention,
)
from statekv.config import CacheDiscoveryConfig, apply_named_overrides, load_discovery_config
from statekv.oracle_closed_loop import KVBackingStore
from statekv.oracle_policy_comparison import _selection_from_scores
from statekv.oracle_policy_freegen import _check_prompt_truncation, _free_rollout
from statekv.qkv_decomposition import _scoring_forward_per_head, rank_and_margin
from statekv.selectors import mandatory_and_eligible
from statekv.storage import atomic_frame, atomic_json, safe_path_component
from statekv.tasks import load_discovery_tasks
from statekv.output_metrics import exact_distribution_metrics


UTILITIES = ("A_current_qk", "B_fixed_ema", "C_r2_future_attention", "D_cf_attention_value")
DAMAGE_METRICS = ("realized_kl", "realized_delta_nll")


# ------------------------------------------------------- CPU-side primitives


def stratified_candidate_groups(
    *,
    eligible: Sequence[int],
    utility_scores: Mapping[str, Mapping[int, float]],
    sizes: Sequence[int] = (4, 8),
    groups_per_size: int = 12,
    random_groups_per_size: int = 3,
    seed: int = 0,
) -> List[Dict[str, Any]]:
    """Stratified candidate groups over age x consensus-salience strata.

    Age is the position rank among eligible; salience is the mean normalized
    rank of the provided utilities (A/B/C at minimum).  Strata cover the full
    salience range so low-utility and random tokens are sampled as well —
    the panel must not be restricted to salient tokens.
    """

    eligible = [int(value) for value in eligible]
    if len(eligible) < max(int(size) for size in sizes):
        raise RuntimeError("eligible pool is smaller than the largest group size")
    rng = np.random.default_rng(int(seed))
    n = len(eligible)
    age_rank = np.empty(n, dtype=np.float64)
    age_rank[np.argsort(np.asarray(eligible, dtype=np.int64), kind="stable")] = (
        np.arange(n) / max(1, n - 1)
    )
    consensus = np.zeros(n, dtype=np.float64)
    for name in sorted(utility_scores):
        values = np.asarray(
            [float(utility_scores[name][position]) for position in eligible],
            dtype=np.float64,
        )
        if not np.isfinite(values).all():
            raise RuntimeError(f"utility {name} has non-finite scores on eligible set")
        ranks = np.empty(n, dtype=np.float64)
        ranks[np.argsort(values, kind="stable")] = np.arange(n) / max(1, n - 1)
        consensus += ranks
    consensus /= max(1, len(utility_scores))
    age_stratum = np.minimum((age_rank * 3).astype(np.int64), 2)
    salience_stratum = np.minimum((consensus * 3).astype(np.int64), 2)
    strata: Dict[Tuple[int, int], List[int]] = {}
    for index, position in enumerate(eligible):
        key = (int(age_stratum[index]), int(salience_stratum[index]))
        strata.setdefault(key, []).append(int(position))

    groups: List[Dict[str, Any]] = []
    group_id = 0
    stratum_keys = sorted(strata)
    stratified_per_size = int(groups_per_size) - int(random_groups_per_size)
    if stratified_per_size < len(stratum_keys):
        raise RuntimeError("groups_per_size cannot cover every stratum")
    for size in sizes:
        size = int(size)
        for rep in range(stratified_per_size):
            key = stratum_keys[rep % len(stratum_keys)]
            pool = strata[key]
            if len(pool) < size:
                raise RuntimeError(
                    f"stratum {key} has {len(pool)} tokens, below group size {size}"
                )
            members = sorted(
                int(value)
                for value in rng.choice(len(pool), size=size, replace=False)
            )
            groups.append(
                {
                    "group_id": group_id,
                    "size": size,
                    "stratum": f"age{key[0]}_salience{key[1]}",
                    "positions": [pool[index] for index in members],
                }
            )
            group_id += 1
        for _ in range(int(random_groups_per_size)):
            members = sorted(
                int(value) for value in rng.choice(n, size=size, replace=False)
            )
            groups.append(
                {
                    "group_id": group_id,
                    "size": size,
                    "stratum": "random",
                    "positions": [eligible[index] for index in members],
                }
            )
            group_id += 1
    return groups


def ranking_metrics(
    damage: np.ndarray,
    scores: np.ndarray,
    top_fraction: float = 0.25,
) -> Dict[str, float]:
    """Spearman, pairwise accuracy, and top-damage recall of one utility."""

    damage = np.asarray(damage, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    if damage.shape != scores.shape or damage.ndim != 1 or damage.size < 4:
        raise RuntimeError("ranking metrics require aligned 1-D arrays of >= 4 groups")
    if not (np.isfinite(damage).all() and np.isfinite(scores).all()):
        raise RuntimeError("ranking metrics require finite damage and scores")
    rho = spearmanr(damage, scores).statistic
    k = max(1, int(round(damage.size * float(top_fraction))))
    top_damage = set(int(value) for value in topk_indices(damage, k).tolist())
    top_predicted = set(int(value) for value in topk_indices(scores, k).tolist())
    return {
        "spearman": float(0.0 if not np.isfinite(rho) else rho),
        "pairwise_accuracy": float(deterministic_pairwise_accuracy(damage, scores)),
        "top_damage_recall": float(len(top_damage & top_predicted) / k),
        "groups": int(damage.size),
    }


def attention_value_damage(
    attention_rows: np.ndarray,
    values: np.ndarray,
    group_columns: Sequence[Sequence[int]],
) -> np.ndarray:
    """First-order counterfactual removal damage (utility D).

    attention_rows: (steps, layers, heads, candidates) rollout attention.
    values: (layers, heads, candidates, head_dim) cached value vectors.
    Returns one damage value per group: the norm of the removed
    attention-output contribution, summed over steps, layers, and heads.
    """

    attention = np.asarray(attention_rows, dtype=np.float64)
    value_array = np.asarray(values, dtype=np.float64)
    if attention.ndim != 4 or value_array.ndim != 4:
        raise RuntimeError("attention must be (t, l, h, n) and values (l, h, n, d)")
    steps, layers, heads, candidates = attention.shape
    if value_array.shape[:3] != (layers, heads, candidates):
        raise RuntimeError(
            f"value shape {value_array.shape} misaligned with attention {attention.shape}"
        )
    damages = np.zeros(len(group_columns), dtype=np.float64)
    for group_index, columns in enumerate(group_columns):
        columns = [int(value) for value in columns]
        if not columns or min(columns) < 0 or max(columns) >= candidates:
            raise RuntimeError("group columns are outside the candidate range")
        contribution = np.einsum(
            "tlhn,lhnd->tlhd",
            attention[:, :, :, columns],
            value_array[:, :, columns, :],
        )
        damages[group_index] = float(
            np.linalg.norm(contribution, axis=-1).sum()
        )
    return damages


def teacher_position_scores(
    teacher: Mapping[str, np.ndarray],
    cycle: int,
    horizon: int,
    eligible: Sequence[int],
) -> np.ndarray:
    """Join dumped R2 scores onto the boundary eligible set (fails loudly)."""

    cycles = [int(value) for value in teacher["cycles"]]
    horizons = [int(value) for value in teacher["horizons"]]
    if int(cycle) not in cycles:
        raise RuntimeError(f"teacher dump lacks cycle {cycle}: has {cycles}")
    if int(horizon) not in horizons:
        raise RuntimeError(f"teacher dump lacks horizon {horizon}: has {horizons}")
    cycle_index = cycles.index(int(cycle))
    count = int(teacher["position_lengths"][cycle_index])
    teacher_positions = [
        int(value) for value in teacher["position_ids"][cycle_index, :count]
    ]
    if teacher_positions != [int(value) for value in eligible]:
        raise RuntimeError(
            f"teacher positions do not match boundary eligible set at cycle {cycle}"
        )
    scores = np.asarray(
        teacher["scores"][cycle_index, horizons.index(int(horizon)), :, :, :count],
        dtype=np.float64,
    )
    if not np.isfinite(scores).all():
        raise RuntimeError(f"teacher scores are non-finite at cycle {cycle}")
    return scores.mean(axis=(0, 1))


def aggregate_diagnostic_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-sample means, then per-family and overall aggregation."""

    metrics = ["spearman", "pairwise_accuracy", "top_damage_recall"]
    per_sample = (
        frame.groupby(["sample_id", "task", "utility", "damage_metric"], as_index=False)
        [metrics]
        .mean()
    )

    def _summarize(df: pd.DataFrame) -> pd.DataFrame:
        return df.groupby(["task", "utility", "damage_metric"], as_index=False).agg(
            spearman_mean=("spearman", "mean"),
            spearman_std=("spearman", "std"),
            pairwise_accuracy_mean=("pairwise_accuracy", "mean"),
            pairwise_accuracy_std=("pairwise_accuracy", "std"),
            top_damage_recall_mean=("top_damage_recall", "mean"),
            top_damage_recall_std=("top_damage_recall", "std"),
            sequences=("sample_id", "nunique"),
        )

    per_family = _summarize(per_sample)
    overall = _summarize(per_sample.assign(task="ALL"))
    return pd.concat([per_family, overall], ignore_index=True)


# ------------------------------------------------------------- model runner


def _baseline_continuation(
    runner: CandidatePullbackRunner,
    state: Any,
    current_token: int,
    candidate_positions: Sequence[int],
    layers: Sequence[int],
    horizon: int,
) -> Dict[str, Any]:
    """Greedy continuation from a temporary state; keeps the state owned by
    the caller (unlike _causal_self_rollout, which releases it)."""

    logits_rows: List[torch.Tensor] = []
    generated: List[int] = []
    attention_rows: List[np.ndarray] = []
    token = int(current_token)
    logits, _, _ = runner.model.forward_one(state, token, capture_attention=True)
    logits_rows.append(logits.detach().float().cpu())
    token = int(torch.argmax(logits.float()).item())
    generated.append(token)
    for _ in range(int(horizon)):
        logits, record, _ = runner.model.forward_one(
            state, token, capture_attention=True
        )
        attention_rows.append(
            _record_pool_attention(record, state, candidate_positions, layers)
        )
        logits_rows.append(logits.detach().float().cpu())
        token = int(torch.argmax(logits.float()).item())
        generated.append(token)
    return {
        "generated": generated,
        "logits": logits_rows,
        "attention": np.stack(attention_rows, axis=0),
    }


def run_counterfactual_diagnostic(
    config_path: Path,
    repository_root: Path,
    max_samples: Optional[int] = None,
    sample_ids: Optional[Sequence[str]] = None,
) -> Path:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    repository_root = Path(repository_root)
    cfg = load_discovery_config(str(repository_root / str(config["base_config"])))
    cfg.experiment_name = str(config["experiment_name"])
    for key, value in dict(config.get("model_overrides") or {}).items():
        setattr(cfg.model, str(key), value)
    apply_named_overrides(cfg.runtime, config.get("runtime_overrides"), "runtime")
    cfg.tasks = task_overrides(config)
    cfg.runtime.seed = int(config["data_seed"])
    cfg.runtime.run_id = str(config["runtime_run_id"])
    cfg.anchor_steps = [0]
    layers = [int(value) for value in config["diagnostic_layers"]]
    cfg.diagnostics.explicit_layers = layers
    cfg.diagnostics.explicit_heads = [
        int(value) for value in config["diagnostic_query_heads"]
    ]

    diagnostic = dict(config["diagnostic"])
    boundaries = sorted(int(value) for value in diagnostic["boundaries"])
    horizon = int(diagnostic["horizon"])
    panel_ids = [
        sample_id_for(str(family), int(index))
        for family in config["task_families"]
        for index in diagnostic["train_indices"]
    ]
    if sample_ids:
        requested = {str(value) for value in sample_ids}
        unknown = requested - set(panel_ids)
        if unknown:
            raise ValueError(f"diagnostic sample IDs outside the panel: {sorted(unknown)}")
        panel_ids = [value for value in panel_ids if value in requested]
    if max_samples is not None:
        panel_ids = panel_ids[: int(max_samples)]

    source_run = repository_root / str(config["source_run"])
    teacher_root = repository_root / str(config["teacher_scores"]) / "train"
    fixed_rhos = json.loads(
        (source_run / "models" / "fixed_baseline_tuning.json").read_text(
            encoding="utf-8"
        )
    )["per_head"]
    output_root = repository_root / str(config["output_run"])
    output_root.mkdir(parents=True, exist_ok=True)

    runner = CandidatePullbackRunner(cfg, repository_root)
    samples, _ = load_discovery_tasks(cfg)
    by_id = {str(sample.sample_id): sample for sample in samples}
    missing = sorted(set(panel_ids) - set(by_id))
    if missing:
        raise RuntimeError(f"diagnostic panel samples were not loaded: {missing}")

    sink_size = int(config["sink_size"])
    recent_size = int(config["recent_size"])
    core_budget = int(config["core_budget"])
    total_budget = int(config["total_budget"])
    group_rows: List[Dict[str, Any]] = []
    metric_rows: List[Dict[str, Any]] = []
    cost_rows: List[Dict[str, Any]] = []
    runner.model.load()
    try:
        for ordinal, sample_id in enumerate(panel_ids, start=1):
            sample = by_id[sample_id]
            artifact = _load_npz(
                source_run / "artifacts" / "train" / f"{safe_path_component(sample_id)}.npz"
            )
            teacher = _load_npz(teacher_root / f"{safe_path_component(sample_id)}.npz")
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
                state, _ = runner.model.state_from_anchor(
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
                processed_tokens = [
                    int(value) for value in reference.prompt_token_ids[:-1]
                ]
                history_rows: List[np.ndarray] = []
                for cycle in range(max(boundaries) + 1):
                    # One persistent backing accumulates every position ever
                    # seen; _clone_full_state rebuilds the scoring state from
                    # it, exactly as in run_causal_rollout_study.
                    backing.update(runner, state)
                    per_head, positions, _ = _scoring_forward_per_head(
                        runner, state, backing, current_token
                    )
                    artifact_count = int(artifact["position_lengths"][cycle])
                    artifact_positions = [
                        int(value)
                        for value in artifact["position_ids"][cycle, :artifact_count]
                    ]
                    if positions != artifact_positions:
                        raise RuntimeError(
                            f"{sample_id} cycle {cycle}: replayed positions diverge "
                            "from the collected trajectory"
                        )
                    stacked = np.stack(
                        [np.asarray(per_head[layer], dtype=np.float32) for layer in layers]
                    )
                    history_rows.append(stacked)
                    _, _, eligible = mandatory_and_eligible(
                        positions, sink_size, recent_size
                    )
                    if cycle in boundaries:
                        rows, metrics, costs = _boundary_diagnostic(
                            runner,
                            sample,
                            sample_id,
                            cycle,
                            positions,
                            eligible,
                            stacked,
                            history_rows,
                            teacher,
                            processed_tokens,
                            current_token,
                            layers,
                            fixed_rhos,
                            diagnostic,
                            int(config["data_seed"]),
                        )
                        group_rows.extend(rows)
                        metric_rows.extend(metrics)
                        cost_rows.append(costs)
                        print(
                            f"[cf-diagnostic] {ordinal}/{len(panel_ids)} {sample_id} "
                            f"cycle {cycle} done",
                            flush=True,
                        )
                    cores_by_layer: Dict[int, Tuple[int, ...]] = {}
                    scores_by_layer: Dict[int, np.ndarray] = {}
                    for layer in range(len(state.cache)):
                        mean_attention = np.asarray(
                            per_head[layer], dtype=np.float64
                        ).mean(axis=0)
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
                        state,
                        backing,
                        current_token,
                        selection,
                        1,
                        initial_cache,
                        rolling_cache,
                    )
                    processed_tokens.append(int(current_token))
                    runner.model.release(state)
                    state = rollout.state
                    current_token = int(new_tokens[-1])
                runner.model.release(state)
            finally:
                runner.model.release(reference)
    finally:
        runner.model.close()

    groups_frame = pd.DataFrame(group_rows)
    metrics_frame = pd.DataFrame(metric_rows)
    atomic_frame(groups_frame, output_root / "group_rows.parquet")
    atomic_frame(metrics_frame, output_root / "boundary_metrics.parquet")
    atomic_frame(pd.DataFrame(cost_rows), output_root / "costs.csv")
    atomic_frame(
        aggregate_diagnostic_rows(metrics_frame), output_root / "summary.csv"
    )
    atomic_json(
        output_root / "protocol_summary.json",
        {
            "dev_only": True,
            "split": "train",
            "panel": panel_ids,
            "boundaries": boundaries,
            "horizon": horizon,
            "group_sizes": [int(value) for value in diagnostic["group_sizes"]],
            "groups_per_size": int(diagnostic["groups_per_size"]),
            "utilities": list(UTILITIES),
            "damage_metrics": list(DAMAGE_METRICS),
            "realized_damage": (
                "physical group deletion on a temporary prefix-recompute branch; "
                "forced continuation with the baseline greedy tokens; summed exact "
                "KL and delta-NLL vs the unmodified continuation"
            ),
            "utility_d": (
                "first-order counterfactual predictor: sum over continuation steps, "
                "layers, heads of ||sum_{i in group} a_{t,i} v_i||; no removal branch"
            ),
            "runtime_future_access": False,
        },
    )
    return output_root


def _cache_probe(keys: Sequence[Any], values: Sequence[Any], length: int) -> float:
    """Checksum of strided slices over the logical [0, length) cache region.

    The recompute buffers may have spare capacity beyond ``length``;
    continuations legitimately append there, so only the shared logical
    region is checked.
    """

    import mlx.core as mx

    total = 0.0
    for key_array, value_array in zip(keys, values):
        for array in (key_array, value_array):
            stride = max(1, int(length) // 16)
            probe = array[:, :, : int(length) : stride, :8].astype(mx.float32)
            total += float(probe.sum())
    return total


def _shared_cache_branch(pristine_state: Any, group: Sequence[int]) -> Any:
    """Branch state sharing the pristine cache buffers read-only.

    KVCache.update_and_fetch always reallocates (concatenate into a fresh
    buffer) before any in-place write when the cache is exactly full, which
    holds for the prefix-recompute state, and _delete_positions reassigns
    through mx.take — so the shared pristine buffers are never written.  The
    caller verifies this with _cache_probe.  _clone_state cannot be used:
    it round-trips through numpy, which bf16 MLX caches do not support.
    """

    from mlx_lm.models.cache import KVCache

    from statekv.backend_mlx import MLXReplayState

    caches = []
    for cache in pristine_state.cache:
        offset = int(cache.offset)
        cloned = KVCache()
        cloned.state = (
            cache.keys[:, :, :offset, :],
            cache.values[:, :, :offset, :],
        )
        caches.append(cloned)
    branch = MLXReplayState(
        cache=caches,
        position_maps={
            int(layer): positions.detach().clone()
            for layer, positions in pristine_state.position_maps.items()
        },
        logical_next_position=int(pristine_state.logical_next_position),
    )
    _delete_positions(branch, group)
    return branch


def _removal_branch_scores(
    runner: CandidatePullbackRunner,
    branches: Sequence[Any],
    input_tokens: Sequence[int],
    reference_logits: Sequence[torch.Tensor],
    horizon: int,
) -> List[Dict[str, float]]:
    """Realized removal damage per group on temporary physical branches.

    Mirrors _counterfactual_group_scores: forced continuation with the
    baseline tokens, summed exact KL / delta-NLL against the unmodified
    continuation logits.  Branches are pre-built shared-cache deletions of
    the prefix-recompute state (the R2 teacher's construction).
    """

    rows: List[Dict[str, float]] = []
    for group_id, branch in enumerate(branches):
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


def _boundary_diagnostic(
    runner: CandidatePullbackRunner,
    sample: Any,
    sample_id: str,
    cycle: int,
    positions: Sequence[int],
    eligible: Sequence[int],
    stacked_attention: np.ndarray,
    history_rows: Sequence[np.ndarray],
    teacher: Mapping[str, np.ndarray],
    processed_tokens: Sequence[int],
    current_token: int,
    layers: Sequence[int],
    fixed_rhos: Mapping[str, float],
    diagnostic: Mapping[str, Any],
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    horizon = int(diagnostic["horizon"])
    started = time.perf_counter()
    # (A) current QK attention, mean over diagnostic layers and KV heads.
    row_by_position = {int(position): row for row, position in enumerate(positions)}
    eligible_rows = np.asarray(
        [row_by_position[int(position)] for position in eligible], dtype=np.int64
    )
    utility_a = stacked_attention.astype(np.float64).mean(axis=(0, 1))[eligible_rows]
    # (B) fixed per-head EMA of the attention history with train-tuned rhos.
    width = int(stacked_attention.shape[-1])
    utility_b = np.zeros((len(layers), stacked_attention.shape[1], width))
    for layer_index in range(len(layers)):
        for head in range(stacked_attention.shape[1]):
            history = np.full((len(history_rows), width), np.nan, dtype=np.float32)
            for step, rows in enumerate(history_rows):
                history[step, : rows.shape[-1]] = rows[layer_index, head]
            utility_b[layer_index, head] = ema_score(
                history,
                float(fixed_rhos[_rho_key(horizon, layer_index, head)]),
            )
    utility_b = utility_b.mean(axis=(0, 1))[eligible_rows]
    # (C) dumped R2 future-attention utility at the matching horizon.
    utility_c = teacher_position_scores(teacher, cycle, horizon, eligible)

    # Baseline state: one prefix recompute (the R2 teacher's exact state
    # construction) serves value extraction, the baseline continuation, and —
    # read-only — every removal branch.
    recompute_started = time.perf_counter()
    baseline_state = _prefix_recompute_state(runner, processed_tokens, horizon + 2)
    original_keys = [cache.keys for cache in baseline_state.cache]
    original_values = [cache.values for cache in baseline_state.cache]
    logical_length = int(baseline_state.logical_next_position)
    probe_before = _cache_probe(original_keys, original_values, logical_length)
    try:
        # Cached values are extracted before the continuation; appending
        # generated tokens never changes existing rows.
        value_backing = KVBackingStore()
        value_backing.update(runner, baseline_state)
        recompute_s = time.perf_counter() - recompute_started
        # Candidate groups, stratified over age x consensus(A/B/C) salience.
        utilities_per_position = {
            "A": {int(p): float(v) for p, v in zip(eligible, utility_a)},
            "B": {int(p): float(v) for p, v in zip(eligible, utility_b)},
            "C": {int(p): float(v) for p, v in zip(eligible, utility_c)},
        }
        groups = stratified_candidate_groups(
            eligible=eligible,
            utility_scores=utilities_per_position,
            sizes=[int(value) for value in diagnostic["group_sizes"]],
            groups_per_size=int(diagnostic["groups_per_size"]),
            random_groups_per_size=int(diagnostic["random_groups_per_size"]),
            seed=int(seed) + int(cycle) + sum(ord(char) for char in sample_id),
        )
        # Removal branches are built BEFORE the continuation: they share the
        # pristine cache buffers, which the continuation then detaches from by
        # reallocation (verified by the cache probe below).
        branches = [
            _shared_cache_branch(baseline_state, group["positions"])
            for group in groups
        ]
        continuation_started = time.perf_counter()
        baseline = _baseline_continuation(
            runner, baseline_state, current_token, eligible, layers, horizon
        )
        continuation_s = time.perf_counter() - continuation_started
        # Consistency: the live R2 future-attention sum must reproduce the dump.
        live_r2 = (
            baseline["attention"].astype(np.float64).sum(axis=0).mean(axis=(0, 1))
        )
        rho = spearmanr(live_r2, utility_c).statistic
        if not np.isfinite(rho) or float(rho) < 0.98:
            raise RuntimeError(
                f"{sample_id} cycle {cycle}: live R2 rollout diverges from the "
                f"dumped teacher scores (spearman={rho})"
            )

        # (D) first-order counterfactual damage from the baseline rollout.
        value_positions = value_backing.positions()
        value_row = {
            int(position): row for row, position in enumerate(value_positions)
        }
        eligible_value_rows = np.asarray(
            [value_row[int(position)] for position in eligible], dtype=np.int64
        )
        value_rows = []
        for layer in layers:
            _, layer_values = value_backing.layer_arrays(int(layer))
            value_rows.append(
                layer_values[0].detach().float().cpu().numpy()[:, eligible_value_rows, :]
            )
        values = np.stack(value_rows, axis=0)
        eligible_column = {
            int(position): column for column, position in enumerate(eligible)
        }
        group_columns = [
            [eligible_column[int(position)] for position in group["positions"]]
            for group in groups
        ]
        utility_d = attention_value_damage(baseline["attention"], values, group_columns)

        # Realized damage: physical deletion branches + forced continuation.
        removal_started = time.perf_counter()
        realized = _removal_branch_scores(
            runner,
            branches,
            [int(current_token)] + baseline["generated"],
            baseline["logits"],
            horizon,
        )
        removal_s = time.perf_counter() - removal_started
        probe_after = _cache_probe(original_keys, original_values, logical_length)
        if abs(probe_after - probe_before) > 1.0e-3 * max(1.0, abs(probe_before)):
            raise RuntimeError(
                f"{sample_id} cycle {cycle}: shared pristine cache buffers were "
                "mutated by a continuation or removal branch"
            )
    finally:
        runner.model.release(baseline_state)

    rows: List[Dict[str, Any]] = []
    for group, realized_row, columns in zip(groups, realized, group_columns):
        row = {
            "sample_id": sample_id,
            "task": str(sample.task),
            "cycle": int(cycle),
            "group_id": int(group["group_id"]),
            "group_size": int(group["size"]),
            "stratum": str(group["stratum"]),
            "positions": ",".join(str(value) for value in group["positions"]),
            "realized_kl": float(realized_row["causal_kl"]),
            "realized_delta_nll": float(realized_row["causal_delta_nll"]),
            "A_current_qk": float(utility_a[columns].sum()),
            "B_fixed_ema": float(utility_b[columns].sum()),
            "C_r2_future_attention": float(utility_c[columns].sum()),
            "D_cf_attention_value": float(utility_d[int(group["group_id"])]),
        }
        rows.append(row)
    metrics: List[Dict[str, Any]] = []
    for damage_metric in DAMAGE_METRICS:
        damage = np.asarray([row[damage_metric] for row in rows], dtype=np.float64)
        for utility in UTILITIES:
            scores = np.asarray([row[utility] for row in rows], dtype=np.float64)
            metrics.append(
                {
                    "sample_id": sample_id,
                    "task": str(sample.task),
                    "cycle": int(cycle),
                    "utility": utility,
                    "damage_metric": damage_metric,
                    **ranking_metrics(
                        damage, scores, float(diagnostic["top_fraction"])
                    ),
                }
            )
    costs = {
        "sample_id": sample_id,
        "task": str(sample.task),
        "cycle": int(cycle),
        "groups": len(groups),
        "horizon": horizon,
        "prefix_recompute_s": float(recompute_s),
        "baseline_continuation_s": float(continuation_s),
        "removal_branches_s": float(removal_s),
        "boundary_total_s": float(time.perf_counter() - started),
    }
    return rows, metrics, costs


__all__ = [
    "DAMAGE_METRICS",
    "UTILITIES",
    "aggregate_diagnostic_rows",
    "attention_value_damage",
    "ranking_metrics",
    "run_counterfactual_diagnostic",
    "stratified_candidate_groups",
    "teacher_position_scores",
]
