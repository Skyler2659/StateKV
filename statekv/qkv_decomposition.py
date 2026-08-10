"""QK–V decomposition dataset builder (discovery protocol Phase 1).

Runs the recoverable qk_pool trajectory (the R0 strongest baseline) with
rich per-cycle instrumentation and writes three tables:

- token rows (T): per cycle x layer x position, head-mean QK and V features:
  attention, exact single-token attention-output removal perturbation
  (delta = a/(1-a) * ||(v - o) W_O||, zero extra forwards), projected-V
  norm, raw V norm, attention x projected-V, rank / margin-to-cutoff /
  in-core flags, token metadata.
- head rows (H): per query-head features for a token subset (dev samples).
- swap rows (S): exact 1-step same-input KL of budget-preserving
  all-layer cutoff swaps (the exact downstream target for the near-tie
  question), with both tokens' feature vectors.

All semantics follow the R0 recoverable protocol: shared KVBackingStore,
budget 256 = sink 4 + recent 32 + core 220, refresh every cycle, greedy,
same-input exact KL trajectory metrics.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

from statekv.candidate_pullback import CandidatePullbackRunner
from statekv.config import (
    CacheDiscoveryConfig,
    apply_named_overrides,
    load_discovery_config,
)
from statekv.oracle_closed_loop import (
    KVBackingStore,
    _rollout_candidate,
    _top_core,
)
from statekv.oracle_policy_comparison import _selection_from_scores
from statekv.oracle_policy_freegen import (
    _advance_full_state,
    _check_prompt_truncation,
    _clone_full_state,
    _free_rollout,
    _full_reference_segment,
    _metric_row,
)
from statekv.selectors import CoreSelection, LayerSelection, mandatory_and_eligible
from statekv.storage import atomic_frame, atomic_json, atomic_text
from statekv.tasks import load_discovery_tasks


# ---------------------------------------------------------------- features


def kv_head_for_query(query_head: int, query_heads: int, kv_heads: int) -> int:
    """GQA mapping: query head -> its KV head."""

    group = int(query_heads) // int(kv_heads)
    return int(query_head) // group


def head_projector_grams(
    runner: CandidatePullbackRunner,
    layers: Sequence[int],
    query_heads: int,
    head_dim: int,
) -> Dict[int, torch.Tensor]:
    """Precompute G_h = W_h^T W_h per layer per query head.

    W_h is the o_proj input slice of head h; ||x W_h^T||^2 = x G_h x^T.
    Handles QuantizedLinear via mx.dequantize.  Returns [heads, dim, dim]
    float32 torch tensors.
    """

    import mlx.core as mx

    grams: Dict[int, torch.Tensor] = {}
    for layer in layers:
        o_proj = runner.model.runner.model.model.layers[
            int(layer)
        ].self_attn.o_proj
        if hasattr(o_proj, "scales"):
            weight = mx.dequantize(
                o_proj.weight,
                o_proj.scales,
                o_proj.biases,
                group_size=64,
                bits=4,
            )
        else:
            weight = o_proj.weight
        dense = torch.from_numpy(
            np.array(weight.astype(mx.float32), copy=True)
        )
        # MLX Linear: y = x W^T, weight [out, in]; head slice of the input.
        slices = dense[:, : int(query_heads) * int(head_dim)].reshape(
            dense.shape[0], int(query_heads), int(head_dim)
        )
        grams[int(layer)] = torch.einsum("ohd,ohj->hdj", slices, slices).float()
    return grams


def exact_removal_delta(
    attn: torch.Tensor,
    values: torch.Tensor,
    gram: torch.Tensor,
    kv_map: Sequence[int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact single-token attention-output removal+renorm perturbation.

    attn: [heads, n] softmax weights; values: [kv_heads, n, dim];
    gram: [heads, dim, dim]; kv_map: query head -> kv head.
    Returns (delta, projected_v_norm, output_per_head):
      delta[h, i]  = a/(1-a) * ||(v_i - o_h) W_h||
      pv[h, i]     = ||v_i W_h||
      output[h]    = o_h (attention output per head)
    Exactness: o' = (o - a_i v_i)/(1 - a_i) => o' - o = a_i/(1-a_i)(o - v_i);
    ||(o'-o) W_h|| = a_i/(1-a_i) ||(v_i - o) W_h||.
    """

    heads, count = attn.shape
    outputs = torch.zeros(heads, values.shape[-1], dtype=torch.float64)
    for head in range(heads):
        outputs[head] = attn[head].double() @ values[int(kv_map[head])].double()
    expanded_v = values[torch.as_tensor(list(kv_map))]  # [heads, n, dim]
    diff = expanded_v.double() - outputs.unsqueeze(1)  # v_i - o_h
    delta_sq = torch.einsum("hni,hij,hnj->hn", diff, gram.double(), diff)
    pv_sq = torch.einsum(
        "hni,hij,hnj->hn", expanded_v.double(), gram.double(), expanded_v.double()
    )
    weight = attn.double() / (1.0 - attn.double()).clamp(min=1.0e-12)
    delta = weight * delta_sq.clamp(min=0.0).sqrt()
    pv = pv_sq.clamp(min=0.0).sqrt()
    return delta.float(), pv.float(), outputs.float()


def rank_and_margin(
    attn_mean: np.ndarray, positions: Sequence[int], eligible: Sequence[int], core_budget: int
) -> Tuple[Dict[int, int], Dict[int, float], Tuple[int, ...]]:
    """Rank (1 = highest attention) within eligible, margin to the cutoff.

    margin_i = attn_cutoff - attn_i (positive inside the core, negative
    outside).  Returns (rank_by_position, margin_by_position, core).
    """

    row_by_position = {int(p): r for r, p in enumerate(positions)}
    ordered = sorted(
        (int(p) for p in eligible),
        key=lambda p: (-float(attn_mean[row_by_position[p]]), p),
    )
    take = min(int(core_budget), len(ordered))
    core = tuple(sorted(ordered[:take]))
    cutoff_score = float(attn_mean[row_by_position[ordered[take - 1]]]) if take else 0.0
    rank = {int(p): index + 1 for index, p in enumerate(ordered)}
    margin = {
        int(p): cutoff_score - float(attn_mean[row_by_position[p]])
        for p in ordered
    }
    return rank, margin, core


# ---------------------------------------------------------------- swap oracle


def swap_selection_all_layers(
    selection: CoreSelection,
    remove_position: int,
    add_position: int,
) -> Tuple[CoreSelection, int]:
    """Swap remove->add in every layer core that contains remove and lacks add.

    Returns (new selection, number of layers swapped).  Budget preserved per
    layer by construction.
    """

    by_layer = {}
    swapped = 0
    for layer, current in selection.by_layer.items():
        core = [int(v) for v in current.selected_positions]
        if int(remove_position) in core and int(add_position) not in core:
            core = sorted(set(core) - {int(remove_position)} | {int(add_position)})
            swapped += 1
        by_layer[int(layer)] = LayerSelection(
            layer=int(layer),
            selected_positions=core,
            eligible_positions=list(current.eligible_positions),
            aggregate_scores=list(current.aggregate_scores),
            metadata=dict(current.metadata),
        )
    return (
        CoreSelection(
            strategy="swap",
            horizon_condition=None,
            by_layer=by_layer,
            metadata={"swap": True, "remove": int(remove_position), "add": int(add_position)},
        ),
        swapped,
    )


# ---------------------------------------------------------------- targets


def add_future_targets(
    frame: pd.DataFrame, horizons: Sequence[int], core_budget: int
) -> pd.DataFrame:
    """Post-hoc future-relevance targets from the token table itself.

    Per (sample, layer, position): fut_attn_h = mean attention at cycles
    c+1..c+h; fut_min_rank_h = best rank within h cycles; revival_h =
    currently outside the core but inside within h cycles.
    """

    frame = frame.sort_values(["sample_id", "layer", "position", "cycle"])
    for horizon in horizons:
        h = int(horizon)
        fut_attn = np.full(len(frame), np.nan)
        fut_rank = np.full(len(frame), np.nan)
        revival = np.zeros(len(frame), dtype=bool)
        attn = frame["attn"].to_numpy(dtype=np.float64)
        rank = frame["rank"].to_numpy(dtype=np.float64)
        in_core = frame["in_core"].to_numpy(dtype=bool)
        group_keys = frame.groupby(["sample_id", "layer", "position"]).indices
        for indices in group_keys.values():
            order = np.asarray(indices)
            values_attn = attn[order]
            values_rank = rank[order]
            values_core = in_core[order]
            count = len(order)
            if count < 2:
                continue
            shifts_attn = np.full((count, h), np.nan)
            shifts_rank = np.full((count, h), np.nan)
            shifts_core = np.zeros((count, h), dtype=bool)
            for k in range(1, h + 1):
                if count - k <= 0:
                    break
                shifts_attn[: count - k, k - 1] = values_attn[k:]
                shifts_rank[: count - k, k - 1] = values_rank[k:]
                shifts_core[: count - k, k - 1] = values_core[k:]
            with np.errstate(invalid="ignore"):
                fut_attn[order] = np.nanmean(shifts_attn, axis=1)
                fut_rank[order] = np.nanmin(shifts_rank, axis=1)
            revival[order] = (~values_core) & shifts_core.any(axis=1)
        frame[f"fut_attn_{h}"] = fut_attn
        frame[f"fut_min_rank_{h}"] = fut_rank
        frame[f"revival_{h}"] = revival
    return frame


# ---------------------------------------------------------------- token metadata


def classify_token(piece: str, count_in_prompt: int) -> str:
    """Coarse runtime-free token class for residual stratification."""

    text = piece.replace("Ġ", "").replace("▁", "").strip()
    if not text:
        return "structural"
    if any(ch.isdigit() for ch in text):
        return "numeric"
    if all(not ch.isalnum() for ch in text):
        return "punctuation"
    if count_in_prompt <= 2:
        return "rare"
    if text[:1].isupper():
        return "capitalized"
    return "common"


# ---------------------------------------------------------------- runner


def _scoring_forward_per_head(
    runner: CandidatePullbackRunner,
    state: Any,
    backing: KVBackingStore,
    current_token: int,
) -> Tuple[Dict[int, np.ndarray], List[int], float]:
    """Full-pool scoring forward returning per-query-head attention.

    Returns ({layer: [query_heads, n_positions] float64}, positions, seconds).
    """

    scoring_state = _clone_full_state(runner, state, backing, int(current_token), 1)
    positions = backing.positions()
    try:
        _, record, forward_s = runner.model.forward_one(
            scoring_state, int(current_token), capture_attention=True
        )
        per_head: Dict[int, np.ndarray] = {}
        for layer in range(len(scoring_state.cache)):
            maps = [
                int(v) for v in scoring_state.position_maps[int(layer)].tolist()
            ]
            index = {int(p): row for row, p in enumerate(maps)}
            raw = (
                record.oracle_attention_by_layer[int(layer)]
                .detach()
                .float()
                .cpu()
                .numpy()
                .astype(np.float64)
                .reshape(-1, len(maps))
            )
            keep = [index[int(p)] for p in positions]
            per_head[int(layer)] = raw[:, keep]
    finally:
        runner.model.release(scoring_state)
    return per_head, positions, float(forward_s)


def _headwise_rows(
    sample: Any,
    cycle: int,
    layer: int,
    attn: "torch.Tensor",
    positions: Sequence[int],
    eligible: Sequence[int],
    core: Sequence[int],
    core_budget: int,
) -> List[Dict[str, Any]]:
    """Per-KV-head captured-mass comparison: shared head-mean top-k core vs
    each KV head's own top-k core (same per-head budget).  ``attn`` is
    [kv_heads, n] query-group-mean attention aligned with ``positions``.
    """
    row_of = {int(p): i for i, p in enumerate(positions)}
    mandatory_cols = [row_of[int(p)] for p in positions if int(p) not in set(int(e) for e in eligible)]
    eligible_cols = [row_of[int(p)] for p in eligible]
    core_cols = [row_of[int(p)] for p in core]
    attn_np = attn.double().numpy()
    rows: List[Dict[str, Any]] = []
    for head in range(attn_np.shape[0]):
        a = attn_np[head]
        total = float(a.sum())
        if total <= 0:
            continue
        own_order = sorted(eligible_cols, key=lambda c: -a[c])[: int(core_budget)]
        shared_mass = float(a[mandatory_cols].sum() + a[core_cols].sum()) / total
        own_mass = float(a[mandatory_cols].sum() + a[own_order].sum()) / total
        overlap = len(set(own_order) & set(core_cols)) / max(1, len(core_cols))
        rows.append(
            {
                "sample_id": str(sample.sample_id),
                "task": str(sample.task),
                "cycle": int(cycle),
                "layer": int(layer),
                "head": int(head),
                "shared_mass": shared_mass,
                "own_mass": own_mass,
                "own_shared_overlap": float(overlap),
            }
        )
    return rows


def run_qkv_decomposition(config_path: Path, repository_root: Path) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cfg = load_discovery_config(str(repository_root / str(config["base_config"])))
    cfg.experiment_name = str(config["experiment_name"])
    for key, value in dict(config.get("model_overrides") or {}).items():
        setattr(cfg.model, str(key), value)
    apply_named_overrides(cfg.runtime, config.get("runtime_overrides"), "runtime")
    allow_prompt_truncation = bool(config.get("allow_prompt_truncation", False))
    cfg.tasks = dict(config["task_overrides"])
    cfg.runtime.seed = int(config["data_seed"])
    cfg.runtime.run_id = str(config["runtime_run_id"])
    sample_ids = set(str(v) for v in config["sample_ids"])
    expected = int(config.get("expected_sample_count", len(sample_ids)))
    output_root = repository_root / str(config["output_run"])
    output_root.mkdir(parents=True, exist_ok=True)

    cycles = int(config["control_cycles"])
    total_budget = int(config["total_budget"])
    sink_size = int(config["sink_size"])
    recent_size = int(config["recent_size"])
    core_budget = int(config["core_budget"])
    probe = dict(config.get("decomposition_probe") or {})
    dev_ids = set(str(v) for v in probe.get("dev_sample_ids", []))
    head_stride = int(probe.get("head_cycle_stride", 4))
    swap_stride = int(probe.get("swap_cycle_stride", 4))
    swap_offsets = [int(v) for v in probe.get("swap_offsets", [1, 2, 4, 8, 16, 32])]
    head_window = int(probe.get("head_window", 16))
    head_top = int(probe.get("head_top", 8))
    head_random = int(probe.get("head_random", 16))
    headwise_probe = bool(probe.get("headwise_probe", False))
    rng = np.random.default_rng(int(config["data_seed"]))

    cfg.anchor_steps = [0]
    runner = CandidatePullbackRunner(cfg, repository_root)
    samples, _ = load_discovery_tasks(cfg)
    selected = [s for s in samples if str(s.sample_id) in sample_ids]
    if {str(s.sample_id) for s in selected} != sample_ids or len(selected) != expected:
        raise RuntimeError("configured decomposition samples were not loaded")

    token_parts: List[pd.DataFrame] = []
    head_parts: List[pd.DataFrame] = []
    headwise_rows: List[Dict[str, Any]] = []
    swap_parts: List[pd.DataFrame] = []
    step_rows: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    started = time.perf_counter()
    model_info = runner.model.load()
    query_heads = int(model_info["num_attention_heads"])
    kv_heads = int(model_info["num_key_value_heads"])
    head_dim = int(model_info.get("head_dim") or 0) or int(
        model_info["hidden_size"]
    ) // query_heads
    layers = list(range(int(model_info["num_layers"])))
    # The attention hook records per-KV-head attention (query-group means):
    # 8 rows for Qwen3-8B.  V is also per KV head, so the exact removal
    # perturbation is computed at KV-group granularity with group-mean
    # projector grams (mean of the 4 query heads' W_O slices per group).
    grams_query = head_projector_grams(runner, layers, query_heads, head_dim)
    group_size = query_heads // kv_heads
    grams = {
        layer: gram.reshape(kv_heads, group_size, head_dim, head_dim).mean(dim=1)
        for layer, gram in grams_query.items()
    }
    kv_map = list(range(kv_heads))

    try:
        for sample_index, sample in enumerate(selected, start=1):
            reference = runner.model.generate_reference(
                sample.sample_id, sample.task, sample.prompt
            )
            _check_prompt_truncation(
                reference, str(sample.sample_id), allow_prompt_truncation
            )
            is_dev = str(sample.sample_id) in dev_ids
            try:
                prompt_ids = [int(v) for v in reference.prompt_token_ids]
                pieces = [
                    runner.model.tokenizer.convert_ids_to_tokens([value])[0]
                    for value in prompt_ids
                ]
                counts = pd.Series(pieces).value_counts()
                token_class = {
                    position: classify_token(piece, int(counts[piece]))
                    for position, piece in enumerate(pieces)
                }
                needle_positions = _needle_positions(sample, pieces)
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
                full_backing = KVBackingStore()
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
                sample_token_rows: List[pd.DataFrame] = []
                for cycle in range(cycles):
                    backing.update(runner, compressed_state)
                    full_backing.update(runner, full_state)
                    per_head, positions, scoring_s = _scoring_forward_per_head(
                        runner, compressed_state, backing, current_token
                    )
                    attn_mean_by_layer: Dict[int, np.ndarray] = {}
                    frame_rows: Dict[str, List[Any]] = {
                        key: []
                        for key in (
                            "cycle",
                            "layer",
                            "position",
                            "attn",
                            "delta",
                            "pv",
                            "vn",
                            "apv",
                            "rank",
                            "margin",
                            "in_core",
                            "token_class",
                            "is_needle",
                        )
                    }
                    cores_by_layer: Dict[int, Tuple[int, ...]] = {}
                    scores_by_layer: Dict[int, np.ndarray] = {}
                    features_by_layer: Dict[int, Dict[str, np.ndarray]] = {}
                    _, _, eligible = mandatory_and_eligible(
                        positions, sink_size, recent_size
                    )
                    head_subset_by_layer: Dict[int, List[int]] = {}
                    for layer in layers:
                        attn = torch.from_numpy(per_head[layer])  # [32, n]
                        _, values = backing.layer_arrays(layer)
                        v = values[0].float()  # [kv, n, dim]
                        delta, pv, _ = exact_removal_delta(
                            attn, v, grams[layer], kv_map
                        )
                        attn_mean = attn.double().mean(dim=0).numpy()
                        vn = v.norm(dim=-1).mean(dim=0)  # mean over kv heads
                        rank, margin, core = rank_and_margin(
                            attn_mean, positions, eligible, core_budget
                        )
                        cores_by_layer[layer] = core
                        scores_by_layer[layer] = attn_mean
                        attn_mean_by_layer[layer] = attn_mean
                        in_core = set(core)
                        delta_mean = delta.double().mean(dim=0).numpy()
                        pv_mean = pv.double().mean(dim=0).numpy()
                        apv_mean = (attn * pv).double().mean(dim=0).numpy()
                        features_by_layer[layer] = {
                            "delta": delta_mean,
                            "pv": pv_mean,
                            "vn": vn,
                        }
                        frame_rows["cycle"].extend([cycle] * len(positions))
                        frame_rows["layer"].extend([layer] * len(positions))
                        frame_rows["position"].extend(positions)
                        frame_rows["attn"].extend(attn_mean.tolist())
                        frame_rows["delta"].extend(delta_mean.tolist())
                        frame_rows["pv"].extend(pv_mean.tolist())
                        frame_rows["vn"].extend(vn.tolist())
                        frame_rows["apv"].extend(apv_mean.tolist())
                        frame_rows["rank"].extend(
                            [rank.get(int(p), -1) for p in positions]
                        )
                        frame_rows["margin"].extend(
                            [margin.get(int(p), float("nan")) for p in positions]
                        )
                        frame_rows["in_core"].extend(
                            [int(p) in in_core for p in positions]
                        )
                        frame_rows["token_class"].extend(
                            [token_class.get(int(p), "generated") for p in positions]
                        )
                        frame_rows["is_needle"].extend(
                            [int(p) in needle_positions for p in positions]
                        )
                        if is_dev and cycle % head_stride == 0:
                            head_subset_by_layer[layer] = _head_subset(
                                positions,
                                eligible,
                                rank,
                                core_budget,
                                head_window,
                                head_top,
                                head_random,
                                rng,
                            )
                            _append_head_rows(
                                head_parts,
                                sample,
                                cycle,
                                layer,
                                head_subset_by_layer[layer],
                                positions,
                                attn,
                                delta,
                                pv,
                            )
                        if headwise_probe:
                            headwise_rows.extend(
                                _headwise_rows(
                                    sample,
                                    cycle,
                                    layer,
                                    attn,
                                    positions,
                                    eligible,
                                    core,
                                    core_budget,
                                )
                            )
                    selection = _selection_from_scores(
                        "qk_pool",
                        positions,
                        eligible,
                        cores_by_layer,
                        scores_by_layer,
                    )
                    sample_token_rows.append(pd.DataFrame(frame_rows))
                    if is_dev and cycle % swap_stride == 0:
                        swap_parts.extend(
                            _swap_oracle_rows(
                                runner,
                                sample,
                                cycle,
                                compressed_state,
                                backing,
                                full_state,
                                full_backing,
                                current_token,
                                selection,
                                positions,
                                eligible,
                                attn_mean_by_layer,
                                features_by_layer,
                                core_budget,
                                swap_offsets,
                                initial_cache,
                                rolling_cache,
                            )
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
                    trajectory_rows = _advance_full_state(
                        runner, full_state, current_token, new_tokens, rollout.logits
                    )
                    step_rows.append(
                        {
                            "sample_id": str(sample.sample_id),
                            "task": str(sample.task),
                            "cycle": int(cycle),
                            "generated_token_id": int(new_tokens[0]),
                            "pool_scoring_forward_time_s": scoring_s,
                            **trajectory_rows[0],
                        }
                    )
                    compressed_state = rollout.state
                    current_token = int(new_tokens[-1])
                    generated.extend(int(v) for v in new_tokens)
                sample_frame = pd.concat(sample_token_rows, ignore_index=True)
                sample_frame["sample_id"] = str(sample.sample_id)
                sample_frame["task"] = str(sample.task)
                token_parts.append(sample_frame)
                mean_kl = float(np.mean([r["exact_kl"] for r in step_rows[-cycles:]]))
                summaries.append(
                    {
                        "sample_id": str(sample.sample_id),
                        "task": str(sample.task),
                        "mean_trajectory_exact_kl": mean_kl,
                        **_metric_row(runner, sample, "qk_pool", generated, mean_kl),
                    }
                )
                atomic_frame(
                    pd.concat(token_parts, ignore_index=True),
                    output_root / "partial_token_rows.parquet",
                )
                atomic_json(
                    output_root / "progress.json",
                    {
                        "completed_samples": sample_index,
                        "expected_samples": expected,
                        "elapsed_s": float(time.perf_counter() - started),
                    },
                )
                print(
                    "[qkv] sample %d/%d %s kl=%.6f"
                    % (sample_index, len(selected), sample.sample_id, mean_kl),
                    flush=True,
                )
            finally:
                runner.model.release(reference)
    finally:
        runner.model.close()
    token_frame = pd.concat(token_parts, ignore_index=True)
    atomic_frame(token_frame, output_root / "token_rows.parquet")
    if head_parts:
        atomic_frame(
            pd.concat(head_parts, ignore_index=True),
            output_root / "head_rows.parquet",
        )
    if headwise_rows:
        atomic_frame(
            pd.DataFrame(headwise_rows),
            output_root / "headwise_rows.parquet",
        )
    if swap_parts:
        atomic_frame(
            pd.concat(swap_parts, ignore_index=True),
            output_root / "swap_rows.parquet",
        )
    atomic_frame(pd.DataFrame(step_rows), output_root / "step_rows.parquet")
    atomic_frame(pd.DataFrame(summaries), output_root / "sample_summary.csv")
    atomic_json(
        output_root / "summary.json",
        {
            "experiment": str(config["experiment_name"]),
            "samples": sorted(sample_ids),
            "control_cycles": cycles,
            "total_budget": total_budget,
            "core_budget": core_budget,
            "model_info": model_info,
            "query_heads": query_heads,
            "kv_heads": kv_heads,
            "head_dim": head_dim,
            "feature_head_granularity": "kv_head",
            "sample_summaries": summaries,
            "collection_elapsed_s": float(time.perf_counter() - started),
        },
    )
    atomic_text(
        output_root / "config.yaml",
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
    )
    return output_root


def _needle_positions(sample: Any, pieces: Sequence[str]) -> set:
    """Locate needle tokens: prompt tokens containing the reference answer."""

    needles: set = set()
    references = [str(v).strip() for v in getattr(sample, "references", [])]
    if not references:
        return needles
    answer = max(references, key=len).strip()
    if len(answer) < 2:
        return needles
    compact = answer.replace(" ", "")
    for position, piece in enumerate(pieces):
        text = piece.replace("Ġ", "").replace("▁", "").strip()
        if len(text) >= 2 and (text in compact or compact == text):
            needles.add(int(position))
    return needles


def _head_subset(
    positions: Sequence[int],
    eligible: Sequence[int],
    rank: Mapping[int, int],
    core_budget: int,
    window: int,
    top: int,
    random_count: int,
    rng: np.random.Generator,
) -> List[int]:
    ordered = sorted(
        (int(p) for p in eligible), key=lambda p: rank.get(int(p), 10**9)
    )
    take = min(int(core_budget), len(ordered))
    subset = list(ordered[: int(top)])
    lo = max(0, take - int(window))
    hi = min(len(ordered), take + int(window))
    subset.extend(ordered[lo:hi])
    pool = [p for p in ordered if p not in set(subset)]
    if pool and random_count:
        chosen = rng.choice(len(pool), size=min(int(random_count), len(pool)), replace=False)
        subset.extend(pool[int(i)] for i in chosen)
    return sorted(set(int(p) for p in subset))


def _append_head_rows(
    parts: List[pd.DataFrame],
    sample: Any,
    cycle: int,
    layer: int,
    subset: Sequence[int],
    positions: Sequence[int],
    attn: torch.Tensor,
    delta: torch.Tensor,
    pv: torch.Tensor,
) -> None:
    index = {int(p): row for row, p in enumerate(positions)}
    rows = []
    for position in subset:
        row = index[int(position)]
        for head in range(attn.shape[0]):
            rows.append(
                {
                    "sample_id": str(sample.sample_id),
                    "task": str(sample.task),
                    "cycle": int(cycle),
                    "layer": int(layer),
                    "head": int(head),
                    "position": int(position),
                    "attn": float(attn[head, row]),
                    "delta": float(delta[head, row]),
                    "pv": float(pv[head, row]),
                }
            )
    parts.append(pd.DataFrame(rows))


def _swap_oracle_rows(
    runner: CandidatePullbackRunner,
    sample: Any,
    cycle: int,
    compressed_state: Any,
    backing: KVBackingStore,
    full_state: Any,
    full_backing: KVBackingStore,
    current_token: int,
    selection: CoreSelection,
    positions: Sequence[int],
    eligible: Sequence[int],
    attn_mean_by_layer: Mapping[int, np.ndarray],
    features_by_layer: Mapping[int, Mapping[str, np.ndarray]],
    core_budget: int,
    swap_offsets: Sequence[int],
    initial_cache: CacheDiscoveryConfig,
    rolling_cache: CacheDiscoveryConfig,
) -> List[pd.DataFrame]:
    """Exact 1-step KL of all-layer cutoff swaps (same-input, teacher-forced).

    Each row is one cutoff pair (inside/outside token) with its exact swap
    regret (swap_kl - base_kl) and both tokens' QK and V feature vectors
    (head-mean; layer 0 and layer-mean over all layers).
    """

    segment = _full_reference_segment(
        runner, full_state, full_backing, int(current_token), 1
    )
    base = _rollout_candidate(
        runner,
        segment,
        compressed_state,
        backing,
        int(current_token),
        selection,
        0,
        1,
        initial_cache,
        rolling_cache,
    )
    base_kl = float(base.step_rows[0]["exact_kl"])
    base.state = None
    layer0 = 0
    attn_mean = attn_mean_by_layer[int(layer0)]
    row_by_position = {int(p): r for r, p in enumerate(positions)}
    ordered = sorted(
        (int(p) for p in eligible),
        key=lambda p: (-float(attn_mean[row_by_position[p]]), p),
    )
    take = min(int(core_budget), len(ordered))

    def _feature(feature: str, position: int, layer: int) -> float:
        return float(features_by_layer[int(layer)][feature][row_by_position[position]])

    def _feature_mean(feature: str, position: int) -> float:
        return float(
            np.mean(
                [
                    _feature(feature, position, layer)
                    for layer in sorted(features_by_layer)
                ]
            )
        )

    rows = []
    for offset in swap_offsets:
        inside_index = take - int(offset)
        outside_index = take + int(offset) - 1
        if inside_index < 0 or outside_index >= len(ordered):
            continue
        inside = int(ordered[inside_index])
        outside = int(ordered[outside_index])
        swapped, layers_swapped = swap_selection_all_layers(selection, inside, outside)
        outcome = _rollout_candidate(
            runner,
            segment,
            compressed_state,
            backing,
            int(current_token),
            swapped,
            0,
            1,
            initial_cache,
            rolling_cache,
        )
        swap_kl = float(outcome.step_rows[0]["exact_kl"])
        outcome.state = None
        row = {
            "sample_id": str(sample.sample_id),
            "task": str(sample.task),
            "cycle": int(cycle),
            "offset": int(offset),
            "inside_position": inside,
            "outside_position": outside,
            "inside_attn": float(attn_mean[row_by_position[inside]]),
            "outside_attn": float(attn_mean[row_by_position[outside]]),
            "attn_margin": float(
                attn_mean[row_by_position[inside]]
                - attn_mean[row_by_position[outside]]
            ),
            "layers_swapped": int(layers_swapped),
            "base_kl": base_kl,
            "swap_kl": swap_kl,
            "swap_regret": float(swap_kl - base_kl),
        }
        for side, position in (("inside", inside), ("outside", outside)):
            for feature in ("delta", "pv", "vn"):
                row[f"{side}_{feature}_l0"] = _feature(feature, position, layer0)
                row[f"{side}_{feature}_mean"] = _feature_mean(feature, position)
        rows.append(row)
    runner.model.release()
    return [pd.DataFrame(rows)]
