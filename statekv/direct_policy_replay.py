"""Teacher-forced physical replay for direct training-free shared cache sets."""
from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import yaml

from statekv.candidate_pullback import CandidatePullbackRunner
from statekv.config import load_discovery_config
from statekv.direct_coreset_pilot import _record_attention
from statekv.direct_policy_runtime import protected_rescue_score
from statekv.direct_policy_signals import (
    adjacent_value_change_score,
    attention_head_peak_score,
    attention_temporal_volatility_score,
    diagonal_leverage_score,
    uniform_position_coverage_score,
)
from statekv.functional_probe import _condition_cache
from statekv.selectors import (
    CoreSelection,
    LayerSelection,
    mandatory_and_eligible,
)
from statekv.storage import atomic_frame, atomic_json, atomic_text
from statekv.tasks import load_discovery_tasks
from statekv.training_free_routes import scenario_token_scores
from statekv.trajectory_model import exact_distribution_metrics


PROTECTED_RESCUE_POLICIES = {
    slots: "protected_attention_rescue_m%d_shared" % slots
    for slots in (4, 8, 16)
}


def _normalized_score(score: np.ndarray, eligible_rows: np.ndarray) -> np.ndarray:
    values = np.maximum(np.asarray(score, dtype=np.float64), 0.0)
    output = np.zeros_like(values)
    denominator = float(values[eligible_rows].sum())
    if denominator > 0.0:
        output[eligible_rows] = values[eligible_rows] / denominator
    return output


def _policy_score_vectors(
    runner: CandidatePullbackRunner,
    reference: Any,
    anchor: int,
    diagnostic_layers: Sequence[int],
    windows: Sequence[int],
    sink_size: int,
    recent_size: int,
    policies: Optional[Sequence[str]] = None,
    core_budget: Optional[int] = None,
) -> Mapping[str, np.ndarray]:
    state = reference.anchors[int(anchor)]
    positions = [int(value) for value in state.position_maps[0].tolist()]
    _, _, eligible_positions = mandatory_and_eligible(
        positions, int(sink_size), int(recent_size)
    )
    row_by_position = {position: row for row, position in enumerate(positions)}
    eligible_rows = np.asarray(
        [row_by_position[position] for position in eligible_positions],
        dtype=np.int64,
    )
    protected_policies = PROTECTED_RESCUE_POLICIES
    available = {
        "attention_mean_w1_shared",
        "attention_mean_w4_shared",
        "contribution_mean_w1_shared",
        "contribution_mean_w4_shared",
        "contribution_q75_w4_shared",
        "blend_attention_contribution_25_w4_shared",
        "blend_attention_contribution_50_w4_shared",
        "blend_attention_contribution_75_w4_shared",
        "attention_head_peak_w4_shared",
        "attention_temporal_volatility_w4_shared",
        "key_diagonal_leverage_shared",
        "value_diagonal_leverage_shared",
        "value_adjacent_change_shared",
        "uniform_position_coverage_shared",
        *protected_policies.values(),
    }
    requested = set(str(value) for value in (policies or sorted(available)))
    resolved_core_budget = int(
        core_budget
        if core_budget is not None
        else runner.cfg.cache.selected_core_budget
    )
    unknown = requested - available
    if unknown:
        raise ValueError("unknown direct policies=%s" % sorted(unknown))
    blend_policies = {
        percent: "blend_attention_contribution_%d_w4_shared" % percent
        for percent in (25, 50, 75)
    }
    primitive_needed = {
        policy
        for policy in requested
        if policy
        not in set(blend_policies.values()) | set(protected_policies.values())
    }
    if requested & (
        set(blend_policies.values()) | set(protected_policies.values())
    ):
        primitive_needed.update(
            {"attention_mean_w1_shared", "contribution_mean_w4_shared"}
        )
    per_policy: Dict[str, List[np.ndarray]] = {
        policy: [] for policy in requested | primitive_needed
    }
    for layer in diagnostic_layers:
        values = (
            state.values[int(layer)][0]
            .detach()
            .float()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        keys = None
        if "key_diagonal_leverage_shared" in primitive_needed:
            keys = (
                state.keys[int(layer)][0]
                .detach()
                .float()
                .cpu()
                .numpy()
                .astype(np.float64)
            )
        if int(values.shape[1]) != len(positions):
            raise RuntimeError("diagnostic-layer token maps do not align")
        records = reference.query_records[
            max(0, int(anchor) - max(windows)) : int(anchor)
        ]
        bank = np.stack(
            [
                _record_attention(record, int(layer), int(values.shape[1]))
                for record in records
            ],
            axis=0,
        )
        specifications = {
            "attention_mean_w1_shared": (bank[-1:], "mean", False),
            "attention_mean_w4_shared": (bank[-4:], "mean", False),
            "contribution_mean_w1_shared": (bank[-1:], "mean", True),
            "contribution_mean_w4_shared": (bank[-4:], "mean", True),
            "contribution_q75_w4_shared": (bank[-4:], "q75", True),
        }
        raw: Dict[str, np.ndarray] = {}
        for policy in primitive_needed:
            if policy in specifications:
                raw[policy] = scenario_token_scores(
                    specifications[policy][0],
                    values,
                    specifications[policy][1],
                    contribution_weighted=specifications[policy][2],
                )
            elif policy == "attention_head_peak_w4_shared":
                raw[policy] = attention_head_peak_score(bank[-4:])
            elif policy == "attention_temporal_volatility_w4_shared":
                raw[policy] = attention_temporal_volatility_score(bank[-4:])
            elif policy == "key_diagonal_leverage_shared":
                if keys is None:
                    raise RuntimeError("key leverage requires anchor keys")
                raw[policy] = diagonal_leverage_score(keys, eligible_rows)
            elif policy == "value_diagonal_leverage_shared":
                raw[policy] = diagonal_leverage_score(values, eligible_rows)
            elif policy == "value_adjacent_change_shared":
                raw[policy] = adjacent_value_change_score(values)
            elif policy == "uniform_position_coverage_shared":
                raw[policy] = uniform_position_coverage_score(
                    len(positions), eligible_rows, resolved_core_budget
                )
            else:
                raise ValueError("unimplemented direct policy=%s" % policy)
        normalized = {
            policy: _normalized_score(score, eligible_rows)
            for policy, score in raw.items()
        }
        for policy, score in normalized.items():
            per_policy[policy].append(score)
        if requested & set(blend_policies.values()):
            attention_score = normalized["attention_mean_w1_shared"]
            contribution_score = normalized["contribution_mean_w4_shared"]
            for percent in (25, 50, 75):
                policy = blend_policies[percent]
                if policy not in requested:
                    continue
                weight = percent / 100.0
                per_policy[policy].append(
                    (1.0 - weight) * attention_score
                    + weight * contribution_score
                )
    aggregated = {
        policy: np.mean(np.stack(scores, axis=0), axis=0)
        for policy, scores in per_policy.items()
        if scores
    }
    for slots, policy in protected_policies.items():
        if policy not in requested:
            continue
        aggregated[policy] = protected_rescue_score(
            aggregated["attention_mean_w1_shared"],
            aggregated["contribution_mean_w4_shared"],
            eligible_rows,
            core_budget=resolved_core_budget,
            rescue_slots=slots,
        )
    return {policy: aggregated[policy] for policy in requested}


def _shared_selection(
    reference: Any,
    anchor: int,
    score: np.ndarray,
    core_budget: int,
    sink_size: int,
    recent_size: int,
    policy: str,
) -> CoreSelection:
    state = reference.anchors[int(anchor)]
    reference_positions = [
        int(value) for value in state.position_maps[0].tolist()
    ]
    _, _, eligible_positions = mandatory_and_eligible(
        reference_positions, int(sink_size), int(recent_size)
    )
    row_by_position = {
        position: row for row, position in enumerate(reference_positions)
    }
    eligible_rows = np.asarray(
        [row_by_position[position] for position in eligible_positions],
        dtype=np.int64,
    )
    take = min(int(core_budget), int(eligible_rows.size))
    chosen_rows = eligible_rows[
        np.argsort(-np.asarray(score)[eligible_rows], kind="stable")[:take]
    ]
    chosen_positions = sorted(reference_positions[int(row)] for row in chosen_rows)
    by_layer: Dict[int, LayerSelection] = {}
    for layer, position_map in state.position_maps.items():
        positions = [int(value) for value in position_map.tolist()]
        if positions != reference_positions:
            raise RuntimeError("shared policy requires aligned layer position maps")
        _, _, eligible = mandatory_and_eligible(
            positions, int(sink_size), int(recent_size)
        )
        by_layer[int(layer)] = LayerSelection(
            layer=int(layer),
            selected_positions=chosen_positions,
            eligible_positions=eligible,
            aggregate_scores=[float(value) for value in score],
            metadata={
                "source": policy,
                "physical_shared_mask": True,
                "uses_future_query": False,
                "candidate_algorithms_run": 0,
            },
        )
    digest = hashlib.sha256(
        np.asarray(chosen_positions, dtype=np.int64).tobytes()
    ).hexdigest()
    return CoreSelection(
        strategy=policy,
        horizon_condition=None,
        by_layer=by_layer,
        metadata={
            "physical_shared_mask": True,
            "selection_hash": digest,
            "candidate_algorithms_run": 0,
        },
    )


def _replay(
    runner: CandidatePullbackRunner,
    reference: Any,
    anchor: int,
    horizon: int,
    selection: CoreSelection,
    total_budget: int,
    recent_size: int,
) -> List[Dict[str, Any]]:
    cache_cfg = _condition_cache(
        runner.cfg, int(total_budget), int(recent_size)
    )
    state, fixed = runner.model.state_from_anchor(
        reference.anchors[int(anchor)], selection, cache_config=cache_cfg
    )
    current_token = int(reference.anchors[int(anchor)].query_token_id)
    rows: List[Dict[str, Any]] = []
    try:
        for offset in range(1, int(horizon) + 1):
            target_index = int(anchor + offset - 1)
            if offset > 1:
                runner.model.prune_recent_before_query(
                    state, fixed, cache_config=cache_cfg
                )
            runner._clear_controls()
            logits, record, forward_s = runner.model.forward_one(
                state, current_token, capture_attention=True
            )
            runner.model.validate_active_budget(state, cache_config=cache_cfg)
            reference_record = reference.query_records[target_index]
            if int(record.query_position) != int(reference_record.query_position):
                raise RuntimeError("compressed replay position is misaligned")
            target_token = int(reference.generated_token_ids[target_index])
            metrics = exact_distribution_metrics(
                reference.probe_logits[target_index], logits, target_token
            )
            active = [int(cache.offset) for cache in state.cache]
            rows.append(
                {
                    "horizon_offset": offset,
                    "target_index": target_index,
                    "forward_time_s": float(forward_s),
                    "active_cache_tokens_max": max(active),
                    "active_cache_tokens_min": min(active),
                    **metrics,
                }
            )
            current_token = target_token
    finally:
        runner.model.release(state)
    return rows


def _summarize(rows: pd.DataFrame, baseline: str) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    base = rows[rows["policy"] == baseline][
        ["sample_id", "anchor", "horizon_offset", "exact_kl"]
    ].rename(columns={"exact_kl": "baseline_exact_kl"})
    for policy, current in rows.groupby("policy", sort=True):
        merged = current.merge(
            base,
            on=["sample_id", "anchor", "horizon_offset"],
            validate="one_to_one",
        )
        sequence = current.groupby("sample_id")["exact_kl"].mean()
        base_sequence = rows[rows["policy"] == baseline].groupby("sample_id")[
            "exact_kl"
        ].mean()
        tail_threshold = float(current["exact_kl"].quantile(0.95))
        tail = current[current["exact_kl"] >= tail_threshold]
        records.append(
            {
                "policy": policy,
                "steps": int(len(current)),
                "sequences": int(current["sample_id"].nunique()),
                "mean_exact_kl": float(current["exact_kl"].mean()),
                "median_exact_kl": float(current["exact_kl"].median()),
                "p95_exact_kl": float(current["exact_kl"].quantile(0.95)),
                "cvar95_exact_kl": float(tail["exact_kl"].mean()),
                "maximum_exact_kl": float(current["exact_kl"].max()),
                "large_loss_rate": float((current["exact_kl"] >= 1.0).mean()),
                "mean_delta_nll": float(current["delta_nll"].mean()),
                "p95_delta_nll": float(current["delta_nll"].quantile(0.95)),
                "mean_forward_time_s": float(current["forward_time_s"].mean()),
                "step_win_rate_vs_baseline": float(
                    (merged["exact_kl"] < merged["baseline_exact_kl"]).mean()
                ),
                "sequence_win_rate_vs_baseline": float(
                    (sequence < base_sequence).mean()
                ),
                "mean_exact_kl_reduction": float(
                    (merged["baseline_exact_kl"] - merged["exact_kl"]).mean()
                ),
            }
        )
    return pd.DataFrame(records)


def _stratified_metrics(
    rows: pd.DataFrame, baseline: str, primary: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: List[Dict[str, Any]] = []
    for stratum, column in (("task", "task"), ("anchor", "anchor")):
        for value, current in rows.groupby(column, sort=True):
            for policy, policy_rows in current.groupby("policy", sort=True):
                records.append(
                    {
                        "stratum": stratum,
                        "value": str(value),
                        "policy": policy,
                        "steps": int(len(policy_rows)),
                        "mean_exact_kl": float(policy_rows["exact_kl"].mean()),
                        "p95_exact_kl": float(
                            policy_rows["exact_kl"].quantile(0.95)
                        ),
                        "maximum_exact_kl": float(
                            policy_rows["exact_kl"].max()
                        ),
                        "mean_delta_nll": float(
                            policy_rows["delta_nll"].mean()
                        ),
                    }
                )
    matched = rows.pivot_table(
        index=["sample_id", "task", "anchor"],
        columns="policy",
        values="exact_kl",
        aggfunc="mean",
    ).reset_index()
    if baseline not in matched or primary not in matched:
        raise RuntimeError("matched baseline/primary units are incomplete")
    matched["mean_exact_kl_reduction"] = matched[baseline] - matched[primary]
    matched["primary_wins"] = matched["mean_exact_kl_reduction"] > 0.0
    matched["primary_nonworse"] = (
        matched["mean_exact_kl_reduction"] >= -1.0e-12
    )
    return pd.DataFrame(records), matched


def run_direct_policy_replay(config_path: Path, repository_root: Path) -> Path:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    cfg = load_discovery_config(str(repository_root / str(config["base_config"])))
    if "task_overrides" in config:
        cfg.tasks = dict(config["task_overrides"])
    if "data_seed" in config:
        cfg.runtime.seed = int(config["data_seed"])
    if "runtime_run_id" in config:
        cfg.runtime.run_id = str(config["runtime_run_id"])
    output_root = repository_root / str(config["output_run"])
    output_root.mkdir(parents=True, exist_ok=True)
    sample_ids = set(str(value) for value in config["sample_ids"])
    anchors = [int(value) for value in config["anchors"]]
    diagnostic_layers = [int(value) for value in config["diagnostic_layers"]]
    windows = [int(value) for value in config["scenario_windows"]]
    horizon = int(config["horizon"])
    total_budget = int(config["total_budget"])
    sink_size = int(config["sink_size"])
    recent_size = int(config["recent_size"])
    core_budget = int(config["core_budget"])
    baseline = str(config["baseline"])
    primary = str(config["primary"])
    policies = [
        str(value)
        for value in config.get(
            "policies",
            [
                "attention_mean_w1_shared",
                "attention_mean_w4_shared",
                "contribution_mean_w1_shared",
                "contribution_mean_w4_shared",
                "contribution_q75_w4_shared",
            ],
        )
    ]
    if baseline not in policies or primary not in policies:
        raise ValueError("policies must include the baseline and primary")

    runner = CandidatePullbackRunner(cfg, repository_root)
    samples, _ = load_discovery_tasks(cfg)
    selected_samples = [
        sample for sample in samples if str(sample.sample_id) in sample_ids
    ]
    if {str(sample.sample_id) for sample in selected_samples} != sample_ids:
        raise RuntimeError("configured replay samples were not loaded")

    replay_rows: List[Dict[str, Any]] = []
    inventory_rows: List[Dict[str, Any]] = []
    started = time.perf_counter()
    runner.model.load()
    try:
        for sample in selected_samples:
            requested_probe_indices = sorted(
                {
                    int(anchor + offset)
                    for anchor in anchors
                    for offset in range(horizon)
                }
            )
            reference = runner.model.generate_reference(
                sample.sample_id,
                sample.task,
                sample.prompt,
                extra_probe_target_indices=requested_probe_indices,
            )
            try:
                for anchor in anchors:
                    policy_scores = _policy_score_vectors(
                        runner,
                        reference,
                        anchor,
                        diagnostic_layers,
                        windows,
                        sink_size,
                        recent_size,
                        policies,
                        core_budget,
                    )
                    baseline_selection = _shared_selection(
                        reference,
                        anchor,
                        policy_scores[baseline],
                        core_budget,
                        sink_size,
                        recent_size,
                        baseline,
                    )
                    baseline_core = set(
                        baseline_selection.by_layer[0].selected_positions
                    )
                    for policy in policies:
                        if policy not in policy_scores:
                            raise ValueError("unknown direct policy=%s" % policy)
                        score = policy_scores[policy]
                        selection = _shared_selection(
                            reference,
                            anchor,
                            score,
                            core_budget,
                            sink_size,
                            recent_size,
                            policy,
                        )
                        selected_core = set(
                            selection.by_layer[0].selected_positions
                        )
                        rescue_slots = next(
                            (
                                slots
                                for slots, name in PROTECTED_RESCUE_POLICIES.items()
                                if name == policy
                            ),
                            0,
                        )
                        inventory_rows.append(
                            {
                                "sample_id": str(sample.sample_id),
                                "task": str(sample.task),
                                "anchor": anchor,
                                "policy": policy,
                                "selected_core_tokens": core_budget,
                                "total_budget": total_budget,
                                "candidate_algorithms_run": 0,
                                "rescue_slots": int(rescue_slots),
                                "protected_attention_tokens": int(
                                    core_budget - rescue_slots
                                    if rescue_slots
                                    else core_budget
                                ),
                                "attention_baseline_core_overlap": int(
                                    len(selected_core & baseline_core)
                                ),
                                "core_changes_vs_attention": int(
                                    core_budget - len(selected_core & baseline_core)
                                ),
                                "selection_hash": str(
                                    selection.metadata["selection_hash"]
                                ),
                            }
                        )
                        for row in _replay(
                            runner,
                            reference,
                            anchor,
                            horizon,
                            selection,
                            total_budget,
                            recent_size,
                        ):
                            replay_rows.append(
                                {
                                    "sample_id": str(sample.sample_id),
                                    "task": str(sample.task),
                                    "anchor": anchor,
                                    "policy": policy,
                                    **row,
                                }
                            )
            finally:
                runner.model.release(reference)
    finally:
        runner.model.close()

    rows = pd.DataFrame(replay_rows)
    summary = _summarize(rows, baseline)
    stratified, matched = _stratified_metrics(rows, baseline, primary)
    selected = summary[summary["policy"] == primary]
    base = summary[summary["policy"] == baseline]
    if len(selected) != 1 or len(base) != 1:
        raise RuntimeError("primary or baseline replay summary is missing")
    primary_row = selected.iloc[0]
    baseline_row = base.iloc[0]
    gate_config = dict(config.get("gate", {}))
    gate_mode = str(config.get("gate_mode", "general_improvement"))
    task_means = stratified[stratified["stratum"] == "task"].pivot(
        index="value", columns="policy", values="mean_exact_kl"
    )
    anchor_means = stratified[stratified["stratum"] == "anchor"].pivot(
        index="value", columns="policy", values="mean_exact_kl"
    )
    task_improvement = {
        str(task): bool(row[primary] < row[baseline])
        for task, row in task_means.iterrows()
    }
    anchor_improvement = {
        str(anchor): bool(row[primary] < row[baseline])
        for anchor, row in anchor_means.iterrows()
    }
    sample_anchor_win_rate = float(matched["primary_wins"].mean())
    sample_anchor_nonworse_rate = float(matched["primary_nonworse"].mean())
    general_checks = {
        "mean_exact_kl_improves": bool(
            primary_row["mean_exact_kl"] < baseline_row["mean_exact_kl"]
        ),
        "p95_exact_kl_improves": bool(
            primary_row["p95_exact_kl"] < baseline_row["p95_exact_kl"]
        ),
        "majority_sequence_win_rate": bool(
            primary_row["sequence_win_rate_vs_baseline"] > 0.5
        ),
        "budget_respected": bool(
            int(rows["active_cache_tokens_max"].max()) <= total_budget
        ),
        "maximum_exact_kl_improves": bool(
            primary_row["maximum_exact_kl"]
            < baseline_row["maximum_exact_kl"]
        ),
        "all_task_means_improve": bool(all(task_improvement.values())),
        "anchor_mean_wins": bool(
            sum(anchor_improvement.values())
            >= int(gate_config.get("minimum_anchor_mean_wins", 1))
        ),
        "sample_anchor_win_rate": bool(
            sample_anchor_win_rate
            >= float(gate_config.get("minimum_sample_anchor_win_rate", 0.5))
        ),
    }
    paired_steps = rows[rows["policy"] == baseline][
        ["sample_id", "task", "anchor", "horizon_offset", "exact_kl"]
    ].rename(columns={"exact_kl": "baseline_exact_kl"}).merge(
        rows[rows["policy"] == primary][
            ["sample_id", "task", "anchor", "horizon_offset", "exact_kl"]
        ].rename(columns={"exact_kl": "primary_exact_kl"}),
        on=["sample_id", "task", "anchor", "horizon_offset"],
        validate="one_to_one",
    )
    baseline_tail_threshold = float(
        paired_steps["baseline_exact_kl"].quantile(0.95)
    )
    baseline_tail = paired_steps[
        paired_steps["baseline_exact_kl"] >= baseline_tail_threshold
    ].copy()
    paired_tail = {
        "quantile": 0.95,
        "baseline_tail_threshold": baseline_tail_threshold,
        "steps": int(len(baseline_tail)),
        "baseline_tail_mean_exact_kl": float(
            baseline_tail["baseline_exact_kl"].mean()
        ),
        "primary_on_baseline_tail_mean_exact_kl": float(
            baseline_tail["primary_exact_kl"].mean()
        ),
        "paired_baseline_tail_mean_reduction": float(
            (
                baseline_tail["baseline_exact_kl"]
                - baseline_tail["primary_exact_kl"]
            ).mean()
        ),
    }
    if gate_mode == "general_improvement":
        checks = general_checks
    elif gate_mode == "development_screen":
        checks = {
            "budget_respected": general_checks["budget_respected"],
        }
    elif gate_mode == "tail_risk":
        checks = {
            "mean_exact_kl_improves": general_checks[
                "mean_exact_kl_improves"
            ],
            "p95_exact_kl_improves": general_checks[
                "p95_exact_kl_improves"
            ],
            "cvar95_exact_kl_improves": bool(
                primary_row["cvar95_exact_kl"]
                < baseline_row["cvar95_exact_kl"]
            ),
            "paired_baseline_tail_improves": bool(
                paired_tail["paired_baseline_tail_mean_reduction"] > 0.0
            ),
            "maximum_exact_kl_nonworse": bool(
                primary_row["maximum_exact_kl"]
                <= baseline_row["maximum_exact_kl"]
            ),
            "large_loss_rate_nonworse": bool(
                primary_row["large_loss_rate"]
                <= baseline_row["large_loss_rate"]
            ),
            "all_task_means_improve": general_checks[
                "all_task_means_improve"
            ],
            "anchor_mean_wins": general_checks["anchor_mean_wins"],
            "budget_respected": general_checks["budget_respected"],
        }
    elif gate_mode == "conservative_direct":
        primary_inventory = pd.DataFrame(inventory_rows)
        primary_inventory = primary_inventory[
            primary_inventory["policy"] == primary
        ]
        maximum_core_changes = int(
            primary_inventory["core_changes_vs_attention"].max()
        )
        configured_rescue_slots = int(
            primary_inventory["rescue_slots"].max()
        )
        checks = {
            "mean_exact_kl_improves": general_checks[
                "mean_exact_kl_improves"
            ],
            "p95_exact_kl_improves": general_checks[
                "p95_exact_kl_improves"
            ],
            "cvar95_exact_kl_nonworse": bool(
                primary_row["cvar95_exact_kl"]
                <= baseline_row["cvar95_exact_kl"]
            ),
            "paired_baseline_tail_improves": bool(
                paired_tail["paired_baseline_tail_mean_reduction"] > 0.0
            ),
            "maximum_exact_kl_nonworse": bool(
                primary_row["maximum_exact_kl"]
                <= baseline_row["maximum_exact_kl"]
            ),
            "large_loss_rate_nonworse": bool(
                primary_row["large_loss_rate"]
                <= baseline_row["large_loss_rate"]
            ),
            "all_task_means_improve": general_checks[
                "all_task_means_improve"
            ],
            "anchor_mean_wins": general_checks["anchor_mean_wins"],
            "sample_anchor_nonworse_rate": bool(
                sample_anchor_nonworse_rate
                >= float(
                    gate_config.get(
                        "minimum_sample_anchor_nonworse_rate", 0.80
                    )
                )
            ),
            "action_radius_respected": bool(
                maximum_core_changes <= configured_rescue_slots
            ),
            "budget_respected": general_checks["budget_respected"],
        }
    else:
        raise ValueError("unknown gate_mode=%s" % gate_mode)
    gate = {
        "experiment": str(config["experiment_name"]),
        "status": "held_out_teacher_forced_physical_replay_pilot",
        "confirmatory_evidence": False,
        "samples": sorted(sample_ids),
        "task_overrides": config.get("task_overrides"),
        "data_seed": int(cfg.runtime.seed),
        "runtime_run_id": str(cfg.runtime.run_id),
        "anchors": anchors,
        "diagnostic_layers": diagnostic_layers,
        "horizon": horizon,
        "total_budget": total_budget,
        "candidate_algorithms_run_per_decision": 0,
        "policies_evaluated_for_research": int(rows["policy"].nunique()),
        "policies": policies,
        "primary": primary,
        "baseline": baseline,
        "primary_values": primary_row.to_dict(),
        "baseline_values": baseline_row.to_dict(),
        "checks": checks,
        "stratified_values": {
            "task_mean_improvement": task_improvement,
            "anchor_mean_improvement": anchor_improvement,
            "anchor_mean_wins": int(sum(anchor_improvement.values())),
            "sample_anchor_win_rate": sample_anchor_win_rate,
            "sample_anchor_nonworse_rate": sample_anchor_nonworse_rate,
        },
        "gate_config": gate_config,
        "gate_mode": gate_mode,
        "paired_tail_values": paired_tail,
        "action_radius_values": {
            "applicable": bool(primary in PROTECTED_RESCUE_POLICIES.values()),
            "maximum_core_changes_vs_attention": int(
                max(
                    row["core_changes_vs_attention"]
                    for row in inventory_rows
                    if row["policy"] == primary
                )
            ),
            "configured_rescue_slots": (
                int(
                    max(
                        row["rescue_slots"]
                        for row in inventory_rows
                        if row["policy"] == primary
                    )
                )
                if primary in PROTECTED_RESCUE_POLICIES.values()
                else None
            ),
        },
        "passed": bool(all(checks.values())),
        "collection_elapsed_s": float(time.perf_counter() - started),
        "scope": (
            "One shared retained set is generated from past query/value scenarios "
            "and applied to all 28 layers. Evaluation is teacher-forced physical "
            "cache replay with full-vocabulary KL, not free generation or latency."
        ),
    }
    atomic_frame(rows, output_root / "physical_replay_rows.parquet")
    atomic_frame(pd.DataFrame(inventory_rows), output_root / "selection_inventory.parquet")
    atomic_frame(summary, output_root / "metrics.csv")
    analysis_root = output_root / "analysis"
    analysis_root.mkdir(parents=True, exist_ok=True)
    atomic_frame(stratified, analysis_root / "stratified_metrics.csv")
    atomic_frame(matched, analysis_root / "matched_sample_anchor.csv")
    atomic_json(output_root / "summary.json", gate)
    atomic_text(
        output_root / "config.yaml",
        yaml.safe_dump(config, sort_keys=True, allow_unicode=True),
    )
    return output_root


__all__ = ["run_direct_policy_replay"]
