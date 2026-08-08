"""Development audit for risk-consistent training-free cache proxies.

The audit is deliberately offline.  It compares several proxy families on a
shared action panel, but a deployed policy would instantiate exactly one proxy,
emit one retained set, and derive refresh from that proxy's own set regret.
"""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml
from scipy import stats

from statekv.candidate_pullback import CandidatePullbackRunner
from statekv.config import load_discovery_config
from statekv.core.decision import additive_retained_set_risk
from statekv.direct_policy_replay import (
    _policy_score_vectors,
    _replay,
    _shared_selection,
)
from statekv.selectors import mandatory_and_eligible
from statekv.storage import atomic_frame, atomic_json, atomic_text
from statekv.tasks import load_discovery_tasks


def _finite_spearman(left: Iterable[float], right: Iterable[float]) -> float:
    x = np.asarray(tuple(left), dtype=np.float64)
    y = np.asarray(tuple(right), dtype=np.float64)
    if x.size < 2 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    return float(stats.spearmanr(x, y).statistic)


def _pairwise_accuracy(prediction: np.ndarray, truth: np.ndarray) -> float:
    predicted_difference = prediction[:, None] - prediction[None, :]
    truth_difference = truth[:, None] - truth[None, :]
    mask = np.triu(np.ones(truth_difference.shape, dtype=bool), k=1)
    mask &= np.abs(truth_difference) > 1.0e-15
    if not np.any(mask):
        return float("nan")
    return float(
        np.mean(
            np.sign(predicted_difference[mask])
            == np.sign(truth_difference[mask])
        )
    )


def proxy_action_risk(
    score: np.ndarray,
    positions: Sequence[int],
    eligible_positions: Sequence[int],
    retained_core: Iterable[int],
) -> float:
    """Evaluate one retained-set action with an aligned token-cost vector."""

    values = np.asarray(score, dtype=np.float64).reshape(-1)
    normalized_positions = [int(position) for position in positions]
    if len(values) != len(normalized_positions):
        raise ValueError("score and logical positions must align")
    row_by_position = {
        position: row for row, position in enumerate(normalized_positions)
    }
    eligible = [int(position) for position in eligible_positions]
    if any(position not in row_by_position for position in eligible):
        raise ValueError("eligible position is outside the score universe")
    costs = {
        position: max(float(values[row_by_position[position]]), 0.0)
        for position in eligible
    }
    retained = set(int(position) for position in retained_core) & set(eligible)
    return additive_retained_set_risk(costs, retained)


def repair_stale_core(
    previous_core: Iterable[int],
    positions: Sequence[int],
    score: np.ndarray,
    *,
    core_budget: int,
    sink_size: int,
    recent_size: int,
) -> Tuple[int, ...]:
    """Make an old core legal at a new anchor without replacing valid members."""

    normalized_positions = [int(position) for position in positions]
    values = np.asarray(score, dtype=np.float64).reshape(-1)
    if len(values) != len(normalized_positions):
        raise ValueError("score and logical positions must align")
    _, _, eligible = mandatory_and_eligible(
        normalized_positions, int(sink_size), int(recent_size)
    )
    eligible_set = set(eligible)
    preserved = sorted(
        set(int(position) for position in previous_core) & eligible_set
    )
    take = min(int(core_budget), len(eligible))
    if len(preserved) > take:
        row_by_position = {
            position: row for row, position in enumerate(normalized_positions)
        }
        preserved = sorted(
            sorted(
                preserved,
                key=lambda position: (
                    -float(values[row_by_position[position]]),
                    position,
                ),
            )[:take]
        )
    row_by_position = {
        position: row for row, position in enumerate(normalized_positions)
    }
    remaining = [position for position in eligible if position not in set(preserved)]
    fill = sorted(
        remaining,
        key=lambda position: (-float(values[row_by_position[position]]), position),
    )[: max(0, take - len(preserved))]
    repaired = tuple(sorted(preserved + fill))
    if len(repaired) != take:
        raise RuntimeError("stale action repair did not fill the core budget")
    return repaired


def _priority_for_core(
    positions: Sequence[int], retained_core: Iterable[int]
) -> np.ndarray:
    retained = set(int(position) for position in retained_core)
    return np.asarray(
        [1.0 if int(position) in retained else 0.0 for position in positions],
        dtype=np.float64,
    )


def _decision_metrics(current: pd.DataFrame) -> Dict[str, Any]:
    actions = (
        current.groupby("candidate_action_id", as_index=False)
        .agg(
            proxy_risk=("proxy_risk", "first"),
            teacher_risk=("teacher_risk", "mean"),
            candidate_policy=("candidate_policy", "first"),
        )
        .sort_values("candidate_action_id", kind="stable")
        .reset_index(drop=True)
    )
    prediction = actions["proxy_risk"].to_numpy(dtype=np.float64)
    truth = actions["teacher_risk"].to_numpy(dtype=np.float64)
    selected = int(np.lexsort((actions["candidate_action_id"], prediction))[0])
    oracle = int(np.lexsort((actions["candidate_action_id"], truth))[0])
    span = float(np.max(truth) - np.min(truth))
    regret = float(truth[selected] - truth[oracle])
    return {
        "candidate_actions": int(len(actions)),
        "spearman": _finite_spearman(prediction, truth),
        "pairwise_accuracy": _pairwise_accuracy(prediction, truth),
        "top1_accuracy": float(selected == oracle),
        "normalized_regret": regret / max(span, 1.0e-15),
        "selected_action_id": str(actions.iloc[selected]["candidate_action_id"]),
        "selected_candidate_policy": str(
            actions.iloc[selected]["candidate_policy"]
        ),
        "oracle_action_id": str(actions.iloc[oracle]["candidate_action_id"]),
        "oracle_candidate_policy": str(actions.iloc[oracle]["candidate_policy"]),
        "selected_teacher_risk": float(truth[selected]),
        "oracle_teacher_risk": float(truth[oracle]),
    }


def summarize_alignment(rows: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate cross-action ranking without pooling decision units."""

    unit_keys = ["sample_id", "task", "anchor", "horizon", "proxy"]
    unit_records = []
    for values, current in rows.groupby(unit_keys, sort=True):
        unit_records.append(
            {**dict(zip(unit_keys, values)), **_decision_metrics(current)}
        )
    units = pd.DataFrame(unit_records)
    summaries = []
    expanded = [units, units.assign(task="all")]
    for current in expanded:
        for (proxy, task), group in current.groupby(["proxy", "task"], sort=True):
            summaries.append(
                {
                    "proxy": str(proxy),
                    "task": str(task),
                    "decision_units": int(len(group)),
                    "median_spearman": float(group["spearman"].median()),
                    "mean_pairwise_accuracy": float(
                        group["pairwise_accuracy"].mean()
                    ),
                    "mean_top1_accuracy": float(group["top1_accuracy"].mean()),
                    "mean_normalized_regret": float(
                        group["normalized_regret"].mean()
                    ),
                }
            )
    return units, pd.DataFrame(summaries)


def _binary_auc(score: np.ndarray, label: np.ndarray) -> float:
    positives = int(label.sum())
    negatives = int(len(label) - positives)
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = stats.rankdata(score, method="average")
    positive_rank_sum = float(ranks[label].sum())
    return float(
        (positive_rank_sum - positives * (positives + 1) / 2.0)
        / (positives * negatives)
    )


def summarize_refresh(rows: pd.DataFrame) -> pd.DataFrame:
    """Summarize whether proxy regret orders the benefit of physical refresh."""

    summaries = []
    expanded = [rows, rows.assign(task="all")]
    for current in expanded:
        for (proxy, task), group in current.groupby(["proxy", "task"], sort=True):
            regret = group["proxy_regret"].to_numpy(dtype=np.float64)
            benefit = group["teacher_refresh_benefit"].to_numpy(dtype=np.float64)
            beneficial = benefit > 0.0
            summaries.append(
                {
                    "proxy": str(proxy),
                    "task": str(task),
                    "refresh_events": int(len(group)),
                    "mean_proxy_regret": float(np.mean(regret)),
                    "mean_teacher_refresh_benefit": float(np.mean(benefit)),
                    "beneficial_refresh_fraction": float(np.mean(beneficial)),
                    "proxy_teacher_spearman": _finite_spearman(regret, benefit),
                    "beneficial_refresh_auc": _binary_auc(regret, beneficial),
                }
            )
    return pd.DataFrame(summaries)


def _gate(
    alignment: pd.DataFrame,
    refresh: pd.DataFrame,
    policies: Sequence[str],
    gate_config: Mapping[str, Any],
) -> Dict[str, Any]:
    records = []
    tasks = sorted(
        task for task in alignment["task"].astype(str).unique() if task != "all"
    )
    for policy in policies:
        action_rows = alignment[alignment["proxy"] == policy].set_index("task")
        refresh_rows = refresh[refresh["proxy"] == policy].set_index("task")
        checks = {
            "all_task_action_spearman": bool(
                all(
                    float(action_rows.loc[task, "median_spearman"])
                    >= float(gate_config["minimum_task_median_action_spearman"])
                    for task in tasks
                )
            ),
            "overall_action_regret": bool(
                float(action_rows.loc["all", "mean_normalized_regret"])
                <= float(gate_config["maximum_mean_normalized_regret"])
            ),
            "overall_refresh_spearman": bool(
                float(refresh_rows.loc["all", "proxy_teacher_spearman"])
                >= float(gate_config["minimum_refresh_spearman"])
            ),
            "both_tasks_refresh_nonharmful": bool(
                all(
                    float(refresh_rows.loc[task, "mean_teacher_refresh_benefit"])
                    >= 0.0
                    for task in tasks
                )
            ),
        }
        records.append(
            {
                "proxy": str(policy),
                "checks": checks,
                "passed": bool(all(checks.values())),
                "overall_action_median_spearman": float(
                    action_rows.loc["all", "median_spearman"]
                ),
                "overall_mean_normalized_regret": float(
                    action_rows.loc["all", "mean_normalized_regret"]
                ),
                "overall_refresh_spearman": float(
                    refresh_rows.loc["all", "proxy_teacher_spearman"]
                ),
                "overall_mean_teacher_refresh_benefit": float(
                    refresh_rows.loc["all", "mean_teacher_refresh_benefit"]
                ),
            }
        )
    eligible = [record for record in records if record["passed"]]
    eligible.sort(
        key=lambda record: (
            -record["overall_action_median_spearman"],
            record["overall_mean_normalized_regret"],
            -record["overall_refresh_spearman"],
            record["proxy"],
        )
    )
    return {
        "gate_config": dict(gate_config),
        "proxy_results": records,
        "eligible_proxies": [record["proxy"] for record in eligible],
        "selected_proxy": eligible[0]["proxy"] if eligible else None,
        "passed": bool(eligible),
    }


def run_proxy_alignment_audit(
    config_path: Path, repository_root: Path
) -> Path:
    """Run cross-action alignment and same-proxy refresh-regret replay."""

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    source_root = repository_root / str(config["source_run"])
    source_rows = pd.read_parquet(source_root / "physical_replay_rows.parquet")
    source_inventory = pd.read_parquet(source_root / "selection_inventory.parquet")
    cfg = load_discovery_config(str(repository_root / str(config["base_config"])))
    cfg.tasks = dict(config["task_overrides"])
    cfg.runtime.seed = int(config["data_seed"])
    cfg.runtime.run_id = str(config["runtime_run_id"])

    output_root = repository_root / str(config["output_run"])
    output_root.mkdir(parents=True, exist_ok=True)
    sample_ids = set(str(value) for value in config["sample_ids"])
    anchors = [int(value) for value in config["anchors"]]
    horizons = [int(value) for value in config["evaluation_horizons"]]
    horizon = max(horizons)
    policies = [str(value) for value in config["policies"]]
    proxies = [str(value) for value in config.get("proxies", policies)]
    refresh_proxies = [
        str(value) for value in config.get("refresh_proxies", proxies)
    ]
    if not set(proxies) <= set(policies):
        raise ValueError("alignment proxies must be included in policies")
    if not set(refresh_proxies) <= set(proxies):
        raise ValueError("refresh proxies must be included in alignment proxies")
    diagnostic_layers = [int(value) for value in config["diagnostic_layers"]]
    windows = [int(value) for value in config["scenario_windows"]]
    total_budget = int(config["total_budget"])
    sink_size = int(config["sink_size"])
    recent_size = int(config["recent_size"])
    core_budget = int(config["core_budget"])

    if set(source_rows["sample_id"].astype(str)) != sample_ids:
        raise RuntimeError("source replay samples do not match the audit protocol")
    if set(source_rows["policy"].astype(str)) != set(policies):
        raise RuntimeError("source replay policies do not match the audit protocol")

    runner = CandidatePullbackRunner(cfg, repository_root)
    samples, _ = load_discovery_tasks(cfg)
    selected_samples = [
        sample for sample in samples if str(sample.sample_id) in sample_ids
    ]
    if {str(sample.sample_id) for sample in selected_samples} != sample_ids:
        raise RuntimeError("configured audit samples were not loaded")

    alignment_rows: List[Dict[str, Any]] = []
    refresh_rows: List[Dict[str, Any]] = []
    stale_replay_rows: List[Dict[str, Any]] = []
    validation_rows: List[Dict[str, Any]] = []
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
                scores_by_anchor: Dict[int, Mapping[str, np.ndarray]] = {}
                selections_by_anchor: Dict[int, Mapping[str, Any]] = {}
                for anchor in anchors:
                    scores = _policy_score_vectors(
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
                    selections = {
                        policy: _shared_selection(
                            reference,
                            anchor,
                            scores[policy],
                            core_budget,
                            sink_size,
                            recent_size,
                            policy,
                        )
                        for policy in policies
                    }
                    scores_by_anchor[anchor] = scores
                    selections_by_anchor[anchor] = selections

                    state = reference.anchors[anchor]
                    positions = [
                        int(value) for value in state.position_maps[0].tolist()
                    ]
                    _, _, eligible = mandatory_and_eligible(
                        positions, sink_size, recent_size
                    )
                    inventory = source_inventory[
                        (source_inventory["sample_id"].astype(str) == str(sample.sample_id))
                        & (source_inventory["anchor"].astype(int) == anchor)
                    ].set_index("policy")
                    for policy, selection in selections.items():
                        observed_hash = str(selection.metadata["selection_hash"])
                        expected_hash = str(inventory.loc[policy, "selection_hash"])
                        validation_rows.append(
                            {
                                "sample_id": str(sample.sample_id),
                                "task": str(sample.task),
                                "anchor": anchor,
                                "policy": policy,
                                "observed_selection_hash": observed_hash,
                                "expected_selection_hash": expected_hash,
                                "selection_hash_matches": observed_hash == expected_hash,
                            }
                        )
                    for evaluation_horizon in horizons:
                        teacher = (
                            source_rows[
                                (source_rows["sample_id"].astype(str) == str(sample.sample_id))
                                & (source_rows["anchor"].astype(int) == anchor)
                                & (source_rows["horizon_offset"].astype(int) <= evaluation_horizon)
                            ]
                            .groupby("policy", as_index=True)["exact_kl"]
                            .mean()
                        )
                        for proxy in proxies:
                            for candidate_policy in policies:
                                candidate = selections[candidate_policy]
                                core = candidate.by_layer[0].selected_positions
                                alignment_rows.append(
                                    {
                                        "sample_id": str(sample.sample_id),
                                        "task": str(sample.task),
                                        "anchor": anchor,
                                        "horizon": evaluation_horizon,
                                        "proxy": proxy,
                                        "candidate_policy": candidate_policy,
                                        "candidate_action_id": str(
                                            candidate.metadata["selection_hash"]
                                        ),
                                        "proxy_risk": proxy_action_risk(
                                            scores[proxy], positions, eligible, core
                                        ),
                                        "teacher_risk": float(
                                            teacher.loc[candidate_policy]
                                        ),
                                    }
                                )

                for previous_anchor, anchor in zip(anchors[:-1], anchors[1:]):
                    state = reference.anchors[anchor]
                    positions = [
                        int(value) for value in state.position_maps[0].tolist()
                    ]
                    _, _, eligible = mandatory_and_eligible(
                        positions, sink_size, recent_size
                    )
                    for proxy in refresh_proxies:
                        fresh = selections_by_anchor[anchor][proxy]
                        fresh_core = tuple(fresh.by_layer[0].selected_positions)
                        previous_core = selections_by_anchor[previous_anchor][
                            proxy
                        ].by_layer[0].selected_positions
                        stale_core = repair_stale_core(
                            previous_core,
                            positions,
                            scores_by_anchor[anchor][proxy],
                            core_budget=core_budget,
                            sink_size=sink_size,
                            recent_size=recent_size,
                        )
                        stale = _shared_selection(
                            reference,
                            anchor,
                            _priority_for_core(positions, stale_core),
                            core_budget,
                            sink_size,
                            recent_size,
                            "stale_" + proxy,
                        )
                        stale_rows = _replay(
                            runner,
                            reference,
                            anchor,
                            horizon,
                            stale,
                            total_budget,
                            recent_size,
                        )
                        for row in stale_rows:
                            stale_replay_rows.append(
                                {
                                    "sample_id": str(sample.sample_id),
                                    "task": str(sample.task),
                                    "previous_anchor": previous_anchor,
                                    "anchor": anchor,
                                    "proxy": proxy,
                                    "stale_action_id": str(
                                        stale.metadata["selection_hash"]
                                    ),
                                    "fresh_action_id": str(
                                        fresh.metadata["selection_hash"]
                                    ),
                                    **row,
                                }
                            )
                        stale_frame = pd.DataFrame(stale_rows)
                        fresh_source = source_rows[
                            (source_rows["sample_id"].astype(str) == str(sample.sample_id))
                            & (source_rows["anchor"].astype(int) == anchor)
                            & (source_rows["policy"].astype(str) == proxy)
                        ]
                        fresh_proxy_risk = proxy_action_risk(
                            scores_by_anchor[anchor][proxy],
                            positions,
                            eligible,
                            fresh_core,
                        )
                        stale_proxy_risk = proxy_action_risk(
                            scores_by_anchor[anchor][proxy],
                            positions,
                            eligible,
                            stale_core,
                        )
                        for evaluation_horizon in horizons:
                            fresh_teacher = float(
                                fresh_source[
                                    fresh_source["horizon_offset"].astype(int)
                                    <= evaluation_horizon
                                ]["exact_kl"].mean()
                            )
                            stale_teacher = float(
                                stale_frame[
                                    stale_frame["horizon_offset"].astype(int)
                                    <= evaluation_horizon
                                ]["exact_kl"].mean()
                            )
                            refresh_rows.append(
                                {
                                    "sample_id": str(sample.sample_id),
                                    "task": str(sample.task),
                                    "previous_anchor": previous_anchor,
                                    "anchor": anchor,
                                    "horizon": evaluation_horizon,
                                    "proxy": proxy,
                                    "proxy_regret": max(
                                        float(stale_proxy_risk - fresh_proxy_risk),
                                        0.0,
                                    ),
                                    "fresh_proxy_risk": fresh_proxy_risk,
                                    "stale_proxy_risk": stale_proxy_risk,
                                    "fresh_teacher_risk": fresh_teacher,
                                    "stale_teacher_risk": stale_teacher,
                                    "teacher_refresh_benefit": float(
                                        stale_teacher - fresh_teacher
                                    ),
                                    "fresh_action_id": str(
                                        fresh.metadata["selection_hash"]
                                    ),
                                    "stale_action_id": str(
                                        stale.metadata["selection_hash"]
                                    ),
                                }
                            )
            finally:
                runner.model.release(reference)
    finally:
        runner.model.close()

    validation = pd.DataFrame(validation_rows)
    if validation.empty or not bool(validation["selection_hash_matches"].all()):
        raise RuntimeError("recomputed fresh actions do not match the source run")
    alignment_raw = pd.DataFrame(alignment_rows)
    alignment_units, alignment_summary = summarize_alignment(alignment_raw)
    refresh_raw = pd.DataFrame(refresh_rows)
    refresh_summary = summarize_refresh(refresh_raw)
    gate = _gate(
        alignment_summary,
        refresh_summary,
        proxies,
        config["gate"],
    )
    result = {
        "experiment": str(config["experiment_name"]),
        "status": str(
            config.get(
                "evidence_status",
                "development_teacher_alignment_and_refresh_regret_audit",
            )
        ),
        "confirmatory_evidence": bool(config.get("confirmatory_evidence", False)),
        "source_run": str(config["source_run"]),
        "samples": sorted(sample_ids),
        "policies": policies,
        "alignment_proxies": proxies,
        "refresh_proxies": refresh_proxies,
        "offline_candidate_panel_only": True,
        "deployment_candidate_algorithms_run": 0,
        "selection_refresh_contract": (
            "one additive retained-set proxy risk; selection minimizes it and "
            "refresh uses stale-action regret under the same proxy"
        ),
        "fresh_selection_hashes_verified": True,
        "gate": gate,
        "collection_elapsed_s": float(time.perf_counter() - started),
        "scope": str(
            config.get(
                "scope",
                "Teacher-forced physical replay alignment audit; not free "
                "generation or a deployment latency result.",
            )
        ),
    }
    atomic_frame(alignment_raw, output_root / "cross_action_rows.parquet")
    atomic_frame(alignment_units, output_root / "alignment_units.parquet")
    atomic_frame(alignment_summary, output_root / "alignment_summary.csv")
    atomic_frame(refresh_raw, output_root / "refresh_regret_rows.parquet")
    atomic_frame(refresh_summary, output_root / "refresh_summary.csv")
    atomic_frame(
        pd.DataFrame(stale_replay_rows), output_root / "stale_replay_rows.parquet"
    )
    atomic_frame(validation, output_root / "fresh_hash_validation.csv")
    atomic_json(output_root / "summary.json", result)
    atomic_text(
        output_root / "config.yaml",
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
    )
    return output_root


__all__ = [
    "proxy_action_risk",
    "repair_stale_core",
    "run_proxy_alignment_audit",
    "summarize_alignment",
    "summarize_refresh",
]
