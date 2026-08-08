"""Training-free selective trigger for the direct cache policy.

The trigger is developed and validated without additional replay forwards.  It
uses observable disagreement between the attention and contribution score
vectors to choose one of two already-evaluated physical cache actions at each
sample/anchor unit.
"""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

from statekv.candidate_pullback import CandidatePullbackRunner
from statekv.config import load_discovery_config
from statekv.direct_policy_replay import (
    _policy_score_vectors,
    _shared_selection,
)
from statekv.selectors import mandatory_and_eligible
from statekv.storage import atomic_frame, atomic_json, atomic_text
from statekv.tasks import load_discovery_tasks


def _normalized_entropy(score: np.ndarray, eligible_rows: np.ndarray) -> float:
    probability = np.asarray(score, dtype=np.float64)[eligible_rows]
    probability = probability[probability > 0.0]
    if probability.size <= 1:
        return 0.0
    return float(-np.dot(probability, np.log(probability)) / math.log(len(eligible_rows)))


def direct_policy_features(
    reference: Any,
    anchor: int,
    policy_scores: Mapping[str, np.ndarray],
    core_budget: int,
    sink_size: int,
    recent_size: int,
    baseline: str,
    alternative: str,
    contribution: str,
) -> Dict[str, Any]:
    """Return observable score disagreement and deterministic selection hashes."""

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
    attention = np.asarray(policy_scores[baseline], dtype=np.float64)
    blend = np.asarray(policy_scores[alternative], dtype=np.float64)
    contribution_score = np.asarray(
        policy_scores[contribution], dtype=np.float64
    )
    take = min(int(core_budget), int(eligible_rows.size))
    attention_top = eligible_rows[
        np.argsort(-attention[eligible_rows], kind="stable")[:take]
    ]
    blend_top = eligible_rows[
        np.argsort(-blend[eligible_rows], kind="stable")[:take]
    ]
    attention_set = set(attention_top.tolist())
    blend_set = set(blend_top.tolist())
    intersection = len(attention_set & blend_set)
    union = len(attention_set | blend_set)
    ranked_attention = np.sort(attention[eligible_rows])[::-1]
    margin = (
        float(ranked_attention[take - 1] - ranked_attention[take])
        if 0 < take < len(ranked_attention)
        else 0.0
    )
    attention_norm = float(np.linalg.norm(attention[eligible_rows]))
    contribution_norm = float(
        np.linalg.norm(contribution_score[eligible_rows])
    )
    cosine = float(
        np.dot(attention[eligible_rows], contribution_score[eligible_rows])
        / max(attention_norm * contribution_norm, 1.0e-30)
    )
    attention_selection = _shared_selection(
        reference,
        anchor,
        attention,
        core_budget,
        sink_size,
        recent_size,
        baseline,
    )
    alternative_selection = _shared_selection(
        reference,
        anchor,
        blend,
        core_budget,
        sink_size,
        recent_size,
        alternative,
    )
    contribution_on_attention_core = float(
        contribution_score[attention_top].sum()
    )
    return {
        "token_count": int(len(positions)),
        "eligible_token_count": int(len(eligible_rows)),
        "score_tv": float(
            0.5
            * np.abs(
                attention[eligible_rows] - contribution_score[eligible_rows]
            ).sum()
        ),
        "score_cosine": cosine,
        "selection_disagreement": float(1.0 - intersection / max(union, 1)),
        "attention_entropy": _normalized_entropy(attention, eligible_rows),
        "contribution_entropy": _normalized_entropy(
            contribution_score, eligible_rows
        ),
        "entropy_difference": float(
            _normalized_entropy(contribution_score, eligible_rows)
            - _normalized_entropy(attention, eligible_rows)
        ),
        "attention_topk_mass": float(attention[attention_top].sum()),
        "contribution_on_attention_topk_mass": contribution_on_attention_core,
        "contribution_rescue_mass": float(
            max(0.0, 1.0 - contribution_on_attention_core)
        ),
        "attention_topk_margin": margin,
        "baseline_selection_hash": str(
            attention_selection.metadata["selection_hash"]
        ),
        "alternative_selection_hash": str(
            alternative_selection.metadata["selection_hash"]
        ),
    }


def _activation(features: pd.DataFrame, rule: Mapping[str, Any]) -> pd.Series:
    values = features[str(rule["feature"])].astype(float)
    threshold = float(rule["threshold"])
    direction = str(rule["direction"])
    if direction == "high":
        return values >= threshold
    if direction == "low":
        return values <= threshold
    raise ValueError("trigger direction must be high or low")


def _conditional_rows(
    replay: pd.DataFrame,
    features: pd.DataFrame,
    baseline: str,
    alternative: str,
    rule: Mapping[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["sample_id", "task", "anchor"]
    feature_rows = features.copy()
    feature_rows["activated"] = _activation(feature_rows, rule)
    base = replay[replay["policy"] == baseline].merge(
        feature_rows[keys + ["activated"]], on=keys, validate="many_to_one"
    )
    alternate = replay[replay["policy"] == alternative].merge(
        feature_rows[keys + ["activated"]], on=keys, validate="many_to_one"
    )
    selected = pd.concat(
        [base[~base["activated"]], alternate[alternate["activated"]]],
        ignore_index=True,
    )
    selected["policy"] = str(rule["name"])
    expected = int(len(base))
    if len(selected) != expected:
        raise RuntimeError("conditional replay is not one-to-one")
    return selected, feature_rows


def _conditional_metrics(
    replay: pd.DataFrame,
    selected: pd.DataFrame,
    feature_rows: pd.DataFrame,
    baseline: str,
    rule: Mapping[str, Any],
) -> Dict[str, Any]:
    base = replay[replay["policy"] == baseline].copy()
    keys = ["sample_id", "task", "anchor"]
    base_unit = base.groupby(keys, as_index=False)["exact_kl"].mean().rename(
        columns={"exact_kl": "baseline_unit_exact_kl"}
    )
    selected_unit = (
        selected.groupby(keys, as_index=False)["exact_kl"]
        .mean()
        .rename(columns={"exact_kl": "selected_unit_exact_kl"})
    )
    units = base_unit.merge(selected_unit, on=keys, validate="one_to_one").merge(
        feature_rows[keys + ["activated"]], on=keys, validate="one_to_one"
    )
    units["unit_reduction"] = (
        units["baseline_unit_exact_kl"] - units["selected_unit_exact_kl"]
    )
    activated = units[units["activated"]]
    base_sequence = base.groupby(["sample_id", "task"])["exact_kl"].mean()
    selected_sequence = selected.groupby(["sample_id", "task"])["exact_kl"].mean()
    task_base = base.groupby("task")["exact_kl"].mean()
    task_selected = selected.groupby("task")["exact_kl"].mean()
    anchor_base = base.groupby("anchor")["exact_kl"].mean()
    anchor_selected = selected.groupby("anchor")["exact_kl"].mean()
    return {
        "rule": str(rule["name"]),
        "feature": str(rule["feature"]),
        "direction": str(rule["direction"]),
        "threshold": float(rule["threshold"]),
        "steps": int(len(selected)),
        "sequences": int(selected["sample_id"].nunique()),
        "sample_anchor_units": int(len(units)),
        "activated_units": int(len(activated)),
        "activation_rate": float(units["activated"].mean()),
        "mean_exact_kl": float(selected["exact_kl"].mean()),
        "baseline_mean_exact_kl": float(base["exact_kl"].mean()),
        "mean_exact_kl_reduction": float(
            base["exact_kl"].mean() - selected["exact_kl"].mean()
        ),
        "p95_exact_kl": float(selected["exact_kl"].quantile(0.95)),
        "baseline_p95_exact_kl": float(base["exact_kl"].quantile(0.95)),
        "maximum_exact_kl": float(selected["exact_kl"].max()),
        "baseline_maximum_exact_kl": float(base["exact_kl"].max()),
        "activated_win_rate": float(
            (activated["unit_reduction"] > 0.0).mean()
        )
        if len(activated)
        else 0.0,
        "activated_mean_unit_reduction": float(
            activated["unit_reduction"].mean()
        )
        if len(activated)
        else 0.0,
        "maximum_activated_unit_harm": float(
            np.maximum(-activated["unit_reduction"], 0.0).max()
        )
        if len(activated)
        else 0.0,
        "sample_anchor_strict_win_rate": float(
            (units["unit_reduction"] > 0.0).mean()
        ),
        "sample_anchor_nonworse_rate": float(
            (units["unit_reduction"] >= -1.0e-12).mean()
        ),
        "sequence_strict_win_rate": float(
            (selected_sequence < base_sequence).mean()
        ),
        "task_mean_improvement": {
            str(task): bool(task_selected.loc[task] < task_base.loc[task])
            for task in task_base.index
        },
        "task_mean_reduction": {
            str(task): float(task_base.loc[task] - task_selected.loc[task])
            for task in task_base.index
        },
        "anchor_mean_improvement": {
            str(anchor): bool(anchor_selected.loc[anchor] < anchor_base.loc[anchor])
            for anchor in anchor_base.index
        },
    }


def _screen_rules(features: pd.DataFrame, config: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rules: List[Dict[str, Any]] = []
    for feature in config["screen_features"]:
        for quantile in config.get("screen_quantiles", [0.25, 0.5, 0.75]):
            threshold = float(features[str(feature)].quantile(float(quantile)))
            for direction in config.get("screen_directions", ["high", "low"]):
                rules.append(
                    {
                        "name": "%s_%s_q%02d"
                        % (feature, direction, int(round(100 * float(quantile)))),
                        "feature": str(feature),
                        "direction": str(direction),
                        "threshold": threshold,
                    }
                )
    return rules


def _gate(metrics: Mapping[str, Any], gate: Mapping[str, Any]) -> Dict[str, bool]:
    activation_rate = float(metrics["activation_rate"])
    return {
        "mean_exact_kl_improves": bool(metrics["mean_exact_kl_reduction"] > 0.0),
        "p95_exact_kl_improves": bool(
            metrics["p95_exact_kl"] < metrics["baseline_p95_exact_kl"]
        ),
        "maximum_exact_kl_nonworse": bool(
            metrics["maximum_exact_kl"]
            <= metrics["baseline_maximum_exact_kl"]
        ),
        "all_task_means_improve": bool(
            all(metrics["task_mean_improvement"].values())
        ),
        "activation_rate_in_range": bool(
            activation_rate >= float(gate["minimum_activation_rate"])
            and activation_rate <= float(gate["maximum_activation_rate"])
        ),
        "activated_win_rate": bool(
            metrics["activated_win_rate"]
            >= float(gate["minimum_activated_win_rate"])
        ),
        "sample_anchor_nonworse_rate": bool(
            metrics["sample_anchor_nonworse_rate"]
            >= float(gate["minimum_sample_anchor_nonworse_rate"])
        ),
    }


def run_direct_policy_trigger(config_path: Path, repository_root: Path) -> Path:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    cfg = load_discovery_config(str(repository_root / str(config["base_config"])))
    if "task_overrides" in config:
        cfg.tasks = dict(config["task_overrides"])
    if "data_seed" in config:
        cfg.runtime.seed = int(config["data_seed"])
    if "runtime_run_id" in config:
        cfg.runtime.run_id = str(config["runtime_run_id"])
    # Retain only the final observation window during reference generation.
    setattr(cfg, "direct_policy_capture_only", True)

    output_root = repository_root / str(config["output_run"])
    output_root.mkdir(parents=True, exist_ok=True)
    source_run = repository_root / str(config["source_replay_run"])
    replay = pd.read_parquet(source_run / "physical_replay_rows.parquet")
    inventory = pd.read_parquet(source_run / "selection_inventory.parquet")
    sample_ids = set(str(value) for value in config["sample_ids"])
    anchors = [int(value) for value in config["anchors"]]
    layers = [int(value) for value in config["diagnostic_layers"]]
    windows = [int(value) for value in config["scenario_windows"]]
    baseline = str(config["baseline"])
    alternative = str(config["alternative"])
    contribution = str(config["contribution_policy"])
    score_policies = [baseline, alternative, contribution]
    core_budget = int(config["core_budget"])
    sink_size = int(config["sink_size"])
    recent_size = int(config["recent_size"])

    runner = CandidatePullbackRunner(cfg, repository_root)
    samples, _ = load_discovery_tasks(cfg)
    selected_samples = [
        sample for sample in samples if str(sample.sample_id) in sample_ids
    ]
    if {str(sample.sample_id) for sample in selected_samples} != sample_ids:
        raise RuntimeError("configured trigger samples were not loaded")
    feature_records: List[Dict[str, Any]] = []
    started = time.perf_counter()
    runner.model.load()
    try:
        for sample in selected_samples:
            reference = runner.model.generate_reference(
                sample.sample_id, sample.task, sample.prompt
            )
            try:
                for anchor in anchors:
                    scores = _policy_score_vectors(
                        runner,
                        reference,
                        anchor,
                        layers,
                        windows,
                        sink_size,
                        recent_size,
                        score_policies,
                        core_budget,
                    )
                    feature_records.append(
                        {
                            "sample_id": str(sample.sample_id),
                            "task": str(sample.task),
                            "anchor": int(anchor),
                            **direct_policy_features(
                                reference,
                                anchor,
                                scores,
                                core_budget,
                                sink_size,
                                recent_size,
                                baseline,
                                alternative,
                                contribution,
                            ),
                        }
                    )
            finally:
                runner.model.release(reference)
    finally:
        runner.model.close()

    features = pd.DataFrame(feature_records)
    source_hashes = inventory[inventory["policy"].isin([baseline, alternative])][
        ["sample_id", "task", "anchor", "policy", "selection_hash"]
    ].pivot(
        index=["sample_id", "task", "anchor"],
        columns="policy",
        values="selection_hash",
    ).reset_index()
    checked = features.merge(
        source_hashes,
        on=["sample_id", "task", "anchor"],
        validate="one_to_one",
    )
    baseline_hash_match = checked["baseline_selection_hash"] == checked[baseline]
    alternative_hash_match = (
        checked["alternative_selection_hash"] == checked[alternative]
    )
    selection_hash_match_rate = float(
        pd.concat([baseline_hash_match, alternative_hash_match]).mean()
    )
    if selection_hash_match_rate != 1.0:
        raise RuntimeError("recomputed policy selections do not match source replay")

    mode = str(config["mode"])
    if mode == "screen":
        rules = _screen_rules(features, config)
    elif mode == "validate":
        rules = [dict(config["locked_rule"])]
    else:
        raise ValueError("trigger mode must be screen or validate")

    metrics_records: List[Dict[str, Any]] = []
    conditional_by_rule: Dict[str, pd.DataFrame] = {}
    for rule in rules:
        conditional, feature_activation = _conditional_rows(
            replay, features, baseline, alternative, rule
        )
        metrics = _conditional_metrics(
            replay, conditional, feature_activation, baseline, rule
        )
        metrics_records.append(
            {
                key: value
                for key, value in metrics.items()
                if not isinstance(value, dict)
            }
        )
        conditional_by_rule[str(rule["name"])] = conditional
    metrics_frame = pd.DataFrame(metrics_records).sort_values(
        ["mean_exact_kl_reduction", "activated_win_rate"],
        ascending=[False, False],
    )

    result: Dict[str, Any] = {
        "experiment": str(config["experiment_name"]),
        "mode": mode,
        "status": "development_screen" if mode == "screen" else "locked_validation",
        "confirmatory_evidence": False,
        "source_replay_run": str(config["source_replay_run"]),
        "baseline": baseline,
        "alternative": alternative,
        "candidate_algorithms_run_per_decision": 0,
        "additional_model_replay_for_trigger_analysis": 0,
        "samples": sorted(sample_ids),
        "anchors": anchors,
        "feature_units": int(len(features)),
        "selection_hash_match_rate": selection_hash_match_rate,
        "collection_elapsed_s": float(time.perf_counter() - started),
        "scope": (
            "Observable score features are recomputed from deterministic reference "
            "trajectories and matched to stored physical baseline/alternative "
            "replays. Conditional outcomes select one stored action per sample-anchor; "
            "no additional replay forward is used."
        ),
    }
    if mode == "screen":
        result["rules_screened"] = int(len(metrics_frame))
        result["top_rules_by_mean_reduction"] = metrics_frame.head(10).to_dict(
            orient="records"
        )
    else:
        rule = rules[0]
        full_metrics = _conditional_metrics(
            replay,
            conditional_by_rule[str(rule["name"])],
            features.assign(activated=_activation(features, rule)),
            baseline,
            rule,
        )
        checks = _gate(full_metrics, config["gate"])
        result.update(
            {
                "locked_rule": rule,
                "metrics": full_metrics,
                "gate": dict(config["gate"]),
                "checks": checks,
                "passed": bool(all(checks.values())),
            }
        )
        atomic_frame(
            conditional_by_rule[str(rule["name"])],
            output_root / "conditional_replay_rows.parquet",
        )

    atomic_frame(features, output_root / "policy_features.csv")
    atomic_frame(metrics_frame, output_root / "trigger_metrics.csv")
    atomic_json(output_root / "summary.json", result)
    atomic_text(
        output_root / "config.yaml",
        yaml.safe_dump(config, sort_keys=True, allow_unicode=True),
    )
    return output_root


__all__ = ["direct_policy_features", "run_direct_policy_trigger"]
