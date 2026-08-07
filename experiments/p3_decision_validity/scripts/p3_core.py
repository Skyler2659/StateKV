"""Pure analysis primitives for P3 decision-validity experiments.

The module deliberately contains no model-loading code.  It is the single
source of truth for labels, detector inputs, component swaps, cost accounting,
and sequence-first metrics.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


FORBIDDEN_FEATURE_TOKENS = (
    "exact_kl",
    "full_reference",
    "full_state",
    "history_minus_full",
    "fresh_two_midpoint",
    "endpoint_nonlinear",
    "future_token",
    "future_attention",
    "formal_label",
)

ZERO_COST_FEATURES = (
    "retained_overlap",
    "core_turnover",
    "token_age_mean",
    "token_age_std",
    "recent_window_exits",
    "retained_attention_mass",
    "cache_occupancy",
    "selector_score_drift",
    "selected_core_score_margin",
    "compressed_residual_norm_drift",
    "compressed_sketch_l2",
    "query_norm_drift",
    "key_query_alignment_mean",
    "key_query_alignment_std",
    "attention_entropy",
    "attention_concentration",
    "sink_attention_mass",
    "recent_attention_mass",
    "core_attention_mass",
    "layer_attention_summary_drift",
    "action_norm_median",
    "action_norm_spread",
    "action_to_compressed_state_ratio",
    "reused_margin",
    "action_only_margin",
    "cheap_rank_disagreement",
)

LOW_COST_FEATURES = ("top_reused_one_midpoint_shift",)

UNIT_COLUMNS = (
    "sample_id",
    "task",
    "stage",
    "horizon",
    "target_anchor",
    "layer",
    "history_id",
)

COMPONENT_COLUMNS = {
    "all_old": "risk_all_old",
    "update_g": "risk_update_g",
    "update_f": "risk_update_f",
    "update_path": "risk_update_path",
    "update_gf": "risk_update_gf",
    "update_gp": "risk_update_gp",
    "update_fp": "risk_update_fp",
    "full_fresh": "risk_full_fresh",
    "path_q1_only": "risk_path_q1_only",
    "path_q3_only": "risk_path_q3_only",
    "single_midpoint": "risk_single_midpoint",
}

FORWARD_COST_PER_CANDIDATE = {
    "all_old": 0,
    "update_g": 0,
    "update_f": 0,
    "update_gf": 0,
    "path_q1_only": 2,
    "path_q3_only": 2,
    "single_midpoint": 2,
    "update_path": 4,
    "update_gp": 4,
    "update_fp": 4,
    "full_fresh": 4,
}


def atomic_json(path: Path, value: Any) -> None:
    """Write valid JSON atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    os.close(descriptor)
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(
                _json_safe(value),
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_frame(path: Path, frame: pd.DataFrame) -> None:
    """Write a Parquet table atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".parquet", dir=str(path.parent)
    )
    os.close(descriptor)
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fisher_variance(probability: Any, displacement: Any) -> float:
    p = np.asarray(probability, dtype=np.float64).reshape(-1)
    d = np.asarray(displacement, dtype=np.float64).reshape(-1)
    if p.shape != d.shape:
        raise ValueError("probability and displacement must have equal shape")
    centered = d - float(np.dot(p, d))
    return max(0.0, float(np.dot(p, centered * centered)))


def scalar_risk(
    gradient: Any, probability: Any, path_displacement: Any
) -> float:
    """Evaluate ``g^T P + 1/2 P^T F P`` without materializing Fisher."""
    g = np.asarray(gradient, dtype=np.float64).reshape(-1)
    p = np.asarray(probability, dtype=np.float64).reshape(-1)
    displacement = np.asarray(
        path_displacement, dtype=np.float64
    ).reshape(-1)
    if not (g.shape == p.shape == displacement.shape):
        raise ValueError("risk components must have equal shape")
    return float(
        np.dot(g, displacement)
        + 0.5 * fisher_variance(p, displacement)
    )


def component_swap_scores(
    old_gradient: Any,
    old_probability: Any,
    old_q1: Any,
    old_q3: Any,
    current_gradient: Any,
    current_probability: Any,
    current_q1: Any,
    current_q3: Any,
    current_midpoint: Any | None = None,
) -> Dict[str, float]:
    """Return the frozen old/current component matrix for one candidate."""
    old_path = 0.5 * (
        np.asarray(old_q1, dtype=np.float64)
        + np.asarray(old_q3, dtype=np.float64)
    )
    current_path = 0.5 * (
        np.asarray(current_q1, dtype=np.float64)
        + np.asarray(current_q3, dtype=np.float64)
    )
    mixed_q1 = 0.5 * (
        np.asarray(current_q1, dtype=np.float64)
        + np.asarray(old_q3, dtype=np.float64)
    )
    mixed_q3 = 0.5 * (
        np.asarray(old_q1, dtype=np.float64)
        + np.asarray(current_q3, dtype=np.float64)
    )
    single_midpoint = np.asarray(
        (
            0.5
            * (
                np.asarray(current_q1, dtype=np.float64)
                + np.asarray(current_q3, dtype=np.float64)
            )
            if current_midpoint is None
            else current_midpoint
        ),
        dtype=np.float64,
    )
    return {
        "risk_all_old": scalar_risk(
            old_gradient, old_probability, old_path
        ),
        "risk_update_g": scalar_risk(
            current_gradient, old_probability, old_path
        ),
        "risk_update_f": scalar_risk(
            old_gradient, current_probability, old_path
        ),
        "risk_update_path": scalar_risk(
            old_gradient, old_probability, current_path
        ),
        "risk_update_gf": scalar_risk(
            current_gradient, current_probability, old_path
        ),
        "risk_update_gp": scalar_risk(
            current_gradient, old_probability, current_path
        ),
        "risk_update_fp": scalar_risk(
            old_gradient, current_probability, current_path
        ),
        "risk_full_fresh": scalar_risk(
            current_gradient, current_probability, current_path
        ),
        "risk_path_q1_only": scalar_risk(
            old_gradient, old_probability, mixed_q1
        ),
        "risk_path_q3_only": scalar_risk(
            old_gradient, old_probability, mixed_q3
        ),
        # The one-midpoint rule uses the average direction as its sole
        # operating point.  The runner stores that derivative separately.
        "risk_single_midpoint": scalar_risk(
            old_gradient, old_probability, single_midpoint
        ),
    }


def deterministic_projection(
    input_dimension: int, output_dimension: int, seed: int
) -> np.ndarray:
    """Frozen Gaussian Johnson-Lindenstrauss projection."""
    if input_dimension < 1 or output_dimension < 1:
        raise ValueError("projection dimensions must be positive")
    generator = np.random.default_rng(int(seed))
    return generator.normal(
        0.0,
        1.0 / math.sqrt(float(output_dimension)),
        size=(int(output_dimension), int(input_dimension)),
    ).astype(np.float64)


def projected_l2(
    old_vector: Any,
    current_vector: Any,
    *,
    output_dimension: int,
    seed: int,
) -> float:
    old = np.asarray(old_vector, dtype=np.float64).reshape(-1)
    current = np.asarray(current_vector, dtype=np.float64).reshape(-1)
    if old.shape != current.shape:
        raise ValueError("projection inputs must have equal shape")
    projection = deterministic_projection(
        old.size, output_dimension, seed
    )
    return float(np.linalg.norm(projection @ (current - old)))


def validate_feature_schema(
    features: Sequence[str],
    *,
    allow_low_cost: bool = True,
) -> tuple[str, ...]:
    names = tuple(str(name) for name in features)
    if len(names) != len(set(names)):
        raise ValueError("detector feature names must be unique")
    allowed = set(ZERO_COST_FEATURES)
    if allow_low_cost:
        allowed.update(LOW_COST_FEATURES)
    unknown = sorted(set(names) - allowed)
    forbidden = sorted(
        name
        for name in names
        if any(token in name.lower() for token in FORBIDDEN_FEATURE_TOKENS)
    )
    if unknown or forbidden:
        raise ValueError(
            f"invalid detector schema: unknown={unknown}, "
            f"forbidden={forbidden}"
        )
    return names


def assert_no_future_columns(columns: Iterable[str]) -> None:
    lowered = [str(column).lower() for column in columns]
    invalid = [
        column
        for column in lowered
        if "future_token" in column or "future_attention" in column
    ]
    if invalid:
        raise ValueError(f"future information is forbidden: {invalid}")


def retained_overlap(old_positions: Sequence[int], current: Sequence[int]) -> float:
    old_set = set(int(value) for value in old_positions)
    current_set = set(int(value) for value in current)
    union = old_set | current_set
    return (
        float(len(old_set & current_set) / len(union))
        if union
        else 1.0
    )


def token_age_statistics(
    positions: Sequence[int], current_position: int
) -> Dict[str, float]:
    ages = np.asarray(
        [int(current_position) - int(value) for value in positions],
        dtype=np.float64,
    )
    if ages.size == 0:
        return {"token_age_mean": 0.0, "token_age_std": 0.0}
    return {
        "token_age_mean": float(ages.mean()),
        "token_age_std": float(ages.std()),
    }


def probability_entropy(probability: Any) -> float:
    p = np.asarray(probability, dtype=np.float64).reshape(-1)
    p = p[p > 0.0]
    return float(-np.dot(p, np.log(p))) if p.size else 0.0


def normalized_regret(truth: Any, predicted: Any) -> float:
    """Normalized minimization regret of the predicted top-1 choice."""
    target = np.asarray(truth, dtype=np.float64).reshape(-1)
    score = np.asarray(predicted, dtype=np.float64).reshape(-1)
    if target.shape != score.shape or target.size == 0:
        raise ValueError("truth and predicted arrays must be equal and nonempty")
    best = float(np.min(target))
    worst = float(np.max(target))
    chosen = float(target[int(np.argmin(score))])
    return float((chosen - best) / max(worst - best, 1.0e-12))


def ranking_spearman(left: Any, right: Any) -> float:
    first = np.asarray(left, dtype=np.float64).reshape(-1)
    second = np.asarray(right, dtype=np.float64).reshape(-1)
    if first.shape != second.shape or first.size < 2:
        raise ValueError("rank arrays must be equal and contain at least 2 values")
    value = spearmanr(first, second).statistic
    return 0.0 if not np.isfinite(value) else float(value)


def pairwise_accuracy(truth: Any, predicted: Any) -> float:
    target = np.asarray(truth, dtype=np.float64).reshape(-1)
    score = np.asarray(predicted, dtype=np.float64).reshape(-1)
    if target.shape != score.shape or target.size < 2:
        raise ValueError("pairwise arrays must be equal and contain >=2 values")
    correct = 0.0
    count = 0
    for left in range(target.size):
        for right in range(left + 1, target.size):
            truth_sign = np.sign(target[left] - target[right])
            score_sign = np.sign(score[left] - score[right])
            correct += (
                1.0
                if truth_sign == score_sign
                else 0.5
                if truth_sign == 0.0 or score_sign == 0.0
                else 0.0
            )
            count += 1
    return float(correct / count)


def top_margin(scores: Any) -> float:
    values = np.sort(np.asarray(scores, dtype=np.float64).reshape(-1))
    if values.size < 2:
        raise ValueError("top margin requires at least two candidates")
    return float(values[1] - values[0])


def decision_event(
    exact_risk: Any,
    fresh_risk: Any,
    reused_risk: Any,
    epsilon: float,
) -> Dict[str, Any]:
    """Create all frozen decision-staleness labels for one unit."""
    exact = np.asarray(exact_risk, dtype=np.float64).reshape(-1)
    fresh = np.asarray(fresh_risk, dtype=np.float64).reshape(-1)
    reused = np.asarray(reused_risk, dtype=np.float64).reshape(-1)
    if not (exact.shape == fresh.shape == reused.shape) or exact.size < 2:
        raise ValueError("decision arrays must be equal and contain >=2 values")
    exact_top = int(np.argmin(exact))
    fresh_top = int(np.argmin(fresh))
    reused_top = int(np.argmin(reused))
    reuse_regret = normalized_regret(exact, reused)
    fresh_regret = normalized_regret(exact, fresh)
    return {
        "exact_top_index": exact_top,
        "fresh_top_index": fresh_top,
        "reused_top_index": reused_top,
        "top1_stale": reused_top != fresh_top,
        "exact_optimum_stale": reused_top != exact_top,
        "harmful_stale": reuse_regret > float(epsilon),
        "reuse_normalized_regret": reuse_regret,
        "fresh_normalized_regret": fresh_regret,
        "refresh_benefit": reuse_regret - fresh_regret,
        "fresh_reused_spearman": ranking_spearman(fresh, reused),
        "fresh_exact_spearman": ranking_spearman(fresh, exact),
        "reused_exact_spearman": ranking_spearman(reused, exact),
        "fresh_pairwise_accuracy": pairwise_accuracy(exact, fresh),
        "reused_pairwise_accuracy": pairwise_accuracy(exact, reused),
        "pairwise_degradation": (
            pairwise_accuracy(exact, fresh)
            - pairwise_accuracy(exact, reused)
        ),
        "fresh_margin": top_margin(fresh),
        "reused_margin": top_margin(reused),
    }


def choose_harmful_epsilon(
    regrets: Any, grid: Sequence[float]
) -> Dict[str, Any]:
    values = np.asarray(regrets, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("epsilon selection needs at least one regret")
    rows = [
        {
            "epsilon": float(epsilon),
            "event_rate": float(np.mean(values > float(epsilon))),
        }
        for epsilon in grid
    ]
    feasible = [
        row for row in rows if 0.10 <= row["event_rate"] <= 0.60
    ]
    selected = (
        min(feasible, key=lambda row: row["epsilon"])
        if feasible
        else min(
            rows,
            key=lambda row: (
                abs(row["event_rate"] - 0.30),
                row["epsilon"],
            ),
        )
    )
    return {"selected": selected["epsilon"], "rows": rows}


def detector_metrics(
    label: Any,
    decision: Any,
    regret: Any,
    *,
    task: Sequence[str] | None = None,
) -> Dict[str, Any]:
    truth = np.asarray(label, dtype=bool).reshape(-1)
    predicted = np.asarray(decision, dtype=bool).reshape(-1)
    losses = np.asarray(regret, dtype=np.float64).reshape(-1)
    if not (truth.shape == predicted.shape == losses.shape):
        raise ValueError("detector metric inputs must have equal shape")
    positives = int(truth.sum())
    missed = truth & ~predicted
    result: Dict[str, Any] = {
        "harmful_recall": (
            float((truth & predicted).sum() / positives)
            if positives
            else 1.0
        ),
        "refresh_coverage": float(predicted.mean()),
        "missed_normalized_regret": float(
            losses[missed].sum() / max(int(truth.sum()), 1)
        ),
        "missed_event_count": int(missed.sum()),
        "harmful_event_count": positives,
    }
    if task is not None:
        task_values = np.asarray(task, dtype=object).reshape(-1)
        if task_values.shape != truth.shape:
            raise ValueError("task array must match labels")
        result["task_recall"] = {}
        for name in sorted(set(task_values.tolist())):
            mask = task_values == name
            positive = int(truth[mask].sum())
            result["task_recall"][str(name)] = (
                float((truth[mask] & predicted[mask]).sum() / positive)
                if positive
                else 1.0
            )
    return result


def threshold_decision(
    values: Any, threshold: float, direction: str
) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    if direction == "gt":
        return data > float(threshold)
    if direction == "lt":
        return data < float(threshold)
    raise ValueError("threshold direction must be 'gt' or 'lt'")


def prefilter_coverage(
    prefilter_scores: Any,
    target_scores: Any,
    k: int,
) -> Dict[str, float]:
    cheap = np.asarray(prefilter_scores, dtype=np.float64).reshape(-1)
    target = np.asarray(target_scores, dtype=np.float64).reshape(-1)
    if cheap.shape != target.shape or not 1 <= int(k) <= cheap.size:
        raise ValueError("invalid prefilter arrays or k")
    kept = set(np.argsort(cheap, kind="stable")[: int(k)].tolist())
    optimum = int(np.argmin(target))
    return {
        "coverage": float(optimum in kept),
        "false_elimination": float(optimum not in kept),
        "probe_savings": float(1.0 - int(k) / cheap.size),
    }


def forward_cost(
    primitive: str,
    *,
    candidate_count: int,
    probed_count: int | None = None,
) -> int:
    if primitive not in FORWARD_COST_PER_CANDIDATE:
        raise ValueError(f"unknown refresh primitive: {primitive}")
    count = int(
        candidate_count if probed_count is None else probed_count
    )
    if count < 0 or count > int(candidate_count):
        raise ValueError("invalid probed candidate count")
    return int(FORWARD_COST_PER_CANDIDATE[primitive] * count)


def mean_rank_disagreement(score_sets: Sequence[Any]) -> float:
    arrays = [
        np.asarray(values, dtype=np.float64).reshape(-1)
        for values in score_sets
    ]
    if len(arrays) < 2 or len({array.shape for array in arrays}) != 1:
        raise ValueError("rank disagreement needs >=2 equal score arrays")
    ranks = np.stack([rankdata(array) for array in arrays], axis=0)
    return float(np.mean(np.std(ranks, axis=0)))


def sequence_first(
    rows: pd.DataFrame,
    value_columns: Sequence[str],
    group_columns: Sequence[str] = ("sample_id", "task", "stage"),
) -> pd.DataFrame:
    missing = sorted(
        set(group_columns).union(value_columns) - set(rows.columns)
    )
    if missing:
        raise ValueError(f"sequence-first columns missing: {missing}")
    return (
        rows.groupby(list(group_columns), sort=True)[list(value_columns)]
        .mean()
        .reset_index()
    )


def all_numeric_finite(frame: pd.DataFrame) -> bool:
    numeric = frame.select_dtypes(include=[np.number])
    return bool(np.isfinite(numeric.to_numpy()).all())


def isolation_check(
    allocations: Mapping[str, Sequence[str]],
    forbidden_ids: Sequence[str],
) -> Dict[str, Any]:
    sets = {
        str(stage): set(str(value) for value in values)
        for stage, values in allocations.items()
    }
    overlaps: Dict[str, list[str]] = {}
    names = sorted(sets)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = sorted(sets[left] & sets[right])
            if overlap:
                overlaps[f"{left}__{right}"] = overlap
    excluded = sorted(
        set().union(*sets.values()) & set(str(value) for value in forbidden_ids)
    )
    return {
        "passed": not overlaps and not excluded,
        "stage_overlaps": overlaps,
        "forbidden_intersection": excluded,
    }


def physical_alignment(
    controlled: pd.DataFrame,
    physical: pd.DataFrame,
    keys: Sequence[str],
) -> pd.DataFrame:
    if controlled.duplicated(list(keys)).any():
        raise ValueError("controlled physical keys are not unique")
    if physical.duplicated(list(keys)).any():
        raise ValueError("physical physical keys are not unique")
    merged = controlled.merge(
        physical,
        on=list(keys),
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    if set(merged["_merge"]) != {"both"}:
        raise ValueError("controlled and physical candidate sets differ")
    return merged.drop(columns="_merge")


__all__ = [
    "COMPONENT_COLUMNS",
    "FORWARD_COST_PER_CANDIDATE",
    "LOW_COST_FEATURES",
    "UNIT_COLUMNS",
    "ZERO_COST_FEATURES",
    "all_numeric_finite",
    "assert_no_future_columns",
    "atomic_frame",
    "atomic_json",
    "choose_harmful_epsilon",
    "component_swap_scores",
    "decision_event",
    "detector_metrics",
    "deterministic_projection",
    "forward_cost",
    "isolation_check",
    "mean_rank_disagreement",
    "normalized_regret",
    "pairwise_accuracy",
    "physical_alignment",
    "prefilter_coverage",
    "probability_entropy",
    "projected_l2",
    "ranking_spearman",
    "retained_overlap",
    "scalar_risk",
    "sequence_first",
    "sha256_file",
    "threshold_decision",
    "token_age_statistics",
    "top_margin",
    "validate_feature_schema",
]
