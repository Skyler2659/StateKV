"""Evidence analysis for the P1 dynamicity, P2 eviction, and P3 tail gates."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

from statekv.storage import atomic_frame, atomic_json


def _load_config(path: Path) -> Mapping[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _run_path(config: Mapping[str, Any], root: Path, stage: str) -> Path:
    return root / str(config[stage]["output_run"])


def _p1_comparisons(samples: pd.DataFrame, steps: pd.DataFrame) -> List[Dict[str, Any]]:
    dynamic = samples[samples["policy"] == "dynamic_b3"].set_index("sample_id")
    rows: List[Dict[str, Any]] = []
    for control in (
        "b2_uniform",
        "static_adaptive",
        "layer_shuffled_b3",
        "stale_b3",
    ):
        baseline = samples[samples["policy"] == control].set_index("sample_id")
        paired = dynamic.join(
            baseline,
            how="inner",
            lsuffix="_dynamic",
            rsuffix="_control",
        )
        kl_delta = (
            paired["mean_trajectory_exact_kl_dynamic"]
            - paired["mean_trajectory_exact_kl_control"]
        ).astype(float)
        quality_delta = (
            paired["official_score_dynamic"]
            - paired["official_score_control"]
        ).astype(float)
        dynamic_steps = steps[steps["policy"] == "dynamic_b3"]["exact_kl"].astype(float)
        control_steps = steps[steps["policy"] == control]["exact_kl"].astype(float)
        dynamic_p95 = float(dynamic_steps.quantile(0.95))
        control_p95 = float(control_steps.quantile(0.95))
        dynamic_cvar = float(dynamic_steps[dynamic_steps >= dynamic_p95].mean())
        control_cvar = float(control_steps[control_steps >= control_p95].mean())
        rows.append(
            {
                "control": control,
                "paired_samples": int(len(paired)),
                "dynamic_minus_control_mean_exact_kl": float(kl_delta.mean()),
                "dynamic_kl_sample_wins": int((kl_delta < -1.0e-12).sum()),
                "dynamic_kl_sample_ties": int((kl_delta.abs() <= 1.0e-12).sum()),
                "dynamic_kl_sample_losses": int((kl_delta > 1.0e-12).sum()),
                "dynamic_minus_control_official_score": float(quality_delta.mean()),
                "dynamic_quality_sample_wins": int((quality_delta > 1.0e-12).sum()),
                "dynamic_quality_sample_ties": int((quality_delta.abs() <= 1.0e-12).sum()),
                "dynamic_quality_sample_losses": int((quality_delta < -1.0e-12).sum()),
                "dynamic_minus_control_p95_exact_kl": dynamic_p95 - control_p95,
                "dynamic_minus_control_cvar95_exact_kl": dynamic_cvar - control_cvar,
                "mean_or_tail_kl_improved": bool(
                    kl_delta.mean() < 0.0
                    or dynamic_p95 < control_p95
                    or dynamic_cvar < control_cvar
                ),
                "advantage_spans_multiple_samples": bool(
                    int((kl_delta < -1.0e-12).sum()) >= 2
                ),
            }
        )
    return rows


def analyze_p1(config_path: Path, root: Path) -> Path:
    config = _load_config(config_path)
    path = _run_path(config, root, "p1")
    samples = pd.read_csv(path / "sample_results.csv")
    steps = pd.read_parquet(path / "step_rows.parquet")
    cycles = pd.read_parquet(path / "cycle_rows.parquet")
    aggregate = pd.read_csv(path / "aggregate_results.csv")
    comparisons = _p1_comparisons(samples, steps)
    mechanism_controls = {
        "static_adaptive", "layer_shuffled_b3", "stale_b3"
    }
    central = [row for row in comparisons if row["control"] in mechanism_controls]
    dynamic = aggregate[aggregate["policy"] == "dynamic_b3"].iloc[0]
    controls = aggregate[aggregate["policy"].isin(mechanism_controls)]
    result = {
        "stage": "p1_dynamic_budget_mechanism",
        "samples": sorted(samples["sample_id"].astype(str).unique()),
        "comparisons": comparisons,
        "fixed_global_requested_core_budget": bool(
            cycles["requested_core_tokens_total"].nunique() == 1
            and int(cycles["requested_core_tokens_total"].iloc[0])
            == len(json.loads(cycles["budget_by_layer_json"].iloc[0]))
            * int(config["p1"]["core_budget"])
        ),
        "dynamic_lower_mean_kl_than_each_mechanism_control": bool(
            all(row["dynamic_minus_control_mean_exact_kl"] < 0.0 for row in central)
        ),
        "dynamic_lower_p95_or_cvar_than_each_mechanism_control": bool(
            all(
                row["dynamic_minus_control_p95_exact_kl"] < 0.0
                or row["dynamic_minus_control_cvar95_exact_kl"] < 0.0
                for row in central
            )
        ),
        "dynamic_quality_not_lower_than_each_mechanism_control": bool(
            all(row["dynamic_minus_control_official_score"] >= 0.0 for row in central)
        ),
        "dynamic_advantage_spans_multiple_samples_against_each_control": bool(
            all(row["advantage_spans_multiple_samples"] for row in central)
        ),
        "dynamic_mean_exact_kl": float(dynamic["mean_exact_kl"]),
        "best_control_mean_exact_kl": float(controls["mean_exact_kl"].min()),
    }
    result["core_mechanism_supported"] = bool(
        result["fixed_global_requested_core_budget"]
        and result["dynamic_lower_mean_kl_than_each_mechanism_control"]
        and result["dynamic_quality_not_lower_than_each_mechanism_control"]
        and result[
            "dynamic_advantage_spans_multiple_samples_against_each_control"
        ]
    )
    atomic_frame(pd.DataFrame(comparisons), path / "p1_paired_comparisons.csv")
    atomic_json(path / "p1_analysis.json", result)
    return path / "p1_analysis.json"


def analyze_p2(config_path: Path, root: Path) -> Path:
    config = _load_config(config_path)
    path = _run_path(config, root, "p2")
    samples = pd.read_csv(path / "sample_results.csv")
    cycles = pd.read_parquet(path / "cycle_rows.parquet")
    aggregate = pd.read_csv(path / "aggregate_results.csv")
    profile_path = path / "runtime_profile_summary.json"
    profile = (
        json.loads(profile_path.read_text(encoding="utf-8"))
        if profile_path.exists()
        else None
    )
    comparisons: List[Dict[str, Any]] = []
    for budget, current in aggregate.groupby("total_budget", sort=True):
        b3 = current[current["policy"] == "dynamic_b3"].iloc[0]
        for policy in current["policy"]:
            if policy == "dynamic_b3":
                continue
            baseline = current[current["policy"] == policy].iloc[0]
            comparisons.append(
                {
                    "total_budget": int(budget),
                    "baseline": str(policy),
                    "baseline_minus_b3_mean_exact_kl": float(
                        baseline["mean_exact_kl"] - b3["mean_exact_kl"]
                    ),
                    "baseline_minus_b3_p95_exact_kl": float(
                        baseline["p95_exact_kl"] - b3["p95_exact_kl"]
                    ),
                    "baseline_minus_b3_cvar95_exact_kl": float(
                        baseline["cvar95_exact_kl"] - b3["cvar95_exact_kl"]
                    ),
                    "b3_minus_baseline_official_score": float(
                        b3["mean_official_score"] - baseline["mean_official_score"]
                    ),
                    "b3_minus_baseline_end_to_end_tokens_per_s": float(
                        b3["mean_end_to_end_tokens_per_s"]
                        - baseline["mean_end_to_end_tokens_per_s"]
                    ),
                }
            )
    result = {
        "stage": "p2_pure_eviction",
        "samples": sorted(samples["sample_id"].astype(str).unique()),
        "budgets": sorted(int(value) for value in samples["total_budget"].unique()),
        "persistent_cpu_kv_backing": False,
        "all_irreversible_set_inclusions_hold": bool(
            cycles["irreversible_set_inclusion"].all()
        ),
        "deleted_token_recovery_events": 0,
        "comparisons": comparisons,
        "controller_only_runtime_profile": (
            profile["policy_aggregates"] if profile is not None else None
        ),
        "memory_measurement_scope": (
            profile["measurement_scope"]
            if profile is not None
            else "missing_controller_only_profile"
        ),
        "execution_valid": bool(
            cycles["irreversible_set_inclusion"].all()
            and samples.groupby(["total_budget", "policy"])["sample_id"].nunique().min()
            == len(config["p2"]["sample_ids"])
            and profile is not None
            and bool(profile["execution_valid"])
        ),
    }
    atomic_frame(pd.DataFrame(comparisons), path / "p2_policy_comparisons.csv")
    atomic_json(path / "p2_analysis.json", result)
    return path / "p2_analysis.json"


def _auc(labels: np.ndarray, score: np.ndarray) -> float:
    positive = score[labels]
    negative = score[~labels]
    if positive.size == 0 or negative.size == 0:
        return float("nan")
    wins = 0.0
    for value in positive:
        wins += float(np.sum(value > negative))
        wins += 0.5 * float(np.sum(value == negative))
    return float(wins / (positive.size * negative.size))


def _gate_rows(cycles: pd.DataFrame, steps: pd.DataFrame) -> pd.DataFrame:
    current = cycles[cycles["policy"] == "dynamic_b3"].copy()
    step = steps[steps["policy"] == "dynamic_b3"].copy()
    columns = [
        "sample_id",
        "cycle",
        "exact_kl",
        "compressed_margin",
        "compressed_entropy",
    ]
    current = current.merge(step[columns], on=["sample_id", "cycle"], how="inner")
    current = current.sort_values(["sample_id", "cycle"])
    current["target_next_exact_kl"] = current.groupby("sample_id")["exact_kl"].shift(-1)
    return current.dropna(subset=["target_next_exact_kl"]).reset_index(drop=True)


def _fit_scalar_gate(validation: pd.DataFrame) -> Dict[str, Any]:
    target_threshold = float(validation["target_next_exact_kl"].quantile(0.95))
    labels = validation["target_next_exact_kl"].to_numpy(float) >= target_threshold
    candidates: Sequence[Tuple[str, str]] = (
        ("budget_l1_change", "high"),
        ("a2_mask_mean_jaccard", "low"),
        ("attention_mask_mean_jaccard", "low"),
        ("compressed_margin", "low"),
        ("compressed_entropy", "high"),
        ("maximum_layer_volatility", "high"),
        ("mean_layer_effective_support", "high"),
    )
    options: List[Dict[str, Any]] = []
    for feature, direction in candidates:
        raw = validation[feature].to_numpy(float)
        score = raw if direction == "high" else -raw
        auc = _auc(labels, score)
        for quantile in np.linspace(0.50, 0.95, 10):
            threshold = float(np.quantile(score, quantile))
            prediction = score >= threshold
            true_positive = int(np.sum(prediction & labels))
            false_positive = int(np.sum(prediction & ~labels))
            false_negative = int(np.sum(~prediction & labels))
            precision = true_positive / max(1, true_positive + false_positive)
            recall = true_positive / max(1, true_positive + false_negative)
            f1 = 2.0 * precision * recall / max(1.0e-12, precision + recall)
            options.append(
                {
                    "feature": feature,
                    "direction": direction,
                    "score_threshold": threshold,
                    "raw_threshold": threshold if direction == "high" else -threshold,
                    "validation_quantile": float(quantile),
                    "validation_auc": auc,
                    "validation_precision": precision,
                    "validation_recall": recall,
                    "validation_f1": f1,
                    "validation_alert_rate": float(np.mean(prediction)),
                }
            )
    best = max(
        options,
        key=lambda row: (
            row["validation_f1"],
            row["validation_precision"],
            row["validation_auc"] if np.isfinite(row["validation_auc"]) else -1.0,
            -row["validation_alert_rate"],
            row["feature"],
        ),
    )
    return {
        **best,
        "target_exact_kl_threshold": target_threshold,
        "validation_tail_events": int(labels.sum()),
        "validation_rows": int(len(validation)),
        "all_candidate_gates": options,
    }


def _evaluate_gate(gate: Mapping[str, Any], test: pd.DataFrame) -> Dict[str, Any]:
    raw = test[str(gate["feature"])].to_numpy(float)
    score = raw if gate["direction"] == "high" else -raw
    labels = (
        test["target_next_exact_kl"].to_numpy(float)
        >= float(gate["target_exact_kl_threshold"])
    )
    prediction = score >= float(gate["score_threshold"])
    true_positive = int(np.sum(prediction & labels))
    false_positive = int(np.sum(prediction & ~labels))
    false_negative = int(np.sum(~prediction & labels))
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    prevalence = float(np.mean(labels)) if labels.size else 0.0
    return {
        "test_rows": int(len(test)),
        "test_tail_events": int(labels.sum()),
        "test_alerts": int(prediction.sum()),
        "test_precision": precision,
        "test_recall": recall,
        "test_alert_rate": float(np.mean(prediction)) if prediction.size else 0.0,
        "test_tail_prevalence": prevalence,
        "test_auc": _auc(labels, score),
        "validated_tail_signal": bool(
            labels.sum() > 0
            and precision > prevalence
            and recall >= 0.5
            and _auc(labels, score) >= 0.60
        ),
    }


def _gov81_case(calibration: Path) -> Tuple[Dict[str, Any], pd.DataFrame]:
    cycles = pd.read_parquet(calibration / "cycle_rows.parquet")
    steps = pd.read_parquet(calibration / "step_rows.parquet")
    samples = pd.read_csv(calibration / "sample_results.csv")
    cycle = cycles[cycles["sample_id"] == "gov_report:81"].copy()
    step = steps[steps["sample_id"] == "gov_report:81"].copy()
    timeline = cycle.merge(step, on=["sample_id", "task", "policy", "cycle"], how="inner")
    timeline = timeline.sort_values("cycle")
    sample = samples[samples["sample_id"] == "gov_report:81"].iloc[0]
    encodings = json.loads(sample["monitor_encodings_json"])
    encoding_17 = encodings.get("probability:token_17:17", [])
    prefix_token = int(encoding_17[-2]) if len(encoding_17) >= 2 else None
    divergence = timeline[timeline["argmax_diverged"]]
    first_divergence = int(divergence["cycle"].iloc[0]) if len(divergence) else None
    layer_count = len(json.loads(timeline["budget_by_layer_json"].iloc[0]))
    lost = timeline[
        timeline["evidence_layers_with_full_survival"] < layer_count
    ]
    first_evidence_loss = int(lost["cycle"].iloc[0]) if len(lost) else None
    risk_threshold = float(timeline["exact_kl"].quantile(0.90))
    spikes = timeline[timeline["exact_kl"] >= risk_threshold]
    first_spike = int(spikes["cycle"].iloc[0]) if len(spikes) else None
    factual_branch = (
        timeline[timeline["input_token_id"] == prefix_token]
        if prefix_token is not None
        else timeline.iloc[0:0]
    )
    factual_row = factual_branch.iloc[0].to_dict() if len(factual_branch) else {}
    result = {
        "sample_id": "gov_report:81",
        "role": "diagnostic_case_only",
        "generation_text": str(sample["generation_text"]),
        "evidence_positions": json.loads(sample["evidence_positions_json"]),
        "first_argmax_divergence_cycle": first_divergence,
        "factual_17_vs_14_choice_cycle": (
            int(factual_branch["cycle"].iloc[0]) if len(factual_branch) else None
        ),
        "first_evidence_survival_loss_cycle": first_evidence_loss,
        "first_p90_risk_spike_cycle": first_spike,
        "p90_within_case_exact_kl": risk_threshold,
        "probability_semantics": (
            "conditional probability of the final digit token after the shared live prefix"
        ),
        "full_probability_token_17_at_divergence": factual_row.get(
            "full_probability_token_17"
        ),
        "compressed_probability_token_17_at_divergence": factual_row.get(
            "compressed_probability_token_17"
        ),
        "full_probability_token_14_at_divergence": factual_row.get(
            "full_probability_token_14"
        ),
        "compressed_probability_token_14_at_divergence": factual_row.get(
            "compressed_probability_token_14"
        ),
        "budget_change_precedes_first_risk_spike": bool(
            first_spike is not None
            and first_spike > 0
            and timeline.loc[
                timeline["cycle"].between(max(0, first_spike - 4), first_spike - 1),
                "budget_l1_change",
            ].max()
            > timeline["budget_l1_change"].median()
        ),
    }
    return result, timeline


def analyze_p3(config_path: Path, root: Path) -> Path:
    config = _load_config(config_path)
    calibration = _run_path(config, root, "calibration")
    validation_path = _run_path(config, root, "p1")
    test_path = _run_path(config, root, "p3")
    gov81, timeline = _gov81_case(calibration)
    validation = _gate_rows(
        pd.read_parquet(validation_path / "cycle_rows.parquet"),
        pd.read_parquet(validation_path / "step_rows.parquet"),
    )
    test = _gate_rows(
        pd.read_parquet(test_path / "cycle_rows.parquet"),
        pd.read_parquet(test_path / "step_rows.parquet"),
    )
    gate = _fit_scalar_gate(validation)
    test_result = _evaluate_gate(gate, test)
    result = {
        "stage": "p3_tail_risk_telemetry",
        "govreport_81": gov81,
        "gate": {**gate, **test_result},
        "a2_fallback_assumed": False,
        "gate_action": "alert_only_no_fallback_policy_claim",
        "validation_samples": sorted(validation["sample_id"].astype(str).unique()),
        "test_samples": sorted(test["sample_id"].astype(str).unique()),
    }
    atomic_frame(timeline, calibration / "govreport_81_timeline.csv")
    atomic_frame(validation, test_path / "tail_gate_validation_rows.csv")
    atomic_frame(test, test_path / "tail_gate_test_rows.csv")
    atomic_json(test_path / "p3_analysis.json", result)
    return test_path / "p3_analysis.json"


# ---------------------------------------------------------------------------
# R2b 4-arm selective-refresh gate
# ---------------------------------------------------------------------------

R2B_CAPTURE_MARGIN = 0.05  # (b) trigger must beat interpolated fixed-k by this
R2B_QUALITY_TOLERANCE = 0.5  # (c) trigger vs every-arm mean official_score
R2B_CAPTURE_DENOM_EPS = 1.0e-9

R2B_PREDECLARED_GATE = (
    "R2B PREDECLARED GATE (fixed before the run): per policy, PASS iff "
    "(a) trigger Capture > 0, where Capture = (KL_never - KL_X) / (KL_never - KL_fresh) "
    "per sample with fresh = every-refresh arm (|denominator| < 1e-9 -> NaN); "
    "(b) trigger mean Capture >= interpolated fixed-k mean Capture + 0.05 at the "
    "trigger's matched mean refresh count (linear interpolation between adjacent "
    "fixed-k arms in refresh-count space; if the trigger's count is outside the "
    "fixed-k range the nearest arm is used and the comparison is flagged); "
    "(c) trigger mean official_score >= every-arm mean official_score - 0.5; "
    "(d) all irreversibility/budget flags hold (irreversible_set_inclusion and "
    "global_kv_budget_respected on every cycle of every arm). "
    "Overall PASS = at least one policy passes AND the other policy still "
    "satisfies (d). If no trigger arm ran, verdict = "
    "'no-trigger-frozen-interval-curve-only' and only the fixed-k curve plus "
    "the never/fresh bounds are reported."
)


def _capture(kl_never: float, kl_x: float, kl_fresh: float) -> float:
    """Fraction of the never->fresh KL gap recovered by arm X."""
    denominator = float(kl_never) - float(kl_fresh)
    if not np.isfinite(denominator) or abs(denominator) < R2B_CAPTURE_DENOM_EPS:
        return float("nan")
    return float((float(kl_never) - float(kl_x)) / denominator)


def _interpolated_fixed_k_capture(
    points: Sequence[Tuple[float, float]], refresh_count: float
) -> Dict[str, Any]:
    """Fixed-k capture at a matched refresh count.

    ``points`` are (mean_refresh_count, mean_capture) pairs of the fixed-k
    arms.  Inside their range: linear interpolation between adjacent arms.
    Outside: the nearest arm's capture, flagged.
    """
    ordered = sorted(
        (float(count), float(capture)) for count, capture in points
    )
    if not ordered:
        return {"reference_capture": float("nan"), "interpolated": False,
                "outside_fixed_k_range": True, "reference_arm": None}
    counts = [point[0] for point in ordered]
    target = float(refresh_count)
    if counts[0] <= target <= counts[-1] and len(ordered) >= 2:
        value = float(np.interp(target, counts, [point[1] for point in ordered]))
        return {"reference_capture": value, "interpolated": True,
                "outside_fixed_k_range": False, "reference_arm": "interpolated"}
    index = int(np.argmin([abs(count - target) for count in counts]))
    return {
        "reference_capture": float(ordered[index][1]),
        "interpolated": False,
        "outside_fixed_k_range": True,
        "reference_arm": f"nearest_fixed_k_at_{counts[index]}",
    }


def _r2b_per_sample_table(
    samples: pd.DataFrame, steps: pd.DataFrame, cycles: pd.DataFrame
) -> pd.DataFrame:
    """One row per sample x policy x arm: KL, refresh count, quality, capture."""
    kl = (
        steps.dropna(subset=["exact_kl"])
        .groupby(["sample_id", "policy", "arm"])["exact_kl"]
        .mean()
        .rename("mean_trajectory_exact_kl")
        .reset_index()
    )
    if "refresh_count" in cycles.columns:
        counts = (
            cycles.groupby(["sample_id", "policy", "arm"])["refresh_count"]
            .max()
            .rename("refresh_count")
            .reset_index()
        )
    else:
        counts = (
            cycles.groupby(["sample_id", "policy", "arm"])["ranking_refreshed"]
            .sum()
            .rename("refresh_count")
            .reset_index()
        )
    quality_columns = [
        column
        for column in (
            "sample_id", "task", "policy", "arm",
            "official_score", "rouge_l", "needle_retrieval_accuracy",
        )
        if column in samples.columns
    ]
    table = samples[quality_columns].merge(
        kl, on=["sample_id", "policy", "arm"], how="left"
    ).merge(counts, on=["sample_id", "policy", "arm"], how="left")
    table["capture"] = float("nan")
    for policy in sorted(table["policy"].unique()):
        current = table[table["policy"] == policy]
        pivot = current.set_index(["sample_id", "arm"])["mean_trajectory_exact_kl"]
        for (sample_id, arm), kl_x in pivot.items():
            try:
                kl_never = float(pivot[(sample_id, "never")])
                kl_fresh = float(pivot[(sample_id, "every")])
            except KeyError:
                continue
            table.loc[
                (table["policy"] == policy)
                & (table["sample_id"] == sample_id)
                & (table["arm"] == arm),
                "capture",
            ] = _capture(kl_never, kl_x, kl_fresh)
    return table


def _evaluate_r2b_gate(
    table: pd.DataFrame,
    cycles: pd.DataFrame,
    capture_margin: float = R2B_CAPTURE_MARGIN,
    quality_tolerance: float = R2B_QUALITY_TOLERANCE,
) -> Dict[str, Any]:
    """Evaluate the predeclared R2b gate from the per-sample table."""
    policies = sorted(table["policy"].unique())
    flags = {
        "irreversible_set_inclusion": bool(cycles["irreversible_set_inclusion"].all()),
        "global_kv_budget_respected": bool(cycles["global_kv_budget_respected"].all()),
    }
    flags_hold = bool(all(flags.values()))

    efficiency_rows: List[Dict[str, Any]] = []
    for (policy, arm), current in table.groupby(["policy", "arm"], sort=True):
        efficiency_rows.append(
            {
                "policy": str(policy),
                "arm": str(arm),
                "samples": int(current["sample_id"].nunique()),
                "mean_refresh_count": float(current["refresh_count"].mean()),
                "mean_capture": float(current["capture"].mean()),
                "capture_samples_valid": int(current["capture"].notna().sum()),
                "mean_exact_kl": float(current["mean_trajectory_exact_kl"].mean()),
                "mean_official_score": float(current["official_score"].mean()),
            }
        )
    efficiency = pd.DataFrame(efficiency_rows)

    has_trigger = bool((table["arm"] == "trigger").any())
    if not has_trigger:
        return {
            "verdict": "no-trigger-frozen-interval-curve-only",
            "reason": "no frozen trigger rule was available; the trigger arm did not run",
            "budget_flags": flags,
            "budget_flags_hold": flags_hold,
            "per_policy": [],
            "refresh_efficiency": efficiency_rows,
        }

    per_policy: List[Dict[str, Any]] = []
    for policy in policies:
        current = efficiency[efficiency["policy"] == policy].set_index("arm")
        trigger = current.loc["trigger"]
        every = current.loc["every"]
        fixed_points = [
            (float(row["mean_refresh_count"]), float(row["mean_capture"]))
            for arm, row in current.iterrows()
            if str(arm).startswith("fixed_k")
        ]
        matched = _interpolated_fixed_k_capture(
            fixed_points, float(trigger["mean_refresh_count"])
        )
        condition_a = bool(float(trigger["mean_capture"]) > 0.0)
        condition_b = bool(
            np.isfinite(matched["reference_capture"])
            and float(trigger["mean_capture"])
            >= float(matched["reference_capture"]) + capture_margin
        )
        condition_c = bool(
            float(trigger["mean_official_score"])
            >= float(every["mean_official_score"]) - quality_tolerance
        )
        condition_d = flags_hold
        per_policy.append(
            {
                "policy": str(policy),
                "trigger_mean_capture": float(trigger["mean_capture"]),
                "trigger_mean_refresh_count": float(trigger["mean_refresh_count"]),
                "trigger_mean_official_score": float(trigger["mean_official_score"]),
                "every_mean_official_score": float(every["mean_official_score"]),
                "never_mean_exact_kl": float(current.loc["never"]["mean_exact_kl"]),
                "every_mean_exact_kl": float(every["mean_exact_kl"]),
                "matched_fixed_k": matched,
                "capture_margin": float(capture_margin),
                "condition_a_capture_positive": condition_a,
                "condition_b_beats_matched_fixed_k": condition_b,
                "condition_c_quality_within_tolerance": condition_c,
                "condition_d_budget_flags_hold": condition_d,
                "gate_pass": bool(
                    condition_a and condition_b and condition_c and condition_d
                ),
            }
        )
    any_pass = any(row["gate_pass"] for row in per_policy)
    verdict = "pass" if (any_pass and flags_hold) else "fail"
    return {
        "verdict": verdict,
        "reason": (
            "at least one policy passes and budget flags hold for all policies"
            if verdict == "pass"
            else "no policy satisfied all four predeclared conditions"
        ),
        "budget_flags": flags,
        "budget_flags_hold": flags_hold,
        "per_policy": per_policy,
        "refresh_efficiency": efficiency_rows,
    }


def analyze_r2b(config_path: Path, root: Path) -> Path:
    config = _load_config(config_path)
    path = _run_path(config, root, "r2b_gate")
    samples = pd.read_csv(path / "sample_results.csv")
    steps = pd.read_parquet(path / "step_rows.parquet")
    cycles = pd.read_parquet(path / "cycle_rows.parquet")
    table = _r2b_per_sample_table(samples, steps, cycles)
    gate = _evaluate_r2b_gate(table, cycles)
    stage = dict(config["r2b_gate"])
    expected_samples = len(stage["sample_ids"])
    arms_present = sorted(table["arm"].astype(str).unique())
    result = {
        "stage": "r2b_selective_refresh_gate",
        "samples": sorted(table["sample_id"].astype(str).unique()),
        "policies": sorted(table["policy"].astype(str).unique()),
        "arms": arms_present,
        "predeclared_gate": R2B_PREDECLARED_GATE,
        "gate": gate,
        "execution_valid": bool(
            table.groupby(["policy", "arm"])["sample_id"].nunique().min()
            == expected_samples
        ),
    }
    atomic_frame(table, path / "r2b_capture_by_sample.csv")
    atomic_frame(
        pd.DataFrame(gate["refresh_efficiency"]),
        path / "r2b_refresh_efficiency.csv",
    )
    atomic_json(path / "gate_analysis.json", result)
    summary_path = path / "summary.json"
    summary: Dict[str, Any] = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.exists()
        else {}
    )
    summary["predeclared_gate"] = R2B_PREDECLARED_GATE
    summary["gate"] = gate
    atomic_json(summary_path, summary)
    return path / "gate_analysis.json"


__all__ = ["analyze_p1", "analyze_p2", "analyze_p3", "analyze_r2b"]
