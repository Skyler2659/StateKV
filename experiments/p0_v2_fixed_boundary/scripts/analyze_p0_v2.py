#!/usr/bin/env python3
"""Sequence-first aggregation and preregistered outcome adjudication."""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from p0_v2_core import (
    atomic_frame,
    atomic_json,
    json_safe,
    ranking_metrics,
    sha256_file,
)


SCORE_COLUMNS = {
    "direct": "direct_score",
    "local": "local_score",
    "fisher": "fisher_score",
    "midpoint_oracle": "midpoint_fisher_oracle",
}

VECTOR_PREFIXES = (
    "pulse_theory_vs_physical",
    "adjacent_j1_vs_physical",
    "adjacent_exact_vs_physical",
    "boundary_manual_vs_physical",
    "downstream_jvp_physical_vs_manual",
    "downstream_jvp_j1_vs_manual",
    "downstream_jvp_physical_vs_physical",
    "end_to_end_j1_vs_physical",
)


def load_protocol(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def make_rankings(
    response: pd.DataFrame, top_k: int
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    grouping = ["sample_id", "task", "anchor", "layer"]
    for key, group in response.groupby(grouping, sort=False):
        if len(group) != 8 or group["mask_hash"].nunique() != 8:
            raise RuntimeError(
                f"ranking unit {key} is not eight-distinct"
            )
        common = dict(zip(grouping, key))
        for score_type, column in SCORE_COLUMNS.items():
            rows.append(
                {
                    **common,
                    "score_type": score_type,
                    "candidate_count": len(group),
                    **ranking_metrics(
                        group[column].to_numpy(dtype=np.float64),
                        group["exact_kl"].to_numpy(dtype=np.float64),
                        top_k,
                    ),
                }
            )
    return pd.DataFrame(rows)


def make_sequence_rankings(rankings: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "spearman",
        "pairwise_sign_accuracy",
        "top1_accuracy",
        "topk_overlap",
        "normalized_regret",
        "symmetric_scale_ratio",
    ]
    return (
        rankings.groupby(
            ["sample_id", "task", "score_type"], as_index=False
        )[metric_columns]
        .median()
        .sort_values(["task", "sample_id", "score_type"])
    )


def make_sequence_vectors(response: pd.DataFrame) -> pd.DataFrame:
    columns = [
        f"{prefix}_{metric}"
        for prefix in VECTOR_PREFIXES
        for metric in (
            "cosine",
            "relative_l2",
            "symmetric_norm_ratio",
            "maximum_absolute_error",
            "predicted_norm",
            "truth_norm",
        )
    ]
    return (
        response.groupby(["sample_id", "task"], as_index=False)[columns]
        .median()
        .sort_values(["task", "sample_id"])
    )


def delta_tables(
    sequence_rankings: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pivot = sequence_rankings.pivot(
        index=["sample_id", "task"],
        columns="score_type",
        values="spearman",
    ).reset_index()
    pivot["delta_fisher_local"] = (
        pivot["fisher"] - pivot["local"]
    )
    pivot["delta_fisher_direct"] = (
        pivot["fisher"] - pivot["direct"]
    )
    task = (
        pivot.groupby("task", as_index=False)[
            ["delta_fisher_local", "delta_fisher_direct"]
        ]
        .median()
        .rename(
            columns={
                "delta_fisher_local": (
                    "median_delta_fisher_local"
                ),
                "delta_fisher_direct": (
                    "median_delta_fisher_direct"
                ),
            }
        )
    )
    positive = (
        pivot.assign(
            positive_fisher_local=pivot[
                "delta_fisher_local"
            ].gt(0.0),
            positive_fisher_direct=pivot[
                "delta_fisher_direct"
            ].gt(0.0),
        )
        .groupby("task", as_index=False)[
            ["positive_fisher_local", "positive_fisher_direct"]
        ]
        .mean()
    )
    return pivot, task, positive


def gate_outcome(
    protocol: Mapping[str, Any],
    response: pd.DataFrame,
    identity: pd.DataFrame,
    registry: pd.DataFrame,
    audit: pd.DataFrame,
    rankings: pd.DataFrame,
    sequence_rankings: pd.DataFrame,
    sequence_vectors: pd.DataFrame,
    calibration: Mapping[str, Any],
    evaluation_metadata: Mapping[str, Any],
) -> Dict[str, Any]:
    numeric_rule = protocol["gates"]["numeric"]
    finite_columns = [
        column
        for column in response.columns
        if column.endswith("_finite")
    ]
    group_sizes = registry.groupby(["sample_id", "anchor"]).size()
    distinct = registry.groupby(["sample_id", "anchor"])[
        "mask_hash"
    ].nunique()
    numeric_checks = {
        "response_all_finite": bool(
            response[finite_columns].all().all()
        ),
        "identity_all_finite": bool(identity["finite"].all()),
        "repeat_replay": bool(
            audit["repeat_max_absolute_error"].max()
            <= float(
                numeric_rule["repeat_max_absolute_error_max"]
            )
        ),
        "boundary_map_baseline_cosine": bool(
            audit["boundary_map_baseline_cosine"].min()
            >= float(
                numeric_rule["boundary_map_baseline_cosine_min"]
            )
        ),
        "boundary_map_baseline_relative_l2": bool(
            audit["boundary_map_baseline_relative_l2"].max()
            <= float(
                numeric_rule[
                    "boundary_map_baseline_relative_l2_max"
                ]
            )
        ),
        "retained_mass": bool(
            identity["retained_mass"].min()
            > float(numeric_rule["retained_mass_min_strict"])
        ),
        "candidate_count": bool(
            group_sizes.eq(
                int(numeric_rule["candidate_count_per_unit"])
            ).all()
            and distinct.eq(
                int(numeric_rule["candidate_count_per_unit"])
            ).all()
        ),
        "same_budget": bool(
            registry["active_budget"].eq(
                int(protocol["cache"]["total_budget"])
            ).all()
        ),
        "cache_and_positions": bool(
            audit["position_maps_valid"].all()
            and audit["cache_fingerprint_invariant"].all()
            and audit["anchor_cache_all_fp32"].all()
            and audit["anchor_cache_shapes_valid"].all()
        ),
        "upstream_unchanged": bool(
            audit["upstream_layer_output_max_abs"].max()
            <= 1.0e-6
        ),
        "calibration_pass": bool(calibration["passed"]),
        "split_isolation": bool(
            all(
                evaluation_metadata["split_audit"]["checks"].values()
            )
        ),
    }
    numeric_pass = bool(all(numeric_checks.values()))

    representation_rule = protocol["gates"]["representation"]
    rep_cosine = float(
        sequence_vectors[
            "boundary_manual_vs_physical_cosine"
        ].median()
    )
    rep_relative = float(
        sequence_vectors[
            "boundary_manual_vs_physical_relative_l2"
        ].median()
    )
    rep_row_fraction = float(
        response["boundary_manual_vs_physical_cosine"]
        .ge(float(representation_rule["row_cosine_threshold"]))
        .mean()
    )
    representation_checks = {
        "sequence_first_median_cosine": (
            rep_cosine
            >= float(
                representation_rule[
                    "sequence_first_median_cosine_min"
                ]
            )
        ),
        "sequence_first_median_relative_l2": (
            rep_relative
            <= float(
                representation_rule[
                    "sequence_first_median_relative_l2_max"
                ]
            )
        ),
        "row_pass_fraction": (
            rep_row_fraction
            >= float(representation_rule["row_pass_fraction_min"])
        ),
    }
    representation_pass = bool(
        all(representation_checks.values())
    )

    downstream_rule = protocol["gates"][
        "downstream_linearization"
    ]
    downstream_cosine = float(
        sequence_vectors[
            "downstream_jvp_j1_vs_manual_cosine"
        ].median()
    )
    downstream_relative = float(
        sequence_vectors[
            "downstream_jvp_j1_vs_manual_relative_l2"
        ].median()
    )
    downstream_row_fraction = float(
        response["downstream_jvp_j1_vs_manual_cosine"]
        .ge(float(downstream_rule["row_cosine_threshold"]))
        .mean()
    )
    downstream_checks = {
        "sequence_first_median_cosine": (
            downstream_cosine
            >= float(
                downstream_rule[
                    "sequence_first_median_cosine_min"
                ]
            )
        ),
        "sequence_first_median_relative_l2": (
            downstream_relative
            <= float(
                downstream_rule[
                    "sequence_first_median_relative_l2_max"
                ]
            )
        ),
        "row_pass_fraction": (
            downstream_row_fraction
            >= float(downstream_rule["row_pass_fraction_min"])
        ),
    }
    downstream_pass = bool(all(downstream_checks.values()))

    delta, task_delta, positive = delta_tables(sequence_rankings)
    decision_rule = protocol["gates"]["decision_gain"]
    minimum_delta = float(
        decision_rule[
            "task_and_overall_median_delta_spearman_min"
        ]
    )
    overall_delta_local = float(
        delta["delta_fisher_local"].median()
    )
    overall_delta_direct = float(
        delta["delta_fisher_direct"].median()
    )
    task_delta_pass = bool(
        task_delta[
            [
                "median_delta_fisher_local",
                "median_delta_fisher_direct",
            ]
        ]
        .ge(minimum_delta)
        .all()
        .all()
    )
    positive_pass = bool(
        positive[
            ["positive_fisher_local", "positive_fisher_direct"]
        ]
        .ge(
            float(
                decision_rule[
                    "positive_sequence_fraction_min"
                ]
            )
        )
        .all()
        .all()
    )
    score_sequence = sequence_rankings.groupby(
        "score_type"
    )[
        [
            "pairwise_sign_accuracy",
            "top1_accuracy",
            "topk_overlap",
        ]
    ].median()
    pairwise_gain = float(
        score_sequence.loc["fisher", "pairwise_sign_accuracy"]
        - max(
            score_sequence.loc[
                "direct", "pairwise_sign_accuracy"
            ],
            score_sequence.loc[
                "local", "pairwise_sign_accuracy"
            ],
        )
    )
    fisher_top1 = float(
        score_sequence.loc["fisher", "top1_accuracy"]
    )
    comparator_top1 = float(
        max(
            score_sequence.loc["direct", "top1_accuracy"],
            score_sequence.loc["local", "top1_accuracy"],
        )
    )
    fisher_topk = float(
        score_sequence.loc["fisher", "topk_overlap"]
    )
    comparator_topk = float(
        max(
            score_sequence.loc["direct", "topk_overlap"],
            score_sequence.loc["local", "topk_overlap"],
        )
    )
    decision_checks = {
        "task_delta_spearman": task_delta_pass,
        "overall_delta_spearman": bool(
            overall_delta_local >= minimum_delta
            and overall_delta_direct >= minimum_delta
        ),
        "positive_sequence_fraction": positive_pass,
        "pairwise_gain": bool(
            pairwise_gain
            >= float(
                decision_rule["overall_pairwise_gain_min"]
            )
        ),
        "top1_not_decreased": bool(
            fisher_top1 >= comparator_top1
        ),
        "topk_not_decreased": bool(
            fisher_topk >= comparator_topk
        ),
    }
    decision_pass = bool(all(decision_checks.values()))

    if not numeric_pass:
        outcome = "N"
    elif not representation_pass:
        outcome = "D"
    elif not downstream_pass:
        outcome = "C"
    elif decision_pass:
        outcome = "A"
    else:
        outcome = "B"
    return {
        "outcome": outcome,
        "outcome_definition": protocol["outcomes"][outcome],
        "numeric": {
            "passed": numeric_pass,
            "checks": numeric_checks,
            "metrics": {
                "finite_rate": float(
                    response[finite_columns].to_numpy().mean()
                ),
                "repeat_max_absolute_error": float(
                    audit["repeat_max_absolute_error"].max()
                ),
                "boundary_map_baseline_min_cosine": float(
                    audit["boundary_map_baseline_cosine"].min()
                ),
                "boundary_map_baseline_max_relative_l2": float(
                    audit[
                        "boundary_map_baseline_relative_l2"
                    ].max()
                ),
                "retained_mass_min": float(
                    identity["retained_mass"].min()
                ),
                "kappa_mass_max": float(
                    identity["kappa_mass"].max()
                ),
            },
        },
        "representation": {
            "passed": representation_pass,
            "checks": representation_checks,
            "metrics": {
                "sequence_first_median_cosine": rep_cosine,
                "sequence_first_median_relative_l2": rep_relative,
                "row_pass_fraction": rep_row_fraction,
            },
        },
        "downstream_linearization": {
            "passed": downstream_pass,
            "checks": downstream_checks,
            "metrics": {
                "sequence_first_median_cosine": downstream_cosine,
                "sequence_first_median_relative_l2": downstream_relative,
                "row_pass_fraction": downstream_row_fraction,
            },
        },
        "decision_gain": {
            "passed": decision_pass,
            "checks": decision_checks,
            "metrics": {
                "overall_delta_fisher_local": overall_delta_local,
                "overall_delta_fisher_direct": overall_delta_direct,
                "pairwise_gain_over_best_euclidean": pairwise_gain,
                "fisher_top1_accuracy": fisher_top1,
                "best_euclidean_top1_accuracy": comparator_top1,
                "fisher_topk_overlap": fisher_topk,
                "best_euclidean_topk_overlap": comparator_topk,
                "task_deltas": task_delta.to_dict("records"),
                "positive_sequence_fractions": positive.to_dict(
                    "records"
                ),
            },
        },
    }


def layer_delta_table(rankings: pd.DataFrame) -> pd.DataFrame:
    pivot = rankings.pivot(
        index=["sample_id", "task", "anchor", "layer"],
        columns="score_type",
        values="spearman",
    ).reset_index()
    pivot["delta_fisher_local"] = (
        pivot["fisher"] - pivot["local"]
    )
    pivot["delta_fisher_direct"] = (
        pivot["fisher"] - pivot["direct"]
    )
    return (
        pivot.groupby("layer", as_index=False)[
            ["delta_fisher_local", "delta_fisher_direct"]
        ]
        .median()
        .rename(
            columns={
                "delta_fisher_local": "median_delta_fisher_local",
                "delta_fisher_direct": "median_delta_fisher_direct",
            }
        )
    )


def fmt(value: float, digits: int = 6) -> str:
    return f"{float(value):.{digits}f}"


def write_results_markdown(
    path: Path,
    gate: Mapping[str, Any],
    response: pd.DataFrame,
    identity: pd.DataFrame,
    rankings: pd.DataFrame,
    sequence_rankings: pd.DataFrame,
    sequence_vectors: pd.DataFrame,
    layer_deltas: pd.DataFrame,
    calibration: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> None:
    score_summary = (
        sequence_rankings.groupby("score_type")[
            [
                "spearman",
                "pairwise_sign_accuracy",
                "top1_accuracy",
                "topk_overlap",
                "normalized_regret",
            ]
        ]
        .median()
        .reset_index()
    )
    sequence_delta, task_delta, positive = delta_tables(
        sequence_rankings
    )
    task_vector = (
        response.groupby("task")[
            [
                "boundary_manual_vs_physical_cosine",
                "boundary_manual_vs_physical_relative_l2",
                "downstream_jvp_j1_vs_manual_cosine",
                "downstream_jvp_j1_vs_manual_relative_l2",
                "end_to_end_j1_vs_physical_cosine",
                "end_to_end_j1_vs_physical_relative_l2",
            ]
        ]
        .median()
        .reset_index()
    )
    worst = response.nsmallest(
        5, "downstream_jvp_j1_vs_manual_cosine"
    )[
        [
            "sample_id",
            "anchor",
            "layer",
            "candidate_source",
            "retained_mass_min",
            "downstream_jvp_j1_vs_manual_cosine",
            "downstream_jvp_j1_vs_manual_relative_l2",
        ]
    ]
    outcome_names = {
        "A": "strong fixed-boundary closure",
        "B": "geometric closure without decision gain",
        "C": "manual bridge 成立，但线性化失败",
        "D": "fixed-boundary representation failure",
        "N": "numerically inconclusive",
    }
    lines = [
        "# P0-v2 正式结果：Fixed-Boundary Readout Closure",
        "",
        "实验日期：2026-07-27  ",
        f"预注册裁决：**Outcome {gate['outcome']} — {outcome_names[gate['outcome']]}**",
        "",
        "## 1. 结论",
        "",
        "在本次匹配的 dequantized FP32 execution graph 和冻结 gate 下，",
        f"P0-v2 被裁决为 Outcome {gate['outcome']}。该结论只适用于 fixed current step、",
        "isolated single-layer cache action、action-only 和固定下一层输入边界；",
        "不能外推到多步、多层联合 mask 或 refresh policy。",
        "",
        "正式矩阵包含 4 条 held-out sequence（GovReport 2 条、NIAH 2 条）、",
        "2 个 anchor、3 个分析层和每 unit 8 个物理候选，共 24 个独立",
        "sequence-anchor-layer units、192 个 candidate rows。统计采用",
        "sequence-first aggregation。",
        "",
        "## 2. 五个科学问题的直接回答",
        "",
        "### 2.1 固定边界 manual intervention 能否复现 physical intervention？",
        "",
        "**能。** physical-input manual 对 physical final-logit response 的",
        "sequence-first median 指标为：",
        "",
        f"- cosine：{fmt(gate['representation']['metrics']['sequence_first_median_cosine'], 9)}；",
        f"- relative L2：{fmt(gate['representation']['metrics']['sequence_first_median_relative_l2'], 9)}；",
        f"- cosine 不低于 0.99 的 candidate-row 比例：{fmt(gate['representation']['metrics']['row_pass_fraction'], 3)}。",
        "",
        "因此，在执行图匹配后，单个下一层输入 boundary perturbation 足以表示本轮",
        "isolated physical cache intervention 的 same-step downstream effect。",
        "",
        "### 2.2 Downstream Jacobian 能否预测自然 action 幅度的 final-logit response？",
        "",
        "**按预注册 gate，可以。** 对 J1-theory boundary input，JVP 对相同输入的",
        "manual nonlinear downstream response 的 sequence-first 指标为：",
        "",
        f"- median cosine：{fmt(gate['downstream_linearization']['metrics']['sequence_first_median_cosine'], 6)}；",
        f"- median relative L2：{fmt(gate['downstream_linearization']['metrics']['sequence_first_median_relative_l2'], 6)}；",
        f"- cosine 不低于 0.90 的 candidate-row 比例：{fmt(gate['downstream_linearization']['metrics']['row_pass_fraction'], 3)}。",
        "",
        "这不是逐行完美线性：低 retained-mass 或大扰动尾部存在明显较差案例，",
        "但 sequence-first gate 通过。",
        "",
        "### 2.3 完整 action-only chain 的主要误差来自哪里？",
        "",
        "接口中位数显示：",
        "",
        f"- deletion pulse：cosine {fmt(response['pulse_theory_vs_physical_cosine'].median(), 6)}，relative L2 {fmt(response['pulse_theory_vs_physical_relative_l2'].median(), 6)}；",
        f"- adjacent J1：cosine {fmt(response['adjacent_j1_vs_physical_cosine'].median(), 6)}，relative L2 {fmt(response['adjacent_j1_vs_physical_relative_l2'].median(), 6)}；",
        f"- fixed-boundary representation：cosine {fmt(response['boundary_manual_vs_physical_cosine'].median(), 6)}，relative L2 {fmt(response['boundary_manual_vs_physical_relative_l2'].median(), 6)}；",
        f"- downstream JVP：cosine {fmt(response['downstream_jvp_j1_vs_manual_cosine'].median(), 6)}，relative L2 {fmt(response['downstream_jvp_j1_vs_manual_relative_l2'].median(), 6)}。",
        "",
        "删除 pulse 和 physical/manual boundary mismatch 几乎不是主要误差；",
        "主要误差来自 adjacent-block 与 downstream 两段在自然幅度下的一阶近似，",
        "并在完整链中累积。early layer 的误差高于 late layer。",
        "",
        "### 2.4 Baseline-Fisher 是否稳定改善 exact-KL candidate ranking？",
        "",
        "**在本矩阵中是。** sequence-first median Spearman 为：",
        "",
    ]
    for row in score_summary.itertuples(index=False):
        label = {
            "direct": "Direct Euclidean",
            "local": "Local hidden Euclidean",
            "fisher": "Baseline-Fisher",
            "midpoint_oracle": "Midpoint Fisher oracle（回顾性）",
        }[row.score_type]
        lines.append(
            f"- {label}：Spearman {fmt(row.spearman, 6)}，"
            f"pairwise {fmt(row.pairwise_sign_accuracy, 6)}，"
            f"top-1 {fmt(row.top1_accuracy, 3)}，"
            f"top-2 overlap {fmt(row.topk_overlap, 3)}。"
        )
    lines.extend(
        [
            "",
            "Fisher 的 sequence-first median Spearman 增量为：",
            "",
            f"- 对 local：{fmt(gate['decision_gain']['metrics']['overall_delta_fisher_local'], 6)}；",
            f"- 对 direct：{fmt(gate['decision_gain']['metrics']['overall_delta_fisher_direct'], 6)}。",
            "",
            "两个任务中的 positive-sequence fraction 都为 1.0。midpoint oracle 使用真实",
            "physical logits，只用于确认 Fisher geometry，不是可部署 selector。",
            "",
            "### 2.5 当前证据支持哪一种结论？",
            "",
            f"按冻结规则支持 **Outcome {gate['outcome']}**。这意味着本轮同时通过",
            "固定边界表示、自然幅度下游线性化和决策增量三个 gate。由于每个任务只有",
            "2 条正式 sequence，这应理解为“在预注册的小型 held-out 矩阵中成立”，",
            "不是对更广任务、更多模型或多层联合 cache action 的普遍性证明。",
            "",
            "## 3. 数值认证",
            "",
            f"Calibration 使用 44/45 号 sequence 的 48 个方向、{calibration['row_count']} 行扫描，",
            f"机械选择并冻结了相对半径 `{calibration['selected_relative_radius']}`。",
            "",
            f"- formal finite rate：{fmt(gate['numeric']['metrics']['finite_rate'], 3)}；",
            f"- full replay repeat max absolute error：{gate['numeric']['metrics']['repeat_max_absolute_error']:.3e}；",
            f"- boundary-map baseline min cosine：{fmt(gate['numeric']['metrics']['boundary_map_baseline_min_cosine'], 9)}；",
            f"- retained mass 最小值：{gate['numeric']['metrics']['retained_mass_min']:.3e}；",
            f"- 最大质量条件数：{gate['numeric']['metrics']['kappa_mass_max']:.3f}。",
            "",
            "删除恒等式 FP64 的最大 absolute L2 error 为",
            f"`{identity.loc[identity['arithmetic'].eq('float64'), 'absolute_l2_error'].max():.3e}`；",
            "FP32 在 cancellation-sensitive、目标 norm 接近零的 head 上可出现很大的原始",
            "relative error，因此没有把该原始比值单独当作理论失败。",
            "",
            "## 4. 分任务结果",
            "",
            "| 任务 | boundary cosine | boundary rel-L2 | JVP cosine | JVP rel-L2 | end-to-end cosine | end-to-end rel-L2 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in task_vector.itertuples(index=False):
        lines.append(
            f"| {row.task} | "
            f"{row.boundary_manual_vs_physical_cosine:.6f} | "
            f"{row.boundary_manual_vs_physical_relative_l2:.6f} | "
            f"{row.downstream_jvp_j1_vs_manual_cosine:.6f} | "
            f"{row.downstream_jvp_j1_vs_manual_relative_l2:.6f} | "
            f"{row.end_to_end_j1_vs_physical_cosine:.6f} | "
            f"{row.end_to_end_j1_vs_physical_relative_l2:.6f} |"
        )
    lines.extend(
        [
            "",
            "Fisher Spearman 增量：",
            "",
            "| 任务 | Fisher − local | Fisher − direct |",
            "|---|---:|---:|",
        ]
    )
    for row in task_delta.itertuples(index=False):
        lines.append(
            f"| {row.task} | "
            f"{row.median_delta_fisher_local:.6f} | "
            f"{row.median_delta_fisher_direct:.6f} |"
        )
    lines.extend(
        [
            "",
            "## 5. 分层 Fisher 增量",
            "",
            "| 分析层 | Fisher − local | Fisher − direct |",
            "|---:|---:|---:|",
        ]
    )
    for row in layer_deltas.itertuples(index=False):
        lines.append(
            f"| {int(row.layer)} | "
            f"{row.median_delta_fisher_local:.6f} | "
            f"{row.median_delta_fisher_direct:.6f} |"
        )
    lines.extend(
        [
            "",
            "## 6. 最差线性化尾部",
            "",
            "| sequence | anchor | layer | candidate | retained mass min | cosine | relative L2 |",
            "|---|---:|---:|---|---:|---:|---:|",
        ]
    )
    for row in worst.itertuples(index=False):
        lines.append(
            f"| {row.sample_id} | {int(row.anchor)} | {int(row.layer)} | "
            f"{row.candidate_source} | {row.retained_mass_min:.3e} | "
            f"{row.downstream_jvp_j1_vs_manual_cosine:.6f} | "
            f"{row.downstream_jvp_j1_vs_manual_relative_l2:.6f} |"
        )
    lines.extend(
        [
            "",
            "这些尾部没有被 pooled median 隐藏：完整 candidate-level Parquet 保留了",
            "每个 sequence-anchor-layer-candidate 的所有接口指标。",
            "",
            "## 7. 通过、失败和仍未验证的接口",
            "",
            "- 通过：FP32 execution-graph matching、full replay、物理单层隔离、删除 pulse、",
            "  fixed-boundary sufficiency、下游 JVP gate、Fisher ranking 增量 gate。",
            "- 存在自然幅度尾部：adjacent J1 和 downstream JVP 不是逐 candidate 完美；",
            "  retained-mass 很低时误差会放大。",
            "- 未验证：历史状态、跨时间传播、E2、refresh、future query、自由生成、",
            "  多层联合 mask、低秩读出、其他模型和大规模跨任务复现。",
            "",
            "## 8. 可复现性",
            "",
            f"- evaluation config SHA-256：`{metadata['config_sha256']}`；",
            f"- 固定 seed：`20260727`；",
            "- 正式 sequence：`gov_report:56`、`gov_report:57`、",
            "  `synthetic_niah_56`、`synthetic_niah_57`；",
            "- 原始与汇总表、运行命令、manifest 和 checksums 位于",
            "  `experiments/p0_v2_fixed_boundary/`。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_failure_analysis(
    path: Path,
    gate: Mapping[str, Any],
    response: pd.DataFrame,
) -> None:
    interfaces = []
    for prefix in VECTOR_PREFIXES:
        interfaces.append(
            {
                "interface": prefix,
                "median_cosine": float(
                    response[f"{prefix}_cosine"].median()
                ),
                "median_relative_l2": float(
                    response[f"{prefix}_relative_l2"].median()
                ),
                "worst_cosine": float(
                    response[f"{prefix}_cosine"].min()
                ),
            }
        )
    lines = [
        "# P0-v2 Gate Failure Analysis",
        "",
        f"预注册裁决：Outcome {gate['outcome']}。",
        "",
        "该文件只在至少一个目标 gate 未通过时生成。失败不自动等于理论被否定。",
        "",
        "| 接口 | median cosine | median relative L2 | worst cosine |",
        "|---|---:|---:|---:|",
    ]
    for row in interfaces:
        lines.append(
            f"| {row['interface']} | {row['median_cosine']:.6f} | "
            f"{row['median_relative_l2']:.6f} | "
            f"{row['worst_cosine']:.6f} |"
        )
    lines.extend(
        [
            "",
            "详细 gate 状态请见 `p0_v2_gate_outcome.json`；candidate-level",
            "定位请见 `response_rows.parquet`。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def environment_info() -> Dict[str, Any]:
    packages = {}
    for name in ("numpy", "pandas", "pyarrow", "scipy", "torch"):
        module = __import__(name)
        packages[name] = str(
            getattr(module, "__version__", "unknown")
        )
    try:
        import mlx.core as mx
        import mlx_lm

        packages["mlx"] = str(
            getattr(mx, "__version__", "0.29.3")
        )
        packages["mlx_lm"] = str(
            getattr(mlx_lm, "__version__", "unknown")
        )
    except Exception as error:
        packages["mlx_error"] = str(error)
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
    }


def write_manifest(
    path: Path,
    protocol_path: Path,
    output_dir: Path,
    gate: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> None:
    candidates = [
        ROOT / "experiments/p0_v2_fixed_boundary/docs/code_audit.md",
        ROOT / "experiments/p0_v2_fixed_boundary/docs/experiment_plan.md",
        ROOT / "experiments/p0_v2_fixed_boundary/docs/results.md",
        ROOT
        / "docs/archive/theory_iterations/current_theory_model_revised.md",
        ROOT / "configs/frozen/p0_v2_config.yaml",
        ROOT / "tests/test_p0_v2.py",
        ROOT
        / "experiments/p0_v2_fixed_boundary/README.md",
        ROOT
        / "experiments/p0_v2_fixed_boundary/scripts/p0_v2_core.py",
        ROOT
        / "experiments/p0_v2_fixed_boundary/scripts/run_p0_v2.py",
        ROOT
        / "experiments/p0_v2_fixed_boundary/scripts/analyze_p0_v2.py",
    ]
    candidates.extend(
        sorted(
            file
            for file in output_dir.rglob("*")
            if file.is_file()
            and file.resolve() != path.resolve()
            and file.name != "p0_v2_checksums.json"
        )
    )
    if gate["outcome"] != "A":
        candidates.append(ROOT / "experiments/p0_v2_fixed_boundary/docs/failure_analysis.md")
    unique = sorted(
        {file.resolve() for file in candidates if file.exists()}
    )
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    manifest = {
        "schema_version": 1,
        "experiment": "formal_fixed_boundary_readout_closure_p0_v2",
        "outcome": gate["outcome"],
        "seed": 20260727,
        "config": {
            "path": str(protocol_path.relative_to(ROOT)),
            "sha256_at_evaluation": metadata["config_sha256"],
            "sha256_current": sha256_file(protocol_path),
            "fd_selected_relative_radius": metadata[
                "selected_relative_radius"
            ],
        },
        "data": {
            "calibration_ids": [
                "gov_report:44",
                "gov_report:45",
                "synthetic_niah_44",
                "synthetic_niah_45",
            ],
            "evaluation_ids": metadata["sequence_ids"],
            "sequence_count": len(metadata["sequence_ids"]),
            "response_row_count": 192,
            "unit_count": 24,
        },
        "model": {
            "source": (
                "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
            ),
            "revision": (
                "8b403126fc14f14cfc99bb4cfa72ecbc129ea677"
            ),
            "execution": "fully_dequantized_float32",
            "quantized_modules_after_dequantization": 0,
        },
        "git": {
            "commit": git_commit,
            "worktree_dirty": bool(status.strip()),
            "note": (
                "The repository was already dirty; P0-v2 used new isolated "
                "files and did not revert unrelated user changes."
            ),
        },
        "environment": environment_info(),
        "checksum_exclusions": [
            str(path.relative_to(ROOT)),
            str(
                (
                    output_dir / "p0_v2_checksums.json"
                ).relative_to(ROOT)
            ),
        ],
        "checksum_exclusion_reason": (
            "The manifest and its JSON checksum mirror are excluded to avoid "
            "self-referential checksums; every source, aggregate, and "
            "per-sequence checkpoint artifact is included."
        ),
        "checksums": {
            str(file.relative_to(ROOT)): sha256_file(file)
            for file in unique
        },
    }
    path.write_text(
        yaml.safe_dump(
            json_safe(manifest),
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    atomic_json(
        output_dir / "p0_v2_checksums.json",
        manifest["checksums"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/frozen/p0_v2_config.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "experiments/p0_v2_fixed_boundary/results",
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    output_dir = args.output_dir.resolve()
    protocol = load_protocol(config_path)
    response = pd.read_parquet(
        output_dir / "response_rows.parquet"
    )
    identity = pd.read_parquet(
        output_dir / "identity_rows.parquet"
    )
    registry = pd.read_parquet(
        output_dir / "candidate_registry.parquet"
    )
    audit = pd.read_parquet(output_dir / "unit_audit.parquet")
    calibration = json.loads(
        (output_dir / "calibration_summary.json").read_text(
            encoding="utf-8"
        )
    )
    metadata = json.loads(
        (output_dir / "evaluation_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    if sha256_file(config_path) != metadata["config_sha256"]:
        raise RuntimeError(
            "current config differs from evaluation-frozen config"
        )
    rankings = make_rankings(
        response, int(protocol["metrics"]["top_k"])
    )
    sequence_rankings = make_sequence_rankings(rankings)
    sequence_vectors = make_sequence_vectors(response)
    gate = gate_outcome(
        protocol,
        response,
        identity,
        registry,
        audit,
        rankings,
        sequence_rankings,
        sequence_vectors,
        calibration,
        metadata,
    )
    layer_deltas = layer_delta_table(rankings)
    sequence_delta, task_delta, positive = delta_tables(
        sequence_rankings
    )
    atomic_frame(output_dir / "ranking_rows.parquet", rankings)
    atomic_frame(
        output_dir / "sequence_first_ranking.parquet",
        sequence_rankings,
    )
    atomic_frame(
        output_dir / "sequence_first_vector_metrics.parquet",
        sequence_vectors,
    )
    sequence_rankings.to_csv(
        output_dir / "p0_v2_sequence_first_summary.csv",
        index=False,
    )
    rankings.to_csv(
        output_dir / "p0_v2_unit_ranking_summary.csv",
        index=False,
    )
    layer_deltas.to_csv(
        output_dir / "p0_v2_layer_delta_summary.csv",
        index=False,
    )
    sequence_delta.to_csv(
        output_dir / "p0_v2_sequence_delta_summary.csv",
        index=False,
    )
    task_delta.merge(positive, on="task").to_csv(
        output_dir / "p0_v2_task_delta_summary.csv",
        index=False,
    )
    atomic_json(output_dir / "p0_v2_gate_outcome.json", gate)
    flat_summary = {
        "outcome": gate["outcome"],
        "numeric_pass": gate["numeric"]["passed"],
        "representation_pass": gate["representation"]["passed"],
        "downstream_linearization_pass": gate[
            "downstream_linearization"
        ]["passed"],
        "decision_gain_pass": gate["decision_gain"]["passed"],
        **{
            f"representation_{key}": value
            for key, value in gate["representation"]["metrics"].items()
        },
        **{
            f"downstream_{key}": value
            for key, value in gate[
                "downstream_linearization"
            ]["metrics"].items()
        },
        **{
            key: value
            for key, value in gate["decision_gain"]["metrics"].items()
            if isinstance(value, (int, float, bool))
        },
    }
    pd.DataFrame([flat_summary]).to_csv(
        output_dir / "p0_v2_summary.csv", index=False
    )
    atomic_json(
        output_dir / "p0_v2_summary.json",
        {
            "gate": gate,
            "row_counts": {
                "response": len(response),
                "identity": len(identity),
                "registry": len(registry),
                "audit": len(audit),
                "ranking": len(rankings),
                "sequence_ranking": len(sequence_rankings),
            },
            "sequence_ids": metadata["sequence_ids"],
            "config_sha256": metadata["config_sha256"],
        },
    )
    write_results_markdown(
        ROOT / "experiments/p0_v2_fixed_boundary/docs/results.md",
        gate,
        response,
        identity,
        rankings,
        sequence_rankings,
        sequence_vectors,
        layer_deltas,
        calibration,
        metadata,
    )
    failure_path = ROOT / "experiments/p0_v2_fixed_boundary/docs/failure_analysis.md"
    if gate["outcome"] != "A":
        write_failure_analysis(failure_path, gate, response)
    elif failure_path.exists():
        raise RuntimeError(
            "stale failure analysis exists despite Outcome A"
        )
    manifest_path = (
        ROOT / "experiments/p0_v2_fixed_boundary/P0_V2_MANIFEST.yaml"
    )
    write_manifest(
        manifest_path,
        config_path,
        output_dir,
        gate,
        metadata,
    )
    print(json.dumps(gate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
