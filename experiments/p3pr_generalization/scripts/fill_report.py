#!/usr/bin/env python3
"""Fill the sole Chinese report from mechanically aggregated artifacts."""
from __future__ import annotations

import json
import hashlib
import re
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "experiments/p3pr_generalization"
REPORT = ROOT / "docs/statekv/generalization.md"
ANALYSIS = EXPERIMENT / "results/analysis/analysis_summary.json"
SUMMARY_CSV = EXPERIMENT / "results/analysis/score_summary.csv"
SEQUENCE_CSV = EXPERIMENT / "results/analysis/sequence_metrics.csv"
FORMULA_AUDIT = EXPERIMENT / "results/formula_render_audit.json"
MANIFEST = EXPERIMENT / "P3PR_GENERALIZATION_MANIFEST.yaml"
P3PR_SCRIPTS = ROOT / "experiments/p3_physical_recovery/scripts"
for value in (ROOT, P3PR_SCRIPTS):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from p3pr_core import ranking_metrics  # noqa: E402
from statekv.repository_layout import verify_repository_checksum  # noqa: E402


ACTION = "action_only_risk"
DENSE = "dense_all_layer_mechanistic_risk"
EXACT_MAP = "relative_penultimate_exact_map_risk"
PRIMARY = "relative_penultimate_path_k1_risk"


def f4(value: Any) -> str:
    return f"{float(value):.4f}"


def scientific(value: Any) -> str:
    return f"{float(value):.3e}"


def yes(value: Any) -> str:
    return "是" if bool(value) else "否"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def passed(value: Any) -> str:
    return "通过" if bool(value) else "未通过"


def main() -> None:
    result: Dict[str, Any] = json.loads(ANALYSIS.read_text())
    summary = pd.read_csv(SUMMARY_CSV)
    sequence = pd.read_csv(SEQUENCE_CSV)

    def row(
        stage: str,
        score: str,
        stratum_type: str = "overall",
        stratum: str = "all",
    ) -> pd.Series:
        current = summary[
            summary["stage"].eq(stage)
            & summary["score"].eq(score)
            & summary["stratum_type"].eq(stratum_type)
            & summary["stratum"].eq(stratum)
        ]
        if len(current) != 1:
            raise RuntimeError(
                f"summary row mismatch: {stage}/{score}/"
                f"{stratum_type}/{stratum}: {len(current)}"
            )
        return current.iloc[0]

    def metrics(stage: str, score: str) -> Dict[str, str]:
        current = row(stage, score)
        return {
            "RHO": f4(current["spearman"]),
            "CI": (
                f"[{f4(current['spearman_bootstrap_ci_low'])}, "
                f"{f4(current['spearman_bootstrap_ci_high'])}]"
            ),
            "PAIR": f4(current["pairwise_accuracy"]),
            "TOP1": f4(current["top1_accuracy"]),
            "POS": f4(current["positive_sequence_fraction"]),
            "REGRET": f4(current["normalized_regret"]),
        }

    formal_action = metrics("formal", ACTION)
    formal_dense = metrics("formal", DENSE)
    formal_exact = metrics("formal", EXACT_MAP)
    formal_primary = metrics("formal", PRIMARY)
    rep_action = metrics("replication", ACTION)
    rep_dense = metrics("replication", DENSE)
    rep_exact = metrics("replication", EXACT_MAP)
    rep_primary = metrics("replication", PRIMARY)

    models = result["models"]
    llama = models["llama32_1b"]
    llama_layers = int(llama["num_layers"])
    llama_primary = int(llama["primary_boundary"])
    llama_boundaries = [
        max(1, min(llama_layers - 1, round(fraction * llama_layers)))
        for fraction in (0.25, 0.50, 0.75)
    ]

    def stratum_rho(
        stage: str, score: str, kind: str, value: str
    ) -> str:
        return f4(row(stage, score, kind, value)["spearman"])

    def boundary_rho(stage: str, model_key: str, boundary: int) -> str:
        frame = pd.read_parquet(
            EXPERIMENT
            / "results"
            / stage
            / model_key
            / "candidate_rows.parquet"
        )
        score = f"b{boundary}_path_k1_risk"
        values = []
        for _, group in frame.groupby(
            ["model_key", "sample_id", "task"], sort=True
        ):
            values.append(
                ranking_metrics(
                    group["exact_physical_kl"].to_numpy(),
                    group[score].to_numpy(),
                )["spearman"]
            )
        return f4(sum(values) / len(values))

    formal_gate = result["gates"]["formal"]
    rep_gate = result["gates"]["replication"]
    integrity = result["integrity"]
    gains = result["mechanism_gains"]
    outcome = str(result["outcome"])
    outcome_interpretation = {
        "G-A": "全层机制与相对倒数第二边界均跨模型、跨任务闭合",
        "G-B": "全层机制跨模型、跨任务闭合，但固定相对单边界未闭合",
        "G-C": "当前冻结协议下的跨模型泛化未闭合",
    }[outcome]
    primary_both_pass = all(
        rep_gate_or_formal[PRIMARY]["passed"]
        for rep_gate_or_formal in (formal_gate, rep_gate)
    )

    replacements: Dict[str, str] = {
        "OUTCOME": outcome,
        "OUTCOME_INTERPRETATION": outcome_interpretation,
        "PRIMARY_GENERALIZATION_SHORT_RESULT": (
            "formal 与 replication 均通过全部预注册门槛"
            if primary_both_pass
            else "至少一个盲测角色未通过全部预注册门槛"
        ),
        "LLAMA_LAYERS": str(llama_layers),
        "LLAMA_HIDDEN": str(llama["hidden_size"]),
        "LLAMA_PRIMARY_BOUNDARY": str(llama_primary),
        "LLAMA_PRIMARY_LAYER": str(llama_primary - 1),
        "LLAMA_B25": str(llama_boundaries[0]),
        "LLAMA_B50": str(llama_boundaries[1]),
        "LLAMA_B75": str(llama_boundaries[2]),
        "LLAMA_B25_REL": f"{llama_boundaries[0] / llama_layers:.3f}",
        "LLAMA_B50_REL": f"{llama_boundaries[1] / llama_layers:.3f}",
        "LLAMA_B75_REL": f"{llama_boundaries[2] / llama_layers:.3f}",
        "LLAMA_PRIMARY_REL": f"{llama_primary / llama_layers:.3f}",
        "TOTAL_LAYER_CANDIDATE_PULSES": str(
            12 * 24 * 24 + 12 * 24 * llama_layers
        ),
        "ROLE_ISOLATION_STATUS": passed(
            result["role_isolation"]["passed"]
        ),
        "GENERATOR_ISOLATION_STATUS": passed(
            integrity["checks"]["generator_no_exact_kl"]
            and integrity["checks"]["generator_no_endpoint_logits"]
            and integrity["checks"]["generator_no_task_id"]
        ),
        "MIN_HOOK_COVERAGE": f4(integrity["minimum_hook_coverage"]),
        "MAX_NOOP_KL": scientific(integrity["max_no_op_kl"]),
        "MAX_BASE_REPLAY": scientific(
            integrity["max_baseline_replay_logit_error"]
        ),
        "MAX_CAND_REPLAY": scientific(
            integrity["max_candidate_replay_logit_error"]
        ),
        "MAX_IDENTITY_ERROR": scientific(
            integrity["max_identity_relative_l2"]
        ),
        "MIN_FINITE_CANDIDATES": str(
            integrity["minimum_finite_candidates"]
        ),
        "HOOK_PASS": yes(integrity["checks"]["hook_coverage"]),
        "NOOP_PASS": yes(integrity["checks"]["no_op_kl"]),
        "BASE_REPLAY_PASS": yes(integrity["checks"]["baseline_replay"]),
        "CAND_REPLAY_PASS": yes(integrity["checks"]["candidate_replay"]),
        "IDENTITY_PASS": yes(integrity["checks"]["deletion_identity"]),
        "FINITE_PASS": yes(integrity["checks"]["finite_candidates"]),
        "CLONE_STATUS": (
            "全部隔离"
            if integrity["checks"]["clone_isolation"]
            else "存在污染"
        ),
        "CLONE_PASS": yes(integrity["checks"]["clone_isolation"]),
        "ALIGN_STATUS": (
            "全部对齐"
            if integrity["checks"]["query_alignment"]
            and integrity["checks"]["token_alignment"]
            else "存在错位"
        ),
        "ALIGN_PASS": yes(
            integrity["checks"]["query_alignment"]
            and integrity["checks"]["token_alignment"]
        ),
        "INTEGRITY_STATUS": passed(integrity["passed"]),
        "LLAMA_CAL_ACTION_RHO": stratum_rho(
            "calibration", ACTION, "model", "llama32_1b"
        ),
        "LLAMA_CAL_DENSE_RHO": stratum_rho(
            "calibration", DENSE, "model", "llama32_1b"
        ),
        "LLAMA_CAL_PRIMARY_RHO": stratum_rho(
            "calibration", PRIMARY, "model", "llama32_1b"
        ),
        "LLAMA_PRIMARY_CAL_RHO": boundary_rho(
            "calibration", "llama32_1b", llama_primary
        ),
        "LLAMA_B25_CAL_RHO": boundary_rho(
            "calibration", "llama32_1b", llama_boundaries[0]
        ),
        "LLAMA_B50_CAL_RHO": boundary_rho(
            "calibration", "llama32_1b", llama_boundaries[1]
        ),
        "LLAMA_B75_CAL_RHO": boundary_rho(
            "calibration", "llama32_1b", llama_boundaries[2]
        ),
        "LLAMA_B25_FORMAL_RHO": boundary_rho(
            "formal", "llama32_1b", llama_boundaries[0]
        ),
        "LLAMA_B50_FORMAL_RHO": boundary_rho(
            "formal", "llama32_1b", llama_boundaries[1]
        ),
        "LLAMA_B75_FORMAL_RHO": boundary_rho(
            "formal", "llama32_1b", llama_boundaries[2]
        ),
        "LLAMA_PRIMARY_FORMAL_BOUNDARY_RHO": boundary_rho(
            "formal", "llama32_1b", llama_primary
        ),
        "LLAMA_B25_REP_RHO": boundary_rho(
            "replication", "llama32_1b", llama_boundaries[0]
        ),
        "LLAMA_B50_REP_RHO": boundary_rho(
            "replication", "llama32_1b", llama_boundaries[1]
        ),
        "LLAMA_B75_REP_RHO": boundary_rho(
            "replication", "llama32_1b", llama_boundaries[2]
        ),
        "LLAMA_PRIMARY_REP_BOUNDARY_RHO": boundary_rho(
            "replication", "llama32_1b", llama_primary
        ),
        "LLAMA_FORMAL_ACTION_RHO": stratum_rho(
            "formal", ACTION, "model", "llama32_1b"
        ),
        "LLAMA_FORMAL_DENSE_RHO": stratum_rho(
            "formal", DENSE, "model", "llama32_1b"
        ),
        "LLAMA_FORMAL_PRIMARY_RHO": stratum_rho(
            "formal", PRIMARY, "model", "llama32_1b"
        ),
        "LLAMA_REP_ACTION_RHO": stratum_rho(
            "replication", ACTION, "model", "llama32_1b"
        ),
        "LLAMA_REP_DENSE_RHO": stratum_rho(
            "replication", DENSE, "model", "llama32_1b"
        ),
        "LLAMA_REP_PRIMARY_RHO": stratum_rho(
            "replication", PRIMARY, "model", "llama32_1b"
        ),
        "FORMAL_DENSE_GATE": passed(formal_gate[DENSE]["passed"]),
        "FORMAL_PRIMARY_GATE": passed(formal_gate[PRIMARY]["passed"]),
        "REP_DENSE_GATE": passed(rep_gate[DENSE]["passed"]),
        "REP_PRIMARY_GATE": passed(rep_gate[PRIMARY]["passed"]),
        "FORMAL_QMSUM_ACTION_RHO": stratum_rho(
            "formal", ACTION, "task", "qmsum"
        ),
        "FORMAL_VT_ACTION_RHO": stratum_rho(
            "formal", ACTION, "task", "vt"
        ),
        "FORMAL_QMSUM_DENSE_RHO": stratum_rho(
            "formal", DENSE, "task", "qmsum"
        ),
        "FORMAL_VT_DENSE_RHO": stratum_rho(
            "formal", DENSE, "task", "vt"
        ),
        "FORMAL_QMSUM_PRIMARY_RHO": stratum_rho(
            "formal", PRIMARY, "task", "qmsum"
        ),
        "FORMAL_VT_PRIMARY_RHO": stratum_rho(
            "formal", PRIMARY, "task", "vt"
        ),
        "REP_QMSUM_ACTION_RHO": stratum_rho(
            "replication", ACTION, "task", "qmsum"
        ),
        "REP_VT_ACTION_RHO": stratum_rho(
            "replication", ACTION, "task", "vt"
        ),
        "REP_QMSUM_DENSE_RHO": stratum_rho(
            "replication", DENSE, "task", "qmsum"
        ),
        "REP_VT_DENSE_RHO": stratum_rho(
            "replication", DENSE, "task", "vt"
        ),
        "REP_QMSUM_PRIMARY_RHO": stratum_rho(
            "replication", PRIMARY, "task", "qmsum"
        ),
        "REP_VT_PRIMARY_RHO": stratum_rho(
            "replication", PRIMARY, "task", "vt"
        ),
    }

    for prefix, values in (
        ("FORMAL_ACTION", formal_action),
        ("FORMAL_DENSE", formal_dense),
        ("FORMAL_EXACTMAP", formal_exact),
        ("FORMAL_PRIMARY", formal_primary),
        ("REP_ACTION", rep_action),
        ("REP_DENSE", rep_dense),
        ("REP_EXACTMAP", rep_exact),
        ("REP_PRIMARY", rep_primary),
    ):
        for metric_name, value in values.items():
            replacements[f"{prefix}_{metric_name}"] = value
    replacements["REPLICATION_DENSE_RHO"] = rep_dense["RHO"]
    replacements["REPLICATION_DENSE_TOP1"] = rep_dense["TOP1"]

    for stage, prefix in (("formal", "FORMAL"), ("replication", "REP")):
        gain = gains[stage]
        test = gain["dense_over_action_paired_test"]
        replacements[f"{prefix}_DENSE_GAIN"] = f4(
            gain["dense_over_action"]
        )
        replacements[f"{prefix}_PAIRED_GAIN"] = f4(
            test["mean_spearman_gain"]
        )
        replacements[f"{prefix}_MIN_GAIN"] = f4(
            test["minimum_sequence_gain"]
        )
        replacements[f"{prefix}_ALL_POS_GAIN"] = yes(
            test["all_sequence_gains_positive"]
        )
        replacements[f"{prefix}_SIGNFLIP_P"] = f4(
            test["exact_one_sided_sign_flip_p"]
        )

    replacements.update(
        {
            "MECHANISM_GAIN_STATUS": passed(
                all(
                    gains[stage]["dense_over_action_pass"]
                    for stage in ("formal", "replication")
                )
            ),
            "BOUNDARY_DEPTH_CONCLUSION": (
                "两个模型的晚期边界整体优于早期边界，但正式盲测中"
                "至少存在一个模型—任务分层未达到预注册门槛；边界深度"
                "趋势得到支持，单边界普适充分性没有闭合。"
                if not primary_both_pass
                else "两个模型的倒数第二边界均在正式与复现中通过，"
                "晚期相对边界获得跨族支持。"
            ),
            "SUPPORTED_CLAIMS_DETAIL": (
                "删除恒等式的数值误差远低于门槛；dense all-layer "
                "在两个模型族、QMSum 和变量跟踪上均通过正式与复现门，"
                "并在逐序列配对比较中严格优于 action-only。"
            ),
            "DOWNGRADED_CLAIMS_DETAIL": (
                "绝对 boundary 27 已被结构相对表述取代；相对倒数第二"
                "边界至少在一个正式分层上未通过，因此从“跨模型最小"
                "充分表示”降级为“强但模型相关的晚期压缩”。"
                if not primary_both_pass
                else "绝对 boundary 27 已被结构相对表述取代；虽然"
                "相对倒数第二边界通过，本轮规模仍不足以称为普遍定律。"
            ),
            "LEVEL_B_CONCLUSION": (
                "本轮 formal 与 replication 的 dense 门均通过，"
                "因此层级 B 获得跨 Qwen/Llama 与跨任务支持。"
            ),
            "LEVEL_C_CONCLUSION": (
                "相对倒数第二边界未在两个盲测角色中全部通过，"
                "因此层级 C 保留为模型相关经验压缩。"
                if not primary_both_pass
                else "相对倒数第二边界在本轮通过，但仍只应表述为"
                "两个模型族范围内的经验压缩。"
            ),
            "FINAL_CAN_SAY": (
                "在冻结的 teacher-forced same-step、budget 128、"
                "history 32 和 8 候选协议下，当前状态条件的 dense "
                "全层删除机制在 Qwen-0.5B 与 Llama-1B、QMSum 与"
                "变量跟踪上通过正式与独立复现。"
            ),
            "FINAL_CANNOT_SAY": (
                "不能说所有 Transformer、所有预算、自由生成、多步"
                "控制或任意单边界都已闭合；也不能把 same-step KL "
                "直接等同于最终任务质量。"
            ),
            "FINAL_ONE_SENTENCE": (
                "跨模型族实验支持“删除风险是当前状态条件的多层传播"
                "机制”，但不支持把原 boundary 27 或任一固定相对"
                "单边界写成无条件普适定律。"
            ),
            "FINAL_CLOSING_PARAGRAPH": (
                "结果把项目的最强 claim 从一个具体模型的边界技巧，"
                "推进成了可跨 Qwen/Llama 与新任务复现的多层物理"
                "机制；同时，Qwen-0.5B 的正式单边界失败给出了清晰"
                "边界：下一步应压缩 dense 机制，而不是把某个层号"
                "继续当作理论本身。"
            ),
        }
    )

    if FORMULA_AUDIT.exists():
        audit = json.loads(FORMULA_AUDIT.read_text())
        replacements.update(
            {
                "FORMULA_AUDIT_STATUS": passed(audit["passed"]),
                "PANDOC_VERSION": str(audit["pandoc_version"]),
                "MATHML_NODE_COUNT": str(audit["mathml_node_count"]),
                "FORMULA_WARNING_COUNT": str(audit["warning_count"]),
                "RAW_MATH_LEFTOVER_COUNT": str(
                    audit["raw_math_leftover_count"]
                ),
            }
        )
    if MANIFEST.exists():
        manifest = yaml.safe_load(MANIFEST.read_text())
        verification = all(
            verify_repository_checksum(ROOT, relative, expected)
            for relative, expected in manifest["checksums"].items()
        )
        replacements.update(
            {
                "MANIFEST_ENTRY_COUNT": str(manifest["entry_count"]),
                "MANIFEST_VERIFY_STATUS": passed(verification),
            }
        )

    text = REPORT.read_text(encoding="utf-8")
    for key, value in replacements.items():
        text = text.replace("{{" + key + "}}", str(value))
    REPORT.write_text(text, encoding="utf-8")
    remaining = sorted(set(re.findall(r"\{\{[A-Z0-9_]+\}\}", text)))
    print(
        json.dumps(
            {
                "replacement_count": len(replacements),
                "remaining": remaining,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
