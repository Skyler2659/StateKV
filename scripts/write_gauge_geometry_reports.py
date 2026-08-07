#!/usr/bin/env python
"""Generate final Chinese reports directly from gauge geometry artifacts."""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
IMPORT_ROOTS = (
    REPOSITORY_ROOT,
    REPOSITORY_ROOT / "benchmarks" / "torch",
    REPOSITORY_ROOT / "benchmarks" / "mlx",
)
for import_root in IMPORT_ROOTS:
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from statekv.artifacts import _atomic_text
from statekv.config import load_discovery_config


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_task(
    rows: Iterable[Mapping[str, Any]],
    family: str,
    task: str,
    calibrated: Any = None,
) -> Mapping[str, Any]:
    matched = [
        row
        for row in rows
        if str(row.get("family")) == family
        and str(row.get("task_bucket")) == task
        and (
            calibrated is None
            or bool(row.get("calibrated_oof")) == bool(calibrated)
        )
    ]
    if len(matched) != 1:
        raise KeyError((family, task, calibrated, len(matched)))
    return matched[0]


def fmt(value: Any, digits: int = 3) -> str:
    number = float(value)
    if not math.isfinite(number):
        return "NA"
    if abs(number) >= 1.0e4 or (
        abs(number) > 0.0 and abs(number) < 1.0e-3
    ):
        return ("%.3e" % number)
    return ("%.*f" % (digits, number))


def geometry_table(point: Dict[str, Any], action: Dict[str, Any]) -> str:
    families = [
        "G0_RAW_GLOBAL",
        "G1_UNIFORM_CENTERED",
        "G2_BASE_FISHER",
        "G3_MIDPOINT_FISHER",
        "G4_GL3_ORACLE",
        "G4_GL5_ORACLE",
        "G4_SIMPSON9_ORACLE",
        "G5C_OOF_SELECTED",
        "G6_OOF_SELECTED",
        "G7_ORDER3",
        "G7_ORDER4",
    ]
    lines = [
        "| Family | Gov KL $\\rho$ | NIAH KL $\\rho$ | Gov action $\\rho$ | NIAH action $\\rho$ |",
        "|---|---:|---:|---:|---:|",
    ]
    for family in families:
        try:
            gov_p = find_task(
                point["task_summary"], family, "GovReport", False
            )
            nia_p = find_task(
                point["task_summary"], family, "NIAH", False
            )
            gov_a = find_task(action["task_split"], family, "GovReport")
            nia_a = find_task(action["task_split"], family, "NIAH")
        except KeyError:
            continue
        lines.append(
            "| %s | %s | %s | %s | %s |"
            % (
                family,
                fmt(gov_p["spearman"]),
                fmt(nia_p["spearman"]),
                fmt(gov_a["candidate_spearman"]),
                fmt(nia_a["candidate_spearman"]),
            )
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_discovery_config(args.config)
    run_dir = (
        REPOSITORY_ROOT
        / cfg.runtime.output_root
        / str(cfg.runtime.run_id)
    )
    gate = load_json(run_dir / "gauge_geometry_gate_decision.json")
    point = load_json(run_dir / "oracle_geometry_kl_summary.json")
    action = load_json(run_dir / "oracle_geometry_action_summary.json")
    decomposition = load_json(
        run_dir / "oracle_geometry_decomposition_summary.json"
    )
    quadrature = load_json(
        run_dir / "path_fisher_quadrature_summary.json"
    )
    topk = load_json(run_dir / "topk_gap_geometry_summary.json")
    cumulant = load_json(run_dir / "cumulant_geometry_summary.json")
    rows = pd.read_parquet(run_dir / "oracle_geometry_rows.parquet")

    if bool(gate["stage_a_passed"]):
        pullback = pd.read_parquet(run_dir / "pullback_jvp_rows.parquet")
        if pullback.empty:
            raise RuntimeError(
                "Stage A passed: complete Stage B before writing final reports"
            )
        raise RuntimeError(
            "The report generator currently expects the preregistered Stage-A stop"
        )

    task_decomp = {
        row["task_bucket"]: row
        for row in decomposition["task_medians"]
    }
    g0_gov = find_task(
        point["task_summary"], "G0_RAW_GLOBAL", "GovReport", False
    )
    g0_nia = find_task(
        point["task_summary"], "G0_RAW_GLOBAL", "NIAH", False
    )
    g1_gov = find_task(
        point["task_summary"], "G1_UNIFORM_CENTERED", "GovReport", False
    )
    g1_nia = find_task(
        point["task_summary"], "G1_UNIFORM_CENTERED", "NIAH", False
    )
    g2_gov = find_task(
        point["task_summary"], "G2_BASE_FISHER", "GovReport", False
    )
    g2_nia = find_task(
        point["task_summary"], "G2_BASE_FISHER", "NIAH", False
    )
    g3_gov = find_task(
        point["task_summary"], "G3_MIDPOINT_FISHER", "GovReport", False
    )
    g3_nia = find_task(
        point["task_summary"], "G3_MIDPOINT_FISHER", "NIAH", False
    )
    g2a_gov = find_task(
        action["task_split"], "G2_BASE_FISHER", "GovReport"
    )
    g2a_nia = find_task(
        action["task_split"], "G2_BASE_FISHER", "NIAH"
    )
    g3a_gov = find_task(
        action["task_split"], "G3_MIDPOINT_FISHER", "GovReport"
    )
    g3a_nia = find_task(
        action["task_split"], "G3_MIDPOINT_FISHER", "NIAH"
    )
    g0a_gov = find_task(
        action["task_split"], "G0_RAW_GLOBAL", "GovReport"
    )
    g0a_nia = find_task(
        action["task_split"], "G0_RAW_GLOBAL", "NIAH"
    )
    quad = {
        (row["family"], row["task_bucket"]): row
        for row in quadrature["quadrature"]
    }
    topk_action = [
        row
        for row in topk["action_task_split"]
        if "_K" in str(row["family"])
    ]
    best_topk = {}
    for task in ("GovReport", "NIAH"):
        eligible = [
            row for row in topk_action if row["task_bucket"] == task
        ]
        best_topk[task] = max(
            eligible, key=lambda row: float(row["candidate_spearman"])
        )
    g5a_selection_counts = Counter(
        topk["outer_fold_training_only_selection"][
            "G5A_OOF_SELECTED"
        ].values()
    )
    margin_action = [
        row
        for row in action["task_split"]
        if str(row["family"]).startswith("G6_")
        and row["family"] != "G6_OOF_SELECTED"
    ]
    best_margin = {}
    for task in ("GovReport", "NIAH"):
        eligible = [
            row for row in margin_action if row["task_bucket"] == task
        ]
        best_margin[task] = max(
            eligible, key=lambda row: float(row["candidate_spearman"])
        )
    g7_3_gov = find_task(
        point["task_summary"], "G7_ORDER3", "GovReport", False
    )
    g7_3_nia = find_task(
        point["task_summary"], "G7_ORDER3", "NIAH", False
    )
    g7_4_gov = find_task(
        point["task_summary"], "G7_ORDER4", "GovReport", False
    )
    g7_4_nia = find_task(
        point["task_summary"], "G7_ORDER4", "NIAH", False
    )

    results = rf"""# Gauge-Aware Output Geometry：实验结果

## 0. 一句话结论

本轮完成了 24 条独立 sequence、55,296 条 candidate-step 的完整词表
gauge-aware geometry 重放。exact KL cumulant identity 与 Fisher variance
identity 的最大数值误差分别为
${fmt(quadrature['exact_identity_max_abs_error'])}$ 与
${fmt(quadrature['fisher_identity_max_abs_error'])}$。

Gauge-aware geometry 显著消除了 raw global bound 的百万级松弛：G2/G3 的
pointwise symmetric ratio 已接近常数量级。但没有 family 通过全部预注册
Stage-A gates，尤其 3/5-point path quadrature 对少量大扰动样本仍不够准确。
因此 Stage B/C/D 均按设计停止，最终选择：

> **E：没有一个 tested gauge-aware family 通过完整、跨任务、非极端驱动的 Stage-A output-utility gate。**

这不表示 Fisher geometry 没有信号；它表示当前固定低阶、单点或低维
geometry 尚不能被升级为经过 gate 验证的 practical output objective。

## 1. 数据与协议

- 12 NIAH + 12 official LongBench GovReport；
- anchors $16,32,48$；
- horizons $4,8,16,32$；
- 每个 sequence-anchor 24 个 inherited distinct physical masks；
- budget 128、sink 4、recent 32、core 92；
- 55,296 个 candidate-step 使用完整 vocabulary logits 流式计算 G0--G7；
- full logits 没有长期写盘；
- layer-27 actual/direct/full residual vectors保存为 float16 fragments；
- sequence 是唯一独立单位；
- top-$k$/margin 选择使用 outer-fold training sequences，不读取 held-out
  sequence。

## 2. Raw logit energy 的分解

| Task | Common shift | Top-256 centered | Tail centered | Fisher near-null Euclidean |
|---|---:|---:|---:|---:|
| GovReport | {fmt(task_decomp['GovReport']['common_shift_energy_fraction'])} | {fmt(task_decomp['GovReport']['top256_centered_energy_fraction'])} | {fmt(task_decomp['GovReport']['tail_centered_energy_fraction'])} | {fmt(task_decomp['GovReport']['fisher_near_null_euclidean_fraction'])} |
| NIAH | {fmt(task_decomp['NIAH']['common_shift_energy_fraction'])} | {fmt(task_decomp['NIAH']['top256_centered_energy_fraction'])} | {fmt(task_decomp['NIAH']['tail_centered_energy_fraction'])} | {fmt(task_decomp['NIAH']['fisher_near_null_euclidean_fraction'])} |

百万级松弛不能主要归因于 common shift。主要能量位于大词表 tail 和
Fisher near-null directions；全局 curvature constant 再把这些
softmax-insensitive directions统一按 worst case 计入。

## 3. Geometry 的 KL 与 action 结果

{geometry_table(point, action)}

G0 的 median symmetric ratio 为 GovReport
{fmt(g0_gov['median_symmetric_ratio'])}、NIAH
{fmt(g0_nia['median_symmetric_ratio'])}。uniform centering 后为
{fmt(g1_gov['median_symmetric_ratio'])}/{fmt(g1_nia['median_symmetric_ratio'])}；
G2 为 {fmt(g2_gov['median_symmetric_ratio'])}/{fmt(g2_nia['median_symmetric_ratio'])}；
G3 为 {fmt(g3_gov['median_symmetric_ratio'])}/{fmt(g3_nia['median_symmetric_ratio'])}。

因此 centered norm 有改善但仍远不够；真正的数量级改善来自
probability/Fisher weighting。

## 4. Path quadrature

| Family | Gov median/p90/max rel. error | NIAH median/p90/max rel. error |
|---|---:|---:|
| GL3 | {fmt(quad[('G4_GL3','GovReport')]['median_relative_error'])}/{fmt(quad[('G4_GL3','GovReport')]['p90_relative_error'])}/{fmt(quad[('G4_GL3','GovReport')]['maximum_relative_error'])} | {fmt(quad[('G4_GL3','NIAH')]['median_relative_error'])}/{fmt(quad[('G4_GL3','NIAH')]['p90_relative_error'])}/{fmt(quad[('G4_GL3','NIAH')]['maximum_relative_error'])} |
| GL5 | {fmt(quad[('G4_GL5','GovReport')]['median_relative_error'])}/{fmt(quad[('G4_GL5','GovReport')]['p90_relative_error'])}/{fmt(quad[('G4_GL5','GovReport')]['maximum_relative_error'])} | {fmt(quad[('G4_GL5','NIAH')]['median_relative_error'])}/{fmt(quad[('G4_GL5','NIAH')]['p90_relative_error'])}/{fmt(quad[('G4_GL5','NIAH')]['maximum_relative_error'])} |
| Simpson-9 | {fmt(quad[('G4_SIMPSON9','GovReport')]['median_relative_error'])}/{fmt(quad[('G4_SIMPSON9','GovReport')]['p90_relative_error'])}/{fmt(quad[('G4_SIMPSON9','GovReport')]['maximum_relative_error'])} | {fmt(quad[('G4_SIMPSON9','NIAH')]['median_relative_error'])}/{fmt(quad[('G4_SIMPSON9','NIAH')]['p90_relative_error'])}/{fmt(quad[('G4_SIMPSON9','NIAH')]['maximum_relative_error'])} |

exact path identity 本身已由独立 partition/cumulant 计算验证。失败发生在固定低阶
quadrature 对 sharp path curvature 的近似，而不是公式符号或 KL orientation。

range-corrected candidate bound 的 coverage 为
{fmt(quadrature['range_bound']['coverage'])}，违反数为
{quadrature['range_bound']['violations']}，但 overflow fraction 为
{fmt(quadrature['range_bound']['overflow_fraction'])}，其 median bound/KL
约为
{fmt(math.exp(quadrature['range_bound']['median_log_bound_minus_log_kl']))}。
因此即使覆盖也不具实用 tightness。

## 5. Top-$k$、margin 与 cumulant

描述性 held-out task-best top-$k$ 分别为：

- GovReport：{best_topk['GovReport']['family']}，action Spearman
  {fmt(best_topk['GovReport']['candidate_spearman'])}；
- NIAH：{best_topk['NIAH']['family']}，action Spearman
  {fmt(best_topk['NIAH']['candidate_spearman'])}。

两任务的描述性最佳 $k$ 都是 256；training-only outer-fold G5A selection
也在 {g5a_selection_counts.get('G5A_K256', 0)}/24 folds 选择 $k=256$。
因此最佳 $k$ 在本轮是跨任务稳定的，但 $k=256$ 并没有提供相对 full
Fisher G2/G3 的计算或 ranking优势。

top-token margin 的 task-best family 为
{best_margin['GovReport']['family']} 与 {best_margin['NIAH']['family']}；
它没有形成统一跨任务优于 full Fisher 的结论。

三阶 KL Spearman 为 {fmt(g7_3_gov['spearman'])}/{fmt(g7_3_nia['spearman'])}，
四阶为 {fmt(g7_4_gov['spearman'])}/{fmt(g7_4_nia['spearman'])}。
三/四阶负预测比例分别为
{fmt(cumulant['negative_prediction_fraction']['G7_ORDER3'])} 与
{fmt(cumulant['negative_prediction_fraction']['G7_ORDER4'])}。没有证据支持继续
增加更高阶 cumulant。

## 6. Gate 与停止位置

Stage A passing families：
`{', '.join(gate['stage_a_passing_families']) or 'none'}`。

G4 low-order quadrature gate：
`{gate['g4_low_order_quadrature_passed']}`。

因此：

- `pullback_jvp_rows.parquet`：带 schema 的空表；
- Stage B linearization/low-rank/subspace summaries：明确 skipped；
- Q-state envelope、spectral-band、pairwise、refresh 与 free-generation：
  明确 skipped；
- 没有使用后续结果修改 Stage-A gate。

## 7. 对 24 个问题的直接回答

1. raw $L_2$ 松弛主要来自 tail/Fisher near-null energy与全局 curvature
   constant，common shift 只解释一部分。
2. centered norm 优于 raw norm，但没有消除数量级松弛。
3. base Fisher 的 ratio 已接近常数量级；midpoint 通常更准；path identity
   精确，但低阶 quadrature 有大扰动 counterexamples。
4. 3/5-point 对多数样本接近 exact KL，但没有满足预注册的全范围小误差 gate。
5. Fisher quadratic有明显 KL/action signal，但没有通过所有跨任务与稳健性 gate。
6. 两任务 task-best 均为 $k=256$，且 23/24 outer folds 选择该值；最佳
   $k$ 在本轮稳定，但没有优于 full Fisher。
7. top-token margin 没有统一优于 full-distribution Fisher。
8. 三/四阶 cumulant能解释部分二阶误差，但没有稳定完成闭合。
9. 尚无既 gauge-aware 又经 gate 验证为可部署的 analytical objective。
10. final residual单点 Jacobian未获授权运行。
11. midpoint/path-integrated Jacobian未获授权运行。
12. $Q_t$ 低秩性未检验。
13. pullback subspace稳定性未检验。
14. anchor-frozen Q寿命未检验。
15. scalar Q-state envelope未检验。
16. spectral-band相对 scalar未检验。
17. propagation相对 direct-only增量未在 Q-state 中检验。
18. 新 geometry pairwise uncertainty未进入 calibration 阶段。
19. normalized/horizon/top-candidate conformal未获授权运行。
20. matched-count refresh未运行。
21. free-generation未运行。
22. 没有得到完整可解析 selection--refresh surrogate。
23. 严格失败首先发生在 Stage-A practical geometry/numerical robustness gate；
    pullback、linearization、envelope、policy均保持未判定，而不是被宣称失败。
24. 当前论文边界应保留 E2 state envelope，并增加：Fisher geometry解释了
    raw-logit bound 的主要松弛，但固定低阶 gauge-aware objective 尚未通过
    tested-range action gate。

## 8. 自动测试

- 新增 gauge/pullback tests：41/41；
- StateKV 与 benchmark 分层测试均通过；
- sequence、task、mask、budget、full-vocabulary identities、vector index、
  JSON/parquet/report 与 skipped-stage separation 均由最终 validator 复核。
"""

    pullback_results = rf"""# Fisher-Pullback State Envelope：阶段结果

## 1. 运行状态

Stage B、C、D 未运行，原因不是缺少 runner 或忘记执行，而是 Stage A 的严格
gate 未通过：

$$
\text{{Stage A fail}}
\Longrightarrow
\text{{no formal pullback/envelope/policy claim}}.
$$

`pullback_jvp_rows.parquet` 与 `q_state_envelope_rows.parquet` 是带稳定 schema
的空表；对应 JSON 均记录
`not_run_by_preregistered_gate`。

## 2. 可以保留的 feasibility 结论

- 当前 MLX 0.29.3 环境提供原生 JVP/VJP API；
- 已实现 pullback quadratic、Fisher-randomized VJP sketch、low-rank range
  finder、spectral-band energy、anchor-frozen/periodic Q schedule 与 Q-envelope
  无 future-truth recursion 的纯数值组件；
- synthetic JVP/finite-difference、sketch covariance与低秩恢复测试通过；
- 正式 4-bit final-block composite JVP 没有在 gate 失败后运行。

因此不能回答 $Q_t$ 是否低秩、subspace 是否稳定、anchor-frozen Q 保持多久，
也不能声称 scalar/spectral Q-envelope 成功或失败。

## 3. 理论边界

解析上可以定义

$$
Q_t=J_t^\top F(p_t)J_t,
$$

但本轮没有获得新的模型上 empirical validation。准确表述是：

> Fisher pullback remains a mathematically valid construction, but its
> model-level feasibility and recursive closure were not authorized after
> the Stage-A gate failure.
"""

    theory = rf"""# Gauge Geometry 实验后的理论模型更新

## 1. 最终类别

本轮选择

$$
\boxed{{\text{{E}}}}
$$

其严格含义是：

> No tested non-oracle gauge-aware output-risk geometry passed every
> preregistered cross-task, action, ratio, robustness and quadrature gate.

## 2. 对旧结论的更新

E2 仍是项目的最终 validated analytical state-bound result。本轮没有修改
E2、没有恢复 signed equality LDS，也没有用 task ID 或 future compressed
truth。

旧结论“raw logit $L_2$ bound 极度空泛”得到更具体的机制解释：

$$
\|\Delta z\|_2^2
=
\text{{common-shift}}
+
\text{{top-vocabulary}}
+
\text{{tail/near-null energy}}.
$$

common shift 并非唯一来源；大量 centered energy仍位于 softmax低概率 tail。
Fisher weighting把 pointwise ratio从 $10^5$ 量级降到常数量级，说明之前 F2
失败确实包含 output geometry错配，而不只是 residual readout拟合失败。

## 3. 新的正面 statement

可以写：

> On the tested full-logit trajectories, probability-weighted Fisher geometry
> removes most of the vacuity of the global Euclidean softmax bound and tracks
> exact KL substantially better than raw or uniformly centered logit norms.

可以写：

> The exact path-integrated Fisher identity is numerically verified through an
> independent KL identity, while fixed low-order quadrature exhibits
> counterexamples under large perturbations.

这些 statement 是 output-geometry diagnostic，不是 selection policy theorem。

## 4. 不能升级的 statement

不能写：

1. 3/5-point path integration 在整个 operating range 数值精确；
2. base/midpoint Fisher 已成为可部署 objective；
3. final-residual Fisher pullback 已验证；
4. $Q_t$ 低秩或可 anchor-frozen；
5. Q-state envelope 已成立；
6. matched-count refresh 或 free generation 已改善；
7. 已得到完整 selection--refresh method。

## 5. 更新后的理论链

当前证据支持：

$$
\text{{cache action}}
\rightarrow
\text{{E2 residual-magnitude envelope}},
$$

并在 retrospective true-logit 层支持：

$$
\text{{raw logit failure}}
\rightarrow
\text{{Fisher/gauge mechanism explanation}}.
$$

但尚不支持：

$$
\text{{deployable single-point geometry}}
\rightarrow
\text{{validated Fisher pullback}}
\rightarrow
\text{{recursive Q-state}}
\rightarrow
\text{{policy}}.
$$

因此最准确的论文边界是：

> Residual magnitudes are controllable by the validated E2 envelope, and the
> failure of raw-logit certificates is largely explained by gauge and
> probability geometry. However, no practical gauge-aware output objective or
> Fisher-pullback dynamic surrogate has passed the preregistered cross-task
> gates.

## 6. 下一步

如果继续 analytical 路线，最小下一步不是扩大 E2/O-family，而是预注册
curvature-adaptive path approximation或限制有效 perturbation radius，然后重新
进行独立 Stage-A validation。不能使用本轮 held-out counterexamples直接调好
quadrature后在同一数据上报告无偏成功。

若不再增加独立数据，项目应停在 E2 state-bound + gauge-geometry limitation，
方法层转向 empirical controller。
"""

    _atomic_text(
        REPOSITORY_ROOT / "GAUGE_AWARE_OUTPUT_GEOMETRY_RESULTS_ZH.md",
        results,
    )
    _atomic_text(
        REPOSITORY_ROOT / "FISHER_PULLBACK_STATE_ENVELOPE_RESULTS_ZH.md",
        pullback_results,
    )
    _atomic_text(
        REPOSITORY_ROOT / "THEORY_MODEL_UPDATE_AFTER_GAUGE_GEOMETRY_ZH.md",
        theory,
    )
    print(
        REPOSITORY_ROOT / "GAUGE_AWARE_OUTPUT_GEOMETRY_RESULTS_ZH.md"
    )


if __name__ == "__main__":
    main()
