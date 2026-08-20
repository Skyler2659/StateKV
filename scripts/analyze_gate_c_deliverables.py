#!/usr/bin/env python3
"""Gate C deliverables: derived CSVs, verdict JSON, and plots.

Reads only the merged formal artifacts under
results/statekv_existence/causal_existence_qwen3_8b_v1/closed_loop/closed_loop_test/
(merged by scripts/run_strict_causal_closed_loop.py --merge-shards) and the
official gate decision (gate_c_failed.json / gate_c_passed.json written by
scripts/report_statekv_existence.py --evaluate-closed-loop). It changes no
experiment inputs and recomputes no arms.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/statekv_existence/causal_existence_qwen3_8b_v1"
CL = RUN / "closed_loop" / "closed_loop_test"
PLOTS = ROOT / "plots/statekv_existence"
PLOTS.mkdir(parents=True, exist_ok=True)

R2 = "STRICT_CAUSAL_ROLLOUT_R2"
EMA = "STRICT_BEST_PER_HEAD_FIXED_EMA"
FULL = "FULL_CACHE_REFERENCE"
POLICY_ORDER = [
    "STRICT_QK_CURRENT",
    "STRICT_H2O_CUMULATIVE",
    "STRICT_SNAPKV_OBSWIN",
    EMA,
    R2,
]
SHORT = {
    "STRICT_QK_CURRENT": "QK-current",
    "STRICT_H2O_CUMULATIVE": "H2O",
    "STRICT_SNAPKV_OBSWIN": "SnapKV",
    EMA: "fixed-EMA",
    R2: "R2 causal",
    FULL: "full-cache",
}


def main() -> None:
    summary = pd.read_csv(CL / "sample_summary.csv")
    steps = pd.read_parquet(CL / "step_rows.parquet")
    paired = pd.read_csv(CL / "paired_comparison.csv")
    gate_path = RUN / "gate_c_failed.json"
    gate_passed = False
    if not gate_path.exists():
        gate_path = RUN / "gate_c_passed.json"
        gate_passed = True
    gate = json.loads(gate_path.read_text())

    matched = summary[summary["policy"] != FULL].copy()

    # 1. sequence-level metrics (drop free-text generation)
    seq_cols = [
        "sample_id", "task", "task_bucket", "policy", "budget",
        "mean_trajectory_exact_kl", "needle_retrieval_accuracy",
        "official_score", "rouge_l", "rouge_1", "rouge_2",
        "repetition_4gram_rate", "generation_length_tokens",
        "wall_time_s", "causal_teacher_time_s", "causal_teacher_refreshes",
        "peak_active_cache_tokens", "refresh_frequency",
        "strict_pure_eviction", "recoverable_cold_tokens",
    ]
    summary[seq_cols].to_csv(CL / "closed_loop_sequence_metrics.csv", index=False)

    # 2. step-level secondary metrics per sequence (mean over cycles)
    step_agg = (
        steps.groupby(["sample_id", "policy", "budget"], as_index=False)[
            ["exact_kl", "exact_js", "delta_nll", "js", "logit_l2_sq",
             "fisher_quadratic", "pool_scoring_time_s", "causal_teacher_time_s"]
        ].mean()
    )
    step_agg.to_csv(CL / "closed_loop_step_metrics_by_sequence.csv", index=False)

    # 3. policy x budget aggregate
    def _num(frame, col):
        return pd.to_numeric(frame[col], errors="coerce")

    agg_rows = []
    for (policy, budget), g in matched.groupby(["policy", "budget"]):
        s = step_agg[(step_agg["policy"] == policy) & (step_agg["budget"] == budget)]
        agg_rows.append({
            "policy": policy,
            "budget": int(budget),
            "n_sequences": int(g["sample_id"].nunique()),
            "mean_exact_kl": _num(g, "mean_trajectory_exact_kl").mean(),
            "median_exact_kl": _num(g, "mean_trajectory_exact_kl").median(),
            "mean_exact_js": s["exact_js"].mean(),
            "mean_delta_nll": s["delta_nll"].mean(),
            "mean_logit_l2_sq": s["logit_l2_sq"].mean(),
            "mean_needle_accuracy": _num(g, "needle_retrieval_accuracy").mean(),
            "mean_official_score": _num(g, "official_score").mean(),
            "mean_rouge_l": _num(g, "rouge_l").mean(),
            "mean_wall_time_s": _num(g, "wall_time_s").mean(),
            "mean_teacher_time_s": _num(g, "causal_teacher_time_s").mean(),
        })
    aggregate = pd.DataFrame(agg_rows).sort_values(["budget", "mean_exact_kl"])
    aggregate.to_csv(CL / "closed_loop_aggregate.csv", index=False)

    # 4. paired bootstrap table (already 20k cluster bootstrap from merge)
    paired.to_csv(CL / "closed_loop_paired_bootstrap.csv", index=False)

    # 5. task breakdown of the primary comparison
    pivot_kl = matched.pivot_table(
        index=["sample_id", "budget"], columns="policy",
        values="mean_trajectory_exact_kl",
    )
    buckets = matched.drop_duplicates("sample_id").set_index("sample_id")["task_bucket"]
    tb_rows = []
    for budget in (128, 256):
        sub = pivot_kl.xs(budget, level="budget")
        delta = sub[EMA] - sub[R2]
        for bucket, vals in delta.groupby(buckets):
            tb_rows.append({
                "budget": budget,
                "task_bucket": bucket,
                "n_sequences": int(len(vals)),
                "mean_kl_improvement": float(vals.mean()),
                "median_kl_improvement": float(vals.median()),
                "win_rate": float((vals > 0).mean()),
            })
    task_breakdown = pd.DataFrame(tb_rows)
    task_breakdown.to_csv(CL / "closed_loop_task_breakdown.csv", index=False)

    # 6. runtime costs (vs full-cache reference wall time)
    full_wall = pd.to_numeric(
        summary.loc[summary["policy"] == FULL, "wall_time_s"], errors="coerce"
    ).mean()
    rt_rows = []
    for (policy, budget), g in matched.groupby(["policy", "budget"]):
        wall = _num(g, "wall_time_s").mean()
        rt_rows.append({
            "policy": policy,
            "budget": int(budget),
            "mean_wall_time_s": wall,
            "runtime_multiplier_vs_full_cache": wall / full_wall,
            "mean_teacher_time_s": _num(g, "causal_teacher_time_s").mean(),
            "mean_teacher_refreshes": _num(g, "causal_teacher_refreshes").mean(),
            "mean_pool_scoring_time_s": step_agg[
                (step_agg["policy"] == policy) & (step_agg["budget"] == budget)
            ]["pool_scoring_time_s"].mean(),
        })
    rt_rows.append({
        "policy": FULL, "budget": 0, "mean_wall_time_s": full_wall,
        "runtime_multiplier_vs_full_cache": 1.0,
        "mean_teacher_time_s": 0.0, "mean_teacher_refreshes": 0.0,
        "mean_pool_scoring_time_s": 0.0,
    })
    runtime = pd.DataFrame(rt_rows)
    runtime.to_csv(CL / "runtime_costs.csv", index=False)

    # 7. verdict json
    verdict = {
        "evaluated_at": gate["evaluated_at"],
        "gate_c_verdict": "PASS" if gate_passed else "FAIL",
        "gate_c_closed_loop_usefulness": gate["gate_c_closed_loop_usefulness"],
        "rule": {
            "primary_baseline": gate["primary_baseline"],
            "require_all_budgets": gate["require_all_budgets"],
            "require_ci_lower_above": 0.0,
            "require_majority_sequence_wins": True,
            "bootstrap_repetitions": 20000,
        },
        "budget_results": gate["budget_results"],
        "all_baseline_comparisons": paired.to_dict("records"),
        "task_breakdown_primary": task_breakdown.to_dict("records"),
        "artifacts": {
            "merged_dir": str(CL.relative_to(ROOT)),
            "official_gate_file": gate_path.name,
        },
    }
    (RUN / "gate_c_verdict.json").write_text(json.dumps(verdict, indent=2))

    # ---- plots ----
    # P1: exact KL by method x budget
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.35
    for i, budget in enumerate((128, 256)):
        vals = [
            aggregate[(aggregate["policy"] == p) & (aggregate["budget"] == budget)]
            ["mean_exact_kl"].iloc[0]
            for p in POLICY_ORDER
        ]
        ax.bar([x + i * width for x in range(len(POLICY_ORDER))], vals, width,
               label=f"budget {budget}")
    ax.set_xticks([x + width / 2 for x in range(len(POLICY_ORDER))])
    ax.set_xticklabels([SHORT[p] for p in POLICY_ORDER])
    ax.set_ylabel("mean trajectory exact KL (vs full cache)")
    ax.legend()
    ax.set_title("Gate C closed loop: exact KL by method x budget (n=30 each)")
    fig.tight_layout()
    fig.savefig(PLOTS / "gate_c_kl_by_method_budget.png", dpi=150)
    plt.close(fig)

    # P2: paired R2 improvement distribution
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for ax, budget in zip(axes, (128, 256)):
        sub = pivot_kl.xs(budget, level="budget")
        delta = sub[EMA] - sub[R2]
        ax.hist(delta, bins=15)
        ax.axvline(0, color="k", lw=1)
        row = paired[(paired["budget"] == budget) & (paired["primary_comparison"])]
        ax.set_title(
            f"budget {budget}\nmean {row['mean_kl_improvement'].iloc[0]:+.3f} "
            f"CI [{row['ci_low'].iloc[0]:+.3f}, {row['ci_high'].iloc[0]:+.3f}]"
        )
        ax.set_xlabel("paired KL improvement (EMA - R2)")
    axes[0].set_ylabel("sequences")
    fig.suptitle("Gate C primary comparison: R2 vs fixed-EMA")
    fig.tight_layout()
    fig.savefig(PLOTS / "gate_c_paired_r2_improvement.png", dpi=150)
    plt.close(fig)

    # P3: task breakdown
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for i, budget in enumerate((128, 256)):
        sub = task_breakdown[task_breakdown["budget"] == budget].set_index("task_bucket")
        ax.bar(
            [x + i * width for x in range(len(sub))],
            sub["mean_kl_improvement"], width, label=f"budget {budget}",
        )
    labels = task_breakdown[task_breakdown["budget"] == 128]["task_bucket"]
    ax.set_xticks([x + width / 2 for x in range(len(labels))])
    ax.set_xticklabels(labels)
    ax.axhline(0, color="k", lw=1)
    ax.set_ylabel("mean paired KL improvement (EMA - R2)")
    ax.set_title("Gate C primary comparison by task bucket")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "gate_c_task_breakdown.png", dpi=150)
    plt.close(fig)

    # P4: quality vs compute
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for budget, marker in ((128, "o"), (256, "s")):
        for p in POLICY_ORDER:
            row = aggregate[(aggregate["policy"] == p) & (aggregate["budget"] == budget)]
            rrow = runtime[(runtime["policy"] == p) & (runtime["budget"] == budget)]
            ax.scatter(
                rrow["runtime_multiplier_vs_full_cache"].iloc[0],
                row["mean_exact_kl"].iloc[0],
                marker=marker, s=70,
            )
            ax.annotate(
                f"{SHORT[p]}@{budget}",
                (rrow["runtime_multiplier_vs_full_cache"].iloc[0],
                 row["mean_exact_kl"].iloc[0]),
                fontsize=8,
            )
    ax.set_xscale("log")
    ax.set_xlabel("runtime multiplier vs full cache (log)")
    ax.set_ylabel("mean trajectory exact KL")
    ax.set_title("Gate C: quality vs compute")
    fig.tight_layout()
    fig.savefig(PLOTS / "gate_c_quality_vs_compute.png", dpi=150)
    plt.close(fig)

    # P5: causal predictability recovery vs closed-loop quality
    lb = pd.read_csv(ROOT / "results/statekv_existence/leaderboard.csv")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    h32 = lb[(lb["future_horizon"] == 32) & (lb["test_split"] == "validation")]
    for _, row in h32.iterrows():
        ax.scatter(row["runtime_multiplier"], row["oracle_gap_recovery"], s=60)
        ax.annotate(row["method"], (row["runtime_multiplier"], row["oracle_gap_recovery"]),
                    fontsize=7)
    r2_kl_gain = paired[paired["primary_comparison"]]["mean_kl_improvement"].mean()
    ax.set_xscale("log")
    ax.set_xlabel("runtime multiplier (log)")
    ax.set_ylabel("oracle-gap recovery (validation, H=32)")
    ax.set_title(
        "Predictability is not closed-loop quality: Gate A/B recovery vs compute\n"
        f"(R2 closed-loop mean KL improvement vs EMA: {r2_kl_gain:+.3f}, Gate C FAIL)"
    )
    fig.tight_layout()
    fig.savefig(PLOTS / "gate_c_recovery_vs_closed_loop.png", dpi=150)
    plt.close(fig)

    print("deliverables written:")
    for name in ("closed_loop_sequence_metrics.csv",
                 "closed_loop_step_metrics_by_sequence.csv",
                 "closed_loop_aggregate.csv", "closed_loop_paired_bootstrap.csv",
                 "closed_loop_task_breakdown.csv", "runtime_costs.csv"):
        print(" ", CL / name)
    print(" ", RUN / "gate_c_verdict.json")
    for png in sorted(PLOTS.glob("gate_c_*.png")):
        print(" ", png)


if __name__ == "__main__":
    main()
