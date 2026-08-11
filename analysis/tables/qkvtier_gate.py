"""QK-route, V-tier method gate analysis (preregistered in
docs/evidence/statekv_qkvtier_gate.md section 4; thresholds hard-coded, do not
tune after seeing results).

P  premise: tiered-256 KL <= 1.10 x qk256 KL and quality non-worse
G1 headroom: tiered-352 KL <= 0.80 x qk256 KL
G2 stability: tiered-352 paired wins >= 8/10 vs qk256
G3 tail: tiered-352 p95 step KL <= 1.05 x qk256 p95
G4 quality: tiered-352 non-worse (NIAH -0, GovReport >= baseline - 1.0)
G5 tiering fidelity: tiered-352 KL <= 1.10 x fp16-352 KL
G6 fairness: execution_valid + budgets in all runs

GO = P and G1..G6.  NO-GO subclass: PREMISE_FAILED / TIERING_LOSSY (G5) /
COVERAGE_WORTHLESS (G1/G2 with premise ok).

Usage:
  .venv/bin/python analysis/tables/qkvtier_gate.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
R0 = ROOT / "results/temporal_cache_discovery/statekv_recoverable_r0_qwen3_8b_v1"
RUNS = {
    "qk_pool_256": R0,
    "qk_tiered_v_256": ROOT
    / "results/temporal_cache_discovery/statekv_qkvtier_gate_256t_v1",
    "qk_pool_352": ROOT
    / "results/temporal_cache_discovery/statekv_qkvtier_gate_352f_v1",
    "qk_tiered_v_352": ROOT
    / "results/temporal_cache_discovery/statekv_qkvtier_gate_352t_v1",
}
OUT = ROOT / "analysis/tables"
MD_DIR = ROOT / "docs/evidence/tables"

P_MAX_RATIO = 1.10
G1_MAX_RATIO = 0.80
G2_MIN_WINS = 8
G3_MAX_RATIO = 1.05
G4_MAX_DROP = 1.0
G5_MAX_RATIO = 1.10


ARM_POLICIES = {
    "qk_pool_256": "qk_pool",
    "qk_tiered_v_256": "qk_tiered_v",
    "qk_pool_352": "qk_pool",
    "qk_tiered_v_352": "qk_tiered_v",
}


def _load(arm):
    samples = pd.read_csv(RUNS[arm] / "sample_results.csv")
    steps = pd.read_parquet(RUNS[arm] / "step_rows.parquet")
    summary = json.loads((RUNS[arm] / "summary.json").read_text())
    # every run also writes full_cache ceiling rows (KL=0 by construction);
    # restrict strictly to the arm's own policy rows
    samples = samples[samples["policy"] == ARM_POLICIES[arm]]
    steps = steps[steps["policy"] == ARM_POLICIES[arm]]
    return samples, steps, summary


def _agg(samples, steps):
    return {
        "mean_kl": float(samples["mean_trajectory_exact_kl"].mean()),
        "median_kl": float(samples["mean_trajectory_exact_kl"].median()),
        "p95_step_kl": float(steps["exact_kl"].quantile(0.95)),
        "mean_niah": float(samples["needle_retrieval_accuracy"].mean()),
        "mean_gov_official": float(
            samples.loc[samples["task_bucket"] == "GovReport", "official_score"].mean()
        ),
    }


def main() -> None:
    data = {arm: _load(arm) for arm in RUNS}
    agg = {arm: _agg(samples, steps) for arm, (samples, steps, _) in data.items()}
    table = pd.DataFrame(agg).T.reset_index().rename(columns={"index": "arm"})
    table.to_csv(OUT / "qkvtier_gate_main.csv", index=False)

    base = data["qk_pool_256"][0].set_index("sample_id")["mean_trajectory_exact_kl"]
    paired_rows = []
    for arm in ("qk_tiered_v_256", "qk_pool_352", "qk_tiered_v_352"):
        other = data[arm][0].set_index("sample_id")["mean_trajectory_exact_kl"]
        common = sorted(set(base.index) & set(other.index))
        diffs = np.asarray([other[s] - base[s] for s in common])
        paired_rows.append(
            {
                "arm": arm,
                "paired": len(common),
                "mean_arm_minus_qk256": float(diffs.mean()),
                "wins_vs_qk256": int(np.sum(diffs < -1e-12)),
                "losses_vs_qk256": int(np.sum(diffs > 1e-12)),
                "ties": int(np.sum(np.abs(diffs) <= 1e-12)),
            }
        )
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(OUT / "qkvtier_gate_paired.csv", index=False)

    # verdict
    qk256, t256, f352, t352 = (agg[a] for a in ("qk_pool_256", "qk_tiered_v_256", "qk_pool_352", "qk_tiered_v_352"))
    premise = bool(
        t256["mean_kl"] <= P_MAX_RATIO * qk256["mean_kl"]
        and t256["mean_niah"] >= qk256["mean_niah"]
        and t256["mean_gov_official"] >= qk256["mean_gov_official"] - G4_MAX_DROP
    )
    g1 = bool(t352["mean_kl"] <= G1_MAX_RATIO * qk256["mean_kl"])
    g2 = bool(
        int(paired.loc[paired["arm"] == "qk_tiered_v_352", "wins_vs_qk256"].iloc[0])
        >= G2_MIN_WINS
    )
    g3 = bool(t352["p95_step_kl"] <= G3_MAX_RATIO * qk256["p95_step_kl"])
    g4 = bool(
        t352["mean_niah"] >= qk256["mean_niah"]
        and t352["mean_gov_official"] >= qk256["mean_gov_official"] - G4_MAX_DROP
    )
    g5 = bool(t352["mean_kl"] <= G5_MAX_RATIO * f352["mean_kl"])
    g6 = bool(
        all(
            summary.get("execution_valid") and summary.get("all_budgets_respected")
            for _, _, summary in data.values()
        )
    )
    go = bool(premise and g1 and g2 and g3 and g4 and g5 and g6)
    if not premise:
        subclass = "PREMISE_FAILED"
    elif not g5:
        subclass = "TIERING_LOSSY"
    else:
        subclass = "COVERAGE_WORTHLESS"
    verdict = "GO" if go else "NO_GO"

    note = [
        "# QK-route, V-tier method gate (preregistered verdict)",
        "",
        table.to_markdown(index=False),
        "",
        paired.to_markdown(index=False),
        "",
        f"P premise (tiered-256 within {P_MAX_RATIO}x qk256 + quality): {premise} "
        f"(ratio {t256['mean_kl']/qk256['mean_kl']:.3f})",
        f"C coverage worth: fp16-352 KL {f352['mean_kl']:.4f} vs qk256 {qk256['mean_kl']:.4f} "
        f"(ratio {f352['mean_kl']/qk256['mean_kl']:.3f})",
        f"G1 tiered-352 <= {G1_MAX_RATIO}x qk256: {g1} (ratio {t352['mean_kl']/qk256['mean_kl']:.3f})",
        f"G2 wins >= {G2_MIN_WINS}/10: {g2}",
        f"G3 p95 tiered-352 {t352['p95_step_kl']:.4f} <= {G3_MAX_RATIO}x qk256 {qk256['p95_step_kl']:.4f}: {g3}",
        f"G4 quality non-worse: {g4}",
        f"G5 tiered-352 <= {G5_MAX_RATIO}x fp16-352: {g5} (ratio {t352['mean_kl']/f352['mean_kl']:.3f})",
        f"G6 fairness flags: {g6}",
        "",
        f"**Gate verdict (preregistered): {verdict}**" + ("" if go else f" ({subclass})"),
        "",
    ]
    (MD_DIR / "qkvtier_gate_main.md").write_text("\n".join(note), encoding="utf-8")
    print("\n".join(note))


if __name__ == "__main__":
    main()
