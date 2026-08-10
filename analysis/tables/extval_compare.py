"""External-validity gate: main comparison tables.

Per-run, per-policy, per-task reads (no cross-task averaging — GovReport
~6 and NIAH 0/1 must never share a mean; see open-search artifact note).

Outputs (analysis/tables/extval_*.csv):
- extval_main.csv: run x policy x task: mean/median/p95 step KL, trajectory
  KL, task score means, repetition, prompt tokens, retained fraction.
- extval_paired.csv: paired qk_pool-vs-arm per sample (trajectory KL).
- extval_telemetry.csv: cycle-level churn / recovery / scoring-time means.
- extval_cadence.csv: qk_pool across cadence regimes (h1/h4/h16) per task.

Usage:
  .venv/bin/python analysis/tables/extval_compare.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/temporal_cache_discovery"
OUT_DIR = ROOT / "analysis/tables"

RUNS = {
    "s0_768_256": "statekv_recoverable_r0_qwen3_8b_v1",
    "s0_768_128": "statekv_openstress_768_128_v1",
    "s0_768_64": "statekv_openstress_768_64_v1",
    "s0_768_64_h4": "statekv_opencorner_768_64_h4_v1",
    "s0_768_64_h16": "statekv_opencorner_768_64_h16_v1",
    "s1_3072_256": "statekv_extval_3072_256_v1",
    "s1_3072_256_h4": "statekv_extval_3072_256_h4_v1",
    "s1_3072_256_h16": "statekv_extval_3072_256_h16_v1",
    "s2_3072_64": "statekv_extval_3072_64_v1",
    "s2_3072_64_h4": "statekv_extval_3072_64_h4_v1",
    "s2_3072_64_h16": "statekv_extval_3072_64_h16_v1",
    "s1_3072_256_mk": "statekv_extval_3072_256_mk_v1",
    "s1_3072_256_reasoning_af": "statekv_extval_3072_256_reasoning_af_v1",
    "s1_3072_256_qwen25_7b": "statekv_extval_3072_256_qwen25_7b_v1",
}


def _task_label(task: str, bucket: str) -> str:
    if bucket == "Reasoning" or "reasoning" in str(task).lower():
        return "reasoning"
    if "niah" in str(task).lower():
        return "niah"
    if "gov" in str(task).lower():
        return "govreport"
    return str(task)


def main() -> None:
    main_rows = []
    paired_rows = []
    telemetry_rows = []
    for regime, dirname in RUNS.items():
        run = RESULTS / dirname
        samples_path = run / "sample_results.csv"
        if not samples_path.exists():
            print("[skip] %s (no sample_results.csv yet)" % regime)
            continue
        samples = pd.read_csv(samples_path)
        samples["task_label"] = [
            _task_label(t, b)
            for t, b in zip(samples["task"], samples["task_bucket"])
        ]
        steps = pd.read_parquet(run / "step_rows.parquet")
        steps["task_label"] = steps["sample_id"].map(
            samples.drop_duplicates("sample_id").set_index("sample_id")[
                "task_label"
            ]
        )
        for (policy, task_label), group in samples.groupby(
            ["policy", "task_label"]
        ):
            step_group = steps[
                (steps["policy"] == policy)
                & (steps["task_label"] == task_label)
            ]
            main_rows.append(
                {
                    "regime": regime,
                    "policy": policy,
                    "task": task_label,
                    "n_samples": int(group["sample_id"].nunique()),
                    "prompt_tokens_mean": float(group["prompt_tokens"].mean()),
                    "retained_fraction_mean": float(
                        group["retained_prompt_fraction"].mean()
                    ),
                    "traj_kl_mean": float(
                        group["mean_trajectory_exact_kl"].mean()
                    ),
                    "traj_kl_median": float(
                        group["mean_trajectory_exact_kl"].median()
                    ),
                    "step_kl_mean": float(step_group["exact_kl"].mean()),
                    "step_kl_p95": float(step_group["exact_kl"].quantile(0.95)),
                    "niah_retrieval": float(
                        group["needle_retrieval_accuracy"].dropna().mean()
                    )
                    if group["needle_retrieval_accuracy"].notna().any()
                    else np.nan,
                    "govreport_rouge_l": float(group["rouge_l"].dropna().mean())
                    if group["rouge_l"].notna().any()
                    else np.nan,
                    "official_score": float(
                        group["official_score"].dropna().mean()
                    ),
                    "repetition_4gram": float(
                        group["repetition_4gram_rate"].mean()
                    ),
                }
            )
        qk = samples[samples["policy"] == "qk_pool"].set_index("sample_id")
        for policy in sorted(samples["policy"].unique()):
            if policy in {"qk_pool", "full_cache"}:
                continue
            arm = samples[samples["policy"] == policy].set_index("sample_id")
            shared = qk.index.intersection(arm.index)
            for sample_id in shared:
                paired_rows.append(
                    {
                        "regime": regime,
                        "arm": policy,
                        "sample_id": sample_id,
                        "task": _task_label(
                            arm.loc[sample_id, "task"],
                            arm.loc[sample_id, "task_bucket"],
                        ),
                        "qk_pool_kl": float(
                            qk.loc[sample_id, "mean_trajectory_exact_kl"]
                        ),
                        "arm_kl": float(
                            arm.loc[sample_id, "mean_trajectory_exact_kl"]
                        ),
                        "qk_wins": bool(
                            qk.loc[sample_id, "mean_trajectory_exact_kl"]
                            < arm.loc[sample_id, "mean_trajectory_exact_kl"]
                        ),
                    }
                )
        cycles_path = run / "cycle_rows.parquet"
        if cycles_path.exists():
            cycles = pd.read_parquet(cycles_path)
            for (policy, task), group in cycles.groupby(["policy", "task"]):
                telemetry_rows.append(
                    {
                        "regime": regime,
                        "policy": policy,
                        "task": _task_label(task, task),
                        "churn_mean": float(
                            group["selected_churn_layer_mean"].mean()
                        ),
                        "recovered_fraction_mean": float(
                            group["selected_recovered_fraction"].mean()
                        ),
                        "universe_mean": float(
                            group["candidate_universe_size"].mean()
                        ),
                        "scoring_forward_s_mean": float(
                            group["pool_scoring_forward_time_s"].mean()
                        ),
                    }
                )
    pd.DataFrame(main_rows).to_csv(OUT_DIR / "extval_main.csv", index=False)
    pd.DataFrame(paired_rows).to_csv(OUT_DIR / "extval_paired.csv", index=False)
    pd.DataFrame(telemetry_rows).to_csv(
        OUT_DIR / "extval_telemetry.csv", index=False
    )
    main = pd.DataFrame(main_rows)
    cadence = main[
        (main["policy"] == "qk_pool")
        & (main["regime"].str.contains("h4|h16|256|64"))
    ]
    cadence.to_csv(OUT_DIR / "extval_cadence.csv", index=False)
    print("[extval] wrote extval_{main,paired,telemetry,cadence}.csv")
    print(
        main[main["policy"] == "qk_pool"][
            ["regime", "task", "traj_kl_mean", "niah_retrieval",
             "govreport_rouge_l", "official_score"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
