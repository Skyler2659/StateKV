"""External-validity probes on the 3072 decomposition run.

Probe 1 (coverage classification): attention mass beyond the S1 core
cutoff (rank > 220) and beyond the S2 core cutoff (rank > 28), per task.
If qk_pool fails at S2 while mass-beyond-28 is large, the failure is
coverage, not ranking (no in-budget ranking can hold that mass).

Probe 2 (hard-cycle predictability, HF1b retest at long context):
cycle-level features from token_rows (missed core mass, attention
entropy, top-1, margin stats, cycle index) vs cycle exact KL from
step_rows.  At 768 no observable predicted hard cycles (best |rho| 0.36
for cycle index, which is not a runtime action signal).

Probe 3 (R-A swap-oracle rule): swap_rows at 3072 vs 768 — fraction of
budget-preserving cutoff swaps that *improve* exact 1-step KL, median
regret, and regret as a fraction of base KL, per task and offset.

Usage:
  .venv/bin/python analysis/tables/extval_probes.py \
      --decomp results/temporal_cache_discovery/statekv_extval_decomp_3072_256_v1 \
      --decomp-768 results/temporal_cache_discovery/statekv_qkv_decomposition_qwen3_8b_v1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "analysis/tables"
CORE_256 = 220
CORE_64 = 28


def _task_label(task: str) -> str:
    task = str(task).lower()
    if "reasoning" in task:
        return "reasoning"
    if "niah" in task:
        return "niah"
    if "gov" in task:
        return "govreport"
    return str(task)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    return float(np.corrcoef(rx, ry)[0, 1])


def probe_coverage(token: pd.DataFrame) -> pd.DataFrame:
    rows = []
    eligible = token[token["rank"] > 0]
    for (sample_id, task, cycle, layer), g in eligible.groupby(
        ["sample_id", "task", "cycle", "layer"]
    ):
        total = float(g["attn"].sum())
        beyond_220 = float(g.loc[g["rank"] > CORE_256, "attn"].sum())
        beyond_28 = float(g.loc[g["rank"] > CORE_64, "attn"].sum())
        rows.append(
            {
                "sample_id": sample_id,
                "task": _task_label(task),
                "cycle": int(cycle),
                "layer": int(layer),
                "mass_beyond_core256": beyond_220 / max(total, 1e-12),
                "mass_beyond_core64": beyond_28 / max(total, 1e-12),
            }
        )
    frame = pd.DataFrame(rows)
    summary = (
        frame.groupby("task")
        .agg(
            mass_beyond_core256_mean=("mass_beyond_core256", "mean"),
            mass_beyond_core256_p95=(
                "mass_beyond_core256",
                lambda s: s.quantile(0.95),
            ),
            mass_beyond_core64_mean=("mass_beyond_core64", "mean"),
            mass_beyond_core64_p95=(
                "mass_beyond_core64",
                lambda s: s.quantile(0.95),
            ),
        )
        .reset_index()
    )
    return summary


def probe_hard_cycles(token: pd.DataFrame, steps: pd.DataFrame) -> pd.DataFrame:
    eligible = token[token["rank"] > 0]
    feats = []
    for (sample_id, task, cycle), g in eligible.groupby(
        ["sample_id", "task", "cycle"]
    ):
        per_layer = g.groupby("layer")["attn"].sum()
        total = float(per_layer.sum())
        missed = float(
            g.loc[~g["in_core"]].groupby("layer")["attn"].sum().sum()
        )
        attn = g["attn"].to_numpy()
        attn = attn / max(attn.sum(), 1e-12)
        entropy = float(-(attn * np.log(attn + 1e-12)).sum())
        feats.append(
            {
                "sample_id": sample_id,
                "task": _task_label(task),
                "cycle": int(cycle),
                "missed_core_mass": missed / max(total, 1e-12),
                "attn_entropy": entropy,
                "cycle_index": int(cycle),
            }
        )
    feat_frame = pd.DataFrame(feats)
    cycle_kl = steps.groupby(["sample_id", "cycle"])["exact_kl"].mean().reset_index()
    data = feat_frame.merge(cycle_kl, on=["sample_id", "cycle"])
    rows = []
    for task, g in data.groupby("task"):
        kl = g["exact_kl"].to_numpy()
        threshold = np.quantile(kl, 0.9)
        hard = kl >= threshold
        base_rate = float(hard.mean())
        row = {"task": task, "cycles": int(len(g)), "p90_kl": float(threshold)}
        for feature in ["missed_core_mass", "attn_entropy", "cycle_index"]:
            row["rho_%s" % feature] = _spearman(
                g[feature].to_numpy(), kl
            )
            top = g[feature] >= g[feature].quantile(0.9)
            row["lift_%s" % feature] = float(
                (hard & top.to_numpy()).sum() / max(top.sum(), 1) / base_rate
            )
        rows.append(row)
    return pd.DataFrame(rows)


def probe_swaps(run: Path, label: str) -> pd.DataFrame:
    swap = pd.read_parquet(run / "swap_rows.parquet")
    swap["task_label"] = swap["task"].map(_task_label)
    swap["improves"] = swap["swap_kl"] < swap["base_kl"] - 1e-12
    rows = []
    for (task, offset), g in swap.groupby(["task_label", "offset"]):
        base = g["base_kl"].clip(lower=1e-12)
        rows.append(
            {
                "context": label,
                "task": task,
                "offset": int(offset),
                "pairs": int(len(g)),
                "cycles": int(g[["sample_id", "cycle"]].drop_duplicates().shape[0]),
                "frac_improves": float(g["improves"].mean()),
                "median_regret": float(g["swap_regret"].median()),
                "p05_regret": float(g["swap_regret"].quantile(0.05)),
                "median_rel_improvement": float(
                    ((g["base_kl"] - g["swap_kl"]) / base).median()
                ),
                "best_rel_improvement": float(
                    ((g["base_kl"] - g["swap_kl"]) / base).max()
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decomp", type=Path, required=True)
    parser.add_argument("--decomp-768", type=Path, required=True)
    args = parser.parse_args()

    columns = [
        "sample_id", "task", "cycle", "layer", "attn", "rank", "in_core",
    ]
    token = pd.read_parquet(args.decomp / "token_rows.parquet", columns=columns)
    steps = pd.read_parquet(args.decomp / "step_rows.parquet")

    coverage = probe_coverage(token)
    coverage.to_csv(OUT_DIR / "extval_coverage_classification.csv", index=False)
    print("[probe 1] coverage classification (attention mass beyond cutoff)")
    print(coverage.to_string(index=False))

    hard = probe_hard_cycles(token, steps)
    hard.to_csv(OUT_DIR / "extval_hard_cycle_predictability.csv", index=False)
    print("[probe 2] hard-cycle predictability @3072")
    print(hard.to_string(index=False))

    swaps = pd.concat(
        [
            probe_swaps(args.decomp, "3072"),
            probe_swaps(args.decomp_768, "768"),
        ]
    )
    swaps.to_csv(OUT_DIR / "extval_swap_regret.csv", index=False)
    print("[probe 3] swap-oracle regret by context/task/offset")
    print(swaps.to_string(index=False))

    summary = {
        "coverage": coverage.to_dict("records"),
        "hard_cycles": hard.to_dict("records"),
        "swap_frac_improves_3072": float(
            swaps[swaps["context"] == "3072"]["frac_improves"].mean()
        ),
        "swap_frac_improves_768": float(
            swaps[swaps["context"] == "768"]["frac_improves"].mean()
        ),
    }
    (OUT_DIR / "extval_probes_summary.json").write_text(
        json.dumps(summary, indent=2)
    )


if __name__ == "__main__":
    main()
