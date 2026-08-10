"""QK-V residual analysis battery (discovery protocol Phase 2, questions A-F).

Consumes the decomposition dataset (token_rows / head_rows / swap_rows)
and answers, in order:

A. dynamic range: Var(log attention) vs Var(log V features) per layer/task/phase
B. ranking overlap: Top-K(attn) vs Top-K(delta/apv), Jaccard, recall of
   future-relevant tokens
C. near-tie: QK-conditioned residual of V features by cutoff bucket +
   swap-oracle exact-target analysis (sign/magnitude of swap regret)
D. head/layer specialization of the residual
E. token-type specialization
F. horizon: cheap features -> future relevance at h=1/2/4/8, dev-fit /
   test-eval linear probe

The core quantity everywhere is I(target; V | QK): partial (residualized)
rank correlation of V features against targets after removing the
attention component.

Usage:
  .venv/bin/python analysis/tables/qkv_residual_analysis.py \
      --run results/temporal_cache_discovery/statekv_qkv_decomposition_qwen3_8b_v1
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = (
    ROOT / "results/temporal_cache_discovery/statekv_qkv_decomposition_qwen3_8b_v1"
)
OUT_DIR = ROOT / "analysis/tables"
HORIZONS = (1, 2, 4, 8)
CORE_BUDGET = 220
DEV_SAMPLES = {"synthetic_niah_86", "synthetic_niah_87", "gov_report:86"}


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    return float(np.corrcoef(rx, ry)[0, 1])


def _residualize(values: np.ndarray, control: np.ndarray) -> np.ndarray:
    """Rank-residualize values on control (linear fit on ranks)."""

    mask = np.isfinite(values) & np.isfinite(control)
    out = np.full(len(values), np.nan)
    if mask.sum() < 3:
        return out
    rv = pd.Series(values[mask]).rank().to_numpy()
    rc = pd.Series(control[mask]).rank().to_numpy()
    rc = (rc - rc.mean()) / (rc.std() + 1e-12)
    beta = float(np.dot(rv - rv.mean(), rc) / (np.dot(rc, rc) + 1e-12))
    out[mask] = (rv - rv.mean()) - beta * rc
    return out


def _partial_spearman(feature: np.ndarray, target: np.ndarray, control: np.ndarray) -> float:
    return _spearman(_residualize(feature, control), _residualize(target, control))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--out-prefix", default="qkv_")
    parser.add_argument("--dev-samples", nargs="*", default=None)
    args = parser.parse_args()
    run = args.run.resolve()
    out_prefix = str(args.out_prefix)
    dev_samples = set(args.dev_samples) if args.dev_samples else DEV_SAMPLES

    summary = json.loads((run / "summary.json").read_text())
    columns = [
        "sample_id", "task", "cycle", "layer", "position", "attn", "delta",
        "pv", "vn", "apv", "rank", "margin", "in_core", "token_class", "is_needle",
    ]
    token = pd.read_parquet(run / "token_rows.parquet", columns=columns)
    token = token[token["rank"] > 0].copy()  # eligible only
    from statekv.qkv_decomposition import add_future_targets

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        token = add_future_targets(token, HORIZONS, CORE_BUDGET)

    outputs = {}

    # ---- consistency: trajectory KL must match R0 qk_pool ----
    sample_summary = pd.read_csv(run / "sample_summary.csv")
    outputs["trajectory_kl_mean"] = float(
        sample_summary["mean_trajectory_exact_kl"].mean()
    )

    # ---- A. dynamic range ----
    a_rows = []
    eligible = token
    for (layer, task), group in eligible.groupby(["layer", "task"]):
        a_rows.append(
            {
                "layer": int(layer),
                "task": str(task),
                "var_log_attn": float(np.var(np.log(group["attn"] + 1e-12))),
                "var_log_vn": float(np.var(np.log(group["vn"] + 1e-12))),
                "var_log_pv": float(np.var(np.log(group["pv"] + 1e-12))),
                "var_log_delta": float(np.var(np.log(group["delta"] + 1e-12))),
                "rows": int(len(group)),
            }
        )
    table_a = pd.DataFrame(a_rows)
    table_a.to_csv(OUT_DIR / f"{out_prefix}a_dynamic_range.csv", index=False)
    phase = pd.cut(token["cycle"], bins=[-1, 15, 31, 47, 63], labels=["q1", "q2", "q3", "q4"])
    table_a_phase = (
        token.groupby(phase)
        .apply(
            lambda g: pd.Series(
                {
                    "var_log_attn": float(np.var(np.log(g["attn"] + 1e-12))),
                    "var_log_pv": float(np.var(np.log(g["pv"] + 1e-12))),
                    "var_log_delta": float(np.var(np.log(g["delta"] + 1e-12))),
                }
            )
        )
        .reset_index()
    )
    table_a_phase.to_csv(OUT_DIR / f"{out_prefix}a_dynamic_range_phase.csv", index=False)

    # ---- B. ranking overlap + future-relevance recall ----
    b_rows = []
    for (sample_id, cycle, layer), group in token.groupby(["sample_id", "cycle", "layer"]):
        ordered_attn = group.sort_values(["attn", "position"], ascending=[False, True])
        ordered_delta = group.sort_values(["delta", "position"], ascending=[False, True])
        ordered_apv = group.sort_values(["apv", "position"], ascending=[False, True])
        take = min(CORE_BUDGET, len(group))
        top_attn = set(ordered_attn["position"].head(take))
        top_delta = set(ordered_delta["position"].head(take))
        top_apv = set(ordered_apv["position"].head(take))
        future = group.dropna(subset=["fut_attn_4"])
        if len(future) >= take:
            top_future = set(
                future.sort_values(["fut_attn_4", "position"], ascending=[False, True])[
                    "position"
                ].head(take)
            )
            recall_attn = len(top_attn & top_future) / take
            recall_delta = len(top_delta & top_future) / take
            recall_apv = len(top_apv & top_future) / take
        else:
            recall_attn = recall_delta = recall_apv = float("nan")
        b_rows.append(
            {
                "sample_id": sample_id,
                "cycle": int(cycle),
                "layer": int(layer),
                "jaccard_attn_delta": len(top_attn & top_delta) / take,
                "jaccard_attn_apv": len(top_attn & top_apv) / take,
                "recall_future_attn": recall_attn,
                "recall_future_delta": recall_delta,
                "recall_future_apv": recall_apv,
            }
        )
    table_b = pd.DataFrame(b_rows)
    table_b.to_csv(OUT_DIR / f"{out_prefix}b_ranking_overlap.csv", index=False)
    table_b_summary = table_b.drop(columns=["sample_id", "cycle", "layer"]).mean()
    outputs["B_mean"] = {k: float(v) for k, v in table_b_summary.items()}

    # ---- C1. near-tie buckets: partial spearman of V features vs future ----
    distance = token["rank"] - CORE_BUDGET
    buckets = pd.cut(
        distance,
        bins=[-10**9, -64, -16, 0, 16, 64, 10**9],
        labels=["deep_inside", "near_inside", "boundary_in", "boundary_out", "near_out", "far_out"],
    )
    c_rows = []
    for bucket, group in token.groupby(buckets):
        for feature in ("delta", "pv", "vn", "apv"):
            c_rows.append(
                {
                    "bucket": str(bucket),
                    "feature": feature,
                    "rows": int(len(group)),
                    "spearman_raw": _spearman(
                        group[feature].to_numpy(), group["fut_attn_4"].to_numpy()
                    ),
                    "partial_spearman_given_attn": _partial_spearman(
                        group[feature].to_numpy(),
                        group["fut_attn_4"].to_numpy(),
                        group["attn"].to_numpy(),
                    ),
                    "partial_spearman_attn_given_feature": _partial_spearman(
                        group["attn"].to_numpy(),
                        group["fut_attn_4"].to_numpy(),
                        group[feature].to_numpy(),
                    ),
                }
            )
    table_c1 = pd.DataFrame(c_rows)
    table_c1.to_csv(OUT_DIR / f"{out_prefix}c1_neartie_partial.csv", index=False)

    # ---- C2. swap oracle: exact downstream target at the cutoff ----
    swap_path = run / "swap_rows.parquet"
    outputs["C2"] = {}
    if swap_path.exists():
        swap = pd.read_parquet(swap_path)
        swap["delta_diff"] = swap["inside_delta_mean"] - swap["outside_delta_mean"]
        swap["pv_diff"] = swap["inside_pv_mean"] - swap["outside_pv_mean"]
        swap["vn_diff"] = swap["inside_vn_mean"] - swap["outside_vn_mean"]
        swap["regret_negative"] = swap["swap_regret"] < -1e-9
        swap["regret_positive"] = swap["swap_regret"] > 1e-9
        c2 = {
            "pairs": int(len(swap)),
            "regret_mean": float(swap["swap_regret"].mean()),
            "regret_median": float(swap["swap_regret"].median()),
            "regret_p95_abs": float(swap["swap_regret"].abs().quantile(0.95)),
            "fraction_positive": float(swap["regret_positive"].mean()),
            "fraction_negative": float(swap["regret_negative"].mean()),
            "fraction_flat_1e-4": float((swap["swap_regret"].abs() < 1e-4).mean()),
            "spearman_margin_vs_regret": _spearman(
                swap["attn_margin"].to_numpy(), swap["swap_regret"].to_numpy()
            ),
            "spearman_delta_diff_vs_regret": _spearman(
                swap["delta_diff"].to_numpy(), swap["swap_regret"].to_numpy()
            ),
            "partial_delta_given_margin": _partial_spearman(
                swap["delta_diff"].to_numpy(),
                swap["swap_regret"].to_numpy(),
                swap["attn_margin"].to_numpy(),
            ),
            "partial_pv_given_margin": _partial_spearman(
                swap["pv_diff"].to_numpy(),
                swap["swap_regret"].to_numpy(),
                swap["attn_margin"].to_numpy(),
            ),
        }
        by_offset = (
            swap.groupby("offset")["swap_regret"]
            .agg(["mean", "median", "size"])
            .reset_index()
        )
        by_offset.to_csv(OUT_DIR / f"{out_prefix}c2_swap_by_offset.csv", index=False)
        outputs["C2"] = c2
        swap.to_csv(OUT_DIR / f"{out_prefix}c2_swap_rows.csv", index=False)

    # ---- D. per-layer and per-head residual ----
    d_rows = []
    for layer, group in token.groupby("layer"):
        d_rows.append(
            {
                "layer": int(layer),
                "partial_delta_given_attn": _partial_spearman(
                    group["delta"].to_numpy(),
                    group["fut_attn_4"].to_numpy(),
                    group["attn"].to_numpy(),
                ),
                "partial_pv_given_attn": _partial_spearman(
                    group["pv"].to_numpy(),
                    group["fut_attn_4"].to_numpy(),
                    group["attn"].to_numpy(),
                ),
                "spearman_attn_future": _spearman(
                    group["attn"].to_numpy(), group["fut_attn_4"].to_numpy()
                ),
            }
        )
    table_d = pd.DataFrame(d_rows)
    table_d.to_csv(OUT_DIR / f"{out_prefix}d_layer_residual.csv", index=False)
    head_path = run / "head_rows.parquet"
    if head_path.exists():
        head = pd.read_parquet(head_path)
        # head-level target: future attention per head is not stored; use
        # within-cycle correlation of delta vs attn (routing-payload
        # decoupling) as the head-specialization signal.
        h_rows = []
        for (layer, head_id), group in head.groupby(["layer", "head"]):
            h_rows.append(
                {
                    "layer": int(layer),
                    "head": int(head_id),
                    "spearman_attn_delta": _spearman(
                        group["attn"].to_numpy(), group["delta"].to_numpy()
                    ),
                    "var_log_attn": float(np.var(np.log(group["attn"] + 1e-12))),
                    "var_log_pv": float(np.var(np.log(group["pv"] + 1e-12))),
                }
            )
        table_h = pd.DataFrame(h_rows)
        table_h.to_csv(OUT_DIR / f"{out_prefix}d_head_decoupling.csv", index=False)

    # ---- E. token-type specialization ----
    e_rows = []
    for klass, group in token.groupby("token_class"):
        e_rows.append(
            {
                "token_class": str(klass),
                "rows": int(len(group)),
                "mean_attn": float(group["attn"].mean()),
                "mean_fut_attn_4": float(group["fut_attn_4"].mean()),
                "spearman_attn_future": _spearman(
                    group["attn"].to_numpy(), group["fut_attn_4"].to_numpy()
                ),
                "partial_delta_given_attn": _partial_spearman(
                    group["delta"].to_numpy(),
                    group["fut_attn_4"].to_numpy(),
                    group["attn"].to_numpy(),
                ),
                "revival_4_rate": float(group["revival_4"].mean()),
            }
        )
    table_e = pd.DataFrame(e_rows)
    table_e.to_csv(OUT_DIR / f"{out_prefix}e_token_type.csv", index=False)

    # needle tokens specifically
    needle = token[token["is_needle"]]
    if len(needle):
        outputs["E_needle"] = {
            "rows": int(len(needle)),
            "mean_attn": float(needle["attn"].mean()),
            "in_core_rate": float(needle["in_core"].mean()),
            "mean_fut_attn_4": float(needle["fut_attn_4"].mean()),
            "partial_delta_given_attn": _partial_spearman(
                needle["delta"].to_numpy(),
                needle["fut_attn_4"].to_numpy(),
                needle["attn"].to_numpy(),
            ),
        }

    # ---- F. horizon probe: dev-fit linear model on ranks, test-eval ----
    from sklearn.linear_model import LinearRegression

    f_rows = []
    features_qk = ["attn"]
    features_qkv = ["attn", "delta", "pv", "vn", "apv"]
    for horizon in HORIZONS:
        target = f"fut_attn_{horizon}"
        data = token.dropna(subset=[target])
        data = data[data["cycle"] <= 63 - horizon]
        dev = data[data["sample_id"].isin(dev_samples)]
        test = data[~data["sample_id"].isin(dev_samples)]
        eval_note = "heldout"
        if len(test) == 0:
            test = dev
            eval_note = "dev_only"

        def _rank_matrix(frame, cols):
            return np.column_stack(
                [pd.Series(frame[c].to_numpy()).rank().to_numpy() for c in cols]
            )

        for name, cols in (("qk_only", features_qk), ("qk_plus_v", features_qkv)):
            model = LinearRegression()
            x_dev = _rank_matrix(dev, cols)
            y_dev = pd.Series(dev[target].to_numpy()).rank().to_numpy()
            model.fit(x_dev, y_dev)
            pred = model.predict(_rank_matrix(test, cols))
            f_rows.append(
                {
                    "horizon": int(horizon),
                    "model": name,
                    "eval_set": eval_note,
                    "test_spearman": _spearman(
                        pred, test[target].to_numpy()
                    ),
                    "test_r2": float(
                        1
                        - np.var(pd.Series(test[target].to_numpy()).rank().to_numpy() - pred)
                        / np.var(pd.Series(test[target].to_numpy()).rank().to_numpy())
                    ),
                }
            )
    table_f = pd.DataFrame(f_rows)
    table_f.to_csv(OUT_DIR / f"{out_prefix}f_horizon_probe.csv", index=False)

    # ---- revival prediction (the cliff question) ----
    r_rows = []
    for horizon in HORIZONS:
        target = f"revival_{horizon}"
        data = token[~token["in_core"]]
        rate = float(data[target].mean())
        # does any feature separate revivers from non-revivers?
        rev = data[data[target]]
        non = data[~data[target]]
        row = {"horizon": int(horizon), "revival_rate": rate}
        for feature in ("attn", "delta", "pv", "vn"):
            row[f"mean_{feature}_revived"] = float(rev[feature].mean()) if len(rev) else float("nan")
            row[f"mean_{feature}_not_revived"] = float(non[feature].mean()) if len(non) else float("nan")
            row[f"spearman_{feature}_revival"] = _spearman(
                data[feature].to_numpy(), data[target].to_numpy(dtype=float)
            )
        r_rows.append(row)
    table_r = pd.DataFrame(r_rows)
    table_r.to_csv(OUT_DIR / f"{out_prefix}revival_prediction.csv", index=False)

    (OUT_DIR / f"{out_prefix}analysis_outputs.json").write_text(
        json.dumps(outputs, indent=2), encoding="utf-8"
    )
    print(json.dumps(outputs, indent=2))
    print("tables written to", OUT_DIR)


if __name__ == "__main__":
    main()
