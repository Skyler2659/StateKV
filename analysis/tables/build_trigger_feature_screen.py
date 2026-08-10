"""Offline action-conditioned trigger feature screen for StateKV refresh decisions.

Builds candidate trigger features from stored artifacts only (no model runs):
  - DEV (fit/select):  statekv_risk_consistent_proxy_alignment_p22_v1 (canonical 48 events)
  - TEST (frozen):     statekv_risk_consistent_proxy_independent_p23b_v1 (48 events)

Outputs (all under analysis/tables/):
  trigger_features_p22.csv / trigger_features_p23b.csv  per-event feature tables
  trigger_feature_metrics.csv                            Spearman / rank-AUC / precision@0.25
  trigger_rules_frozen.csv                               top-2 conjunction + product rules
  trigger_screen_report.md                               computability notes + verdict
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
P22 = ROOT / "results/temporal_cache_discovery/statekv_risk_consistent_proxy_alignment_p22_v1"
P23B = ROOT / "results/temporal_cache_discovery/statekv_risk_consistent_proxy_independent_p23b_v1"
OUT = ROOT / "analysis/tables"
PROXY = "attention_mean_w1_shared"
ALERT_RATE = 0.25


def rank_auc(feature: np.ndarray, positive: np.ndarray) -> float:
    """P(f_pos > f_neg) + 0.5 * P(tie); rank-AUC vs benefit>0."""
    pos = feature[positive]
    neg = feature[~positive]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = np.mean(pos[:, None] > neg[None, :])
    eq = np.mean(pos[:, None] == neg[None, :])
    return float(gt + 0.5 * eq)


def precision_at_rate(feature: np.ndarray, positive: np.ndarray, rate: float) -> float:
    k = int(round(len(feature) * rate))
    order = np.argsort(-feature, kind="stable")
    return float(np.mean(positive[order[:k]]))


def build_events(base: Path) -> pd.DataFrame:
    rr = pd.read_parquet(base / "refresh_regret_rows.parquet")
    ca = pd.read_parquet(base / "cross_action_rows.parquet")
    sr = pd.read_parquet(base / "stale_replay_rows.parquet")

    events = rr[rr.proxy == PROXY].copy().reset_index(drop=True)

    # ---- panel features (cross_action_rows, canonical proxy, 7 candidates) ----
    panel = ca[ca.proxy == PROXY].copy()
    rows = []
    for (sample, anchor, horizon), grp in panel.groupby(["sample_id", "anchor", "horizon"]):
        t = np.sort(grp.teacher_risk.to_numpy())
        p = np.sort(grp.proxy_risk.to_numpy())
        rows.append(
            {
                "sample_id": sample,
                "anchor": anchor,
                "horizon": horizon,
                "panel_margin_teacher": t[1] - t[0],
                "panel_spread_teacher": t[-1] - t[0],
                "panel_margin_proxy": p[1] - p[0],
                "panel_spread_proxy": p[-1] - p[0],
            }
        )
    panel_feat = pd.DataFrame(rows)
    events = events.merge(panel_feat, on=["sample_id", "anchor", "horizon"], how="left")

    # ---- stale trajectory drift (stale_replay_rows, horizon-independent) ----
    drift_rows = []
    for (sample, prev, anchor), grp in sr[sr.proxy == PROXY].groupby(
        ["sample_id", "previous_anchor", "anchor"]
    ):
        grp = grp.sort_values("horizon_offset")
        off = grp.horizon_offset.to_numpy(dtype=float)
        early = grp[grp.horizon_offset <= 4]
        x = early.horizon_offset.to_numpy(dtype=float)
        kl = early.exact_kl.to_numpy(dtype=float)
        slope = float(np.polyfit(x, kl, 1)[0])
        drift_rows.append(
            {
                "sample_id": sample,
                "previous_anchor": prev,
                "anchor": anchor,
                "kl_offset1": float(grp.exact_kl.iloc[0]),
                "kl_slope_early": slope,
                "fisher_mean_early": float(early.fisher_quadratic.mean()),
                "delta_nll_mean_early": float(early.delta_nll.mean()),
                "logit_l2_mean_early": float(early.logit_l2_sq.mean()),
                "kl_max_early": float(kl.max()),
                "n_offsets": int(off.max()),
            }
        )
    drift = pd.DataFrame(drift_rows)
    events = events.merge(drift, on=["sample_id", "previous_anchor", "anchor"], how="left")

    # ---- proxy-risk shape ----
    events["proxy_risk_ratio"] = events.stale_proxy_risk / events.fresh_proxy_risk.clip(lower=1e-12)

    # ---- churn: stale core == previous fresh core for 100% of events (hash-verified),
    # but consecutive-core intersection is NOT stored and proxy score vectors are NOT
    # persisted, so churn_jaccard is uncomputable offline. churn_binary (stale != fresh)
    # is constant 1 in both datasets -> dropped as a feature. ----
    events["churn_binary"] = (events.stale_action_id != events.fresh_action_id).astype(float)

    return events


def metric_table(events: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    benefit = events.teacher_refresh_benefit.to_numpy()
    positive = benefit > 0
    rows = []
    for f in features:
        x = events[f].to_numpy(dtype=float)
        if np.isnan(x).any() or np.allclose(x, x[0]):
            rows.append({"feature": f, "spearman": np.nan, "rank_auc": np.nan,
                         "precision_at_0.25": np.nan, "note": "constant or NaN"})
            continue
        rows.append(
            {
                "feature": f,
                "spearman": float(stats.spearmanr(x, benefit).statistic),
                "rank_auc": rank_auc(x, positive),
                "precision_at_0.25": precision_at_rate(x, positive, ALERT_RATE),
                "note": "",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    dev = build_events(P22)
    test = build_events(P23B)
    dev.to_csv(OUT / "trigger_features_p22.csv", index=False)
    test.to_csv(OUT / "trigger_features_p23b.csv", index=False)

    features = [
        "panel_margin_teacher",
        "panel_margin_proxy",
        "panel_spread_teacher",
        "panel_spread_proxy",
        "kl_offset1",
        "kl_slope_early",
        "kl_max_early",
        "fisher_mean_early",
        "delta_nll_mean_early",
        "logit_l2_mean_early",
        "fresh_proxy_risk",
        "stale_proxy_risk",
        "proxy_risk_ratio",
        "proxy_regret",   # failed baseline trigger
        "horizon",        # fixed-interval baseline
        "churn_binary",   # degenerate churn proxy (expected constant)
    ]
    m_dev = metric_table(dev, features).add_suffix("_p22").rename(columns={"feature_p22": "feature"})
    m_test = metric_table(test, features).add_suffix("_p23b").rename(columns={"feature_p23b": "feature"})
    metrics = m_dev.merge(m_test, on="feature", how="outer")
    metrics.to_csv(OUT / "trigger_feature_metrics.csv", index=False)
    pd.set_option("display.width", 200)
    print(metrics.to_string(index=False))

    # ---- select top-2 non-baseline features on DEV (rank_auc primary, spearman tiebreak) ----
    non_baseline = [f for f in features if f not in {"proxy_regret", "horizon", "churn_binary"}]
    sel = (
        metrics[metrics.feature.isin(non_baseline)]
        .sort_values(["rank_auc_p22", "spearman_p22"], ascending=False)
        .head(2)
    )
    f1, f2 = sel.feature.tolist()
    print("selected top-2 on P22:", f1, f2)

    # conjunction rule: thresholds chosen on DEV, alert rate ~0.25, max DEV precision
    best = None
    pos_dev = (dev.teacher_refresh_benefit > 0).to_numpy()
    for q1 in np.arange(0.40, 0.80, 0.05):
        for q2 in np.arange(0.40, 0.80, 0.05):
            t1, t2 = dev[f1].quantile(q1), dev[f2].quantile(q2)
            alert = (dev[f1] >= t1) & (dev[f2] >= t2)
            rate = alert.mean()
            if not (0.18 <= rate <= 0.32):
                continue
            prec = pos_dev[alert.to_numpy()].mean()
            if best is None or prec > best[0]:
                best = (prec, float(t1), float(t2), float(rate))
    _, t1, t2, rate_dev = best
    rule_rows = []
    for tag, df in [("p22", dev), ("p23b", test)]:
        pos = (df.teacher_refresh_benefit > 0).to_numpy()
        alert = ((df[f1] >= t1) & (df[f2] >= t2)).to_numpy()
        rule_rows.append(
            {
                "rule": f"conj({f1}>={t1:.6g} AND {f2}>={t2:.6g})",
                "dataset": tag,
                "alert_rate": float(alert.mean()),
                "precision": float(pos[alert].mean()) if alert.any() else float("nan"),
            }
        )
        # product rule on ranks (dataset-internal ranking)
        r1 = df[f1].rank(pct=True).to_numpy()
        r2 = df[f2].rank(pct=True).to_numpy()
        prod = r1 * r2
        rule_rows.append(
            {
                "rule": f"product(rank {f1} x rank {f2})",
                "dataset": tag,
                "alert_rate": np.nan,
                "precision": np.nan,
                "spearman": float(stats.spearmanr(prod, df.teacher_refresh_benefit).statistic),
                "rank_auc": rank_auc(prod, pos),
                "precision_at_0.25": precision_at_rate(prod, pos, ALERT_RATE),
            }
        )
    rules = pd.DataFrame(rule_rows)
    rules.to_csv(OUT / "trigger_rules_frozen.csv", index=False)
    print(rules.to_string(index=False))

    # quick diagnostic: what is stale_teacher_risk relative to stale replay KL trajectory?
    sr = pd.read_parquet(P23B / "stale_replay_rows.parquet")
    sr = sr[sr.proxy == PROXY]
    diag = []
    for r in test.itertuples():
        grp = sr[
            (sr.sample_id == r.sample_id)
            & (sr.previous_anchor == r.previous_anchor)
            & (sr.anchor == r.anchor)
            & (sr.horizon_offset <= r.horizon)
        ]
        diag.append(
            {
                "stale_teacher": r.stale_teacher_risk,
                "kl_at_horizon": float(grp.exact_kl.iloc[-1]),
                "kl_mean": float(grp.exact_kl.mean()),
                "kl_max": float(grp.exact_kl.max()),
            }
        )
    diag = pd.DataFrame(diag)
    for c in ["kl_at_horizon", "kl_mean", "kl_max"]:
        print(f"stale_teacher vs {c}: max abs diff {(diag.stale_teacher - diag[c]).abs().max():.6f}")


if __name__ == "__main__":
    main()
