"""Offline action-conditioned trigger pre-screen on P3 decision-validity event rows.

Strictly out-of-sample: univariate screen + rule fitting on calibration only;
frozen evaluation on evaluation/ and replication/ splits.
Outputs written to analysis/tables/.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

ROOT = Path("experiments/p3_decision_validity/results")
OUT = Path("analysis/tables")
OUT.mkdir(parents=True, exist_ok=True)

STAGES = ["calibration", "evaluation", "replication"]

# Concept classification of the zero-cost observable feature columns.
CONCEPT = {
    # (1) decision-boundary instability
    "retained_overlap": "boundary",
    "core_turnover": "boundary",
    "selector_score_drift": "boundary",
    "selected_core_score_margin": "boundary",
    "action_only_margin": "boundary",
    "cheap_rank_disagreement": "boundary",
    "top_reused_one_midpoint_shift": "boundary",
    "recent_window_exits": "boundary",
    "compressed_residual_norm_drift": "boundary",
    # (2) retained-set coverage drift
    "retained_attention_mass": "coverage",
    "core_attention_mass": "coverage",
    "recent_attention_mass": "coverage",
    "sink_attention_mass": "coverage",
    "key_query_alignment_mean": "coverage",
    "key_query_alignment_std": "coverage",
    "cache_occupancy": "coverage",
    # (3) generic state scalars (baselines)
    "attention_entropy": "generic",
    "attention_concentration": "generic",
    "token_age_mean": "generic",
    "token_age_std": "generic",
    "query_norm_drift": "generic",
    "compressed_sketch_l2": "generic",
    "layer_attention_summary_drift": "generic",
    "action_norm_median": "generic",
    "action_norm_spread": "generic",
    "action_to_compressed_state_ratio": "generic",
    "horizon": "structural",
}

LABEL_PRIMARY = "benefit_pos"  # refresh_benefit > 0  (== harmful_stale in this data)
LABEL_EPS = "harmful_stale_eps_0p1"  # secondary, epsilon used by the P3 frozen detector


def rank_auc(x, y):
    """Mann-Whitney rank AUC of score x for binary label y (tie-corrected)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan
    r = rankdata(x)
    return float((r[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def prf(alert, y):
    alert = np.asarray(alert, dtype=bool)
    y = np.asarray(y, dtype=bool)
    tp = int((alert & y).sum())
    fp = int((alert & ~y).sum())
    fn = int((~alert & y).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def load(stage):
    df = pd.read_parquet(ROOT / stage / "event_rows.parquet")
    df["benefit_pos"] = df["refresh_benefit"] > 0
    return df


data = {s: load(s) for s in STAGES}
cal = data["calibration"]

FEATURES = [f for f in CONCEPT if cal[f].nunique() > 1]  # drop constants

# ---------------------------------------------------------------- 1. univariate
rows = []
for f in FEATURES:
    x = cal[f].to_numpy(dtype=float)
    rho, p = spearmanr(x, cal["refresh_benefit"])
    auc_b = rank_auc(x, cal[LABEL_PRIMARY])
    auc_h = rank_auc(x, cal["harmful_stale"])
    auc_e = rank_auc(x, cal[LABEL_EPS])
    rows.append(
        dict(feature=f, concept=CONCEPT[f], spearman=rho, spearman_p=p,
             abs_spearman=abs(rho), auc_benefit_pos=auc_b, auc_harmful=auc_h,
             auc_harmful_eps_0p1=auc_e,
             cal_min=np.min(x), cal_median=np.median(x), cal_max=np.max(x))
    )
uni = pd.DataFrame(rows).sort_values("abs_spearman", ascending=False).reset_index(drop=True)
uni.to_csv(OUT / "p3_trigger_prescreen_univariate.csv", index=False)

# ------------------------------------------------------------- 2. rule fitting
# Top 8 by |Spearman|, guaranteeing >=1 per concept class.
top = list(uni["feature"].head(8))
for cls in ["boundary", "coverage", "generic"]:
    if not any(CONCEPT[f] == cls for f in top):
        top.append(uni.loc[uni["concept"] == cls, "feature"].iloc[0])

def fit_threshold(df, feat, label):
    """Best (direction, threshold) by F1 on df; tie-break precision, then recall."""
    x = df[feat].to_numpy(dtype=float)
    y = df[label].to_numpy(dtype=bool)
    vals = np.unique(x)
    cands = (vals[:-1] + vals[1:]) / 2.0
    best = None
    for t in cands:
        for d in ("gt", "le"):
            alert = x > t if d == "gt" else x <= t
            prec, rec, f1 = prf(alert, y)
            key = (f1, prec, rec)
            if best is None or key > best[0]:
                best = (key, d, float(t))
    (_, prec, rec), d, t = (best[0][1], best[0][2], best[0][0]), best[1], best[2]
    return {"feature": feat, "direction": d, "threshold": t,
            "cal_f1": best[0][0], "cal_precision": prec, "cal_recall": rec}

def apply_rule(df, rule):
    x = df[rule["feature"]].to_numpy(dtype=float)
    return x > rule["threshold"] if rule["direction"] == "gt" else x <= rule["threshold"]

single_rules = [fit_threshold(cal, f, LABEL_PRIMARY) for f in top]
for r in single_rules:
    r["kind"] = "single"
    r["name"] = f"{r['feature']}_{r['direction']}_{r['threshold']:.6g}"

# 2-feature AND conjunctions among the top 4 (AND of the individually fitted rules).
top4 = top[:4]
conj_rules = []
for i in range(len(top4)):
    for j in range(i + 1, len(top4)):
        r1 = fit_threshold(cal, top4[i], LABEL_PRIMARY)
        r2 = fit_threshold(cal, top4[j], LABEL_PRIMARY)
        n1 = f"{r1['feature']}_{r1['direction']}_{r1['threshold']:.6g}"
        n2 = f"{r2['feature']}_{r2['direction']}_{r2['threshold']:.6g}"
        alert = apply_rule(cal, r1) & apply_rule(cal, r2)
        prec, rec, f1 = prf(alert, cal[LABEL_PRIMARY])
        conj_rules.append({"kind": "and", "name": f"AND({n1},{n2})",
                           "parts": [r1, r2], "cal_f1": f1, "cal_precision": prec,
                           "cal_recall": rec})

# boundary x coverage product rule: orient each feature so that HIGH = risky
# (using the sign of its calibration Spearman), min-max scale to [0,1] on
# calibration, multiply, then fit a single threshold on the product.
b_best = uni.loc[uni["concept"] == "boundary"].iloc[0]
c_best = uni.loc[uni["concept"] == "coverage"].iloc[0]
prod_spec = {"kind": "product",
             "name": f"PROD({b_best['feature']},{c_best['feature']})",
             "f1": b_best["feature"], "f2": c_best["feature"],
             "s1": float(np.sign(b_best["spearman"]) or 1.0),
             "s2": float(np.sign(c_best["spearman"]) or 1.0)}

def product_score(df, spec):
    def scaled(f, s):
        x = df[f].to_numpy(dtype=float) * s
        lo, hi = np.min(x), np.max(x)
        return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)
    return scaled(spec["f1"], spec["s1"]) * scaled(spec["f2"], spec["s2"])

# fit product threshold on calibration
ps = product_score(cal, prod_spec)
y = cal[LABEL_PRIMARY].to_numpy(dtype=bool)
vals = np.unique(ps)
best = None
for t in (vals[:-1] + vals[1:]) / 2.0:
    prec, rec, f1 = prf(ps > t, y)
    if best is None or (f1, prec, rec) > best[0]:
        best = ((f1, prec, rec), float(t))
prod_spec["threshold"] = best[1]
prod_spec["cal_f1"], prod_spec["cal_precision"], prod_spec["cal_recall"] = best[0]

def apply_any(df, rule):
    if rule["kind"] == "single":
        return apply_rule(df, rule)
    if rule["kind"] == "and":
        return apply_rule(df, rule["parts"][0]) & apply_rule(df, rule["parts"][1])
    return product_score(df, rule) > rule["threshold"]

# reference: the existing P3 frozen detector (fit on this same calibration set upstream)
frozen_ref = {"kind": "single", "name": "REF_frozen_query_norm_drift",
              "feature": "query_norm_drift", "direction": "gt",
              "threshold": 0.0005792366292015205}

all_rules = single_rules + conj_rules + [prod_spec, frozen_ref]

# ---------------------------------------------------------- 3. frozen evaluation
eval_rows = []
for rule in all_rules:
    for stage in STAGES:
        df = data[stage]
        alert = apply_any(df, rule)
        ylab = df[LABEL_PRIMARY].to_numpy(dtype=bool)
        prec, rec, f1 = prf(alert, ylab)
        # feature-level rank AUC where meaningful
        if rule["kind"] == "single":
            auc = rank_auc(df[rule["feature"]], ylab)
        elif rule["kind"] == "product":
            auc = rank_auc(product_score(df, rule), ylab)
        else:
            auc = np.nan
        ben = df["refresh_benefit"].to_numpy(dtype=float)
        mb_a = float(ben[alert].mean()) if alert.any() else np.nan
        mb_n = float(ben[~alert].mean()) if (~alert).any() else np.nan
        eval_rows.append(dict(
            rule=rule["name"], kind=rule["kind"], split=stage,
            precision=prec, recall=rec, f1=f1, feature_auc=auc,
            alert_rate=float(alert.mean()), n_alert=int(alert.sum()),
            mean_benefit_alerted=mb_a, mean_benefit_not_alerted=mb_n,
            benefit_lift=(mb_a - mb_n) if np.isfinite(mb_a) and np.isfinite(mb_n) else np.nan,
        ))
ev = pd.DataFrame(eval_rows)
ev.to_csv(OUT / "p3_trigger_prescreen_rules.csv", index=False)

# frozen rule definitions (thresholds frozen on calibration)
frozen_defs = []
for rule in all_rules:
    d = {"name": rule["name"], "kind": rule["kind"], "primary_label": "refresh_benefit > 0"}
    if rule["kind"] == "single":
        d.update(feature=rule["feature"], direction=rule["direction"], threshold=rule["threshold"])
    elif rule["kind"] == "and":
        d["parts"] = [{"feature": p["feature"], "direction": p["direction"],
                       "threshold": p["threshold"]} for p in rule["parts"]]
    else:
        d.update(f1=rule["f1"], f2=rule["f2"], sign1=rule["s1"], sign2=rule["s2"],
                 scaling="minmax_on_calibration_after_sign_orientation",
                 threshold=rule["threshold"])
    frozen_defs.append(d)
(OUT / "p3_trigger_prescreen_frozen_rules.json").write_text(json.dumps(frozen_defs, indent=2))

# ------------------------------------------------------------------ console dump
pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 200)
print("=== UNIVARIATE (calibration, sorted by |Spearman|) ===")
print(uni[["feature", "concept", "spearman", "spearman_p", "auc_benefit_pos",
           "auc_harmful_eps_0p1"]].to_string(index=False,
           float_format=lambda v: f"{v:.4f}"))
print("\nfeatures fitted:", top)
print("\n=== RULE METRICS PER SPLIT (primary label = refresh_benefit > 0) ===")
show = ev[ev["rule"].isin([r["name"] for r in all_rules])]
print(show.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print("\n=== FROZEN RULE DEFINITIONS ===")
print(json.dumps(frozen_defs, indent=2))
