"""Fit and honestly validate a cheap state-aware KV-cache refresh trigger.

Reads a refresh-event table (one row per sample x policy x step) produced by the
selective-refresh label-collection runs, engineers tail-event labels from the
teacher-side refresh-benefit columns, screens cheap online-computable features,
and fits threshold rules with sample-level leave-one-sample-out (LOSO) CV.

Predeclared freeze gate (hard-coded, not tuned post hoc):
  the selected rule must reach LOSO-held-out rank-AUC >= 0.65 AND show
  alerted-mean-benefit > non-alerted-mean-benefit in >= 80% of folds
  (>= 8/10 on the full 10-sample validation run), else NO_FREEZE.

Usage:
  .venv/bin/python analysis/tables/fit_refresh_trigger.py \
      --events results/temporal_cache_discovery/statekv_selective_refresh_labels_r2a_v1/refresh_event_rows.parquet

Outputs (under analysis/tables/):
  refresh_trigger_label_stats.csv        benefit distribution + label rates
  refresh_trigger_feature_screen.csv     per-feature x label x policy metrics
  refresh_trigger_loso_rules.csv         LOSO fold-level + aggregate rule metrics
  refresh_trigger_fixed_interval.csv     fixed-interval baseline (alert every k steps)
  refresh_trigger_fit_summary.md         human-readable summary + verdict
  refresh_trigger_frozen_rule.json       frozen rule (only if the gate passes)
  refresh_trigger_no_freeze.json         diagnostics (only if the gate fails)
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVENTS = (
    ROOT
    / "results/temporal_cache_discovery/statekv_selective_refresh_labels_r2a_v1/refresh_event_rows.parquet"
)
OUT_DIR = ROOT / "analysis/tables"

# ---- predeclared analysis constants (do not tune after seeing results) ----
TAU = 0.05  # primary tail-event threshold on refresh_benefit_lag4
TAU_CANDIDATES = [0.01, 0.05, 0.10, 0.25]
ALERT_RATES = [0.05, 0.10, 0.25]
FREEZE_MIN_AUC = 0.65
FREEZE_MIN_LIFT_FOLD_FRAC = 0.8
MIN_SAMPLES_FOR_FREEZE = 10  # smoke runs are plumbing-only, never freeze
MAX_SCREEN_FEATURES_FOR_RULES = 8  # cap pair search combinatorics
THRESHOLD_QUANTILES = np.linspace(0.50, 0.99, 20)
MAX_ALERT_RATE_RULE = 0.5
INTERVAL_BASELINE_KS = [2, 3, 4, 6, 8, 12]

# Cheap features: computable online without the teacher (full-model) pass.
# Teacher-side columns (exact_kl, js, *_nll, fisher_quadratic, logit_l2_sq,
# stale_exact_kl_*, refresh_benefit_*, full_*/compressed-argmax/probability,
# argmax_diverged) are reference labels only and are never used as features.
BASE_CHEAP_FEATURES = [
    "churn_jaccard_mean",
    "churn_jaccard_min",
    "churn_jaccard_max",
    "score_tv_mean",
    "boundary_margin_mean",
    "coverage_mass_mean",
    "stale_action_l1_lag4",
    "stale_action_l1_lag16",
    "compressed_margin",
    "compressed_entropy",
]
DERIVED_FEATURES = {
    # churn gated by how close the boundary decision is (low margin = risky)
    "churn_x_1minus_margin": lambda df: df["churn_jaccard_mean"] * (1.0 - df["boundary_margin_mean"]),
    # score drift interacting with selection churn
    "tv_x_churn": lambda df: df["score_tv_mean"] * df["churn_jaccard_mean"],
}
LABEL_COLS = {"tail_event": None, "any_benefit": None}  # filled after engineering


# ---------------------------------------------------------------------------
# small metric helpers
# ---------------------------------------------------------------------------
def rank_auc(score: np.ndarray, positive: np.ndarray) -> float:
    """P(s_pos > s_neg) + 0.5 * P(tie); NaN if a class is missing."""
    pos, neg = score[positive], score[~positive]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = np.mean(pos[:, None] > neg[None, :])
    eq = np.mean(pos[:, None] == neg[None, :])
    return float(gt + 0.5 * eq)


def precision_at_rate(score: np.ndarray, positive: np.ndarray, rate: float) -> float:
    k = max(1, int(round(len(score) * rate)))
    order = np.argsort(-score, kind="stable")
    return float(np.mean(positive[order[:k]]))


def alerted_benefit_split(score: np.ndarray, benefit: np.ndarray, rate: float) -> tuple[float, float, float]:
    """Mean benefit in top-`rate` alerted vs the rest."""
    k = max(1, int(round(len(score) * rate)))
    order = np.argsort(-score, kind="stable")
    alerted, rest = order[:k], order[k:]
    if len(rest) == 0:
        return float("nan"), float("nan"), float("nan")
    return (
        float(np.mean(benefit[alerted])),
        float(np.mean(benefit[rest])),
        float(np.mean(benefit[alerted]) - np.mean(benefit[rest])),
    )


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return float("nan")
    return float(stats.spearmanr(x, y).statistic)


# ---------------------------------------------------------------------------
# 1. label engineering
# ---------------------------------------------------------------------------
def engineer_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    df = df.copy()
    b4 = df["refresh_benefit_lag4"].to_numpy(dtype=float)
    df["any_benefit"] = b4 > 0.0
    df["tail_event"] = b4 >= TAU

    lag_relation = {}
    lag_cols = sorted(
        (c for c in df.columns if c.startswith("refresh_benefit_lag")),
        key=lambda c: int(c.rsplit("lag", 1)[1]),
    )
    lag_relation["lag_columns"] = ",".join(lag_cols) if lag_cols else "MISSING"
    for stem in ("refresh_benefit_lag", "stale_exact_kl_lag", "stale_action_l1_lag"):
        for lag in ("4", "16", "32"):
            col = f"{stem}{lag}"
            if col not in df.columns:
                lag_relation[col] = "MISSING"
    if {"refresh_benefit_lag4", "refresh_benefit_lag16"} <= set(df.columns):
        diff = np.abs(df["refresh_benefit_lag4"] - df["refresh_benefit_lag16"]).max()
        lag_relation["lag4_vs_lag16_max_abs_diff"] = float(diff)
        lag_relation["lag4_lag16_identical"] = bool(diff < 1e-12)

    rows = [
        {"stat": "mean", "value": float(np.mean(b4))},
        {"stat": "median", "value": float(np.median(b4))},
        {"stat": "p90", "value": float(np.quantile(b4, 0.9))},
        {"stat": "p99", "value": float(np.quantile(b4, 0.99))},
        {"stat": "max", "value": float(np.max(b4))},
        {"stat": "min", "value": float(np.min(b4))},
        {"stat": "frac_gt_0", "value": float(np.mean(b4 > 0))},
    ]
    for tau in TAU_CANDIDATES:
        rows.append({"stat": f"frac_ge_{tau}", "value": float(np.mean(b4 >= tau))})
    rows.append({"stat": "label_rate_any_benefit", "value": float(df["any_benefit"].mean())})
    rows.append({"stat": "label_rate_tail_event", "value": float(df["tail_event"].mean())})
    for col in lag_cols:
        if col == "refresh_benefit_lag4":
            continue
        values = df[col].to_numpy(dtype=float)
        lag = col.rsplit("lag", 1)[1]
        rows.append({"stat": f"lag{lag}_mean", "value": float(np.mean(values))})
        rows.append({"stat": f"lag{lag}_median", "value": float(np.median(values))})
        rows.append({"stat": f"lag{lag}_p90", "value": float(np.quantile(values, 0.9))})
        rows.append({"stat": f"lag{lag}_frac_gt_0", "value": float(np.mean(values > 0))})
    return df, pd.DataFrame(rows), lag_relation


# ---------------------------------------------------------------------------
# 2. feature screening
# ---------------------------------------------------------------------------
def build_feature_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    feats = pd.DataFrame(index=df.index)
    names = []
    for col in BASE_CHEAP_FEATURES:
        if col in df.columns:
            feats[col] = pd.to_numeric(df[col], errors="coerce")
            names.append(col)
    for name, fn in DERIVED_FEATURES.items():
        deps_ok = True
        probe = None
        try:
            probe = fn(df)
        except KeyError:
            deps_ok = False
        if deps_ok and probe is not None:
            feats[name] = pd.to_numeric(probe, errors="coerce")
            names.append(name)
    return feats, names


def screen_features(df: pd.DataFrame, feats: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    benefit = df["refresh_benefit_lag4"].to_numpy(dtype=float)
    rows = []
    policies = ["ALL"] + sorted(df["policy"].unique().tolist())
    for policy in policies:
        mask = np.ones(len(df), dtype=bool) if policy == "ALL" else (df["policy"] == policy).to_numpy()
        for feat in names:
            x = feats[feat].to_numpy(dtype=float)
            valid = mask & np.isfinite(x)
            xv, bv = x[valid], benefit[valid]
            for label in ("tail_event", "any_benefit"):
                yv = df[label].to_numpy()[valid]
                row = {
                    "policy": policy,
                    "feature": feat,
                    "label": label,
                    "n": int(valid.sum()),
                    "n_positive": int(yv.sum()),
                    "spearman": spearman(xv, bv),
                    "rank_auc": rank_auc(xv, yv),
                }
                for rate in ALERT_RATES:
                    row[f"precision_at_{rate}"] = precision_at_rate(xv, yv, rate)
                    ma, mn, lift = alerted_benefit_split(xv, bv, rate)
                    row[f"benefit_alerted_at_{rate}"] = ma
                    row[f"benefit_nonalerted_at_{rate}"] = mn
                    row[f"benefit_lift_at_{rate}"] = lift
                rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. rule fitting with LOSO CV
# ---------------------------------------------------------------------------
def within_sample_rank(values: pd.Series, samples: pd.Series) -> pd.Series:
    return values.groupby(samples).rank(pct=True)


def candidate_rules(df: pd.DataFrame, feats: pd.DataFrame, screen: pd.DataFrame) -> dict[str, dict]:
    """Build candidate rule forms. Scores are NaN-safe (NaN -> -inf, never alerts)."""
    allp = screen[(screen.policy == "ALL") & (screen.label == "tail_event")].copy()
    allp["abs_spearman"] = allp["spearman"].abs()
    top = (
        allp.sort_values("abs_spearman", ascending=False, na_position="last")
        .head(MAX_SCREEN_FEATURES_FOR_RULES)["feature"]
        .tolist()
    )
    if not top:  # fallback if screen degenerate
        top = list(feats.columns[: min(4, feats.shape[1])])

    neg_inf = feats[top[0]].to_numpy(dtype=float) * 0.0  # for NaN handling
    del neg_inf

    def col(name: str) -> np.ndarray:
        v = feats[name].to_numpy(dtype=float)
        return np.where(np.isfinite(v), v, -np.inf)

    rules: dict[str, dict] = {}
    for f in top:
        rules[f"single:{f}"] = {"kind": "single", "features": [f], "score": col(f)}

    for a, b in combinations(top, 2):
        ra = within_sample_rank(feats[a], df["sample_id"]).fillna(0.0).to_numpy()
        rb = within_sample_rank(feats[b], df["sample_id"]).fillna(0.0).to_numpy()
        # AND rule: score = min of within-sample ranks (threshold on min == AND of thresholds)
        rules[f"and:{a}&{b}"] = {
            "kind": "and",
            "features": [a, b],
            "score": np.minimum(ra, rb),
        }
        rules[f"rankprod:{a}*{b}"] = {
            "kind": "rankprod",
            "features": [a, b],
            "score": ra * rb,
        }
    return rules


def fit_threshold(score: np.ndarray, benefit: np.ndarray) -> tuple[float, float]:
    """Pick threshold maximizing alerted-minus-nonalerted mean benefit.

    Constraint: alert rate in (0, MAX_ALERT_RATE_RULE]. Returns (threshold, lift).
    """
    finite = np.isfinite(score)
    if finite.sum() < 10 or np.unique(score[finite]).size < 2:
        return float("nan"), float("nan")
    best_t, best_lift = float("nan"), -np.inf
    for q in THRESHOLD_QUANTILES:
        t = float(np.quantile(score[finite], q))
        alerted = score >= t
        rate = alerted.mean()
        if rate <= 0 or rate > MAX_ALERT_RATE_RULE:
            continue
        if (~alerted).sum() == 0:
            continue
        lift = float(np.mean(benefit[alerted]) - np.mean(benefit[~alerted]))
        if lift > best_lift:
            best_lift, best_t = lift, t
    return best_t, best_lift


def loso_validate(df: pd.DataFrame, rules: dict[str, dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    benefit = df["refresh_benefit_lag4"].to_numpy(dtype=float)
    y = df["tail_event"].to_numpy()
    samples = df["sample_id"].to_numpy()
    uniq_samples = np.unique(samples)

    fold_rows = []
    for rule_name, rule in rules.items():
        score = rule["score"]
        for held in uniq_samples:
            te = samples == held
            tr = ~te
            thr, _ = fit_threshold(score[tr], benefit[tr])
            if not np.isfinite(thr):
                fold_rows.append(
                    {"rule": rule_name, "held_out_sample": held, "threshold": np.nan,
                     "alert_rate": np.nan, "auc": np.nan, "precision": np.nan,
                     "recall": np.nan, "benefit_alerted": np.nan,
                     "benefit_nonalerted": np.nan, "lift_positive": False}
                )
                continue
            alert_te = score[te] >= thr
            b_te, y_te = benefit[te], y[te]
            n_alert = int(alert_te.sum())
            ba = float(np.mean(b_te[alert_te])) if n_alert else float("nan")
            bn = float(np.mean(b_te[~alert_te])) if (~alert_te).any() else float("nan")
            fold_rows.append(
                {
                    "rule": rule_name,
                    "held_out_sample": held,
                    "threshold": thr,
                    "alert_rate": float(n_alert / te.sum()),
                    "auc": rank_auc(score[te], y_te),
                    "precision": float(np.mean(y_te[alert_te])) if n_alert else float("nan"),
                    "recall": float(y_te[alert_te].sum() / y_te.sum()) if y_te.sum() else float("nan"),
                    "benefit_alerted": ba,
                    "benefit_nonalerted": bn,
                    "lift_positive": bool(np.isfinite(ba) and np.isfinite(bn) and ba > bn),
                }
            )
    folds = pd.DataFrame(fold_rows)

    agg = (
        folds.groupby("rule")
        .agg(
            n_folds=("held_out_sample", "count"),
            threshold_median=("threshold", "median"),
            threshold_min=("threshold", "min"),
            threshold_max=("threshold", "max"),
            auc_mean=("auc", "mean"),
            auc_min=("auc", "min"),
            precision_mean=("precision", "mean"),
            recall_mean=("recall", "mean"),
            alert_rate_mean=("alert_rate", "mean"),
            benefit_alerted_mean=("benefit_alerted", "mean"),
            benefit_nonalerted_mean=("benefit_nonalerted", "mean"),
            lift_positive_folds=("lift_positive", "sum"),
        )
        .reset_index()
    )
    agg["lift_fold_frac"] = agg["lift_positive_folds"] / agg["n_folds"]
    return folds, agg


def fixed_interval_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """Mean benefit when alerting every k-th step (step % k == 0), plus always/never."""
    rows = []
    for policy in ["ALL"] + sorted(df["policy"].unique().tolist()):
        sub = df if policy == "ALL" else df[df["policy"] == policy]
        b = sub["refresh_benefit_lag4"].to_numpy(dtype=float)
        steps = sub["step"].to_numpy(dtype=int)
        rows.append({"policy": policy, "k": 1, "alert_frac": 1.0,
                     "mean_benefit_alerted": float(np.mean(b)),
                     "mean_benefit_nonalerted": float("nan")})
        for k in INTERVAL_BASELINE_KS:
            alert = steps % k == 0
            if alert.sum() == 0 or (~alert).sum() == 0:
                continue
            rows.append({
                "policy": policy,
                "k": k,
                "alert_frac": float(alert.mean()),
                "mean_benefit_alerted": float(np.mean(b[alert])),
                "mean_benefit_nonalerted": float(np.mean(b[~alert])),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. report + freeze decision
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--events", type=Path, default=DEFAULT_EVENTS, help="refresh_event_rows.parquet path")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    df = pd.read_parquet(args.events)
    run_name = args.events.parent.name
    n_samples = df["sample_id"].nunique()
    underpowered = n_samples < MIN_SAMPLES_FOR_FREEZE

    df, label_stats, lag_relation = engineer_labels(df)
    feats, feat_names = build_feature_frame(df)
    screen = screen_features(df, feats, feat_names)
    rules = candidate_rules(df, feats, screen)
    folds, agg = loso_validate(df, rules)
    fixed = fixed_interval_baseline(df)

    # ---- freeze gate (predeclared) ----
    min_lift_folds = int(np.ceil(FREEZE_MIN_LIFT_FOLD_FRAC * n_samples))
    agg = agg.sort_values("auc_mean", ascending=False, na_position="last").reset_index(drop=True)
    selected = agg.iloc[0] if len(agg) else None
    gate_pass = (
        selected is not None
        and not underpowered
        and np.isfinite(selected["auc_mean"])
        and selected["auc_mean"] >= FREEZE_MIN_AUC
        and selected["lift_positive_folds"] >= min_lift_folds
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    label_stats.to_csv(args.out_dir / "refresh_trigger_label_stats.csv", index=False)
    screen.to_csv(args.out_dir / "refresh_trigger_feature_screen.csv", index=False)
    folds.to_csv(args.out_dir / "refresh_trigger_loso_folds.csv", index=False)
    agg.to_csv(args.out_dir / "refresh_trigger_loso_rules.csv", index=False)
    fixed.to_csv(args.out_dir / "refresh_trigger_fixed_interval.csv", index=False)

    verdict = "FREEZE" if gate_pass else "NO_FREEZE"
    if underpowered:
        verdict = "NO_FREEZE (UNDERPOWERED: plumbing run, freeze gate disabled)"

    if gate_pass:
        best_rule = rules[selected["rule"]]
        payload = {
            "rule_name": selected["rule"],
            "kind": best_rule["kind"],
            "features": best_rule["features"],
            "threshold": float(selected["threshold_median"]),
            "threshold_semantics": (
                "alert when score >= threshold; score is the raw feature for single rules, "
                "min of within-sample percentile ranks for AND rules, product of within-sample "
                "percentile ranks for rankprod rules; NaN features never alert"
            ),
            "tau": TAU,
            "label": "tail_event = refresh_benefit_lag4 >= tau",
            "loso": {
                "auc_mean": float(selected["auc_mean"]),
                "auc_min": float(selected["auc_min"]),
                "precision_mean": float(selected["precision_mean"]),
                "recall_mean": float(selected["recall_mean"]),
                "alert_rate_mean": float(selected["alert_rate_mean"]),
                "lift_positive_folds": int(selected["lift_positive_folds"]),
                "n_folds": int(selected["n_folds"]),
            },
            "source_run": run_name,
            "gate": {"min_auc": FREEZE_MIN_AUC, "min_lift_folds": min_lift_folds},
        }
        (args.out_dir / "refresh_trigger_frozen_rule.json").write_text(json.dumps(payload, indent=2))
    else:
        top_rules = agg.head(5)
        payload = {
            "verdict": "NO_FREEZE",
            "underpowered": bool(underpowered),
            "reason": (
                "underpowered run (n_samples < %d); freeze gate disabled" % MIN_SAMPLES_FOR_FREEZE
                if underpowered
                else "best rule failed the predeclared gate "
                f"(auc_mean >= {FREEZE_MIN_AUC} and lift-positive in >= {min_lift_folds}/{n_samples} folds)"
            ),
            "n_rows": int(len(df)),
            "n_samples": int(n_samples),
            "tau": TAU,
            "label_rate_tail_event": float(df["tail_event"].mean()),
            "label_rate_any_benefit": float(df["any_benefit"].mean()),
            "best_rules_by_loso_auc": json.loads(top_rules.to_json(orient="records")) if len(top_rules) else [],
            "lag_relation": lag_relation,
            "source_run": run_name,
        }
        (args.out_dir / "refresh_trigger_no_freeze.json").write_text(json.dumps(payload, indent=2))

    # ---- markdown summary ----
    lines = [
        "# Refresh trigger fit summary",
        "",
        f"- source events: `{args.events.relative_to(ROOT) if args.events.is_relative_to(ROOT) else args.events}` (run `{run_name}`)",
        f"- rows: {len(df)} | samples: {n_samples} | policies: {', '.join(sorted(df['policy'].unique()))}",
        f"- steps per sample-policy: {df.groupby(['sample_id','policy']).size().median():.0f} (median)",
        f"- **VERDICT: {verdict}**",
        "",
    ]
    if underpowered:
        lines += [
            "> UNDERPOWERED RUN: fewer than %d samples — all numbers below are plumbing" % MIN_SAMPLES_FOR_FREEZE,
            "> checks only, not evidence. Re-run on the full validation run before interpreting.",
            "",
        ]
    lines += [
        "## 1. Label engineering",
        "",
        f"- primary label `tail_event` = refresh_benefit_lag4 >= {TAU} (rate {df['tail_event'].mean():.4f})",
        f"- secondary label `any_benefit` = refresh_benefit_lag4 > 0 (rate {df['any_benefit'].mean():.4f})",
        "",
        label_stats.to_markdown(index=False),
        "",
        "lag4/lag16 relation:",
    ]
    for k, v in lag_relation.items():
        lines.append(f"- {k}: {v}")
    if lag_relation.get("lag4_lag16_identical"):
        lines.append(
            "- NOTE: lag4 and lag16 benefit columns are identical here (12-step horizon vs 32-token "
            "recent window). Trigger fitting uses lag4 only; lag16 adds no information in this run."
        )
    lines += ["", "## 2. Feature screening (policy=ALL, label=tail_event, top by |Spearman|)", ""]
    top_screen = (
        screen[(screen.policy == "ALL") & (screen.label == "tail_event")]
        .assign(abs_s=lambda d: d["spearman"].abs())
        .sort_values("abs_s", ascending=False)
        .head(10)
        [["feature", "spearman", "rank_auc", "precision_at_0.1", "benefit_lift_at_0.1"]]
    )
    lines += [top_screen.to_markdown(index=False), "", "Full per-policy tables in `refresh_trigger_feature_screen.csv`.", ""]
    lines += ["## 3. LOSO rule fitting (top 8 rules by held-out AUC)", ""]
    lines += [
        agg.head(8)[
            ["rule", "auc_mean", "auc_min", "precision_mean", "recall_mean",
             "alert_rate_mean", "lift_positive_folds", "n_folds",
             "threshold_median", "threshold_min", "threshold_max"]
        ].to_markdown(index=False),
        "",
        f"Freeze gate: LOSO AUC >= {FREEZE_MIN_AUC} AND benefit lift positive in >= {min_lift_folds}/{n_samples} folds.",
        "",
        "## 3b. Fixed-interval baseline (mean benefit when alerting every k-th step)",
        "",
        fixed[fixed.policy == "ALL"].to_markdown(index=False),
        "",
        "Any frozen trigger must beat these alerted-mean-benefit numbers at comparable alert rates.",
        "",
    ]
    (args.out_dir / "refresh_trigger_fit_summary.md").write_text("\n".join(lines))

    # ---- console summary ----
    print(f"rows={len(df)} samples={n_samples} policies={sorted(df['policy'].unique())}")
    print(f"benefit lag4: mean={label_stats.loc[label_stats.stat=='mean','value'].iloc[0]:.6g} "
          f"median={label_stats.loc[label_stats.stat=='median','value'].iloc[0]:.6g} "
          f"p99={label_stats.loc[label_stats.stat=='p99','value'].iloc[0]:.6g} "
          f"frac>0={df['any_benefit'].mean():.3f} frac>={TAU}={df['tail_event'].mean():.3f}")
    print(f"lag4==lag16: {lag_relation.get('lag4_lag16_identical')}")
    if selected is not None:
        print(f"best LOSO rule: {selected['rule']} auc_mean={selected['auc_mean']:.3f} "
              f"lift_folds={int(selected['lift_positive_folds'])}/{int(selected['n_folds'])}")
    fi = fixed[fixed.policy == "ALL"]
    print("fixed-interval alerted mean benefit: " +
          ", ".join(f"k={int(r.k)}:{r.mean_benefit_alerted:.3g}" for r in fi.itertuples()))
    print(f"VERDICT: {verdict}")
    print(f"outputs written to {args.out_dir}")


if __name__ == "__main__":
    main()
