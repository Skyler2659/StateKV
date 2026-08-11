# Offline trigger-feature screen: action-conditioned refresh benefit prediction

**Question.** Can any feature computable from stored artifacts predict
`teacher_refresh_benefit = stale_teacher_risk − fresh_teacher_risk` better than the failed
`proxy_regret` baseline (P23b: Spearman −0.350, rank-AUC 0.249 — reproduced exactly below)?

**Protocol.** Fit/select on P22 dev (48 canonical events, `proxy == attention_mean_w1_shared`),
frozen-evaluate on P23b independent (48 events, single proxy). Metrics per feature: Spearman vs
benefit, rank-AUC vs `benefit > 0`, precision at alert-rate 0.25 (top-12 alerts).
Builder: `analysis/tables/build_trigger_feature_screen.py`. Per-event tables:
`trigger_features_p22.csv`, `trigger_features_p23b.csv`; metrics: `trigger_feature_metrics.csv`;
rules: `trigger_rules_frozen.csv`.

## Computability audit (what stored artifacts actually contain)

| feature | status | reason |
|---|---|---|
| churn_jaccard (1 − Jaccard(stale, fresh)) | **NOT computable offline** | `selection_inventory.parquet` stores only `selected_core_tokens` = core_budget (a scalar **count**, not the token array), plus `selection_hash` and overlap-vs-attention-baseline counts. Proxy **score vectors are not persisted** by either `statekv/proxy_alignment.py` or `statekv/direct_policy_replay.py`, so `repair_stale_core` (proxy_alignment.py:81) cannot be replayed and consecutive-core intersections exist nowhere. Hash checks: stale core == previous anchor's fresh core for 48/48 events in both datasets (repair preserved everything, zero fill); stale == fresh in 0/48, so `churn_binary` is constant and useless; panel-hash cross-matching (stale id in panel@anchor, fresh id in panel@prev-anchor) recovers 0/48 overlaps. |
| panel_margin / panel_spread (teacher & proxy) | computable | from `cross_action_rows.parquet` (7-candidate panel at same sample×anchor×horizon); verified fresh action = panel row of `attention_mean_w1_shared` (teacher/proxy risk match exactly, diff 0.0). |
| stale-trajectory drift | computable, with offset shift | `stale_replay_rows.parquet` is horizon-independent (16 offsets per transition, max distinct `exact_kl` across horizons = 1). Stored offsets are **1..16, not 0..3**, so "kl at offset 0" → `kl_offset1`; early slope = OLS slope of `exact_kl` over offsets 1..4; `fisher_mean_early` = mean over offsets 1..4. |
| proxy-risk shape (fresh, stale, ratio) | computable | direct columns of `refresh_regret_rows.parquet`. |
| baselines proxy_regret, horizon | computable | `proxy_regret` == `stale_proxy_risk − fresh_proxy_risk` exactly. |

Side finding: `stale_teacher_risk` equals the **mean** of `exact_kl` over offsets 1..horizon
(max abs diff 0.0 on P23b; not the endpoint KL, max diff 3.03).

## Full metric table (dev → frozen test)

| feature | rho P22 | AUC P22 | P@0.25 P22 | rho P23b | AUC P23b | P@0.25 P23b |
|---|---|---|---|---|---|---|
| panel_margin_teacher | −0.252 | 0.229 | 0.333 | +0.073 | 0.172 | 0.417 |
| panel_margin_proxy | +0.410 | 0.641 | 1.000 | +0.005 | 0.249 | 0.583 |
| panel_spread_teacher | +0.167 | 0.357 | 0.583 | +0.074 | 0.211 | 0.500 |
| panel_spread_proxy | −0.122 | 0.586 | 0.583 | −0.205 | 0.623 | 0.833 |
| kl_offset1 | +0.154 | 0.406 | 0.667 | −0.116 | 0.181 | 0.333 |
| **kl_slope_early** | **+0.042** | **0.414** | **0.417** | **+0.395** | **0.878** | **1.000** |
| kl_max_early | +0.247 | 0.422 | 0.417 | +0.208 | 0.367 | 0.583 |
| fisher_mean_early | +0.156 | 0.500 | 0.667 | +0.175 | 0.338 | 0.583 |
| delta_nll_mean_early | −0.037 | 0.359 | 0.417 | −0.107 | 0.367 | 0.583 |
| logit_l2_mean_early | +0.082 | 0.344 | 0.417 | +0.159 | 0.328 | 0.583 |
| fresh_proxy_risk | +0.086 | 0.391 | 0.417 | +0.181 | 0.387 | 0.750 |
| stale_proxy_risk | +0.246 | 0.516 | 0.750 | −0.062 | 0.240 | 0.500 |
| proxy_risk_ratio | +0.172 | 0.578 | 0.917 | −0.375 | 0.436 | 0.833 |
| **proxy_regret (failed baseline)** | +0.378 | 0.664 | 0.750 | **−0.350** | **0.249** | 0.500 |
| horizon (fixed interval) | −0.015 | 0.477 | 0.667 | +0.067 | 0.367 | 0.667 |
| churn_binary | constant 1 (dropped) | | | constant 1 (dropped) | | |

## Top-2 rules (selected on P22, thresholds frozen, applied to P23b)

P22 selection (by dev AUC) picked `panel_margin_proxy` + `panel_spread_proxy`.

- Conjunction `panel_margin_proxy ≥ 0.0189026 AND panel_spread_proxy ≥ 0.554093`:
  P22 alert-rate 0.25, precision 0.917 → **P23b alert-rate 0.00** (thresholds do not transfer;
  proxy-risk scales differ across substrates), precision undefined.
- Rank-product rule: P22 rho +0.352 / AUC 0.695 / P@0.25 0.667 →
  **P23b rho −0.244 / AUC 0.318 / P@0.25 0.583**. Fails.

## The one test-set signal: kl_slope_early

P23b: Spearman +0.395 (p = 0.0055), AUC 0.878, precision@0.25 = 1.000; consistent across tasks
(gov_report +0.48, niah +0.53), anchors (+0.63 / +0.24), horizons ≥ 4 (+0.51..+0.83; −0.35 at
horizon 1 where the benefit target is offset-1 KL itself). Zero correlation with horizon — not a
fixed-interval confound. Mechanistically sensible: early KL growth of the stale continuation
predicts eventual horizon benefit. **But** on P22 dev it is null (rho +0.042, p = 0.78; split
+0.54 at anchor 48 vs −0.30 at anchor 32), so no dev-frozen procedure would ever select it, and
p = 0.0055 does not survive Bonferroni over the 13 screened features (α = 0.0038). It is also a
**teacher-side statistic** (exact_kl needs full-context logits), so even if real it is not a
cheap online trigger as-is — it motivates a cheap probe approximation, not a deployment.

## Verdict

**No dev-selectable offline feature or pair beats proxy_regret out-of-sample.** The dev-selected
top-2 rules collapse on P23b (conjunction never fires; product AUC 0.318, rho −0.244). Every
feature that looked good on P22 (`panel_margin_proxy`, `stale_proxy_risk`, `proxy_risk_ratio`)
reverts to ≤ 0.25 AUC or negative rho on P23b — the same non-transfer failure mode as
proxy_regret itself. Numerically one feature does clear the stated bar on P23b —
`kl_slope_early` (AUC 0.878 ≥ 0.6, rho +0.395 > 0) — but it was invisible on dev and is
post-hoc among 13 candidates, so it is a lead, not a validated trigger.

**Actionable conclusions:**
1. Offline artifact mining cannot deliver the action-conditioned trigger: the primary candidate
   (churn_jaccard) is uncomputable because cores are stored as counts+hashes and proxy score
   vectors are never persisted. If churn is wanted, the online runner must log selected core
   token ids (or score vectors) at each anchor.
2. The only credible signal direction is stale-trajectory early drift (`kl_slope_early`).
   Instrument the online runner to log it (and a cheap probe surrogate, e.g. first-offset
   divergence under a reference-free statistic) across many more refresh events, then re-test
   dev→independent transfer with a frozen selection rule.
3. Until then, horizon (fixed interval) is as good as any offline trigger — which is to say,
   none of them work.
