# P3 decision-validity: action-conditioned trigger pre-screen (offline)

Date: 2026-08-08. Analyst: subagent run. Data: `experiments/p3_decision_validity/results/{calibration,evaluation,replication}/event_rows.parquet` (84 rows each; 4 sequences × 7 target anchors × layers {0,14,26}; horizons 0..32).

## Schema summary

55 columns. Groups:

- Identifiers/meta: `sample_id, task, stage, history_id, tau_anchor, target_anchor, horizon, layer`.
- Decision labels (require exact/fresh/reused risk vectors — NOT zero-cost): `exact/fresh/reused_top_index, top1_stale, exact_optimum_stale, harmful_stale, reuse/fresh_normalized_regret, refresh_benefit, fresh_margin, reused_margin, harmful_stale_eps_{0p01,0p02,0p05,0p1}`, plus ranking-fidelity columns (`fresh_reused_spearman`, `*_pairwise_accuracy`, `pairwise_degradation`).
- Zero-cost observable features (24 usable + `cache_occupancy` constant, dropped): classified into
  - **boundary** (decision-boundary instability): `retained_overlap, core_turnover, selector_score_drift, selected_core_score_margin, action_only_margin, cheap_rank_disagreement, top_reused_one_midpoint_shift, recent_window_exits, compressed_residual_norm_drift`
  - **coverage** (retained-set coverage drift): `retained_attention_mass, core_attention_mass, recent_attention_mass, sink_attention_mass, key_query_alignment_mean/std, cache_occupancy`
  - **generic** state scalars: `attention_entropy, attention_concentration, token_age_mean/std, query_norm_drift, compressed_sketch_l2, layer_attention_summary_drift, action_norm_median/spread, action_to_compressed_state_ratio`
  - **structural**: `horizon` (known at zero cost).

## Label predeclaration

The manifest/ledger defines no trigger gate. `harmful_stale` was produced with epsilon = 0 (`p3_core.py::decision_event`, `harmful_stale = reuse_regret > epsilon`), so **`harmful_stale` ≡ `refresh_benefit > 0` in every split** (verified). Primary label = `refresh_benefit > 0`; secondary = `harmful_stale_eps_0p1` (the epsilon the upstream P3 frozen detector used). Positive rates (benefit>0): calibration 0.679, evaluation 0.571, replication 0.643.

## Headline univariate (calibration, full table in `p3_trigger_prescreen_univariate.csv`)

Top by |Spearman| vs `refresh_benefit`: `compressed_sketch_l2` (0.451, generic), `retained_overlap` (−0.415, boundary), `recent_window_exits` (0.415, boundary), `horizon` (0.415, structural), `top_reused_one_midpoint_shift` (0.358, boundary), `query_norm_drift` (0.290, generic).

**Critical structural finding:** at `horizon == 0` reused == fresh by construction, so `refresh_benefit == 0` for all 12 horizon-0 rows in every split. `retained_overlap`, `recent_window_exits`, `top_reused_one_midpoint_shift` are near-perfectly collinear with `horizon > 0` (identical Spearman 0.4146 and AUC 0.8041). Consequently every fitted threshold rule degenerates to "alert iff horizon > 0": alert rate ≈ 0.86, recall 1.0, precision ≈ base rate — on ALL THREE splits. This is trivially "out-of-sample stable" but carries no decision-boundary information.

## Restricted to horizon > 0 (72 events/split; the non-degenerate decision)

Feature rank-AUC (cal / eval / repl), label = benefit>0 — see `p3_trigger_prescreen_horizon_positive_auc.csv`:

- `compressed_sketch_l2`: 0.739 / 0.667 / 0.614 (decays but stays > 0.6)
- `sink_attention_mass`: 0.680 / 0.697 / 0.671 (most stable)
- `retained_attention_mass`: 0.642 / 0.686 / 0.598
- everything else ≤ 0.62 on calibration and/or collapses out-of-sample (`cheap_rank_disagreement` 0.488/0.337/0.492; `query_norm_drift` 0.336/0.355/0.416 — below chance)

Frozen rules on horizon>0 events (`p3_trigger_prescreen_horizon_positive_rules.csv`): all rules alert 93–100% of events; e.g. `compressed_sketch_l2 > 18.28` gives F1 0.887/0.789/0.826 but alerted-vs-not mean-benefit gap inverts on replication (0.218 alerted vs 0.495 not-alerted). No rule isolates a low-alert-rate high-benefit subgroup.

## Verdict

**Negative result for a selective action-conditioned trigger on this substrate.** Specifically:

1. The only robustly transferable signal is the structural bit `horizon > 0` (equivalently `retained_overlap < 1`, `recent_window_exits > 0`): once the anchor state is stale at all, refresh pays with probability ≈ 0.67–0.79 vs a 0.21–0.33 false-alarm cost. As an online gate this is "always recompute once stale" — a refresh-every-step policy, not a selective trigger. It may still be the correct economic answer if refresh cost is low relative to expected regret 0.24–0.39.
2. Among horizon>0 events, no observable feature — boundary, coverage, or generic — separates beneficial from non-beneficial refreshes out-of-sample at a usable operating point. The best candidates (`sink_attention_mass`, `compressed_sketch_l2`, `retained_attention_mass`; AUC 0.6–0.7) are generic/coverage scalars, NOT decision-boundary features, and are too weak to beat the degenerate horizon rule's recall.
3. Action-conditioned boundary features specifically (`selector_score_drift`, `cheap_rank_disagreement`, `selected_core_score_margin`, `action_only_margin`) fail: AUC ≤ 0.5 out-of-sample on the horizon>0 stratum.
4. Caveats: 84 events/split from only 4 sequences (2 gov_report + 2 synthetic NIAH); events within a sequence are correlated, and the substrate is a single-layer 8-candidate toy. Thresholds fitted on 84 rows are noise-dominated.

**Rules worth carrying to the online closed-loop gate** (frozen on calibration, in `p3_trigger_prescreen_frozen_rules.json`), with the caveat that #1 is the only defensible one:

1. `T1 (structural, recommended baseline arm)`: refresh iff `horizon > 0` — implementable as "anchor state changed since last selection." Recall 1.0 all splits; alert rate 0.857; mean benefit alerted 0.24–0.39 vs 0.00 not-alerted.
2. `T2 (weak selective variant)`: refresh iff `horizon > 0 AND sink_attention_mass` elevated — threshold NOT frozen (AUC-only signal, 0.68/0.70/0.67); only worth an exploratory arm, not a frozen gate.
3. Do NOT carry: `query_norm_drift` rules (below-chance AUC on horizon>0, all splits), any `cheap_rank_disagreement`/`selector_score_drift` rule, and any rule whose only content is retained_overlap (identical to T1 but noisier).

Bottom line: on this substrate the honest trigger policy is "recompute the ranking whenever the retained set has changed since selection"; a *selective* cheap trigger that skips some of those refreshes is not supported by the data. The online experiment should therefore compare T1 against always-refresh and never-refresh arms, and treat any selective arm as exploratory.

## Files

- `p3_trigger_prescreen_univariate.csv` — full 25-feature calibration univariate table (Spearman, p, AUC vs benefit>0 / harmful / harmful_eps_0p1).
- `p3_trigger_prescreen_rules.csv` — 17 rules × 3 splits: precision/recall/F1/feature-AUC/alert-rate/mean-benefit-alerted-vs-not.
- `p3_trigger_prescreen_horizon_positive_auc.csv`, `p3_trigger_prescreen_horizon_positive_rules.csv` — horizon>0-stratum analyses.
- `p3_trigger_prescreen_frozen_rules.json` — frozen rule definitions with thresholds.
- Script: `tmp/trigger_prescreen.py` (reproducible, read-only on experiment data).
