# StateKV gate-rejection retrospective catalog

Status: final archive
Date: 2026-08-10
Purpose: archive **every strategy/policy rejected by a preregistered gate or marked
not-supported / refuted / gate-failed** in the StateKV project, preserving each
phase's original numbers, and classify each rejection as **decisive** or
**marginal / veto-style** (a veto metric slightly worse while other metrics
improved). This document anchors the re-test program: entries in Part 3 are the
re-test targets; entries in Part 2 should not be re-run without new evidence.

Method: every number below was re-verified against the raw artifacts on
2026-08-10 — run-level `results/temporal_cache_discovery/*/summary.json` and
`metrics.csv` / `comparison_table.csv` / `p1_analysis.json`, the derived tables
in `analysis/tables/` (human-readable summaries now under
`docs/evidence/tables/`), and the closure documents in `docs/evidence/`. The claims
registry `configs/ccfa.yaml`, the frozen registry
`experiments/frozen_registry.yaml`, and the README evidence table were used as
cross-checks and as the completeness oracle. Where a number could not be tied
to a raw artifact it is marked `[unverified]` with the reason.

Two substrate eras (non-transferable; see Part 4):

- **Era 1**: Qwen2.5-1.5B-Instruct-4bit, shared-mask selection, budget 128 /
  core 92 (sink 4, recent 32), teacher-forced physical replay or matched-budget
  free generation. Phases P0–P30 and the substrate-B teacher gate.
- **Era 2**: Qwen3-8B-4bit, per-layer selection, budget 256 / core 220
  (sink 4, recent 32), strict pure-eviction or recoverable (KVBackingStore)
  semantics, same-input exact-KL evaluation. Phases P31–P35, Gate 0/1, ladder
  2B/2C, R0, the QK–V discovery battery, the qk_tiered_v gate, R1/R2a/R2,
  refresh arms, and the open-search hypotheses HF1–HF4.
- **Era 0** (pre-discovery frozen phases, Qwen2.5-1.5B frozen protocol):
  recorded in `experiments/frozen_registry.yaml` as frozen-negative-evidence.

---

## Part 1 — master table of all rejected strategies

Substrate shorthand: E1 = Era 1, E2 = Era 2, E0 = Era 0. "indep" = evaluated on
held-out / new sequences; "dev" = development sequences only.

| # | phase / id | mechanism (what was tried) | veto metric + numbers | improved metrics + numbers | n / substrate | dev vs indep | evidence |
|---|---|---|---|---|---|---|---|
| E0-a | predictive_closure | original predictive-closure audit | preregistered P0-formal gate failed: native 4-bit / mismatched execution graphs do not support the closure audit | — (precision diagnostics retained) | E0 | frozen | experiments/frozen_registry.yaml; experiments/predictive_closure/ |
| E0-b | p1_state_conditioned | full-vector state readout | preregistered full-vector readout gate did not close | post-hoc operating-point diagnostics (not formal closure) | E0 | frozen | experiments/frozen_registry.yaml; experiments/p1_state_conditioned/ |
| E0-c | p2_state_local_risk | natural-amplitude full-vector reconstruction | prerequisite gate failed; full-vector reconstruction remains unclosed after replication (p2_recovery) | descriptive ranking signal → motivated finite-action recovery | E0 | frozen | experiments/frozen_registry.yaml; experiments/p2_state_local_risk/ |
| E0-d | p3_decision_validity | controlled-score transfer to physical histories; detector / minimal-refresh / candidate-prefilter | controlled score does not transfer to propagated all-layer physical histories; the three gates did not jointly close | event table later reused for the R1 trigger prescreen | E0 | frozen | experiments/frozen_registry.yaml; experiments/p3_decision_validity/ |
| E0-e | p3pr_generalization | fixed relative late-boundary rule | fails fresh formal generalization | dense all-layer mechanism replicates (Spearman 1.0 formal / 0.9940 replication) | E0 | frozen | experiments/frozen_registry.yaml; experiments/p3pr_generalization/ |
| 1 | TF-P0 sketch | fixed-decay Euclidean recursive sketch of layer-27 action vectors | median Spearman gain −0.0157 (eval) / −0.0217 (replication) at 64-dim, ρ=0.95; pairwise and regret also negative | none at primary; negative over dims 16–128, ρ 0.5–1.0 | 24 stored trajectories, E1 | retrospective eval+replication splits | results/temporal_cache_discovery/statekv_tf_sketch_p0_v1/{summary.json,metrics.csv} |
| 2 | TF-P1 metric repair | unlabeled diagonal-RMS (+EMA) metric scaling | eval mean normalized-regret gain −0.0063 (regret worsens; gate veto) | median Spearman +0.0096 (eval) / +0.0122 (replication); replication regret +0.0113 | stored trajectories, E1 | dev screen (splits descriptive, not confirmatory) | results/temporal_cache_discovery/statekv_tf_metric_repair_p1_v1/{summary.json,metrics.csv} |
| 3 | P2 shared pullback | rank-4 shared randomized Fisher pullback, 32-dim sketch, refresh 4 | median Spearman gain 0.0; pairwise gain −0.0281; regret gain −0.0432 | none | 2 dev seqs × 8 candidates, E1 | dev pilot | results/temporal_cache_discovery/statekv_shared_jvp_pilot_p2_v1/{summary.json,metrics.csv} |
| 4 | P3 Gaussian VJP | predeclared output-side Fisher-Gaussian VJP, 16 dirs, refresh 4 | regret gain −0.0638; pairwise gain −0.0153; Spearman gain 0.0 — predeclared route fails | adjoint identity holds (max rel. err 4.5e-4) | 4 dev seqs, E1 | dev pilot | results/temporal_cache_discovery/statekv_vjp_routes_p3_stress_v1/{summary.json,metrics.csv} |
| 5 | P3 Rademacher VJP (post-hoc) | Rademacher 8-dir / repeated-state variant, refresh 4 (post-hoc best) | pairwise-accuracy gain −0.0026 (misses non-negative by 0.0026) | normalized regret 0.1945 → 0.0696 | 4 dev seqs, E1 | dev-only post-hoc; never independently replicated | results/temporal_cache_discovery/statekv_vjp_routes_p3_stress_v1/metrics.csv (row: fisher_rademacher_vjp_repeated_state, w8, r4) |
| 6 | P5 multi-boundary VJP | post-attention VJP sum at layers 0/14/27, width 8, refresh 4 | regret gain −0.0669; pairwise gain −0.0587; Spearman gain −0.0833 | action reconstruction err < 7.2e-4; adjoint identity holds | 4 dev seqs, E1 | dev pilot; ~6 reverse passes/token amortized | results/temporal_cache_discovery/statekv_post_multiboundary_vjp_p5_v1/{summary.json,metrics.csv} |
| 7 | P7 pure contribution | contribution_mean_w4_shared multi-anchor (16/32/48) | NIAH task mean 0.15993 → 0.16061 (+0.0007; all-task-means gate fails) | mean KL 0.3293 → 0.2569; P95 1.5670 → 1.0743; max 13.66 → 13.65; 9/12 seq wins; all 3 anchor means | 12 seqs × 3 anchors, E1 | indep (held-out) | results/temporal_cache_discovery/statekv_direct_policy_independent_multianchor_p7_v1/summary.json; analysis/stratified_metrics.csv |
| 8 | P9 shrinkage λ=0.25 | blend_attention_contribution_25_w4_shared | sample-anchor unit win rate 17/36 = 47.2% < locked 55% | mean KL 0.3577 → 0.3298; P95 1.9535 → 1.5733; max 12.44 → 12.20; both task means; all 3 anchor means; 9/12 sequences | 12 new seqs × 3 anchors, E1 | indep | results/temporal_cache_discovery/statekv_direct_policy_shrinkage_independent_p9_v1/summary.json |
| 9 | P12 selective trigger | fire shrinkage only when score TV > 0.24735 (P8-locked) | activated-unit win rate 50% < 60%; NIAH task mean slightly worse (reduction −3.8e-5) | mean KL 0.35766 → 0.34125; P95 1.95348 → 1.74473; nonworse rate 94.4% | 6 activated of 36 units (P9 seqs), E1 | indep (locked validation on P9 replays) | results/temporal_cache_discovery/statekv_direct_policy_selective_trigger_independent_p12_v1/summary.json |
| 10 | P13 fixed shrinkage, tail gate | same λ=0.25 blend, CVaR tail-risk gate | CVaR95 6.57390 → 6.63419 (worse); KL≥1 rate 11.63% → 11.81% (removes 4 large-loss steps, creates 5) | mean KL 0.52735 → 0.51766; P95 3.62435 → 2.77128; paired reduction on baseline tail +0.163 | third set of 12 seqs, E1 | indep | results/temporal_cache_discovery/statekv_direct_policy_tail_risk_independent_p13_v1/summary.json; analysis/tail_migration.json |
| 11 | P14 protected rescue | top-attention protection + m∈{4,8,16} contribution rescue slots | no m passes all 6 dev constraints; best-mean m=8: CVaR95 3.2221 → 3.2312, max KL 5.795 → 6.315, KL≥1 8.68% → 9.03% all worsen | m=8: mean KL 0.29580 → 0.29367; P95 1.87740 → 1.60877; P95 ΔNLL 2.116 → 2.057 | 6 dev seqs, E1 | dev-only; independent split untouched (not authorized) | results/temporal_cache_discovery/statekv_direct_policy_protected_rescue_screen_p14_v1/metrics.csv; analysis/protected_rescue_selection.json |
| 12 | P15 signal families | six fixed signal families (head-peak, temporal-volatility, key/value geometry, boundary, position) | five of six families fail ≥1 of 6 joint mean/tail/task constraints (e.g. position coverage mean KL 1.119 vs baseline 0.363) | attention head-peak w4 close: mean KL 0.319 vs 0.363, seq wins 4/6; temporal volatility passes all six (0.307) → promoted to P16 | 6 dev seqs, E1 | dev screen | results/temporal_cache_discovery/statekv_direct_policy_signal_family_screen_p15_v1/metrics.csv; analysis/signal_family_selection.json |
| 13 | P18 temporal volatility freegen | frozen TV policy, matched-budget free generation | overall teacher-forced NLL 1.68318 → 1.68692 (+0.00374; frozen primary gate fails) | GovReport ROUGE-L 7.31 → 7.95; RULER 100 = 100; throughput 95.7% of baseline; P16 indep tail gate had passed all checks (mean 0.524 → 0.471, P95 3.02 → 2.86, CVaR95 5.34 → 4.98, max 13.20 → 12.85, KL≥1 13.7% → 12.3%) | 6 paired samples, E1 | P16 indep-passed; P18 frozen-gate veto | results/temporal_cache_discovery/statekv_temporal_volatility_freegen_p18_v1/summary.json; statekv_direct_policy_temporal_volatility_independent_p16_v1/summary.json |
| 14 | P19 attention-free geometry | KNorm / KeyDiff / VNormL2 static geometry scores | all miss both RULER needles; throughput 35–41% of random control; screen not passed | KeyDiff NLL −0.921 vs random, GovReport +0.40; full-reference peak memory | 4 dev samples/method, E1 | dev-only screen; no replication candidate | results/temporal_cache_discovery/statekv_attention_free_geometry_screen_p19_v1/summary.json |
| 15 | P21 static token rarity | token_rarity_shared (no attention capture) | GovReport ROUGE-L 8.92 → 8.09; overall NLL +0.00334 — cross-task gate fails | RULER 3/3 needles retained (100 = 100); throughput 1.252×; peak memory 0.313× of latest attention | 6 untouched samples, E1 | indep replication (P20 dev screen had passed) | results/temporal_cache_discovery/statekv_token_rarity_replication_p21_v1/summary.json; statekv_static_lexical_screen_p20_v1/summary.json |
| 16 | P22→P23b latest-attention refresh proxy | one additive proxy for selection + refresh ordering | refresh-benefit Spearman: dev +0.378 → indep −0.350 (sign reversal; joint gate fails) | action alignment replicated: median Spearman 0.750, normalized regret 0.118 (dev 0.786 / 0.152) | 6 untouched seqs, E1 | dev → indep (frozen proxy, unchanged thresholds) | results/temporal_cache_discovery/statekv_risk_consistent_proxy_alignment_p22_v1/summary.json; statekv_risk_consistent_proxy_independent_p23b_v1/summary.json |
| 17 | P24 output-aware contribution proxy | attention–Value contribution cost as unified proxy | both proxies fail the dev gate; contribution refresh-benefit Spearman 0.011 vs 0.160 for latest attention | none (new proxy strictly worse on the refresh axis) | P7 seqs reused, E1 | dev-only exploratory | results/temporal_cache_discovery/statekv_risk_consistent_output_aware_proxy_p24_v1/summary.json |
| 18 | P29 oracle H=8 | exact-risk teacher, 8-token control horizon, freegen | loses trajectory KL to SnapKV: 0.2394 vs 0.2253 (paired −0.0142) | beats attention (0.5843) and H2O (0.3879) on KL | 2 dev samples, E1 | dev | results/temporal_cache_discovery/statekv_oracle_policy_freegen_p29_v1/summary.json |
| 19 | P29c oracle H=4 | same teacher, 4-token horizon | trajectory KL 0.26603 vs SnapKV 0.26110 — risk gate missed by 0.005 | task quality better than all baselines: official +2.07 vs SnapKV, +1.50 vs attention, +2.76 vs H2O | 2 dev samples, E1 | dev | results/temporal_cache_discovery/statekv_oracle_policy_freegen_h4_p29c_v1/summary.json |
| 20 | P30 oracle H=1 indep | frozen H=1 teacher, 4 new sequences | task-quality gate fails: NIAH 0/2 for all compressed policies (full cache 2/2); GovReport ROUGE-L trails attention/SnapKV | KL gate passed: mean KL 0.1391 = −66.5% / −33.2% / −65.4% vs attention / SnapKV / H2O | 4 new seqs, E1 | indep | results/temporal_cache_discovery/statekv_oracle_policy_freegen_independent_p30_v1/summary.json; analysis/analysis.json |
| 21 | P32-A3 set-output perturbation | cheap controller via set-output perturbation | collapses to Attention in all 640 decisions (README statement; comparison_table.csv confirms A3 metrics identical to Attention) | none | 10 Qwen3-8B samples, E2 | dev screen | results/temporal_cache_discovery/statekv_cheap_policy_freegen_qwen3_8b_n10_p32_v1/comparison_table.csv |
| 22 | P32-A4 uncertainty cascade | uncertainty-cascade cheap controller | no quality gain (official 53.25 = Attention); diagnostic negative | KL 0.32586 vs attention 0.33572 (small) | 10 Qwen3-8B samples, E2 | dev screen | same comparison_table.csv |
| 23 | P32-B1 historical tiny ranker | learned historical-prior ranker | KL 0.18960 — ~2× worse than A2 temporal volatility 0.09525 | none | 10 Qwen3-8B samples, E2 | dev screen | same comparison_table.csv |
| 24 | P34 dynamic layer budget | per-cycle state-dependent per-layer core budgets | loses to layer-shuffled B3: mean KL +0.0270, CVaR95 +0.4417, 2/10 wins; also loses to stale B3 (mean +0.0111, CVaR95 +0.1461); mechanism gate fails | beats b2_uniform and static_adaptive on mean (not the mechanism controls) | 10 samples (86–90), E2 | mechanism gate | results/temporal_cache_discovery/statekv_dynamic_budget_mechanism_qwen3_8b_p34_v1/{p1_analysis.json,aggregate_results.csv} |
| 25 | Gate 0/1 one-step teacher | per-cycle min-exact-KL action over cheap panel, strict pure eviction | Gate 0 NO_HEADROOM: teacher KL 0.2322 vs b2_uniform 0.0961, paired 2/10 (1/10 vs attention), step p95 1.094 vs 0.412; Gate 1 ACTION_SPACE_DOMINANT: oracle regret 1.7% relative | NIAH unchanged (1.0); plateau mechanism mapped (61.6% cycles tied < 1e-3) | 10 samples, E2 pure eviction | fresh matched split | docs/evidence/statekv_gate0_1step_teacher_negative.md; analysis/tables/gate0_*, gate1_*; results/temporal_cache_discovery/statekv_teacher_gate_qwen3_8b_g0_v1/ |
| 26 | ladder 2B/2C | deep (h∈{1,2,4}) teacher-forced risk ladder + 1-step swap marginals | attention family tied at every horizon (h1 0.053, h4 0.28 for all four; regret ~0.0006–0.0013); 1-step marginals ~1e-5–1e-4 all token classes; pair interactions exactly 0 | only uniform separates with depth (regret 0.08 → 0.49) | 10 fresh samples (101–105), E2 | fresh samples | docs/evidence/statekv_ladder_2b_deep_risk.md; docs/evidence/tables/ladder_2b_risk_depth.md |
| 27 | R0 recoverable teacher | physical-risk teacher under recoverable semantics | teacher 0.0213 vs qk_pool 0.0086 (G1 ratio 2.47); paired 0/10; tail 2.2× worse; D3 = −0.0127 negative residual | teacher GovReport official +0.46 (noise band, cannot carry the method) | 10 samples, E2 recoverable | matched R0 substrate | docs/evidence/statekv_recoverable_r0_results.md; analysis/tables/recoverable_r0_*; results/temporal_cache_discovery/statekv_recoverable_r0_qwen3_8b_v1/ |
| 28 | V-routing residual given QK | V / projected-V features for routing, cutoff, revival, heads/layers | QK-conditioned partial Spearman −0.05..−0.10 in every cutoff bucket; 288-swap exact oracle median regret 2e-15 (92% flat < 1e-4); no layer/head/token/horizon pocket | none (I(target;V\|QK) ≈ 0 everywhere measured) | 25M token rows, E2 recoverable | systematic battery A–F | docs/evidence/statekv_qkv_discovery_results.md; analysis/tables/qkv_*.csv |
| 29 | qk_tiered_v | QK routing + 4-bit cold-V tier (H=96 hot), memory-matched 352 | G5 only: tiered-352 KL 0.004845 vs fp16-352 0.004304, ratio 1.126 > preregistered 1.10 → NO_GO (TIERING_LOSSY) | G1–G4 all pass: 0.562× baseline KL, 10/10 paired wins, p95 0.48×, quality non-worse; premise P passes at 256 (ratio 0.944, 6/10 wins) | 10 samples, E2 recoverable | preregistered gate | docs/evidence/statekv_qkvtier_gate.md; docs/evidence/tables/qkvtier_gate_main.md; analysis/tables/qkvtier_gate_paired.csv; runs statekv_qkvtier_gate_{256t,352f,352t}_v1 |
| 30 | selective refresh trigger R1/R2a/R2 | action-conditioned refresh triggers, offline + online | rankings time-invariant at every quality-valid operating point (coverage 0.9979 / 0.9969 / 0.9952; lag-identical 100%); NO_FREEZE at both fits; P23b AUC 0.249 for the frozen proxy baseline | contrast substrate identified: P23b coverage 0.6982, benefit mean 0.1292, 77.1% positive — genuinely stale, substrate-bound | E2 online labels + E1 P22/P23b events | mixed (offline prescreen + online instrumentation) | docs/evidence/tables/selective_refresh_negative_result_r2.md, refresh_operating_point_comparison.csv, refresh_trigger_no_freeze.json, trigger_screen_report.md |
| 31 | refresh-arm sweep (R2b) | never vs every vs fixed_k16 refresh | every-refresh best on all 10 NIAH samples (mean KL 0.024 vs never 0.346, attention arm); NO_CLEAR_REFRESH_ADVANTAGE for never on both policies | none for selective/fixed refresh | 10 fresh samples (101–105), E2 pure eviction | fresh samples | results/temporal_cache_discovery/statekv_refresh_arms_qwen3_8b_768_256_v1/; docs/evidence/tables/refresh_arms_summary.md |
| 32 | quest_like page approximation | 16-token page-granular full-pool scoring | collapses at budget 64: KL 0.7065 vs qk_pool 0.0819, NIAH 0.8 (first task-level failure); page-max recall upper bound 0.674 / 0.623 / 0.585 / 0.543 @ p4/8/16/32 | 2.8–3.0× qk_pool KL at 256/128 (gap exists but no page granularity closes it) | 10 paired samples × 3 budgets, E2 recoverable | open-search runs + offline recall bound | results/temporal_cache_discovery/statekv_openstress_768_{128,64}_v1/sample_results.csv; analysis/tables/open_hf2_page_recall.csv |
| 33 | qk_obswin(w32) | SnapKV-style observation-window full-pool score at h16, budget 64 | KL 1.0748 vs qk_pool@h16 0.8439; NIAH 0.0 both; paired 2/10 — NO_GO_CORNER (backward-looking window is stale-biased) | none; corner recorded as boundary condition (fix is cadence, not scoring) | 10 samples, E2 recoverable | preregistered corner gate | docs/evidence/statekv_corner_gate.md; results/temporal_cache_discovery/statekv_corner_obswin_768_64_h16_v1/sample_results.csv |
| 34 | HF4 per-KV-head selection | per-head own-top-k at identical total budget | captured-mass gain +0.96pp (0.9768 vs 0.9671) < pre-committed "large gap" action threshold → closed without implementation; literature-covered (Ada-KV/HeadKV/KV-Compress) | p95 +3.6pp, concentrated in diffuse early layers (layer 0: +4.8pp) | 3 samples × 36 layers × 8 KV heads, E2 | offline probe | results/temporal_cache_discovery/statekv_headwise_probe_qwen3_8b_v1/; analysis/tables/open_hf4_* |
| 35 | HF1b hard-cycle conditional budgeting | budget-on-hard-cycles conditioned on runtime observables | no predictable trigger: per-cycle KL vs missed mass −0.02, attention entropy +0.21, top-10 mass −0.29, cycle index +0.36 (pool growth), band margin ~0 → deprioritized before any run | regret concentration mapped (top-10% cycles carry 76.5% of KL) | offline over decomposition records, E2 | offline probe | docs/evidence/statekv_open_search_log.md; analysis/tables/open_hf1_* |
| 36 | teacher gate, substrate B | one-step teacher on the P23b (shared-mask, 128/92) substrate | teacher wins 1/6 vs best cheap (trajectory KL 0.06–0.75 vs 0.04–0.95); NIAH 0.0 for teacher AND every cheap policy → operating point quality-invalid; no headroom | none | 6 samples (P23b split), E1 | fresh gate run | docs/evidence/statekv_teacher_closure_2026-08-09.md §2; results/temporal_cache_discovery/statekv_teacher_gate_qwen25_15b_p23b_v1/ |

Current-status pointers (from `configs/ccfa.yaml`): the pure-eviction chassis
itself passed (P35; `strict-pure-eviction-deployability`
mechanics-supported); the direct four-query contribution policy is
"not-supported-as-default-selective-or-cvar-tail-controller" (entries 7–11);
temporal volatility is "teacher-risk-positive, free-generation primary gate
failed" (entry 13); the proxy controller is "action-alignment-positive,
refresh-ordering-not-supported" (entry 16); oracle freegen is
"distribution-risk supported, task-quality competitive" only at H=1
(entries 18–20); qk_tiered_v is "refuted-gate-nogo-tiering-lossy" (entry 29);
selective refresh on Qwen3-8B is "refuted-substrate-bound" (entry 30).

---

## Part 2 — Decisive rejections (do not re-test without new evidence)

**E0-a..E0-e (Era-0 frozen negatives).** The original predictive-closure audit,
the full-vector state readout, natural-amplitude full-vector reconstruction,
controlled-to-physical score transfer, and the fixed relative late-boundary rule
all failed preregistered formal gates on the frozen Qwen2.5-1.5B protocol, and
the registry froze them as negative evidence. They are closed at the formal-gate
level; the later program exists precisely because these lines failed. Re-testing
means re-opening a frozen registry, not running an experiment.

**1 — TF-P0 Euclidean sketch.** The negative holds at the predeclared point and
across the whole swept grid (dims 16–128, ρ 0.5–1.0) on both eval and
replication splits; every metric moves the wrong way. There is no residual
positive cell to rescue.

**3 — P2 shared Fisher pullback.** Zero Spearman gain with negative pairwise and
regret gains on the bounded pilot; the shared low-rank pullback adds cost
(finite-difference probes) and no ranking signal. A re-test would need a
different metric hypothesis, not more samples of this one.

**4 — P3 predeclared Gaussian VJP route.** The predeclared 16-direction Gaussian
route fails on all three ranking metrics; only the adjoint numerics validate.
The route as preregistered is dead (the post-hoc Rademacher variant is entry 5,
a separate question).

**6 — P5 multi-boundary VJP.** Regret gain −0.0669 with all ranking metrics
negative at ~6 reverse passes per token — strictly dominated on both axes
(accuracy and cost).

**14 — P19 attention-free geometry.** All three candidates miss both RULER
needles and run at 35–41% of the random control's throughput; KeyDiff's NLL gain
is a development-screen curiosity explicitly marked not replication-eligible.
Dev-only, but the failure mode (task quality + speed simultaneously) is not
noise-shaped.

**17 — P24 output-aware contribution proxy.** Both the new contribution proxy
and latest attention fail the development gate, and the new proxy's
refresh-benefit Spearman (0.011) collapses relative to latest attention (0.160)
on the same events. The output-aware repair hypothesis made things worse, not
marginally short — reclassified here as decisive (the draft had it marginal).

**21 / 23 — P32-A3 and P32-B1.** A3's set-output perturbation collapses to the
Attention action in every one of the 640 decisions — a degenerate mechanism, not
a weak one. B1's historical tiny ranker is ~2× worse than A2 on KL (0.18960 vs
0.09525) with no compensating quality gain. Both are screen diagnostics with
clearly negative point estimates on 10 paired samples.

**24 — P34 dynamic per-layer budgets.** The dynamic rule loses to the
layer-shuffled-static control (8/10 samples, mean +0.0270, CVaR95 +0.4417) and to
the stale control — the two controls that destroy exactly the state-condition
information the mechanism claims to use. That is a mechanism-level refutation,
not a metric near-miss.

**25 — Gate 0/1 one-step pure-eviction teacher.** The teacher is 2.4× *worse*
than the best cheap policy (0.2322 vs 0.0961), and Gate 1 shows why: the action
space is degenerate in one-step risk (1.7% oracle regret, 61.6% of cycles tied).
The closure is structural — one-step physical risk cannot rank actions at this
operating point — so no one-step surrogate inherits anything to distill.

**26 — Ladder 2B/2C.** Deep risk exists but is non-discriminative: the whole
attention family stays tied at horizons 1–4, and one-step marginals are flat
(1e-5–1e-4) with exactly zero interactions. The only remaining distillation
channel (depth) is closed by measurement, not by threshold.

**27 — R0 recoverable teacher.** Negative headroom under recoverable semantics:
teacher 0.0213 vs qk_pool 0.0086, 0/10 paired wins, worse tail, and the scorer's
residual D3 = −0.0127 is negative. The decomposition also explains the old P31
gain as a machinery artifact (backing store + quasi-irreversible baselines).
Per protocol the recoverable pivot stopped here.

**28 — V-routing residual given QK.** Systematic falsification: negative partial
correlation in every cutoff bucket, a flat exact swap oracle (median regret
2e-15, 92% flat), no head/layer/token-type/horizon pocket, and revival predicted
by attention, not V. The measurement battery is exhaustive at this substrate;
re-testing needs a new substrate, not a new V feature.

**30 — Selective-refresh triggers (R1/R2a/R2).** The premise is absent on
Qwen3-8B per-layer selection: rankings are time-invariant (coverage
0.995–0.998), so a trigger has nothing to capture — NO_FREEZE at both fitting
gates, and the more aggressive operating point is quality-invalid anyway (NIAH
0.0). The P23b contrast (coverage 0.698, 77% positive benefit) confirms the
phenomenon is real but substrate-bound.

**31 — Refresh-arm sweep.** Every-refresh is best on all 10 NIAH samples (0.024
vs 0.346 never) and the predeclared verdict is NO_CLEAR_REFRESH_ADVANTAGE for
not refreshing. There is no selective-refresh gap to exploit at budget 256.

**32 — quest_like page approximation.** The page-max recall ceiling
(0.674–0.543 across page sizes) is an *offline upper bound* computed with an
oracle within-page max; the realized p16 policy already collapses at budget 64
(KL 0.7065, NIAH 0.8). Token-level exactness is unrecoverable by page metadata
and the gap widens at tight budgets.

**33 — qk_obswin(w32).** The preregistered corner gate failed cleanly: the
observation-window score is worse than the freshest single token at the same
cadence (1.0748 vs 0.8439, NIAH 0.0 both). Recorded as a boundary condition:
at tight coverage the fix is cadence, not scoring.

**35 — HF1b conditional budgeting.** No runtime observable predicts hard cycles
(correlations −0.02 to +0.36, band margin ~0); the hypothesis was
deprioritized before any model run per the pre-committed probe rule. There is
no trigger to build.

**36 — Teacher gate on substrate B.** On the P23b shared-mask substrate the
teacher wins 1/6 and every compressed policy is quality-invalid (NIAH 0.0 while
full cache retrieves). No headroom, and the operating point itself is unusable.

---

## Part 3 — Marginal / veto-style rejections (the re-test targets)

**2 — TF-P1 diagonal metric repair.** Improved median Spearman on both splits
(+0.0096 eval, +0.0122 replication) and regret on replication (+0.0113); vetoed
solely because eval normalized regret worsened by 0.0063. Development screen
with descriptive splits, so neither direction is confirmatory — a cheap replay
re-screen on fresh sequences can settle it.

**5 — P3 Rademacher VJP (post-hoc).** The strongest veto-style case in Era 1:
normalized regret 0.1945 → 0.0696, missing the pairwise-accuracy check by
0.0026. It was found post-hoc inside a dev pilot and never independently
replicated — the veto may be pure multiple-comparison noise, in either
direction.

**7 — P7 pure contribution.** Independent run; mean KL 0.3293 → 0.2569, P95
1.5670 → 1.0743, 9/12 sequence wins — vetoed by a +0.0007 wobble in the NIAH
task mean under an all-task-means-must-improve gate. The veto metric is two
orders of magnitude smaller than the improvements.

**8 — P9 shrinkage λ=0.25.** Everything improves (mean/P95/max KL, both task
means, all three anchor means, 9/12 sequences); vetoed only by the locked 55%
sample-anchor win-rate floor (17/36 = 47.2%). A distributional gate, not a
magnitude gate.

**9 — P12 selective trigger.** Mean KL 0.35766 → 0.34125 and P95 → 1.74473 with
94.4% nonworse units; vetoed because the 6 activated units won only 50% (< 60%)
and the NIAH mean moved −3.8e-5. Tiny activation set, noise-scale veto.

**10 — P13 tail gate.** Mean KL and P95 improve (0.52735 → 0.51766,
3.62435 → 2.77128) and the paired reduction on the baseline's own tail is
+0.163; vetoed because CVaR95 moves +0.060 and the KL≥1 set changes by one net
step (removes 4, creates 5). A one-step migration veto.

**11 — P14 protected rescue.** m=8 improves mean KL, P95 KL, and P95 ΔNLL;
vetoed by CVaR95 (+0.009), max KL, and a +0.35pp KL≥1 rate. Dev-only by design
— the independent split was never touched, so the re-test costs nothing extra
beyond running it.

**12 — P15 signal families.** Five families fail at least one of six joint
constraints, but attention head-peak w4 was close (mean KL 0.319 vs 0.363, 4/6
sequence wins). Rejected without any within-family parameter search, per the
screen's rules.

**13 — P18 temporal volatility freegen.** The flagship veto case: the same
frozen policy passed the full independent tail gate in P16 (all five tail
metrics improve) and then failed the free-generation primary gate on an NLL
delta of +0.00374, while GovReport ROUGE-L improved 7.31 → 7.95, RULER stayed
100, and throughput was 95.7%. Passed-independent-then-frozen-veto.

**15 — P21 token rarity.** Independent replication kept all three 16K RULER
needles with 1.252× throughput and 0.313× peak memory; vetoed by GovReport
ROUGE-L 8.92 → 8.09 and NLL +0.00334. A retrieval-specific success killed by
the cross-task gate.

**16 — P22→P23b refresh proxy.** Action alignment replicated strongly (median
Spearman 0.750, regret 0.118); the veto is the refresh-benefit Spearman
reversing sign (+0.378 → −0.350). The magnitude is not small, but the
action-ranking half of the claim survived independent audit — the re-test
question is whether any refresh ordering is learnable at all (see entry 30:
substrate-bound).

**18 — P29 oracle H=8.** Beats attention (0.5843) and H2O (0.3879) on trajectory
KL; vetoed by SnapKV alone (0.2394 vs 0.2253). A one-baseline veto at long
control horizon.

**19 — P29c oracle H=4.** Misses the SnapKV risk gate by 0.005 (0.26603 vs
0.26110) while posting the best task-quality numbers of any arm (+2.07 official
vs SnapKV). The closest near-miss in the oracle line.

**20 — P30 oracle H=1 independent.** The KL gate passed decisively (−66.5% /
−33.2% / −65.4% vs attention/SnapKV/H2O); the veto is NIAH 0/2 for all
compressed policies on n=2 NIAH samples — a two-sample task-quality veto on an
otherwise large distribution-risk win.

**22 — P32-A4 uncertainty cascade.** KL slightly better than attention
(0.32586 vs 0.33572) with zero quality gain — a weak-positive KL with a
flat-quality veto in a dev screen.

**29 — qk_tiered_v.** G5-only veto: the memory-matched method beats the
256-FP16 baseline on every preregistered gate (0.562× KL, 10/10 wins, 0.48×
tail, quality non-worse) and the premise ablation passes at 256 (ratio 0.944);
it fails only the coverage-fidelity ratio against the 1.375×-memory FP16-352
control (1.126 vs ≤1.10). The comparator is deliberately unequal-memory — see
Part 4.

**34 — HF4 per-KV-head selection.** The probe found a real but sub-threshold
gain (+0.96pp mean captured mass, +3.6pp p95, concentrated in diffuse early
layers) against a pre-committed "large gap" rule, plus a literature-coverage
closure. A threshold veto on a measured positive.

---

## Part 4 — Caveats for the re-test program

1. **Substrate non-transferability is measured, not hypothetical.** The same
   refresh question is degenerate on Era-2 Qwen3-8B per-layer (coverage
   0.995–0.998; entries 30–31) and real on the Era-1 P23b shared-mask substrate
   (coverage 0.698, 77% positive benefit; entries 16, 30). Any re-test must
   state its era; an Era-1 veto says nothing about Era 2 and vice versa. The
   cross-machinery caution is also quantified: P31's attention policy scored
   0.3357 vs P35's 0.0976 at the same budget on the same substrate — numbers
   from different evaluation machinery are not comparable (Gate-0 doc, point 4).
2. **Dev-only entries carry weaker negatives.** Entries 2, 3, 4, 5, 6, 11, 12,
   14, 17, 18, 19, 21, 22, 23, 34, 35 rest on development sequences or offline
   probes only. For the marginal ones (2, 5, 11, 12, 18, 19, 22, 34) this is
   exactly why they are re-test targets; for the decisive ones the mechanism
   (degeneracy, ties, collapse) rather than the effect size is the reason they
   stay closed. Entry 5 is the sharpest case: a post-hoc positive that was
   never independently replicated.
3. **qk_tiered_v's comparator is unequal-memory by design.** G5 pits the
   memory-matched tiered-352 (+0.8% vs the 256-FP16 baseline) against
   FP16-352 at 1.375× memory. The method passed every equal-memory gate
   (G1–G4) and the premise gate P; the 1.126-vs-1.10 miss says "4-bit cold V
   costs 12.6% of the coverage gain", not "the method loses at equal memory".
   A re-test must reproduce the exact arm memories before comparing numbers.
4. **P16 passed independent, P18 vetoed frozen.** Temporal volatility passed
   the full independent tail-risk gate (P16) and then failed the frozen
   free-generation NLL gate (P18) by +0.00374. The two results are both valid;
   the veto is a different deployment surface (free generation vs teacher-forced
   replay), not a contradiction. Note also the P18 scope warning: the MLX
   attention-capture path peaks at 3.19× full-cache memory, so the policy is
   not a low-memory deployment path as implemented.
5. **Metric-boundness.** The program's discriminator is exact same-input KL;
   task metrics saturate on the standard substrates (R0 §6.1 audit: all arms
   quality-equivalent at 64-token ROUGE resolution). Vetoes registered on NIAH
   with n=2 (entry 20) or on 64-token GovReport are low-resolution vetoes.
6. **Completeness sources.** Entries were cross-checked against
   `configs/ccfa.yaml` (claims + experiments status fields), the README
   evidence table, and `experiments/frozen_registry.yaml`. Two registry items
   are deliberately excluded: `multi-boundary-direct-policy-distillation`
   (superseded, not gate-failed) and `statekv-tail-telemetry-qwen3-p36`
   (superseded before being run). The external-validity battery
   (`docs/evidence/statekv_external_validity_report.md`) is a DRAFT with all
   verdicts [PENDING] as of 2026-08-10; it challenges the substrate generality
   of the Era-2 closures (entries 25–35) but has not yet re-scoped any of them.
7. **Single README-sourced number.** Entry 21's "640 decisions" is stated in
   README.md; the raw artifact (comparison_table.csv) confirms A3 is
   metric-identical to Attention but does not itself contain the decision
   count. All other numbers trace to raw run artifacts or derived tables listed
   in Part 1.
