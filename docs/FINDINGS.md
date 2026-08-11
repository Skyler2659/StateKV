# StateKV findings — graded by current evidence

Compiled 2026-08-10. Every entry links its evidence. Grades:
**A** strong (multiple experiments, never overturned) · **B** conditional
(model/task/budget-bound) · **C** observation (insufficient evidence) ·
**D** negative (explicitly falsified) · **E** open question.

The companion `FAILURE_ANALYSIS.md` explains *why* the main line failed;
`EXPERIMENT_REGISTRY.md` lists every experiment behind these
entries.

---

## A. Strong findings

- **A1. The exact set-level deletion identity is numerically exact.**
  Max FP64 L2 error 2.26e-11 at the fixed operating point.
  Evidence: `experiments/p0_v2_fixed_boundary/results/identity_rows.parquet`.
- **A2. State-conditioned evaluation is measurable and accurate.** Cosine
  0.99974, relative L2 0.02255 at the observed compressed state.
  Evidence: `experiments/p1_state_conditioned/results/state_operating_point_summary.json`.
- **A3. A state-local scalar risk evaluator ranks frozen candidate pools
  near-perfectly.** Two-midpoint scalar risk: Spearman 1.0, top-1 gain 1.0,
  evaluation and replication; dense mechanistic risk transfers across two
  model families and two task families (P3PR).
  Evidence: `experiments/p2_recovery/r4_scalar_decision_risk/results/`,
  `experiments/p3pr_generalization/results/analysis/analysis_summary.json`.
- **A4. Exact per-query full-pool QK routing (qk_pool) is the strongest
  working-set policy at every quality-valid operating point tested.**
  Nothing beat it across: recoverable teacher gate (R0, paired 0/10),
  coverage stress (NIAH 1.0 down to 8% coverage at 768 ctx; down to 1.4%
  at 3–4.7K ctx), the open search (7 hypothesis families), and external
  validity (two model families).
  Evidence: `docs/evidence/statekv_recoverable_r0_results.md`,
  `docs/evidence/statekv_open_search_report.md`,
  `docs/evidence/statekv_external_validity_report.md`.
- **A5. The state-conditioned physical-risk teacher has no deployable
  headroom over cheap selectors under strict pure eviction.** Teacher KL
  0.232 vs best cheap 0.096 (paired 2/10); the action space is degenerate
  (oracle regret 1.7%); the P31 headroom was a full-history-access
  artifact. Supported on two substrates.
  Evidence: `docs/evidence/statekv_teacher_closure_2026-08-09.md`,
  `docs/evidence/tables/gate0_teacher_headroom.md`, `analysis/tables/gate0_*`.
- **A6. One-step risk is a plateau; long-run risk is a cliff.** 61.6% of
  cycles tie numerically (48.1% hard-tied); swap marginals ~1e-5–1e-4 with
  exactly zero pair interactions; the cliff becomes visible only 2–4 steps
  before the future query, shared by all panel actions.
  Evidence: `docs/evidence/statekv_ladder_2b_deep_risk.md`,
  `analysis/tables/ladder_2b_*`, `analysis/tables/qkv_c2_swap_by_offset.csv`.
- **A7. The coverage × cadence interaction cliff is the dominant failure
  mode of working-set control.** At 768 ctx / 64 budget: h1 KL 0.082 /
  NIAH 1.0 → h4 0.376 / 0.2 → h16 0.844 / 0.0; reproduces at 3072 ctx; the
  controlling variable is the absolute core budget; no refresh-time
  observable rescues slow cadence (observation-window scoring is *worse*).
  Evidence: `docs/evidence/statekv_corner_gate.md`,
  `docs/evidence/statekv_external_validity_report.md`.
- **A8. Token-level exactness cannot be recovered by page-granular
  metadata.** Page-max recall upper bound of the exact top-220 core: 0.674
  (p4) → 0.543 (p32); quest_like collapses at budget 64 (KL 0.71 vs 0.08).
  Evidence: `analysis/tables/open_hf2_page_recall.csv`,
  `results/temporal_cache_discovery/statekv_openstress_768_64_v1/`.
- **A9. Cold-V 4-bit tiering under QK routing is near-lossless at matched
  budget.** qk_tiered_v KL 0.0081 vs qk_pool 0.0076 with identical task
  scores (n=20 fresh, recoverable freegen); tiered-256 beat qk_pool-256
  6/10 in the original gate. Its original NO_GO rested on an unequal-memory
  comparison (G5 vs 1.375×-memory FP16-352).
  Evidence: `docs/evidence/statekv_qkvtier_gate.md`,
  `results/temporal_cache_discovery/statekv_retest_freegen_qwen3_8b_n20_v1/`.
- **A10. Binary task metrics saturate long before distribution fidelity
  does.** Across the retest panel (n=20), 9 of 12 compressed policies score
  NIAH 10/10 while their mean KL spans 0.008–0.87; GovReport ROUGE-L spans
  0.055–0.061. KL/tail metrics detected every eventual task collapse
  earlier than task scores did.
  Evidence: `docs/evidence/statekv_retest_report.md` (Track B),
  `docs/evidence/statekv_open_search_report.md` §3.

## B. Conditional findings

- **B1. Four-query attention–Value contribution selection beats latest
  attention on Era-1 replay** (Qwen2.5-1.5B, shared mask, budget 128):
  mean KL 0.394 vs 0.408, 54–58% sequence wins on 24 fresh sequences —
  but the family never passed an independent task gate in its own era, and
  Era-2 does not inherit the conclusion (different mask semantics).
  Evidence: `results/temporal_cache_discovery/statekv_retest_replay_era1_n24_v1/metrics.csv`,
  catalog entries #7–#11.
- **B2. Static token rarity is retrieval-specific.** Replicates 3/3 RULER
  needles with 1.25× throughput and 0.31× peak memory (Era-1), NIAH 10/10
  on Qwen3-8B (retest) — but GovReport ROUGE-L trails attention in both
  eras. A retrieval-route candidate, not a general selector.
  Evidence: `results/temporal_cache_discovery/statekv_token_rarity_replication_p21_v1/`,
  retest Track B.
- **B3. Staleness (refresh value) exists only in low-coverage shared-mask
  substrates.** P23b (coverage 0.698) shows real staleness (benefit 0.129,
  77% positive); Qwen3-8B per-layer at coverage 0.995+ is time-invariant.
  Evidence: `analysis/tables/refresh_operating_point_comparison.csv`,
  `docs/evidence/tables/selective_refresh_negative_result_r2.md`.
- **B4. P32 cheap controllers (B1/B2/B3) keep NIAH 10/10 with KL
  0.18–0.24 vs attention 0.33 on Qwen3-8B 768-ctx freegen** — but B3's
  dynamic layer-budget *mechanism* is refuted (loses to shuffled static
  control, P34); what works is a static layer-budget prior. Evidence:
  retest Track B, `results/.../statekv_dynamic_budget_mechanism_qwen3_8b_p34_v1/`.
- **B5. Nearest-value merge and 2/3/4-bit cold-value tiers have positive
  local diagnostics** (merge wins 192/192 matched units; tiers at
  23–34% of FP16 storage) — local mechanism only; keys, retrieval, and
  end-to-end replay never evaluated. Evidence:
  `results/temporal_cache_discovery/statekv_direct_coreset_p4_replication_v1/`.

## C. Observations (insufficient evidence)

- **C1.** qk_pool residual KL is event-driven (top-10% cycles carry 76.5%
  of KL mass) but hard cycles are not predictable from runtime observables
  (correlations ≤ 0.36). Evidence: `docs/evidence/statekv_open_search_report.md` §3.5.
- **C2.** Per-KV-head own-top-k gains +0.96pp captured mass (p95 +3.6pp,
  diffuse early layers) — below the pre-committed action threshold;
  literature-covered (Ada-KV/HeadKV). Evidence:
  `analysis/tables/open_hf4_headwise_by_layer.csv`.
- **C3.** Temporal attention volatility passed the Era-1 independent
  teacher-forced tail gate (all five tail metrics improved) but failed the
  frozen freegen NLL gate (+0.00374) and is not task-competitive on Era-2
  (NIAH 0.8). Whether it has *any* valid operating point is untested.
  Evidence: catalog entries #13, retest Track B.
- **C4.** P29c horizon-4 oracle had better task quality than all baselines
  while missing the KL gate by 0.005 (n=2 development sequences). Too
  small to conclude anything. Evidence: catalog entry #19.

## D. Negative findings (explicitly falsified)

Each of these was a live hypothesis with a designed test:

- **D1.** Deployable state-conditioned risk distillation (Gate 0–2, R0;
  A5/A6 above).
- **D2.** V-side residual signal given QK: partial Spearman −0.05…−0.10 in
  every cutoff bucket; I(target;V|QK) ≈ 0. `docs/evidence/statekv_qkv_discovery_results.md`.
- **D3.** Selective refresh triggers on Qwen3-8B per-layer (premise absent;
  LOSO AUC < 0.65). `docs/evidence/tables/selective_refresh_negative_result_r2.md`.
- **D4.** Dynamic per-layer budget allocation (P34; loses to shuffled
  static and stale controls).
- **D5.** Training-free cheap estimators as controller metrics: Euclidean
  history sketches (TF-P0), diagonal metric repair (TF-P1), shared Fisher
  pullback (P2), Gaussian VJP (P3), multi-boundary VJP (P5).
- **D6.** Rademacher VJP (P3 post-hoc): dev gain was a selection artifact;
  all gains negative on 8 fresh sequences (retest Track D).
- **D7.** Page-granular approximation of exact routing at tight budgets
  (A8); observation-window scoring at slow cadence (corner gate).
- **D8.** Attention-free geometry selectors (KNorm/KeyDiff/VNormL2): miss
  both RULER needles, 35–41% of control throughput (P19).
- **D9.** Conditional hard-cycle budgeting: no observable trigger exists
  (HF1b).
- **D10.** Era-0 closures: predictive_closure, p2_state_local_risk
  (natural-amplitude full-vector reconstruction), p3_decision_validity —
  frozen negatives in `experiments/frozen_registry.yaml`.
- **D11.** Per-head selection beyond action threshold (HF4); the P32 A3
  set-output perturbation (collapses to Attention); A4 cascade (no gain).

## E. Open questions exposed by the experiments

- **E1.** Why do allocation/eviction policies that differ enormously in KL
  converge to identical task scores over wide operating ranges? (A10 is
  descriptive; the mechanism of task-score robustness is unexplained.)
- **E2.** Is there *any* quality-valid, low-coverage operating point where
  staleness exists (B3) and a refresh trigger has value? The P23b
  phenomenon has never been observed at a quality-valid point.
- **E3.** Does the cadence cliff (A7) have a smooth frontier usable for
  deployment-cost modeling (refresh rate × memory × quality)? The corner
  data exists but no cost model was built.
- **E4.** Cross-session / prefix reuse (HF5) was deferred for lack of a
  harness — untouched.
- **E5.** The extval reasoning task scored 0 for every policy including
  full cache at 64-token decode — long-generation scoring at usable
  horizons was never validated as an instrument.
