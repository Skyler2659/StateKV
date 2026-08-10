# StateKV selective-refresh investigation — negative result at the Qwen3-8B operating point (R1/R2a)

Status: final
Date: 2026-08-09
Question: can a cheap action-conditioned trigger predict when re-ranking a stale KV retained set is worth it, under strict pure eviction with per-layer attention/B2 selectors on Qwen3-8B?

Formal fitting-gate verdicts (predeclared in analysis/tables/fit_refresh_trigger.py: freeze requires LOSO AUC ≥ 0.65 and alerted-benefit lift in ≥ 80% of sample folds): NO_FREEZE on the 768/256 run (10 samples; best rule lift 6/10 folds, benefits ~1e-4) and NO_FREEZE on the 4K/128 run (4 samples; best rule lift 2/4, all fixed-interval alerted benefits negative). v3 collection was stopped at 4/10 samples after three consecutive gov_report samples and one NIAH sample reproduced the identical degenerate signature; with the premise absent, further samples could not change the verdict.

## Answer (negative, mechanistically explained)

No — because on this model family and selector family the premise is absent:
**cheap per-layer attention rankings are time-invariant at every tested operating point**, so stale-vs-fresh decisions coincide (up to the deterministic recent-window slide) and the refresh benefit is noise-level. A trigger has nothing to capture.

## Evidence chain

1. Offline pre-screen (R1, no model runs):
   - `experiments/p3_decision_validity` event table (3 splits): only transferable signal is the structural bit "state changed since selection" (≡ always-refresh-when-stale); every selective action-conditioned feature fails out-of-sample (boundary features AUC 0.32–0.49; best generic scalar sink_attention_mass AUC ~0.68). `analysis/tables/p3_trigger_prescreen_*`
   - P22→P23b frozen transfer: no dev-selectable feature beats the failed proxy_regret baseline (P23b AUC 0.249). churn features are not computable offline (cores/scores never persisted). One post-hoc lead (early stale-trajectory KL slope, P23b AUC 0.878) is dev-invisible and teacher-side — not a cheap trigger. `analysis/tables/trigger_screen_report.md`

2. Online instrumentation (R2a, new machinery): per-step cheap features (core churn Jaccard, boundary margin, score TV, coverage mass) + teacher labels (refresh_benefit = stale_exact_kl − fresh_exact_kl via shallow-clone counterfactual forwards; proven non-perturbing, max |ΔKL|=0) under strict pure eviction. 36 unit tests + smoke identity checks pass.

3. Operating-point sweep on Qwen3-8B (all per-layer, pure eviction, 64-token generation):
   `analysis/tables/refresh_operating_point_comparison.csv`
   - 768 ctx, budget 256/core 220 (10 samples): coverage 0.9979, lag4≡lag16 in 100% of steps, Σbenefit/ΣKL = +0.15%
   - 768 ctx, budget 128/core 92 (1 sample, early-stopped): coverage 0.9969, lag-identical 100%, Σbenefit/ΣKL = −0.27%
   - 4K ctx, budget 128/core 92 (4 samples: gov_report 96–98, NIAH 96): coverage 0.9952, lag4≡lag16≡lag32 in 100% of steps, Σbenefit/ΣKL = −0.44%
   - churn per step pinned at exactly the recent-window slide (2 positions/layer); boundary margin ~1e-4 (near-zero tail, position tie-broken)
   Contrast — P23b (Qwen2.5-1.5B, 4–8K LongBench ctx, budget 128/core 92, **shared-mask**): coverage **0.6982**, teacher refresh benefit mean 0.1292, 77.1% of events positive. Staleness is real there.

4. Mechanism (quantified): staleness requires the retained set to hold a limited share of score mass (~0.7), so that *which* tokens are kept matters and ranking drift changes the mass retained. Per-layer attention-family selection on Qwen3-8B concentrates ~99.6–99.8% of mass inside the core at all tested budgets; scores do change (per-step score TV ≈ 0.19–0.24) but the mass redistribution happens entirely *within* the retained set, leaving the core/tail boundary untouched. Two boundary variables identified: per-layer vs shared-mask selection, and model-family attention concentration.

5. Quality double-negative: the more aggressive operating point (128/4K) also breaks the task — NIAH needle retrieval 0.0 for both selectors (vs 1.0 at 256/768 in P35). So: at quality-valid operating points the refresh problem is degenerate; at compression levels where it could exist, the policy fails the task anyway.

## Boundary claims (supported)

- On Qwen3-8B with per-layer latest-attention or B2-blend selectors, ranking staleness is negligible at 768–4K contexts and 128–256 budgets; selective refresh cannot beat (or match) simply not refreshing — fresh re-ranking is itself slightly harmful (−0.3% to −2% KL on degenerate tasks).
- The P23b refresh phenomenon (77% positive, mean 0.129) is substrate-bound: shared-mask selection and/or a more diffuse-attention model family, at retained-mass coverage ≈ 0.7.
- Refresh research on this mainline should only resume if the selector changes the coverage regime (e.g., value-aware or output-aware costs that redistribute mass), or the deployment target moves to shared-mask / small-model settings. The label-collection machinery (R2a) and the 4-arm gate (R2b) are implemented and tested for exactly that check.

## Artifacts

- Label runs: results/temporal_cache_discovery/statekv_selective_refresh_labels_r2a_v1 (768/256, 10 samples), _v2 (768/128, partial), _v3 (4K/128)
- Machinery: statekv/budget_dynamics.py (FrozenRanking, stale_selection, feature fns), statekv/refresh_trigger.py (allowlisted trigger rules), runner refresh modes + label mode, tests/test_refresh_scheduling.py + test_refresh_trigger_gate.py (36 tests)
- Fitting gate: analysis/tables/fit_refresh_trigger.py (predeclared: LOSO AUC ≥ 0.65 and benefit lift in ≥ 80% folds, else NO_FREEZE)
- Tables: analysis/tables/refresh_operating_point_comparison.csv, refresh_gap_decomposition_* (R0), trigger_* (R1)
