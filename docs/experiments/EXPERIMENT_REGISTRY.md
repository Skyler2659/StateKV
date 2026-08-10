# StateKV Experiment Registry

Complete registry of every experiment run in this repository, compiled 2026-08-10
from `configs/ccfa.yaml` (claims + experiment statuses),
`experiments/frozen_registry.yaml` (Era-0 frozen phases),
`analysis/statekv_gate_retrospective_catalog.md` (41 verified rejected-strategy
entries), the README evidence table, and the run directories under
`results/temporal_cache_discovery/`. Every result path below was checked for
existence on disk.

## How to read this registry

**Status vocabulary** (exactly these values):

- `VALID` — the run is sound and its result stands as recorded.
- `NEGATIVE RESULT` — a completed, sound run whose preregistered gate/hypothesis
  failed. A negative result is a result, not a failed run.
- `INCONCLUSIVE` — evidence insufficient to accept or reject.
- `SUPERSEDED` — replaced by a later phase (named in the notes).
- `INVALID` — the run does not measure what it claims (reason noted).
- `BUGGED` — a metric or implementation defect affects the recorded artifacts
  (reason and salvage noted).
- `SANITY CHECK` — smoke/debug scaffolding, never intended as evidence.
- `EXPLORATORY` — probing run without a preregistered gate.

**Substrate shorthand** (eras are non-transferable; see the retrospective
catalog, Part 4):

- `E0` — frozen Qwen2.5-1.5B protocol (`experiments/*`).
- `E1` — Qwen2.5-1.5B-Instruct-4bit, shared-mask selection, budget 128 /
  core 92 (sink 4, recent 32); phases P0–P24 + substrate-B gate.
- `E2` — Qwen3-8B-4bit, per-layer selection, budget 256 / core 220 (sink 4,
  recent 32), recoverable or strict pure-eviction; phases P25–P35, gates,
  R0–R2, QK–V battery, open search, external validity.

**Directory conventions** (`results/temporal_cache_discovery/`):

- *Canonical* run dirs contain `summary.json` + `config.yaml`; these are the
  curated evidence dirs referenced below.
- *Seed-suffixed twins* (e.g. `..._seed20260808_v1`) are runtime scaffolding
  produced during execution. They are not separate experiments and are not
  listed as rows (82 seed-suffixed dirs exist).
- Runtime-fragment dirs (e.g. `statekv_p1_p3_gates_qwen3_8b_seed20260808_v1_p1`)
  are per-stage scratch behind the curated P33/P34/P35 dirs; they exist only as
  fragments with no curated summary.

"Config" and "Runner" columns give repository-relative paths. "Result" is the
canonical artifact (run dir + key file, or analysis table for offline studies).

---

## 0. Era 0 — frozen phases (`experiments/*`, registry: `experiments/frozen_registry.yaml`)

| ID | Research question / hypothesis | Substrate | Config | Runner script | Result location | Main result | Status |
|---|---|---|---|---|---|---|---|
| E0 predictive_closure | Original predictive-closure audit under native 4-bit | E0 | `experiments/predictive_closure/configs/primary.yaml`, `configs/p0_formal_4bit.yaml` | `experiments/predictive_closure/scripts/run_p0_formal.py` | `experiments/predictive_closure/` | P0-formal gate failed: native 4-bit / mismatched execution graphs do not support the closure audit; precision diagnostics retained | NEGATIVE RESULT |
| E0 local_truncated_jacobian | Local Jacobian / finite-difference boundary study | E0 | `experiments/local_truncated_jacobian/configs/local_primary.yaml` | `experiments/local_truncated_jacobian/scripts/run_l0_formal.py` (…l1/l2/l3) | `experiments/local_truncated_jacobian/` | Local action transport informative only within registered depth and numerical regime (frozen-boundary-evidence) | VALID (boundary evidence) |
| E0 p0_v2_fixed_boundary | Matched-graph deletion identity at fixed boundary | E0 | `configs/frozen/p0_v2_config.yaml` | `experiments/p0_v2_fixed_boundary/scripts/run_p0_v2.py` | `experiments/p0_v2_fixed_boundary/results/p0_v2_summary.json` | Set-level deletion identity max FP64 L2 error 2.26e-11; boundary replay cosine ≈ 1, rel L2 8.09e-7 (frozen-positive-evidence) | VALID |
| E0 p1_state_conditioned | Does observed state change action geometry? | E0 | `configs/frozen/p1_state_conditioned_config.yaml` | `experiments/p1_state_conditioned/scripts/run_p1.py` | `experiments/p1_state_conditioned/results/state_operating_point_summary.json` | Operating-point diagnostic cosine 0.99974 / rel L2 0.02255; preregistered full-vector readout gate did not close (frozen-boundary-evidence) | NEGATIVE RESULT (formal gate; boundary evidence retained) |
| E0 p2_state_local_risk | Natural-amplitude full-vector state-local risk reconstruction | E0 | `configs/frozen/p2_state_local_config.yaml` | `experiments/p2_state_local_risk/scripts/run_p2.py` | `experiments/p2_state_local_risk/` | Prerequisite gate failed; full-vector reconstruction unclosed; descriptive ranking signal motivated finite-action recovery | NEGATIVE RESULT |
| E0 p2_recovery | Finite-action path recovery; controlled scalar decision risk | E0 | `experiments/p2_recovery/r{0,1,3,4}*/…_config.yaml` | `experiments/p2_recovery/scripts/run_r0_r1.py`, `run_r3.py`, `analyze_r4.py` | `experiments/p2_recovery/r4_scalar_decision_risk/results/{evaluation,replication}/analysis_summary.json` | R4 two-midpoint scalar risk: Spearman 1.0, top-1 gain 1.0 in both evaluation and replication; R1 trust region cosine 0.99986@1/16 → 0.95463@1. R2 preregistered but not run (not a negative) | VALID (frozen-recovery-evidence) |
| E0 p3_decision_validity | Does controlled score transfer to physical histories? | E0 | `experiments/p3_decision_validity/p3_config.yaml` | `experiments/p3_decision_validity/scripts/run_p3_trajectory.py` | `experiments/p3_decision_validity/` | Controlled score does not transfer to propagated all-layer physical histories; detector/minimal-refresh/prefilter gates did not jointly close; event table reused by R1 | NEGATIVE RESULT |
| E0 p3_physical_recovery | Same-state physical singleton target; teacher-level ranking | E0 | `experiments/p3_physical_recovery/p3pr_config.yaml` | `experiments/p3_physical_recovery/scripts/run_p3pr.py` | `experiments/p3_physical_recovery/` | Clone-based same-state exact-KL target passes integrity; dense current-state mechanism sufficient; boundary-27 teacher closes 8-candidate ranking (high-cost, non-deployable) | VALID (frozen-recovery-evidence) |
| E0 p3pr_generalization | Cross-model/task generalization of P3PR mechanism | E0 | `experiments/p3pr_generalization/p3pr_generalization_config.yaml` | `experiments/p3pr_generalization/scripts/run_generalization.py` | `experiments/p3pr_generalization/results/analysis/analysis_summary.json` | Dense mechanism replicates (Spearman 1.0 formal / 0.9940 replication, top-1 1.0); fixed relative late-boundary rule fails fresh formal generalization | VALID (limited scope; boundary rule NEGATIVE) |

---

## 1. Training-free estimators P0–P5 (E1, stored-trajectory / dev-pilot screens)

| ID | Research question / hypothesis | Substrate | Config | Runner script | Result location | Main result | Status |
|---|---|---|---|---|---|---|---|
| P0 tf_sketch | Fixed-decay Euclidean recursive sketch of layer-27 action vectors improves ranking | E1, 24 stored trajectories | `configs/stages/training_free_sketch_config.yaml` | `scripts/analyze_training_free_sketch.py` | `results/…/statekv_tf_sketch_p0_v1/{summary.json,metrics.csv}` | Median Spearman gain −0.0157 (eval) / −0.0217 (replication) at 64-dim ρ=0.95; negative across dims 16–128, ρ 0.5–1.0 | NEGATIVE RESULT (decisive) |
| P1 metric_repair | Unlabeled diagonal-RMS (+EMA) metric scaling repairs the decision metric | E1, stored trajectories | `configs/stages/training_free_metric_repair_config.yaml` | `scripts/analyze_metric_repair.py` | `results/…/statekv_tf_metric_repair_p1_v1/{summary.json,metrics.csv}` | Median Spearman +0.0096 (eval) / +0.0122 (replication) but eval normalized regret worsens −0.0063 → gate veto | NEGATIVE RESULT (marginal/veto-style; retest target) |
| P2 shared_jvp | Rank-4 shared randomized Fisher pullback (32-dim sketch, refresh 4) repairs ranking | E1, 2 dev seqs × 8 candidates | `configs/stages/shared_jvp_pilot_config.yaml` | `scripts/run_shared_jvp_pilot.py` | `results/…/statekv_shared_jvp_pilot_p2_v1/{summary.json,metrics.csv}` | Median Spearman gain 0.0; pairwise −0.0281; regret −0.0432. MLX forward-mode could not differentiate `Sum` → symmetric finite differences used (documented) | NEGATIVE RESULT (decisive) |
| P3 vjp_routes (pilot) | Output-side VJP controller metric is numerically valid | E1, dev | `configs/stages/vjp_routes_pilot_config.yaml` | `scripts/run_vjp_routes_pilot.py` | `results/…/statekv_vjp_routes_p3_v1/` | Adjoint identity holds (max rel. err 4.5e-4); numerics validated | VALID (numerics only) |
| P3 vjp_routes stress | Predeclared Gaussian 16-dir VJP route improves ranking; post-hoc Rademacher variant | E1, 4 dev seqs | `configs/stages/vjp_routes_stress_config.yaml` | `scripts/run_vjp_routes_pilot.py` | `results/…/statekv_vjp_routes_p3_stress_v1/{summary.json,metrics.csv}` | Predeclared route: regret −0.0638, pairwise −0.0153, Spearman 0.0 → fails. Post-hoc Rademacher 8-dir: regret 0.1945→0.0696 but pairwise −0.0026 (never independently replicated at the time) | NEGATIVE RESULT (predeclared, decisive); Rademacher variant EXPLORATORY → later retested negative (group 5) |
| P4 direct_coreset (pilot) | Direct four-query mean-contribution selector, zero candidate rollouts | E1, 4 dev seqs screen | `configs/stages/direct_coreset_pilot_config.yaml` | `scripts/run_direct_coreset_pilot.py` | `results/…/statekv_direct_coreset_p4_v1/` | Locked the four-query contribution selector on the dev screen | VALID (dev screen) |
| P4 direct_coreset replication | Held-out local replication of the locked selector + merge/tier diagnostics | E1, 8 held-out seqs | `configs/stages/direct_coreset_replication_config.yaml` | `scripts/run_direct_coreset_pilot.py` | `results/…/statekv_direct_coreset_p4_replication_v1/summary.json` | Local projected error 0.0978→0.0709, 70.8% matched-unit wins; nearest-value merge beats hard deletion in 192/192 units; 2/3/4-bit cold-V tier at 23.1/28.6/34.0% of FP16 storage | VALID (held-out local replication positive) |
| P5 multiboundary_vjp | Post-attention VJP sum at layers 0/14/27 (width 8, refresh 4) | E1, 4 dev seqs | `configs/stages/multiboundary_vjp_pilot_config.yaml` | `scripts/run_multiboundary_vjp_pilot.py` | `results/…/statekv_post_multiboundary_vjp_p5_v1/{summary.json,metrics.csv}` | Regret gain −0.0669, pairwise −0.0587, Spearman −0.0833 at ~6 reverse passes/token; action reconstruction err < 7.2e-4 | NEGATIVE RESULT (decisive; dominated on accuracy and cost) |
| — multi-boundary direct-policy distillation | Distill multi-boundary teacher into direct policy | — | — | — | no run dir | Not run as a gated phase | SUPERSEDED by the training-free direct-policy line (P4→P6) |

---

## 2. Direct-policy replay line P6–P24 (E1)

| ID | Research question / hypothesis | Substrate | Config | Runner script | Result location | Main result | Status |
|---|---|---|---|---|---|---|---|
| P6 replay | Shared 92-token contribution core applied to all 28 layers lowers teacher-forced KL | E1, 8 held-out seqs | `configs/stages/direct_policy_replay_config.yaml` | `scripts/run_direct_policy_replay.py` (+ `analyze_direct_policy_replay.py`) | `results/…/statekv_direct_policy_replay_p6_v1/summary.json` | Mean KL 0.0485→0.0199, P95 0.1793→0.1576, max 1.6502→0.1692; 6/8 sequences improve | VALID (held-out dev, teacher-forced) |
| P7 multianchor | Pure contribution policy across 3 anchors on new sequences | E1, 12 seqs × 3 anchors | `configs/stages/direct_policy_independent_multianchor_config.yaml` | `scripts/run_direct_policy_replay.py` | `results/…/statekv_direct_policy_independent_multianchor_p7_v1/summary.json` | Mean KL 0.3293→0.2569, P95 1.5670→1.0743, 9/12 seq wins; NIAH task mean 0.15993→0.16061 (+0.0007) fails all-task-means gate | NEGATIVE RESULT (marginal veto; retested, group 5) |
| P8 shrinkage screen | Development selection of shrinkage coefficient λ | E1, dev | `configs/stages/direct_policy_shrinkage_screen_config.yaml` | `scripts/run_direct_policy_replay.py` | `results/…/statekv_direct_policy_shrinkage_screen_p8_v1/summary.json` | λ=0.25 blend selected and locked | VALID (dev screen) |
| P9 shrinkage independent | Locked λ=0.25 blend on 12 new sequences | E1, 12 seqs × 3 anchors | `configs/stages/direct_policy_shrinkage_independent_config.yaml` | `scripts/run_direct_policy_replay.py` | `results/…/statekv_direct_policy_shrinkage_independent_p9_v1/summary.json` | Mean KL 0.3577→0.3298, P95 1.9535→1.5733, both task means improve; sample-anchor win rate 17/36 = 47.2% < locked 55% | NEGATIVE RESULT (distributional veto; retested, group 5) |
| P10 runtime profile | Is scheduled sparse capture + CPU scoring fast enough? | E1 microbenchmark | `configs/stages/direct_policy_runtime_profile_config.yaml` | `scripts/profile_direct_policy_runtime.py` | `results/…/statekv_direct_policy_runtime_profile_p10_v3/summary.json` | Capture hook adds 0.31 ms to 18.89 ms decode step; ~0.43 ms/step amortized at 16-step refresh; continuous rolling rejected. **Only `_v3` is stored** (no _v1/_v2 dirs exist) | VALID (microbenchmark, not end-to-end) |
| P11 trigger screen | Develop a fire-only-when-drifting trigger | E1, dev | `configs/stages/direct_policy_trigger_screen_config.yaml` | `scripts/run_direct_policy_trigger.py` | `results/…/statekv_direct_policy_selective_trigger_screen_p11_v1/summary.json` | Score total-variation rule with threshold 0.24735 selected and locked | VALID (dev screen) |
| P12 trigger independent | Locked TV trigger on P9 replays | E1, 6 activated of 36 units | `configs/stages/direct_policy_trigger_independent_config.yaml` | `scripts/run_direct_policy_trigger.py` | `results/…/statekv_direct_policy_selective_trigger_independent_p12_v1/summary.json` | Mean KL 0.35766→0.34125, P95 →1.74473, 94.4% nonworse; activated-unit win rate 50% < 60%, NIAH mean −3.8e-5 | NEGATIVE RESULT (noise-scale veto) |
| P13 tail risk | Fixed shrinkage under a CVaR tail-risk gate, third sequence set | E1, 12 new seqs | `configs/stages/direct_policy_tail_risk_independent_config.yaml` | `scripts/run_direct_policy_replay.py` | `results/…/statekv_direct_policy_tail_risk_independent_p13_v1/summary.json` | Mean KL 0.52735→0.51766, P95 3.62435→2.77128; CVaR95 6.57390→6.63419 worse; KL≥1 11.63%→11.81% (removes 4, creates 5) | NEGATIVE RESULT (one-step migration veto) |
| P14 protected rescue | Top-attention protection + m∈{4,8,16} contribution rescue slots | E1, 6 dev seqs | `configs/stages/direct_policy_protected_rescue_screen_config.yaml` | `scripts/run_direct_policy_replay.py` (+ `analyze_protected_rescue_screen.py`) | `results/…/statekv_direct_policy_protected_rescue_screen_p14_v1/` | No m passes all 6 dev constraints; best m=8: mean KL 0.29580→0.29367 but CVaR95 3.2221→3.2312, max KL 5.795→6.315 worse. Independent split never touched | NEGATIVE RESULT (dev-only) |
| P15 signal family screen | Which of six fixed signal families survives a joint screen? | E1, 6 dev seqs | `configs/stages/direct_policy_signal_family_screen_config.yaml` | `scripts/run_direct_policy_replay.py` (+ `analyze_signal_family_screen.py`) | `results/…/statekv_direct_policy_signal_family_screen_p15_v1/` | Only 4-step temporal attention volatility passes all 6 constraints (mean KL 0.36262→0.30730); head-peak close (0.319); five families rejected without parameter search | VALID (dev screen; promoted TV to P16) |
| P16 temporal volatility independent | Frozen TV policy, aggregate tail-risk gate, untouched seqs | E1, 12 seqs | `configs/stages/direct_policy_temporal_volatility_independent_config.yaml` | `scripts/run_direct_policy_replay.py` | `results/…/statekv_direct_policy_temporal_volatility_independent_p16_v1/summary.json` | Mean KL 0.52419→0.47115, P95 3.02075→2.85653, CVaR95 5.33994→4.98184, max 13.19648→12.84552, KL≥1 13.72%→12.33%; 6/12 seq wins, bootstrap crosses zero | VALID (aggregate-risk gate passed; not per-sequence superiority) |
| P17 TV runtime profile | Cost of the rolling TV state | CPU arithmetic microbenchmark | `configs/stages/temporal_volatility_runtime_profile_config.yaml` | `scripts/profile_temporal_volatility_runtime.py` | `results/…/statekv_temporal_volatility_runtime_profile_p17_v1/summary.json` | 96 B per context token; 0.25 ms/step update and 2.82 ms/refresh at 32K tokens (excludes capture and end-to-end latency) | VALID (narrow scope) |
| P18 TV freegen | Frozen TV policy in matched-budget free generation | E1, 6 paired samples | `configs/stages/temporal_volatility_freegen_protocol.yaml` | `scripts/analyze_temporal_volatility_freegen.py` | `results/…/statekv_temporal_volatility_freegen_p18_v1/summary.json` | GovReport ROUGE-L 7.31→7.95, RULER 100=100, throughput 95.7%; overall NLL 1.68318→1.68692 (+0.00374) → frozen primary gate fails. Capture peaks at 3.19× full-cache memory | NEGATIVE RESULT (flagship veto case; retested, group 5) |
| P19 geometry screen | Attention-free static geometry scores (KNorm/KeyDiff/VNormL2) | E1, 4 dev samples/method | `configs/stages/attention_free_geometry_screen_protocol.yaml` | `scripts/analyze_attention_free_geometry_screen.py` | `results/…/statekv_attention_free_geometry_screen_p19_v1/summary.json` | All miss both RULER needles; throughput 35–41% of random control; KeyDiff NLL gain not replication-eligible | NEGATIVE RESULT (decisive) |
| P20 static lexical screen | Attention-free token-rarity selector, dev screen | E1, dev | `configs/stages/static_lexical_screen_protocol.yaml` | `scripts/analyze_static_lexical_screen.py` | `results/…/statekv_static_lexical_screen_p20_v1/summary.json` | GovReport ROUGE-L 10.88 vs 9.60 random; RULER 2/2 vs 0/2 controls; authorizes replication | VALID (dev screen) |
| P21 token rarity replication | Independent replication of token rarity | E1, 6 untouched samples | `configs/stages/token_rarity_replication_protocol.yaml` | `scripts/analyze_token_rarity_replication.py` | `results/…/statekv_token_rarity_replication_p21_v1/summary.json` | RULER 3/3 needles, throughput 1.252×, peak memory 0.313× of latest attention; GovReport ROUGE-L 8.09 vs 8.92, NLL +0.00334 → cross-task gate fails | NEGATIVE RESULT (retrieval-specific success; retested, group 5) |
| P22 proxy alignment | One additive proxy for selection + refresh ordering (7-signal dev audit) | E1, dev | `configs/stages/proxy_alignment_protocol.yaml` | `scripts/run_proxy_alignment.py` | `results/…/statekv_risk_consistent_proxy_alignment_p22_v1/summary.json` | Latest attention only signal passing joint screen: action Spearman 0.786, regret 0.152, refresh-benefit 0.378 | VALID (dev audit) |
| P23a independent source | New-sequence physical replay source for P23b | E1, 6 new seqs | `configs/stages/proxy_alignment_independent_source_protocol.yaml` | `scripts/run_direct_policy_replay.py` | `results/…/statekv_risk_consistent_proxy_independent_source_p23a_v1/summary.json` | Source trajectories/replays generated for the frozen-proxy audit | VALID (infrastructure run) |
| P23b proxy independent | Frozen latest-attention proxy on untouched sequences | E1, 6 seqs | `configs/stages/proxy_alignment_independent_protocol.yaml` | `scripts/run_proxy_alignment.py` | `results/…/statekv_risk_consistent_proxy_independent_p23b_v1/summary.json` | Action alignment replicated (median Spearman 0.750, regret 0.118); refresh-benefit Spearman reverses +0.378→−0.350 → joint gate fails | NEGATIVE RESULT (refresh axis; action axis VALID) |
| P24 output-aware proxy | Attention–Value contribution cost as unified proxy | E1, P7 seqs reused, dev | `configs/stages/proxy_alignment_output_aware_protocol.yaml` | `scripts/run_proxy_alignment.py` | `results/…/statekv_risk_consistent_output_aware_proxy_p24_v1/summary.json` | Contribution refresh-benefit Spearman 0.011 vs 0.160 latest attention; both proxies fail the dev gate | NEGATIVE RESULT (decisive; exploratory dev scope) |

---

## 3. Oracle / cheap-controller line P25–P35 (P25–P30 on E1-era tasks; P31+ E2)

| ID | Research question / hypothesis | Substrate | Config | Runner script | Result location | Main result | Status |
|---|---|---|---|---|---|---|---|
| P25 closed loop | Physical-oracle closed loop with cold recovery: mechanics | E1, dev, 8 sample–strategy runs | `configs/stages/oracle_closed_loop_protocol.yaml` | `scripts/run_oracle_closed_loop.py` (+ `analyze_oracle_closed_loop.py`) | `results/…/statekv_physical_oracle_closed_loop_p25_v1/summary.json` | All 8 runs complete 3 control cycles; budget and state continuity preserved; 7 post-initial refreshes with 7 cold recoveries | VALID (dev mechanics closure) |
| P26 closed loop independent | Same loop on untouched sequences | E1, 4 seqs, 16 loops | `configs/stages/oracle_closed_loop_independent_protocol.yaml` | `scripts/run_oracle_closed_loop.py` | `results/…/statekv_physical_oracle_closed_loop_independent_p26_v1/summary.json` | 16/16 loops complete; 21 refreshes/recoveries; dense risk median Spearman 0.957, top-1 93.75% vs exact KL; selected exact KL 0.3281 vs stale 0.3970 | VALID (independent mechanics + risk ranking) |
| P27 comparison | Teacher-forced StateKV vs fixed policies, each owning its history | E1, dev | `configs/stages/oracle_policy_comparison_protocol.yaml` | `scripts/run_oracle_policy_comparison.py` | `results/…/statekv_oracle_policy_comparison_p27_v1/summary.json` | Development teacher-forced risk superiority passed | VALID (dev) |
| P28 comparison independent | Same comparison on untouched sequences | E1, 4 seqs | `configs/stages/oracle_policy_comparison_independent_protocol.yaml` | `scripts/run_oracle_policy_comparison.py` (+ `analyze_oracle_policy_comparison.py`) | `results/…/statekv_oracle_policy_comparison_independent_p28_v1/summary.json` | StateKV mean KL 0.1482 vs attention 0.3046 / SnapKV 0.7247 / H2O 0.5196; wins every sample-level comparison | VALID |
| P29 freegen H=8 | Exact-risk teacher, 8-token control horizon, free generation | E1, 2 dev samples | `configs/stages/oracle_policy_freegen_protocol.yaml` | `scripts/run_oracle_policy_freegen.py` | `results/…/statekv_oracle_policy_freegen_p29_v1/summary.json` | Trajectory KL 0.2394 vs SnapKV 0.2253 (paired −0.0142) → loses; beats attention (0.5843) and H2O (0.3879) | NEGATIVE RESULT (one-baseline veto) |
| P29b freegen H=1 | Per-token control horizon | E1, dev | `configs/stages/oracle_policy_freegen_h1_protocol.yaml` | `scripts/run_oracle_policy_freegen.py` | `results/…/statekv_oracle_policy_freegen_h1_p29b_v1/summary.json` | H=1 joint risk/quality gate passed (only horizon that passes on dev) | VALID (dev) |
| P29c freegen H=4 | 4-token horizon | E1, 2 dev samples | `configs/stages/oracle_policy_freegen_h4_protocol.yaml` | `scripts/run_oracle_policy_freegen.py` | `results/…/statekv_oracle_policy_freegen_h4_p29c_v1/summary.json` | KL 0.26603 vs SnapKV 0.26110 — risk gate missed by 0.005; best task quality of any arm (+2.07 official vs SnapKV) | NEGATIVE RESULT (closest near-miss) |
| P30 freegen independent | Frozen H=1 teacher on 4 new greedy-generation seqs | E1, 4 seqs | `configs/stages/oracle_policy_freegen_independent_protocol.yaml` | `scripts/run_oracle_policy_freegen.py` | `results/…/statekv_oracle_policy_freegen_independent_p30_v1/summary.json` | Mean KL 0.1391 (−66.5%/−33.2%/−65.4% vs attention/SnapKV/H2O) — KL gate passed; NIAH 0/2 for all compressed policies (full cache 2/2), GovReport trails → task-quality gate fails | NEGATIVE RESULT (two-sample task veto) |
| P31 freegen Qwen3-8B | Per-token exact-risk teacher at 8B scale, budget 256 | E2, 10 seqs | `configs/stages/oracle_policy_freegen_qwen3_8b_n10_protocol.yaml` | `scripts/run_oracle_policy_freegen.py` | `results/…/statekv_oracle_policy_freegen_qwen3_8b_n10_p31_v1/summary.json` | Mean KL 0.05057 vs 0.33572/0.13250/0.54677 (Attention/SnapKV/H2O); NIAH 5/5; GovReport competitive. Later shown to use persistent full-KV backing re-anchor (not strict pure eviction) — see group 4 R0 | VALID (bounded; machinery caveat measured in R0) |
| P32 cheap freegen | Training-free direct cheap controllers, zero candidate rollouts | E2, 10 samples | `configs/stages/cheap_policy_freegen_qwen3_8b_n10_protocol.yaml` | `scripts/run_cheap_policy_freegen.py` | `results/…/statekv_cheap_policy_freegen_qwen3_8b_n10_p32_v1/summary.json` | A2 temporal volatility KL 0.09525, beats Attention 10/10; B3 dynamic layer budgets KL 0.114995, best task-score point estimate; A3 collapses to Attention in all 640 decisions; B1 KL 0.18960 (~2× A2); A4 flat | VALID (screen positive; A3/B1/A4 NEGATIVE diagnostics) |
| P33 budget calibration | Anchor calibration reproduces P32-B3 operating point | E2 | `configs/stages/statekv_p1_p3_gates_qwen3_8b.yaml` (calibration stage) | `scripts/run_statekv_gates.py calibration` | `results/…/statekv_budget_calibration_qwen3_8b_p33_v1/summary.json` | Anchor reproduces P32-B3 | VALID |
| P34 dynamic budget mechanism | Does state-dependent per-layer budgeting beat static/misaligned controls? | E2, 10 samples (86–90) | `configs/stages/statekv_p1_p3_gates_qwen3_8b.yaml` (p1 stage) | `scripts/run_statekv_gates.py p1` | `results/…/statekv_dynamic_budget_mechanism_qwen3_8b_p34_v1/{p1_analysis.json,aggregate_results.csv}` | Dynamic loses to layer-shuffled static (mean KL +0.0270, CVaR95 +0.4417, 2/10 wins) and to stale B3 (+0.0111/+0.1461) → mechanism refuted; uniform budget retained | NEGATIVE RESULT (mechanism-level refutation) |
| P35 pure eviction | Strict pure-eviction (irreversible) chassis mechanics, no CPU backing store | E2, budgets 192/156 and 256/220 | `configs/stages/statekv_p1_p3_gates_qwen3_8b.yaml` (p2 stage) | `scripts/run_statekv_gates.py p2` | `results/…/statekv_pure_eviction_qwen3_8b_p35_v1/{summary.json,aggregate_results.csv}` | Irreversible set inclusion holds at both budgets; mechanics passed; analysis summary stored | VALID |
| — tail telemetry P36 | Tail telemetry phase | — | — | — | no run dir | Never run | SUPERSEDED by the selective-refresh mainline (R0–R2) before execution |

---

## 4. Gate / closure program (E2 unless noted)

| ID | Research question / hypothesis | Substrate | Config | Runner script | Result location | Main result | Status |
|---|---|---|---|---|---|---|---|
| G0/G1 teacher gate | Does a per-cycle min-exact-KL one-step teacher have headroom over cheap policies under strict pure eviction? | E2 pure eviction, 10 samples | `configs/stages/statekv_teacher_gate_g0.yaml` | `scripts/run_statekv_gates.py teacher-gate` | `results/…/statekv_teacher_gate_qwen3_8b_g0_v1/`; `analysis/tables/gate0_*`, `gate1_*` | Gate 0 NO_HEADROOM: teacher KL 0.2322 vs b2_uniform 0.0961, paired 2/10, step p95 1.094 vs 0.412; Gate 1 ACTION_SPACE_DOMINANT: oracle regret 1.7%, 61.6% cycles tied | NEGATIVE RESULT (structural closure) |
| Teacher gate substrate B | Same one-step teacher on the P23b shared-mask substrate | E1 (P23b split, 6 samples) | `configs/stages/statekv_teacher_gate_p23b.yaml` | `scripts/run_statekv_gates.py teacher-gate` | `results/…/statekv_teacher_gate_qwen25_15b_p23b_v1/` | Teacher wins 1/6 vs best cheap; NIAH 0.0 for teacher AND every cheap policy → operating point quality-invalid; no headroom | NEGATIVE RESULT (operating point INVALID for all compressed policies) |
| P2 cheap panel, P23b substrate | Cheap-policy panel mechanics on the substrate-B operating point | E1 (P23b substrate) | `configs/stages/statekv_p2_p23b_cheap.yaml` | `scripts/run_statekv_gates.py p2` | `results/…/statekv_p2_qwen25_15b_p23b_cheap_v1/summary.json` | Execution valid; irreversible inclusions hold at 128/92; supplies the cheap arms for the substrate-B gate | VALID (supporting run) |
| Ladder 2B deep risk | Is deep (h∈{1,2,4}) teacher-forced risk discriminative across panel actions? | E2 pure eviction, 10 fresh samples (101–105) | `configs/stages/statekv_ladder_2b.yaml` | `scripts/run_statekv_gates.py ladder` | `results/…/statekv_ladder_qwen3_8b_2b_v1/`; `analysis/statekv_ladder_2b_deep_risk.md`; `analysis/tables/ladder_2b_*` | DEEP_RISK but non-discriminative: attention family tied at every horizon (mean step KL 0.053@h1, 0.28@h4; regret ~0.0006–0.0013); only uniform separates with depth (0.08→0.49); 55.4% cycles tied at h1; 101/160 cycles valid pre-shift | BUGGED → corrected (committed probe KLs were a different-input metric inflated to 10–46 after a one-token phase shift; the corrected pre-shift, same-input analysis stands and is what the closure doc uses) |
| Marginal 2C additivity | Are one-step boundary-swap marginals informative (additive top-k)? | E2, same ladder run (marginal block in config) | `configs/stages/statekv_ladder_2b.yaml` (`marginal:` block) | `scripts/run_statekv_gates.py ladder` | `analysis/tables/ladder_2b_*`; closure doc above | One-step marginals ~1e-5–1e-4 for every token class; pair interactions exactly 0 → flat at depth 1, information-free | NEGATIVE RESULT (measurement closure) |
| Refresh arms 101–105 | Is any selective/fixed refresh better than always-refresh at budget 256? | E2 pure eviction, 10 fresh NIAH samples | `configs/stages/statekv_refresh_arms_qwen3_8b_768_256.yaml` | `scripts/run_statekv_gates.py r2b-gate` | `results/…/statekv_refresh_arms_qwen3_8b_768_256_v1/`; `analysis/tables/refresh_arms_summary.*` | Every-refresh best on all 10 samples (mean KL 0.024 vs never 0.346, attention arm); NO_CLEAR_REFRESH_ADVANTAGE for not refreshing | NEGATIVE RESULT (no selective-refresh gap at budget 256) |
| Refresh-gap decomposition R0 | Is there a structured, triggerable staleness gap offline? | E1 P23b events + E2 labels | — (offline analysis over stored parquets) | `scripts/analyze_refresh_gap_decomposition.py` | `analysis/tables/refresh_gap_decomposition_summary.csv`; `results/…/statekv_risk_consistent_proxy_independent_p23b_v1/refresh_regret_rows.parquet` | Staleness gap is structured offline → trigger design justified (premise for R1/R2) | VALID (offline decomposition) |
| Trigger prescreen R1 | Does any offline feature predict refresh benefit online? | Offline over decomposition records | — | `analysis/tables/build_trigger_feature_screen.py`, `fit_refresh_trigger.py` | `analysis/tables/trigger_screen_report.md` | No offline feature transfers online → instrumentation required | NEGATIVE RESULT |
| Selective refresh labels R2a v1 | Collect online refresh labels at 768 ctx / 256 budget | E2, 10 samples | `configs/stages/statekv_selective_refresh_r2a.yaml` | `scripts/run_statekv_gates.py r2a-labels` | `results/…/statekv_selective_refresh_labels_r2a_v1/refresh_event_rows.parquet` | Degenerate operating point: 256 of 768 scores time-invariant → no staleness signal at this operating point | NEGATIVE RESULT (evidence retained) |
| Selective refresh labels R2a v2 | More aggressive operating point (128 of 768) | E2 | `configs/stages/statekv_selective_refresh_r2a_v2.yaml` | `scripts/run_statekv_gates.py r2a-labels` | `results/…/statekv_selective_refresh_labels_r2a_v2/partial_*.parquet` (no summary.json; early-stopped) | Early-stopped; partial rows show degenerate operating point (128 of 768) | NEGATIVE RESULT (partial/early-stopped; evidence retained) |
| Selective refresh labels R2a v3 | 4k-ctx / 128-budget operating point | E2 | `configs/stages/statekv_selective_refresh_r2a_v3.yaml` | `scripts/run_statekv_gates.py r2a-labels` | `results/…/statekv_selective_refresh_labels_r2a_v3/partial_*.parquet` (no summary.json) | Degenerate operating point; NIAH 0.0 → quality-invalid at this operating point | NEGATIVE RESULT (partial; quality-invalid) |
| Selective refresh trigger R2 | Is a selective-refresh trigger viable on Qwen3-8B per-layer selection? | E2 labels + E1 P23b contrast | — (offline analysis over R2a labels) | `analysis/tables/` trigger fits | `analysis/tables/selective_refresh_negative_result_r2.md`, `refresh_operating_point_comparison.csv`, `refresh_trigger_no_freeze.json` | Rankings time-invariant at every quality-valid operating point (coverage 0.9979/0.9969/0.9952, lag-identical 100%); NO_FREEZE at both fits; P23b contrast substrate coverage 0.698, 77.1% positive benefit → phenomenon real but substrate-bound | NEGATIVE RESULT (refuted on Qwen3-8B) |
| Recoverable R0 teacher | Does the physical-risk teacher have headroom under recoverable (KVBackingStore) semantics? | E2 recoverable, 10 samples | `configs/stages/statekv_recoverable_r0_qwen3_8b.yaml` | `scripts/run_oracle_policy_freegen.py` (recoverable mode; `statekv/oracle_policy_freegen.py`) | `results/…/statekv_recoverable_r0_qwen3_8b_v1/`; `analysis/statekv_recoverable_r0_results.md`; `analysis/tables/recoverable_r0_*` | Teacher 0.0213 vs qk_pool 0.0086 (ratio 2.47), paired 0/10, tail 2.2× worse, scorer residual D3 = −0.0127 → no headroom; explains P31 gain as machinery artifact; qk_pool is the strongest working-set policy (KL 0.0086, NIAH 1.0) | NEGATIVE RESULT (teacher); VALID (qk_pool strongest arm; P31-artifact explanation) |
| QK–V decomposition battery | Is there V-routing residual given QK? What dominates the QK dynamic range? | E2 recoverable, 25M token rows | `configs/stages/statekv_qkv_decomposition_qwen3_8b.yaml` | `scripts/run_qkv_decomposition.py` | `results/…/statekv_qkv_decomposition_qwen3_8b_v1/`; `analysis/statekv_qkv_discovery_results.md`; `analysis/tables/qkv_*.csv` | V residual refuted: partial Spearman −0.05..−0.10 in every cutoff bucket, 288-swap exact oracle median regret 2e-15 (92% flat), no layer/head/token/horizon pocket. QK dynamic range supported: var log-attention 1.3–4.5 vs var log-projected-V 0.01–0.14 | NEGATIVE RESULT (V residual, decisive); VALID (QK dynamic-range mechanism) |
| qk_tiered_v gate 256t | Premise: QK routing + 4-bit cold-V tier at budget 256 | E2 recoverable, 10 samples | `configs/stages/statekv_qkvtier_gate_256t.yaml` | `scripts/run_oracle_policy_freegen.py` (tier mode) | `results/…/statekv_qkvtier_gate_256t_v1/summary.json` | Premise P passes at 256: ratio 0.944, 6/10 paired wins | VALID (premise gate) |
| qk_tiered_v gate 352f | FP16-352 coverage control (1.375× memory) | E2 recoverable | `configs/stages/statekv_qkvtier_gate_352f.yaml` | `scripts/run_oracle_policy_freegen.py` | `results/…/statekv_qkvtier_gate_352f_v1/summary.json` | Coverage is the binding constraint: qk-pool 352 FP16 KL = 0.499× of 256 baseline, 10/10 wins, tail better | VALID (control arm; coverage claim supported) |
| qk_tiered_v gate 352t | Memory-matched tiered-352 vs FP16-352 (G5) | E2 recoverable | `configs/stages/statekv_qkvtier_gate_352t.yaml` | `scripts/run_oracle_policy_freegen.py` | `results/…/statekv_qkvtier_gate_352t_v1/summary.json`; `analysis/statekv_qkvtier_gate.md` | G5-only veto: tiered KL 0.004845 vs fp16 0.004304, ratio 1.126 > preregistered 1.10 → NO_GO (TIERING_LOSSY); G1–G4 all pass (0.562× baseline KL, 10/10, p95 0.48×). Comparator deliberately unequal-memory | NEGATIVE RESULT (unequal-memory G5 veto; retested matched-budget, group 5) |
| Open stress 768/128 | Does qk_pool track the full cache under coverage stress? | E2 recoverable, 768 ctx, budget 128 | `configs/stages/statekv_openstress_768_128.yaml` | `scripts/run_oracle_policy_freegen.py` | `results/…/statekv_openstress_768_128_v1/sample_results.csv` | qk_pool tracks full cache down to 8% coverage, NIAH 1.0; KL scales ~3× per budget halving; quest_like 2.8–3.0× qk_pool KL at 256/128 | VALID (coverage stress) |
| Open stress 768/64 | Same at budget 64 | E2 recoverable, budget 64 | `configs/stages/statekv_openstress_768_64.yaml` | `scripts/run_oracle_policy_freegen.py` | `results/…/statekv_openstress_768_64_v1/sample_results.csv` | quest_like collapses: KL 0.7065 vs qk_pool 0.0819, NIAH 0.8 (first task-level failure); page-max recall upper bound 0.674–0.543 @ p4–p32 | NEGATIVE RESULT (quest_like page approximation) |
| Open corner h4 | Slow-refresh corner, horizon 4 cadence, budget 64 | E2 recoverable | `configs/stages/statekv_opencorner_768_64_h4.yaml` | `scripts/run_oracle_policy_freegen.py` | `results/…/statekv_opencorner_768_64_h4_v1/sample_results.csv` | h4 corner arm measured; see corner gate doc | VALID (boundary condition) |
| Open corner h16 | Horizon-16 cadence at budget 64 | E2 recoverable | `configs/stages/statekv_opencorner_768_64_h16.yaml` | `scripts/run_oracle_policy_freegen.py` | `results/…/statekv_opencorner_768_64_h16_v1/sample_results.csv` | qk_pool@h16 catastrophic at 64: KL 0.8439, NIAH 0 → slow refresh unsafe at tight coverage | NEGATIVE RESULT (cadence cliff) |
| Corner obswin h16 | SnapKV-style observation-window scoring at the h16 corner | E2 recoverable | `configs/stages/statekv_corner_obswin_768_64_h16.yaml` | `scripts/run_oracle_policy_freegen.py` | `results/…/statekv_corner_obswin_768_64_h16_v1/sample_results.csv`; `analysis/statekv_corner_gate.md` | KL 1.0748 vs qk_pool@h16 0.8439; NIAH 0.0 both; paired 2/10 → NO_GO_CORNER (backward-looking window is stale-biased; fix is cadence, not scoring) | NEGATIVE RESULT (preregistered corner gate) |
| Headwise probe HF4 | Per-KV-head own-top-k selection at identical total budget | E2, offline probe, 3 samples × 36 layers × 8 KV heads | `configs/stages/statekv_headwise_probe_qwen3_8b.yaml` | `scripts/run_qkv_decomposition.py` | `results/…/statekv_headwise_probe_qwen3_8b_v1/headwise_rows.parquet`; `analysis/tables/open_hf4_*` | Captured-mass gain +0.96pp (0.9768 vs 0.9671), p95 +3.6pp, concentrated in diffuse early layers — below pre-committed action threshold; literature-covered (Ada-KV/HeadKV/KV-Compress) → closed without implementation | NEGATIVE RESULT (threshold veto on a measured positive) |
| Extval 3072/256 | Does the qk_pool closure hold at 4.7k context? | E2, 3072 ctx, budget 256 (h1) | `configs/stages/statekv_extval_3072_256.yaml` | `scripts/run_oracle_policy_freegen.py` | `results/…/statekv_extval_3072_256_v1/sample_results.csv`; `analysis/statekv_external_validity_report.md` | Task-perfect down to 1.4% coverage; NIAH 3/3; only KL degradation, coverage-driven | VALID |
| Extval 3072/64 | Same at budget 64 | E2, budget 64 | `configs/stages/statekv_extval_3072_64.yaml` | `scripts/run_oracle_policy_freegen.py` | `results/…/statekv_extval_3072_64_v1/sample_results.csv` | Coverage-stress arm at 3072; feeds cliff/coverage analysis | VALID |
| Extval 3072/64 h4 | Cadence cliff at 3072, h4 | E2 | `configs/stages/statekv_extval_3072_64_h4.yaml` | `scripts/run_oracle_policy_freegen.py` | `results/…/statekv_extval_3072_64_h4_v1/sample_results.csv` | Cadence cliff reproduces at 3072 core 28: NIAH 1/3 | VALID (boundary) |
| Extval 3072/64 h16 | Cadence cliff at 3072, h16 | E2 | `configs/stages/statekv_extval_3072_64_h16.yaml` | `scripts/run_oracle_policy_freegen.py` | `results/…/statekv_extval_3072_64_h16_v1/sample_results.csv` | Cliff family at core 28 (h16) | VALID (boundary) |
| Extval 3072/256 h4 | Cadence arm at full core 220, h4 | E2 | `configs/stages/statekv_extval_3072_256_h4.yaml` | `scripts/run_oracle_policy_freegen.py` | `results/…/statekv_extval_3072_256_h4_v1/sample_results.csv` | Cliff absent at core 220 → controlling variable is the absolute core budget | VALID |
| Extval 3072/256 h16 | Cadence arm at full core 220, h16 | E2 | `configs/stages/statekv_extval_3072_256_h16.yaml` | `scripts/run_oracle_policy_freegen.py` | `results/…/statekv_extval_3072_256_h16_v1/sample_results.csv` | Cliff absent at core 220 (h16) | VALID |
| Extval 3072/256 multikey | Multi-key retrieval workload | E2 | `configs/stages/statekv_extval_3072_256_mk.yaml` | `scripts/run_oracle_policy_freegen.py` | `results/…/statekv_extval_3072_256_mk_v1/sample_results.csv` | Multikey 3/3 | VALID |
| Extval Qwen2.5-7B | Is the closure Qwen3-specific? | Qwen2.5-7B, 3072/256 | `configs/stages/statekv_extval_3072_256_qwen25_7b.yaml` | `scripts/run_oracle_policy_freegen.py` | `results/…/statekv_extval_3072_256_qwen25_7b_v1/sample_results.csv` | qk_pool NIAH 3/3, KL 0.004–0.009, GovReport 0.009–0.055; uniform fails → closure pattern not Qwen3-specific | VALID (family confirmation) |
| Extval reasoning attention-free | Attention-free variant on reasoning workload | E2 | `configs/stages/statekv_extval_3072_256_reasoning_af.yaml` | `scripts/run_oracle_policy_freegen.py` | `results/…/statekv_extval_3072_256_reasoning_af_v1/sample_results.csv` | Reasoning attention-free probe; no dedicated ccfa claim — see `analysis/statekv_external_validity_log.md` | EXPLORATORY |
| Extval decomposition 3072/256 | Swap-oracle / ranking regret at 4× context | E2 | `configs/stages/statekv_extval_decomp_3072_256.yaml` | `scripts/run_qkv_decomposition.py` | `results/…/statekv_extval_decomp_3072_256_v1/swap_rows.parquet`; `analysis/tables/extval_swap_regret.csv` | Swap oracle flat at 3072: 91% pairs < 1e-4, median regret 7.6e-15 → no ranking headroom at 4× context; hard-cycle predictability ρ ≤ 0.36 replicates 768 | VALID (context-invariance of closures) |

---

## 5. Retest program (no-gate re-evaluations on fresh sequences, 2026-08-10)

| ID | Research question / hypothesis | Substrate | Config | Runner script | Result location | Main result | Status |
|---|---|---|---|---|---|---|---|
| Retest replay era1 n24 | Do the Era-1 vetoed contribution-family policies hold on 24 fresh sequences? | E1, 24 fresh seqs, 17 policies | `configs/stages/retest_replay_era1_n24_protocol.yaml` | `scripts/run_direct_policy_replay.py` | `results/…/statekv_retest_replay_era1_n24_v1/metrics.csv`; `analysis/statekv_retest_report.md` | Contribution q75 mean KL 0.3938 vs attention 0.4085; sequence wins 54–58% → P7/P9 vetoes NOT confirmed on fresh data | VALID (no-gate re-evaluation) |
| Retest freegen qwen3 n20 | Matched-budget re-test of qk-tiered-V, token rarity, temporal volatility, cheap controllers | E2 recoverable, 20 fresh seqs, 12 policies | `configs/stages/retest_freegen_qwen3_8b_n20_protocol.yaml` | `scripts/run_retest_freegen.py` | `results/…/statekv_retest_freegen_qwen3_8b_n20_v1/{policy_aggregates.csv,paired_comparisons.csv}` | Tiered-V matched budget: KL 0.008066 vs qk_pool 0.007614, NIAH 10/10, task scores identical (G5 veto was an unequal-memory comparison). Token rarity: NIAH 10/10, GovReport ROUGE-L 0.055 vs attention 0.061, KL 0.869 (retrieval-specific holds, cross-task gap persists). Temporal volatility: NIAH 0.8, KL 0.2406 — not competitive on Era 2. Cheap B1–B3: NIAH 10/10, KL 0.184–0.235 vs attention 0.327 | VALID (no-gate re-evaluation) |
| Retest VJP Rademacher | Independent replication of the post-hoc Rademacher VJP gain (P3 stress) | E1, 8 fresh seqs | `configs/stages/retest_vjp_rademacher_replication.yaml` (base: `retest_vjp_base_config.yaml`) | `scripts/run_vjp_routes_pilot.py` | `results/…/statekv_retest_vjp_rademacher_replication_v1/metrics.csv`; `analysis/statekv_retest_report.md` | All gains negative on 8 fresh sequences → the dev post-hoc gain was a development-set artifact | VALID (replication negative) |

---

## Appendix A — sanity-check / debug scaffolding (SANITY CHECK)

Smoke, compatibility, and probe-check dirs. Never curated as evidence; listed so
the directory census is complete:

- `chk2`, `dec`, `probecheck`, `tokcheck` — early debug fragments.
- `freegen_smoke_runtime`, `policy_smoke_runtime_policy_runtime` — runtime smokes.
- `functional_probe_smoke_4bit_seed42_v1`, `gauge_geometry_smoke_1seq_v1`,
  `gauge_geometry_smoke_1seq_v2`, `gauge_geometry_smoke_2task_v3`,
  `output_sensitivity_smoke_2task_v1`, `robust_envelope_smoke_h2_seed42_v1`,
  `trajectory_model_smoke_h2_seed42_v1` — discovery-era smokes.
- `smoke_seed42`, `smoke_4bit_seed42`, `smoke_4bit_seed42_protocol_v{2,3,4,5}` —
  protocol-development smokes.
- `qkv_smoke_v1`, `qkvtier_smoke_v1`, `statekv_recoverable_r0_smoke_v1`,
  `statekv_extval_7b_smoke_v1`, `retest_freegen_smoke_seed20260810_v1` —
  pre-gate smokes for the closure/retest batteries.
- `statekv_p1_p3_gates_smoke_{calibration,p1,p2,p3}` — gate-runner smokes.
- `statekv_selective_refresh_labels_r2a_smoke`, `..._smoke_nolabel` — label-runner
  smokes (these two do carry curated-style summary/config but are smoke runs).
- `qwen3_8b_compatibility_check_seed20260808_v1` — backend compatibility check.

## Appendix B — deliberate failure-preservation dirs (INVALID)

- `results/…/mlx_qwen25_15b_inst_4bit/longbench/p20a_integration_failed_no_token_stream`
- `results/…/mlx_qwen25_15b_inst_4bit/longbench/p20a_integration_failed_relative_output_retry`

Both are integration failures of the P20a static-lexical LongBench workload
(missing token stream; wrong relative-output wiring), kept deliberately as a
failure trace. The fixed workload ran as `p20a_v1`. Status: **INVALID** — do not
quote numbers from these dirs.

## Appendix C — discovery-era exploration runs (EXPLORATORY)

Pre-P0 temporal-cache-discovery program (Qwen2.5-1.5B-4bit, seed 42 line). These
motivated the phase program; they carry no preregistered gates:

| Run dir | Content |
|---|---|
| `discovery_small_4bit_seed42` (+ `_protocol_v2`, `_v3`, `_v4`) | Core discovery sweeps; protocols v2/v3/v4 coexist with no recorded canonical pick |
| `functional_probe_stage1_4bit_seed42_v1` (+ `_interrupted_pre_prefill_optimization`) | Functional staleness probes (one interrupted run retained) |
| `gauge_geometry_4bit_24seq_seed42_v1` | Gauge-geometry screen, 24 seqs |
| `independent_fisher_4bit_24newseq_seed20260726_v1` | Independent Fisher trajectories (later the source run for the VJP retest) |
| `mechanism_targeted_4bit_seed42_v1` | Targeted mechanism probes |
| `output_sensitivity_4bit_24seq_seed42_v1` | Output-sensitivity screen |
| `robust_trajectory_envelope_4bit_seed42_v1` | Robust trajectory envelope |
| `theory_closing_4bit_seed42_v1` | Theory-closing audit |
| `trajectory_stochastic_model_4bit_seed42_v1` | Stochastic trajectory model |

The `benchmarks/mlx` harness runs backing the P18–P21 workloads live under
`results/…/mlx_qwen25_15b_inst_4bit/{longbench,ruler}/` (`v1`, `p19a_v1`,
`p20a_v1`, `p21a_v1`, `p19b_v1`, `p20b_v1`, `p21b_v1`), driven by
`benchmarks/mlx/configs/experiments/statekv/*.yaml` via
`benchmarks/mlx/scripts/run_benchmark.py`. The two bare `v1` dirs predate the
per-phase naming; their exact config mapping is not recorded (see coverage note).

## Appendix D — runtime scaffolding (not experiments)

- 82 seed-suffixed dirs (e.g. `direct_policy_shrinkage_independent_seed20260813_v1`,
  `statekv_qkvtier_gate_352t_seed20260808_v1`) — execution-time twins of the
  canonical runs.
- `statekv_p1_p3_gates_qwen3_8b_seed20260808_v1_{calibration,p1,p2}` — runtime
  fragments of the gate runner behind P33/P34/P35; no curated summary dirs.
- `statekv_ladder_2b_seed20260808_v1_ladder`,
  `statekv_teacher_gate_*_seed*_teacher_gate`,
  `statekv_selective_refresh_r2a_*_seed*_r2a_labels`,
  `statekv_refresh_arms_768_256_seed20260808_v1_r2b_gate`,
  `statekv_selective_refresh_r2b_smoke_seed20260808_v1_r2b_gate` — per-stage
  scratch dirs for the gate runner.

---

## Coverage note

Entries I could not classify with full confidence:

1. **`statekv_extval_3072_256_reasoning_af_v1`** — has canonical
   summary/config but no dedicated claim in `configs/ccfa.yaml`; classified
   EXPLORATORY on that basis.
2. **Bare `v1` dirs under `mlx_qwen25_15b_inst_4bit/{longbench,ruler}/`** — the
   harness config set contains `p18a/p18b` (temporal volatility) configs, but no
   `p18a_v1`/`p18b_v1`-named result dirs exist; the bare `v1` dirs presumably
   hold those workloads. The config→dir mapping is inferred, not recorded.
3. **`statekv_p2_qwen25_15b_p23b_cheap_v1`** — role (cheap panel for the
   substrate-B teacher gate) inferred from the config name and summary contents;
   it has no ccfa experiment entry of its own.
4. **Discovery-era runs (Appendix C)** — directory-level provenance only; no
   per-dir manifests survive, so the listing is by name and stored artifact
   types, not by a registry.
5. **`statekv_openstress_3072_256.yaml`** — config exists with no corresponding
   run dir (the 3072 regime was covered by the extval runs instead); not an
   experiment, noted here so the config is not mistaken for missing evidence.
6. **`analysis/manifest.json` staleness** — the manifest references
   `analysis/README.md`, `analysis/david_update.md`, and
   `analysis/data_schema_report.md`, none of which exist; the
   `analysis/generate_report.py` pipeline that produced them is partially dead.
   This affects analysis plumbing, not any registry row above.
