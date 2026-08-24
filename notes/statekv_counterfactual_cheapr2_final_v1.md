# Cheap-R2 Final Report: R2 → Cheap-R2 Pareto Campaign

Date: 2026-08-24. Branch: `codex/statekv-counterfactual-utility`.
Scope (user directive 2026-08-23): stop static-student distillation; find the
minimal future computation that preserves R2's strict-eviction advantage.
Phases: horizon sweep → refresh sweep → selective trigger → (draft rollout:
evaluated as unnecessary, see §6).

All runs: Qwen3-8B-4bit, strict pure eviction, cold recovery = 0, sink 4 /
recent 32, budget 256 (core 220), 64 decode cycles, train panels (multikey /
multiquery 200–209; 2wikimqa / gov_report 0–9). Integrity audits passed on
every sweep (unique samples, no shard overlap, strict flags, peak ≤ 256,
clean exits). Frozen references: teacher PK (`teacher_pk_v1_all_arms.csv`,
H=1/f=2 R2 and baselines) and Full-cache references rerun per config.

## 1. Phase 1 — Horizon sweep (multikey@256, refresh=2)

| H | score | teacher s/arm | wall s/arm | ×QK wall (71.7s) |
|---|-------|---------------|------------|-------------------|
| 1  | 32.5 (frozen) | 454.5 | 580.3 | 8.1× |
| 2  | 37.5 | 503.0 | 560.2 | 7.8× |
| 4  | 52.5 | 511.8 | 569.6 | 7.9× |
| 8  | 55.0 | 526.7 | 588.3 | 8.2× |
| 16 | 60.0 | 564.5 | 618.7 | 8.6× |
| 32 | **72.5** | 651.5 | 707.1 | 9.9× |

Rollout steps are nearly free (≈0.2 s/step); the per-refresh prefix
recompute (≈14.2 s) dominates. H=32 exceeds the Full-cache reference (70.0)
on this panel — H=1 was never the R2 family's ceiling. (Gate C stays frozen;
this is a new evaluation of the same teacher family.)

## 2. Phase 2 — Refresh sweep (multikey@256, H=32)

| refresh | score | refreshes/arm | teacher s | wall s | ×QK |
|---------|-------|---------------|-----------|--------|-----|
| 1  | 65.0 | 64 | 1287 | 1347 | 18.8× |
| 2  | 72.5 | 32 | 651 | 707 | 9.9× |
| 4  | 67.5 | 16 | 325 | 379 | 5.3× |
| 8  | 67.5 | 8 | 162 | 215 | 3.0× |
| 16 | **67.5** | **4** | **80** | **132** | **1.8×** |

Flat 67.5 from f=4 to f=16: only 4 rollouts per sequence suffice. The lost
5 points vs f=2 concentrate in samples 200 and 209. f=1 is NOT better than
f=2 (65.0 vs 72.5; n=10 noise-level but non-monotonic).

## 3. Phase 3 — Selective trigger (multikey@256, H=32)

Trigger statistic (QK normalized margin at the cutoff) is always tiny
(max 0.029 over 256 logged cycles; median 0.0034) — thresholds 0.1/0.2
degenerate to always-fire.

| variant | semantics | score | invocation | refreshes | wall | ×QK |
|---------|-----------|-------|------------|-----------|------|-----|
| pm_t001  | QK between triggers | 45.0 | 24% | 15.6 | 356 | 5.0× |
| pm_t001c | cached R2 between triggers | 65.0 | 24% | 15.3 | 348 | 4.9× |
| pm_t0025c | cached R2 between triggers | 67.5 | 43% | 27.4 | 581 | 8.1× |

Two findings:

1. **QK-between-triggers fails** (45.0 vs 67.5): QK-guided eviction cycles do
   irreversible damage; the cache must always be governed by (possibly stale)
   R2 rankings. A trigger may only decide WHEN to refresh, never WHAT ranks.
2. **Trigger timing adds nothing over uniform periodic**: pm_t0025c uses 27
   targeted refreshes to reach the same 67.5 that f=16 reaches with 4 uniform
   refreshes. On this panel, only the refresh COUNT matters, and 4 suffice.

## 4. Winner validation on secondary/control tasks (H=32, f=16)

| task @256 | Full | QK | SnapKV | H2O | R2-H1-f2 (frozen) | **R2-H32-f16** |
|-----------|------|----|--------|-----|--------------------|----------------|
| multikey    | 70 | 20 | 2.5 | 5 | 32.5 | **67.5** |
| multiquery  | 100 | 100 | 0 | 30 | 100 | **100** |
| 2wikimqa    | 30 | 40 | 20 | 20 | 40 | **30** |
| gov_report  | 6.27 | 5.83 | 5.93 | 5.41 | 5.73 | **6.08** |

- multiquery: ties Full (100); task saturated at 256.
- 2wikimqa: 30 vs QK/R2-H1 40 — the entire gap is one sample (2wikimqa:7,
  the known "compression denoising" sample where Full=0 but QK/R2-H1=100).
  n=10 binary; not a systematic regression, but R2-H32-f16 adds nothing here.
- gov_report (control): 6.08 vs Full 6.27 (97%) — no collapse; best of all
  compression policies measured.

## 5. Final Pareto (multikey@256) and the two requested answers

| variant | score | wall ×QK | R2 gain retention |
|---------|-------|----------|-------------------|
| QK | 20.0 | 1.0× | 0% |
| R2-H1-f2 (frozen) | 32.5 | 8.1× | 100% (ref) |
| R2-H32-f2 | 72.5 | 9.9× | 190% |
| **R2-H32-f16** | **67.5** | **1.8×** | **380% of frozen R2-H1 gain; 90.5% of H32-f2 gain** |
| hybrid-cached best | 67.5 | 8.1× | dominated by f16 |

- **Cheapest variant clearly above QK: R2-H32-f16** (67.5 vs QK 20, 1.8×QK
  wall, 4 rollouts/sequence).
- **Closest to full R2 at much lower cost: also R2-H32-f16** (5 points below
  H32-f2 at 5.4× lower wall time). The simplest configuration — longest
  horizon, sparsest uniform refresh — wins the whole campaign.

## 6. Phase 4 decision: draft-model rollout SKIPPED (justified)

Target was "runtime from ~19× to as close to 1–2× as possible". H32-f16
achieves 1.8×QK without any draft model, so the draft branch has no headroom
to justify its engineering risk. Recorded as evaluated-and-unnecessary.

## 7. Scientific takeaways

1. R2's real ceiling was severely underestimated at H=1: H=32 nearly doubles
   it (32.5→72.5) and exceeds Full cache on multikey (denoising effect).
2. Cost is 100% in prefix recompute frequency, ~0% in rollout length.
3. Eviction must always follow (cached) R2 rankings; any QK-governed cycle
   irreversibly damages the cache under strict pure eviction.
4. Margin-based selective triggering has no value over uniform sparse
   refresh on this panel — timing doesn't matter, count does, and 4/64
   cycles suffice.
5. Residual gap (67.5 vs 72.5) lives in samples 200/209, which want denser
   early refreshes; a "dense-first-8-cycles then f16" schedule is a candidate
   follow-up, not pursued here.

Artifacts: `results/statekv_counterfactual/cheapr2_{h2,h4,h8,h16,h32,h32f16_*,
hybrid_*}_v1/`. Configs: `configs/statekv_counterfactual/cheapr2_*.yaml`.
Related: horizon sweep note `notes/statekv_counterfactual_cheapr2_horizon_sweep_v1.md`.
