# StateKV Open Search Report — open-ended research search after three closures

Status: final
Date: 2026-08-10
Verdict: **CLOSE** (no method candidate survives; sharpest new findings are
boundary conditions of exact query-aware routing)
Search log: `statekv_open_search_log.md` (full search tree, including every
falsified branch)

## 0. Starting position

Three preregistered closures bound this search: strict-pure-eviction teacher
(Gate 0-2), recoverable physical-risk teacher (R0: qk_pool 0.0086 vs teacher
0.0213, 10/10 paired), QK–V routing discovery (I(target;V|QK)≈0 everywhere;
qk_tiered_v gate NO_GO).  From the older lineage: training-free direct
signal policies (P8-P24), dynamic per-layer budgets (P34), selective refresh
triggers (R1/R2) — all closed.

## 1. What this search established as the remaining question

At horizon-1 refresh with full-pool visibility, the selection at cycle t is
scored by the *same* query that the same-input KL measurement uses (verified
in `_free_rollout` / `_full_pool_scores`: selection query == measurement
query; the full reference advances along the policy's own trajectory).
Therefore qk_pool @ horizon-1 is the **optimal memoryless head-mean
token-level selector** for its evaluation, and its residual KL (0.0086 @
256) bounds what any selector can gain there.  The search therefore asked:
which of qk_pool's *assumptions* break first when the regime changes?

Assumptions tested: generous coverage (HF1), cheap approximation of its
score (HF2), per-step refresh cadence (HF3 + corner), head-mean aggregation
across GQA KV heads (HF4), single-session decode (HF5, deferred),
trainability of a better selector (HF6, deprioritized — labels ≈ qk_pool).

## 2. Hypothesis families covered and their fates

| family | test | outcome |
|---|---|---|
| HF1 coverage stress (23%→15%→8%) | 2 real runs, 10 paired samples each | regime discriminates; qk_pool stays task-perfect; nothing closes in |
| HF2 approximation frontier | offline recall bounds + stress runs | pages cannot recover token-level exactness; gap widens with tight budgets |
| HF3 cadence @ 256 | offline probes | closed (confirms R2a time-invariance) |
| HF3' cadence × coverage corner | 3 real runs (h1/h4/h16 @ 64) + preregistered rescue arm | **cliff found**; rescue FAILED (NO_GO_CORNER) |
| HF4 per-KV-head selection | instrumented probe (3 samples, 36 layers × 8 heads) | closed: +0.96pp mass, below action threshold |
| HF1b hard-cycle predictability | offline feature screen | no observable trigger; conditional-budget idea dead before runs |
| HF5 cross-session reuse | — | deferred (no harness; NEW PROJECT candidate) |
| HF6 learned selector | — | deprioritized (labels ≈ qk_pool) |

## 3. Main empirical results

### 3.1 Coverage stress (768 ctx, exact same-input KL, 10 paired samples)

| arm | @256 | @128 | @64 |
|---|---|---|---|
| qk_pool | 0.0086 | 0.0257 | 0.0819 |
| quest_like (p16) | 0.0243 | 0.0767 | 0.7065 |
| uniform | 0.8952 | 1.6298 | 2.0964 |

qk_pool KL scales ~3.2× per budget halving but keeps **NIAH 1.00 at every
budget** and GovReport at the full-cache level (all arms ≈ 6 official on
this low-resolution metric).  quest_like collapses at 64 (8.6× qk_pool,
NIAH 0.8, failures concentrated on NIAH samples).

### 3.2 Cadence × coverage corner (the cliff)

| qk_pool @ 64 | mean KL | NIAH |
|---|---|---|
| h1 | 0.0819 | 1.0 |
| h4 | 0.3761 | 0.2 |
| h16 | 0.8439 | 0.0 |
| qk_obswin(w32) @ h16 | 1.0748 | 0.0 (paired 2/10 vs qk_pool h16) |

The preregistered corner gate (`statekv_corner_gate.md`) ruled GO_CORNER
requires NIAH 1.0 + KL ≤ 0.25; the observation-window arm is *worse* than
single-token scoring.  Verdict NO_GO_CORNER: at tight coverage, exact
routing must refresh (near) every step; no cheaper refresh-time scoring
preserves quality at slow cadence.  The fix is cadence, not scoring.

### 3.3 Approximation frontier (HF2)

Page-max recall of the exact top-220 core (true attention as within-page
oracle — upper bound for Quest-style metadata): p4 0.674, p8 0.623,
p16 0.585, p32 0.543.  Combined with 3.1: token-level exactness is worth
2.8-8.6× KL depending on budget, and page granularity provably cannot
recover it on this substrate.

### 3.4 Per-head selection (HF4)

Per-KV-head own-top-220 vs shared core at identical budget: +0.96pp mean
captured mass (p95 +3.6pp, concentrated in diffuse early layers 0-1).
Below the pre-committed action threshold; per-head decode masks not
implemented.  Literature: Ada-KV/HeadKV/KV-Compress cover this family.

### 3.5 Negative micro-results

- qk_pool residual KL is event-driven (top-10% cycles = 76.5% of KL mass)
  but hard cycles are NOT predictable from runtime observables (missed mass
  −0.02, entropy +0.21, cycle index +0.36): conditional budgeting dead.
- Core-set Jaccard(c,c+1) = 0.653; stale-mass decay dominated by the sliding
  recent window — no cadence headroom at 256 (R2a confirmed independently).

## 4. Answers to the 18 required questions

1. **Remaining scientific question**: which assumptions of exact current-query
   full-pool routing break under regime stress, and does any observable
   residual structure survive for a method?  (Answered: cadence at tight
   coverage breaks; no exploitable residual.)
2. **Families covered**: §2 table (7 distinct families + 2 deferred).
3. **Regimes lacking discrimination**: 768/256 (saturated: qk_pool 0.0086,
   every full-pool arm quality-equivalent); cadence stress at 256 (R2a).
4. **Regimes exposing failure/headroom**: 768/64 discriminates strongly
   (uniform/quest collapse; qk_pool KL 10× its 256 value); the cadence
   corner h4/h16@64 is the only regime where qk_pool itself fails the task.
5. **Where qk_pool ≈ oracle**: every quality-valid operating point with
   per-step refresh, down to 8% coverage (NIAH 1.0 throughout).
6. **Where systematic regret appears**: slow refresh at tight coverage —
   KL 4.6× by h4, NIAH 0 by h16.  This is an interaction effect, invisible
   in R2's budget-256 sweep.
7. **Residual signals found**: none exploitable.  (The cadence cliff is a
   *regime* finding, not a signal: no refresh-time observable rescues it.)
8. **Signals falsified this round**: observation-window scoring (worse than
   last-token), per-head selection (mass gain too small), hard-cycle
   conditional budgeting (no trigger), page-metadata approximation (upper
   bound below token-level).
9. **New mechanism**: the cadence × coverage interaction cliff, plus the
   optimality argument (selection query == measurement query at horizon 1)
   explaining *why* nothing beats qk_pool at h1.
10. **Method formed?** No.  qk_obswin was implemented and gated; NO_GO.
11. **Lineage continuity**: the search stayed on the StateKV question
    (state-conditioned working-set control); the closure is that state
    conditioning beyond the current query does not pay in any tested regime.
12. **Advantage over strongest baseline**: none — nothing beat qk_pool at
    its own cadence; at slow cadence everything failed equally.
13. **Stability**: all conclusions paired across 10 samples × 2 tasks × 3
    budgets × 3 cadences.
14. **New artifacts/evaluator issues**: two found and fixed —
    (a) headwise probe recorder indentation bug (only layer 35 recorded;
    fixed, rerun, regression-checked by row counts);
    (b) open_stress_compare.py mixed GovReport≈6 with NIAH=100 in one
    average (caught; all reported figures use direct per-task reads).
    No earlier conclusions affected.
15. **Formally closed directions**: V-routing (prior), physical-risk teacher
    (prior), per-head selection, observation-window/slow-refresh scoring,
    page-approximation of qk_pool, conditional hard-cycle budgeting,
    temporal triggers (reconfirmed).
16. **Worth another round**: cross-session/prefix reuse (HF5, needs a new
    harness — NEW PROJECT); a true long-context substrate (4K+, machinery
    ready, config prepared but unrun — expected to reproduce the coverage
    ladder rather than change conclusions); deployment-cost modeling of
    qk_pool (transfer volume accounting exists in R0 telemetry).
17. **Most plausible paper route**: a negative-results/benchmark-style
    contribution: exact full-pool query-aware routing as an oracle reference
    for hierarchical KV, its optimality argument at per-step refresh, the
    coverage/cadence frontier map, and the demonstration that celebrated
    cheaper alternatives (pages, observation windows, risk scorers,
    per-head splits) each fail for identifiable mechanistic reasons.
18. **Continue or end?**  End the method search.  The remaining space is
    either literature-covered (per-head), machinery-missing (cross-session),
    or dominated by a known expensive answer (refresh exact QK every step).

## 5. Artifacts

| artifact | path |
|---|---|
| Search log (full tree) | docs/evidence/statekv_open_search_log.md |
| Corner gate (preregistered + verdict) | docs/evidence/statekv_corner_gate.md |
| Probe builder | analysis/tables/open_search_probes.py (outputs open_hf1_*/open_hf2_*/open_hf3_*.csv) |
| Stress runs | results/.../statekv_openstress_768_{128,64}_v1/ |
| Corner runs | results/.../statekv_opencorner_768_64_{h4,h16}_v1/, statekv_corner_obswin_768_64_h16_v1/ |
| Headwise probe | results/.../statekv_headwise_probe_qwen3_8b_v1/headwise_rows.parquet; analysis/tables/open_hf4_* |
| New machinery | `qk_obswin` (obswin full-pool scoring, `oracle_policy_freegen.py`), `_headwise_rows` (`qkv_decomposition.py`), tests in test_recoverable_r0.py / test_qkv_decomposition.py |
| Configs | configs/stages/statekv_openstress_768_{128,64}.yaml, statekv_opencorner_768_64_{h4,h16}.yaml, statekv_corner_obswin_768_64_h16.yaml, statekv_headwise_probe_qwen3_8b.yaml, statekv_openstress_3072_256.yaml (prepared, unrun) |
| Claim registry | configs/ccfa.yaml (stage closed-negative-open-search-*; 4 new claims) |

Full test suite: 476 collected, 470 passed, 5 skipped, 1 pre-existing
failure (`test_repository_architecture`, predates this session).
