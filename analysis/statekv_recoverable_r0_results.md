# StateKV Recoverable Gate R0 — results and verdict

Status: final
Verdict: **NO-GO (NO_HEADROOM)** — preregistered rules in
`statekv_recoverable_r0_protocol.md` §7, computed by
`analysis/tables/recoverable_r0_headroom.py` (no threshold was touched after
results were observed).
Date: 2026-08-09

## 1. What R0 tested

> Under identical recoverable backing-store semantics and identical active
> budget, does state-conditioned physical-risk scoring improve working-set
> selection over standard query-aware retrieval?

All arms share: `KVBackingStore` full-history pool (re-entry legal for every
arm, cheap ones included), budget 256 = sink 4 + recent 32 + core 220 per
layer, refresh every cycle (horizon 1, 64 cycles), the same backing candidate
universe (mean 1118.8 positions), greedy generation, same-input exact-KL
evaluation.  Substrate is byte-identical to Gate 0: Qwen3-8B-4bit,
gov_report:86-90 + synthetic_niah:86-90, seed 20260808.

Baseline-semantics audit that shaped the arm set (protocol §2.1): the P31
`attention` baseline scored only the *active* set (evicted positions score
0.0 via `_score_on_universe`), so it was de facto quasi-irreversible.  R0
adds true full-pool query-aware baselines (`qk_pool`: exact current-query
head-mean attention over all backing positions, one full-pool scoring
forward per cycle; `quest_like`: 16-token page-granular variant) and keeps
P31-style `attention` as the labeled quasi-irreversible control.

## 2. Main table (10 paired samples, mean per-sample trajectory KL)

| arm | mean KL | median | step p95 | NIAH | GovReport official | recovered frac | churn/layer |
|---|---|---|---|---|---|---|---|
| full_cache (ceiling) | 0.0000 | 0.0000 | — | 1.00 | 6.27 | — | — |
| **qk_pool (B\*)** | **0.0086** | 0.0089 | **0.0509** | 1.00 | 5.83 | 0.215 | 48.0 |
| statekv teacher | 0.0213 | 0.0181 | 0.1124 | 1.00 | **6.29** | 0.519 | 116.0 |
| quest_like | 0.0243 | 0.0254 | 0.1189 | 1.00 | 5.92 | 0.213 | 47.5 |
| attention (P31-style, quasi-irreversible) | 0.3458 | 0.3701 | 1.9268 | 1.00 | 5.93 | 0.000 | 0.0 |
| recency | 0.7952 | 0.7633 | 4.6153 | 0.00 | 5.47 | 0.004 | 1.0 |
| uniform | 0.8952 | 0.8408 | 4.6151 | 0.00 | 5.78 | 0.492 | 110.0 |

Fairness: `execution_valid` and `all_budgets_respected` both true;
`maximum_active_cache_tokens` = 256 for every arm and cycle; identical
universe, cadence, and mandatory structure verified by unit tests
(`tests/test_recoverable_r0.py`, 9 tests) and run flags (G5 pass).
Substrate is quality-valid: full_cache NIAH = 1.00 (G4 validity clause pass).

## 3. Verdict evaluation (preregistered G1–G5)

B\* = qk_pool (strongest cheap recoverable, mean KL 0.0086).

- **G1 (headroom ≥ 30%): FAIL.**  teacher/B\* ratio = **2.47** (teacher is
  2.5× *worse*, not 30% better).
- **G2 (paired stability): FAIL.**  qk_pool beats the teacher on **10/10**
  samples; paired bootstrap 95% CI of (B\* − teacher) =
  [−0.0207, −0.0059] — excludes 0 in B\*'s favor.
- **G3 (tail): FAIL.**  teacher p95 step KL 0.1124 = 2.2× qk_pool's 0.0509.
- G4 (quality): pass — teacher task quality is not worse (NIAH 1.00 both;
  GovReport official 6.29 vs 5.83 — inside the noise band audited in §6.1).
  Quality cannot rescue G1–G3.
- G5 (fairness): pass.

**R0 verdict: NO-GO, subclass NO_HEADROOM.**  The StateKV physical-risk
scorer has *negative* residual headroom (D3 = −0.0127) over plain full-pool
query-aware retrieval.  Per protocol §8 the recoverable pivot stops here:
no student, no reranker, no system work.

## 4. Where the old P31 gain came from (decomposition)

Ladder over the same 10 samples (`recoverable_r0_ladder.{csv,md}`):

| component | Δ KL |
|---|---|
| D1 recoverability, same rule (pure attention 0.0976 → rec attention 0.3458) | −0.2482 |
| D1 recoverability, best cheap (pure b2_uniform 0.0961 → rec qk_pool 0.0086) | +0.0875 |
| D1 recoverability, teacher (pure teacher 0.2322 → rec teacher 0.0213) | +0.2109 |
| D2 query-aware retrieval (best simple recoverable 0.7952 → qk_pool 0.0086) | **+0.7866** |
| D3 physical-risk scorer (qk_pool 0.0086 → teacher 0.0213) | **−0.0127** |

Reading:

1. **Recoverability rescued the teacher** (+0.21): the strict-pure-eviction
   teacher (Gate 0: 0.2322, worse than cheap) becomes 0.0213 once it may
   re-fetch history.  That is the entire "P31 looks beautiful" effect —
   P31's teacher ran on a persistent backing store.
2. **Recoverability alone does nothing for a same-rule cheap baseline**
   (−0.25 for attention): P31's cheap baselines were quasi-irreversible, so
   they could not cash in the backing store.  This baseline handicap is the
   second half of why P31's teacher margin looked large.
3. **The dominant term is full-pool query-aware retrieval** (+0.79 over
   simple recoverable controls): once the current query may score *all*
   historical positions, a one-line top-k rule reaches KL 0.0086 — 2.5×
   below the expensive teacher, with better tails and 10/10 paired wins.
4. **The StateKV scorer's residual value is negative** (−0.013): oracle
   1-step exact-KL selection over a 10-candidate panel (which *includes*
   qk_pool) loses to committing the qk_pool action every cycle.

## 5. Mechanism: why the teacher loses to its own panel member

The teacher's panel contains `qk_pool`; an omniscient selector could match
the qk_pool arm by picking it every cycle.  Instead the teacher picked
qk_pool in only **148/640 cycles (23%)**, spreading the rest across
h2o/snapkv/attention/value_norm/quest_like/recency/key_norm/uniform — and
the cycles where it picked qk_pool are the *harder* ones (mean cycle KL
0.029 vs 0.019 for other picks), i.e. selection is counter-cyclical noise,
not signal.  This is the Gate 0 plateau/cliff finding reproduced under
recoverable semantics: one-step exact-KL panel scores sit on a plateau and
cannot identify "retain what the current query attends to over the full
pool" as the winning action; the teacher random-walks across candidates
(churn 116/layer, recovered fraction 0.52) and intermittently commits
actions that drop tokens later queries need, while the committed qk_pool
arm applies the right action uniformly (churn 48/layer).

The valuable action in this regime is directly computable from one
full-pool scoring forward; physical-risk evaluation of candidate actions
adds cost and noise, not accuracy.

## 6. Boundary conditions (what this closure does NOT cover)

- One substrate: Qwen3-8B-4bit, ~0.9–1.2K prompts, budget 256 (~20–26%
  coverage), 64 generated tokens, NIAH + GovReport.  Longer generations,
  much lower coverage regimes, or tasks where current-query attention is a
  poor predictor of future queries (if any exist) were not tested.
- `qk_pool` here uses an *exact* full-pool scoring forward per cycle — an
  oracle-grade version of query-aware retrieval (real Quest/retrieval
  systems approximate it with key metadata).  R0 measures algorithmic
  headroom only; the cost accounting (`pool_scoring_forward_time_s`,
  mean 8.3 s/sample-arm, plus transfer volume) is recorded for future
  system work but was excluded from the verdict by design.
- The teacher's quality edge on GovReport (+0.46 official over qk_pool) is
  within the 64-token ROUGE noise band (paired bootstrap 95% CI
  [+0.02, +0.88], barely excluding zero; see §6.1) and cannot carry a
  method (G1–G3 fail decisively).

### 6.1 GovReport output audit (qualitative, post-verdict, no rule changes)

All five GovReport generations were read in full for every arm.  Findings:

- **No arm degenerates.**  Every recoverable policy (including uniform and
  recency) produces fluent, topically correct 64-token summaries.  Absolute
  ROUGE-L is 0.04-0.08 for everyone because 64 tokens are compared against
  a full-length reference summary; ±1 official point corresponds to
  roughly 1-2 more/fewer content words overlapping the reference.
- **"teacher > full_cache" is noise.**  Mean teacher−full = +0.02 points,
  paired bootstrap 95% CI [−0.11, +0.18].  On 3/5 samples the teacher's
  text is near-identical to full_cache's (identical scores); on gov90 the
  teacher writes "Consolidated Services Pension Fund" — a factual error
  that full_cache, qk_pool, and attention all avoid ("Central States").
- **teacher > qk_pool (+0.46) is weakly significant but tiny.**  Paired
  bootstrap CI [+0.02, +0.88]; the difference is 1-2 reference-overlapping
  words per sample.  Notably, qk_pool's text is on average *more* similar
  to the full_cache text than the teacher's is (content-word Jaccard 0.788
  vs 0.744), consistent with its 2.5x lower KL; the teacher's ROUGE edge
  comes from incidental phrasing overlap with the reference, not from
  better fidelity or better content.
- Bottom line: on this substrate the three full-pool arms (teacher,
  qk_pool, quest_like) are **quality-equivalent** at the resolution this
  metric can see; G4 passes as "quality non-worse", but there is no
  meaningful StateKV quality advantage, and the KL verdict (G1-G3) stands
  untouched.

## 7. Why P31 looked beautiful (the answer to the pivot question)

Old P31 gain = recoverability (backing store for the teacher) + a
quasi-irreversible baseline handicap (attention/snapkv could not score
evicted tokens) + the absence of a true full-pool query-aware control.
Under unified recoverable semantics with that control present, the
StateKV scorer's independent oracle headroom is **negative**.  The pivot
hypothesis — "state-conditioned physical risk is for recoverable
working-set management" — is **refuted** at R0: recoverability matters,
and it is best exploited by direct query-aware retrieval, not by
physical-risk scoring.

## 8. Artifacts

| artifact | path |
|---|---|
| Protocol (preregistered) | analysis/statekv_recoverable_r0_protocol.md |
| Raw results | results/temporal_cache_discovery/statekv_recoverable_r0_qwen3_8b_v1/ (sample_results.csv, cycle_rows.parquet, step_rows.parquet, summary.json, config.yaml) |
| Config | configs/stages/statekv_recoverable_r0_qwen3_8b.yaml |
| Analysis | analysis/tables/recoverable_r0_main.{csv,md}, recoverable_r0_paired.csv, recoverable_r0_ladder.{csv,md} (builder: recoverable_r0_headroom.py) |
| New machinery | `recency_core`, `quest_like_core` (statekv/oracle_closed_loop.py); `qk_pool`/`quest_like`/`recency` panel candidates + pool-score plumbing (statekv/oracle_policy_comparison.py); `_full_pool_scores`, churn/recovery/cost telemetry, generalized aggregation (statekv/oracle_policy_freegen.py) |
| Tests | tests/test_recoverable_r0.py (9 tests; full suite 479 passed, only the pre-existing architecture-test failure remains) |
| Smoke run | tmp/r0_smoke_run/ (2 samples × 8 cycles, mechanics validation) |

## 9. Next steps

Per protocol §8: **stop the recoverable StateKV method line.**  No student,
no candidate-reranking pipeline (R1 is cancelled — its premise, positive
teacher headroom, does not hold), no scorer-complexity increases.  The
positive, repo-relevant residue: (i) the recoverable machinery and its
fairness tests are reusable; (ii) `qk_pool` (exact full-pool query-aware
top-k at KL 0.0086, NIAH 1.0) is the strongest observed working-set policy
on this substrate and the correct baseline for any future hierarchical-KV
work; (iii) the closure record above bounds what any future "risk-scored
retrieval" proposal must beat.
