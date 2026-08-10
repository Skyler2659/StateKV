# StateKV Recoverable Gate R0 — protocol and preregistered verdict

Status: preregistered (written before any R0 experiment was run)
Date: 2026-08-09
Supersedes-question: the Gate 0-4 closure (`statekv_teacher_closure_2026-08-09.md`)
closed the *strict pure eviction* method line.  R0 asks a new question under
*recoverable* semantics.

## 1. The question

> Under identical recoverable backing-store semantics and an identical active
> GPU KV budget, does state-conditioned physical-risk scoring improve
> working-set selection over standard query-aware retrieval?

Concretely: was the old P31 teacher headroom (KL 0.0506 vs 0.3357 attention)
primarily a **recoverability effect** (any policy that may re-fetch historical
KV wins), or does the StateKV physical-risk **scorer** retain significant
independent oracle headroom once every baseline shares the same backing store?

This is a headroom gate, not a deployability gate.  The teacher is allowed to
be expensive.  CPU↔GPU transfer latency is excluded from the verdict but
access volumes and scorer costs are recorded.

## 2. Recoverable semantics (binding on every arm)

1. All historical KV is kept in a CPU-resident backing store
   (`KVBackingStore`, exact fp16 K/V rows for every token ever processed).
2. Every policy may re-select positions that are not in the active cache
   (re-entry from backing is legal for all arms, including cheap ones).
3. Identical active budget: `total_budget = 256 = sink 4 + recent 32 + core 220`
   per layer, enforced per cycle by `validate_active_budget` and the
   `budget_respected`/`all_budgets_respected` run flags.
4. Identical refresh cadence: every arm re-selects its working set at every
   cycle (`control_horizon = 1`, 64 cycles).
5. Identical fixed-structure constraints: `mandatory_and_eligible` with
   sink 4 / recent 32 for every arm; the core is chosen from the same
   eligible universe.
6. Identical candidate universe: `backing.positions()` (full history) for
   every arm and every teacher panel candidate, every cycle.
7. The teacher's oracle privilege is *only* the exact-KL counterfactual
   evaluation of panel actions against the same-input full-cache reference.
   It receives no other information (no future tokens beyond the one-step
   horizon every rollout shares, no privileged candidate universe).
8. No arm may exceed the active budget at any point of any cycle.
9. Same-input metric semantics: the trajectory KL compares, per step, the
   full-cache state and the committed compressed state on the *same* input
   token (the compressed arm's own generated token).  Teacher candidate
   scoring is teacher-forced on the full-cache greedy continuation of the
   same current token.  No probe-logit / phase-mismatched comparisons
   (the Gate-2B ladder bug class) are used anywhere in R0.
10. Backing-store access is a common legal capability of all recoverable
    policies, not an artifact.  Access volumes (candidate universe size,
    pool-scoring forwards, teacher candidate evaluations) are recorded per
    cycle.

### 2.1 Audit finding that shapes the baseline set

The P31 machinery (`oracle_policy_freegen.py`) was already recoverable for
the *commit* path (all policies re-anchor from `KVBackingStore`).  However,
the P31 `attention` baseline scores positions with `memory.latest`, which
only covers the *currently active* set; evicted positions score 0.0 via
`_score_on_universe` and can therefore never be re-selected.  P31 `attention`
(and `snapkv`, whose window covers only recently-active positions) were
**de facto quasi-irreversible**.  Only `h2o` (cumulative), `key_norm`,
`value_norm`, and `uniform` scored the full pool.

R0 therefore adds true full-pool query-aware baselines (`qk_pool`,
`quest_like`) whose scores come from one full-pool scoring forward per cycle
(exact per-head attention of the current query over *all* backing
positions), and keeps the P31-style `attention` arm as an explicitly labeled
quasi-irreversible control.

## 3. Arms

| Arm | Role | Selection rule (core from full backing pool each cycle) |
|---|---|---|
| `full_cache` | ceiling (not budget-matched) | full history active; KL = 0 by construction; quality ceiling |
| `uniform` | recoverable simple control | deterministic evenly-spaced core over eligible (position coverage) |
| `recency` | recoverable simple control | most recent `core_budget` eligible positions (+ mandatory sink/recent) |
| `attention` | P31 control, quasi-irreversible | top-k by latest-query head-mean attention over the *active* set only |
| `qk_pool` | query-aware retrieval (exact QK/attention top-k) | top-k by current query's exact head-mean attention over *all* backing positions (one full-pool scoring forward/cycle) |
| `quest_like` | Quest-like recoverable control | page-granular (16 tokens) query-aware retrieval: pages ranked by max token score from the same full-pool scoring forward; top whole pages, last page truncated by token score to fill budget exactly.  Uses *exact* attention rather than Quest's min/max-key metadata bound — an upper-bound-strong simplification, labeled Quest-like, not a Quest reproduction |
| `statekv_exact_mean` | recoverable StateKV teacher/oracle | per cycle commits the minimum one-step exact-KL action from a 10-candidate panel: `stale, attention, snapkv, h2o, key_norm, value_norm, uniform, recency, qk_pool, quest_like`; each candidate evaluated teacher-forced against the same-input full-cache reference (P31 machinery) |

All budgeted arms: same samples, same budget, same cadence, same universe,
same mandatory structure, greedy generation, deterministic tie-breaks.

Irrecoverable references (H2O/SnapKV/old B2/Gate-0 arms) are *not* R0
competitors; they enter only the decomposition ladder (§6).

## 4. Substrate

Identical to Gate 0 (paired at the sample level with the Gate 0 runs):

- Model: `mlx-community/Qwen3-8B-4bit` @ `545dc425`, chat template,
  `enable_thinking: false`, greedy.
- Samples: `gov_report:86-90` + `synthetic_niah:86-90` (10 total),
  `data_seed 20260808`, ruler_niah ctx 768 offset 86, gov_report max_words
  700 indices 86-90 — byte-identical task overrides to
  `configs/stages/statekv_teacher_gate_g0.yaml`.
- 64 control cycles, horizon 1, budget 256/220, sink 4, recent 32.

## 5. Metrics

Primary: per-sample mean trajectory exact same-input KL over 64 steps;
aggregates = mean over the 10 samples.

Secondary/required:
- median and p95 of step-level exact KL per arm (pooled over samples×steps);
- paired per-sample wins/ties/losses and paired bootstrap 95% CI of the
  per-sample mean-KL difference (seed = data_seed, 20000 draws);
- task quality: NIAH `needle_retrieval_accuracy`, GovReport `rouge_l` and
  `official_score` (longbench), per arm and vs `full_cache`;
- fairness invariants: `all_budgets_respected`, per-cycle
  `maximum_active_cache_tokens ≤ 256`, candidate-universe equality (unit
  tests + recorded `candidate_universe_size`), refresh-cadence equality
  (one committed selection per cycle for every arm);
- recoverability behavior: `selected_recovered_layer_tokens` (re-entries
  from backing), `recovery_events`, per-layer churn
  (`selected_churn_layer_mean`), re-entry fraction;
- cost accounting: `pool_scoring_forward_time_s`, teacher candidates
  evaluated per cycle, forward counts, wall-clock per arm.

Quality-validity: the substrate is declared quality-valid iff
`full_cache` mean NIAH retrieval ≥ 0.8 on the 5 NIAH samples.  If not, the
gate is invalid and no verdict is issued.

## 6. Gap decomposition ladder (reported regardless of verdict)

```
pure best cheap (P35 b2_uniform, same samples)             [irreversible]
pure uniform / attention / snapkv (P35 arms)               [irreversible]
pure teacher_panel (Gate 0 arm)                            [irreversible]
    → recoverable uniform / recency                        [+ recoverability]
    → recoverable qk_pool / quest_like                     [+ query-aware retrieval]
    → recoverable statekv teacher                          [+ physical-risk scoring]
    → full_cache                                           [ceiling]
```

- D1 (recoverability): P35 `uniform` vs R0 `uniform` is the exact
  same-rule pair; P35 best cheap (b2_uniform) vs R0 best cheap is the
  practical pair.  P35 `attention` vs R0 `attention` is same-rule;
  P35 `attention` vs R0 `qk_pool` additionally includes the
  active-pool → full-pool fix and is labeled as such.
- D2 (query-aware retrieval): best simple recoverable (min of
  uniform/recency) vs best query-aware recoverable (min of qk_pool/quest_like).
- D3 (scorer headroom): best cheap recoverable B* vs teacher.

## 7. Preregistered verdict

Definitions: B* = the cheapest-kl non-teacher budgeted arm (lowest
mean trajectory KL among {uniform, recency, attention, qk_pool,
quest_like}).  "teacher" = `statekv_exact_mean`.

**GO requires ALL of:**

- G1 (headroom size): teacher mean KL ≤ 0.70 × B* mean KL
  (≥ 30% relative reduction; per the task brief, 1-3% margins are not GO,
  and ≈2× reductions like 0.10→0.05 are strong GO).
- G2 (stability): teacher beats B* on ≥ 8/10 paired samples AND the paired
  bootstrap 95% CI of (B* − teacher) per-sample mean KL excludes 0 above.
- G3 (tail): teacher p95 step KL ≤ 1.05 × B* p95 step KL.
- G4 (quality, on the quality-valid substrate): teacher mean official score
  ≥ B* − 1.0 point within each task bucket, and teacher mean NIAH
  retrieval ≥ B* − 0.1.
- G5 (fairness): all invariants in §2 verified (run flags + unit tests).

**NO-GO if any of G1-G5 fails.**  Sub-classification reported with the
verdict:
- `NO_HEADROOM`: teacher ≥ B* (ratio ≥ 1.0).
- `INSUFFICIENT_HEADROOM`: 0.70 < ratio < 1.0 or G2 fails — advantage too
  small/unstable to justify a student/reranker track.
- `INVALID_SUBSTRATE`: quality-validity check fails (no verdict on headroom).

No threshold in this section may be relaxed after results are observed.
Any post-hoc analysis is labeled exploratory.

## 8. Consequences

- GO → R1: candidate-generation + StateKV reranking feasibility (cheap
  retrieval proposes Top-M, physical-risk reranks to Top-K; measure the M
  needed to preserve the teacher headroom).  Not started before the GO
  verdict is written.
- NO-GO → clean negative closure of the recoverable pivot: document the
  decomposition (how much of P31 was recoverability vs retrieval vs
  scorer), the strongest recoverable baseline, and stop the method line.
  No scorer complexity increases, no hyperparameter rescue, no student.

## 9. Irrecoverable references — narrative boundary

SnapKV/H2O/old-B2/pure-eviction arms appear only in the decomposition
ladder.  Forbidden narrative: "StateKV Recoverable beats SnapKV → better
KV compression".  The only licensed claim shape is §1's question.

## 10. Implementation and artifacts (planned)

- Code: `statekv/oracle_policy_comparison.py` (new panel candidates
  `recency`, `qk_pool`, `quest_like` + pool-score plumbing),
  `statekv/oracle_policy_freegen.py` (pool scoring forward, churn/cost
  telemetry, generalized aggregation).
- Config: `configs/stages/statekv_recoverable_r0_qwen3_8b.yaml`.
- Tests: `tests/test_recoverable_r0.py` (re-entry legality, universe
  equality, budget invariants, same-input KL, quest page rule, recency
  rule, pool-score requirement, cadence).
- Results: `results/temporal_cache_discovery/statekv_recoverable_r0_qwen3_8b_v1/`.
- Analysis: `analysis/tables/recoverable_r0_headroom.{md,csv}` (+ ladder
  table), built by `analysis/tables/recoverable_r0_headroom.py`.
- Results doc: `analysis/statekv_recoverable_r0_results.md`.
- Claim registry: `configs/ccfa.yaml` sync.
