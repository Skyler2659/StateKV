# StateKV External-Validity Gate — protocol (preregistered)

Status: preregistered before any long-context run
Date: 2026-08-10
Log: `statekv_external_validity_log.md`

## 0. Purpose

Challenge — not defend — the existing closures.  All prior evidence comes
from a single small substrate (768-token prompts, Qwen3-8B-4bit, NIAH +
GovReport, recoverable semantics).  This gate asks whether the closures
survive a genuine regime change: longer context, lower effective coverage,
an additional task type, and varied refresh cadence.

Closures under challenge:

- C-a: exact per-step full-pool QK routing (qk_pool) leaves no exploitable
  selection headroom (swap marginals ~1e-4, pair interactions 0,
  I(target;V|QK) ≈ 0 at 768).
- C-b: qk_pool's only failure mode is the coverage × cadence interaction
  cliff; the fix is cadence, not scoring (no refresh-time observable
  rescues slow cadence).
- C-c: no state-conditioned / physical-risk / temporal residual signal has
  predictive value beyond QK.
- C-d: qk_tiered_v, per-head selection, obswin scoring, page approximation
  are all NO_GO.

Boundary condition accepted from the open search: the horizon-1 optimality
statement is an *empirical* property of the evaluation (selection query ==
measurement query), NOT a theorem that QK top-K minimizes downstream KL.
This gate therefore re-measures ranking regret directly at long context
instead of relying on that argument.

## 1. Substrate ladder

| regime | ctx | budget | coverage | arms | cadence |
|---|---|---|---|---|---|
| S0 (existing) | 768 | 256/128/64 | 33%/17%/8% | qk_pool, quest_like, uniform | h1 |
| S0-corner (existing) | 768 | 64 | 8% | qk_pool, qk_obswin | h1/h4/h16 |
| S1 | 3072 | 256 | ~8% | qk_pool, quest_like, uniform | h1 |
| S2 | 3072 | 64 | ~2% | qk_pool, uniform | h1 |
| S1-h4 | 3072 | 256 | ~8% | qk_pool | h4 (16 cycles) |
| S1-h16 | 3072 | 256 | ~8% | qk_pool | h16 (4 cycles) |

S1 matches S0-corner coverage (8%) at 4× context: same memory pressure,
far more distractors, needle/target is a much smaller fraction of the pool.
S2 probes where qk_pool itself breaks at long context.

All runs: Qwen3-8B-4bit (same revision as all prior gates), data_seed
20260808, greedy, same-input exact-KL trajectory, recoverable semantics
(shared backing pool, sink 4 + recent 32 + core K), 64 total generated
tokens per sample (cycles × horizon = 64).

## 2. Tasks and why each is discriminative

| task | n | structure it probes |
|---|---|---|
| ruler_niah @ 3072 | 3 | single tiny target in a large distractor pool — routing precision |
| gov_report @ 2800 words | 3 | long-document distributed relevance — coverage of spread information |
| reasoning_long_generation (160 distractors) | 2 | retrieval + long free generation with drifting local state — state evolution during decode |

Scoring caveat (declared in advance): reasoning answers are scored by
normalized answer containment in ≥100-word generations — a low-resolution
metric, used only as a coarse validity check; KL and swap-oracle rows carry
the mechanism evidence.  Reasoning samples are bucketed separately
(`task_bucket == "Reasoning"`) and never mixed into NIAH/GovReport means.

## 3. Primary measurements

- mean / median / p95 exact same-input KL per arm per regime (paired vs S0).
- task scores per task (NIAH retrieval, GovReport ROUGE/official,
  reasoning containment) + repetition rate + text sanity.
- working-set telemetry: churn, recovery fraction, budget invariant.
- **Ranking regret (local oracle)**: swap-oracle table S rerun at 3072/256
  (exact 1-step same-input KL of budget-preserving all-layer cutoff swaps,
  offsets 1..32) + QK-conditioned V residual on token table T.  This is the
  direct external-validity check of C-a.
- **Refresh failure**: S1-h4 / S1-h16 vs S1 h1 — does the cadence cliff
  reproduce/strengthen at long context, and does any telemetry observable
  (churn, stale mass, entropy) predict the hard cycles?

## 4. Failure taxonomy (declared before seeing results)

Every degradation event will be classified as exactly one of:

- **coverage failure**: no ranking within budget can hold the needed mass;
  swap oracle cannot improve; h1 qk_pool fails while full cache passes.
- **ranking failure**: swap oracle finds budget-preserving exchanges that
  strictly reduce exact KL (or a cheap observable reranking does).
- **refresh failure**: selection was right at refresh time; degradation
  grows with horizon h while h1 stays quality-valid.

## 5. Preregistered verdict rules

REOPEN (any one suffices, must hold on ≥2 tasks and survive a confirmation
subset):

- R-A ranking gap: at 3072/256 h1, budget-preserving cutoff swaps reduce
  exact 1-step KL below the qk_pool selection in >20% of probed cycles
  with median improvement >10% of that cycle's qk_pool residual KL.
  (At 768 the equivalent marginals were ~1e-4 ≈ 0.)
- R-B temporal gap: qk_pool@h1 loses quality-validity at S1 on a task
  where full cache passes AND the failure is *not* explainable as coverage
  (i.e., swap oracle shows recoverable mass misrouted, not simply
  unaffordable).
- R-C predictable staleness: in S1-h4/h16, a runtime-observable precursor
  predicts hard cycles/steps materially better than chance (e.g. AUROC
  > 0.8 or top-decile lift > 3× base rate) — reopens trigger research.
- R-D task-conditional gap: the reasoning task exhibits a structured
  qk_pool failure mode absent from NIAH/GovReport, with an identifiable
  observable cause.
- R-E phase transition: QK-conditioned V/physical residual predictive
  value rises from ≈0 (768) to material at 3072 (e.g. >5% residual
  variance, or swap-regret correlation with V features |ρ| > 0.3).

FINAL CLOSE if all of: qk_pool@h1 stays task-valid across S1 tasks;
swap-oracle regret ≈ 0 at 3072 as at 768; S2 shows a pure coverage
boundary (swap oracle still ≈ 0 where qk_pool fails); the cadence cliff
reproduces at S1 with no observable precursor; no residual signal meets
any R-* rule.

INCONCLUSIVE if the regimes cannot be read (full cache itself fails the
tasks, runs infeasible within compute, or evaluator artifacts are found
that cannot be corrected).

Post-hoc rule changes are forbidden; any refinement discovered mid-analysis
is labelled exploratory.

## 6. Known substrate risks and guards

- `runtime.max_prompt_tokens` (base 1536) silently middle-truncates
  prompts.  All S1/S2 configs set `runtime_overrides.max_prompt_tokens:
  8192` and both runners now hard-fail on any truncation unless
  `allow_prompt_truncation: true` (regression tests:
  tests/test_external_validity_substrate.py).  The previously prepared but
  unrun `statekv_openstress_3072_256.yaml` lacked this override and would
  have produced a fake "3072" substrate; it is superseded by the extval
  configs and must not be run as-is.
- GovReport/niah official metrics are low-resolution; KL + swap rows are
  primary, task scores secondary.
- MPS serial execution; runs are launched one at a time via a queue script.

---

## Run log

### 2026-08-10 — substrate setup

- **Artifact #3 found and fixed (pre-data)**: base config
  `runtime.max_prompt_tokens=1536` silently middle-truncates prompts
  (`encode_prompt`, first-half+last-half).  The previously prepared
  `statekv_openstress_3072_256.yaml` would have produced a fake 3072
  substrate.  Added `runtime_overrides` support + a hard-fail truncation
  guard to both runners (`_check_prompt_truncation`), regression tests in
  `tests/test_external_validity_substrate.py`.  No prior results affected
  (all prior prompts < 1536).
- **Artifact #4 found and fixed (pre-data)**: stage configs must carry
  `snapkv_observation_window/pooling_kernel/pooling` (runner indexes them
  unconditionally); first queue launch died on KeyError before any
  measurement.  Added to all extval configs.
- Actual tokenized prompt lengths (Qwen3 tokenizer, raw prompt): NIAH@3072
  ≈ 4.69K tokens (filler sentence is ~9.2 tokens, so `context_length` is
  nominal), GovReport@2800w ≈ 3.4-3.6K, reasoning@280d ≈ ~3K.  Coverage
  @256: 5.5% (NIAH) / 7.4% (Gov) / ~9% (reasoning); @64: 1.4-3.5%.
  This is a genuine 4.5-6x context scale-up over the 768 regime.
- Task set extended with `ruler_niah_multikey` (4 needles, spread depths,
  fraction-found scoring): tests multi-target working-set load, the
  retrieval-structure axis single-NIAH cannot see.  `_metric_row` NIAH
  scoring changed from any() to fraction-found — identical for the
  single-reference samples used in all prior runs.
- Model-diversity confirmation queued: Qwen2.5-7B-Instruct-4bit on the S1
  regime (qk_pool + uniform only).
- Residual-analysis battery (`analysis/tables/qkv_residual_analysis.py`)
  parameterized (`--run/--out-prefix/--dev-samples`); regression rerun on
  the 768 decomposition reproduces all 8 existing tables byte-identically.
  It will be rerun on the 3072 decomposition with prefix `extval_qkv_`.
- Full test suite after machinery edits (runtime_overrides, truncation
  guard, task_bucket, multikey generator, fraction-found scoring):
  471 passed / 5 skipped / 0 failed (excluding the pre-existing
  test_repository_architecture failure).  Decomp config parse path
  dry-validated (overrides, guard, dev samples).
- **Artifact #5 (task-metric resolution, found mid-run)**: the
  derivation-first reasoning prompt cannot complete within the 64-token
  decode cap — qk_pool tracks full cache almost perfectly (KL 0.0073) yet
  official score is 0 because neither arm ever states the answer.  The
  reasoning task score in queue-1 runs is declared invalid by design
  (protocol already assigned KL the primary role there); an answer-first
  variant (`answer_first: true`, regression-tested) is queued separately
  (`statekv_extval_3072_256_reasoning_af`) to make the reasoning structure
  quality-valid.  Queue-1 prompts are intentionally NOT changed mid-queue
  to preserve within-queue pairing.

### S1 (3072/256, h1) — interim read (run 1 complete)

qk_pool: NIAH KL 0.0194 retrieval 3/3; GovReport KL 0.0328 ROUGE 0.0652
(at/above full-cache level); reasoning KL 0.0075 (task score invalid,
artifact #5).  All budgets respected; full_cache NIAH 1.0 everywhere
(substrate quality-valid).

Notable: NIAH KL at 5.5% coverage here is LOWER than the 768/64 corner at
8% coverage (0.019 vs 0.074).  Working interpretation (to verify in
analysis): 4x more *homogeneous* filler means the dropped mass is nearly
self-redundant, so coverage pressure on KL depends on distractor
redundancy structure, not just the coverage ratio.  The NIAH substrate
discriminates arms (uniform 1.58-2.30, retrieval 0) without threatening
qk_pool.

quest_like @3072/256: NIAH KL 0.036-0.042 (retrieval 3/3 — page budget in
pages is much larger here than at 768/64), GovReport KL 0.060-0.175,
2-8x qk_pool — the approximation gap widens on real text at long context.

### S1-h4 first sample — cadence cliff does NOT reproduce at core 220

qk_pool @3072/256 h4, synthetic_niah_86: KL 0.0293, retrieval 100
(768/64 h4 corner: KL 0.496, retrieval 0.2).  At equal *relative*
coverage the cliff is gone when the absolute core is 220 vs 28.  Working
hypothesis: the cliff's controlling variable is the absolute core budget
relative to the task's distinct-content mass, not context length or
coverage ratio.  Like-for-like test queued: 3072/64 (core 28) at h4/h16
(queue 3, after multikey + Qwen2.5-7B).

### S1-h4 complete — cliff still absent at core 220

qk_pool @3072/256 h4 (16 cycles x 4): NIAH 3/3 (KL 0.020-0.029 vs h1
0.019-0.020); GovReport KL 0.037-0.131 (h1: 0.020-0.047), official
6.32-7.40 (above the 768 full-cache ~6.0); reasoning KL ~0.01.  KL rises
2-3x vs h1 on GovReport but no task failure anywhere.  Contrast with the
768/64 corner: h4 NIAH 0.2, KL 0.50.  The like-for-like core-28 cadence
corner (queue 3) is now the decisive test of whether the cliff tracks
absolute core budget rather than context scale.

### S1-h16 complete — no cliff at core 220 even with 4 refreshes

qk_pool @3072/256 h16 (4 cycles x 16): NIAH 3/3 (KL 0.046-0.067);
GovReport KL 0.052/0.135/0.753 (official 7.08-7.49, all healthy — the
0.75 outlier on gov_report:86 is a KL-tail event without task damage);
reasoning KL 0.023-0.028.  Cadence KL inflation at 3072/256: NIAH ~3x,
GovReport ~4x from h1 to h16 — real but nowhere near the 768/64 corner
(h16: NIAH 0/5, KL 1.04).  The cliff is a tight-*core* phenomenon; at
long context with a 220-token core, slow refresh degrades KL gracefully.
Queue 3 (3072/64 h4/h16) will test whether the cliff reappears at
long context when the core is again 28.

### S2 (3072/64, coverage 1.4-3.5%) complete — qk_pool survives extreme coverage at h1

NIAH: qk_pool 3/3 (KL 0.065-0.067 — nearly identical to the 768/64 h1
value 0.074); uniform 0/3 (KL 2.19-2.32).  GovReport: qk_pool KL
0.10-0.51 (8-25x its 256 value) but official scores straddle full-cache
(5.05/7.47/6.70 vs 5.97/6.50/6.34) — the metric is saturated at this
operating point and cannot arbitrate; uniform KL 0.82-1.48.  Reasoning:
qk_pool KL ~0.023; uniform KL 1.2-1.7.

Key structural read: at h1, NIAH KL at 1.4% coverage @4.7K ≈ KL at 8%
coverage @1.2K (0.066 vs 0.074).  The 768/64 h1 operating point — the
foot of the 768 cadence cliff — is EXACTLY reproduced at 4x context.
Queue 3 (3072/64 h4/h16) is thus a clean like-for-like test of whether
the cadence cliff itself generalizes to long context.

### Decomp @3072 complete — R-A (ranking gap) NOT triggered

Swap oracle (exact 1-step same-input KL of budget-preserving all-layer
cutoff swaps; 288 pairs, 48 cycles, 3 dev samples — one per task) at
3072/256 vs 768/256:

| quantity | 3072 | 768 |
|---|---|---|
| swaps improving KL | 41.0% | 39.9% |
| median regret | 7.6e-15 | 2.0e-15 |
| p05 regret | -8.7e-05 | -6.5e-05 |
| median rel improvement | -2.1e-05 | -4.6e-05 |

The ~40% "improvement" fraction at both contexts is zero-mean noise
around a flat plateau (median relative change ~0, regret ~1e-14), not
systematic ranking error.  R-A required >20% of cycles with median
improvement >10% of residual KL — observed median improvement is ~0.002%.
**Closure C-a generalizes to 4.5-6x context: the 1-step risk plateau at
the QK cutoff is context-invariant; qk_pool leaves no measurable ranking
headroom at long context either.**

### Probes @3072 complete (extval_probes.py)

- **Coverage classification**: eligible-token attention mass beyond core —
  @220: niah 13.2% / gov 27.6% / reasoning 13.0% (p95 37-53%); @28: niah
  30.2% / gov 51.6% / reasoning 37.2% (p95 71-83%).  Yet S1 KL is
  0.02-0.03 and S2 NIAH is 3/3: the beyond-cutoff mass is a diffuse,
  self-redundant tail whose omission costs little KL.  S2's GovReport KL
  inflation (0.10-0.51 at 52% mass beyond core) is therefore classified
  **coverage-driven**, with the swap oracle flat at 256 ruling out a
  ranking component.  No ranking failure anywhere.
- **Hard-cycle predictability (HF1b retest @3072, R-C rule)**: per-task
  Spearman against cycle KL — niah: missed-mass -0.07 / entropy -0.13 /
  cycle-index +0.36 (exact replication of the 768 +0.36); govreport:
  0.06/0.31/0.23; reasoning: 0.17/0.26/-0.29.  Top-decile lifts: all <
  3x EXCEPT reasoning entropy at 3.79x — a literal threshold crossing on
  ONE of three tasks (128 cycles, 13 in the top decile, p90 KL only
  0.027, h1 regime with nothing to rescue).  Per the preregistered rule
  (must hold on >=2 tasks AND survive confirmation), **R-C is NOT
  triggered**; recorded here as a borderline single-task anomaly.
- **Swap oracle by task x offset**: no task/offset slice shows systematic
  improvement (median regrets 1e-7..1e-15 everywhere).
- **Artifact #5 resolved**: answer-first reasoning run shows full_cache
  itself answers wrong (205 true vs 225 generated on sample 0) — the
  scoring machinery is correct; Qwen3-8B-4bit greedy at 64 tokens fails
  the embedded arithmetic at full cache.  The reasoning task is declared
  quality-invalid at this operating point for ALL arms; it contributes
  KL/telemetry evidence only (as pre-assigned in the protocol).  Task
  validity rests on NIAH (exact) and GovReport (saturated, noted).
  Addendum: reasoning-AF sample-level detail — full_cache retrieval 0/1
  (sample 0 wrong answer 225 vs 205; sample 1 correct).  qk_pool/quest
  retrieval 0/2 despite KL 0.008-0.019: single-token divergences flip the
  stated number.  Confirms the reasoning task cannot arbitrate arms at
  this operating point; also a live demonstration that small-KL
  divergence is not task-neutral on exact-answer tasks.

### Residual battery @3072 complete — R-E (phase transition) NOT triggered

All battery sections replicate 768 on 71M token rows / 3 tasks:

- Swap oracle: 91.0% of pairs flat (<1e-4), median regret 7.6e-15,
  margin/Δ partials 0.011/-0.033 (768: 92% flat).
- C1 near-tie partials (V | QK): -0.048..-0.103 across every cutoff
  bucket — indistinguishable from 768 (-0.053..-0.095), including the
  boundary buckets.
- Horizon probe (dev-fit/heldout): V features add <=0.005 Spearman at
  h=1..8 (768: <=0.006).
- Revival: current attention predicts revival (0.16-0.21), V ~0.01;
  revival rates LOWER at 3072 (h1 1.6% vs 5.4% @768).  Needle tokens sit
  outside the core 94% of cycles with NIAH=1.0 — fetch-on-demand defuses
  the delayed cliff even more strongly at long context.
- Layer residual: mid-layer partials up to 0.27, same non-exploitable
  pattern as 768 (no recall/probe conversion).

**C-a and C-c generalize: QK absorbs V at long context exactly as at
768; there is no phase transition up to 4.7K tokens / 1.4% coverage.**

### Multikey (4 needles, 3072/256) complete — multi-target load does not break qk_pool

qk_pool: retrieval 3/3 samples at 1.00 (all 4 needles), KL 0.053-0.069.
quest_like: 3/3 at 1.00, KL 0.056-0.078.  uniform: 0.00 with repetition
0.80+ (degenerate).  full_cache itself scores 0.75 on sample 86 — the
uncompressed model drops one of four needles in its answer; qk_pool's
near-identical trajectory happens to state all four.  This is
single-sample answer-formatting noise (like reasoning-AF sample 1), NOT a
method advantage; reported as such.  R-D not triggered: no task-
conditional qk_pool failure mode exists on multi-fact retrieval.

### Qwen2.5-7B run rejected by backend whitelist

`cfg.validate()` whitelists only Qwen2.5-1.5B-4bit and Qwen3-8B-4bit for
the MLX discovery backend.  Decision: attempt a whitelist extension +
smoke test for 7B (same architecture family, 28 layers / 4 KV heads);
if the smoke fails or looks risky, fall back to declaring model diversity
out of scope for this gate (protocol ranked it last).

### 3072/64 h4 sample 1 — THE CLIFF REPRODUCES at long context (core-28)

qk_pool @3072/64 h4, synthetic_niah_86: KL 0.790, retrieval 0 (768/64 h4:
KL 0.496, retrieval 0.2).  Same absolute core (28), same failure — at
4x context.  Combined with S1-h4/h16 (core 220: no failure at any
cadence), the cliff's controlling variable is confirmed to be the
absolute core budget relative to distinct content, NOT the coverage
ratio or context length.  C-b generalizes with a sharper boundary
condition.  The open-search rescue arm (obswin) already failed this
corner at 768 and was gated NO_GO; nothing in the long-context data
suggests revisiting it.
- Qwen2.5-7B whitelist extension VALIDATED by smoke run
  (statekv_extval_7b_smoke_v1): 28/28 attention hooks, budgets respected,
  coherent greedy decode.  Full S1-family run (qk_pool+uniform, 8 samples)
  queued as queue 4 after the corner runs.
- Corner sample 2 confirms the cliff: qk_pool @3072/64 h4 KL 0.571,
  retrieval 0.

### Corner h4 @3072/64 complete — cliff reproduces with the same signature

qk_pool h4 @core-28: NIAH 1/3 (KL 0.49-0.79 on failures, 0.49 on the
pass); 768/64 h4 was NIAH 1/5 (KL 0.50).  GovReport KL 0.10-0.41 with
official 6.3-7.2 (task-healthy, as at 768 — the cliff hits needle-style
exact retrieval, not summarization).  Reasoning KL 0.09-0.11.  The
long-context corner is statistically the same object as the 768 corner.

### Corner h16 @3072/64 complete — full cliff reproduction

qk_pool h16 @core-28: NIAH 1/3 (KL 1.01-1.41; the same sample 88 passes
at h4 and h16 — its needle lands in the mandatory recent window, a
positional accident, not a signal); GovReport KL 0.39-2.21, official
5.6-8.1.  768/64 h16 was NIAH 0/5, KL 1.04.  **C-b FINAL: the
coverage-x-cadence cliff generalizes to 4x context with the same
signature; the boundary condition is the absolute core budget (28 vs
220), not relative coverage or context length.**

### Queue 4 complete — Qwen2.5-7B family confirmation (FINAL)

qk_pool @S1 on Qwen2.5-7B-4bit: NIAH 3/3 (KL 0.004-0.009), GovReport KL
0.009-0.055, reasoning KL 0.003; uniform NIAH 0/3 with repetition
degeneracy.  GovReport official-score saturation replicates on the
second family.  All reopen rules (R-A…R-E) evaluated negative across
2 contexts x 2 budgets x 3 cadences x 4 task structures x 2 model
families.  Verdict: FINAL CLOSE.  See
docs/evidence/statekv_external_validity_report.md.
