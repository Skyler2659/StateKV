# StateKV counterfactual-utility phase — preregistration

Status: **FROZEN before any new test opening** (2026-08-20).
Branch: `codex/statekv-counterfactual-utility`.
Previous phase frozen at tag `statekv-existence-gates-abc`
(Gate A FAIL / Gate B STRONG PASS / Gate C formal FAIL, success level S3).
Nothing in this document reinterprets or modifies those verdicts.

## 1. Hypotheses

**H1 — Reactivation Hypothesis.** The practical value of state-conditioned
future KV utility is concentrated in *relevance-reactivation* events:
information that is dormant (low importance) for an extended period becomes
critical later as the decoding state changes. Falsifiable prediction (§5):
per-sequence method gain correlates positively with the Reactivation Index.

**H2 — Utility Alignment Hypothesis.** Future attention is a misaligned
proxy for eviction utility. The quantity that should be predicted is
*counterfactual eviction utility*: how much future model behavior is
damaged if this token/group/action is removed now. Falsifiable prediction
(Gate D, §7): causal counterfactual utility correlates with realized
physical damage significantly better than R2 future-attention utility.

## 2. Task hierarchy (frozen)

Mechanism-first; tasks are NOT weighted equally.

**Tier A — primary mechanistic (efficacy benchmark):**
- `ruler_niah_multikey` (existing custom generator, `statekv/tasks.py`)
- `variable_tracking` (existing, `benchmarks/mlx/src/benchmarks/ruler.py`)
- `ruler_niah_multiquery` (to be added: same needle inserted under several
  distinct queries, all queried at generation time — minimal extension of
  the existing multikey generator)
- single `ruler_niah` as calibration only
- Linked/Sequential NIAH: deferred; build only if variable_tracking proves
  insufficient as a chain-reactivation probe (avoid benchmark engineering
  before the core experiment)

**Tier B — realistic retrieval/reasoning (generalization):**
- `passage_retrieval_en` (LongBench wrapper exists)
- `hotpotqa` (LongBench wrapper exists; multi-hop QA pick)

**Tier C — control / non-inferiority:**
- `gov_report`. No improvement required. Must not degrade beyond a
  non-inferiority margin δ (§8) frozen from validation before any test.

## 3. Splits (frozen index ranges, per task family)

Previous phase used offsets 151–160 (closed-loop test) and 161–199
(collection debug/train/validation/fresh_test). This phase uses disjoint
ranges:

Synthetic families (ruler_niah, ruler_niah_multikey, ruler_niah_multiquery,
ruler_variable_tracking) are generated with unlimited offsets:

| split | indices | role |
|---|---|---|
| train | 200–219 | RI parameter + damage-metric + baseline selection |
| validation | 220–229 | all frozen choices confirmed; Gate D rehearsal |
| fresh diagnostic test | 230–249 | Gate D, opened once |
| fresh closed-loop test | 250–279 | Gate E, opened once |

LongBench families (gov_report, passage_retrieval_en, hotpotqa): the local
THUDM/LongBench snapshot has exactly 200 rows per task, and rows 24–29,
61–63, 86–90, 96–105, 118–127, 151–199 were consumed by earlier phases
(conservative union over all historical configs, any task). The remaining
rows are allocated as:

| split | LongBench rows | count |
|---|---|---|
| train | 0–19 | 20 |
| validation | 30–39 | 10 |
| fresh diagnostic test | 40–59 | 20 |
| fresh closed-loop test | 64–85, 106–113 | 30 |

(Amended 2026-08-20 before any new collection or test opening; the
200–279 range does not exist for LongBench.)

Gate C test data (151–160) may be read for mechanism development only;
no confirmatory claim uses it. Each fresh test opening is recorded in a
ledger with a one-opening limit, same discipline as the previous phase.

## 4. Reactivation Index (RI) — full causal timeline

Computed **only from full-cache trajectories**; no compressed-policy output
is read. The timeline spans BOTH prefill query positions and decode cycles:
for each historical token i we record its received-attention trajectory
a_i(q) over subsequent query positions q (prefill queries aggregated into
fixed-size blocks, decode queries per cycle). Prefill is NOT collapsed into
a single scalar — that would average the filler dormancy and the
question-time reactivation together.

Definitions (parameters [K, L, τ_low, τ_high / rank quantiles, block size]
selected on train/validation only, then frozen before any test):

- **Dormancy**: token i stays below τ_low (or out of top-K) for ≥ L
  consecutive query positions/blocks.
- **Reactivation**: afterwards its rank crosses τ_high (or re-enters
  top-K) at some query/state q′.
- **Event** = dormant→important transition; per event record dormancy
  duration D_i, reactivation distance, amplitude A_i = r_{i,q′} − recent
  historical level.
- `RI_count` / `RI_fraction` (= events / all future-important events),
  sequence- and task-level.

Reactivation TYPE is reported per event (mechanism decomposition, not
separate hypotheses):

- **Type I — prompt-boundary**: dormancy ends at late-prefill query
  positions (canonical NIAH: needle dormant through filler, reactivated
  when the question is read).
- **Type II — decode-state**: dormancy ends during decode (variable
  tracking / sequential reasoning — the purest state-conditioned case).
- **Type III — persistent importance** (never dormant) is tracked as the
  complementary non-target pattern.

Expected task signature (to be verified on train/validation, NOT assumed):
single NIAH = high Type I / low Type II; multi-key NIAH = high I / medium
II; variable tracking = medium I / high II; GovReport = low/dispersed.

Mandatory sanity check before trusting any RI scalar: plot single-token
attention trajectories (needle, filler, GovReport-salient, VT variable
tokens) over the full prefill+decode timeline on train samples — a needle
must show "visible at insertion → long low plateau → sharp spike at
question/state change". If trajectories do not show this, the RI
definition is wrong, not the hypothesis.

Pre-registered revision rule: RI parameters may be revised on
train/validation only; the version frozen before the fresh diagnostic test
is final. Corrected hypothesis statement (supersedes any decode-only
framing):

**StateKV targets state-triggered relevance reactivation after dormancy,
regardless of whether the transition occurs during prefill or decoding.**

## 5. Pre-registered H1 prediction

x: sequence-level RI (frozen definition). y: method gain (counterfactual
teacher or R2 vs validation-selected strong baseline) on the primary task
metric. Report Pearson, Spearman, sequence-bootstrap 95% CI. Support for
H1 = significantly positive correlation with high-RI tasks benefiting and
GovReport neutral. No correlation → H1 reported as not supported.

## 6. Counterfactual teacher (CAUSAL_COUNTERFACTUAL_TEACHER)

At decoding boundary t, uses only current prefix + model + past state.

- Baseline branch: causal rollout from the current prefix (same machinery
  as R2 prefix-recomputation).
- Counterfactual branch: temporarily remove a candidate token/group, roll
  forward from the same causal state, measure future damage.

Two versions, reported separately:
- **CF-A forced-token (primary diagnostic):** removal branch replays the
  baseline branch's generated tokens as inputs; isolates the KV-removal
  effect from trajectory divergence.
- **CF-B free-generation (secondary):** removal branch generates freely;
  measures trajectory divergence + task behavior.

Candidate damage metrics: trajectory KL, JS, logit-L2, sequence NLL delta,
task-functional damage. The choice is made on validation and frozen;
no damage-metric selection on any test.

## 7. Gate D — Utility Alignment Gate (fresh diagnostic test)

Diagnostic: at sampled decoding boundaries, stratified candidate groups
(group sizes {4, 8}) drawn across age × current attention × EMA rank ×
R2 score × random strata (no cherry-picking salient tokens). For each
group compute four utilities — (A) current-QK, (B) fixed-EMA history,
(C) R2 future attention, (D) counterfactual — and independently the
**realized physical damage**: actually evict the group on a temporary
strict branch, continue a fixed horizon, measure damage. (D is never used
as its own ground truth.)

Frozen pass criteria (all on fresh diagnostic test, per budget):
1. mean Spearman(D, realized) − Spearman(C, realized) > 0 with
   sequence-bootstrap 95% CI lower > 0;
2. majority of sequences show D > C;
3. top-damage recall of D exceeds C by a margin [m] fixed on validation.

Also reported: Spearman / pairwise accuracy / NDCG for all four utilities.
Gate D FAIL → no counterfactual closed-loop; analyze the target instead.

## 8. Gate E — Counterfactual closed loop (only if Gate D PASS)

Method: CAUSAL_COUNTERFACTUAL_ACTION_TEACHER — enumerate matched-budget
candidate retention sets, causal-rollout each, execute the lowest-damage
action physically. Cost is allowed to be high (existence proof).

Baselines: NOT hand-picked. On validation, per task, select the strongest
of {QK-current, H2O, SnapKV, fixed-EMA} by the task's PRIMARY metric;
freeze as VALIDATION_SELECTED_STRONG_BASELINE.

Primary comparisons: (1) counterfactual teacher vs R2 future-attention
teacher (is the target the problem?); (2) counterfactual teacher vs
validation-selected strongest baseline (is it competitive?). Full cache is
reference only.

Metric ordering (frozen): Tier A/B primary = task success (needle accuracy
/ EM / official score); secondary = exact KL, JS, NLL. GovReport primary =
official score as non-inferiority control.

Pass conditions (frozen):
- Tier A: significant primary-metric improvement vs strongest baseline
  (paired bootstrap CI > 0, majority wins) in **≥2 distinct reactivation
  task families**;
- Tier B: stable positive generalization (not synthetic-only);
- GovReport: within the non-inferiority margin δ, where δ is frozen from
  validation (official score; distribution metrics secondary).

## 9. Cache settings, cost, forbidden actions

Budgets 128 and 256 retained; retention ratio reported alongside because
context length varies across tasks. Any ratio-based setting is frozen on
validation. Full cost accounting: wall time, teacher time, #rollouts,
FLOP proxy, peak/persistent memory; quality-vs-compute Pareto required.

Forbidden: rewriting frozen Gate A/B/C verdicts; defining reactivation or
δ after seeing tests; task names as a substitute for RI measurement; new
EMA/rank-drift heuristics; optimizing future-attention predictors;
merging counterfactual and R2 into one experiment; damage-metric
selection on test; claiming universal long-context improvement from
synthetic-only gains.
