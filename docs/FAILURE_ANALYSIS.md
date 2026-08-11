# StateKV failure analysis

Compiled 2026-08-10. Question addressed:

> Why did StateKV's mechanism experiments look decisive while the final
> deployable KV-compression policy showed no advantage?

Each explanation is graded: **SUPPORTED BY EVIDENCE** (a designed experiment
directly shows it), **PLAUSIBLE BUT UNPROVEN** (consistent with the data, no
direct test), **SPECULATION** (no test exists). Where no experiment can
answer, the question is listed as open.

---

## 1. Hypothesis failure — partially, and late

The narrow hypothesis "compression-history-conditioned physical
retained-set interventions can be propagated to downstream output risk" was
**not** falsified — it was confirmed (P0/P1/R4/P3PR, FINDINGS A1–A3). What
failed is the stronger deployability thesis built on top of it: that this
risk signal, made cheap, improves a deployable controller.

**SUPPORTED BY EVIDENCE**: the failure is not estimator error but the
*absence of a decision-relevant signal at decision time*. Gate 1 measured
oracle action regret of 1.7% inside the fixed panel; the ladder showed
one-step risk is numerically tied for every good action at every horizon
(plateau), while the long-run differences live in cliff events (a
future-queried token missing from the state: risk ~2, binary) that become
visible only 2–4 steps ahead — after every cheap selector has already
dropped the token and pure eviction forbids recovery
(`docs/evidence/statekv_teacher_closure_2026-08-09.md` §5).

In the vocabulary of the audit categories: this is neither proxy failure
nor optimization failure. It is a structural property of the operating
regime — the hypothesis that *graded, exploitable* state-conditioned risk
exists at decision time is what failed.

## 2. Oracle vs deployable gap — the central artifact

**SUPPORTED BY EVIDENCE**: the flagship teacher result (P31, KL 0.0506 vs
attention 0.336) was produced by machinery that evaluated every candidate
through a persistent full-history backing store — i.e. with access to
tokens already deleted from the live trajectory. Under deployment-faithful
semantics (strict pure eviction), the same teacher loses to the best cheap
policy (0.232 vs 0.096, paired 2/10). The gap between "teacher-forced
oracle with history access" and "deployable policy" was not a performance
gap; it was an access-to-forbidden-information gap
(`docs/evidence/statekv_gate0_1step_teacher_negative.md`).

Corollary that survives: under recoverable semantics, the comparison is
honest — and there the exact per-query full-pool router (qk_pool) beats the
physical-risk teacher 0.0086 vs 0.0213, paired 0/10 (R0). The expensive
risk machinery never won a deployment-faithful comparison anywhere.

## 3. Teacher-forced vs autoregressive gap

**SUPPORTED BY EVIDENCE**: P29 showed control horizon dominates: only H=1
survives free generation; H=4/H=8 lose trajectory KL to SnapKV because the
teacher-forced lookahead diverges from the realized greedy prefix
(`results/.../statekv_oracle_policy_freegen_independent_p30_v1/analysis/horizon_ablation.csv`).
Policies tuned under teacher-forced replay (P6–P16) repeatedly failed to
transfer to free generation (P18 NLL veto; P30 task gate).

## 4. Metric mismatch — real, and it cut both ways

**SUPPORTED BY EVIDENCE**:
- The joint gates gave KL/NLL/tail metrics veto power over task scores that
  were too coarse to discriminate (NIAH binary, 3–6 GovReport samples). The
  retest program showed some vetoes were miscalibrated (P9 win-rate gate:
  the policy leads on 24 fresh sequences; qk_tiered_v G5: unequal-memory
  comparator) — while others coincided with real boundaries (token rarity's
  GovReport gap persists; temporal volatility is not Era-2 competitive).
- Conversely, task-score ties did not imply equivalence: KL/tail metrics
  detected every eventual task collapse earlier (FINDINGS A10). So: KL is
  a *leading* indicator with no calibrated mapping to task harm; both
  "KL vetoed a good policy" and "task scores hid a bad one" occurred.

## 5. Small-sample and selection artifacts

**SUPPORTED BY EVIDENCE**: P30's NIAH gate rested on n=2; P29c's "better
task quality" on n=2; the Rademacher VJP dev gain evaporated on 8 fresh
sequences (retest Track D). The dev-screen → independent-reversal pattern
(P20→P21, P22→P23b) recurred often enough that development-set positives
in this codebase should be treated as hypotheses, not results.

## 6. Substrate dependence (model / mask semantics / coverage regime)

**SUPPORTED BY EVIDENCE**: the two eras behave oppositely on staleness
(P23b shared-mask coverage 0.70: real refresh value; Qwen3-8B per-layer
coverage 0.995+: time-invariant rankings, R2). Era-1 policy conclusions do
not transfer to Era-2 (temporal volatility; contribution family untested
there). External validity showed the *closures* do generalize across
context length and one additional model family — the negative results are
the robust part.

## 7. Evaluation-noise / instrumentation incidents

**SUPPORTED BY EVIDENCE** (documented, corrected): the ladder 2B committed
probe KLs were inflated by a different-input metric after a one-token phase
shift (`docs/evidence/statekv_ladder_2b_deep_risk.md`); the headwise probe
recorder had an indentation bug recording only layer 35 (fixed, rerun);
`open_stress_compare.py` mixed GovReport≈6 with NIAH=100 in one average
(caught). None changed earlier conclusions, but see CODE_AUDIT.md for the
silent-fallback patterns that made such incidents likely.

## 8. What does NOT explain the failure

- **Estimator inaccuracy**: the evaluator is exact (A1–A3). RULED OUT.
- **Optimization failure of the controllers**: cheap controllers achieved
  near-oracle one-step risk; the panel's oracle regret is 1.7%. RULED OUT
  by Gate 1.
- **Implementation bugs in the final policies**: selection hashes matched
  source replays exactly where audited (P12). RULED OUT for the main
  closures.

## 9. PLAUSIBLE BUT UNPROVEN

- The plateau is a *coverage* phenomenon: at retained-mass coverage ≈1
  every action keeps the next-queried token, so one-step risk ties. If a
  quality-valid operating point with coverage ~0.7 existed, graded risk
  might return (the P23b refresh phenomenon hints at it). No tested
  operating point is simultaneously quality-valid and low-coverage.
- Task-score robustness (A10) may reflect the metrics' insensitivity
  (ROUGE/containment) rather than true generation-quality robustness.
- qk_pool's dominance may partly be an evaluation-geometry artifact: at
  horizon 1 the selection query equals the measurement query (noted in the
  open-search report §1; external validity re-measured ranking regret
  directly and found none, which weakens but does not kill this concern).

## 10. SPECULATION (recorded, untested)

- A multi-step-lookahead action *generator* (not scorer) could escape the
  plateau — but Gate 2 showed the lookahead is exactly what requires the
  expensive rollouts.
- Value-aware redistribution of retained mass could re-enter a different
  coverage regime — P24's attempt failed, other variants untested.

## 11. Open questions the failure exposed

See FINDINGS.md §E (E1–E5). The single sharpest one: **what task-relevant
information does KL carry that current task metrics cannot see, and at what
KL does task harm actually begin?** No experiment in this repo calibrated
that mapping — it is the missing bridge between the two metric families
that drove every gate dispute.
