# StateKV External-Validity Gate — report

Status: final
Date: 2026-08-10
Protocol & search tree: `statekv_external_validity_log.md` (preregistered
verdict rules R-A…R-E, substrate artifacts #3/#4/#5, all interim reads)
Verdict: **FINAL CLOSE** (all five reopen rules evaluated negative; the
768 closures generalize to 4.5–6× context, three task structures, two
budgets, three cadences)

## 0. What was challenged

All prior StateKV closures were established on a single small substrate
(768-token prompts, Qwen3-8B-4bit, NIAH + GovReport).  This gate
re-tested them at 3.1–4.7K-token prompts (4.5–6×), coverage 5.5–14%
@256 and 1.4–3.5% @64, with four task structures (single-needle NIAH,
4-needle multikey NIAH, GovReport summarization, distractor-heavy
reasoning), cadence h1/h4/h16, and a Qwen2.5-7B family confirmation.

Closures under challenge: C-a (qk_pool leaves no selection headroom),
C-b (its only failure is the coverage × cadence cliff; fix is cadence),
C-c (no QK-conditioned residual signal), C-d (all approximation variants
NO_GO).

## 1. Regimes tested (all runs completed, all budgets respected)

| regime | ctx (tokens) | core | cadence | arms |
|---|---|---|---|---|
| S0 (prior) | ~1.2K | 220/92/28 | h1, h4, h16 | qk_pool, quest_like, uniform |
| S1 3072/256 | 3.1–4.7K | 220 | h1 / h4 / h16 | qk_pool, quest_like, uniform |
| S2 3072/64 | 3.1–4.7K | 28 | h1 / h4 / h16 | qk_pool, uniform |
| S1-multikey | 4.8K | 220 | h1 | qk_pool, quest_like, uniform |
| S1-reasoning-AF | 3.1K | 220 | h1 | qk_pool, quest_like, uniform |
| S1-Qwen2.5-7B | 3.1–4.7K | 220 | h1 | qk_pool, uniform |
| decomp probe | 3.1–4.7K | 220 | h1 | qk_pool trajectory + swap oracle |

Substrate guard: `max_prompt_tokens` override + hard-fail truncation
invariant (artifact #3); tokenized prompt lengths verified per sample.

## 2. qk_pool at long context (per-step refresh)

| regime | NIAH KL / retrieval | GovReport KL | reasoning KL |
|---|---|---|---|
| S1 h1 | 0.019 / 3/3 | 0.033 | 0.007 |
| S1 h4 | 0.026 / 3/3 | 0.091 | 0.010 |
| S1 h16 | 0.059 / 3/3 | 0.313 | 0.025 |
| S2 h1 (core 28) | 0.066 / 3/3 | 0.248 | 0.024 |
| S2 h4 | 0.62 / 1/3 | 0.27 | 0.096 |
| S2 h16 | 1.15 / 1/3 | 1.16 | 0.57 |
| S1 multikey h1 | 0.061 / 3/3 (all 4 needles) | — | — |

Paired trajectory KL at S1: qk_pool beats quest_like 8/8 and uniform
8/8.  At 768 the corner cliff (NIAH failure at h4/h16) appeared at
core 28; at 3072 the identical signature appears at core 28 (NIAH 1/3
at h4 and h16, KL ~0.5–1.4) and is entirely absent at core 220 (NIAH
3/3 at every cadence, KL ≤ 0.06 even at h16).

## 3. Local-oracle (ranking) regret vs context — R-A: NOT triggered

Swap oracle (exact 1-step same-input KL of budget-preserving all-layer
cutoff swaps; 288 pairs / 48 cycles / 3 dev tasks per context):

| quantity | 768 | 3072 |
|---|---|---|
| pairs flat (<1e-4) | 92% | 91% |
| median regret | 2.0e-15 | 7.6e-15 |
| median rel. improvement | −4.6e-05 | −2.1e-05 |
| margin/Δ partial vs regret | ~0 | 0.011 / −0.033 |

R-A required >20% of cycles with median improvement >10% of residual
KL; observed median improvement is ~0.002%.  **The 1-step cutoff plateau
is context-invariant: no ranking headroom at 4.7K.**

## 4. Failure classification (every observed degradation)

- **S2 GovReport KL inflation (0.10–0.51)** at 52% eligible mass beyond
  the 28-token core → **coverage failure**; swap oracle flat at 256 and
  task score unreadable (saturated) — no ranking component.
- **Corner NIAH failures (S2 h4/h16)** → **refresh failure**: the same
  samples pass at h1 with KL 0.066; staleness of a 28-token core over
  4–16 steps, identical signature to 768.  Not ranking: at h1 the same
  budget routes perfectly.
- **Ranking failure: none observed anywhere** (swap oracle §3; quest_like
  loses 8/8 paired KL to qk_pool at S1 — approximation, not ranking).
- Boundary refinement discovered this round: the cliff's controlling
  variable is the **absolute core budget** (28 fails, 220 survives at
  every cadence), not relative coverage (S1 NIAH at 5.5% has lower KL
  than 768/64 at 8%: 0.019 vs 0.074 — homogeneous filler makes dropped
  mass self-redundant).

## 5. Staleness predictability — R-C: NOT triggered

Hard-cycle predictability at 3072 (cycle features vs cycle exact KL):
NIAH cycle-index ρ 0.36 (exact replication of the 768 +0.36), all
runtime features |ρ| ≤ 0.31, top-decile lifts < 3× — except reasoning
entropy lift 3.79×, a single-task anomaly (128 cycles, p90 KL 0.027 in
a regime with nothing to rescue).  The preregistered rule required ≥2
tasks + confirmation; not met.  No observable staleness precursor at
long context either.

## 6. Residual signals (QK-conditioned) — R-E: NOT triggered

Full residual battery on 71M token rows at 3072 replicates 768 exactly:
C1 near-tie partials (V | QK) −0.048..−0.103 in every cutoff bucket
(768: −0.053..−0.095); V features add ≤0.005 heldout Spearman to
future-relevance probes at h=1..8 (768: ≤0.006); revival predicted by
current attention (0.16–0.21), not V (~0.01); needle tokens sit outside
the core 94% of cycles with NIAH 1.0 — fetch-on-demand defuses delayed
cliffs even more strongly at long context.  No phase transition.

## 7. Task-conditional gaps — R-D: NOT triggered

Multi-fact retrieval (4 needles, 5.3% coverage): qk_pool 3/3 samples at
full retrieval, beats quest_like 2/3 paired KL (one sample ties within
0.004).  Reasoning: task-invalid at this operating point because the
full-cache model itself miscomputes (artifact #5) — contributes KL
evidence only.  No task structure produced a qk_pool failure at h1.

## 8. Model-family confirmation (Qwen2.5-7B)

S1 regime rerun on Qwen2.5-7B-Instruct-4bit (whitelist extension
smoke-validated: 28/28 hooks, budgets respected): qk_pool NIAH 3/3 with
KL 0.004–0.009 (lower than Qwen3-8B's 0.019), GovReport KL 0.009–0.055,
reasoning KL 0.003; uniform fails NIAH 0/3 (KL 0.9–2.0, repetition
degeneracy on two samples).  GovReport official saturation replicates on
the second family (uniform 8.84 > qk_pool 6.36 on gov_report:88).  The
closure pattern is not Qwen3-specific.

## 9. Artifacts / evaluator issues found this round

- #3 silent prompt middle-truncation at `max_prompt_tokens=1536` —
  would have faked the long-context substrate; fixed with
  `runtime_overrides` + hard-fail guard + regression tests.  The
  previously prepared (unrun) `statekv_openstress_3072_256.yaml` is
  marked superseded.
- #4 stage configs require `snapkv_*` keys (first queue launch died
  pre-measurement).
- #5 reasoning task unreadable at 64-token decode (derivation-first
  prompt; then full-cache arithmetic failure under answer-first) —
  reasoning declared KL-only at this operating point.
- GovReport official score saturated once more (uniform ≥ qk_pool on
  individual samples at S1/S2); used only as a sanity floor.
- Multikey full_cache drops 1/4 needles on one sample while qk_pool
  states all four — answer-formatting noise, explicitly not counted as
  a method advantage.

## 10. Answers to the 18 required questions

1. **Does the 768 closure generalize to true long context?**  Yes —
   every closure (C-a…C-d) replicated at 3.1–4.8K tokens.
2. **Regimes tested**: §1 (2 contexts × 2 budgets × 3 cadences × 4 task
   structures + swap-oracle probes + family check).
3. **qk_pool task quality / KL by regime**: §2 — task-perfect at h1 in
   every quality-readable regime down to 1.4% coverage.
4. **Local-oracle regret vs context**: flat at both contexts (§3);
   regret does not grow with context.
5. **New ranking failure at long context?**  None (swap oracle, paired
   KL, multikey).
6. **Is degradation still mostly coverage?**  Yes — the only h1
   degradation (GovReport KL at core 28) is coverage-driven; no task
   failure at h1 anywhere.
7. **Does the cadence × coverage cliff reproduce/strengthen?**  Yes,
   at the same absolute core budget (28), with the same per-task
   signature (NIAH breaks, GovReport KL-inflates, both ~1.0 KL at h16);
   and it is absent at core 220 at any cadence — a sharper boundary
   condition than the 768 round established.
8. **New revival/state-drift structure?**  No — revival rates are
   *lower* at long context (h1 1.6% vs 5.4%) and remain attention-
   predicted.
9. **Is staleness predictable?**  No (§5; one single-task anomaly below
   the preregistered bar).
10. **Any residual signal beyond QK?**  No (§6; 71M-row battery
    replicates 768 point-for-point).
11. **Does StateKV physical/state-conditioned machinery regain headroom
    in any new regime?**  No — the regime where it was already dead
    (h1 exact refresh) stays dead; the regime where everything fails
    (core-28 slow cadence) fails for qk_pool too, and the open search
    already gated the cheap-rescue arm NO_GO at 768; nothing at long
    context changes that calculus.
12. **Cross-task stable conclusions**: qk_pool h1 validity; plateau;
    QK⊇V; cliff signature.  All hold on NIAH + multikey + GovReport
    (reasoning KL-only).
13. **Task-specific conclusions**: GovReport official metric is
    non-discriminating at every operating point tested; NIAH-style exact
    retrieval is the only task that ever distinguishes arms.
14. **New evaluator/artifact issues**: §9 (#3, #4, #5) — all fixed or
    declared; no prior conclusion affected.
15. **Reopen method search?**  No rule triggered; see verdict.
16. **(If reopen)** — n/a.
17. **Directions to mark NEW PROJECT** (not StateKV continuation):
    systems acceleration of exact QK routing (transfer/latency
    accounting exists in R0 telemetry), coverage engineering (absolute
    core sizing), cross-session prefix reuse (needs harness), page
    approximation (bounded below token-level, §HF2 of the open search).
18. **Verdict: FINAL CLOSE.**  The closures survive a deliberate
    long-context, multi-task, multi-pressure challenge; the one genuine
    regime finding (the cliff) was refined, not overturned.

## 11. Artifacts

| artifact | path |
|---|---|
| Protocol + search log | docs/evidence/statekv_external_validity_log.md |
| Main tables | analysis/tables/extval_{main,paired,telemetry,cadence}.csv |
| Probes | analysis/tables/extval_{coverage_classification,hard_cycle_predictability,swap_regret}.csv, extval_probes_summary.json |
| Residual battery @3072 | analysis/tables/extval_qkv_*.csv/json |
| Runs | results/.../statekv_extval_{3072_256,3072_256_h4,3072_256_h16,3072_64,3072_64_h4,3072_64_h16,3072_256_mk,3072_256_reasoning_af,decomp_3072_256,7b_smoke}_v1/ |
| Configs | configs/stages/statekv_extval_*.yaml |
| Machinery | runtime_overrides + truncation guard (oracle_policy_freegen.py, qkv_decomposition.py), multikey generator + answer-first reasoning (tasks.py), fraction-found NIAH scoring, Qwen2.5-7B whitelist (config.py), parameterized residual battery |
| Tests | tests/test_external_validity_substrate.py (11 tests) |
