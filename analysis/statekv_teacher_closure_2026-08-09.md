# StateKV teacher-headroom closure — negative result with mechanism, 2026-08-09

Status: final
Scope: the principal scientific question "why would the expensive
state-conditioned physical-risk teacher beat cheap selectors, and can that
signal be distilled into a cheap strict-pure-eviction policy?" — answered as a
**negative closure** under the predeclared Gate 0-4 protocol, with the
mechanism mapped and reproducible artifacts for every gate.

## 1. The question and the protocol

The StateKV core premise — deletion risk depends on the current compressed
state — is supported in controlled settings (P1/P2).  The open question was
whether the *expensive state-conditioned physical-risk teacher* has headroom
over cheap selectors under **strict pure eviction** (deleted history never
restored), and whether that headroom can be distilled.

Gates (predeclared in the takeover brief):
- Gate 0: teacher vs best cheap, matched, strict pure eviction.  No headroom → STOP.
- Gate 1: fixed action space — scoring vs action-space advantage.
- Gate 2: teacher-gap decomposition (state-conditioning, propagation depth,
  additivity, boundary concentration, layer/trajectory structure).
- Gate 3/4: surrogate training and closed-loop (never reached).

## 2. Gate 0 — verdict: NO_HEADROOM (both substrates tested)

Substrate A (quality-valid): Qwen3-8B 4bit, 768 ctx, budget 256/core 220,
10 samples (gov_report:86-90, synthetic_niah:86-90), 64 tokens, strict pure
eviction.  Teacher = per-cycle minimum exact-KL action over a fixed panel of
cheap legal actions (attention, b2_uniform, a2_temporal_volatility, uniform,
snapkv, stale_prev), each evaluated as a counterfactual clone of the surviving
cache (new machinery `run_pure_eviction_policy(refresh_mode="teacher")`,
8 new unit tests).

- Teacher trajectory mean KL **0.2322** vs best cheap b2_uniform **0.0961**
  (teacher 2.4x worse; relative gain -141%); paired wins 2/10 vs b2_uniform,
  1/10 vs attention; step p95 1.094 vs 0.412; NIAH quality unchanged (1.0).
- Mechanism: **61.6% of cycles have panel one-step KL spread < 1e-3 (48.1%
  hard-tied)**.  The teacher selects the min among numerically-tied risks —
  a noise-driven random walk across the panel (selection distribution identical
  in tied and separated cycles: stale_prev 44%, uniform 21%, snapkv 17%,
  attention 5%) — and occasionally commits actions whose cores drop
  rarely-queried tokens (irreversible; the NIAH needle), causing late
  trajectory degradation.
- The P31 "teacher headroom" (statekv_exact_mean 0.0506 vs attention 0.3357)
  is a **machinery artifact**: P31 evaluated every candidate from the
  full-history anchor through a persistent KVBackingStore, i.e. with access to
  tokens already deleted from the live trajectory — the access that
  deployment semantics forbid.  Under the same score with state-conditioned
  evaluation, the teacher is worse than the best cheap policy.

Substrate B (quality-invalid): Qwen2.5-1.5B 4bit, 768 ctx, budget 128/core 92
(the shared-mask / lower-coverage regime where the refresh phenomenon was
real; P23b split, 6 samples).  Teacher wins 1/6 vs best cheap in trajectory KL
(0.06-0.75 vs 0.04-0.95); NIAH retrieval 0.0 for the teacher AND every cheap
policy (all compressed trajectories degenerate into repetition at this
budget), so the operating point is quality-invalid for all compressed
policies; no headroom in KL either.

## 3. Gate 1 — verdict: ACTION_SPACE_DOMINANT

On the teacher's own roll-in, the fixed action space is degenerate: mean
oracle action regret of the best cheap panel candidate is 0.0039 vs mean KL
0.2361 (1.7% relative; regret > 0 in 73.8% of cycles but the size is
noise-level).  The teacher's (absent) advantage is not a scoring advantage —
the panel actions are indistinguishable in one-step risk; a better action
generator (retaining future-queried tokens) is what would be needed, and none
of the cheap selectors can propose it.

## 4. Gate 2 — decomposition

- **2A state-conditioning**: moot at this operating point — one-step risk is
  flat for every action (Gate 1), so conditioning on the current state changes
  nothing.  The state-dependence of risk is real but cliff-shaped (binary:
  a state missing a future-queried token has risk ~2 for every action; a
  state retaining it has ~0), not graded.
- **2B propagation depth: DEEP_RISK but non-discriminative.**  Teacher-forced
  multi-horizon ladder (new machinery `run_ladder`, horizons {1,2,4} on clones
  of the surviving cache, 10 fresh samples): 55.4% of valid cycles tie at
  horizon 1 (spread < 1e-3); h1-vs-h4 top-1 ranking agreement 62.4%; 20% of
  cycles show the cliff signature (step KL ~0 at depth 1, > 0.1 at depth >= 2).
  But the attention family (attention/b2/a2/snapkv) stays tied at *every*
  horizon (h1 0.053, h4 0.28; regret ~0.001); only uniform separates (regret
  0.08 -> 0.49 with depth).  A deeper teacher would avoid uniform and gain
  nothing over the best cheap policy.
  - Protocol note: the ladder's committed probe KLs were initially inflated
    (different-input KL after a benign one-token "phase shift" in the
    repetitive filler; same-input full-state KL stays ~0; quality unaffected,
    NIAH 1.0).  All ladder statistics use the corrected pre-shift rows; the
    bug is documented in `statekv_ladder_2b_deep_risk.md`.
- **2C additivity: degenerate at depth 1.**  One-step boundary swap marginals
  are ~1e-5-1e-4 for every token class (just-below/just-above/mid-core/far-out)
  and pair interactions are exactly zero.  Additive top-k scoring over
  one-step marginals is well-defined but information-free.
- **2D boundary concentration: none exploitable.**  The risk carried by
  future-queried tokens is invisible at depth 1 (the only cheap observable),
  and the deep signal appears only 2-4 steps before the event — too late, and
  shared by every panel action.
- **2E layer/trajectory structure**: the degeneracy is uniform across tasks
  (NIAH and GovReport), samples (86-90, 96-100, 101-105), and both model
  families tested; no stable error structure to compress.

## 5. Why the teacher cannot win (synthesis)

At the tested operating points, one-step physical risk — the only quantity a
greedy teacher scores — sits on a plateau: every legal action has near-equal
one-step KL because the retained-mass coverage is high and the token the model
queries next is almost always retained by every action.  The long-run risk
differences between actions are carried by a small set of tokens queried later
(the needle), where the risk is a cliff (0 or ~2), not a graded marginal; the
cliff becomes visible to any evaluator only 2-4 steps before the query, at
which point every cheap selector has already dropped the token and pure
eviction forbids restoration.  Consequently: (i) the teacher cannot rank
actions (ties), (ii) the deep teacher cannot rank the good actions either
(attention-family ties at all horizons), (iii) the only thing that would help
— an action generator that retains future-queried tokens — is exactly the
multi-step lookahead the teacher itself requires, i.e. there is no cheap
signal to distill.

## 6. Boundary conditions (where this closure does NOT apply)

- Substrates/selectors that materially reduce retained-mass coverage below
  ~0.9 (shared-mask + low budgets + diffuse-attention small models) can
  restore graded risk (the P23b refresh phenomenon at 4-8K contexts,
  coverage 0.698, is real).  At the tested 768/128 budget the same substrate
  is quality-invalid (degeneration), so the teacher headroom question there
  remains open in principle; the existing refresh label machinery (R2a/R2b)
  is ready for exactly that check when a quality-valid low-coverage operating
  point is found.
- Longer generations, larger contexts, or different tasks were not tested.
- Value-aware / output-aware selectors that redistribute retained mass could
  re-enter a different coverage regime; none was found to work in this repo
  (P24 output-aware proxies: gate failed).

## 7. Positive mechanism findings retained

- The state-conditioned physical-risk *evaluator* is sound and cheap to
  compute per action (exact KL via counterfactual clones; P1/P2).
- The risk structure is cliff-shaped, and the cliff is predictable 2-4 steps
  ahead only at depth >= 2 (2B ladder) — a negative result for 1-step
  distillation, but a precise boundary condition for any future work.
- Strict pure eviction machinery (P35, teacher panel, ladder, marginal,
  refresh arms) is implemented, tested (43+ tests), and reusable.
- The P31 headroom is fully explained as a full-history-access artifact.

## 8. Reproducible artifacts

| Gate | Config | Raw results | Analysis |
|---|---|---|---|
| Gate 0/1 | configs/stages/statekv_teacher_gate_g0.yaml | results/temporal_cache_discovery/statekv_teacher_gate_qwen3_8b_g0_v1/ | analysis/tables/gate0_teacher_headroom.{md,csv}, gate0_paired_comparisons.csv, gate1_fixed_action_space.csv; analysis/statekv_gate0_1step_teacher_negative.md |
| 2B/2C | configs/stages/statekv_ladder_2b.yaml | results/temporal_cache_discovery/statekv_ladder_qwen3_8b_2b_v1/ | analysis/tables/ladder_2b_risk_depth.{md,csv}; analysis/statekv_ladder_2b_deep_risk.md |
| Refresh arms | configs/stages/statekv_refresh_arms_qwen3_8b_768_256.yaml | results/temporal_cache_discovery/statekv_refresh_arms_qwen3_8b_768_256_v1/ | analysis/tables/refresh_arms_summary.{md,csv} |
| P23b gate | configs/stages/statekv_teacher_gate_p23b.yaml, statekv_p2_p23b_cheap.yaml | results/temporal_cache_discovery/statekv_teacher_gate_qwen25_15b_p23b_v1/, statekv_p2_qwen25_15b_p23b_cheap_v1/ | (this document, section 2) |
| Machinery | statekv/statekv_gate_runner.py (teacher mode, ladder, marginal), statekv/budget_dynamics.py (uniform, shared_attention), tests/test_teacher_gate.py | | 43 tests pass |

All runs: strict pure eviction (`S_t ⊆ S_{t-1} ∪ {x_t}`, irreversible
inclusion verified per cycle, no persistent CPU KV backing), deterministic
greedy, matched samples and budgets, same-input exact-KL metrics.

## 9. Verdict

The expensive state-conditioned physical-risk teacher has **no deployable
headroom over cheap selectors under strict pure eviction at any tested
quality-valid operating point**, and the gap decomposition shows the reason is
structural (plateau + cliff, no discriminative signal at any horizon within
the panel).  Per the predeclared protocol, the StateKV method track is closed
as a **negative closure**; the mechanism, boundary conditions, and artifacts
above are the closure record.  The state-conditioned risk *evaluator* remains
a sound research instrument; the *teacher-as-selector* and any 1-step
distillation target are not supported.
