# Gate 0/1 negative result — the one-step state-conditioned teacher has no headroom under strict pure eviction

Status: final (2026-08-09)
Substrate: Qwen3-8B 4bit, 768 ctx, budget 256/core 220 (sink 4, recent 32), 64 generated
tokens, strict pure eviction (`S_t ⊆ S_{t-1} ∪ {x_t}`, no CPU backing, irreversible
inclusion verified per cycle), 10 samples (gov_report:86-90, synthetic_niah:86-90),
exactly the P35 p2 split so the teacher arm is matched against the existing P35
cheap trajectories on the same samples and budget.

## Question

Under deployment semantics (strict pure eviction, state-conditioned counterfactual
evaluation from the surviving cache), does the expensive state-conditioned physical
teacher (per-cycle minimum exact-KL action over a fixed panel of cheap legal
actions) still beat the best cheap selector?

## Machinery (new, tested)

`statekv/statekv_gate_runner.py::run_pure_eviction_policy(refresh_mode="teacher")`:
every cycle, forward the Full-KV reference, build a panel of cheap legal actions
(attention, b2_uniform, a2_temporal_volatility, uniform, snapkv, stale_prev) from
the *surviving* view, evaluate each action as a shallow-clone counterfactual forward
(one-step exact KL vs the Full-KV reference), commit the minimum-risk action, and
never restore deleted history.  This is the P31 teacher's exact score but with the
P31 re-anchor-from-full-history evaluation replaced by evaluation from the current
compressed state.  8 new unit tests (`tests/test_teacher_gate.py`, 43 total pass).

## Verdicts (predeclared, analysis/tables/gate0_teacher_headroom.py)

**Gate 0: NO_HEADROOM.**  Teacher trajectory mean KL **0.2322** vs best cheap
b2_uniform **0.0961** (relative gain −141%); teacher wins only 2/10 paired samples
vs b2_uniform (1/10 vs attention); step p95 KL 1.094 vs 0.412 (tail worse); NIAH
quality unchanged (1.0 vs 1.0).

**Gate 1: ACTION_SPACE_DOMINANT.**  On the teacher's own roll-in, the fixed
action space is degenerate: all panel candidates have nearly identical one-step KL
(best cheap panel candidate a2 mean 0.2361 vs min 0.2361−0.0039 regret = 1.7%
relative; regret > 0 in 73.8% of cycles but the *size* of the regret is noise-level).

## Mechanism (quantified)

1. **One-step risk is degenerate at this operating point.**  61.6% of the 640
   cycles have panel KL spread < 1e-3 (48.1% hard-tied at 1e-4).  The panel
   candidates differ in which ~220/1200 tokens they retain, but the retained-mass
   coverage is 0.998 (R2): the token the model actually queries at the *next* step
   is almost always retained by every panel action, so every action's one-step KL
   is ~0 or small.

2. **The teacher's selection is therefore a random walk over tied actions.**
   Teacher selection distribution is identical in tied vs separated cycles
   (stale_prev 44%, uniform 21%, snapkv 17-19%, attention 5%).  The min-KL action
   among ~0 values is chosen by floating-point noise.  The committed trajectory
   mixes attention-family actions with uniform/stale actions that spread the core
   evenly and drop rarely-queried-but-critical tokens (e.g., the NIAH needle at
   position ~933, outside the recent window).

3. **Risk is cliff-shaped, not graded.**  Dropping the needle is invisible in
   one-step risk (the needle is queried only at a specific later step), but when
   the model queries it the trajectory diverges: committed step KL jumps to
   ~2.1 and never recovers (sample synthetic_niah_86: KL 3.6e-4 at cycle 19 →
   2.10 at cycle 20 for *every* legal action from the post-commit state).  The
   per-cycle panel spread is bimodal: 61.6% tied, 2.0% > 0.1 (cliff).

4. **The P31 "teacher headroom" (statekv_exact_mean 0.0506 vs attention 0.3357,
   KL) was a machinery artifact.**  P31 evaluated every candidate from the
   full-history anchor through a persistent `KVBackingStore`
   (`statekv/oracle_closed_loop.py::_run_strategy` + `_rollout_candidate`), so the
   teacher could always see tokens already deleted from the live trajectory — the
   exact access the deployment semantics forbid.  Under strict pure eviction the
   same score with state-conditioned evaluation is *worse* than the best cheap
   policy.  Cross-machinery comparability is also invalid: P31's attention policy
   (0.3357) vs P35's attention (0.0976) at the same budget differ 3.4x on the same
   substrate.

## Why this is not "attention already wins, just use attention"

The finding is stronger than a method comparison.  It establishes that **one-step
physical risk cannot rank eviction actions at this operating point at all** — the
teacher itself fails, not just the cheap proxies.  Any one-step-based surrogate
(student, marginal scorer) inherits the same information bottleneck: 1-step KL sits
on a plateau, and the long-run consequence of the action (cliff vs no-cliff) is not
in the 1-step signal.  The interesting remaining question is the propagation depth
at which the cliff becomes visible (ladder 2B run, `statekv_ladder_2b` stage), and
whether the attention-family actions separate at depth (which would be the only
remaining distillable signal).

## Artifacts

- Teacher run: results/temporal_cache_discovery/statekv_teacher_gate_qwen3_8b_g0_v1/
  (sample_results.csv, step_rows.parquet, panel_rows.parquet, cycle_rows.parquet, summary.json)
- Cheap trajectories (matched): results/temporal_cache_discovery/statekv_pure_eviction_qwen3_8b_p35_v1/
- Tables: analysis/tables/gate0_teacher_headroom.{md,csv}, gate0_paired_comparisons.csv,
  gate0_step_tail.csv, gate1_fixed_action_space.csv, gate1_action_choice.csv
- Machinery: statekv/statekv_gate_runner.py (teacher mode + panel evaluator),
  tests/test_teacher_gate.py
- Config: configs/stages/statekv_teacher_gate_g0.yaml
