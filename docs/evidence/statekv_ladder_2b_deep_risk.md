# StateKV 2B/2C — candidate risk is deep and non-discriminative; one-step risk is flat

Status: final (2026-08-09; corrected 2026-08-09 after a probe-metric bug in the
ladder committed KLs was identified and fixed)
Substrate: Qwen3-8B 4bit, 768 ctx, budget 256/core 220, strict pure eviction, 10
fresh samples (gov_report:101-105, synthetic_niah:101-105), attention trajectory
committed every cycle; at every 4th cycle the panel (attention, b2_uniform,
a2_temporal_volatility, uniform, snapkv) is rolled out teacher-forced at horizons
{1,2,4} on clones of the surviving cache, plus 1-step boundary swap marginals at
every 8th cycle.

## Metric protocol (amended, documented)

The ladder's committed/probe KLs compare the candidate logits against the
reference probe logits whose input is the *reference* token.  Once the compressed
trajectory deviates from the reference (a benign one-token "phase shift" in the
repetitive filler; see below), the probe KL is a different-input KL and inflates
to 10-46.  The valid same-input metric (full-state forward per cycle, the repo
standard) stays ~0 after the shift.  All ladder statistics below use only
pre-shift cycles (shift detected per sample as the first cycle where the probe KL
departs from the arms same-input KL by > 1.0; 101 of 160 measured cycles valid).

## Verdicts

**2B (propagation depth): DEEP_RISK, but non-discriminative.**  55.4% of valid
cycles have all panel candidates tied (spread < 1e-3) at horizon 1; the
horizon-1 vs horizon-4 top-1 ranking agrees only 62.4% of the time; 20% of cycles
have a candidate whose step-KL is ~0 at depth 1 but explodes (>0.1) at depth >= 2
(the cliff signature).  Crucially, the deep separation is *only* between uniform
and the attention family: attention/b2_uniform/a2/snapkv stay tied at every
horizon (mean step KL 0.053 @ h1, 0.28 @ h4 for all four; horizon-4 oracle regret
~0.0006-0.0013), while uniform's regret grows with depth (0.08 @ h1 -> 0.49 @ h4).
A deeper teacher would avoid uniform but gains nothing over the attention family.

**2C (additivity): degenerate at depth 1.**  One-step boundary swap marginals are
~1e-5-1e-4 for every token class (just-below-core, just-above-core, mid-core,
far-out) and pair interactions are exactly zero.  The one-step risk landscape is
flat: additive top-k scoring over one-step marginals is well-defined but
information-free.

## The mechanism (sample synthetic_niah_101, quantified)

1. The NIAH needle value is queried at generated step ~0-1 (the reference's first
   token is the number); the attention core from prefill contains it; both the
   reference and the compressed model emit it correctly (retrieval 1.0).
2. At cycle ~41-44, the compressed model skips one filler token ("the" in the
   repetitive "the sky is blue and the grass is green" phrase) — a one-token
   phase shift.  The cache-fidelity loss that precedes it is visible at depth 2
   only 1-2 steps ahead (h2@40 = 0.84-0.95 for every panel action) and at depth 4
   up to ~16 steps ahead (h4@24-32 = 0.04-0.06), but is invisible at depth 1
   (h1 = 5.9e-7 at cycle 40).
3. The shift is quality-neutral: the generated text still contains the needle
   value; NIAH retrieval 1.0 and official score 100 for every arm; the repo's
   same-input KL stays ~0.02-0.06 after the shift.
4. All panel actions share the deep dips (no action discriminates), and no panel
   action can prevent the phase shift — the teacher cannot help even with
   horizon 2-4 (all candidates tie at the decisive cycles).

## Boundary condition (R2 correction — none needed)

The R2 conclusion ("rankings time-invariant on qwen3-8b per-layer at coverage
0.998; refresh benefit noise") holds on the fresh samples too: the refresh-arm
comparison (never vs every vs fixed_k16, all same-input metrics) shows every
refreshing is best on all 10 NIAH samples (mean KL 0.024 vs never 0.346) and
NO_CLEAR_REFRESH_ADVANTAGE for never.  The earlier "attention catastrophically
diverges on 4/5 fresh samples" reading was a probe-metric artifact; the real
event is the benign one-token phase shift.

## Implication for the teacher headroom question

The teacher (any horizon) cannot improve over the attention family at this
operating point: its panel contains no action that discriminates at any horizon,
and the deep-risk events it could detect (phase shifts) are quality-neutral.
Combined with Gate 0 (NO_HEADROOM for the 1-step teacher) and the flat 1-step
marginal structure (2C), the risk structure leaves no exploitable signal for a
state-conditioned physical-risk surrogate at this operating point.

## Artifacts

- Ladder run: results/temporal_cache_discovery/statekv_ladder_qwen3_8b_2b_v1/
- Refresh arms: results/temporal_cache_discovery/statekv_refresh_arms_qwen3_8b_768_256_v1/
- Analysis: docs/evidence/tables/ladder_2b_risk_depth.md, refresh_arms_summary.md
  (CSVs of the same names in analysis/tables/)
- Machinery: statekv/statekv_gate_runner.py (_ladder_rollout, _marginal_measurement,
  _swap_selection, run_ladder stage), tests/test_teacher_gate.py
- Configs: configs/stages/statekv_ladder_2b.yaml, statekv_refresh_arms_qwen3_8b_768_256.yaml
