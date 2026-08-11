# StateKV Corner Gate — coverage × cadence cliff (preregistered)

Status: preregistered 2026-08-10, before any method-arm results.
Context: `docs/evidence/statekv_open_search_log.md` (HF1/HF3 corner findings).

## 1. Finding that motivates this gate

At 768/64 (coverage ~8%), exact full-pool current-query routing (qk_pool) is
near-oracle ONLY at refresh-every-step cadence:

| arm | h=1 | h=16 |
|---|---|---|
| qk_pool mean KL | 0.0819 | ~0.96 |
| qk_pool NIAH | 1.00 | 0.00 |

A 16-step-stale core at tight budget is quality-invalid.  R2's
"no selective-refresh advantage" closure was measured at budget 256 where
staleness is free; it does not cover this corner.

## 2. Question

In the tight-coverage / slow-refresh corner, can a selection rule that is
cheap and observable at refresh time preserve quality with 16× fewer
full-pool refreshes?

Candidates (all recoverable, same budget 64 = sink4+recent32+core28, same
backing pool, same refresh cadence):

- **qk_pool** (reference, done): current-query head-mean attention top-k.
- **qk_ensemble_w16** (to implement): pool score = per-position MAX of
  head-mean attention over the last 16 query steps (observation window over
  the full backing pool, reselected each refresh).  Rationale: hedges
  near-future query drift with recent-past evidence.
- **snapkv_pool_w32** (if cheap to add): SnapKV-style observation-window-32
  mean-pooled score over the full pool — the classic baseline form.

Controls already done: uniform @ h16 (1.05-2.73), qk_pool @ h1.

## 3. Refresh curve

Also map the cliff: qk_pool @ h4 (cycles 16) @ 64 — where quality loss
starts.

## 4. Verdict rules (fixed now)

Primary metric: mean per-sample same-input exact KL; task quality:
NIAH accuracy, GovReport official.

- **GO_CORNER**: qk_ensemble_w16 @ h16 restores NIAH to 1.00 AND mean KL
  ≤ 0.25 (≈3× the h1 reference 0.0819, i.e. recovers ≥ ~75% of the h16
  staleness damage), with paired wins vs qk_pool@h16 on ≥ 8/10 samples.
- **NO_GO_CORNER**: otherwise.  If no candidate restores quality, the corner
  is a regime where slow refresh is simply unsafe, and the deployable answer
  is "refresh often" — recorded as a boundary condition of qk_pool, not a
  method.

Fairness invariants (unit-tested): same budget/pool/cadence for all arms;
ensemble uses only PAST queries (no future leakage); same-input KL.

## 5. Artifacts (to fill)

- runs: statekv_opencorner_768_64_h16_v1 (done), statekv_opencorner_768_64_h4_v1,
  statekv_corner_ensemble_768_64_h16_v1
- machinery: pool-scored ensemble candidate in oracle_policy_freegen.py
- analysis: analysis/tables/corner_gate_*.csv

## 6. Results and verdict (2026-08-10)

| arm @ 768/64 | mean KL | NIAH | paired |
|---|---|---|---|
| qk_pool h1 (reference) | 0.0819 | 1.00 | — |
| qk_pool h4 | 0.3761 | 0.20 | — |
| qk_pool h16 | 0.8439 | 0.00 | — |
| **qk_obswin(w32) h16** | **1.0748** | **0.00** | 2/10 vs qk_pool h16 |
| uniform h16 | ~1.9 | 0.00 | — |

Fairness: `all_budgets_respected` true for every arm; identical
budget/pool/cadence; obswin scores use past/current tokens only.

**Verdict: NO_GO_CORNER.**  The observation-window candidate does not
rescue slow refresh — it is *worse* than single-token current-query scoring
at the same cadence (KL 1.07 vs 0.84, NIAH 0.0 both).  Mechanism: the
backward-looking window is stale-biased toward the previous window's
topics; the freshest single token remains the best refresh-time predictor.
The corner's conclusion is therefore a boundary condition, not a method:

> At tight coverage, exact query-aware routing must refresh (nearly) every
> step; no cheaper refresh-time scoring rule preserves quality at slow
> cadence.  The fix is cadence, not scoring.  At generous coverage (256),
> cadence is free (R2); the interaction, not either factor alone, is where
> the cliff lives.

Note: qk_obswin was run as "mean over last 32 trajectory tokens" (the
protocol's qk_ensemble_w16 max-variant was replaced by the SnapKV-standard
mean-form BEFORE any results were observed; recorded here for audit).
