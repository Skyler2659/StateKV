# Rank-Migration Mechanism Analysis: Why Cheap-R2 Wins on Multi-key but Ties on HotpotQA (v1)

Date: 2026-08-25
Branch: `codex/statekv-counterfactual-utility`
Method frozen throughout: Cheap-R2 = STRICT_CAUSAL_ROLLOUT_R2, H=32, refresh f=16, budget 256.
No algorithm changes were made or evaluated. This campaign is pure mechanism diagnostics.

Data: per-token rank-migration instrumentation added to the strict closed loop
(`diagnostic_sink`, default-off; determinism verified bit-exact against the fresh runs).
110 diagnostic arms: multikey n=50 (280–329), hotpotqa/2wikimqa/gov_report n=20 each (40–59).
npz: `results/statekv_counterfactual/cheapr2_fresh_{multikey,longbench}_v1/rank_migration/_shards/`.
Analysis: `scripts/analyze_rank_migration.py` → `results/statekv_counterfactual/rank_migration_v1/`.

Core quantities per refresh cycle (R2 arm): for each eligible token i,
`rank_now(i)` = rank of current QK attention within the eligible set;
`rank_future_best(i)` = best rank of i across the 32 simulated rollout steps.
Reactivated := rank_now > B and rank_future_best ≤ B (B = core budget 220; B=256 robustness variant reported in CSVs).

---

## 1. Workload-level: generic rank churn does NOT distinguish tasks

| Metric | multikey | hotpotqa | 2wikimqa |
|---|---|---|---|
| reactivation_rate (mean over cycles) | 0.815 | 0.813 | 0.816 |
| reactivation_count_norm | 0.33 | 0.81 | 0.84 |
| reactivation_mass | 0.041 | 0.062 | 0.073 |
| severe_reactivation_rate | 0.034 | 0.048 | 0.050 |
| migration_magnitude (pct median) | 0.49 | 0.64 | 0.64 |
| top-B overlap (current vs future, mean) | 0.928 | 0.895 | 0.887 |

**Naive reactivation hypothesis is refuted at workload level:** LongBench QA has *more*
rank churn, not less — yet Cheap-R2 gains nothing there (35 = 35). The volume of
current→future rank migration is not the discriminating variable. This is exactly the
"filler churn" failure mode the old RI definitions were vulnerable to, now confirmed
with per-token ground truth from the actual rollout traces.

## 2. Cycle-level: all actionable future-demand shift happens at cycle 0

Per refresh-cycle breakdown (reactivation_count_norm / top-B overlap):

| Task | cycle 0 | cycle 16 | cycle 32 | cycle 48 |
|---|---|---|---|---|
| multikey count_norm / overlap | 1.32 / 0.723 | 0.005 / 0.996 | 0.005 / 0.996 | 0.005 / 0.996 |
| hotpotqa count_norm / overlap | 3.23 / 0.591 | 0.005 / 0.996 | 0.005 / 0.996 | 0.005 / 0.996 |
| 2wikimqa count_norm / overlap | 3.34 / 0.560 | 0.005 / 0.996 | 0.005 / 0.996 | 0.005 / 0.996 |

Two facts with direct methodological consequences:

1. **During steady decoding, current QK is a near-perfect predictor of future utility
   (overlap 0.996) in every workload.** The only state where "the future differs from
   the present" is the query-onset refresh (cycle 0). This retroactively explains two
   earlier negative results: why refresh triggers found nothing (there is no
   mid-decode surprise to trigger on) and why f16 suffices (refreshes after cycle 0
   are almost no-ops).
2. Multikey's cycle-0 overlap (0.72) is *higher* than hotpotqa's (0.59) — so again,
   raw instability does not predict gain.

## 3. The discriminator: reactivation lands on task-critical tokens

Multikey cycle 0 (the decision point that matters):

| Group | reactivated fraction |
|---|---|
| needle tokens | **47.3%** |
| filler tokens | 20.0% |

Needle tokens are ~2.4× more likely than filler to be current-dormant→future-important
at query onset. Retention outcomes (mean over refresh cycles):

- needle retained by R2: **93.1%**; by counterfactual-QK (same state): **86.2%**
- at cycle 0 the R2-arm state is identical to the QK-arm state (pre-divergence), so the
  cycle-0 counterfactual QK decisions are the QK arm's *actual* decisions.

## 4. Sample-level: gain is predicted by *what* is retained, not *how much* churns

Spearman ρ between per-sample mechanism metrics and Gain = Score(R2) − Score(QK),
multikey n=50:

| Metric | ρ | 95% CI |
|---|---|---|
| needle retained by R2 | **+0.53** | [+0.29, +0.71] |
| needle dropped by both (needle_neither) | **−0.59** | p<0.0001 |
| needle kept by R2 but dropped by QK (needle_r2_only) | +0.42 | p=0.002 |
| filler retained by R2 | **−0.56** | [−0.72, −0.35] |
| filler reactivated frac | −0.43 | [−0.64, −0.17] |
| generic reactivation rate/mass | ≤0.43, mixed signs | — |

Win-group (R2>QK, n=37) vs tie-group (n=13): needle_retained_r2 93.4% vs 92.4%
(diff +0.96pp, CI [+0.36, +1.57]pp) — small per-sample differences, consistently ordered
(Spearman picks up the monotonic relation; per-token margins are small but every
additional dropped needle sentence is a large score quantum).

On hotpotqa/2wikimqa no metric correlates with gain (all |ρ| ≤ 0.32, CIs cross 0) —
consistent with gain being noise around zero there.

## 5. Token-level causal chain (clean case)

Sample `synthetic_niah_multikey_287`, refresh cycle 0 (details in
`results/statekv_counterfactual/rank_migration_v1/case_synthetic_niah_multikey_287.csv`):

- needle position 202: rank_now **260** (outside top-220 → QK drops it) →
  rank_future_best **3** in the rollout → R2 retains it → realized real-decode
  attention over the next 16 cycles 0.0022 (~5× the median retained-token usage).
- positions 190, 486–487: same pattern (rank_now 226–254 → QK drops; future rank
  55–76 → R2 keeps; nonzero realized usage).

The full hypothesized chain — current-invisible → rollout-visible → R2-protected →
actually used later — is directly observed on needle tokens. Contrast case
`hotpotqa:40`: R2 likewise retains reactivated tokens that QK drops (e.g. rank 231→2),
but per-sample outcomes are identical to QK — the churn is not task-critical.

## 6. Verdict: **supported in sharpened form (between A and B)**

- **Refuted:** "reactivation happens more in multikey" — generic rank churn is
  ubiquitous and is, if anything, larger in LongBench QA.
- **Supported:**
  1. Future demand diverges from current importance essentially only at query onset
     (cycle 0); steady-state decode is rank-stable (overlap 0.996).
  2. In multikey, the current-dormant→future-important tokens at that boundary are
     precisely the task-critical needles (47% vs 20% filler).
  3. R2's advantage is retention-selective: it protects needles (ρ=+0.53 with gain)
     at the expense of filler (ρ=−0.56).
  4. In hotpotqa/2wikimqa the same or larger churn exists but is task-irrelevant:
     retention differences do not change outcomes.

**Paper-ready mechanism statement:**

> The value of future-aware eviction is not predicted by the *volume* of importance
> churn — churn is ubiquitous across workloads — but by its *semantic content* and
> *timing*. Future-aware retention pays off exactly at query-onset state transitions,
> and exactly when the tokens invisible to the current query are the ones the task
> will later require. Under aggressive budgets, current-query scoring systematically
> evicts dormant-but-future-critical evidence (multi-key: 47% of evidence tokens);
> a short causal rollout exposes this demand and protects it (needle retention
> 93.1% vs 86.2%), converting a 21% task score into 69.5%. On workloads where churn
> is task-irrelevant, the same machinery is neutral (35 = 35).

## 7. Honest limitations

- Needle ground truth exists only for multikey; the "task-irrelevant churn" claim for
  LongBench is inferred from outcome-equivalence, not from evidence-span labels.
- Counterfactual-QK retention is computed in the R2 arm's state; after cycle 0 the
  arms' trajectories diverge, so mid-decode counterfactuals are approximate. Cycle-0
  decisions are exact (identical pre-divergence state).
- needle_retained_r2 is partly near-tautological (keeping evidence helps); the
  non-tautological content is the *dormancy* of those needles at decision time
  (rank_now > B for ~17% of needle-token-refresh observations) plus the rollout's
  ability to see their future rank.
- gov_report collected as control but not central; see cycle_level.csv.
