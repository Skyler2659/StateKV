# StateKV existence study — final report (Gate C complete)

Date: 2026-08-20. Run: `statekv_causal_existence_qwen3_8b_seed20260820_v1`.
Model: `mlx-community/Qwen3-8B-4bit` (revision 545dc425, chat template,
thinking disabled). Config: `configs/statekv_existence/causal_existence_qwen3_8b.yaml`.
All gates evaluated with the frozen protocol; the formal test split
(`closed_loop_test`, 30 sequences: 10 synthetic NIAH + 10 GovReport +
10 multikey NIAH, offsets 151–160) was opened exactly once
(`test_open_ledger.json`).

## Verdict summary

| Gate | Question | Result |
|---|---|---|
| Gate A | Can a cheap learned predictor (hist_gbdt, H=32) recover oracle headroom? | **FAIL** (recovery 21.39%, CI [18.45%, 24.36%] < 25% threshold; 24/24 sequence wins — a stable positive signal, not a null) |
| Gate B | Can expensive causal rollout (R2 prefix-recomputation) recover it? | **STRONG PASS** (92.57%–96.24% recovery across H=1–32, 24/24 wins, CI lower ≥ 86.34%) |
| Gate C | Does R2 improve matched strict pure-eviction closed loop? | **FAIL** by the frozen rule (`gate_c_failed.json`) |

Success level: **S3** (expensive causal rollout recovers >50% of oracle gap).
S4 (causal method improves strict closed-loop pure eviction) is **not**
attained under the frozen criteria.

## Gate C formal result

Primary comparison: R2 causal rollout vs validation-frozen primary baseline
(per-head fixed EMA), sequence-level paired mean trajectory exact-KL
improvement, 20,000 cluster-bootstrap resamples, per budget:

| budget | mean KL improvement | 95% CI | sequence win rate | frozen rule |
|---|---|---|---|---|
| 128 | +0.343 | [−0.017, +0.712] | 20/30 (0.667) | FAIL (CI lower ≤ 0) |
| 256 | +0.133 | [−0.062, +0.334] | 16/30 (0.533) | FAIL (CI lower ≤ 0) |

Both point estimates are positive and both win rates exceed 0.5, but the
frozen rule requires CI lower > 0 at **all** budgets. Neither budget clears
it, so Gate C = FAIL. The sign consistency is real but the effect is not
statistically resolved at n=30.

All-baseline paired comparisons (mean KL improvement of R2 vs each; CI):

| baseline | b128 | b256 |
|---|---|---|
| QK-current | +0.439 [+0.011, +0.895] | +0.147 [−0.050, +0.361] |
| H2O cumulative | +0.439 [−0.029, +0.905] | **+1.055 [+0.686, +1.433]** |
| SnapKV obswin | **−0.463 [−0.931, −0.028]** | −0.046 [−0.217, +0.129] |
| fixed-EMA (primary) | +0.343 [−0.017, +0.712] | +0.133 [−0.062, +0.334] |

R2 does **not** dominate the field on KL: SnapKV's observation window is
significantly better at budget 128 and tied at 256.

## Task breakdown (primary comparison)

| task | b128 mean (win rate) | b256 mean (win rate) |
|---|---|---|
| NIAH (single + multikey, n=20) | +0.566 (75%) | +0.154 (55%) |
| GovReport (n=10) | −0.101 (50%) | +0.090 (50%) |

The KL benefit is entirely a NIAH phenomenon; on GovReport R2 is a coin
flip against the EMA baseline.

## Task metrics vs KL — the structural finding

Mean needle accuracy / official score (matched policies):

| policy | b128 needle / official | b256 needle / official |
|---|---|---|
| R2 causal | **0.575 / 40.7** | **0.663 / 46.3** |
| fixed-EMA | 0.300 / 22.3 | 0.588 / 41.2 |
| QK-current | 0.300 / 22.3 | 0.588 / 41.2 |
| SnapKV obswin | 0.013 / 3.1 | 0.463 / 33.0 |
| H2O | 0.000 / 2.1 | 0.000 / 2.1 |

R2 has the best task metrics at both budgets — including over SnapKV, which
wins the KL comparison at 128 while scoring near zero on needle retrieval.
Trajectory KL against the full-cache reference and task success are
measuring different things in this regime; a policy can track the
full-cache distribution closely (SnapKV) while evicting the tokens the task
actually needs, and vice versa. This mirrors the earlier FINDINGS A10
(KL–task decoupling) and limits what "KL improvement" alone can certify.

## Cost

R2 wall time ≈ 389–403 s/sequence (**≈20× full-cache reference**,
~5–6× the cheap policies), of which ~331–347 s is teacher recomputation
(32 refreshes at the frozen frequency 2). Gate B-level predictability is
therefore available only at a cost no deployment would accept; the
efficiency question (S5, distillation) is open — the validation
rollout-distilled student recovered only ~2.3% of the oracle gap
(leaderboard.csv, `rollout_distilled_mlp`).

## Answers to the standing questions

1. **Dynamic token-time oracle headroom exists?** YES (oracle decomposition:
   token-time oracle 0.8577 vs per-head fixed 0.7212 recall).
2. **Cheap causal predictor?** Partial: 21.4% stable recovery, below the 25%
   bar. History-only features carry ~nothing; the signal lives in
   query-conditioned token/state features.
3. **Expensive causal rollout?** YES, strongly (Gate B).
4. **Does it improve strict closed-loop pure eviction?** Not to the frozen
   standard. Positive but unresolved KL effect vs the EMA baseline;
   significantly worse KL than SnapKV at budget 128; best task metrics at
   both budgets.
5. **Consistency across budgets/tasks?** No: effect concentrated in NIAH and
   budget 128; GovReport flat.
6. **Cost?** ~20× full-cache wall time.
7. **Current bottleneck?** Not information (oracle headroom exists) and not
   causal predictability (Gate B). The failure sits in the
   **utility-target → eviction-quality conversion**: better future-attention
   ranking does not reliably translate into strict physical eviction quality
   measured by trajectory KL, and KL itself is partially decoupled from task
   success. Candidate mechanisms: utility-target mismatch (future attention
   ≠ counterfactual importance), irreversible-eviction interactions,
   shared-mask aggregation.
8. **Success level:** S3. S4 not attained; S5 (distillation) untested beyond
   the weak validation student.

## Interpretation (per protocol §14)

Gate B PASS + Gate C FAIL means: future-attention utility is causally
predictable, **but** better future-attention ranking does not translate into
strict physical eviction quality under the frozen metric. This is not
"StateKV is dead"; it localizes the gap. The preregistered next branch is
the **causal counterfactual KL teacher** (debug signal: removal-KL vs real
future attention Spearman ≈ 0.929), which targets counterfactual importance
directly rather than future attention mass. Any such follow-up is a new
experiment; Gate C as frozen is complete and closed.

A second, metric-level lesson: future protocols should pair any
distributional primary metric with a task-success requirement, because the
two diverged systematically in this test (SnapKV).

## Artifacts

- Merged formal data: `results/statekv_existence/causal_existence_qwen3_8b_v1/closed_loop/closed_loop_test/`
  (`sample_summary.csv`, `step_rows.parquet`, `paired_comparison.csv`,
  `protocol_summary.json`, `closed_loop_sequence_metrics.csv`,
  `closed_loop_step_metrics_by_sequence.csv`, `closed_loop_aggregate.csv`,
  `closed_loop_paired_bootstrap.csv`, `closed_loop_task_breakdown.csv`,
  `runtime_costs.csv`; shard provenance in `_shards/`)
- Gate decision: `gate_c_failed.json` (official pipeline),
  `gate_c_verdict.json` (full comparison pack)
- Plots: `plots/statekv_existence/gate_c_*.png`
- Analysis script: `scripts/analyze_gate_c_deliverables.py`
- Structural audit (2026-08-20): 330/330 arms, 30 unique samples, no
  duplicates/overlap; every matched arm strict_pure_eviction=True,
  recoverable_cold_tokens=0, peak active tokens exactly == budget; R2
  refresh_frequency=2 everywhere; shard protocol summaries identical.
