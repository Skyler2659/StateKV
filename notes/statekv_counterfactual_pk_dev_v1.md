# StateKV Student — first dev closed-loop PK (2026-08-21)

Status: DEV-SCALE signal check. n=3 per synthetic family, n=2 per LongBench
family. All differences below are within sampling noise unless stated.

## Setup

- Backbone: Qwen3-8B-4bit (4B disqualified at backbone qualification:
  multikey full-cache ceiling 0.325 vs 0.700).
- Student: `r2_student_mlp_v2.pt` (v2.1 objective: within-boundary percentile
  BCE + log-utility regression + per-horizon cutoff-straddling pairwise loss
  at core budgets 92/220, weight 0.1, 12 epochs). Validation vs R2 teacher at
  deployment horizon H1: Spearman 0.847, top-K recall 0.678 @ core 92 /
  0.744 @ core 220, ndcg +0.03 over the frozen v1 student.
- CF route: CLOSED (dev diagnostic, 540 groups x 9 sequences): first-order
  counterfactual utility is the worst of four utilities vs realized removal
  damage (KL Spearman 0.132 vs R2 0.174); within fixed group size every
  utility falls to rho 0.02-0.07. The earlier 0.929 debug signal was a
  salient-only selection artifact. No CFStudent will be built.
- Protocol: strict pure eviction, cold recovery 0, shared-token core,
  refresh=2 (frozen), budgets 128/256, 64 decode cycles, all structural
  audits green (170 matched arms, zero budget violations).
- Panels: `pk_dev_synthetic_v1` (multikey/multiquery/vt train 200-202),
  `pk_dev_longbench_v1` (passage/2wikimqa/hotpotqa/govreport train 0-1).
- Table: `results/statekv_counterfactual/pk_dev_v1_table.csv`
  (rebuild: `scripts/build_pk_table.py`).

## Official task score (dev, noisy)

| task | full | EMA 128/256 | H2O 128/256 | QK 128/256 | SnapKV 128/256 | Student 128/256 |
|---|---|---|---|---|---|---|
| multikey | 83 | 0 / 17 | 0 / 0 | 0 / 17 | 0 / 0 | 0 / 8 |
| multiquery | 100 | 0 / 100 | 0 / 33 | 0 / 100 | 0 / 0 | 0 / 100 |
| vt | 100 | 0 / 0 | 0 / 0 | 0 / 0 | 33 / 0 | 0 / 0 |
| passage | 50 | 50 / 50 | 50 / 50 | 50 / 50 | 100 / 50 | 50 / 50 |
| 2wikimqa | 50 | 0 / 50 | 0 / 0 | 0 / 50 | 0 / 0 | 0 / 50 |
| hotpotqa | 100 | 50 / 0 | 0 / 0 | 50 / 50 | 0 / 50 | 0 / 0 |
| gov_report | 6.5 | 6.1 / 5.5 | 5.9 / 5.1 | 6.4 / 5.5 | 6.4 / 5.3 | 5.8 / 5.3 |

## Read

- Student ties the best baseline on multiquery@256 and 2wikimqa@256, is
  mid-pack on multikey@256, loses hotpotqa@256 and vt in this tiny sample.
- Budget 128 does not discriminate on 768-token synthetic tasks (retention
  1/6): every matched policy scores 0 on multikey/multiquery/vt. Signal at
  128 exists only on LongBench.
- GovReport non-inferiority: student 5.8/5.3 vs full 6.5, baselines
  5.1-6.4 — within the metric's noise band.
- Student wall-time overhead: 55.4s vs 48.9-50.7s mean per arm (+~11%);
  R2 teacher arms would be orders of magnitude more expensive.
- R2 teacher column is absent by design (that cost is what the student
  replaces); Gate C's frozen R2 numbers stand for the old panel only.

## Next

- This n is too small to rank policies; any scale-up decision needs >= 10
  per family. The synthetic 128-budget regime is uninformative and should be
  replaced by a longer context or dropped for synthetic panels.
