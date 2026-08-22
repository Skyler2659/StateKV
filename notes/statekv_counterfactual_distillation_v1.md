# R2 → Student distillation campaign — final report (2026-08-22)

Question: can a cheap causal student reproduce R2's retention decisions
closely enough to preserve its strict-eviction task gains?

## Data audit (passed before any retraining)

- teacher/feature (sample, cycle, position) join independently verified:
  teacher H=1 score vs artifact's own next-cycle attention Spearman 0.996
  (n=9792); 204 boundaries aligned; determinism checks pass;
  train/validation sequence sets disjoint.
  (`scripts/audit_distillation_data.py`)
- Noted: 39/120 feature columns are constant on training boundaries.

## Distillation variants (validation, deployment aggregation: H=1,
## layer/head-mean, full eligible sets)

| variant | top-256 recall | top-512 recall | Jaccard@256 | cutoff pair acc | band pair acc |
|---|---|---|---|---|---|
| v1 frozen (topk-BCE + reg) | 0.713 | 0.771 | 0.560 | 0.901 | 0.503 |
| v2.1 (percentile + cutoff pairs) | 0.713 | 0.773 | 0.561 | 0.897 | 0.507 |
| v3 base (pure ranking) | 0.698 | 0.753 | 0.543 | 0.889 | 0.503 |
| v3 + hard-neg round 1 | 0.633 | 0.714 | 0.479 | 0.838 | 0.503 |
| v3 + hard-neg round 2 | 0.697 | 0.772 | 0.542 | 0.892 | 0.503 |
| v2.1 + 4x data | 0.711 | 0.773 | 0.558 | 0.898 | 0.506 |
| v2.1 + wide MLP (256-256-128) | 0.717 | 0.781 | 0.566 | 0.901 | 0.508 |

Every lever in the diagnosis list was pulled: objective shaping (three
objectives), hard-negative mining (harmful: errors 34k -> 86k after round
1), 4x training data (no movement), model capacity (+0.004, noise).
Supporting measurement: a single (layer, head) teacher top-220 overlaps
the layer/head-mean consensus top-220 only 0.48-0.59 — the deployed target
is a cross-head consensus ranking.

**Conclusion: the 120-dim causal feature set caps distillation at ~0.71-0.72
top-256 recall, with near-chance ordering (0.51) inside the decisive cutoff
band. The binding constraint is feature information, not objective, data,
mining, or capacity.**

## Closed-loop @256 (same samples as the frozen teacher PK, strict pure
## eviction, refresh=2, cold recovery 0; all audits green)

| task | Full | R2 | QK | SnapKV | H2O | Student v2.1 |
|---|---|---|---|---|---|---|
| multikey | 70.0 | 32.5 | 20.0 | 2.5 | 5.0 | **15.0** |
| multiquery | 100.0 | 100.0 | 100.0 | 0.0 | 30.0 | **90.0** |
| 2wikimqa | 30.0 | 40.0 | 40.0 | 20.0 | 20.0 | **30.0** |

Inference overhead: student 72-86s per arm vs QK 68-70s (+4-26%); R2 teacher
524-772s (student is 7-9x cheaper than the teacher).

## Where the gap lives

- multiquery: student 90 preserves most of the (non-distinctive) score.
- 2wikimqa: student 30 = midpoint of R2/QK (40) and SnapKV/H2O (20) —
  preserves ~50% of the teacher's gain over weak baselines.
- multikey (the one task where R2 is DISTINCTIVELY better than QK):
  student 15.0 < QK 20.0. Per-sample: student ties R2 on 7/10 samples
  (25=25) but scores exactly 0 on the three samples (201/207/208) where
  R2's future-aware signal produces 25/50/75. The student's failures
  concentrate precisely where the teacher's unique information matters.

## Answers

1. Best distillation: v2.1 (or wide, within noise) at 0.713-0.717 top-256
   recall. No variant escapes the plateau.
2. Gain preserved: ~90% on multiquery, ~50% on 2wikimqa, and NONE of the
   distinctive teacher advantage on multikey (student below the QK baseline
   there).
3. Gap attribution (in the preregistered diagnosis order): not alignment
   (audit pass), not objective, not mining, not data volume, not capacity.
   The ceiling is the causal feature set; a secondary structural issue is
   that the teacher training data contains no multiquery/2wikimqa/passage
   sequences (existence-era families only), so deployment families are
   out-of-distribution for the student.

Implication: closing the multikey gap requires either richer causal
features or R2 teacher dumps on the actual deployment families (GPU cost),
not further objective/model tuning on the current data.
