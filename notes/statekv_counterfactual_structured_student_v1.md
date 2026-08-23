# STATEKV_STRUCTURED_STUDENT v1 — structured R2 distillation

Date: 2026-08-23. Branch: `codex/statekv-counterfactual-utility`.
Code: `statekv/structured_student.py`, driver `scripts/train_structured_student.py`,
tests `tests/test_structured_student.py`, configs
`configs/statekv_counterfactual/structured_student_qwen3_8b.yaml` (training) and
`structured_student_pk_synthetic_qwen3_8b.yaml` (closed loop).

## Motivation

The pooled student (120-dim `artifact_boundary` features, small MLP) plateaus at
top-220 recall ~0.71 and near-cutoff band pair accuracy ~0.51 against the R2
teacher (H=1). Hypothesis: averaging over 6 layers x 8 KV heads and per-token
independent scoring destroys the signal. The structured student keeps one
19-dim feature vector per (token, layer, head) — attention-history scalars,
current QK geometry vs the 4-head query group, query-trajectory interactions
(lags 1/2/4), K/V norms — plus per-layer state features (40), token features
(2), globals (6). The model encodes each (l, h) with a shared MLP plus
layer/head embeddings, aggregates the 48 heads by mean/max/attention-pool,
adds a per-layer state summary and a DeepSets context over eligible tokens,
and scores each token once. A per-head readout is supervised with a
percentile-BCE auxiliary against the per-head teacher. One feature builder
(`structured_boundary`) consumes artifact-shaped mappings from both the
training npz and `RuntimeFeatureHistory.artifact_view()`, so train/serve
consistency is by construction (parity and causality tests in
`tests/test_structured_student.py`).

Training: 30 sequences x 4 teacher cycles = 120 full boundaries, H=1 targets,
AdamW 1e-3 / wd 1e-4, 20 epochs on CPU (MPS verified working but slower at
this model size), cutoff-weighted pairwise loss at k in {92, 220} + 0.1 soft
percentile BCE + 0.3 per-head percentile BCE. Scalers fit on train only and
stored in the checkpoint (`kind="structured_mlp"`).

## Validation (21 sequences x 4 cycles = 84 boundaries, one code path)

`student_mlp_v2` row reproduces the frozen v2.1 numbers exactly
(0.7133 / 0.5070), confirming comparability.

| method | recall@220 | jaccard@220 | cutoff_pair@220 | band_pair@220 | recall@476 |
|---|---|---|---|---|---|
| student_mlp_v2 (old) | 0.7133 | 0.5608 | 0.8971 | 0.5070 | 0.7731 |
| structured_full | 0.7889 | 0.6623 | 0.9350 | 0.5108 | 0.7995 |
| structured_no_query_traj | 0.7969 | 0.6733 | 0.9420 | 0.5088 | 0.8113 |
| structured_no_head_identity | 0.7970 | 0.6702 | 0.9430 | 0.5002 | 0.8085 |
| structured_no_context | 0.7996 | 0.6765 | 0.9405 | 0.5090 | 0.8151 |
| structured_no_head_structure | 0.7929 | 0.6674 | 0.9346 | 0.5106 | 0.8027 |
| structured_no_perhead_aux | 0.7937 | 0.6684 | 0.9390 | 0.4957 | 0.8157 |

Per-task topk_recall@220:

| task | old v2.1 | full | best ablation |
|---|---|---|---|
| ruler_niah | 0.6989 | 0.8211 | 0.8318 (no_context / no_query_traj) |
| govreport_or_qmsum | 0.6474 | 0.6894 | 0.7179 (no_head_identity) |
| ruler_niah_multikey | 0.7935 | 0.8562 | 0.8619 (no_context) |

Ablation reading: no single feature group is responsible for the gain —
every ablation stays within ~1 recall point of `full` (and several are
slightly above it). The improvement comes from the richer per-head feature
set itself (structured boundary), not from head identity embeddings, the
query-trajectory terms, the DeepSets context, the preserved (l, h)
aggregation, or the per-head auxiliary loss. `no_head_structure` — which
pools the same 19-dim features over (l, h) into a plain MLP — loses only
~0.4 recall points, so "keeping structure" per se is not what matters.

Crucially, `band_pair_accuracy@220` stays at ~0.50-0.51 for **every**
variant, including the old student. The near-cutoff ordering plateau is not
broken.

CSVs: `results/statekv_counterfactual/student_models/structured_validation_cutoff.csv`,
`structured_comparison.csv`, `structured_comparison_by_task.csv`.

## Closed loop (synthetic multikey @256, train samples 200-209, STRICT_STATEKV_STUDENT)

Run: `results/statekv_counterfactual/structured_student_pk_synthetic_v1/closed_loop/train/_shards/structured_v1`.

| sample | Full | R2 | QK | SnapKV | H2O | student v2.1 | structured |
|---|---|---|---|---|---|---|---|
| 200 | 75 | 25 | 0 | 0 | 0 | 25 | 0 |
| 201 | 100 | 25 | 25 | 0 | 0 | 0 | 0 |
| 202 | 75 | 25 | 25 | 0 | 0 | 0 | 25 |
| 203 | 100 | 25 | 25 | 25 | 50 | 25 | 50 |
| 204 | 25 | 25 | 0 | 0 | 0 | 25 | 0 |
| 205 | 25 | 25 | 25 | 0 | 0 | 25 | 0 |
| 206 | 25 | 25 | 25 | 0 | 0 | 25 | 0 |
| 207 | 100 | 50 | 25 | 0 | 0 | 0 | 25 |
| 208 | 100 | 75 | 25 | 0 | 0 | 0 | 25 |
| 209 | 75 | 25 | 25 | 0 | 0 | 25 | 25 |
| **mean** | **70** | **32.5** | **20** | **2.5** | **5** | **15** | **15** |

Structured student scoring overhead: 0.0795 s/cycle mean vs 0.2482 s/cycle
for the old pooled student (~3x cheaper, one boundary build + one forward per
cycle instead of 48 `artifact_boundary` builds); per-sample student time
5.09 s vs 15.88 s.

## Verdict

- The top-220 recall plateau **is broken**: 0.713 -> 0.789 (full) / 0.800
  (no_context), with gains on all three task families, largest on
  ruler_niah (+0.12) and multikey (+0.06).
- The near-cutoff band plateau is **not broken**: band pair accuracy remains
  ~0.51 (chance) for every variant.
- Closed loop **did not improve**: structured student multikey@256 = 15.0,
  identical to the old student, below QK=20 and far below R2=32.5. Samples
  201/207/208 score 0/25/25 vs R2 25/50/75 — 207 and 208 improve over the
  old student (0 -> 25) but only to QK level; 201 stays 0.
- Interpretation: the R2 teacher's eviction-critical signal is the
  near-cutoff ordering, and that ordering is not recoverable from
  runtime-causal per-head features either — consistent with an information
  ceiling rather than a feature-pooling artifact. The recall gain is real
  but lives in the easy part of the ranking (cutoff pair accuracy
  0.897 -> 0.935), which does not move eviction decisions at the cutoff.

No tuning on validation beyond the listed ablations; frozen teacher-PK
results, old checkpoints, and `artifact_boundary` were not modified.
