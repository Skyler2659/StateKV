# StateKV method gate: QK-route, V-tier (qk_tiered_v)

Status: preregistered before any gate run
Date: 2026-08-09
Parent: `statekv_qkv_discovery_results.md` (routing closure + Family C pivot)

## 1. Method (derived from the mechanism finding)

Mechanism premise: on this substrate, routing carries the information
(Var log attention 1.3-4.5 vs Var log ‖vW_O‖ 0.01-0.14) and the V payload
is nearly uniform.  Therefore V *precision* should be cheap to sacrifice,
and the freed memory convertible into QK coverage.

Method rule (the entire method):
1. Score the full historical pool with the current query's exact head-mean
   attention (identical to qk_pool; one scoring forward per cycle).
2. Retain top-(M − sink − recent) eligible tokens per layer
   (M = 352; sink 4, recent 32 → core 316).
3. V-precision tiering of the *active* rows: hot set = sink + recent +
   top-H attention within the retained core (H = 96) keeps FP16 V; all
   other retained ("cold") rows carry V at 4-bit precision
   (per-token-per-head symmetric absmax quantization, group 64 over the
   128-dim head vector, quantize→dequantize at cache build; K stays FP16
   everywhere).

Memory model (arithmetic, per token per layer; 8 KV heads × 128 dim):
K FP16 = 8×128×2 B = 2048 B; V FP16 = 2048 B → FP16 token = 4096 B.
Cold tiered V = 8 × (128×0.5 B + 2 scales×2 B) = 544 B → cold token =
2048 + 544 = 2592 B (0.633× FP16).
Arm memory per layer: 256 FP16 = 1,048,576 B; 352 tiered (96 hot) =
96×4096 + 256×2592 = 1,056,768 B (**+0.8% vs 256 FP16 — memory-matched**);
352 FP16 = 1,441,792 B (1.375×, same-coverage control).

Novelty scope (pre-registered honesty): V-quantization families (KIVI et
al.) exist.  This gate does not claim a new quantizer; it tests the scoped
claim that *under QK routing on the recoverable working-set regime, V
precision tiering converts memory into coverage nearly losslessly*.

## 2. Which qk_pool failure mode it addresses

The discovery battery found no ranking failure of qk_pool; its only
structural weakness is coverage (256 of ~1100 positions).  The method
attacks exactly that constraint, at equal active memory.

## 3. Arms (R0 substrate, same 10 samples, 64 cycles, recoverable semantics)

| arm | budget | V precision | role |
|---|---|---|---|
| qk_pool 256 FP16 | 256/220 | FP16 | baseline (R0 numbers reused) |
| qk_tiered_v 256/4bit/H96 | 256/220 | cold 4-bit | premise ablation: is V precision free at fixed coverage? |
| qk_pool 352 FP16 | 352/316 | FP16 | same-coverage control (1.375× memory) |
| qk_tiered_v 352/4bit/H96 | 352/316 | cold 4-bit | **the method** (memory-matched to baseline, +0.8%) |

All arms: identical scoring forward, refresh cadence, universe, sink/recent
structure, greedy, same-input exact-KL evaluation vs the FP16 full-cache
reference.

## 4. Preregistered verdict

- **P (premise)**: tiered-256 vs qk_pool-256: mean KL within +10% and
  task quality non-worse (NIAH −0, GovReport official ≥ baseline −1.0).
  If P fails → method premise dead; record and close.
- **C (coverage worth)**: reported, not gated: qk_pool-352 vs qk_pool-256
  (what coverage alone buys at 1.375× memory).
- **GO requires ALL**:
  - G1: tiered-352 mean KL ≤ 0.80 × qk_pool-256 mean KL (≥20% reduction);
  - G2: tiered-352 paired wins ≥ 8/10 vs qk_pool-256;
  - G3: tiered-352 p95 step KL ≤ 1.05 × qk_pool-256 p95;
  - G4: quality non-worse (same rule as P);
  - G5: tiered-352 captures most of the coverage gain:
    tiered-352 KL ≤ 1.10 × qk_pool-352 FP16 KL (i.e., 4-bit V gives up
    ≤10% of the coverage benefit at identical coverage);
  - G6: fairness invariants (budget flags, same universe/cadence).
- **NO-GO otherwise** (subclasses: PREMISE_FAILED / TIERING_LOSSY /
  COVERAGE_WORTHLESS).

No threshold may be relaxed after results.  The method is abandoned (not
retuned) on NO-GO: no bit-width/H/M sweeps beyond reporting the single
preregistered point.

## 5. Required answers (brief's method questions)

1-2. Score/rule and mechanism derivation: §1.  3. Failure mode: §2.
4-5. Needs V?  Only its precision is sacrificed — the gate itself measures
whether V content matters (if P passes with a large margin, the "V
payload is nearly free" mechanism is confirmed in vivo; if P fails, V
content matters more than the norm statistics suggested and the method
dies).  6. Without tiering the method reduces to qk_pool — G5 isolates
the tiering contribution.  7-9. Paired gain, tail, quality: G1-G4.
10. Overhead: none beyond qk_pool's scoring forward + per-build
quantize/dequantize of cold rows (arithmetic; recorded per cycle).
11. Budgets: 256 and 352 points.  12. Tasks/samples: R0 10-sample paired
substrate.  13. Failure regime existence: coverage-limited regime is
exactly the R0 substrate (coverage 23-28%).

## 6. Implementation plan

- `statekv/backend_mlx.py::state_from_anchor`: keyword-only optional
  `cold_positions` + `quant_bits`/`quant_group` (default None → no-op,
  backward compatible); quantize-dequantize selected cold V rows (torch,
  before mx.array).
- `statekv/oracle_policy_freegen.py::_free_rollout`: optional cold-set
  passthrough.  `_run_free_policy`: new policy `qk_tiered_v` — qk_pool
  selection + hot-set computation (top-H pool score within selected) +
  cold quantization on commit; teacher path untouched.
- Tests: quantization exactness/idempotence/error bound; hot/cold
  partition correctness; budget invariants; fp16 path unchanged when no
  cold set; cold rows actually carry ≤4-bit distinct values.
- Configs: three single-policy runs under
  `configs/stages/statekv_qkvtier_gate_{256t,352f,352t}.yaml` →
  `results/temporal_cache_discovery/statekv_qkvtier_gate_*_v1/`.
- Analysis: `analysis/tables/qkvtier_gate.py` → main/paired tables +
  verdict per §4.
