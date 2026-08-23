# Cheap-R2 Phase 1: R2 Horizon Sweep (multikey@256)

Date: 2026-08-23. Branch: `codex/statekv-counterfactual-utility`.
Panel: `ruler_niah_multikey` train 200–209 (n=10), budget 256, sink 4, recent 32,
refresh_frequency 2, policy STRICT_CAUSAL_ROLLOUT_R2, Qwen3-8B-4bit.
Identical panel/conditions to the frozen teacher PK (`teacher_pk_synthetic_v1`),
whose H=1 R2 arm (32.5) is reused here as the H=1 point.

## Integrity audit (passed)

- 10 unique samples per horizon, no duplicate (sample, H) rows
- strict_pure_eviction=True and recoverable_cold_tokens=0 on every arm
- peak_active_cache_tokens ≤ 256 everywhere; budgets [256]; refresh [2]
- Both shards exited 0 (logs: bash-pivrpgfu / bash-mkci6aux)

## Results

| H | multikey@256 | teacher s/arm | wall s/arm | ×QK wall (71.7s) |
|---|--------------|---------------|------------|-------------------|
| 1  | 32.5 (frozen teacher PK) | 454.5 | 580.3 | 8.1× |
| 2  | 37.5 | 503.0 | 560.2 | 7.8× |
| 4  | 52.5 | 511.8 | 569.6 | 7.9× |
| 8  | 55.0 | 526.7 | 588.3 | 8.2× |
| 16 | 60.0 | 564.5 | 618.7 | 8.6× |
| 32 | **72.5** | 651.5 | 707.1 | 9.9× |

References: Full cache 70.0, QK 20.0, H2O 5.0, SnapKV 2.5, structured student 15.0.

Per-sample (H=32): 200→75, 201→100, 202→100, 203→100, 204→25, 205→25,
206→25, 207→100, 208→100, 209→75.

## Findings

1. **Longer rollout horizon is dramatically better, not just costlier.**
   H=1→32 moves 32.5→72.5 (+40 pts), reaching and slightly exceeding the
   full-cache reference (70.0) on this panel. The Gate C / teacher-PK
   evaluation used H=1; that verdict stays frozen, but H=1 is clearly not the
   R2 family's real ceiling. (Compression exceeding full cache on NIAH is
   plausible via denoising; 3 samples stay at 25 under every horizon.)
2. **Rollout steps are nearly free; prefix recompute dominates cost.**
   Teacher time grows 454s→651s from H=1→32 over 32 refreshes:
   per-refresh fixed cost ≈ 14.2s (prefix recompute), per rollout step
   ≈ 0.2s. The expensive part of R2 is *how often* you recompute, not *how
   far* you roll out.
3. Consequence for Cheap-R2: the horizon lever is exhausted (take H=32, it is
   ~free); the refresh-frequency and selective-trigger levers are where the
   cost must come from. Phase 2 sweeps refresh {1,4,8,16} at H=32.

Artifacts: `results/statekv_counterfactual/cheapr2_h{2,4,8,16,32}_v1/closed_loop/train/_shards/{s0,s1}/`.
Configs: `configs/statekv_counterfactual/cheapr2_h*_multikey_qwen3_8b.yaml` (commit 63ef7b0).
