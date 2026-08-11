# StateKV Gate 0/1 — strict-pure-eviction teacher headroom (Qwen3-8B, 768 ctx, budget 256/core 220)

Teacher arm: `/Users/wangsikai/l1-robust-kv-cache/results/temporal_cache_discovery/statekv_teacher_gate_qwen3_8b_g0_v1`; cheap trajectories: `/Users/wangsikai/l1-robust-kv-cache/results/temporal_cache_discovery/statekv_pure_eviction_qwen3_8b_p35_v1` (matched samples, same budget, same substrate).
Teacher trajectory KL: 0.2322 vs best cheap (b2_uniform): 0.0961 (relative gain -141.6%).
Paired per-sample: teacher wins 2/10 vs b2_uniform.
Step tail p95: teacher 1.0941 vs b2_uniform 0.4122 (tail_ok=False); NIAH teacher 1.00 vs b2_uniform 1.00 (quality_ok=True).

**Gate 0 verdict (predeclared): NO_HEADROOM**

Gate 1 (fixed action space, teacher roll-in):
best cheap panel candidate: a2_temporal_volatility mean KL 0.2361, mean oracle regret 0.0039 (relative 1.7%), regret>0 in 73.8% of cycles.

**Gate 1 verdict (predeclared): ACTION_SPACE_DOMINANT**

Teacher selected-candidate counts: a2_temporal_volatility=49; attention=34; b2_uniform=29; snapkv=112; stale_prev=282; uniform=134
