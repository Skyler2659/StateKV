# StateKV gate-retest report (no verdicts)

Date: 2026-08-10.  Re-tests of policies that earlier phases rejected through preregistered joint gates, under the no-hard-gate contract: task scores are the primary endpoint, KL/NLL are diagnostics, and every number is reported continuously (point estimate, paired bootstrap CI, win/tie/loss).  Fresh sample offsets throughout (Track A 106-117, Track B 118-127, Track D 14-17); recoverable backing-pool semantics in Track B.

## Track A — Era-1 replay policy families (P7–P15)

Era-1 teacher-forced replay (Qwen2.5-1.5B, shared mask, budget 128/core 92, 24 fresh sequences at offsets 106-117, anchors 16/32/48).  This re-screens the P7/P9/P13/P14/P15 policy families with no selection gate.

| policy | mean KL ↓ | P95 KL ↓ | CVaR95 ↓ | KL≥1 rate ↓ | mean ΔNLL ↓ | seq win vs attn |
|---|---:|---:|---:|---:|---:|---:|
| contribution_q75_w4_shared | 0.3938 | 2.325 | 5.014 | 0.0938 | 0.3988 | 0.542 |
| contribution_mean_w4_shared | 0.3957 | 2.389 | 4.988 | 0.0911 | 0.4087 | 0.583 |
| contribution_mean_w1_shared | 0.4009 | 2.423 | 5.448 | 0.0877 | 0.4113 | 0.500 |
| attention_mean_w1_shared | 0.4085 | 2.592 | 5.311 | 0.0929 | 0.4078 | 0.000 |
| attention_temporal_volatility_w4_shared | 0.4099 | 2.550 | 4.861 | 0.1068 | 0.4175 | 0.458 |
| blend_attention_contribution_25_w4_shared | 0.4289 | 2.581 | 5.547 | 0.0990 | 0.4362 | 0.333 |
| blend_attention_contribution_75_w4_shared | 0.4292 | 2.646 | 5.453 | 0.0972 | 0.4365 | 0.417 |
| protected_attention_rescue_m4_shared | 0.4298 | 2.635 | 5.585 | 0.0938 | 0.4332 | 0.417 |
| protected_attention_rescue_m8_shared | 0.4321 | 2.706 | 5.592 | 0.0981 | 0.4352 | 0.375 |
| protected_attention_rescue_m16_shared | 0.4401 | 2.751 | 5.604 | 0.1033 | 0.4491 | 0.417 |
| blend_attention_contribution_50_w4_shared | 0.4416 | 2.769 | 5.594 | 0.1033 | 0.4475 | 0.292 |
| attention_mean_w4_shared | 0.4425 | 2.644 | 5.364 | 0.1059 | 0.4526 | 0.375 |
| attention_head_peak_w4_shared | 0.4429 | 2.774 | 5.680 | 0.1007 | 0.4272 | 0.458 |
| value_diagonal_leverage_shared | 0.6100 | 2.983 | 5.118 | 0.1736 | 0.6768 | 0.333 |
| key_diagonal_leverage_shared | 0.6995 | 3.437 | 5.883 | 0.1979 | 0.7847 | 0.250 |
| value_adjacent_change_shared | 0.9956 | 4.909 | 7.702 | 0.2361 | 1.0548 | 0.083 |
| uniform_position_coverage_shared | 1.2520 | 5.328 | 8.415 | 0.3186 | 1.3300 | 0.000 |

## Track B — Era-2 recoverable freegen panel (P16–P32, QKV-tier)

Era-2 recoverable free generation (Qwen3-8B-4bit, budget 256/core 220, 20 fresh sequences at offsets 118-127, 64 tokens, horizon 1).  Primary endpoint: task scores; KL and ΔNLL are diagnostics.  Non-inferiority reference lines: ΔNLL +0.01, ROUGE-L -0.5.

| policy | n | official ↑ | GovReport ROUGE-L ↑ | NIAH ↑ | mean KL ↓ | mean ΔNLL ↓ |
|---|---:|---:|---:|---:|---:|---:|
| b1_historical_tiny_ranker | 20 | 53.0715 | 0.061 | 1.000 | 0.235240 | -0.21263 |
| a4_uncertainty_cascade | 20 | 53.0260 | 0.061 | 1.000 | 0.326599 | -0.32950 |
| attention | 20 | 53.0260 | 0.061 | 1.000 | 0.326599 | -0.32950 |
| b3_layer_adaptive_budget | 20 | 52.9164 | 0.058 | 1.000 | 0.189236 | -0.16580 |
| b2_direct_action_generator | 20 | 52.8684 | 0.057 | 1.000 | 0.184253 | -0.16394 |
| full_cache | 20 | 52.7890 | 0.056 | 1.000 | 0.000000 | — |
| qk_pool | 20 | 52.7614 | 0.055 | 1.000 | 0.007614 | -0.01917 |
| token_rarity | 20 | 52.7571 | 0.055 | 1.000 | 0.868886 | -0.79052 |
| qk_tiered_v | 20 | 52.7365 | 0.055 | 1.000 | 0.008066 | -0.01909 |
| a2_temporal_volatility | 20 | 42.9100 | 0.058 | 0.800 | 0.240597 | -0.19652 |
| snapkv | 20 | 37.8659 | 0.057 | 0.700 | 0.215554 | -0.24811 |
| h2o | 20 | 2.8805 | 0.058 | 0.000 | 0.796621 | -0.74510 |
| uniform | 20 | 2.8046 | 0.056 | 0.000 | 1.109721 | -1.03595 |

Paired official-score deltas vs **attention**:

| policy | Δ | CI95 | wins/ties/losses |
|---|---:|---|---|
| a4_uncertainty_cascade | +0.0000 | [+0.0000, +0.0000] | 0/20/0 |
| a2_temporal_volatility | -10.1160 | [-25.1255, +0.0414] | 5/8/7 |

Paired official-score deltas vs **qk_pool**:

| policy | Δ | CI95 | wins/ties/losses |
|---|---:|---|---|
| b1_historical_tiny_ranker | +0.3101 | [+0.0402, +0.6777] | 7/10/3 |
| a4_uncertainty_cascade | +0.2646 | [-0.0630, +0.7206] | 6/10/4 |
| attention | +0.2646 | [-0.0634, +0.7122] | 6/10/4 |
| b3_layer_adaptive_budget | +0.1550 | [-0.0935, +0.4455] | 6/11/3 |
| b2_direct_action_generator | +0.1070 | [-0.1325, +0.3520] | 6/10/4 |
| full_cache | +0.0276 | [-0.1125, +0.1780] | 5/11/4 |
| a2_temporal_volatility | -9.8515 | [-24.8976, +0.2503] | 6/10/4 |
| h2o | -49.8810 | [-70.0115, -29.7057] | 8/0/12 |

## Track D — Rademacher VJP replication (P3)

Rademacher VJP independent-sample replication (8 untouched sequences, offsets 14-17, from the frozen P3 source run).  Gains vs the `hidden_l2_action` baseline at width 8 / refresh 4, averaged over both tasks.  The original P3 post-hoc variant had normalized regret 0.1945 → 0.0696 (gain +0.125) with pairwise accuracy -0.0026 on the development sequences.

| method | Δ median spearman | Δ pairwise acc | Δ normalized regret |
|---|---:|---:|---:|
| output_fisher_oracle | +0.3095 | +0.2423 | +0.1770 |
| margin_vjp_action | -0.0119 | +0.0051 | +0.0131 |
| hidden_l2_action | +0.0000 | +0.0000 | +0.0000 |
| fisher_gaussian_vjp_action | -0.0238 | -0.0108 | -0.0087 |
| fisher_rademacher_vjp_action | -0.0278 | -0.0204 | -0.0141 |
| fisher_rademacher_vjp_innovation_state | -0.0556 | -0.0293 | -0.0229 |
| fisher_rademacher_vjp_repeated_state | -0.0635 | -0.0383 | -0.0428 |
| fisher_gaussian_vjp_innovation_state | -0.0317 | -0.0274 | -0.0464 |
| entropy_vjp_action | -0.1071 | -0.0510 | -0.0503 |
| fisher_gaussian_vjp_repeated_state | -0.0556 | -0.0338 | -0.0561 |

## Standing caveats

- Era-1 and Era-2 substrates are not comparable (different model, mask semantics, budget); cross-era rows are never averaged.
- Bootstrap CIs at these sample sizes are wide; treat reference-line crossings as descriptive, not adjudicated.
- Raw artifacts: `statekv_retest_replay_era1_n24_v1`, `statekv_retest_freegen_qwen3_8b_n20_v1`, `statekv_retest_vjp_rademacher_replication_v1`.
