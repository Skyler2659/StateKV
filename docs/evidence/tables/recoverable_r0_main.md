# StateKV Recoverable Gate R0 — unified recoverable-semantics headroom

Run: `/Users/wangsikai/l1-robust-kv-cache/results/temporal_cache_discovery/statekv_recoverable_r0_qwen3_8b_v1`; ladder references: `/Users/wangsikai/l1-robust-kv-cache/results/temporal_cache_discovery/statekv_pure_eviction_qwen3_8b_p35_v1` (pure cheap) and `/Users/wangsikai/l1-robust-kv-cache/results/temporal_cache_discovery/statekv_teacher_gate_qwen3_8b_g0_v1` (pure teacher), same samples.
Protocol: docs/evidence/statekv_recoverable_r0_protocol.md (G1-G5 preregistered).

## Per-arm aggregates

| policy             |   samples |   mean_trajectory_kl |   median_trajectory_kl |   p95_trajectory_kl |   mean_official_score |   mean_govreport_rouge_l |   mean_niah_retrieval |   mean_recovered_fraction |   mean_churn_layer_mean |   recovery_events |   mean_candidate_universe |   pool_scoring_time_s |   official_govreport |   official_niah |   p95_step_kl |   p99_step_kl |   max_step_kl |
|:-------------------|----------:|---------------------:|-----------------------:|--------------------:|----------------------:|-------------------------:|----------------------:|--------------------------:|------------------------:|------------------:|--------------------------:|----------------------:|---------------------:|----------------:|--------------:|--------------:|--------------:|
| attention          |        10 |           0.345801   |             0.370115   |           0.657874  |              52.9642  |                0.0592832 |                     1 |                0          |                  0      |               0   |                    1118.8 |               0       |              5.92832 |             100 |     1.92676   |      4.69607  |     10.4144   |
| full_cache         |        10 |           0          |             0          |           0         |              53.1355  |                0.0627097 |                     1 |              nan          |                nan      |               0   |                     nan   |             nan       |              6.27096 |             100 |   nan         |    nan        |    nan        |
| qk_pool            |        10 |           0.00862282 |             0.00886767 |           0.012575  |              52.9148  |                0.0582955 |                     1 |                0.214752   |                 47.9953 |              63   |                    1118.8 |               8.30752 |              5.82952 |             100 |     0.0509281 |      0.12839  |      0.368487 |
| quest_like         |        10 |           0.02426    |             0.0253538  |           0.0360197 |              52.9598  |                0.0591959 |                     1 |                0.212629   |                 47.5209 |              63   |                    1118.8 |               8.24745 |              5.91958 |             100 |     0.118907  |      0.357322 |      0.860278 |
| recency            |        10 |           0.795189   |             0.763305   |           1.45537   |               2.73445 |                0.0546892 |                     0 |                0.00447443 |                  1      |              63   |                    1118.8 |               0       |              5.4689  |               0 |     4.61532   |     10.4927   |     15.5422   |
| statekv_exact_mean |        10 |           0.0213002  |             0.0181496  |           0.0459091 |              53.1472  |                0.0629438 |                     1 |                0.518878   |                115.965  |              55.5 |                    1118.8 |               9.26535 |              6.29438 |             100 |     0.112375  |      0.366558 |      0.921873 |
| uniform            |        10 |           0.895196   |             0.840829   |           1.57177   |               2.89043 |                0.0578085 |                     0 |                0.492188   |                110      |              63   |                    1118.8 |               0       |              5.78086 |               0 |     4.61507   |     11.0074   |     24.3101   |

## Paired vs teacher (baseline minus teacher, positive = teacher better)

| baseline   |   paired_samples |   mean_baseline_minus_teacher_kl |   ci95_lower |   ci95_upper |   teacher_wins |   ties |   teacher_losses |   min_diff |    max_diff |
|:-----------|-----------------:|---------------------------------:|-------------:|-------------:|---------------:|-------:|-----------------:|-----------:|------------:|
| uniform    |               10 |                        0.873895  |   0.647803   |   1.12067    |             10 |      0 |                0 |  0.299936  |  1.55351    |
| recency    |               10 |                        0.773889  |   0.484175   |   1.07523    |             10 |      0 |                0 |  0.186907  |  1.5665     |
| attention  |               10 |                        0.324501  |   0.211417   |   0.451378   |             10 |      0 |                0 |  0.102415  |  0.765687   |
| qk_pool    |               10 |                       -0.0126774 |  -0.0206812  |  -0.00589609 |              0 |      0 |               10 | -0.0387329 | -0.00052215 |
| quest_like |               10 |                        0.0029598 |  -0.00591234 |   0.0101157  |              7 |      0 |                3 | -0.0278828 |  0.0155803  |

## Verdict

B* (strongest cheap recoverable): qk_pool mean KL 0.0086; teacher 0.0213 (ratio 2.470).
G1 headroom ratio <= 0.7: False
G2 wins 0/10 >= 8 and CI95 [-0.0207, -0.0059] excludes 0: False
G3 p95 teacher 0.1124 <= 1.05x B* 0.0509: False
G4 quality-valid (full_cache NIAH 1.00 >= 0.8) and quality non-worse: True
G5 fairness flags: True

**R0 verdict (preregistered): NO_GO** (NO_HEADROOM)

## Decomposition

| component                                                      |   delta_kl |
|:---------------------------------------------------------------|-----------:|
| D1 recoverability, same rule (attention)                       | -0.24822   |
| D1 recoverability, best cheap (pure b2_uniform -> rec qk_pool) |  0.0874668 |
| D1 recoverability, teacher (pure -> recoverable)               |  0.210856  |
| D2 query-aware retrieval (simple -> qk/quest)                  |  0.786567  |
| D3 physical-risk scorer (B* -> teacher)                        | -0.0126774 |
