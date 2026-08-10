# Recoverable Gate R0 — decomposition ladder (same 10 samples)

| stage                               | arm                         |   mean_trajectory_kl |
|:------------------------------------|:----------------------------|---------------------:|
| irreversible (Gate 0 pure eviction) | pure_a2_temporal_volatility |           0.263763   |
| irreversible (Gate 0 pure eviction) | pure_attention              |           0.0975807  |
| irreversible (Gate 0 pure eviction) | pure_b2_uniform             |           0.0960896  |
| irreversible (Gate 0 pure eviction) | pure_dynamic_b3             |           0.118775   |
| irreversible (Gate 0 pure eviction) | pure_snapkv                 |           0.159635   |
| irreversible (Gate 0 pure eviction) | pure_teacher_panel          |           0.232156   |
| recoverable simple/control          | rec_attention               |           0.345801   |
| ceiling                             | rec_full_cache              |           0          |
| recoverable query-aware             | rec_qk_pool                 |           0.00862282 |
| recoverable query-aware             | rec_quest_like              |           0.02426    |
| recoverable simple/control          | rec_recency                 |           0.795189   |
| recoverable teacher                 | rec_statekv_exact_mean      |           0.0213002  |
| recoverable simple/control          | rec_uniform                 |           0.895196   |

| component                                                      |   delta_kl |
|:---------------------------------------------------------------|-----------:|
| D1 recoverability, same rule (attention)                       | -0.24822   |
| D1 recoverability, best cheap (pure b2_uniform -> rec qk_pool) |  0.0874668 |
| D1 recoverability, teacher (pure -> recoverable)               |  0.210856  |
| D2 query-aware retrieval (simple -> qk/quest)                  |  0.786567  |
| D3 physical-risk scorer (B* -> teacher)                        | -0.0126774 |
