# QK-route, V-tier method gate (preregistered verdict)

| arm             |    mean_kl |   median_kl |   p95_step_kl |   mean_niah |   mean_gov_official |
|:----------------|-----------:|------------:|--------------:|------------:|--------------------:|
| qk_pool_256     | 0.00862282 |  0.00886767 |     0.0509281 |           1 |             5.82952 |
| qk_tiered_v_256 | 0.00814225 |  0.00818889 |     0.0400657 |           1 |             6.22386 |
| qk_pool_352     | 0.0043039  |  0.00454406 |     0.0223101 |           1 |             6.02216 |
| qk_tiered_v_352 | 0.00484458 |  0.00441419 |     0.0243007 |           1 |             6.16292 |

| arm             |   paired |   mean_arm_minus_qk256 |   wins_vs_qk256 |   losses_vs_qk256 |   ties |
|:----------------|---------:|-----------------------:|----------------:|------------------:|-------:|
| qk_tiered_v_256 |       10 |           -0.000480574 |               6 |                 4 |      0 |
| qk_pool_352     |       10 |           -0.00431892  |              10 |                 0 |      0 |
| qk_tiered_v_352 |       10 |           -0.00377824  |              10 |                 0 |      0 |

P premise (tiered-256 within 1.1x qk256 + quality): True (ratio 0.944)
C coverage worth: fp16-352 KL 0.0043 vs qk256 0.0086 (ratio 0.499)
G1 tiered-352 <= 0.8x qk256: True (ratio 0.562)
G2 wins >= 8/10: True
G3 p95 tiered-352 0.0243 <= 1.05x qk256 0.0509: True
G4 quality non-worse: True
G5 tiered-352 <= 1.1x fp16-352: False (ratio 1.126)
G6 fairness flags: True

**Gate verdict (preregistered): NO_GO** (TIERING_LOSSY)
