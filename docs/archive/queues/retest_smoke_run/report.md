# StateKV gate-retest panel (no verdicts)

Primary endpoint: task scores.  KL / delta-NLL are diagnostics.  All comparisons are continuous (point estimate + paired bootstrap CI + win/tie/loss); this report contains no pass/fail judgement.

| policy | n | official ↑ | GovReport ROUGE-L ↑ | NIAH ↑ | Reasoning ↑ | mean KL ↓ | mean ΔNLL ↓ | repetition ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| a2_temporal_volatility | 2 | 0.3278 | 0.007 | 0.000 | — | 0.000216 | -0.00059 | 0.0000 |
| attention | 2 | 0.3278 | 0.007 | 0.000 | — | 0.000387 | -0.00145 | 0.0000 |
| b2_direct_action_generator | 2 | 0.3278 | 0.007 | 0.000 | — | 0.000349 | -0.00151 | 0.0000 |
| full_cache | 2 | 0.3278 | 0.007 | 0.000 | — | 0.000000 | — | 0.0000 |
| qk_pool | 2 | 0.3278 | 0.007 | 0.000 | — | 0.000227 | -0.00125 | 0.0000 |
| qk_tiered_v | 2 | 0.3278 | 0.007 | 0.000 | — | 0.000223 | -0.00125 | 0.0000 |
| token_rarity | 2 | 0.0000 | 0.000 | 0.000 | — | 0.287315 | -0.49039 | 0.0000 |

## Paired comparisons vs references

### vs attention

| policy | Δ official | CI95 | wins/ties/losses |
|---|---:|---|---|
| a2_temporal_volatility | +0.0000 | [+0.0000, +0.0000] | 0/2/0 |

### vs qk_pool

| policy | Δ official | CI95 | wins/ties/losses |
|---|---:|---|---|
| a2_temporal_volatility | +0.0000 | [+0.0000, +0.0000] | 0/2/0 |
| attention | +0.0000 | [+0.0000, +0.0000] | 0/2/0 |
| b2_direct_action_generator | +0.0000 | [+0.0000, +0.0000] | 0/2/0 |
| full_cache | +0.0000 | [+0.0000, +0.0000] | 0/2/0 |

Full all-pairs results (all metrics, all buckets) live in `paired_comparisons.csv`.  Collection took 131.9 s.