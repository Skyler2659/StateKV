# Counterfactual utility and distillation

## Goal

R2 predicts future attention, while cache eviction ultimately changes model outputs and task success.
This stage compares these signals, measures the task-level headroom of the R2 teacher,
and tests whether a learned student can approximate its ranking.

## Teacher headroom

With Qwen3-8B-4bit, strict eviction, 64 decode cycles, and budgets 256/512,
the R2 teacher's clearest advantage appears on multikey retrieval.

| Task | Full | R2 @256 | Current-QK @256 | R2 @512 |
|---|---:|---:|---:|---:|
| Multikey | 70.0 | **32.5** | 20.0 | 57.5 |
| Multiquery | 100.0 | 100.0 | 100.0 | 100.0 |
| Passage retrieval | 30.0 | **40.0** | 30.0 | 30.0 |
| 2WikiMQA | 30.0 | **40.0** | 40.0 | 30.0 |
| GovReport | 6.3 | 5.7 | 5.8 | 5.9 |

The counterfactual diagnostic finds local agreement between future attention and removal damage,
but not a stable low-cost ranking near the eviction boundary.
This shifted the practical question from reproducing every counterfactual calculation
to preserving the R2 teacher's retrieval benefit.

## Student results

| Model | Recall@220 | Near-cutoff pair accuracy | Multikey @256 |
|---|---:|---:|---:|
| Pooled student | 0.713 | 0.507 | 15.0 |
| Structured student | 0.789 | 0.511 | 15.0 |
| R2 teacher | — | — | **32.5** |
| Current-QK | — | — | 20.0 |

The structured model improves global top-k recovery while reducing scoring overhead
from `0.2482 s` to `0.0795 s` per cycle. It does not improve the task-critical ordering
near the cutoff: both students reach 15.0 on the workload where R2 exceeds current-QK.

## Outcome

Static causal features can approximate broad R2 rankings but do not preserve the small set
of delayed-demand tokens that drive its retrieval gain. The next iteration therefore reduces
the number of teacher calls instead of distilling the full dynamic ranking:
a single query-onset rollout becomes Cheap-R2.

Training configurations and model artifacts are under
[`configs/statekv_counterfactual/`](../../configs/statekv_counterfactual/)
and `results/statekv_counterfactual/student_models/`.
