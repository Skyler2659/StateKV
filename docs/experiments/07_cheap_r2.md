# Cheap-R2

## Method

**Cheap-R2 performs one full-cache foresight pass when a query begins.** The target model rolls out
32 future tokens on the uncompressed prefix, sums each historical token's future attention utility,
and makes one strict physical eviction. The chosen ranking is reused for the remaining decode cycles.

The final setup uses Qwen3-8B-4bit, 4 sink tokens, 32 recent tokens, 64 decode cycles,
and KV budgets of 128, 256, and 512. The primary validation set is 50 fresh multikey instances;
LongBench HotpotQA, 2WikiMQA, and GovReport use 20 instances each at budget 256.

## Selecting the final policy

### Horizon

With refresh fixed at every two steps, increasing the foresight horizon from 1 to 32
raises multikey@256 from 32.5 to 72.5. The useful demand window is therefore much longer
than a one-step lookahead, and `H=32` is used in the final method.

### Refresh

At H=32, slowing refresh from every two steps to every 16 steps retains 90.5% of the multikey gain
while lowering cost substantially. A direct comparison then shows that one query-onset action
is equivalent to periodic refresh on the tested workloads:

| Task and budget | Cycle-0 minus refresh-16 | 95% CI |
|---|---:|---:|
| Multikey @128 | +0.5 | `[0.0, +1.5]` |
| Multikey @256 | +0.5 | `[-2.0, +3.5]` |
| Multikey @512 | -0.5 | `[-1.5, 0.0]` |
| HotpotQA / 2WikiMQA @256 | 0.0 | `[0, 0]` |

The current and future top-budget sets overlap by `0.996` after onset. The query-onset ranking
contains the useful intervention; later refresh adds cost without a measurable benefit.

## Main benchmark

| Task and budget | Full | Current-QK | SnapKV | H2O | LAQ | LAQ++ | Cheap-R2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Multikey @128 | 82.0 | 1.0 | 0.5 | 0.0 | 0.0 | 0.0 | **48.5** |
| Multikey @256 | 82.0 | 21.0 | 0.0 | 5.0 | 29.5 | 21.0 | **70.0** |
| Multikey @512 | 82.0 | 25.5 | 55.0 | 16.0 | **84.5** | 68.5 | 81.0 |
| HotpotQA @256 | 40.0 | 35.0 | 35.0 | 35.0 | 30.0 | 25.0 | 35.0 |
| 2WikiMQA @256 | 45.0 | 55.0 | 50.0 | 45.0 | 45.0 | 40.0 | 45.0 |
| GovReport @256 | 6.25 | 6.49 | 6.09 | 6.23 | 6.24 | 6.01 | 6.03 |

Against LAQ, Cheap-R2 improves multikey by `+48.5` at budget 128 and `+40.5` at budget 256;
both paired 95% intervals are positive. At budget 512 their scores are comparable.
Cheap-R2's core difference is that its future queries are generated before cache compression,
so a weak but task-critical span can still shape the ranking.

## Runtime

At multikey@256, Cheap-R2 takes `75.5 s` per sequence versus `53.4 s` for current-QK (1.41x),
while raising the score from 21.0 to 70.0. On the tested QA slices it does not improve quality
over current-QK, so the additional computation is not justified there.

## Scope

Cheap-R2 is a retrieval-critical working-set controller for aggressive KV budgets.
The result is strongest when several pieces of evidence must survive until a later query.
It is not a general replacement for current-query cache policies: wider budgets reduce its advantage,
and the evaluated QA workloads show no stable gain. See the
[benchmark report](08_benchmark_results.md) for task-specific interpretation.
