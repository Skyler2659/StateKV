# Benchmark results

## Evaluation setup

The final panel uses Qwen3-8B-4bit, strict physical eviction, no cold recovery, and 64 decode cycles.
Multikey uses 50 fresh sequences at budgets 128, 256, and 512. HotpotQA, 2WikiMQA, and GovReport
each use 20 sequences at budget 256. The panel compares Full cache, current-QK, SnapKV, H2O, LAQ,
LAQ++, and Cheap-R2.

## Multikey retrieval

Each example contains four distributed seven-digit values; the score is exact recovery accuracy.
Cheap-R2 is the best compressed policy at budgets 128 and 256.

| Budget | Full | Current-QK | LAQ | LAQ++ | Cheap-R2 | R2–QK | R2–LAQ |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 82.0 | 1.0 | 0.0 | 0.0 | **48.5** | +47.5 `[+40.0, +55.5]` | +48.5 `[+41.0, +56.0]` |
| 256 | 82.0 | 21.0 | 29.5 | 21.0 | **70.0** | +49.0 `[+39.0, +58.5]` | +40.5 `[+29.5, +51.0]` |
| 512 | 82.0 | 25.5 | **84.5** | 68.5 | 81.0 | +55.5 `[+46.5, +63.5]` | -3.5 `[-11.5, +4.0]` |

At budget 256, the one-shot rollout costs 1.41x current-QK wall time and improves accuracy by 49 points.
Token-level inspection attributes the gain to future query information: R2 moves delayed needle spans
from median current-QK ranks 217–260 to future ranks 29–34, while LAQ reaches only 152–177
after looking ahead from its already compressed cache.

## Question answering

| Task | Full | Current-QK | Cheap-R2 | LAQ | LAQ++ |
|---|---:|---:|---:|---:|---:|
| HotpotQA | 40 | 35 | 35 | 30 | 25 |
| 2WikiMQA | 45 | 55 | 45 | 45 | 40 |

The original substring score treats text in a model's explanatory segment as an answer and misses valid aliases.
A 20-example answer-level review gives HotpotQA 40 for both current-QK and Cheap-R2;
for 2WikiMQA it gives 40 and 45. With this sample size, the project treats the QA comparison
as no stable method difference, not as a Cheap-R2 win.

## Long-form generation

The stored GovReport runs stop at 64 tokens, before any system produces the substantive report content.
Their ROUGE-L values (Full 6.25, current-QK 6.49, Cheap-R2 6.03) measure overlap in a generic
opening sentence rather than summary quality. This workload remains in the repository for continuity
but is not used in the headline comparison.

## Practical choice

| Workload | Recommended policy | Reason |
|---|---|---|
| Multikey @128 / @256 | Cheap-R2 | Large retrieval gain for a modest one-shot rollout cost |
| Multikey @512 | LAQ or Cheap-R2 | Comparable quality; LAQ has the simpler lookahead path |
| HotpotQA / 2WikiMQA @256 | Current-QK | No demonstrated Cheap-R2 benefit at lower cost |
| Long-form generation | Evaluate with a longer generation protocol | The stored 64-token setting is not informative |

The benchmark supports a focused conclusion: full-cache future foresight is valuable when a tight cache
must preserve dispersed evidence for an imminent retrieval query. Its value outside that regime
remains an open empirical question.
