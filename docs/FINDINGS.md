# Research results

StateKV studies how a compressed KV cache can retain information that becomes useful only after
the model's query state changes. The work combines exact state-conditioned evaluation,
current-query working-set policies, and causal future-utility control.

## What the project establishes

### State-conditioned cache actions are measurable

A fixed-boundary deletion identity matches replay to FP64 maximum L2 error `2.26e-11`.
At an observed compressed state, state-conditioned action responses reach cosine `0.99974`.
For finite candidate panels, the path-corrected scalar evaluator recovers rankings with
Spearman `1.0` in the reported evaluation and replication. These results provide a precise
instrument for studying state-dependent cache control.

### Current-query QK routing is a strong working-set baseline

Exact full-pool QK routing selects the historical keys most useful to the current query.
It performs strongly across the recoverable free-generation panel, long-context stress tests,
and both evaluated model families. The most important practical control variable is
**coverage × cadence**: when the active core is small, stale rankings rapidly lose task-critical
evidence; with a sufficiently large core, the same policy remains stable at longer contexts.

Cold-V 4-bit tiering retains the task behavior of QK routing at matched budget
(KL `0.008066` versus `0.007614` in the fresh free-generation retest), making it a useful
memory option when more FP16 coverage is unavailable.

### Future utility changes tight-budget retrieval

The adaptive temporal study identifies a token-time future-utility opportunity:
a non-causal temporal oracle improves held-out recall by `+0.1381` over fixed EMA.
R2 target-model rollouts recover 92.57–96.24% of this oracle signal from causal information.

Cheap-R2 turns that insight into a practical one-shot controller. On fresh multikey retrieval,
it scores 48.5 / 70.0 / 81.0 at budgets 128 / 256 / 512. At budget 256, current-QK scores 21.0
and LAQ scores 29.5; Cheap-R2 scores 70.0 at 1.41x current-QK wall time.

## Scope of the result

Cheap-R2 is most effective when a small cache must preserve several dispersed pieces of evidence
until a delayed retrieval query. At budget 512, LAQ and Cheap-R2 are comparable; on the evaluated
HotpotQA and 2WikiMQA slices, Cheap-R2 has no stable advantage over current-QK.
The stored 64-token GovReport runs do not reach substantive report content and are not used
to compare summary quality.

## Design lessons

- Accurate physical evaluation does not by itself produce a low-cost online policy.
- Current-query routing and future-query routing solve complementary problems.
- Broad top-k ranking accuracy is not enough: task quality depends on a small group of tokens
  close to the cache cutoff.
- Strict cache eviction makes the first query-onset decision especially important.

For the complete sequence of studies, see the [experiment reports](experiments/README.md).
For the latest task-level comparisons, see [benchmark results](experiments/08_benchmark_results.md).
