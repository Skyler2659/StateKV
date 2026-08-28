# Future work

The current results point to a clear next set of questions for future-utility cache control.

## Lower-cost causal ranking

Cheap-R2 obtains a useful near-cutoff ranking from one full-prefix target-model rollout.
A next model should preserve that local ranking with fewer recomputations, for example through
a lightweight query-conditioned predictor trained directly on task-critical boundary examples.

## Quality and cost frontiers

Coverage and refresh cadence jointly determine working-set quality. A systematic frontier over
memory, update rate, latency, and task quality would turn the observed cliff behavior into
a deployment model for retrieval systems.

## Broader benchmark coverage

The current positive result is retrieval-focused. The next benchmark should add longer
generations, answer-span scoring for QA, generation caps appropriate for summarization,
additional model families, and varied decode lengths.

## Cache reuse beyond one request

Cross-session and prefix reuse introduce a different form of future demand: a token may matter
to a later request rather than a later decode step. The StateKV cache and evaluation interfaces
provide a starting point for this setting.

## Reusable infrastructure

The repository already provides strict and recoverable cache semantics, exact output metrics,
paired bootstrap statistics, full-pool QK routing, causal rollout collection, and compact
active evaluation runners. These components can support a new memory-control project without
rebuilding the experimental stack.
