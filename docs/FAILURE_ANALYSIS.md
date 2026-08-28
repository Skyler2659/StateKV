# Design lessons and limitations

StateKV began with a state-conditioned physical evaluator and ultimately arrived at
query-onset future-utility control. The transition reveals several useful design lessons
for memory-limited generation systems.

## Physical precision and deployable information are different resources

The same-state physical evaluator accurately ranks controlled candidate panels, but it requires
candidate-specific model computation. Its information does not become a practical controller
simply by replacing it with a cheap local proxy. In strict eviction, the best action is often
tied under one-step risk and the decisive event occurs only when a future query needs
an already-removed token.

## The cache action space matters

An evaluator can only distinguish actions that the policy panel makes available. In the early
strict-eviction studies, many candidate policies retained nearly identical working sets.
Expanding lookahead alone increased measured damage without consistently changing the ranking.
Future-utility control addresses this limitation by changing the information available at
the first cache decision.

## Current attention is a strong baseline, not a complete forecast

QK routing excels when the current query identifies the needed evidence. It does not anticipate
a query whose discriminative tokens are still quiet. Cheap-R2 succeeds on multikey retrieval
because its full-prefix rollout exposes that delayed demand before the cache is compressed.

## Objectives should be chosen for the workload

Trajectory KL, future-attention recall, and task accuracy capture different behavior.
The causal-existence study improves task retrieval while its frozen KL endpoint remains
inconclusive; the student models improve global recall without preserving the important
near-cutoff order. A benchmark should therefore include both distributional telemetry
and workload-level success measures.

## Operational constraints

At tight coverage, refresh cadence is critical: an outdated ranking can discard all retrieval
evidence even when its selection rule is otherwise strong. Cheap-R2 avoids repeated refresh
by acting once at query onset, but still requires one target-model prefix recomputation.
The method is consequently most appropriate for retrieval-critical workloads where that
one-shot cost is justified.

## Next step

The natural extension is a lower-cost causal predictor that preserves R2's near-cutoff ranking,
together with longer-generation and cross-model benchmarks. Proposed directions are collected
in [Future work](NEXT_RESEARCH_DIRECTIONS.md).
