# Project evolution

StateKV evolved from a question about state-dependent cache actions into a focused
future-utility controller for tight-budget retrieval.

## Mechanism foundations

The initial frozen studies established an exact deletion identity, measured how compression
history changes action consequences, and built a same-state physical evaluator for finite
candidate panels. This created a precise experimental instrument for cache interventions.

## Current-state control

The next program explored training-free estimators, contribution policies, temporal signals,
refresh triggers, and direct cache selectors. It clarified the strength of current-query
attention and the limits of using one local signal for both selection and refresh.

## Working-set control

Closed-loop evaluation on Qwen3-8B established exact full-pool QK routing as the strongest
practical baseline. It also mapped the coverage-and-cadence trade-off, demonstrated nearly
lossless cold-value tiering, and separated recoverable from strict cache semantics.

## Future utility and Cheap-R2

Adaptive temporal analysis showed that the useful horizon varies by token and time.
Causal R2 rollouts recovered most of that signal and improved tight-budget retrieval,
but repeated rollouts were expensive. Cheap-R2 distilled the operational insight into
one full-cache H=32 rollout at query onset followed by a fixed strict eviction.

The current project is organized around this final method and its benchmark scope.
The full chronological reports are available in [experiments/](experiments/README.md).
