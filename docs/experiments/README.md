# StateKV experiment reports

The reports below describe the research program as a sequence of questions:
how to measure state-dependent cache actions, how far current-query policies
can go, and when future utility changes the outcome.

## Reading order

1. [Mechanism foundations](01_frozen_mechanism.md) — validates the
   state-conditioned evaluator used throughout the early work.
2. [Training-free and direct policies](02_training_free_and_direct_policy.md)
   — studies practical signals derived from the current cache state.
3. [Closed-loop policies and working-set control](03_oracle_gates_and_retests.md)
   — establishes QK routing, coverage/cadence behavior, and strict eviction.
4. [Adaptive temporal utility](04_adaptive_temporal.md) — identifies a large
   token-time future-utility opportunity beyond fixed attention histories.
5. [Causal existence](05_causal_existence.md) — shows that target-model
   rollouts can recover this future utility.
6. [Counterfactual utility and distillation](06_counterfactual_distillation.md)
   — evaluates task-level R2 headroom and learned approximations.
7. [Cheap-R2](07_cheap_r2.md) — selects the final one-shot policy.
8. [Benchmark results](08_benchmark_results.md) — presents the current main
   results, costs, and workload-specific conclusions.

## Current takeaway

Cheap-R2 performs one H=32 target-model rollout on the complete prefix when a
query begins, uses the resulting future-attention ranking for a single strict
eviction, and leaves that cache unchanged afterwards. It is particularly
effective on tight-budget multikey retrieval: at budget 256 it scores 70.0,
versus 21.0 for current-QK and 29.5 for LAQ. At budget 512, LAQ and Cheap-R2
are comparable; the evaluated LongBench QA slices do not show a stable
Cheap-R2 advantage.

Each report links to the configuration and artifacts needed to inspect its
results. Supporting protocols and per-study tables are available in
[`../evidence/`](../evidence/).
