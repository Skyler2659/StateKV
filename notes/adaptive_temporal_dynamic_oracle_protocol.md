# Dynamic-Horizon Oracle protocol

## Question

Does token-time variation in the best temporal memory retain meaningful future-utility headroom after accounting for simpler global, task, and per-head fixed horizons?

This is a diagnostic gate. It does not test a deployable adaptive algorithm and cannot establish downstream benefit.

## Why a new trace was required

The existing full token table averages attention over heads. Its auxiliary head table covers only three development samples, one cycle in four, and a selected token subset; most `(sample, layer, head, token)` histories occur too few times to form a causal EMA. Mixing that sparse head table with the full head-mean table would make the four levels incomparable.

The matched collection therefore replays the original Qwen3-8B qk_pool trajectory on the same ten samples and 64 decode cycles, while saving full per-KV-head attention only for the six preregistered layers `[0, 7, 14, 15, 21, 27]`. The cache budget remains sink 4 + recent 32 + core 220 = 256. The collection stores compressed matrices per sample and does not materialize future labels.

## Candidate temporal memories and target

Candidates are causal fixed EMAs with `rho` in `{0, 0.25, 0.5, 0.8, 0.9, 0.95, 0.99, 0.995, 0.999}`. `rho=0` is current attention. For future horizon `H` in `{1, 4, 16, 32}`, the noncausal target is the sum of attention received by the token over the next `H` decode steps. The final `H` cycles of each sequence are excluded for that target.

All metrics use the same eligible token set and a per-head top-k of 220. This top-k is a headroom diagnostic; it is not claimed to be the shared-mask physical eviction policy.

## Four levels

1. `global_fixed`: one EMA candidate per future horizon, selected across all development tasks, layers, and heads.
2. `task_fixed`: one EMA candidate per task and future horizon, selected on development sequences of that task.
3. `per_head_fixed`: one EMA candidate per `(layer, KV head, future horizon)`, selected across development tasks.
4. `NON_CAUSAL_TOKEN_TIME_ORACLE`: for each held-out token and decode time, choose the candidate whose within-step percentile rank is closest to the future-utility percentile rank. Percentile normalization prevents raw scale differences between EMAs from creating artificial choices.

The development set is `gov_report:86`, `synthetic_niah_86`, and `synthetic_niah_87`. The remaining seven sequences are held out until all fixed choices and gate thresholds are written.

## Continuation gate

The primary statistic averages future-top-k recall over `H={1,4,16,32}` within each held-out sequence. Paired bootstrap resamples the seven sequences, not tokens, cycles, layers, or heads.

Adaptive-horizon exploration may continue only if the token-time oracle:

- exceeds both `global_fixed` and `per_head_fixed` by at least 0.01 absolute recall;
- has a 95% paired sequence-bootstrap lower bound above zero for both comparisons; and
- wins on at least 5 of 7 held-out sequences for both comparisons.

If any condition fails, rank-drift and new state-conditioned-gate exploration stop and the branch is reported as a negative finding.

