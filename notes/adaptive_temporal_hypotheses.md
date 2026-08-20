# Adaptive Temporal Memory: hypotheses and stop conditions

## Core hypothesis

Importance drift has heterogeneous persistence across tokens, heads, states, sequences, or tasks. A causal estimator that adapts how quickly historical attention is forgotten should therefore predict future token utility better than a fixed EMA selected on a development split. Downstream improvement is a separate claim and requires a matched closed-loop eviction result.

Falsification condition: optimal fixed horizons concentrate tightly, or the adaptive estimator does not beat the development-tuned fixed EMA on held-out future-utility metrics.

## Branch V1 — Fast/slow normalized drift

Hypothesis:

The normalized disagreement between fast and slow attention states detects a change in the local relevance regime.

Mechanism:

Maintain fast and slow EMAs, normalize their difference by an EMA of squared residuals, and map high normalized drift to a shorter retention timescale. Test both a threshold gate and a smooth gate.

Expected evidence:

Higher held-out Spearman correlation and future-important-token recall than the tuned fixed EMA, with gains concentrated around high-drift events and no loss on stable trajectories.

Falsification condition:

No held-out improvement over the best fixed EMA, or all gains reduce to using a globally shorter fixed horizon.

Cost:

Three scalar states per tracked token for fast, slow, and variance, plus the retention score; offline analysis first, then a matched closed-loop run only if the diagnostic gate passes.

## Branch A — Surprise

Hypothesis:

Absolute prediction error `|a_t - R_{t-1}|` is a more direct staleness signal than fast/slow disagreement.

Mechanism:

Normalize surprise by a residual-variance EMA and use it to gate between the same short and long decay rates.

Expected evidence:

Improved event response and held-out future-utility recall when regime shifts are abrupt.

Falsification condition:

Surprise fires mostly on low-magnitude noise or fails to improve held-out metrics.

Cost:

Low; reuse the V1 variance state. Run only if V1 misses abrupt events.

## Branch B — Rank drift

Hypothesis:

Relative rank changes are more stable across attention magnitudes and layers than absolute drift.

Mechanism:

Gate the temporal horizon using recent percentile-rank displacement within the eligible set.

Expected evidence:

Improved top-k recall near the eviction boundary, especially where attention magnitudes differ substantially across layers.

Falsification condition:

Rank gating increases set turnover without improving future-important-token recall.

Cost:

Medium; requires a per-step rank transform. Run only if V1 error is localized near scale-heterogeneous layers.

## Branch C — Variance-normalized drift

Hypothesis:

Separating persistent noise from a regime change is necessary; raw fast/slow difference over-forgets noisy tokens.

Mechanism:

Compare unnormalized and variance-normalized fast/slow gates with identical decay endpoints.

Expected evidence:

Normalized drift preserves stable-token accuracy while matching or improving event response.

Falsification condition:

Normalization has no held-out benefit or suppresses true drift events.

Cost:

Already included in V1; this is a required ablation rather than an independent search.

## Branch D — Head-level horizon

Hypothesis:

Temporal persistence is primarily a head property, so head-shared adaptation captures most of the benefit with lower memory than token-level states.

Mechanism:

Estimate drift statistics per layer/head and apply the resulting gate to tokens routed through that head.

Expected evidence:

Clear between-head best-horizon dispersion and most token-level adaptive gain at substantially lower state cost.

Falsification condition:

Between-head dispersion is weak or head-shared adaptation loses the token-level prediction gain.

Cost:

Low state memory, but existing head traces are sparse. Run the offline head diagnostic before adding a live implementation.

## Branch E — Existing StateKV state signal

Hypothesis:

A cheap StateKV-era observable predicts when attention history becomes stale better than attention drift alone.

Mechanism:

Calibrate a threshold or linear mapping from an already captured causal state feature to the decay gate.

Expected evidence:

Complementarity beyond attention-only adaptive memory on held-out samples.

Falsification condition:

No gain after controlling for the V1 drift score, or the signal requires forbidden future/full-pool information.

Cost:

Medium. Run only if the attention-only V1 passes and a deployable state feature is available on the matched live path.

## Branch F — Dual-memory gate

Hypothesis:

Mixing stable short- and long-term scores is more robust than recursively changing a single EMA coefficient.

Mechanism:

Maintain fixed short and long EMAs and mix them using the normalized drift gate.

Expected evidence:

Lower sensitivity to gate thresholds and stronger stable-event behavior than dynamic-rho V1.

Falsification condition:

No held-out gain over V1 or the best constituent fixed EMA.

Cost:

Low after V1. This is the first fallback if dynamic-rho V1 is unstable.

## Decision gates

1. Drift gate: lagged attention/rank stability must decay measurably.
2. Heterogeneity gate: best fixed horizons must vary materially across at least one meaningful axis and must not be explained entirely by sample length or attention scale.
3. Dynamic-oracle gate: before any further heuristic work, decompose global-fixed, task-fixed, per-head-fixed, and noncausal token-time dynamic-horizon performance on the same per-KV-head traces. Fixed choices are selected on the development split. The continuation test is a paired held-out-sequence bootstrap, averaged over future horizons 1, 4, 16, and 32. Token-time dynamic recall must exceed both global fixed and per-head fixed by at least 0.01, with a 95% bootstrap lower bound above zero and at least 5/7 sequence wins.
4. Adaptive-horizon stop rule: if gate 3 fails, stop rank-drift and state-conditioned-gate exploration and write the result as a negative finding. Apparent token-level heterogeneity alone is not sufficient evidence.
5. Estimator gate: only if gate 3 passes may a causal adaptive method be developed further; it must then beat the development-tuned fixed EMA on held-out future-utility prediction or recall.
6. Closed-loop gate: only methods passing gate 5 enter an expensive policy run.
7. Continuation gate: downstream task/KL benefit determines whether adaptive temporal memory is a KV-compression method or only an estimator diagnostic.
