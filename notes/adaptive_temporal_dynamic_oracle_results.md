# Dynamic-Horizon Oracle results

## Integrity checks

The targeted per-KV-head collection completed all ten preregistered sequences and 64 decode cycles. Across 640 matched steps, the new replay and the original qk_pool baseline have zero maximum absolute difference for exact KL, JS, full and perturbed NLL, delta NLL, logit L2, and Fisher quadratic. The attention files cover six preregistered layers and all eight KV heads.

## Four-level decomposition

Held-out means average seven sequences and future horizons `H={1,4,16,32}`.

| Horizon granularity | Future top-k recall | Mean step Spearman | Gain over global fixed |
|---|---:|---:|---:|
| Global fixed | 0.719671 | 0.805851 | 0.000000 |
| Task fixed | 0.719671 | 0.805851 | 0.000000 |
| Per-head fixed | 0.721195 | 0.806946 | +0.001523 |
| `NON_CAUSAL_TOKEN_TIME_ORACLE` | 0.857740 | 0.930795 | +0.138069 |

Task-specific tuning selected exactly the same fixed EMA as the global selection at every future horizon. Per-head fixed selection produced visible development-set horizon dispersion but only +0.0015 held-out recall. Static task/head specialization therefore explains little of the available headroom.

The token-time oracle improves recall at every future horizon:

| Future horizon | Global fixed | Per-head fixed | Token-time oracle | Oracle − per-head |
|---:|---:|---:|---:|---:|
| 1 | 0.735740 | 0.739040 | 0.874151 | +0.135111 |
| 4 | 0.733835 | 0.735791 | 0.876029 | +0.140238 |
| 16 | 0.718726 | 0.719825 | 0.856819 | +0.136994 |
| 32 | 0.690384 | 0.690122 | 0.823962 | +0.133840 |

## Preregistered continuation gate

Paired inference uses the sequence-level mean over all four future horizons.

| Comparison | Mean gain | 95% paired bootstrap CI | Wins |
|---|---:|---:|---:|
| Token-time oracle − global fixed | +0.138069 | [0.130739, 0.145361] | 7/7 |
| Token-time oracle − per-head fixed | +0.136546 | [0.128282, 0.144715] | 7/7 |

Both comparisons exceed the preregistered +0.01 practical threshold, have confidence intervals entirely above zero, and win at least five sequences. The Dynamic-Horizon Oracle gate passes.

## Interpretation

This result establishes a large upper-bound opportunity for token-time horizon selection on the future-attention ranking target. It does not show that rank drift, Fast/Slow disagreement, or any other causal signal can recover that opportunity. It also does not override the existing strict pure-eviction teacher result: the oracle here is a per-head future-utility diagnostic, not a physically deployable shared-mask controller.

The next allowed step is therefore narrow: test a preregistered causal rank-drift gate against the tuned global and per-head fixed baselines. A causal failure would mean that dynamic headroom exists but the tested observables do not identify it.

