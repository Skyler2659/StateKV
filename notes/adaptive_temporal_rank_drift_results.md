# Rank-drift follow-up result

The Dynamic-Horizon Oracle gate passed before this branch was run. The follow-up was restricted to 36 preregistered configurations of one mechanism: a causal percentile-rank jump controls a fast/slow dual memory. No configuration was added after held-out evaluation.

Development selected:

```text
fast rho        0.5
slow rho        0.9
rank-memory rho 0.8
jump threshold  0.05
gate alpha      10
output space    percentile rank
```

Held-out future-top-k recall:

| Future horizon | Global fixed | Per-head fixed | Rank-jump dual |
|---:|---:|---:|---:|
| 1 | 0.735740 | 0.739040 | 0.741418 |
| 4 | 0.733835 | 0.735791 | 0.741539 |
| 16 | 0.718726 | 0.719825 | 0.712038 |
| 32 | 0.690384 | 0.690122 | 0.679810 |

The gate averages the four horizons within each sequence. Rank-jump dual minus per-head fixed is `-0.002493`, with a paired sequence-bootstrap 95% interval `[-0.003979, -0.001313]` and 0 wins in 7 sequences. Short-horizon gains do not compensate for consistent degradation at H=16 and H=32.

The branch therefore fails. The result is not evidence that token-time horizon headroom is absent—the noncausal oracle established the opposite. It shows that the tested causal observable, percentile-rank jump, does not identify which temporal horizon will match future utility. Together with the earlier Fast/Slow, surprise, normalized/raw drift, and dual-memory results, this rules out further unstructured attention-history gating in the current branch.

No additional StateKV token-time state signal is aligned in the collected held-out per-head trajectories: the exact StateKV teacher is an action-level physical evaluator, while the auxiliary per-head QK/V feature table is development-only and sparsely sampled. Training a new state-conditioned gate on that mismatch would not be a fair continuation. A future branch would require a newly preregistered, matched per-head state-feature collection or a learned future-utility policy with a fresh evaluation split.

