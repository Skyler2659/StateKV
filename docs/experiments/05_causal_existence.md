# Causal future utility

## Question

The adaptive-temporal oracle shows that future attention is useful. This study
asks whether that signal can be predicted from causal information and whether
it improves strict cache eviction.

The formal evaluation uses Qwen3-8B-4bit, 30 sequences across single NIAH,
GovReport, and multikey tasks, strict eviction at budgets 128 and 256, and
paired cluster bootstrap statistics.

## Predicting future utility

| Predictor | Oracle recovery at H=32 | Result |
|---|---:|---|
| History-feature GBDT | 21.39% | Current observable features recover a limited portion of future utility. |
| R2 causal rollout | 92.57–96.24% across H=1–32 | Target-model rollouts recover nearly all of the future-attention signal. |

R2 recomputes the current prefix, autoregressively produces future tokens,
and accumulates the attention utility they assign to historical positions. Its
recovery is positive on all 24 validation sequences, demonstrating that the
important future signal is available causally rather than requiring an oracle
with access to a ground-truth continuation.

## Closed-loop result

The future-attention ranking raises the task measures at both budgets, with
the clearest change on needle retrieval:

| Policy | Budget 128: needle / score | Budget 256: needle / score |
|---|---:|---:|
| R2 | **0.575 / 40.7** | **0.663 / 46.3** |
| Fixed EMA | 0.300 / 22.3 | 0.588 / 41.2 |
| Current-QK | 0.300 / 22.3 | 0.588 / 41.2 |
| SnapKV | 0.013 / 3.1 | 0.463 / 33.0 |
| H2O | 0.000 / 2.1 | 0.000 / 2.1 |

The paired trajectory-KL comparison against a frozen EMA baseline is not
conclusive at either budget (improvements `+0.343` and `+0.133`, with
confidence intervals crossing zero). The result separates three objectives:
future attention, distributional fidelity, and task success need not induce
the same cache ranking.

## Cost and implication

The original R2 controller costs roughly 389–403 seconds per sequence—about
20x the full-cache reference and 5–6x cheap policies—because it repeatedly
recomputes the prefix. It nevertheless proves the key causal claim: future
utility is predictable and useful for retrieval. The following work focuses on
retaining that advantage with far fewer rollouts.

The formal configuration is
[`causal_existence_qwen3_8b.yaml`](../../configs/statekv_existence/causal_existence_qwen3_8b.yaml).
