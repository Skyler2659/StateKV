# Adaptive temporal utility

## Question

Attention history is persistent but non-stationary. This study asks whether a
token's optimal temporal horizon changes over decoding, and whether a causal
attention-history rule can recover that change.

The experiments use Qwen3-8B-4bit, a QK-routed 256-token cache, and future
attention sums over horizons `H ∈ {1, 4, 16, 32}`. All measurements replay
the baseline trajectory exactly before comparing temporal policies.

## Main result

The future-utility opportunity is strongly token-time dependent.

| Predictor family | Held-out recall | Step Spearman | Gain over global fixed |
|---|---:|---:|---:|
| Global fixed EMA | 0.7197 | 0.8059 | 0 |
| Per-head fixed EMA | 0.7212 | 0.8069 | +0.0015 |
| Token-time future oracle | **0.8577** | **0.9308** | **+0.1381** |

The oracle improves every held-out sequence, with a 95% confidence interval
of `[0.1283, 0.1447]` for the gain over per-head fixed EMA. The opportunity is
therefore not explained by a task-level or head-level choice of smoothing.

## Causal policies

Fixed EMAs show substantial short-term persistence (for example, GovReport
rank Spearman is `0.8473` at lag 1 and `0.5608` at lag 32). Dual-memory and
rank-drift rules use this persistence, but their improvements are limited to
short horizons and reverse at longer horizons. The rank-drift follow-up has a
held-out mean gain of `-0.00249` across seven sequences.

## Outcome

The temporal oracle establishes that future utility contains a large,
structured signal. Hand-designed attention-history features do not identify
it reliably, which motivates the causal target-model rollout used by R2.
Configurations live in [`configs/adaptive_temporal/`](../../configs/adaptive_temporal/)
and source artifacts in [`results/adaptive_temporal/`](../../results/adaptive_temporal/).
