# Adaptive Temporal Memory for StateKV: final report

## Decision

The branch produces a mixed but actionable result.

There is large, statistically clear token-time dynamic-horizon headroom on a matched per-KV-head future-attention target. The `NON_CAUSAL_TOKEN_TIME_ORACLE` improves held-out top-k recall by +0.1381 over global fixed EMA and +0.1365 over per-head fixed EMA, with both paired sequence-bootstrap intervals entirely above zero and 7/7 sequence wins.

The tested causal methods do not recover that headroom. Development-tuned Fast/Slow dual memory is mixed—small improvement only at H=4 and losses at H=16/32—and the follow-up rank-jump gate is worse than per-head fixed by -0.00249 with a 95% interval `[-0.00398, -0.00131]`, losing all seven sequences. No adaptive method passes the future-utility estimator gate, so no new adaptive controller enters an expensive closed-loop model run.

The right conclusion is not “dynamic horizons do not matter.” Dynamic horizons matter to the oracle target, but the current causal attention-history signals do not reveal the correct horizon. Stop unstructured adaptive-horizon heuristic exploration in this branch. Reopen it only with a matched token-time state-feature collection or a learned future-utility policy and a fresh evaluation split.

## 1. Repository and physical-semantics audit

This work extends StateKV in an independent `adaptive_temporal` namespace and does not replace its original results. The main reusable substrate is the recoverable qk_pool trajectory, exact same-input KL measurement, KV backing store, attention capture, and strict pure-eviction controls.

The original StateKV physical-risk controller remains closed-negative. Its teacher evaluates candidate actions through state-conditioned output risk; it is not a tokenwise future-utility score. A historical result that used a backing store cannot be compared as strict pure eviction. The matched main setting here is Qwen3-8B 4-bit, greedy decoding, total budget 256 = sink 4 + recent 32 + core 220, with irreversible inclusion enforced in the pure-eviction comparisons.

The baseline reproduction matches the existing qk_pool budget-256 result: mean trajectory exact KL `0.09758068`, official score `52.74872`, NIAH `1.0`, and GovReport ROUGE-L `0.05497446` across ten samples and 640 decode steps.

## 2. Data and split discipline

The original head-mean trace contains 25,777,152 token rows over ten samples, 64 cycles, and 36 layers. The existing auxiliary head table is unsuitable for dynamic-horizon inference because it covers only three development samples, every fourth cycle, and a selected token subset.

For the Dynamic-Horizon Oracle, the qk_pool trajectory was replayed on the same ten samples while recording all eight KV heads for six preregistered layers `[0, 7, 14, 15, 21, 27]`. The compressed trajectory artifacts are about 6 MB per sample. Across all 640 steps, the replay has zero maximum absolute difference from the original baseline in exact KL, JS, NLL terms, logit L2, and Fisher quadratic.

Development samples are `gov_report:86`, `synthetic_niah_86`, and `synthetic_niah_87`; the other seven sequences are held out. Fixed horizon choices and thresholds were written before held-out oracle evaluation. The rank-drift follow-up is explicitly conditional exploratory analysis after the oracle gate; it uses the same development split and does not retune on held-out results.

## 3. Temporal persistence and fixed-memory baselines

Attention importance is persistent but not stationary. Mean Spearman correlation falls from lag 1 to lag 32:

| Task | Lag 1 | Lag 32 |
|---|---:|---:|
| GovReport | 0.8473 | 0.5608 |
| NIAH | 0.8836 | 0.7027 |

The best head-mean fixed EMA on development is `rho=0.5` for future horizons 1 and 4, and `rho=0.8` for horizons 16 and 32. Held-out future-top-k recall is 0.7779, 0.7812, 0.7786, and 0.7611 respectively. Current attention, cumulative attention, and all adaptive methods are compared on the same eligible tokens and future-attention labels.

Fixed EMA has one scalar state per token/layer. The tested dual-memory gate needs three. At budget 256 over 36 layers, the reported fp32 states are 0.035 MiB for fixed EMA and 0.105 MiB for dual memory; these are state-size calculations and offline NumPy update rates, not end-to-end decode latency measurements.

## 4. Fast/Slow adaptive V1 and bounded variants

The bounded development grid contains 36 parameter configurations and four variants: dynamic-rho, smooth dynamic-rho, dual memory, and rank-adaptive smoothing. The selected method is dual memory with fast `0.5`, slow `0.95`, variance rho `0.5`, threshold `0.25`, and alpha `4`.

| Future horizon | Best fixed EMA | Tuned dual memory | Difference |
|---:|---:|---:|---:|
| 1 | 0.777925 | 0.777176 | -0.000749 |
| 4 | 0.781178 | 0.784830 | +0.003651 |
| 16 | 0.778626 | 0.768734 | -0.009892 |
| 32 | 0.761135 | 0.752450 | -0.008685 |

The H=4 gain is sequence-consistent and small, but the method loses every sequence at H=16 and H=32. It fails the across-horizon estimator gate. Surprise, raw drift, normalized drift, and the other untuned variants do not change that conclusion.

## 5. Dynamic-Horizon Oracle decomposition

Candidates are causal fixed EMAs with `rho` in `{0, 0.25, 0.5, 0.8, 0.9, 0.95, 0.99, 0.995, 0.999}`. Future utility is summed attention over the next H steps for `H={1,4,16,32}`. Global, task, and per-head fixed choices are made on development. The token-time oracle noncausally chooses, for each held-out token and cycle, the candidate whose within-step percentile rank is closest to the future-utility percentile rank.

| Granularity | Held-out recall | Step Spearman | Gain over global |
|---|---:|---:|---:|
| Global fixed | 0.719671 | 0.805851 | 0.000000 |
| Task fixed | 0.719671 | 0.805851 | 0.000000 |
| Per-head fixed | 0.721195 | 0.806946 | +0.001523 |
| Token-time dynamic oracle | 0.857740 | 0.930795 | +0.138069 |

The oracle gain is stable across future horizons: +0.1351, +0.1402, +0.1370, and +0.1338 over per-head fixed at H=1, 4, 16, and 32. Sequence-level paired bootstrap gives:

| Comparison | Mean gain | 95% CI | Wins |
|---|---:|---:|---:|
| Oracle − global fixed | +0.138069 | [0.130739, 0.145361] | 7/7 |
| Oracle − per-head fixed | +0.136546 | [0.128282, 0.144715] | 7/7 |

This passes the preregistered gate of at least +0.01 recall, positive 95% lower bound, and at least five sequence wins. Task-level specialization contributes nothing and per-head fixed specialization contributes little; the opportunity is genuinely token-time rather than a disguised task/head lookup.

This is an upper-bound result. Per-head top-k future-attention recall is not the shared-mask physical eviction policy, and the oracle reads future labels.

## 6. Rank-drift follow-up

After the oracle passed, one new bounded mechanism was allowed. The rank-jump dual gate compares current percentile rank with a causal rank memory, using fast memory after a large jump and slow memory when stable. Thirty-six configurations were selected only on development.

The winner improves H=1 and H=4 by +0.0024 and +0.0057 over per-head fixed, but loses -0.0078 and -0.0103 at H=16 and H=32. Averaged within sequence, its gain is `-0.002493`, with paired 95% CI `[-0.003979, -0.001313]` and 0/7 wins. Rank drift does not capture the oracle opportunity.

No further state-conditioned gate was added. The exact StateKV signal available in the repository is action-level, while the per-head QK/V feature table is sparse and development-only. Combining those mismatched artifacts into a token-time gate would not support a fair held-out claim.

## 7. Relationship to prior work

Temporal persistence and adaptive cache structure are occupied research areas. Scissorhands and H2O establish persistent historical importance; SnapKV and Ada-KV motivate head-specific structure; CAKE and LazyEviction explicitly use temporal dynamics; DynamicKV and MemDecay add task/region conditioning. Moment-KV is the closest direct overlap because it uses momentum-based decode-time temporal aggregation. ForesightKV, Expected Attention, LU-KV, and Learning to Evict target future utility more directly. QEvict overlaps with recoverable eviction and attention drift.

The distinct result here is therefore empirical rather than a new generic decay rule: a matched four-level oracle shows large token-time horizon headroom, while a broad bounded family of causal attention-history gates fails to recover it under StateKV's trajectory. Any future positive method must distinguish itself from these prior approaches and must be evaluated against tuned fixed and per-head fixed memories.

## 8. Final claim ledger

| Claim | Status | Evidence |
|---|---|---|
| Attention importance drifts over decode time | Supported | Lagged rank/attention stability declines |
| Fixed temporal horizon depends on prediction horizon | Supported | Development selects rho 0.5 vs 0.8 |
| Task-fixed horizon adds value | Not supported | Identical to global fixed |
| Per-head fixed horizon adds material value | Not supported | +0.0015 recall only |
| Token-time dynamic-horizon headroom exists | Supported as noncausal upper bound | +0.1365 over per-head, positive CI, 7/7 |
| Fast/Slow adaptive memory beats best fixed EMA | Not supported | Mixed; loses long horizons |
| Rank drift captures dynamic headroom | Rejected for tested mechanism | Negative CI, 0/7 wins |
| Adaptive temporal scoring improves physical closed-loop eviction | Not established | Estimator gate failed; run correctly gated out |

## 9. Research recommendation

Do not spend more cycles on hand-designed drift thresholds, extra rho grids, or live-controller tuning. The current evidence separates two facts cleanly: future-dependent horizon selection has substantial value, but causal attention-history heuristics do not identify it.

A justified next project would require both:

1. a matched full per-head collection of causal state features aligned with token-time oracle labels, or an explicitly learned future-utility predictor; and
2. a fresh task/model evaluation split reserved after method design, followed by the same strict pure-eviction physical gate.

Until those inputs exist, the adaptive-horizon algorithm branch should be considered closed as a negative method finding with a positive oracle finding.

