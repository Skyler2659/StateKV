# Adaptive Temporal Memory: literature map

Search date: 2026-08-19. Sources below are primary paper or official conference pages. The search targeted temporal attention persistence, decay, per-head/task adaptation, and future-utility supervision in autoregressive KV-cache eviction. This map excludes serving-only cache placement, pure quantization, diffusion/video caches, and non-primary summaries.

## Closest temporal-persistence line

- [Scissorhands](https://arxiv.org/abs/2305.17118) motivates eviction through a persistence-of-importance hypothesis. It is the clearest early premise that historically important tokens remain important, but it does not establish that a token-time-varying forgetting horizon is necessary.
- [H2O](https://arxiv.org/abs/2306.14048) combines recent tokens with cumulative-attention heavy hitters. It is the canonical long-memory control for this branch.
- [SnapKV](https://arxiv.org/abs/2404.14469) reports head-specific prompt-attention patterns inferred from a prompt-end observation window. It supports measuring head heterogeneity, although its compression decision is not a decode-time adaptive EMA horizon.
- [CAKE](https://arxiv.org/abs/2503.12491) explicitly uses spatial and temporal attention dynamics for layer-aware allocation and a temporally informed eviction indicator. Any claim that temporal dynamics in KV eviction are new would therefore be untenable.
- [LazyEviction](https://arxiv.org/abs/2506.15969) tracks recurrence intervals to preserve tokens that regain importance in long reasoning. It is directly relevant to revival/rank-drift diagnostics.
- [Moment-KV](https://arxiv.org/abs/2605.29873) is the closest algorithmic overlap: it uses momentum-driven temporal attention aggregation with decay during long generation. A StateKV adaptive-forgetting contribution would need to differ through a validated dynamic-horizon mechanism, physical-risk target, or stronger oracle/negative result—not merely by using decayed attention.
- [MemDecay](https://arxiv.org/abs/2607.10582) assigns semantic regions different priorities and decay rates calibrated from attention lifetimes. It makes task/region-conditioned decay an occupied design point and suggests that interpretable structure may be more reliable than unconstrained tokenwise gating.

## Granularity and allocation

- [Ada-KV](https://arxiv.org/abs/2407.11550) allocates cache budget by head. It motivates a per-head fixed baseline but addresses budget allocation, not temporal-memory horizon selection.
- [PyramidKV](https://arxiv.org/abs/2406.02069) allocates different capacities across layers using pyramidal information funneling.
- [DynamicKV](https://arxiv.org/abs/2412.14838) adapts layer retention to task characteristics. It makes task-level adaptation a necessary comparison rather than evidence for token-time adaptation.
- [LU-KV](https://arxiv.org/abs/2602.08585) profiles long-horizon utility and optimizes head-level allocation. It is especially relevant to separating head allocation headroom from token-time horizon headroom.

## Future-utility prediction and learned policies

- [Expected Attention](https://arxiv.org/abs/2510.00636) estimates how future queries will attend to cached entries in closed form. It establishes a training-free future-query alternative to historical-attention heuristics.
- [ForesightKV](https://arxiv.org/abs/2602.03203) constructs a future-attention “Golden Eviction” target and distills it with supervised ranking and reinforcement learning.
- [Learning to Evict from Key-Value Cache](https://arxiv.org/abs/2602.10238) trains lightweight per-head policies against future utility. It directly occupies the learned future-utility-ranking space.
- [Attention-Gate](https://arxiv.org/abs/2410.12876) learns head/layer-varying eviction flags from context. A state-conditioned gate is therefore not novel without a materially different signal, constraint, or evidence result.

## Recoverability and output correction

- [QEvict](https://arxiv.org/abs/2608.05326) reports attention drift and a recoverable quantized tier. It is a near-concurrent overlap with StateKV's recoverable-pool framing; claims must distinguish matched physical semantics and avoid presenting recoverability itself as new.
- [MomentKV](https://arxiv.org/abs/2606.01563) keeps moment statistics for evicted keys/values and corrects the attention output. It addresses directional error rather than adaptive forgetting, but it reinforces that attention mass alone is an incomplete risk proxy.

## Consequence for this branch

The defensible research question is narrow: after controlling for global, task, and per-head fixed temporal horizons, does a noncausal token-time dynamic-horizon oracle retain meaningful held-out future-utility headroom on the exact StateKV trajectory? If not, a new rank-drift or state-conditioned gate lacks an evidence-backed target and the correct contribution is a negative finding. If the oracle does show significant headroom, any causal method must still beat tuned fixed EMA and distinguish itself from Moment-KV, MemDecay, CAKE, Attention-Gate, ForesightKV, and learned per-head future-utility policies.

