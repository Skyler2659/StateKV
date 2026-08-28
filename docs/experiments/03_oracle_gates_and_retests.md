# Closed-loop policies and working-set control

## From physical scoring to live generation

This program evaluates the StateKV physical evaluator, practical cache
policies, and strict eviction semantics in closed-loop generation. It moves
from teacher-forced replay to recoverable and irreversible cache settings on
Qwen3-8B.

## Closed-loop observations

The physical evaluator provides accurate rankings on controlled candidate
panels and works mechanically with cold recovery. In autoregressive
generation, however, one-step physical risk does not distinguish the best
actions often enough to outperform a simple working-set policy. In a
recoverable full-pool setting, exact per-query QK routing (`qk_pool`) reaches
mean KL `0.0086`, while the physical teacher reaches `0.0213`.

The evaluation therefore identifies QK routing as the practical baseline for
this repository. It selects the highest-attended historical keys for the
current query and remains effective at long context on the tested models.

## QK routing, coverage, and cadence

| Setting | Observation | Consequence |
|---|---|---|
| QK versus V routing | Attention has much larger dynamic range than projected values; adding V does not recover a useful residual ranking. | QK is the primary routing signal. |
| Value tiering | 4-bit cold-V tiering gives KL `0.008066` versus QK `0.007614` with identical task scores at matched budget. | Tiering is a near-lossless memory option. |
| Coverage | Raising QK coverage from 256 to 352 FP16 tokens reduces KL to `0.499x` of the 256-token baseline. | Absolute core size is the dominant resource. |
| Cadence | At context 768 / budget 64, slowing refresh from every token to every 16 tokens changes NIAH from `1.0` to `0.0`. | Tight budgets require rapid updates. |
| Longer context | QK routing retains NIAH `3/3` at roughly 4.7K context on Qwen3-8B and Qwen2.5-7B. | The QK working-set behavior is not specific to one model family. |

## Retest summary

Fresh-sequence retests clarify several early observations. Contribution
selection retains a small mean-KL improvement but no large task effect;
matched-budget value tiering is equivalent to QK routing on task scores;
token rarity remains retrieval-specific; and temporal-volatility controllers
are not competitive in the newer free-generation setting.

## Outcome

This phase establishes a practical control baseline and exposes the remaining
problem: current-query routing cannot preserve information needed only by a
future query. The next reports investigate that future-utility signal
directly. Technical details are available in the
[experiment catalog](../EXPERIMENT_REGISTRY.md) and
[`../evidence/`](../evidence/).
