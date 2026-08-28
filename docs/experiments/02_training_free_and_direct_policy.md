# Current-state policy exploration

## Goal

This stage replaces the expensive state-conditioned evaluator with signals
available from the live cache. It uses Qwen2.5-1.5B-Instruct-4bit with a
shared mask and a 128-token budget (4 sink, 32 recent, 92 core tokens).

## Training-free estimators

The first set of experiments tested history sketches, metric normalization,
Fisher pullbacks, and output-side VJP estimators. Their numerical components
work as intended—for example, VJP consistency reaches maximum relative error
`4.5e-4`—but none improves candidate ranking reliably enough to replace the
physical evaluator. The multi-boundary VJP route is both slower and less
accurate than the simpler alternatives.

The useful observation from this group is that direct contribution scores can
reduce local action error. That result led to policy evaluation rather than
more elaborate pullback approximations.

## Direct selection and refresh policies

| Line | Result | Interpretation |
|---|---|---|
| Direct contribution replay | Mean KL falls from `0.0485` to `0.0199` on the initial held-out replay. | Current attention/contribution has useful local signal. |
| Multi-anchor contribution | Mean KL falls from `0.3293` to `0.2569` on 12 fresh sequences; later 24-sequence retest gives `0.3938` versus attention `0.4085`. | The effect is small and workload dependent. |
| Shrinkage | Mean KL improves from `0.3577` to `0.3298`, without consistent per-unit wins. | It smooths aggregate error but does not create a robust controller. |
| Temporal volatility | Improves several replay tail metrics, yet does not carry to free generation. | Attention dynamics alone are insufficient. |
| Token rarity | Preserves retrieval needles and improves resource use, but trails on GovReport. | A retrieval-oriented heuristic, not a general selector. |
| Selection plus refresh | Latest attention aligns with current selection, while refresh-value ordering reverses on the independent set. | Selection and refresh need different information. |

## Design lessons

Current-cache signals support a strong working-set baseline, but they do not
reliably predict which token will be needed after the next change in query
state. In particular, a single observable does not serve both token selection
and refresh timing. This distinction becomes central in the later causal
future-utility experiments.

Configurations and the complete study catalog are available in
[`configs/stages/`](../../configs/stages/) and
[the experiment catalog](../EXPERIMENT_REGISTRY.md).
