# Mechanism foundations

## Question

Can the effect of a KV-cache action be measured at the model state where that
action is taken? This report establishes the state-conditioned evaluator that
motivated the first StateKV line of work.

The frozen studies use Qwen2.5/Llama-scale models, GovReport, and synthetic
retrieval workloads. Their manifests and original artifacts are retained in
[`experiments/`](../../experiments/).

## Results

| Study | Finding | Representative result |
|---|---|---:|
| Fixed-boundary deletion | The analytic deletion-and-renormalization response matches replay on a shared execution graph. | FP64 maximum L2 error `2.26e-11`; replay relative L2 `8.09e-7` |
| State-conditioned response | The same deletion can have a different consequence after a different compression history. | cosine `0.99974`; relative L2 `0.02255` |
| Finite-action scalar risk | A path-corrected scalar risk ranks a frozen candidate panel reliably. | Spearman `1.0`; top-1 gain `1.0` in evaluation and replication |
| Same-state physical teacher | Evaluating candidates in their own physical state recovers the candidate ordering. | 8-candidate ranking recovered at the measured boundary |
| Dense mechanism transfer | The dense risk construction transfers across the tested models and task families. | formal Spearman/top-1 `1.0`; replication Spearman `0.9940` |

## What the studies establish

The cache trajectory matters: an action cannot generally be evaluated from a
state that was produced by a different compression history. For a fixed
candidate panel, the project can compute a physically meaningful scalar risk
and use it to rank actions.

Several simplifying assumptions were also tested. A native 4-bit prediction
path and a full-vector natural-amplitude reconstruction did not supply a
usable common target, while a local Jacobian remained useful only as a
short-range mechanism probe. These observations motivated the same-state,
candidate-specific evaluator rather than a global linear surrogate.

## Outcome

This phase delivered a high-fidelity **research evaluator**, not an online
policy. Its role is to make state-dependent cache decisions measurable; later
reports ask whether its information can be obtained cheaply enough for
generation-time control.

Key artifacts include
[`p0_v2_summary.json`](../../experiments/p0_v2_fixed_boundary/results/p0_v2_summary.json),
the P2 recovery analyses, and the P3 physical-recovery manifests under
[`experiments/`](../../experiments/).
