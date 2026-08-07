# StateKV

**State-conditioned physical risk for KV-cache selection and refresh.**

Repeated KV-cache compression is state-dependent: the consequence of a retained
set depends on the compressed trajectory that produced the current model state.

StateKV treats each candidate retained set as a physical intervention at that
state. It propagates the resulting deletion-and-renormalization response to
downstream logits, estimates the associated increase in output risk, and uses the
same risk object for selection and refresh.

The current repository validates a reference-dependent teacher evaluator in
controlled diagnostics. A low-cost online estimator and deployable refresh policy
remain open system components.

![StateKV architecture](assets/statekv-architecture.png)

The editable [TikZ source](assets/statekv-architecture.tex) and vector
[PDF](assets/statekv-architecture.pdf) are versioned with the repository.

## Research question

Static token scores assume that a candidate has the same consequence whenever it
is evaluated. Repeated compression breaks that assumption: earlier evictions
change later queries, attention patterns, residual streams, and KV contents.
StateKV asks:

> Given the compressed trajectory observed now, which retained set produces the
> smallest increase in output risk, and when does state evolution invalidate
> that choice?

Candidate pools can come from baseline selectors or other proposal mechanisms.
StateKV contributes the state-conditioned evaluator shared by the subsequent
decision stages:

1. **Candidate generation** proposes legal retained sets at a fixed budget.
2. **State-conditioned evaluation** estimates each candidate's finite downstream
   effect and incremental output risk under the current trajectory.
3. **Selection and refresh** choose the lowest-risk candidate and re-evaluate it
   as the functional state evolves.

## Method at a glance

1. **Accumulate state.** Earlier compression decisions move the model away from
   its paired full-cache trajectory.
2. **Intervene at the current state.** Each candidate retained set defines a
   concrete deletion-and-renormalization action.
3. **Measure the downstream response.** The action is transported through the
   remaining network to obtain its effect on output logits.
4. **Rank by local output risk.** A state-conditioned KL approximation converts
   the logit response into a scalar candidate risk.
5. **Refresh when the decision changes.** As the compressed trajectory evolves,
   the candidates are re-evaluated against the new functional state.

The teacher makes this pipeline measurable by pairing the compressed run with a
full-cache reference and using deep per-candidate probes. It supplies diagnostic
risk labels for studying selection and refresh; an online policy must replace
those reference-dependent measurements with a cheaper estimator and trigger.

## Minimal formulation

At boundary $b$, compression history is represented by the displacement between
the observed compressed state and its paired full-cache reference:

```math
\mathbf{s}_{t,b}
=
\mathbf{x}^{\mathrm{hist}}_{t,b}
-
\mathbf{x}^{\mathrm{ref}}_{t,b}.
```

For a retained set $C$, let $\widehat{\Delta\mathbf z}(C)$ denote the transported
finite response in output logits. The teacher computes the exact local
deletion-and-renormalization action and evaluates its second-order increment in
reference KL at the observed state:

```math
\widehat{\mathcal R}_{\mathbf s}(C)
=
\mathbf g_{\mathbf s}^{\top}\widehat{\Delta\mathbf z}(C)
+
\frac{1}{2}
\widehat{\Delta\mathbf z}(C)^{\top}
\mathbf F_{\mathbf s}
\widehat{\Delta\mathbf z}(C).
```

Here $\mathbf g_{\mathbf s}$ and $\mathbf F_{\mathbf s}$ are the local first- and
second-order terms of the reference KL geometry. Selection and refresh use the
same risk object:

```math
C^{\star}_{\mathbf s}=\arg\min_{C\in\mathcal A_{t,\ell}(B)}
\widehat{\mathcal R}_{\mathbf s}(C),
\qquad
\text{oracle refresh if }
C^{\star}_{\mathbf s_t}\ne C^{\star}_{\mathbf s_{t-1}}.
```

The refresh label is currently an oracle diagnostic for evaluating future
state-drift triggers.

## Evidence

The [frozen experiment registry](experiments/frozen_registry.yaml) links each
claim to its protocol, manifest, and stored result artifacts.

| Finding | Stored evidence and scope |
|---|---|
| The exact set-level deletion identity reaches a maximum FP64 L2 error of $2.26\times10^{-11}$. | [P0 identity rows](experiments/p0_v2_fixed_boundary/results/identity_rows.parquet), at the fixed operating point and stored candidate protocol. |
| Physical boundary replay reaches sequence-first cosine $\approx 1$ and relative L2 $8.09\times10^{-7}$. | [P0 summary](experiments/p0_v2_fixed_boundary/results/p0_v2_summary.json), under controlled fixed-boundary replay. |
| Evaluation at the observed state reaches cosine $0.99974$ and relative L2 $0.02255$ in the P1 operating-point diagnostic. | [P1 summary](experiments/p1_state_conditioned/results/state_operating_point_summary.json), on four stored sequences from the frozen Qwen protocol. |
| The finite-action approximation has a visible trust region: cosine falls from $0.99986$ at amplitude $1/16$ to $0.95463$ at amplitude $1$, with median residual slope $1.983$. | [R1 summary](experiments/p2_recovery/r1_amplitude_trust_region/results/r1_summary.json), from the retrospective amplitude study. |
| The two-midpoint state-local scalar-risk evaluator obtains Spearman $1.0$ and top-1 gain $1.0$ in both evaluation and replication splits, outperforming the stored action-only ranking. | [Evaluation](experiments/p2_recovery/r4_scalar_decision_risk/results/evaluation/analysis_summary.json) and [replication](experiments/p2_recovery/r4_scalar_decision_risk/results/replication/analysis_summary.json), on frozen candidate pools. |
| Dense all-layer mechanistic risk transfers across the limited P3PR model/task study (Spearman $1.0$ formal and $0.9940$ replication; top-1 $1.0$), while the tested relative single-boundary shortcut does not. | [P3PR summary](experiments/p3pr_generalization/results/analysis/analysis_summary.json), across two model families, two task families, and the stored splits. |

Together, these results support state-conditioned physical evaluation and a
candidate-specific teacher within the frozen protocols.

## Current scope

- **Teacher evaluator:** The supported evaluator uses a paired full-cache
  reference and deep per-candidate probes; the low-cost estimator and deployable
  state-drift trigger are not yet implemented.
- **Mechanistic boundary:** Natural-amplitude full-vector reconstruction and a
  universal relative single-boundary shortcut did not pass their frozen gates.
- **System evaluation:** End-to-end free-generation quality, latency, memory,
  throughput, and refresh overhead still require matched-budget evaluation.
  Same-step KL is an evaluator target rather than a substitute for generation
  quality.

The current project stage and planned estimator, trigger, and closed-loop work
are tracked in [`configs/ccfa.yaml`](configs/ccfa.yaml).

## Reproduce

StateKV targets Python 3.9+. Create an environment and install the canonical
package with:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -e .
```

Run the dependency-light research-path check and repository test suite:

```bash
PYTHONPYCACHEPREFIX=/tmp/statekv-pycache \
  .venv/bin/python scripts/smoke_test.py
.venv/bin/python -m pytest
```

The smoke test covers configuration loading, cache budgeting, baseline and
candidate generation, functional measurement, output-risk utilities, refresh
scheduling, the exact retained-set action, and the stable state/risk/decision
API.

Model-scale runs require local model weights, dataset access, and an appropriate
MPS or CUDA environment. Install and test a backend harness from its own project
root:

```bash
.venv/bin/python -m pip install -e benchmarks/torch
.venv/bin/python -m pip install -e benchmarks/mlx

(cd benchmarks/torch && ../../.venv/bin/python -m pytest tests)
(cd benchmarks/mlx && ../../.venv/bin/python -m pytest tests)
```

## Core Python API

The backend-independent core exposes the state, action, risk, and decision
objects used in the formulation:

```python
from statekv import (
    functional_history_state,
    select_lowest_risk,
    set_level_attention_delta,
    state_conditioned_quadratic_risk,
)

state = functional_history_state(history_boundary, reference_boundary)
boundary_delta = set_level_attention_delta(
    attention, values, retained_positions
)
risk = state_conditioned_quadratic_risk(
    reference_logits, state_logits, candidate_delta_logits
)
decision = select_lowest_risk(
    {"candidate-a": float(risk_a), "candidate-b": float(risk_b)}
)
```

`set_level_attention_delta` implements the exact fixed-operating-point action.
Finite downstream transport remains model-aware and is exposed through
`statekv.core.midpoint_path_response`.

## Repository map

| Path | Role |
|---|---|
| [`statekv/core/`](statekv/core/) | Stable backend-independent state, action, risk, and decision contracts. |
| [`statekv/`](statekv/) | Reusable model-aware probes, features, instrumentation, and analysis logic. |
| [`experiments/`](experiments/) | Frozen phase protocols, manifests, and structured results. |
| [`benchmarks/`](benchmarks/) | PyTorch/CUDA and MLX execution harnesses and baselines. |
| [`configs/`](configs/) | Discovery, staged, and frozen experiment configurations. |
| [`analysis/`](analysis/) | Derived tables, figures, and publication analysis. |
| [`tests/`](tests/) | Core contract, protocol, and frozen-evidence tests. |
| [`assets/`](assets/) | Architecture source and rendered deliverables. |

Stable contracts live in `statekv.core`; backend-specific execution and frozen
experiment protocols depend on those contracts, not the reverse.

## Citation and license

The StateKV paper is in preparation and has no archival citation identifier. For
now, cite the repository commit and the relevant frozen experiment manifest.

See [LICENSE](LICENSE) for licensing terms.
