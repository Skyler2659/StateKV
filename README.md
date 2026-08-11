# StateKV

**An experimental research codebase documenting an investigation into
state-conditioned KV-cache compression — including both positive and
negative results.**

> **Research status (2026-08-10).** The original StateKV method line — a
> deployable controller driven by state-conditioned physical retained-set
> risk — is **closed as a negative result** under preregistered
> deployment-faithful gates and should not be cited as a validated method.
> What survives is the mechanism-level evaluator, a strong oracle baseline
> (exact per-query full-pool QK routing), a mapped coverage × cadence
> failure frontier, a systematically falsified search space (41 rejected
> strategies with verified numbers), and a reusable evaluation
> infrastructure. Read [`docs/RESEARCH_HISTORY.md`](docs/RESEARCH_HISTORY.md)
> first, then [`docs/FINDINGS.md`](docs/FINDINGS.md).

## What was investigated

Repeated KV-cache compression is state-dependent: the consequence of a
retained set depends on the compressed trajectory that produced the current
model state. StateKV treated each candidate retained set as a physical
intervention at that state, propagated the deletion-and-renormalization
response to output logits, and used the resulting risk for both selection
and refresh. Its supported identity was a same-state physical evaluator:
an expensive, candidate-specific teacher that rolls out a
legal action panel at every boundary, with the intention that a deployable
online policy would eventually replace the teacher's candidate rollouts
with one cheap observable. L1/L2 leverage, attention, recency, value norm,
SnapKV, and H2O appear as baselines, proposal mechanisms, or diagnostics —
not as StateKV itself.

The investigation ran through: frozen mechanism validation (P0–P3PR),
training-free cheap estimators (TF-P0–P5), teacher-forced direct policies
(P6–P24), a physical-oracle closed loop and cheap controllers (P25–P35),
deployment-faithful headroom gates (Gate 0–2, R0), a QK–V mechanism
battery, a selective-refresh trigger line (R1–R2), an open-ended search
(HF1–HF6), an external-validity challenge at longer context, and a final
no-gate retest of every marginally rejected policy.

## What survived current evidence

- The state-conditioned physical risk **evaluator** is exact and accurate
  (identity error 2.26e-11; state-conditioned cosine 0.99974; frozen-pool
  ranking Spearman 1.0). It remains a sound research instrument.
- **Exact per-query full-pool QK routing (qk_pool)** was never beaten by
  any cheap or expensive alternative at any quality-valid operating point,
  across two model families and coverage down to 1.4%.
- The **coverage × cadence cliff**: at tight budgets, slow refresh is
  catastrophic (NIAH 1.0 → 0.0 from h1 to h16 at 8% coverage) and no
  refresh-time scoring rescues it. The controlling variable is the absolute
  core budget.
- **Cold-V 4-bit tiering under QK routing is near-lossless** at matched
  budget (KL within 6% of qk_pool, identical task scores).
- Full graded list: [`docs/FINDINGS.md`](docs/FINDINGS.md).

## What failed

- The deployable distillation of the physical-risk teacher: one-step risk
  is a plateau (61.6% of cycles tied) and long-run risk is a cliff carried
  by future-queried tokens, visible only 2–4 steps ahead — no cheap signal
  exists to distill, and the expensive teacher's apparent headroom was a
  forbidden-information artifact.
- Every training-free cheap estimator tested (sketches, metric repair,
  Fisher pullbacks, VJP routes), dynamic layer budgets, selective refresh
  triggers on high-coverage substrates, page-granular approximations, and
  observation-window scoring.
- Full falsified list: [`docs/FINDINGS.md`](docs/FINDINGS.md) §D and the
  verified 41-entry catalog
  [`docs/evidence/statekv_gate_retrospective_catalog.md`](docs/evidence/statekv_gate_retrospective_catalog.md).
- Why the early experiments looked successful while the final policy had no
  advantage: [`docs/FAILURE_ANALYSIS.md`](docs/FAILURE_ANALYSIS.md).

## Why the repository is still useful

- A validated exact-KL closed-loop evaluation stack (recoverable CPU
  backing store, cold recovery, per-cycle telemetry, paired bootstrap).
- A one-config-key policy panel: attention, SnapKV, H2O, uniform, qk_pool,
  quest_like, qk_obswin, qk_tiered_v, token_rarity, and the A1–B3 cheap
  controllers.
- A 79-method eviction baseline library (`benchmarks/mlx`).
- Frozen mechanistic phases with registries and tests (`experiments/`).
- A curated negative-result archive: every rejected strategy, its veto
  numbers, and its retest outcome.
- Where to start next: [`docs/NEXT_RESEARCH_DIRECTIONS.md`](docs/NEXT_RESEARCH_DIRECTIONS.md).

## Documentation map

| Document | Content |
|---|---|
| [`docs/README.md`](docs/README.md) | Documentation index and reading order |
| [`docs/RESEARCH_HISTORY.md`](docs/RESEARCH_HISTORY.md) | Phase-by-phase history, believed-then vs known-now |
| [`docs/FINDINGS.md`](docs/FINDINGS.md) | Graded findings (strong / conditional / observations / negative / open) |
| [`docs/FAILURE_ANALYSIS.md`](docs/FAILURE_ANALYSIS.md) | Evidence-graded explanation of the main line's failure |
| [`docs/EXPERIMENT_REGISTRY.md`](docs/EXPERIMENT_REGISTRY.md) | Every experiment: question, substrate, config, result, status |
| [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) | Finding → config → script → raw-result chain audit |
| [`docs/CODE_AUDIT.md`](docs/CODE_AUDIT.md) | Module classification, defensive-code risks, duplicates |
| [`docs/NEXT_RESEARCH_DIRECTIONS.md`](docs/NEXT_RESEARCH_DIRECTIONS.md) | Empirical gaps worth building on |
| [`docs/evidence/`](docs/evidence/) | Raw experiment protocols, logs, closure reports, gate verdicts |
| [`docs/proposals/`](docs/proposals/) | Architecture/design notes |
| [`experiments/frozen_registry.yaml`](experiments/frozen_registry.yaml) | Frozen phase protocols, manifests, result links |
| [`configs/ccfa.yaml`](configs/ccfa.yaml) | Claims registry with per-claim evidence paths |
| [`analysis/`](analysis/) | Analysis code and derived CSV/parquet tables |

## Repository map

| Path | Role |
|---|---|
| [`statekv/core/`](statekv/core/) | Stable backend-independent state/action/risk/decision contracts |
| [`statekv/`](statekv/) | Model-aware probes, policies, runners (core vs legacy split in `docs/CODE_AUDIT.md`) |
| [`experiments/`](experiments/) | Frozen mechanistic phases with manifests and tests |
| [`benchmarks/`](benchmarks/) | MLX and torch harnesses; 79-method eviction baseline registry |
| [`configs/`](configs/) | Discovery, staged, and frozen experiment configurations |
| [`analysis/`](analysis/) | Closure documents, derived tables, figures |
| [`results/`](results/) | Run artifacts (canonical summaries tracked; raw tensors local) |
| [`tests/`](tests/) | Contract, invariant, protocol, and frozen-evidence tests |
| [`docs/archive/queues/`](docs/archive/queues/) | Archived run queues and smoke traces |

## Reproduce

Environment (Python 3.9+, Apple-silicon MLX for model runs):

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -e .
.venv/bin/python -m pip install -e benchmarks/torch
.venv/bin/python -m pip install -e benchmarks/mlx
PYTHONPATH=benchmarks/mlx .venv/bin/python -m pytest
PYTHONPATH=benchmarks/mlx PYTHONPYCACHEPREFIX=/tmp/statekv-pycache \
  .venv/bin/python scripts/smoke_test.py
```

Representative canonical commands (full chain audit:
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md); model runs need local
HF caches and are compute-heavy):

```bash
# Era-1 direct-policy replay re-screen (no gates, 24 sequences)
HF_HUB_OFFLINE=1 .venv/bin/python scripts/run_direct_policy_replay.py \
  --config configs/stages/retest_replay_era1_n24_protocol.yaml

# Era-2 no-gate multi-policy recoverable freegen panel
HF_HUB_OFFLINE=1 .venv/bin/python scripts/run_retest_freegen.py \
  --config configs/stages/retest_freegen_qwen3_8b_n20_protocol.yaml

# qk_tiered_v matched-budget arm
HF_HUB_OFFLINE=1 .venv/bin/python scripts/run_oracle_policy_freegen.py \
  --config configs/stages/statekv_qkvtier_gate_256t.yaml

# P31 exact-risk teacher (expensive; per-decision candidate rollouts)
HF_HUB_OFFLINE=1 .venv/bin/python scripts/run_oracle_policy_freegen.py \
  --config configs/stages/oracle_policy_freegen_qwen3_8b_n10_protocol.yaml

# P32 cheap controllers
HF_HUB_OFFLINE=1 .venv/bin/python scripts/run_cheap_policy_freegen.py \
  --config configs/stages/cheap_policy_freegen_qwen3_8b_n10_protocol.yaml

# Retrospective training-free gates (no model run needed)
.venv/bin/python scripts/analyze_training_free_sketch.py \
  --config configs/stages/training_free_sketch_config.yaml
.venv/bin/python scripts/analyze_metric_repair.py \
  --config configs/stages/training_free_metric_repair_config.yaml
```

Earlier-phase reproduction commands (P4–P28) are unchanged from the
previous README revision (git history) and are catalogued with their
configs and result paths in `docs/EXPERIMENT_REGISTRY.md`.

## Core Python API

Backend-independent contracts (stable, tested in `tests/core/`):

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

## Citation and license

StateKV has no archival citation identifier. Cite the repository commit and
the relevant frozen experiment manifest or run summary. See
[LICENSE](LICENSE) (note: the copyright line predates this project — see
`docs/CODE_AUDIT.md` §7).
