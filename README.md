# StateKV

**State-conditioned physical risk for KV-cache selection and refresh.**

StateKV studies how a concrete KV-cache action changes model-output risk under
the observed compressed state created by earlier cache decisions. Selection
chooses among candidate actions in that state; refresh decides when state
evolution makes an earlier choice unreliable. L1/L2 leverage, attention,
SnapKV, H2O, value norm, age, Fisher and related geometry are retained as
baselines, candidate generators or diagnostics, not as StateKV itself.

This is the repository's only Markdown document. Detailed machine-readable
evidence remains in manifests, ledgers, YAML, JSON, CSV and Parquet artifacts.

## Scientific scope

Repeated cache compression is a sequential intervention:

```text
compression history
  -> observed compressed state
  -> candidate retained/deletion action
  -> state-conditioned physical risk
  -> selection validity
  -> refresh decision
```

A static token score omits the first term. Earlier evictions can change later
queries, attention, residual streams and KV contents, so the same candidate can
have different risk in different compressed states. StateKV separates:

1. candidate generation using attention, leverage, age or other selectors;
2. risk evaluation under the current functional or physical state;
3. selection and refresh policy driven by that risk object.

The strongest current positive results are deliberately narrow:

- A controlled reference-dependent evaluator closes scalar ranking for frozen
  retained-set candidate pools after exact deletion injection, finite-action
  transport and state-local KL/Fisher readout.
- A same-state physical evaluator closes candidate ranking for an observed
  compressed prequery state under the frozen singleton-deletion protocol.
- Dense all-layer mechanism evidence transfers across the limited tested
  model/task scope, while the successful candidate-specific teacher remains too
  expensive to be an online policy.

The repository does not yet establish a deployable low-cost controller,
subset-level physical closure, free-generation quality preservation, or full
latency/memory/throughput gains. Same-step KL is an evaluator target, not a
substitute for downstream generation quality.

Important negative results are preserved as scientific evidence:

- controlled single-boundary risk does not directly transfer across propagated
  all-layer physical histories;
- full-vector reconstruction did not replicate where scalar ranking did;
- static single-/multi-boundary geometry and low-dimensional summaries did not
  replace candidate-conditioned deep response;
- the successful late boundary on one model is not a universal rule.

## Repository layout

```text
statekv/               canonical StateKV implementation
statekv/io/            artifact, schema and provenance interfaces
configs/               active, staged and frozen StateKV configurations
scripts/               StateKV experiment, validation and analysis entrypoints
experiments/           frozen phase code, manifests, ledgers and result data
benchmarks/mlx/        Apple-Silicon benchmark and baseline harness
benchmarks/torch/      protocol-aware PyTorch/CUDA benchmark
analysis/              structured StateKV analysis pipeline and tables
results/               StateKV discovery and mechanism artifacts
artifacts/             run-registry schema and artifact governance
tests/                 unit, architecture, provenance and golden tests
```

The benchmark projects are supporting infrastructure. They must not import
phase-specific experiment code or define the StateKV research claim. New shared
research logic belongs under `statekv/`; new backend-specific execution logic
belongs under `benchmarks/<backend>/`.

The repository has no compatibility symlinks. Code uses canonical paths such as
`benchmarks/torch`, `configs/frozen` and `experiments/<phase>` directly.

## Installation

The current local environment is Python 3.9 based and uses both backend
projects. Install in this order:

```bash
python -m venv .venv
.venv/bin/python -m pip install -e benchmarks/torch
.venv/bin/python -m pip install -e benchmarks/mlx
.venv/bin/python -m pip install -e .
```

Backend requirements are declared separately in:

- `pyproject.toml`
- `benchmarks/mlx/pyproject.toml`
- `benchmarks/torch/pyproject.toml`

Model-scale work additionally requires local model weights, dataset access and
appropriate MPS or CUDA hardware. Unit-test success alone does not validate a
full model run.

## Verification

Run all automated suites without writing pytest caches into the repository:

```bash
PYTHONPYCACHEPREFIX=/tmp/statekv-pycache \
  .venv/bin/python -m pytest -q -p no:cacheprovider

cd benchmarks/mlx
PYTHONPYCACHEPREFIX=/tmp/statekv-mlx-pycache \
  ../../.venv/bin/python -m pytest -q -p no:cacheprovider

cd ../torch
PYTHONPYCACHEPREFIX=/tmp/statekv-torch-pycache \
  ../../.venv/bin/python -m pytest -q -p no:cacheprovider
```

Static syntax and dependency checks:

```bash
PYTHONPYCACHEPREFIX=/tmp/statekv-compile \
  .venv/bin/python -m compileall -q statekv scripts tests experiments benchmarks
.venv/bin/python -m pip check
```

Five golden tests are intentionally skipped until a researcher exports audited
`frozen_v1` fixtures. No cache tensor, logit, risk value or ranking is fabricated
to make those tests pass.

## Main StateKV entrypoints

Mechanism-discovery and state-evolution runs:

| Purpose | Entrypoint | Configuration |
|---|---|---|
| Temporal discovery | `scripts/run_temporal_discovery.py` | `configs/discovery/discovery_small.yaml` |
| Functional probe | `scripts/run_functional_probe.py` | `configs/discovery/functional_probe_stage1_4bit.yaml` |
| Mechanism-targeted analysis | `scripts/run_mechanism_targeted.py` | `configs/discovery/mechanism_targeted_4bit.yaml` |
| Gauge geometry | `scripts/run_gauge_geometry.py` | `configs/stages/gauge_geometry_config.yaml` |
| Independent Fisher | `scripts/run_independent_fisher.py` | `configs/stages/independent_fisher_config.yaml` |
| Output sensitivity | `scripts/run_output_sensitivity.py` | `configs/stages/output_sensitivity_config.yaml` |
| Robust envelope | `scripts/run_robust_envelope.py` | `configs/stages/robust_envelope_config.yaml` |
| Trajectory model | `scripts/run_trajectory_model.py` | `configs/stages/trajectory_model_config.yaml` |
| Theory closing | `scripts/run_theory_closing.py` | `configs/stages/theory_closing_config.yaml` |

Reusable logic is implemented in `statekv/backend.py`, `backend_mlx.py`,
`mechanism.py`, `selectors.py`, `signals.py`, `metrics.py`,
`functional_probe.py`, `gauge_geometry.py`, `independent_fisher.py`,
`output_sensitivity.py`, `robust_envelope.py`, `trajectory_model.py` and
`theory_closing.py`.

Backend benchmark entrypoints remain under `benchmarks/mlx/scripts` and
`benchmarks/torch/scripts`. Historical MLX benchmark outputs were removed from
the working tree; StateKV result data remains under
`results/temporal_cache_discovery`.

## Frozen evidence registry

`experiments/frozen_registry.yaml` is the repository-level inventory. Each
phase manifest remains authoritative for the exact evaluation-time protocol,
checksums and bounded claim.

| Phase | Status | Manifest |
|---|---|---|
| Predictive closure | frozen negative evidence | `experiments/predictive_closure/experiment_manifest.yaml` |
| Local truncated Jacobian | frozen boundary evidence | `experiments/local_truncated_jacobian/EXPERIMENT_MANIFEST.yaml` |
| P0-v2 fixed boundary | frozen positive evidence | `experiments/p0_v2_fixed_boundary/P0_V2_MANIFEST.yaml` |
| P1 state-conditioned | frozen boundary evidence | `experiments/p1_state_conditioned/P1_STATE_CONDITIONED_MANIFEST.yaml` |
| P2 state-local risk | frozen negative evidence | `experiments/p2_state_local_risk/P2_STATE_LOCAL_MANIFEST.yaml` |
| P2 recovery | frozen recovery evidence | `experiments/p2_recovery/P2_RECOVERY_MANIFEST.yaml` |
| P3 decision validity | frozen negative evidence | `experiments/p3_decision_validity/P3_DECISION_VALIDITY_MANIFEST.yaml` |
| P3 physical recovery | frozen recovery evidence | `experiments/p3_physical_recovery/P3_PHYSICAL_RECOVERY_MANIFEST.yaml` |
| P3PR generalization | frozen generalization evidence | `experiments/p3pr_generalization/P3PR_GENERALIZATION_MANIFEST.yaml` |

Negative evidence is not a failed run. Failed, interrupted, smoke and obsolete
protocol artifacts remain distinct statuses.

Historical Markdown reports have been retired from the checkout. Their paths
and SHA-256 values are recorded in `experiments/retired_documents.yaml`.
Evaluation-time filenames and hashes remain unchanged inside phase manifests;
`statekv/repository_layout.py` distinguishes an intentionally retired document
from an unrecorded missing source. Path-only source migrations are separately
bound in `experiments/layout_migrations.yaml`.

## Reproducibility rules

Do not rerun a frozen experiment into its original results directory. For a new
run:

1. copy the relevant frozen configuration into a new experiment/config ID;
2. choose a new output directory;
3. record random seeds and deterministic settings;
4. record model and tokenizer identifiers plus immutable revisions;
5. record dataset identifiers, revisions and exact sample IDs;
6. record the Git commit and dirty-diff hash;
7. save the executed command and resolved configuration;
8. write structured JSON/CSV/Parquet artifacts and a run record conforming to
   `artifacts/registry.schema.yaml`;
9. classify the result as complete, negative-result, failed-run,
   interrupted-run, obsolete-protocol, smoke or in-progress.

The frozen P0/P1/P2 inputs live under `configs/frozen`. New discovery configs
belong under `configs/discovery`; later risk-model stages belong under
`configs/stages`. Generic cache-method/model/task benchmark configs belong under
their backend project, not under the StateKV config root.

## Data and output policy

- `results/temporal_cache_discovery` contains StateKV raw and derived mechanism
  artifacts. Large Parquet/CSV candidate, trajectory, Fisher and sensitivity
  tables are research data, not source code.
- `analysis/tables` and `analysis/figures` contain derived paper-analysis
  material.
- `experiments/<phase>/results` contains frozen phase artifacts.
- Generic benchmark outputs belong under `benchmarks/<backend>/results` and are
  disposable when reproducible and not selected as paper evidence.
- Future raw, derived and failed payloads should follow the schemas under
  `artifacts/` or live in an external artifact store.

The repository intentionally keeps exactly one Markdown file: this README.
Intermediate explanations, meeting notes, duplicated reports, figure indexes,
backend mini-READMEs and generated Markdown summaries must not be added. Put
machine-consumable results in JSON, YAML, CSV or Parquet and update this README
only when repository-wide guidance changes.

## Citation and license

The StateKV paper is in preparation and has no archival citation identifier.
Until one is published, cite the repository commit and the relevant frozen
experiment manifest. Do not cite the earlier L1/geometry project title as the
StateKV method name.

The project license is in `LICENSE`.
