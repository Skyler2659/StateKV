# StateKV

**StateKV is a research codebase for KV-cache compression under changing
generation state.** It studies how a cache should retain information that is
not important now but will become important to an upcoming query, and
introduces **Cheap-R2**, a one-shot full-cache foresight policy for this
setting.

The repository includes the active evaluation runtime, frozen experiment
artifacts, and a compact record of the research path from physical risk
modeling to future-utility control.

## Highlights

- **Cheap-R2.** At query onset, the target model rolls out 32 future tokens on
  the full prefix, accumulates each historical token's future attention
  utility, and performs one strict physical eviction. The selected ranking is
  reused for the rest of decoding.
- **Strong tight-budget retrieval result.** On fresh multikey retrieval with
  Qwen3-8B-4bit, Cheap-R2 reaches 48.5 / 70.0 / 81.0 at KV budgets
  128 / 256 / 512. Current-QK reaches 1.0 / 21.0 / 25.5; LAQ reaches
  0.0 / 29.5 / 84.5.
- **A precise operating point.** At budget 256, Cheap-R2 improves multikey
  accuracy from 21.0 to 70.0 over current-QK at 1.41x its wall time. The gain
  is specific to retrieval-heavy, tight-memory workloads; it does not
  translate into a stable advantage on the evaluated LongBench QA slices.
- **A reusable research platform.** The active code supports recoverable and
  strict-eviction evaluation, exact output metrics, paired bootstrap
  statistics, QK routing, Cheap-R2, and causal student experiments.

## Research question

Most KV-cache policies rank tokens from the current query. That is sufficient
when today's attention predicts tomorrow's demand, but it can discard evidence
before a delayed query activates it. StateKV asks whether future token utility
can be estimated causally and whether that estimate improves memory-limited
generation.

The project explored three successive ideas:

1. **State-conditioned physical risk.** An exact same-state evaluator can
   measure the effect of a candidate cache action, but its computational cost
   prevents it from becoming a practical online controller.
2. **Cheap observable policies.** Attention, contribution, geometry, and
   refresh-based signals establish strong working-set baselines, especially
   exact per-query QK routing.
3. **Causal future utility.** R2 uses a target-model rollout to estimate which
   historical tokens an imminent query will need. Cheap-R2 preserves this
   information with a single query-onset action rather than repeated refresh.

## Main results

The current benchmark uses Qwen3-8B-4bit, strict physical eviction, 64 decode
cycles, and fresh multikey sequences. Scores are percentage accuracy.

| Multikey budget | Full cache | Current-QK | LAQ | Cheap-R2 |
|---:|---:|---:|---:|---:|
| 128 | 82.0 | 1.0 | 0.0 | **48.5** |
| 256 | 82.0 | 21.0 | 29.5 | **70.0** |
| 512 | 82.0 | 25.5 | **84.5** | 81.0 |

Cheap-R2 is most useful when a query depends on several dispersed pieces of
evidence that current attention has not yet activated. On the tested HotpotQA
and 2WikiMQA slices it is statistically indistinguishable from the cheaper
current-QK policy. The 64-token GovReport setup is retained as an artifact but
is not used to rank methods because it evaluates only an opening fragment.

See the [benchmark report](docs/experiments/08_benchmark_results.md) for the
full protocol, paired comparisons, and cost breakdown.

## Method: Cheap-R2

```text
full prefix KV ──> target-model H=32 rollout ──> future attention utility
                                                          │
new query ───────────────────────────────────────────────┘
                                                          ↓
                                             one strict eviction
                                                          ↓
                                            fixed cache for decoding
```

Unlike degraded-cache lookahead methods, the rollout happens before eviction,
so evidence that is weak under the current query can still influence the
future ranking. [The Cheap-R2 report](docs/experiments/07_cheap_r2.md)
describes the horizon, refresh, and baseline ablations.

## Getting started

StateKV requires Python 3.9+; model-scale runs use Apple-silicon MLX and local
model/dataset caches.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -e .
.venv/bin/python -m pip install -e benchmarks/torch
.venv/bin/python -m pip install -e benchmarks/mlx

PYTHONPATH=benchmarks/mlx .venv/bin/python -m pytest -q
PYTHONPATH=benchmarks/mlx .venv/bin/python scripts/smoke_test.py
```

Representative model runs:

```bash
# Recoverable free-generation policy panel
HF_HUB_OFFLINE=1 .venv/bin/python scripts/run_retest_freegen.py \
  --config configs/stages/retest_freegen_qwen3_8b_n20_protocol.yaml

# QK routing and value-tier comparison
HF_HUB_OFFLINE=1 .venv/bin/python scripts/run_oracle_policy_freegen.py \
  --config configs/stages/statekv_qkvtier_gate_256t.yaml

# R2 causal rollout study
HF_HUB_OFFLINE=1 .venv/bin/python scripts/run_causal_rollout_study.py \
  --config configs/statekv_counterfactual/r2_student_qwen3_8b.yaml
```

## Documentation

| Start here | What it covers |
|---|---|
| [Project guide](docs/README.md) | A concise map of the public documentation |
| [Experiment reports](docs/experiments/README.md) | The complete research story in eight reports |
| [Cheap-R2](docs/experiments/07_cheap_r2.md) | Final method, ablations, and mechanism |
| [Benchmark results](docs/experiments/08_benchmark_results.md) | Main results, statistics, cost, and task scope |
| [Research results](docs/FINDINGS.md) | Cross-experiment findings and design lessons |
| [Architecture](docs/CODE_AUDIT.md) | Active modules and execution paths |
| [Running experiments](docs/REPRODUCIBILITY.md) | Environment and active experiment entry points |

Historical protocols, run summaries, and supporting tables remain under
[`docs/evidence/`](docs/evidence/) and `results/`. They provide technical
background for readers who want to trace a particular result.

## Repository layout

| Path | Purpose |
|---|---|
| [`statekv/`](statekv/) | Active cache-control, R2, and evaluation runtime |
| [`statekv/core/`](statekv/core/) | Backend-independent state, action, risk, and decision contracts |
| [`benchmarks/`](benchmarks/) | MLX runner and small torch compatibility layer |
| [`configs/`](configs/) | Active and frozen experiment configurations |
| [`docs/`](docs/) | Project guide, reports, and technical appendices |
| [`experiments/`](experiments/) | Frozen manifests and completed experiment artifacts |
| [`results/`](results/) | Canonical run summaries and local raw artifacts |
| [`tests/`](tests/) | Contract and active-runtime tests |

## Citation and license

StateKV does not yet have an archival citation identifier. Please cite the
repository revision together with the relevant experiment configuration and
report. See [LICENSE](LICENSE) for licensing information.
