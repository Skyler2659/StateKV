# Adaptive Temporal Memory: repository audit

Date: 2026-08-19  
Branch: `codex/adaptive-temporal-memory`  
Audited revision: `588d075` plus the seven local commits already present on `main`

## Executive finding

StateKV is not currently a successful deployable eviction method. The repository closes the original state-conditioned physical-risk controller as a negative result under deployment-faithful gates. The reusable assets are nevertheless unusually well matched to the adaptive-temporal question: exact-KL closed-loop evaluation, strict pure-eviction and recoverable runners, full-pool QK routing, attention-memory baselines, and a 25,777,152-row attention trajectory from 10 Qwen3-8B samples.

The new branch must therefore be evaluated as a fresh hypothesis, not presented as an incremental improvement over a validated StateKV policy. Its direct conceptual connection is narrower and testable: StateKV establishes that action consequences depend on the current compressed state; this branch asks whether the reliability of historical attention also changes with state.

## Repository structure

| Path | Actual role |
| --- | --- |
| `statekv/core/` | Backend-independent state, retained-set action, KL-risk, and decision primitives. |
| `statekv/` | Model-aware MLX/HF runners, policies, probes, oracle comparisons, and analysis code. |
| `benchmarks/mlx/` | Main Apple-silicon benchmark harness and the 79-entry eviction-method registry. |
| `benchmarks/torch/` | Hugging Face benchmark harness for LongBench, RULER, and SCBench protocols. |
| `experiments/` | Frozen P0–P3PR mechanistic experiments and their manifests/results. |
| `configs/` | Frozen, discovery, and staged experiment protocols. |
| `results/temporal_cache_discovery/` | Model-run artifacts for the later StateKV program. |
| `analysis/` | Derived tables, figures, mechanism analysis, and literature records. |
| `docs/` | Current research status, complete history, negative-result closure, and reproducibility map. |
| `tests/` | Core contracts, physical cache invariants, gates, and golden-result checks. |

The authoritative reading order is `docs/RESEARCH_HISTORY.md`, `docs/FINDINGS.md`, `docs/FAILURE_ANALYSIS.md`, and `docs/EXPERIMENT_REGISTRY.md`. The root README correctly labels the deployable StateKV line as closed-negative.

## Models

Model-scale StateKV runs use MLX on Apple silicon with locally cached 4-bit checkpoints:

| Model | Observed use |
| --- | --- |
| `mlx-community/Qwen3-8B-4bit` | Principal quality-valid substrate; 36 layers, 32 query heads, 8 KV heads, 40,960-token maximum context. |
| `mlx-community/Qwen2.5-1.5B-Instruct-4bit` | Earlier/shared-mask and lower-coverage substrate; some tight-budget settings are quality-invalid. |
| `mlx-community/Qwen2.5-7B-Instruct-4bit` | External-validity run at longer context. |

The standalone benchmark configs additionally list Qwen2.5-0.5B/1.5B/7B, Llama-3.1-8B, Gemma-2-9B, Mistral-7B, and Yi-1.5-6B variants. These are infrastructure options, not evidence that every model has a completed StateKV experiment.

## Benchmarks and tasks

| Benchmark/task | Status in this repository |
| --- | --- |
| Synthetic single-needle NIAH / RULER-style retrieval | Principal retrieval-sensitive StateKV task; completed runs exist. |
| GovReport | Principal summarization task; completed StateKV runs exist and report ROUGE. |
| QMSum fallback | Loader fallback when GovReport is unavailable; provenance warnings exist. It must not be silently treated as GovReport. |
| LongBench | Implemented in both benchmark harnesses and used in pre-StateKV baseline work; not the principal closed-gate StateKV substrate. |
| RULER | Implemented in both harnesses; completed benchmark artifacts exist. |
| SCBench | Implemented in the torch harness for query-visible and reuse protocols. |
| Multihop / reasoning | MLX benchmark configurations exist; selected later stress runs exist. |

The adaptive-temporal development substrate should remain Qwen3-8B with the matched 5 NIAH + 5 GovReport split, because it already has a full-pool trace and matched strict-pure-eviction results.

## Existing baselines

The MLX registry distinguishes implementation fidelity. Names must follow that registry rather than being inferred from filenames.

| Method | Repository implementation/status |
| --- | --- |
| Full Cache | No-eviction control. |
| Random | Seeded control. |
| Recency / Sink+Recent / StreamingLLM | Implemented. |
| Attention / AccumulatedAttention | Cumulative causal attention implementation. |
| H2O | Core reimplementation using accumulated causal attention and heavy-hitter/recent split; not the original CUDA code. |
| SnapKV | MLX/Qwen core reimplementation with observation-window attention and pooling; not a line-for-line official port. |
| Windowed attention | Implemented with a rolling fixed window. |
| Attention decay | Existing fixed exponential decay implementation, but its update is `score <- gamma*score + attention`, not normalized EMA. Ranking is equivalent to normalized EMA only when token ages and update availability are matched. |
| qk_pool | Exact per-query full-pool QK routing with recoverable backing; strongest validated working-set policy, not pure eviction. |
| StateKV exact teacher | Candidate-specific physical evaluator/oracle; expensive and non-deployable. |
| Value/norm/leverage/hybrid families | Extensive registry and negative-result screens; see `docs/FINDINGS.md`. |

The adaptive branch will name the new normalized fixed-decay control `FixedEMA` and keep it distinct from H2O and from the existing unnormalized `attention_decay` benchmark alias.

## Cache settings and physical semantics

The principal matched Qwen3-8B development setting is:

- total cache budget 256 tokens per layer;
- sink protection 4;
- recent protection 32 (the runner temporarily uses `recent_size + 1` while advancing a query);
- core budget 220;
- 36 layers with a per-layer shared token mask across the 8 KV heads;
- 64 closed-loop decode cycles;
- greedy decoding;
- Qwen3-8B 4-bit weights; KV tensors are not described as quantized in the pure-eviction run;
- strict pure eviction enforces `S_t subseteq S_{t-1} union {x_t}` per layer and has no persistent CPU backing;
- recoverable runs keep a CPU `KVBackingStore`, may score the full pool, and can restore cold tokens.

The repository also contains budgets 64, 128, 192, 256, 352 and longer-context settings. They are not interchangeable: the findings show a sharp coverage-by-refresh-cadence interaction, and several lower-budget settings are task-quality invalid.

## What StateKV actually scores

There are two distinct objects that older prose can blur.

### Mechanistic state-conditioned risk

`statekv/core/actions.py` defines the current functional compression state as the boundary displacement between the history-compressed and reference trajectories. For a retained set, it computes the exact attention-output change at a fixed operating point after deletion and renormalization.

`statekv/core/risk.py` then evaluates a local second-order KL-risk increment at the current compressed state. If `p_ref` and `p_state` are the reference and current-state output distributions and `delta_logits` is the propagated retained-set response, the executed formula is:

```text
(p_state - p_ref)^T delta_logits
+ 0.5 * delta_logits^T F(p_state) delta_logits
```

where `F(p_state)` is the categorical Fisher operator. Candidate actions are selected by minimum finite risk with deterministic identifier tie-breaking.

### Deployment-gate teacher score

The strict Gate-0 teacher in `statekv/statekv_gate_runner.py` does not merely use an attention-importance heuristic. At each live pure-eviction state, it builds a fixed panel of legal candidate retained sets, forwards a clone for each candidate, computes same-input exact `KL(full || candidate)`, and commits the minimum-KL panel action. The state-conditioned evaluator is therefore an action-level physical-risk teacher, not a tokenwise additive importance score.

The best deployable attention-family policies use latest attention, cumulative attention, a fixed observation window (SnapKV), or recent attention volatility. No validated deployable StateKV token score survives the closure.

## Existing temporal evidence to reuse

`results/temporal_cache_discovery/statekv_qkv_decomposition_qwen3_8b_v1/token_rows.parquet` contains 25,777,152 rows over 10 samples, 64 cycles, and 36 layers. Each row records token attention, QK/V perturbation features, rank, core membership, token class, and task. It was collected on the recoverable qk_pool trajectory at budget 256/core 220. A smaller head table contains 774,144 per-head rows on three development samples.

`statekv/qkv_decomposition.py::add_future_targets` already defines post-hoc mean future attention and revival labels. Existing analyses probe horizons up to 8, but they do not compare tuned EMA horizons, estimate per-token/per-head best horizons, or implement adaptive forgetting. `a2_temporal_volatility` scores high recent standard deviation; it is not an adaptive temporal-memory estimator and performed worse than attention in its closed-loop panel.

## Teacher/oracle fairness audit

| Method/run | Cache budget | Candidate universe | Recent/sink | Recoverability | Future/full-history access | Layer budget | Eviction constraint | Closed-loop |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- |
| P35 attention / b2 / a2 / uniform / SnapKV | 256 = 4 + 32 + 220 | Surviving live cache | matched | none | none | 220/layer unless explicitly dynamic | strict irreversible inclusion | yes |
| Gate-0 strict teacher | 256 = 4 + 32 + 220 | Fixed panel built only from surviving live cache | matched | none | full logits for current query only | matched to panel | strict irreversible inclusion | yes |
| Gate-1 panel oracle regret | same | Same legal panel on teacher roll-in | matched | none | current-query exact KL | matched | strict irreversible inclusion | counterfactual one-step rows on a closed-loop roll-in |
| P31 `statekv_exact_mean` | 256 = 4 + 32 + 220 | Full historical pool | matched nominally | CPU backing store | deleted tokens accessible at every cycle | matched nominally | recoverable; not pure eviction | yes |
| R0 qk_pool / statekv recoverable | 256 = 4 + 32 + 220 | Full backing pool | matched | CPU backing store | current-query full-pool QK scoring | matched | recoverable refresh | yes |
| QKV decomposition trace | 256 = 4 + 32 + 220 | Full backing pool under qk_pool | matched | CPU backing store | full-pool current attention; future labels added only offline | 220/layer | recoverable refresh | yes |

The historical P31 claim of teacher headroom is apples-to-oranges relative to strict pure eviction because its backing store allows re-anchoring and restoration. The repository now explicitly records this as an artifact. Gate 0 is the valid matched comparison and finds no teacher headroom: mean trajectory KL 0.2322 versus 0.0961 for the best cheap policy, with the teacher winning 2/10 samples.

The new future-utility oracle must therefore be reported in two forms if both are run:

1. offline `NON_CAUSAL_ORACLE`, for score-quality/headroom diagnostics on the full-pool trace;
2. strict live-candidate oracle, where the scored candidates are limited to the surviving pure-eviction cache and deleted tokens remain unrecoverable.

Results from the first cannot be compared directly with pure-eviction task quality as if physical constraints matched.

## Current scientific status relevant to this branch

- State dependence of physical retained-set consequences is supported in controlled settings.
- The original deployable physical-risk controller is a negative result.
- Strict pure eviction exhibits a plateau-then-cliff structure: useful future-query tokens can become visible only after they are irreversibly gone.
- qk_pool is the strongest validated recoverable policy.
- Recent attention volatility alone is not a successful eviction score.
- The adaptive-horizon hypothesis remains untested: existing evidence neither establishes heterogeneous optimal temporal horizons nor shows that adapting them beats a tuned fixed EMA.

These facts make the branch scientifically meaningful but set a high stop condition. Better future-attention prediction without better task performance would support an estimator result while reinforcing irreversibility as the downstream bottleneck.

## Reproduction checks completed

- `scripts/smoke_test.py`: passed on 2026-08-19.
- Existing QKV trajectory files load with the expected schemas and row counts.
- A derived-table reproduction is tracked separately under `results/adaptive_temporal/reproduction/`.

