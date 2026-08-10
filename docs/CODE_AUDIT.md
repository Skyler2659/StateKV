# StateKV code audit

Compiled 2026-08-10 from a full read of `statekv/` (68 modules), `scripts/`
(58), `configs/`, `benchmarks/`, `tests/`. Method: import-graph analysis +
per-module reading; statuses cross-checked against the claims registry.

Scientific-correctness principle applied throughout: **fail loudly, never
quietly recover** — but no fix may change the semantics of stored
experimental results.

---

## 1. Module classification (`statekv/`, 68 modules)

**CORE** (surviving machinery — recoverable freegen evaluator, replay,
retest): `config.py`, `storage.py`, `artifacts.py`, `tasks.py`,
`selectors.py`, `backend_mlx.py`, `runner.py`, `functional_probe.py`
(base chain), `trajectory_model.py` (only `exact_distribution_metrics` +
base class), `robust_envelope.py`, `output_sensitivity.py`,
`independent_fisher.py`, `gauge_geometry.py` (only `fisher_variance`),
`candidate_pullback.py`, `oracle_closed_loop.py` (only `KVBackingStore`,
`_rollout_candidate`, `_top_core`, `_stale_core`),
`oracle_policy_comparison.py` (only `AttentionPolicyMemory`,
`_physical_candidate_panel`, `_selection_from_scores`, `token_rarity_scores`),
`oracle_policy_freegen.py`, `cheap_policy.py`, `value_tier.py`,
`retest_freegen.py`, `direct_policy_runtime.py` (only
`contribution_token_score`), `metrics.py` (legacy runner only), plus
`core/{actions,risk,decision}.py` (stable contracts — verified: no
back-imports, stdlib+torch only).

**LEGACY** (closed phases, kept for reproduction): `statistics.py`,
`plotting.py`, `mechanism.py`, `functional_features.py`,
`trajectory_analysis.py`, `theory_closing.py` (2138 lines, import-alive only
via an `atomic_frame` alias), `robust_envelope_{analysis,policy}.py`,
`output_sensitivity_{analysis,policy,freegen}.py` (helpers still CORE),
`gauge_geometry_analysis.py`, `independent_fisher_analysis.py`,
`candidate_pullback_analysis.py`, `fisher_pullback.py`, `shared_jvp{,_pilot}.py`,
`training_free_analysis.py`, `training_free_routes.py`,
`metric_repair_analysis.py`, `vjp_routes_pilot.py`,
`multiboundary_vjp_pilot.py`, `direct_coreset_pilot.py`,
`direct_policy_{signals,replay,runtime_profile,trigger,analysis}.py`,
`proxy_alignment.py`, `oracle_closed_loop_analysis.py`,
`oracle_policy_comparison_analysis.py`, `cheap_policy_freegen.py`,
`budget_dynamics.py`, `statekv_gate_runner.py` (3051 lines, largest module),
`statekv_gate_analysis.py`, `refresh_trigger.py`, `qkv_decomposition.py`,
`backend.py` (torch/HF path — Era-1 only; all Era-2 runs are MLX).

**DEAD**: `training_free.py` (372 lines; sole importer is its own test).
Kept as the artifact of the TF-P0/P1 negative line; flagged here rather than
deleted (the line's analysis modules remain LEGACY).

## 2. Over-defensive code — the scientific-validity risks

Top instances (file:line, behavior, risk):

1. **`tasks.py:300-319`** — any LongBench loading exception silently
   substitutes synthetic gov_report samples. Dataset identity changes
   mid-run. → **fixed in this audit** (loud warning + metadata flag).
2. **`artifacts.py:57-63`** — `_git` swallows all exceptions →
   `git_commit: null` in run metadata; silent provenance loss. → **fixed**
   (warning to stderr).
3. `oracle_policy_freegen.py:411,429` — `longbench_score(...) or 0.0`:
   unscorable generation becomes a hard 0.0 task score feeding the primary
   endpoint. Documented known issue; changing it would drift metric
   semantics against stored runs — **not fixed**, flagged.
4. `independent_fisher_analysis.py:24-29` — `_finite(value, fallback=0.0)`
   zeroes unparseable gate inputs inside frozen-gate statistics. Legacy
   only; **not fixed** (frozen evidence reproducibility), flagged.
5. `oracle_policy_comparison.py:51-57` — `_score_on_universe` fills missing
   scores with 0.0, silently reshaping selections. **Not fixed** (live
   semantics), flagged.
6. `backend.py:298-302` — silent prompt middle-truncation in the torch
   path; only the freegen line hard-fails (`_check_prompt_truncation`).
   Era-1 runs were unguarded. Legacy; flagged.
7. `trajectory_model.py:85-96` — the project's primary exact-KL metric runs
   in float32 with a `torch.equal` zero-shortcut; the same comparison is
   float64 in `statekv_gate_runner.py:88-91`. Inconsistent numerics for the
   same quantity; flagged (stored results used float32).
8. `functional_probe.py:166` — `protected_recent=0` silently becomes 1,
   changing cache semantics for 12+ callers. Flagged.
9. `config.py:84` + `runner.py:137-216` — `fail_on_error=False` default:
   failed combos recorded and excluded from aggregates without loud signal.
   Flagged; Era-2 runners set it explicitly.
10. `output_sensitivity.py:131-136`, `independent_fisher.py:520-524`,
    `gauge_geometry.py:615-619` — corrupt fragments treated as "incomplete"
    and silently re-collected (masks corruption as resume). Flagged.

## 3. Duplicate implementations

Confirmed semantic duplicates → **consolidated in this audit**:
- `trajectory_analysis.py:41` `atomic_json` reimplements
  `storage.atomic_json` (~10 modules import the wrong one).
- `cheap_policy.py:38` `_normalize_on_eligible` == `oracle_closed_loop.py:240`.
- `functional_features.py:16` `_relative_ridge` == `selectors.py:77`.

Same-name-different-semantics (must NOT be merged):
- `metrics.approximate_kl` (top-128 approximation, Era-1) vs
  `core/risk.reference_kl` (exact torch) vs
  `trajectory_model.exact_distribution_metrics` (exact, float32).
- `_metric_row` in `oracle_policy_freegen.py:374` vs
  `output_sensitivity_freegen.py:244` (different task buckets).
- Bootstrap intervals ×3 (`oracle_policy_freegen.py:856`,
  `trajectory_analysis.py:61`, `statistics.py:15`) — same algorithm,
  different call sites; left as-is (no shared import worth the churn).
- Keep-set computation (sink+core+recent with budget guard) exists 4×
  (`backend.py:646-676`, `backend_mlx.py:801-853`, `:901-946`, `:966-995`) —
  pure-eviction vs recoverable vs tiered variants; NOT merged (different
  semantics), but flagged: a budget-rule change must touch all copies.

## 4. Configuration audit

Three config systems coexist: (1) dataclass `load_discovery_config` schema,
(2) frozen phase YAMLs, (3) flat base+override YAMLs read via
`config.get(key, <default>)`. System 3's silent defaults are where
experiment-relevant constants hide. Worst instances (full list in the audit
trail): `config.py:34-37` budget defaults (256/4/32/220), five dataclasses
with diagnostic-layers `[0,7,14,15,21,27]` defaults,
`robust_envelope.py:554` uses a *different* hardcoded list `[0,7,14,21,27]`,
`oracle_policy_freegen.py:668-676` value-tier fallbacks, retest/cheap
policy constants duplicated (`cascade_margin=0.15`, `adaptive_budget_delta=44`).
The `statekv_openstress_3072_256.yaml` incident (silent prompt truncation
via `max_prompt_tokens` default) is the documented cost of this pattern;
the fix then was a hard-fail guard — the defaults remain.

**Rule adopted by this audit:** new configs must set every
experiment-relevant key explicitly; the silent-fallback list above is the
migration backlog, documented here rather than chased (changing defaults
risks silent divergence from stored runs).

## 5. Benchmarks and external surface

- `statekv/` depends on exactly 5 modules of `benchmarks/mlx/src`:
  `config`, `model_adapters`, `runners.mlx_runner`,
  `evaluation.official_metrics` (+ `snapkv_pool_scores_numpy`). The rest of
  the 79-method eviction registry is legacy (19 of 79 canonical names are
  referenced by any StateKV-era experiment; `attention_l1_compactor` /
  `attention_l2_compactor` by none at all). Kept: it is the baseline library
  for future work and P18–P21 raw data depends on it.
- `benchmarks/torch` (`kvbench`): only `types.py` (AttentionSignals), the
  `temporal/*` compat shims, and `backends/huggingface.py` (one test) are
  load-bearing. The rest is legacy CUDA harness, kept for Era-1 lineage.
- `pyproject.toml` does not declare `mlx` (Apple-silicon-only) — noted;
  installation is via the editable benchmark harnesses per README.

## 6. Tests

491 collected. Strong coverage of: deterministic selection/tie-break,
attention-delta renormalization identity, risk-metric identities, atomic
writes, protocol split-disjointness, prompt-truncation hard-fail, policy
memory. **Gaps closed in this audit** (`tests/test_cache_invariants.py`):
budget exactness, sink/recent survival, and determinism across the pure
selection primitives (`_top_core`, `deterministic_uniform_core`,
`recency_core`, `quest_like_core`, `token_rarity_scores`,
`mandatory_and_eligible`). Remaining gap: the five `tests/golden/`
fixtures were never exported (all skipped) — frozen reference values do not
exist; recorded as REQUIRES RERUN-style debt in REPRODUCIBILITY.md.

`tests/test_repository_architecture.py` encoded a "single markdown file"
policy that predates this audit's `docs/` tree — **deliberately rewritten**
in this audit to enforce the new documentation policy instead (canonical
docs set present, no root yaml, no `sys.path` mutation in `statekv/`).

## 7. Misc findings

- `LICENSE` names "MIT HAN Lab 2023" — inherited from the vendored
  streaming-llm era; flagging, not changing (attribution may be correct for
  the derived harness code).
- `scripts/audit_current_state_conditioned_physical_risk_theory.py` is dead
  (targets a `docs/statekv/theory.md` that no longer exists).
- `analysis/manifest.json` is stale (references three generated docs absent
  from the tree); `generate_report.py`'s full pipeline is partially dead.
- `tmp/trigger_prescreen.py` was real source code living in the gitignored
  scratch tree → rescued to `analysis/tables/` in this audit.
- No notebooks, no large commented-out code blocks, only 3 substantive
  TODO/HACK comments (one vendored upstream).
