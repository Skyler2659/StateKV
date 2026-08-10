# StateKV QK–V Mechanism Discovery — protocol

Status: preregistered discovery protocol (Phase 0), 2026-08-09
Mode: mechanism discovery → candidate hypothesis → cheap falsification →
controlled validation → simple method → paired evaluation → GO/NO-GO.
This document is written before any discovery experiment is run.

## 1. What is actually proven (and where)

| fact | scope | evidence |
|---|---|---|
| 1-step state-conditioned physical-risk teacher has no headroom under strict pure eviction | Qwen3-8B 768/256 + Qwen2.5-1.5B P23b | `statekv_teacher_closure_2026-08-09.md`, Gate 0 tables |
| The 1-step risk landscape is a plateau (62% cycles tie; swap marginals ~1e-4; pair interactions 0); delayed risk is cliff-shaped | same substrate | Gate 1/2 tables |
| P31's teacher headroom was recoverability (backing store) + quasi-irreversible baselines, not scorer value | R0 substrate | `statekv_recoverable_r0_results.md` |
| Under unified recoverable semantics, exact full-pool query-aware top-k (`qk_pool`) reaches KL 0.0086, beats the recoverable teacher 10/10 paired with 2.5x better tail | R0 substrate (Qwen3-8B, ~0.9-1.2K ctx, budget 256, 64 tokens) | R0 main/paired tables |
| GovReport quality of teacher/qk/quest is equivalent at 64-token ROUGE resolution; "teacher > full_cache" is noise | R0 substrate | R0 results §6.1 |
| attention×‖v−o‖ ("contribution") as a *selector* passes local/dev screens but fails independent all-task, tail, and refresh gates (P7/P9/P13/P14/P24) | P4-P24 substrates | frozen registry summaries |
| value merging / low-bit cold-value tiers have positive *local* diagnostics | local replication | README storage rows |
| attention-free geometry selectors (KNorm/KeyDiff/VNormL2) fail needle tasks | P19 substrate | P19 summary |

Substrate limits: everything above is Qwen3-8B/Qwen2.5-1.5B, ≤1.2K-8K
contexts, 64 generated tokens, NIAH+GovReport.  Nothing is proven for
32K+ contexts, long generations, or other model families.

## 2. qk_pool — exact definition (the baseline to beat)

Per cycle, per layer: one full-pool scoring forward of the current token
against a cache rebuilt from the complete `KVBackingStore`; per-position
score = head-mean softmax attention of that single current query
(`_full_pool_scores`); select top-220 eligible positions (sink 4 + recent
32 mandatory) — deterministic tie-break by position; commit via
re-anchor from backing; refresh every cycle.  No V, no history, no learned
weights, one extra forward per cycle.

## 3. Why the old teacher lost (mechanism to explain, not to relitigate)

1-step exact-KL panel scores sit on a plateau: the teacher picked the
winning action class (qk_pool, which is in its panel) in only 23% of cycles
and the pick was counter-cyclical noise.  The valuable action — "retain
what the current query attends to over the full pool" — is directly
computable; physical-risk scoring adds noise, not accuracy (Gate 0/1, R0 §5).

## 4. The open question (this program)

> Why is QK routing already so strong, and does V / downstream physical
> information carry residual predictive value that QK cannot capture —
> anywhere: specific layers, heads, token types, cutoff regions, horizons,
> or failure regimes?

Primary form of the question (brief §"Suggested first experiment"):

> I(target; V | QK) — after conditioning on QK, how much residual variance /
> ranking error does V explain?

If ≈ 0 everywhere we measure: close the V-ranking direction with evidence.
If > 0 in an identifiable, runtime-detectable regime: build the simplest
conditional method and gate it against qk_pool.

## 5. What may NOT be repackaged as a new claim

- The P31 teacher numbers (recoverability artifact as a *method* claim).
- Raw contribution (α·‖v−o‖) as a global selector — gated negative 5 times.
- value_norm/key_norm panel candidates — no standalone win anywhere.
- "StateKV beats SnapKV/H2O" style irrecoverable comparisons.
- Any 1-step physical-risk scorer as a retrieval ranker (closed twice).

## 6. Reusable machinery

- Recoverable loop + `_full_pool_scores` (exact per-head attention over the
  full backing pool, one forward/cycle) — the dataset chassis.
- `KVBackingStore` + `state_from_anchor` (legal re-entry, rebuilds).
- `contribution_token_score` pattern (α·‖v−o‖, GQA-aware reshapes).
- o_proj access: `runner.model.model.layers[l].self_attn.o_proj`
  (backend_mlx.py:1040 pattern) → projected-V norms ‖v W_O‖ and exact
  single-token removal+renorm perturbation
  Δ_i = a_i/(1−a_i)·‖(v_i − o) W_O‖, zero extra forwards.
- `_swap_selection` pattern (budget-preserving per-layer swaps) → exact
  1-step KL swap oracle for near-cutoff pairs.
- `exact_distribution_metrics`, `_advance_full_state` (same-input KL).
- Future relevance for free: future cycles' scoring forwards give future
  attention/rank/revival of every historical token.
- R2 evidence/monitor machinery for needle-position metadata.

## 7. Phase 1 dataset: QK–V Decomposition Dataset

Run the qk_pool recoverable trajectory on the R0 10 samples (64 cycles),
instrumented.  Three tables:

- **T (token×layer×cycle, head-mean)**: position, attention a_i, rank,
  in-core flag, margin to cutoff, ‖v_i‖, ‖v_i W_O‖, Δ_i (exact
  attention-output removal perturbation), a_i·‖v_i W_O‖.  (~25M rows)
- **H (head-level subset)**: dev samples, every 4th cycle, cutoff window
  ±16 + top-8 + random-16: per query-head a_i, Δ_i, ‖v_i W_O‖.  (~1.4M rows)
- **S (swap oracle, exact downstream target)**: dev samples, every 4th
  cycle, cutoff pairs (in/out): exact 1-step same-input KL of the
  budget-preserving swapped action vs the committed action → sign/magnitude
  of QK ranking error per pair, plus both tokens' full feature vectors.

Targets derived post-hoc: future attention mass/rank at h ∈ {1,2,4,8},
revival (below cutoff now, top-k within h), needle-token labels, token-type
metadata (digit/punct/case/rare/needle/structural).

## 8. Analysis battery (Phase 2 questions A–F)

A. **Dynamic range**: Var(log a) vs Var(log ‖v‖), Var(log ‖v W_O‖),
   Var(log Δ) — per layer, head, task, sample, phase.
B. **Ranking overlap**: Top-K(a) vs Top-K(a·‖u‖) vs Top-K(Δ) vs
   Top-K(future relevance): Jaccard, recall of high-future-relevance
   tokens, Spearman; boundary disagreement census.
C. **Near-tie hypothesis** (highest priority): bucket by margin to
   cutoff; per bucket — (i) partial Spearman of V-features vs future
   relevance given a; (ii) swap oracle: does Δ/‖u‖ predict the *sign and
   size* of exact swap regret after the attention margin is accounted for?
D. **Head/layer specialization**: per-layer and per-head versions of C;
   GQA KV-group differences; identify any layer/head where
   I(target; V | QK) is materially positive.
E. **Token-type specialization**: residual stats by token class.
F. **Horizon**: which cheap current features predict future
   relevance/revival at h = 1/2/4/8 (current a, window mean/std,
   persistence, ‖u‖, a·‖u‖, Δ, recurrence).

Signal probes: linear/logistic models and a tiny ranker (sklearn) are
allowed as *probes*; any positive must be distilled to an analytic rule or
a small gate before becoming a method candidate.

Controls before calling anything "signal": not a monotone transform of
attention; not driven solely by attention magnitude; not a handful of
outliers; no future-oracle leakage; not a metric artifact (same-input
checks throughout).

## 9. Method families (priority order)

A. **QK-first, V-on-demand**: keep QK ranking; rerank only the cutoff
   window (M tokens) with a V-aware score.  Parameters frozen on a dev
   subset before touching test samples.
B. **QK + V residual score** (additive/multiplicative, λ frozen on dev,
   one shot, no tuning loop).
C. **K/V asymmetric management**: QK routes; V decides precision/merge/
   tier (only if the mechanism data forces this reading — V useless for
   *whom to fetch*, useful for *how to keep*).
D. **Future-revival-aware routing** (cheap revival predictor added to QK).
E. **Head/layer-conditional hybrid** (special scorer only where D shows
   residual value).

## 10. Stop / pivot rules

Stop a hypothesis when: QK-conditioned residual ≈ 0; gain only in <1%
unstable cases; paired CI crosses 0 with small effect; overhead dominates;
needs oracle future info; method reproduces qk_pool; a simpler baseline
matches it.  Record negative evidence and move on.

Program-level closure: if A–F show I(target; V | QK) ≈ 0 across layers,
heads, token types, cutoff buckets, and horizons on the dev samples, run
one confirmatory pass on held-out samples; if confirmed, write the
evidence-backed closure (QK absorbs the usable signal on this substrate).

## 11. Method gate (when a candidate exists)

Separate preregistered gate doc per candidate answering the brief's 13
method questions; paired evaluation vs full_cache, qk_pool, quest_like,
old teacher, on the R0 substrate (and a budget variant if overhead claims
are made).  Success = stable paired gain vs qk_pool in an identifiable
regime, tail non-worse, task quality non-worse, overhead path, mechanism
consistent, not a tiny numerical perturbation of qk_pool.

## 12. Artifacts (planned)

- Dataset runs: `results/temporal_cache_discovery/statekv_qkv_decomposition_*`
- Analysis: `analysis/tables/qkv_*.csv/md` (+ figures if informative)
- Results: `analysis/statekv_qkv_discovery_results.md`
- If method: `analysis/statekv_<method>_gate.md`, minimal implementation,
  tests, `configs/stages/statekv_<method>_gate.yaml`
- ccfa.yaml sync at the end.
