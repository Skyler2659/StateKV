# StateKV research history

Reconstructed 2026-08-10 from git history (81 commits), the claims registry
(`configs/ccfa.yaml`), the frozen experiment registry
(`experiments/frozen_registry.yaml`), the closure documents in `analysis/`,
and the verified rejection catalog
(`analysis/statekv_gate_retrospective_catalog.md`).

This document deliberately separates **what was believed at the time** from
**what is known now**. Where the two differ, both are stated.

---

## Era 1 — pre-StateKV eviction baselines (2026-05-09 → 2026-05-13, 49 commits)

The repository began as "L1-Robust KV Cache": a flat scratch repo (`cmpr.py`,
a vendored `streaming-llm-main/`) exploring L1-norm "leverage" scores for KV
eviction on HF models. Most of the era was HF-compatibility plumbing
(`pos_shift` polyfills for DynamicCache 4.50/4.51, GPT-NeoX `layer_idx`
counters, FP16 overflow fixes) and benchmark accretion (PG19, wiki40b, NIAH,
NarrativeQA, HotpotQA).

Two algorithm variants were tried and abandoned within days: K-V joint
scoring (`029cbb5`, reverted) and attention-weighted L1 (ended as pure L1
leverage, `731d655`). An H2O baseline was added and debugged (8 commits,
05-13). HotpotQA multi-hop results on TinyLlama-1.1B were the era's
empirical high point (`c056837`).

*Believed then:* L1 leverage scores are a viable eviction signal.
*Known now:* this line was never validated against strong baselines and was
abandoned, not falsified. Its only survivors are plumbing (pos_shift, H2O)
inside `benchmarks/mlx/`.

## Era 2 — runnable research framework (2026-06-08 → 2026-06-17, 3 commits)

`9b058d3` introduced the root `src/` package (eviction registry, runners,
configs); `b53143a` added the MLX 4-bit benchmark suite; `1d64abc` curated
per-method eviction configs and results. This is the direct ancestor of
`benchmarks/mlx/` — 79 registered eviction methods, most never used by any
StateKV experiment.

## Era 3 — StateKV big-bang refactor (2026-08-07, 4 commits)

`3b29f05` moved the framework into `benchmarks/` and created `statekv/`,
`experiments/`, `analysis/` (2496 files, +711k lines). `4118811` deleted the
provenance layer (`statekv/io/`, `tests/provenance/`) and added
`configs/ccfa.yaml` + the architecture figure. The StateKV hypothesis was
formalized: *repeated compression is state-dependent; evaluate each
candidate retained set as a physical intervention at the current compressed
state, propagate to output risk, and use one risk object for selection and
refresh.*

## Era 4 — the StateKV experiment program (2026-08-07 → 2026-08-10)

### Phase 0 — mechanism validation (frozen, `experiments/`)

P0 v2: the exact set-level deletion identity reaches FP64 L2 error
2.26e-11 — **still valid**. P1: evaluation at the observed (compressed)
state, cosine 0.99974 — **still valid**. P2 recovery: finite-action trust
region mapped (R1 amplitude study); the two-midpoint state-local scalar risk
evaluator hit Spearman 1.0 / top-1 gain 1.0 in evaluation and replication
(R4) — **still valid as a diagnostic**. P3PR: dense all-layer mechanistic
risk transfers across two model families and task families — **valid,
limited scope**. Negative frozen results: predictive_closure,
p2_state_local_risk (natural-amplitude full-vector reconstruction),
p3_decision_validity — documented in `experiments/frozen_registry.yaml`.

*Believed then:* a state-conditioned physical risk evaluator exists and can
rank candidate actions nearly perfectly on frozen pools.
*Known now:* true — but only as an *expensive, teacher-forced* evaluator.
Every later failure is about making it cheap or deployable, not about the
mechanism.

### Phase 1 — cheap estimators (P0–P5, training-free line)

Fixed-decay Euclidean history sketches (TF-P0), diagonal-block metric repair
(TF-P1), shared randomized Fisher pullback (P2), output-side VJP routes
(P3), multi-boundary VJP (P5) — all negative or pilot-negative. P4 (direct
coreset + merge/quant tiers) gave the only positive local diagnostics
(merge beats hard deletion in 192/192 units; 2/3/4-bit cold-value tiers at
23–34% storage).

The P3 post-hoc Rademacher VJP variant looked promising on development data
(normalized regret 0.1945→0.0696). **Believed then:** worth replication.
**Known now (retest Track D, 2026-08-10):** does not replicate on 8 fresh
sequences — all gains negative; the dev gain was a selection artifact.

### Phase 2 — direct policies and proxy controllers (P6–P24, Era 1 substrate)

Teacher-forced replay (P6) showed a four-query contribution selector lowers
mean exact KL 0.0485→0.0199. Then a long dev-positive/independent-negative
pattern: P7 (NIAH mean +0.0007 veto), P9 shrinkage (win rate 47% vs locked
55%), P13 (CVaR95 +0.9% veto while mean/P95 improved), P14 protected rescue
(dev-only tail veto), P16 temporal volatility (passed the independent tail
gate) → P18 (frozen freegen NLL +0.00374 veto), P20 token rarity → P21
(retrieval replicates 3/3, GovReport regresses). P22 latest-attention proxy
(action alignment strong) → P23b (refresh ordering reverses sign on new
data). P24 output-aware proxy: no improvement anywhere.

*Believed then:* each rejection was a hard gate failure closing the line.
*Known now (retest Tracks A/B, 2026-08-10):* the vetoes were a mixed bag.
The contribution family does lead the Era-1 replay table on 24 fresh
sequences (0.394 vs attention 0.408, 54–58% wins) — the P9 55%-win-rate
gate was miscalibrated. Token rarity's retrieval specificity replicates
cross-model (NIAH 10/10 on Qwen3-8B) but its GovReport gap is real.
Temporal volatility is *not* competitive on Era-2 (NIAH 0.8). See
`analysis/statekv_retest_report.md`.

### Phase 3 — physical oracle and cheap controllers (P25–P35, Era 2 substrate)

P25/P26 built the physical closed loop (recoverable CPU backing store, cold
recovery). P28: the exact-risk teacher beats attention/SnapKV/H2O on
teacher-forced KL. P29: only per-token control (H=1) survives free
generation. P30: KL gate passed, task gate failed (small model). P31
(Qwen3-8B): teacher mean KL 0.0506 vs attention 0.336, NIAH 5/5.
P32: cheap zero-rollout controllers — A2 temporal volatility KL 0.095, B3
dynamic-budget 0.115 with best task point estimate. P34: B3's dynamic
mechanism refuted (loses to shuffled static control) — the working part is
a static layer-budget prior. P35: strict pure-eviction mechanics pass.

### Phase 4 — the closures (2026-08-09)

The decisive question: does the expensive state-conditioned teacher have
deployable headroom over cheap selectors under strict pure eviction?

- **Gate 0/1:** no. Teacher KL 0.232 vs best cheap 0.096, paired 2/10; 61.6%
  of cycles have numerically tied one-step risks; the fixed action space is
  degenerate (oracle regret 1.7%). The P31 "headroom" was a machinery
  artifact — candidates were evaluated with access to already-deleted
  tokens via the persistent backing store.
- **Ladder 2B/2C:** risk is plateau + cliff — flat at one step for every
  good action; the deep signal appears 2–4 steps before the event, shared
  by all panel actions. Nothing to distill. (A probe-metric bug in the
  ladder was found, fixed, and documented; corrected rows used.)
- **Recoverable R0:** under recoverable semantics, qk_pool (exact per-query
  full-pool QK routing) KL 0.0086 vs teacher 0.0213, paired 0/10.
- **QK–V battery:** no V-side residual given QK (swap-oracle regret ~1e-15,
  92% flat); qk_tiered_v gate NO_GO only on the G5 unequal-memory
  comparison (retest: at matched budget it is within 6% KL of qk_pool with
  identical task scores).
- **Selective refresh (R1/R2a/R2):** rankings time-invariant on Qwen3-8B
  per-layer (coverage 0.995+); the premise (staleness) absent. The P23b
  substrate (shared mask, coverage 0.70) is genuinely stale but
  quality-invalid at tested budgets.

### Phase 5 — open search and external validity (2026-08-09 → 2026-08-10)

The open search (HF1–HF6) tested every assumption of qk_pool: coverage
stress (tracks full cache down to 8% coverage, NIAH 1.0), approximation
frontier (page metadata cannot recover token-level exactness), cadence ×
coverage (the one cliff: at 64-token budget, h4 KL 0.38 / NIAH 0.2, h16 KL
0.84 / NIAH 0.0 — the fix is cadence, not scoring), per-head selection
(+0.96pp, below action threshold), conditional budgeting (no predictable
trigger). Verdict: CLOSE, no method candidate survives.

External validity then challenged all closures at 3–4.7K context, lower
coverage, more tasks, and a second model family: **FINAL CLOSE** — qk_pool
stays task-perfect down to 1.4% coverage, swap-oracle regret flat at 3072,
the cadence cliff reproduces and its controlling variable is the absolute
core budget, no staleness precursor exists, and the pattern replicates on
Qwen2.5-7B.

### Phase 6 — gate retest (2026-08-10)

A no-hard-gate re-evaluation of the marginally rejected policies (fresh
sample offsets, continuous reporting): Era-1 contribution family confirmed
competitive; qk_tiered_v confirmed at matched budget; token_rarity boundary
confirmed; temporal volatility disconfirmed on Era-2; Rademacher VJP
disconfirmed. See `analysis/statekv_retest_report.md`.

---

## What actually happened, in one paragraph

The mechanism (state-conditioned physical risk) is real and precisely
measured. What failed is the *deployability thesis*: at every
quality-valid operating point tested, one-step risk is flat across all
reasonable actions (so no cheap or expensive selector can improve on
per-step exact QK routing), and the only regime where selection matters —
tight coverage × slow cadence — is one where no refresh-time observable
helps. The project's positive residue is a validated evaluation stack, a
map of the coverage/cadence frontier, a strong oracle baseline (qk_pool),
and a systematically falsified search space.
