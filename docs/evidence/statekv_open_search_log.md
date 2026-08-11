# StateKV Open Search Log — goal-driven research search

Status: active
Started: 2026-08-09 (session 2 of the project; after three closed lines)

## Boundary conditions inherited (do NOT re-test)

| closed line | evidence | status |
|---|---|---|
| 1-step physical-risk teacher, strict pure eviction | Gate 0: teacher KL 0.2322 vs cheap 0.0961; plateau 61.6% cycles | closed-negative |
| recoverable physical-risk teacher vs qk_pool | R0: 0.0213 vs 0.0086, paired 0/10, tails 2.2× worse | closed-negative |
| V residual routing value given QK (all buckets: near-tie, head/layer, token type, horizon, revival) | QK–V discovery battery A–F + 288 exact swap oracles | closed-negative |
| qk_tiered_v (4-bit cold V tier) | preregistered gate: TIERING_LOSSY (G5 1.126 > 1.10) | closed-negative |
| dynamic per-layer budget (per-cycle) | P34: loses to shuffled-static control 8/10 | closed-negative |
| selective/adaptive refresh triggers | R1/R2a/R2b: ranking time-invariant at quality-valid operating points (Qwen3-8B per-layer) | closed-negative |
| training-free direct signal policies (blend, volatility, rarity, geometry, output-aware proxy) | P8–P24, all gates failed | closed-negative |

Strongest current facts:
- `qk_pool` (exact full-pool current-query top-k, refresh every cycle) KL 0.0086, NIAH 1.0 — the reference to beat or to approximate.
- Coverage is the binding constraint: fp16-352 = 0.499× KL of 256, 10/10 paired.
- `quest_like` (16-token pages) 0.0243 = 2.8× qk_pool — an approximation gap exists.
- Old P31 gain = recoverability + quasi-irreversible baseline handicap (R0 ladder D1/D2/D3).

## Hypothesis portfolio (this round)

### HF1 — Regime stress: does qk_pool's near-oracle status survive coverage/length stress? [LEAD]
- Motivation: all closures sit on ONE substrate (Qwen3-8B, ~1.1K ctx, budget 256 ≈ 23% coverage, 64-token generation). "Coverage is binding" predicts tighter coverage → real decisions → ranking quality starts to matter. qk_pool's assumptions: current query ≈ future queries; token-level exact scoring; uniform per-layer budget.
- Cheap probe: offline attention-mass-captured vs K per layer/cycle from decomposition token_rows; per-cycle KL concentration from step_rows.
- Real runs (only if probe predicts discrimination): qk_pool at 768/{128, 64}; possibly longer-context substrate.
- Continue if: qk_pool regret grows AND has observable structure. Stop if: qk_pool stays ≲0.02 KL at 6–12% coverage → ranking genuinely doesn't matter, regime line closes.

### HF2 — Approximation frontier: can cheap candidate generation close the quest_like→qk_pool gap? [SECOND]
- Motivation: qk_pool 0.0086 vs quest_like 0.0243 (2.8×). qk_pool is oracle-grade (full-pool exact scan per cycle); its deployable approximations are untested beyond 16-token pages. Frontier question: recall@M → KL.
- Cheap probe: offline recall simulation (page-max of attn at page sizes 4/8/16, hybrid recent+pages) vs exact top-220 core.
- Novelty risk: Quest/RetrievalAttention territory — check before claiming anything.
- Lineage note: this is "hierarchical working-set management" (R0 residue question), NOT old-StateKV scorer revival. If it produces a method it is likely NEW PROJECT / weak-lineage.

### HF3 — Temporal structure / cadence stress [expected null, closure-grade quantification]
- Motivation: R2a says ranking is time-invariant at quality-valid points; verify directly on the qk_pool trajectory records: set Jaccard across cycles, stale-set attention-mass decay over horizons 1–8.
- Offline only. Stop unless decay is steep (then cadence-adaptive selection becomes interesting).

### HF4 — Head-wise budget heterogeneity [expected null, cheap]
- Motivation: P34 killed dynamic per-LAYER budgets; head-wise (8 GQA KV heads) never tested.
- Offline probe: per-head missed attention mass; greedy reallocation gain at fixed total budget.
- P34's shuffled-static control predicts null; stop unless simulated gain is large.

### HF5 — Cross-sequence / prefix-reuse [deferred]
- No multi-session harness exists; building one is out of this round's compute budget. Recorded as NEW PROJECT candidate.

### HF6 — Learned selection module [deprioritized]
- qk_pool is near-oracle on current substrate → labels ≈ qk_pool → a learned selector can at best mimic QK. Reconsider only if HF1 finds a regime with structured qk_pool regret.

## Experiment log

(hypothesis | motivation | test | result | verdict | next — appended as work proceeds)

- 2026-08-09 HF-portfolio established. Next: offline probes for HF1/HF3/HF4/HF2-recall on `statekv_qkv_decomposition_qwen3_8b_v1` records (25.7M token_rows).

### Probe round 1 (offline, `analysis/tables/open_search_probes.py`)

- **HF1 coverage stress probe** — mean attention mass captured by attn-top-k core:
  k=220 → 0.957, k=92 → 0.913, k=28 → 0.814; hard-cycle p05: 0.864 / 0.760 / 0.593.
  Early layers 0-1 diffuse (0.84 @ k=220). → budget stress SHOULD discriminate; real
  runs at 768/128 and 768/64 launched. Tables: open_hf1_coverage_by_{cycle,layer}.csv.
- **HF1 regret concentration** — qk_pool per-cycle KL: p50 = 0.00018, p95 = 0.051,
  max = 0.368; top-10% cycles carry 76.5% of KL mass; late cycles worse
  (56-63: 0.0123 vs 0-7: 0.0013). Regret is event-driven, not uniform.
  Table: open_hf1_step_kl_sorted.csv.
- **Hard-cycle predictability: WEAK/NEGATIVE.** Per-cycle exact KL vs runtime
  observables (from token_rows): missed mass −0.02, attention entropy +0.21,
  top-10 mass −0.29, cycle index +0.36 (pool growth), band margin ~0. No clean
  observable predictor of the 64 hard cycles → conditional "budget-on-hard-cycles"
  method hypothesis deprioritized before any run.
- **HF3 temporal stability: CLOSED (confirmation of R2a).** Core-set Jaccard(c,c+1)
  = 0.653; stale-set mass at h=1 0.274 (mostly recency-window sliding, not core
  decay — in_core excludes mandatory sink/recent). Combined with R2a's
  time-invariance finding, cadence-stress has no headroom at horizon-1 semantics.
  Tables: open_hf3_jaccard_by_layer.csv, open_hf3_stale_mass_decay.csv.
- **HF2 approximation frontier: REAL GAP CONFIRMED OFFLINE.** Page-max recall of
  exact top-220 core: p4 0.674, p8 0.623, p16 0.585, p32 0.543 — an upper bound
  for page-granular methods (true attn as within-page max oracle). R0: p16 pages
  → KL 0.0243 = 2.8× qk_pool. Token-level exactness is worth real KL; no
  page granularity recovers it. Table: open_hf2_page_recall.csv.

### Theoretical reframing after probe round 1

At horizon-1 refresh + full-pool visibility, qk_pool is the *optimal memoryless
token-level selector for the current query* (exact top-k of the exact score).
Its residual KL is bounded by 0.0086 @ budget 256 — no selector can gain more
on this substrate. Untested selection degrees of freedom at fixed budget:
1. **per-KV-head selection** (head-mean aggregation is a known suboptimality;
   GQA KV heads each own their cache, so per-head top-k is memory-fair) — HF4;
2. per-cycle adaptive total budget — deprioritized (hard cycles not predictable);
3. per-token utility beyond current attention — closed (swap oracle flat,
   V residual zero).

### Run queue (MPS serialized)

1. `statekv_openstress_768_128` (qk_pool/quest_like/uniform @ 768/128) — RUNNING.
   First sample: qk_pool 0.0314 (2.4× its 256-budget KL), quest_like 0.0605,
   uniform 2.34. Tight coverage discriminates as predicted.
2. `statekv_openstress_768_64` (same arms @ 768/64, coverage ~8%) — queued.
3. `statekv_opencorner_768_64_h16` (qk_pool/uniform @ 64, horizon 16) — queued;
   tests the coverage × cadence corner R2 could not reach at budget 256.
4. `statekv_headwise_probe_qwen3_8b` (3 samples, per-KV-head captured mass) —
   queued; sizes HF4 before any policy implementation.
5. `statekv_openstress_3072_256` (6 samples, ~3.2K pool, coverage ~8%) —
   prepared; only if 768/64 saturates.

HF4 literature position (checked 2026-08-09): per-head budget allocation is
covered by Ada-KV / HeadKV / KV-Compress / DuoAttention / RazorAttention.
Equal-budget per-head position selection under exact query-aware scoring is
incremental over that family — HF4's role is closing the last selection DoF
under our semantics, not claiming a novel method family.

HF4 implementation feasibility (checked `backend_mlx.state_from_anchor`):
the cache path materializes ONE shared keep-set per layer into a standard
KVCache; per-head position sets would require per-head decode masks (custom
attention path).  Decision rule fixed in advance: implement the per-head
policy only if the probe shows a large own-vs-shared mass gap; otherwise
close HF4 on probe evidence.

### Probe round 2 (offline): coverage x cadence corner pre-estimate

Question: does a stale core (refresh every 16) lose meaningful attention mass
at TIGHT budget (64 = sink4+recent32+core28), where R2a's budget-256 closure
might not hold?  Simulated on the decomposition trajectory (approximation:
256-budget trajectory records; sink+recent always fresh in both arms).

Result: full-set captured-mass gap (fresh core minus stale core) averages
**1.9%** over h=1..15 (max ~4% at h~11-13).  The core-only gap looks large
(19%) but is almost entirely absorbed by the always-fresh mandatory window.
Given the locally flat KL-vs-missed-mass link (probe 2.2), the predicted KL
effect is small.  Corner run stays queued for closure-grade confirmation
(cheap: 4 selection cycles x 10 samples), but expectation is null.

### Boundary conditions noted for the final report

- **Metric-boundness**: the whole program measures exact same-input KL to the
  full-cache policy; task metrics (NIAH, 64-token GovReport ROUGE) are
  saturated/insensitive on this substrate (R0 §6.1 audit).  A regime where KL
  says "no difference" but downstream quality differs is not visible here.
- **Single model family**: all closure evidence is Qwen3-8B-4bit (GQA 8 KV
  heads) plus the older Qwen2.5-1.5B line.  MHA models, larger contexts
  beyond ~3K, and multi-turn/agentic workloads are untested.
- HF5 (cross-sequence reuse) and any training-based selector (HF6) remain
  untouched for lack of harness/label signal, not because of evidence.

### Run result: statekv_openstress_768_128 (10 paired samples, exact same-input KL)

| arm | mean KL @128 | mean KL @256 (R0) | ratio within regime |
|---|---|---|---|
| qk_pool | 0.0257 | 0.0086 | 1.0 |
| quest_like (p16) | 0.0767 | 0.0243 | 2.98x |
| uniform | 1.6298 | 0.8952 | 63x |
| full_cache | 0 | 0 | — |

Quality-valid: full NIAH 1.0; qk_pool NIAH 1.0 (uniform 0.0).  GovReport
official: full 53.14 vs qk_pool 53.04 (delta 0.1, inside noise).
Reading: halving coverage triples qk_pool KL; the quest_like/qk_pool gap
holds at ~3x; qk_pool still tracks full-cache task quality exactly.  The
regime discriminates (uniform catastrophic), yet no cheap approximation
closes on qk_pool.

### Run result: statekv_openstress_768_64 (10 paired samples)

| arm | mean KL @64 | @128 | @256 |
|---|---|---|---|
| qk_pool | 0.0819 | 0.0257 | 0.0086 |
| quest_like (p16) | 0.7065 | 0.0767 | 0.0243 |
| uniform | 2.0964 | 1.6298 | 0.8952 |

- qk_pool scales ~3.2x per budget halving but keeps NIAH 1.0 everywhere —
  current-query token-level routing stays task-perfect down to 8% coverage.
- quest_like collapses at 64: 8.6x qk_pool KL, NIAH 0.8 (first task-level
  failure of the page approximation; failures concentrate on NIAH samples —
  GovReport stays mild: e.g. 0.068 vs 0.046 on gov88).
- Interpretation: token-level exactness of the core matters exactly when the
  core is small enough that a 16-token page is a large fraction of it AND the
  task needs specific tokens (needle).  This is the sharpest regime-dependent
  separation measured in this search.

### HF4 probe: instrumentation bug found and fixed

First headwise run recorded only layer 35 (indentation error placed the
recorder outside the layer loop — 1536 rows = 3 x 64 x 8 x 1).  Fixed in
`statekv/qkv_decomposition.py`; probe will rerun after the corner run.
The layer-35-only partial data (NOT decision-grade): own top-k captures
+0.62pp attention mass over the shared core at that layer, own/shared
overlap 0.653.  No conclusion drawn yet.

### Corner gate launched (preregistered: docs/evidence/statekv_corner_gate.md)

Corner result (h16@64, 10 samples): qk_pool mean KL ~0.96, NIAH 0/5 —
quality-INVALID; 16-step-stale core at tight budget is catastrophic, while
the same budget at h1 is fine (0.0819, NIAH 1.0).  First large oracle gap
found in this search: coverage x cadence interaction.

Failure mechanism hypothesis: at core 28, the refresh-time single-token
query under-ranks needle-relevant positions; at h1 the next step rescues
them, at h16 the model commits to a wrong answer before the next refresh.
Prediction: an observation-window score (mean full-pool attention over the
last 32 tokens, SnapKV-style but full-pool + recoverable) ranks those
positions correctly at refresh time and restores quality at h16.

Arms queued: qk_pool @ h4 (curve point), qk_obswin(w32) @ h16 (new
machinery: _full_pool_scores_obswin + _observation_window_tokens +
_mean_score_rows; unit tests in test_recoverable_r0.py; past-only by
construction).  Verdict rules fixed in the gate doc BEFORE any arm results.

### HF4 result (fixed probe, 3 samples x 64 cycles x 36 layers x 8 KV heads)

Per-KV-head own-top-220 vs shared head-mean top-220, identical budget:
captured mass 0.9768 vs 0.9671 -> **+0.96pp mean gain**, p95 +3.6pp,
concentrated in diffuse early layers (layer 0: +4.8pp; layer >= 25: <+0.5pp).
Own/shared core overlap 0.677.

Verdict: CLOSED without policy implementation (pre-committed rule: only a
large mass gap justifies per-head decode masks).  Rationale: the KL-vs-miss
link is locally flat (probe 2.2), so a ~1pp mass gain concentrated where
attention is diffuse is very unlikely to produce meaningful KL gains; the
implementation cost (custom per-head attention path) is high; and the
technique family (Ada-KV/KV-Compress/HeadKV) is literature-covered.
Tables: open_hf4_headwise_overall... (open_hf4_headwise_by_layer.csv,
open_hf4_headwise_rows.csv).

### Refresh-cadence cliff at budget 64 (qk_pool, exact full-pool routing)

| cadence | mean KL | NIAH | GovReport official |
|---|---|---|---|
| h1  | 0.0819 | 1.0 | 6.08 |
| h4  | 0.3761 | 0.2 | 6.06 |
| h16 | 0.8439 | 0.0 | 5.47 |

The KL cliff starts immediately (h1->h4 = 4.6x) and task validity breaks by
h4.  GovReport official is low-resolution (~6 for every arm incl. full
cache; earlier "53" figures were a mixed-task averaging bug in
open_stress_compare.py — direct per-task reads are authoritative).
At 256 R2 found no staleness effect; at 64 staleness is the dominant error
source.  This is the sharpest conditional structure found in this search.

### Corner gate verdict: NO_GO_CORNER

qk_obswin(w32) @ h16: KL 1.0748, NIAH 0.0, paired 2/10 vs qk_pool@h16 —
the backward-looking window is stale-biased and scores WORSE than the
freshest single token.  Per preregistered rules the corner closes as a
boundary condition: at tight coverage, exact routing needs (near) per-step
refresh; no cheap refresh-time scoring rescues slow cadence.  Gate doc:
docs/evidence/statekv_corner_gate.md.
