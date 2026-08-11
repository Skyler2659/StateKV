# StateKV QK–V Mechanism Discovery — results (routing closure) and pivot

Status: final — routing closure + Family C gate complete; program CLOSED
Date: 2026-08-09
Protocol: `statekv_qkv_discovery_protocol.md`
Method gate: `statekv_qkvtier_gate.md` (NO_GO / TIERING_LOSSY)
Dataset: `results/temporal_cache_discovery/statekv_qkv_decomposition_qwen3_8b_v1/`
(qk_pool trajectory on the exact R0 substrate; per-sample trajectory KL
matches R0's qk_pool arm to 0.0 on all 10 samples — consistency verified)

## 1. Why qk_pool beats the old teacher — the mechanistic answer

Routing and payload live on entirely different scales.  Per layer, over the
full historical pool (token table T, 25M rows):

| quantity | Var(log) range across layers |
|---|---|
| attention (QK routing) | **1.26 – 4.47** |
| projected-V norm ‖v W_O‖ | 0.01 – 0.14 |
| raw V norm | 0.01 – 0.14 |
| Δ (exact removal perturbation) | 0.95 – 4.45 (≈ attention's) |

QK's log-dynamic-range is **20-100× larger than V's**, and the gap widens
with depth.  Δ = a/(1−a)·‖(v−o)W_O‖ inherits its variance entirely from the
attention factor (head-level Spearman(attn, Δ) = 0.96-0.998 at every one of
288 layer×head cells).  Routing × payload decomposition shows the payload
is nearly constant on this substrate: every V-weighted ranking is a small
monotone perturbation of the QK ranking (Top-K Jaccard 0.932 for both
Δ-ranking and a·‖vW_O‖-ranking), and the exact physical-risk score the old
teacher used is, to first order, a noisy monotone transform of attention —
which is why scoring candidates added noise instead of accuracy (Gate 0/1,
R0 §5).

## 2. I(target; V | QK) ≈ 0 — everywhere we measured

Targets: future attention (h=4), exact 1-step swap KL (oracle), revival.
Conditioning: rank-linear residualization on current attention.

- **Near-tie hypothesis: falsified.**  QK-conditioned partial Spearman of
  Δ/‖vW_O‖/‖v‖/a·‖vW_O‖ vs future relevance is **−0.05 to −0.10 in every
  cutoff bucket** (deep_inside, near_inside, boundary_in, boundary_out,
  near_out, far_out) — including the boundary buckets where V was
  hypothesized to matter.  Attention's own residual given V stays
  0.59-0.81: the information flows one way (QK ⊇ V).
- **Exact swap oracle: nothing to predict.**  288 all-layer budget-
  preserving cutoff swaps with exact 1-step same-input KL: median regret
  2e-15, 92% of pairs flat (<1e-4); regret correlates with neither
  attention margin (−0.06) nor Δ difference (−0.05); partial pv −0.10.
  The Gate-2C plateau replicates at shared-mask level under recoverable
  semantics: near-cutoff 1-step downstream effects are unmeasurable, so no
  V feature can predict them.
- **Horizon probe (dev-fit, heldout-eval):** adding all V features to a
  linear rank probe improves future-relevance prediction by ≤ 0.006
  Spearman at h=1/2/4/8 (0.896→0.898 at h=1).
- **Revival (the cliff): not a V phenomenon.**  Revival rates are 5.4%
  (h=1) to 21.2% (h=8); revivers are predicted by *current attention*
  (Spearman 0.22-0.35), while ‖v‖/‖vW_O‖ predict revival at 0.01-0.02.
  Under recoverable semantics qk_pool re-fetches reviving tokens exactly
  when a query attends them (needle tokens sit outside the core 78% of
  cycles yet NIAH = 1.0) — the cliff is structurally defused by
  fetch-on-demand, not by V.
- **No token-type or needle specialization** (partial Δ|attn per class
  −0.04 to −0.15; needle −0.09).
- **Layer/head specialization: none exploitable.**  Per-head attn–Δ
  coupling 0.96-0.998 everywhere.  Per-layer partial Δ|attn vs future
  relevance is mildly positive (0.13-0.28) in mid layers 8-23 but does not
  translate into any selection gain (B: Δ top-K recall of future-relevant
  tokens 0.762 < attention's 0.771; F: no probe gain) — consistent with
  the nonlinear a/(1−a) factor surviving rank-linear residualization, not
  with usable V information.

## 3. Falsified hypotheses (recorded per stop rules)

1. V / projected-V improves routing over QK — falsified (B, C1, F).
2. V matters at the QK cutoff (near-tie) — falsified (C1 buckets, C2 oracle).
3. V predicts future revival — falsified (revival table).
4. V has head/layer pockets of residual value — falsified for selection
   (D tables; mid-layer partial does not convert to recall or probe gains).
5. Re-tested-and-reconfirmed prior closures: α·‖v−o‖-style contribution
   rankings perturb QK by ~7% and slightly *lose* future-relevance recall.

## 4. Family C gate outcome: NO_GO (TIERING_LOSSY)

Gate: `statekv_qkvtier_gate.md` (preregistered P + G1-G6; analysis:
`analysis/tables/qkvtier_gate.py`, tables `qkvtier_gate_main.{csv,md}`,
`qkvtier_gate_paired.csv`; runs `statekv_qkvtier_gate_{256t,352f,352t}_v1`).

| arm | mean KL | p95 step | wins vs qk256 |
|---|---|---|---|
| qk_pool 256 FP16 (baseline) | 0.00862 | 0.0509 | — |
| qk_tiered_v 256/4bit/H96 | 0.00814 | 0.0401 | 6/10 |
| qk_pool 352 FP16 (coverage control) | 0.00430 | 0.0223 | 10/10 |
| qk_tiered_v 352/4bit/H96 (the method) | 0.00484 | 0.0243 | 10/10 |

- **P (premise) PASS**: at fixed 256 coverage, 4-bit cold V is nearly free
  (ratio 0.944; mixed paired wins 6/10 — noise-level; quality non-worse).
  The flat-payload mechanism finding holds in vivo at this coverage.
- **C (coverage worth)**: coverage is enormously valuable — fp16-352 =
  **0.499×** baseline KL, 10/10 wins, better tail, better GovReport.
  qk_pool's only real constraint on this substrate is memory, not ranking.
- **G1-G4 PASS**: the method (memory-matched 352 tiered) beats the
  baseline decisively: 0.562× KL, 10/10 wins, 0.48× tail, quality
  non-worse.
- **G5 (tiering fidelity) FAIL**: tiered-352 gives up **12.6%** of the
  coverage gain to 4-bit V noise (0.00484 vs fp16-352's 0.00430), missing
  the preregistered ≤10% bar.  Residual V information is small but
  nonzero, and it shows up exactly when coverage pressure drops: at 352
  the marginal core tokens are lower-attention, and their payload content
  matters relatively more than the flat-norm statistics suggested.

**Gate verdict: NO_GO (TIERING_LOSSY)** — recorded per protocol; no
bit-width/H/M retuning was attempted.  The gate is a real informative
negative: "V precision is nearly free" is true to ~5% at 256 coverage but
only to ~13% at 352 coverage — not free enough to beat simply spending
the memory on FP16 coverage.

## 5. Program verdict: CLOSE

1. V **routing** beyond QK: closed with strong evidence (§2).
2. V **storage** tiering: tested with a preregistered method gate —
   near-miss negative (G5 = 1.126 vs ≤1.10); method abandoned per rules.
3. The practical positive answer the evidence supports:
   **QK routing + more FP16 coverage** dominates every alternative
   measured on this substrate (0.50× KL per 1.375× tokens, 10/10).
   qk_pool at the largest affordable budget is the strongest policy;
   no V-derived signal improved it, and no physical-risk scoring beat it.

Closure conditions met: the QK-conditioned residual search was systematic
(buckets, layers, heads, token types, horizons, exact swap oracle), the
single surviving hypothesis family received a fair preregistered method
gate, and both ended in evidence-backed negatives.

## 5. Answers to the brief's mechanism questions (1-8 of 13)

1. **Why qk_pool > old teacher**: routing carries the information
   (dynamic-range ratio 20-100×); the teacher's risk score ≈ noisy
   attention transform on a plateau.
2. **Dynamic ranges**: table A above (per layer/task/phase; phase table
   shows the gap is stable across the generation).
3. **V residual given QK**: ≈ 0 (slightly negative) for every target.
4. **Concentration**: none — no layer/head/token/cutoff/horizon pocket.
5. **Stable qk_pool failure mode**: none found on this substrate; its
   only structural weakness is coverage itself (256 of ~1100 tokens),
   which is a memory constraint, not a ranking error.
6. **Candidate signals found**: Δ exact removal perturbation (computed,
   exact, but ≈ attention); V-precision redundancy (storage axis — gated,
   near-miss negative, §4).
7. **Falsified**: §3 (routing) + V-tier method gate G5 (§4).
8. **Worth continuing**: nothing on this substrate; the supported
   practical policy is qk_pool at the largest affordable FP16 coverage.

## 6. Artifacts

- Dataset: `results/temporal_cache_discovery/statekv_qkv_decomposition_qwen3_8b_v1/`
  (token_rows 25M, head_rows, swap_rows 288 pairs, step_rows, summary)
- Machinery: `statekv/qkv_decomposition.py` (exact removal Δ via o_proj
  grams, per-KV-head features, swap oracle, future-target derivation),
  `scripts/run_qkv_decomposition.py`,
  `configs/stages/statekv_qkv_decomposition_qwen3_8b.yaml`,
  `tests/test_qkv_decomposition.py` (7 tests, incl. Δ-identity vs direct
  projection)
- Analysis: `analysis/tables/qkv_residual_analysis.py` →
  `qkv_a_dynamic_range{,_phase}.csv`, `qkv_b_ranking_overlap.csv`,
  `qkv_c1_neartie_partial.csv`, `qkv_c2_swap_{rows,by_offset}.csv`,
  `qkv_d_{layer_residual,head_decoupling}.csv`, `qkv_e_token_type.csv`,
  `qkv_f_horizon_probe.csv`, `qkv_revival_prediction.csv`,
  `qkv_analysis_outputs.json`
