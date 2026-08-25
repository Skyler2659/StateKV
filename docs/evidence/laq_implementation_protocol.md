# LAQ / LAQ++ implementation protocol (strict closed-loop port)

Scope: port of **Lookahead Q-Cache** (LAQ / LAQ++), EMNLP 2025,
arXiv:2505.20334, into the strict pure-eviction closed loop
(`statekv/causal_closed_loop.py`) as policies `STRICT_LAQ` and
`STRICT_LAQPP`, for a head-to-head novelty stress test against the project's
R2 causal-rollout policy (`STRICT_CAUSAL_ROLLOUT_R2`). Reference
implementation consulted read-only: https://github.com/noforit/Lookahead_Q-Cache
(cloned at `tmp/laq_reference/`; PyTorch/KVCache-Factory, not reused as code).

## 1. The published algorithm

1. Prefill the prompt with a full KV cache.
2. **Lookahead stage**: evict the prompt KV to the target budget with an
   auxiliary compressor (paper/repo default: SnapKV, observation window 32),
   then greedily generate `max_lookahead_size = 8` pseudo response tokens on
   the degraded cache. The per-layer post-RoPE query vectors of these 8
   tokens form the **Q-Cache**.
3. **Re-eviction (stage 2)**: score every original prompt KV position by its
   attention logit mass from the Q-Cache queries, summed over the 8 lookahead
   queries; keep top-B. This is **LAQ** (`stage2_window_size = 0`). **LAQ++**
   adds the local observation window `stage2_window_size = 8` — the queries of
   the last 8 prompt positions — to the observation set, i.e. scores are
   summed over W ∪ Q-Cache (16 queries).
4. Decode with the re-evicted cache. Selection is one-shot; LAQ never
   re-scores.

Reference scoring details (`pyramidkv/pyramidkv_utils.py::update_kv_LAQ`):
per layer, queries (post-RoPE, GQA-expanded) × full prefill keys → softmax →
sum over the observation queries → maxpool (kernel 7) → **per-head, per-layer**
top-(B − window); the last `window` positions are always kept.

## 2. Where this port is faithful

- One-shot selection from pseudo-response queries; scores frozen after the
  decision point and reused for all later cycles (LAQ never re-scores — here
  literally, cycles 1+ reuse the frozen dict).
- Lookahead stage: SnapKV eviction of a full-KV prefix to the **same target
  budget B** (the official `run_LAQ.sh` sets
  `lookahead_max_capacity_prompts = max_capacity_prompts`), observation
  window 32, then 8 greedy pseudo tokens on the evicted branch; the first
  pseudo token comes from the full-KV logits of the last prompt query, as in
  the reference.
- Q-Cache = post-RoPE queries of the 8 pseudo tokens; LAQ++ adds the last 8
  prompt positions' post-RoPE queries captured under full KV (the reference
  caches them during prefill).
- Scores are computed against the **original full prefill keys** (snapshot
  taken before the lookahead eviction), as the reference does
  (`full_past_kv.key_cache`).
- Per (layer, query head), scores are the **sum over observation queries** of
  q·k attention logits, with the reference's causal mask inside the
  observation block reproduced by absolute-position masking (a window query
  cannot score keys after its own position; Q-Cache queries postdate the
  prompt, so their mask is a no-op).
- All 32 Qwen3-8B query heads of every score layer are captured
  (`diagnostic_query_heads` in the LAQ configs), matching the reference's
  all-head scoring.
- No information advantage: LAQ sees only the prompt and its own 8
  self-generated pseudo tokens.

## 3. Deviations forced by the framework

- **Decision point**: the reference evicts at prefill time; we select at
  cycle 0 of the strict loop. These are the same decision point here — the
  strict loop starts eviction at decode onset (cycle 0) and the first
  eviction happens before any token is generated.
- **Shared cross-layer top-core vs per-head/per-layer budgets**: the
  reference picks top-(B−window) indices independently per query head per
  layer; the strict loop selects one token core shared by all layers/heads
  via `rank_and_margin` (identical for every strict policy, including R2).
  Aggregation therefore averages the summed-over-queries logits over query
  heads within a layer and then over the six diagnostic score layers
  (`[0, 7, 14, 15, 21, 27]`, the same layers R2 scores).
- **Raw pre-softmax logits instead of softmax+pooling**: the paper's scoring
  formula is Σ q·k; the reference repo actually applies softmax and
  maxpool (kernel 7) per head before top-k. We implement the paper's raw
  pre-softmax q·k (with the model's attention scale 1/√d); no pooling is
  applied to the LAQ score. The SnapKV pooling kernel is the frozen
  framework value 63 (not the repo's 7), used only for the lookahead-stage
  eviction, consistent with `STRICT_SNAPKV_OBSWIN`.
- **SnapKV lookahead cache inside the strict loop**: the stage-1 SnapKV
  scores come from the anchor's recorded last-32 prompt-query attention rows
  plus the cycle-0 query row (the same vector `STRICT_SNAPKV_OBSWIN` uses),
  with the strict sink=4/recent=32 protection, instead of a fresh per-head
  prefill pass.
- **No early EOS stop in lookahead**: the reference stops the lookahead at
  EOS; we always generate exactly 8 pseudo tokens (the strict loop has no
  EOS stopping), matching `max_lookahead_size` as an exact count.
- **GQA**: the reference repeat-KV-expands keys; we map each query head to
  its KV head (`head // 4` on Qwen3-8B) — algebraically identical.

Fairness invariants kept identical to R2: same `mandatory_and_eligible`
sink/recent protection, same `rank_and_margin` selection, strict physical
eviction unchanged (`_free_rollout` / `apply_selection_in_place`), same
`-inf` masking of non-eligible positions with `current_shared` fallback for
positions that did not exist at cycle 0.

## 4. Hyperparameters and their source

| Parameter | Value | Source |
| --- | --- | --- |
| `lookahead_size` (Q-Cache tokens) | 8 | paper / repo `max_lookahead_size` default |
| Lookahead-stage method | SnapKV | paper / repo `lookahead_method=snapkv` |
| Lookahead-stage observation window | 32 | repo `lookahead_window_size` default |
| Lookahead-stage budget | = target budget B | repo `run_LAQ.sh` (`lookahead_max_capacity_prompts=max_capacity_prompts`) |
| SnapKV pooling kernel | 63 | frozen framework config (`closed_loop.snapkv_pooling_kernel`); repo uses 7 |
| `stage2_window` (LAQ++ local window) | 8 (LAQ: 0) | repo `stage2_window_size` default / `run_LAQ.sh` comment |
| Score layers / heads | layers [0,7,14,15,21,27], all 32 query heads | framework diagnostic layers (same as R2); all heads per the reference |

## 5. Cost accounting

The entire lookahead cost at cycle 0 — full-KV prefix recomputation, the
LAQ++ window replay (7 extra single-token forwards), the current-query
forward, the key snapshot, the SnapKV eviction, the 8 lookahead generation
forwards, and the q·k scoring — is timed end to end in
`_laq_lookahead_scores` and charged to `causal_teacher_time_s` (cycle-0 row)
and hence to `wall_time_s` and the summary `causal_teacher_time_s`;
`causal_teacher_refreshes = 1` marks the one-shot refresh. Cycles 1+ report
zero teacher time. The temporary branch is released after scoring; no
persistent shadow KV is kept (`strict_pure_eviction` remains true and
`peak_active_cache_tokens ≤ budget` on the real decode path).
