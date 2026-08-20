# StateKV Existence Study: Literature Check

Search date: 2026-08-20. The screening log and machine-readable evidence table
are under `literature/statekv_existence/`. Searches covered learned eviction,
future-utility prediction, query-conditioned selection, RL, draft/lookahead
generation, and counterfactual utility. Primary arXiv, OpenReview, and ACL
sources were prioritized; MDPI was excluded.

## Direct overlap

The direction is no longer an unexplored hypothesis. Four 2026 papers overlap
materially with the proposed existence study:

- [ForesightKV](https://arxiv.org/abs/2602.03203) constructs future-attention
  Golden Eviction traces, distills them with pairwise ranking, and adds GRPO.
  This is the closest match to the supervised teacher-label and RL portions.
- [Learning to Evict from Key-Value Cache](https://arxiv.org/abs/2602.10238)
  trains lightweight per-head RL rankers from cached keys, values, and positions
  using a holistic future-utility reward. Its reported zero-shot GovReport and
  RULER transfer is especially relevant to the fresh-split requirement.
- [LookaheadKV](https://arxiv.org/abs/2603.10899) distills true
  future-response attention into learned lookahead tokens and selectively
  activated LoRA modules. It explicitly positions draft generation as a strong
  but expensive causal future proxy and reports Qwen3-8B experiments.
- [Predicting Future Utility / LU-KV](https://arxiv.org/abs/2602.08585)
  emphasizes heterogeneous long-horizon behavior across heads and performs
  global marginal-utility budget allocation. This directly overlaps the
  headroom-decomposition motivation, although its main decision variable is
  head-level allocation rather than token-time horizon selection.

Important antecedents are
[Attention-Gate](https://arxiv.org/abs/2410.12876), a learned global-context
gate; [Lookahead Q-Cache](https://arxiv.org/abs/2505.20334), which generates
pseudo lookahead queries; and
[SpecKV](https://arxiv.org/abs/2506.08373), which uses a smaller draft model to
produce lookahead for KV dropping.

## Consequence for this branch

The scientifically defensible contribution is not a first claim that future KV
utility can be learned or approximated causally. The value of this branch is a
controlled StateKV-specific audit: fixed token-time oracle denominator,
strictly separated feature ladder, fresh sequence split, explicit R1/R2 causal
self-rollout comparison, matched budget, sequence-level uncertainty, and a
test of whether intermediate predictability survives physical closed-loop
eviction. Any positive result must be framed as independent triangulation and
mechanistic decomposition relative to these works. Any negative result must be
scoped to this model/task/budget/state interface rather than generalized to the
field.

No screened paper was found that exactly matches the proposed same-model,
current-prefix, repeated decode-boundary counterfactual group-removal diagnostic
with both logit divergence and downstream strict-pure-eviction validation. That
is a narrower differentiation, not evidence of broad novelty.
