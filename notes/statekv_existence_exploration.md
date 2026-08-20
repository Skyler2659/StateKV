# StateKV Causal Existence Exploration

## Status

This branch is an existence-proof study, not another lightweight adaptive
heuristic sweep. The earlier adaptive-temporal artifacts are frozen. New raw
data and analyses live only under `results/statekv_existence/`.

## Preregistered question

Can a strictly causal estimator, using only the current prefix and state
available at the eviction boundary, recover a material fraction of the
token-time future-attention oracle headroom? If ordinary supervised predictors
do not pass the gate, the study proceeds to causal self-rollout and then to a
counterfactual-removal diagnostic before concluding that the present state is
not exploitable.

## Fresh split and test-opening rule

The split was frozen on 2026-08-20 before collecting any new labels. Three task
families use indices 161--199: single-key NIAH, official LongBench GovReport,
and multi-key NIAH. Indices 161--164 are debug (12 sequences), 165--184 train
(60), 185--191 validation (21), and 192--199 fresh test (24). Expanded sample
IDs and source metadata are written by the collector. The fresh-test evaluation
may be opened once; debugging, feature choice, model choice, and thresholds are
restricted to debug/train/validation.

A separate closed-loop split is reserved at indices 151--160 for the same three
task families (30 sequences). It is disjoint from both the earlier experiments
and the supervised-predictor split and is opened only if Gate B passes. Closed
loop is preregistered at budgets 128 and 256 with strict non-recoverable physical
eviction; the causal teacher uses H=1 prefix recomputation because longer
horizons are reserved for intermediate predictability, not for tuning on this
split. Matched online controls are current-QK, H2O cumulative attention,
SnapKV-style observation-window attention, and train-only tuned per-head fixed
EMA; the latter is the frozen primary Gate-C baseline. Full cache is reported
only as the zero-distortion reference, not as a matched-budget policy.

Before opening that closed-loop split, rollout refresh frequency is selected
only on validation indices 185--186 from each task family, at budget 128 and 16
decode cycles. Frequencies 1/2/4/8 are compared by paired mean exact-KL
improvement against the frozen per-head fixed EMA; the largest improvement is
selected, with lower compute used only as a tie breaker. No closed-loop-test
sample participates in this choice.

The initially proposed range 200--238 was rejected during the pre-label smoke
because the local official GovReport split ends at index 199. No model artifact
or future label was produced before replacing it with the valid 161--199 range;
the latter remains disjoint from all prior adaptive-temporal samples (through
index 90).

The fixed protocol is
`configs/statekv_existence/causal_existence_qwen3_8b.yaml`. Cache budget is 256
tokens (sink 4, recent 32, selected core 220), matching the preceding dynamic
oracle. Horizons are 1, 4, 16, and 32. The primary label is summed future
attention per token and KV head, with H=32 frozen as the learned-predictor
primary horizon. The primary gate is at least 25% oracle-gap
recovery on fresh test, with a sequence-bootstrap confidence interval strictly
above zero and wins on a majority of sequences. At least 50% is a strong
existence result.

## Leakage boundary

Runtime-legal inputs include current and prior attention, the current query,
keys and values already produced by the prefix, current residual/hidden state,
recent query trajectory, token age/position, and global prefix statistics.
Offline future attention supplies labels only. Real future tokens, saved future
attention, reference answers, and full-cache final outcomes are forbidden from
predictor inputs and from causal rollout. Causal rollout must generate its own
continuation from a cloned current-prefix state.

## Planned evidence ladder

The supervised ladder covers history-only probes, query/key/value geometry,
current hidden-state probes, token-wise query-conditioned scoring, a set model,
a temporal model, and a multi-horizon objective. Evaluation reports future
top-K recall, Spearman correlation, pairwise accuracy, NDCG, and oracle-gap
recovery. If Gate A fails, both full-shadow and prefix-recomputation causal
self-rollout are evaluated. If rollout attention is insufficient, temporary
token/group removal is scored through causal rollout using attention mass,
logit KL, target NLL, and logit divergence. Closed-loop eviction is run only
after Gate B passes, on a separately unopened split.

If rollout distillation is entered, teacher generation uses a frozen balanced
30-sequence subset of train (indices 165--174 in each task family). The student
still uses only causal state features and is selected on the full validation
split; its target is R2 prefix-recomputation rollout utility, never offline
real-future attention.

## Results

### EXP-001 — Causal state capture smoke

Question: can the existing Qwen3-8B MLX path expose the preregistered causal
state without modifying model semantics? Method family: data instrumentation.
Causal inputs: current attention, all current pre/post-RoPE queries, existing
K/V, residual and attention inputs, prior attention, and current logits. Future
labels used during training: yes, offline only. Runtime future access: NO. Split:
debug. Result: the real-model smoke produced aligned tensors with shapes
`attention=[2,6,8,1245]`, `query=[2,6,32,128]`,
`residual=[2,6,4096]`, and `K/V=[6,8,1245,128]`; all valid entries were finite.
Interpretation: the requested feature ladder is implementable without a new
backend. Next action: full debug/train/validation collection.

### EXP-002 — Same-model causal rollout smoke

Question: does deterministic current-prefix rollout recover the non-causal
future-attention ranking? Method family: causal rollout. Causal inputs: current
prefix, model weights, and greedy decoding. Future labels used during training:
no. Runtime future access: NO. Split: one debug sequence at cycle 0. Result:
R1 and R2 recovered roughly 99.7--99.9% of the oracle gap across H=1--32;
future top-K recall was 0.986--0.991. R1 cost 2.50x the current scoring forward;
R2 cost 5.53x including prefix recomputation. Interpretation: this is a strong
debug existence signal, not Gate B, because no validation or fresh test has been
opened. Next action: validation and fresh-test sequence-level evaluation.

### EXP-003 — Causal counterfactual group removal

Question: does simulated removal expose action utility beyond rollout attention?
Method family: counterfactual rollout. Causal inputs: current prefix and the
same-model rollout's own tokens/logits. Future labels used during training: no.
Runtime future access: NO. Split: the same single debug boundary, eight disjoint
four-token groups around the retention cutoff, H=16. Result: causal removal KL
had Spearman 0.929 with offline future attention; logit L2 had 0.810 and target
NLL 0.238. Interpretation: distributional divergence is the useful diagnostic;
single-target NLL is too noisy here. Next action: retain KL as the action-teacher
objective and do not tune it on fresh test.

### EXP-004 — Strict physical-eviction smoke

Question: can causal scoring operate without a recoverable cold-token pool?
Method family: strict closed loop. Causal inputs: active physical cache plus
prefix recomputation for the teacher. Future labels used during training: no.
Runtime future access: NO. Split: one debug sequence, budget 128, two cycles.
Result: current-QK, prompt-initialized H2O, prompt-initialized SnapKV,
per-head fixed EMA, and causal-R2 all held exactly 128 active tokens, reported
zero recoverable cold tokens, and completed with a shared token core across
layers. In this one-sequence/two-step smoke, causal-R2 was essentially tied
with current-QK/fixed but worse by about 2.4e-7 mean exact KL; H2O and SnapKV
were much worse. The smoke is not powered for a benefit claim. Interpretation:
the expanded matched execution path is valid, while intermediate recall must
not be treated as closed-loop evidence. Next action: enter the reserved
closed-loop split only after Gate B.

### EXP-005 — Leakage and collector-provenance audit

Question: do early artifacts created before the prefill-only refactor differ
from the final collector path, or allow saved future state to enter features?
Method family: protocol audit. Causal inputs: artifact tensors only. Future
labels used during training: yes, offline only. Runtime future access: NO.
Split: one train and one validation artifact were regenerated and compared
before completing provenance-unified recollection. Result: generated token IDs
were identical and maximum absolute differences for attention, post-RoPE query,
and K/V arrays were exactly zero in the checked pairs. The feature builder was
also changed to align all tensors by physical position ID, mask pre-appearance
history, initialize EMA at first observation, and avoid final-artifact width in
normalization. Interpretation: no measured feature difference came from the
old unused reference-generation call, and the final publication artifacts use
the cleaner prefill-only path. Next action: finish the unified train collection
before fitting any model.

### EXP-006 — Learned causal-state validation

Question: can local causal state recover at least 25% of the H=32 oracle gap?
Method family: classical probes, token-wise neural, set, temporal, and nonlinear
feature ablations. Causal inputs: the frozen 120-dimensional feature ladder.
Future labels used during training: yes, train only. Runtime future access: NO.
Train split: 60 sequences. Validation split: 21 sequences. Test split: sealed.
Result: the frozen `hist_gbdt` candidate recovered 19.01% on validation, with
95% sequence-bootstrap CI [14.77%, 23.35%] and 21/21 sequence wins. It does not
pass the 25% gate. History-only and history+current-query nonlinear models were
negative (-3.37% and -2.37%); adding token keys increased recovery to 13.21%,
then QK geometry 14.73%, value state 16.20%, current hidden state 16.68%, recent
query trajectory 16.62%, and global state/full features 19.01%.
Interpretation: causal state has statistically reliable nonlinear signal, with
the largest increment appearing when token key state is added, but the signal
is below the existence threshold for a cheap/local predictor. Next action:
retain the frozen candidate for the single fresh-test opening and rely on the
preplanned causal rollout branch for Gate B.

### EXP-007 — R1/R2 causal rollout validation

Question: can expensive same-model lookahead recover the oracle gap without
reading saved future tokens? Method family: causal rollout. Causal inputs:
current prefix, model weights, and greedy self-generation. Future labels used
during training: no. Runtime future access: NO. Validation split: 21 sequences.
Test split: sealed. Result: R1 recovered 97.24%, 96.38%, 96.28%, 95.98%,
95.79%, and 94.35% for H=1/2/4/8/16/32; R2 recovered 97.24%, 96.37%, 96.28%,
95.98%, 95.35%, and 93.72%. Every horizon and implementation won on 21/21
sequences. Mean runtime multipliers rose from 2.94x to 32.87x for R1 and from
46.02x to 74.28x for R2. Interpretation: validation strongly supports causal
computability but not deployability. Next action: include both implementations
in the single fresh-test opening.

### EXP-008 — Rollout distillation validation

Question: can a state-only MLP compress the R2 causal teacher? Method family:
rollout distillation. Causal inputs: the same 120-dimensional state features.
Future labels used during training: no real future; only R2 scores on a balanced
30-sequence train subset. Runtime future access: NO. Validation split: 21
sequences. Test split: sealed. Result: the classification head recovered 11.34%
of the real-future oracle gap, CI [8.02%, 14.70%], with 21/21 wins; its utility
head recovered 4.38%. Interpretation: this first student is cheaper but does not
preserve most teacher information. Next action: evaluate it once on fresh test
as preregistered, without further architecture tuning.

### EXP-009 — Single-opening fresh predictability test

Question: do the validation findings replicate on 24 untouched sequences?
Method family: learned predictors, nonlinear feature ablation, causal rollout,
and rollout distillation. Causal inputs: frozen by EXP-006--008. Future labels
used during training: train only for supervised models; no real-future labels
for rollout distillation. Runtime future access: NO. Test split: fresh_test,
opened once and closed after all four registered components. Result: the frozen
`hist_gbdt` H=32 predictor recovered 21.39%, CI [18.45%, 24.36%], with 24/24
wins, so Gate A did not reach 25%. R2 prefix recomputation recovered 95.12%,
95.55%, 96.24%, 95.97%, 93.32%, and 92.57% at H=1/2/4/8/16/32; every horizon
won on 24/24 sequences and every CI lower bound exceeded 86%. R1 recovered
90.85%--95.50%. The distilled classification student recovered 11.12% at H=32,
CI [8.97%, 13.32%]. Interpretation: cheap local state is predictive but below
Gate A; the expensive causal teacher strongly passes Gate B, while the first
student fails to preserve most teacher headroom. Next action: strict closed-loop
testing is authorized.

The fresh nonlinear ladder replicated the mechanism: history/StateKV alone was
-3.42%, adding current query remained negative (-1.69%), and adding token keys
jumped to 15.29%; QK geometry, value, current hidden state, query trajectory,
and global/full state reached 15.99%, 17.22%, 18.73%, 17.66%, and 21.39%.

### EXP-010 — Validation-only closed-loop refresh selection

Question: how often should the expensive R2 teacher be refreshed before the
separate closed-loop test? Method family: strict causal control tuning. Causal
inputs: active KV and prefix-token recomputation. Runtime future access: NO.
Validation split: indices 185--186 from each task family, budget 128, 16 cycles.
Test split: sealed. Result: refresh frequencies 1/2/4/8 improved mean exact KL
over fixed EMA by 1.404/1.427/1.073/1.017; frequency 2 was selected by the
preregistered maximum-improvement rule (CI [0.441, 2.616], 5/6 wins).
Interpretation: every tested cadence was positive on this bounded validation
sweep, with frequency 2 best. Next action: freeze frequency 2 and run the full
30-sequence, two-budget closed-loop test.
