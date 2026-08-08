# StateKV

**State-conditioned physical risk for KV-cache selection and refresh.**

Repeated KV-cache compression is state-dependent: the consequence of a retained
set depends on the compressed trajectory that produced the current model state.

StateKV treats each candidate retained set as a physical intervention at that
state. It propagates the resulting deletion-and-renormalization response to
downstream logits, estimates the associated increase in output risk, and uses the
same risk object for selection and refresh.

The supported identity is a same-state physical evaluator. L1/L2 leverage,
attention, recency, value norm, SnapKV, and H2O remain baselines, proposal
mechanisms, or diagnostics—not as StateKV itself.

The repository now spans both sides of the control problem. P25--P31 establish
an expensive physical teacher: at every boundary it rolls out a legal action
panel, ranks the resulting trajectories by full-vocabulary output risk, and
carries the chosen compressed state forward with cold-token recovery. On ten
Qwen3-8B sequences at a 256-token per-layer budget, P31 reduces mean trajectory
KL to $0.05057$, versus $0.33572/0.13250/0.54677$ for repeatedly applied
Attention/SnapKV/H2O, while preserving all five NIAH needles.

P32 provides the first direct cheap-controller evidence. Its training-free
policies emit one retained set and run zero candidate model rollouts. Temporal
attention volatility reaches mean KL $0.09525$ and beats Attention on all ten
samples. The strongest current variant combines an online token-utility signal
with state-dependent layer budgets: it reaches KL $0.114995$, the best observed
task-score point estimate, and keeps the global KV allocation fixed. These are
screening results rather than final benchmark evidence: the run contains ten
samples and 64 generated tokens, retains a CPU full-KV backing store, and has not
yet separated dynamic state alignment from a static layer-budget prior.

![StateKV architecture](assets/statekv-architecture.png)

The editable [TikZ source](assets/statekv-architecture.tex) and vector
[PDF](assets/statekv-architecture.pdf) are versioned with the repository.

## Novelty boundary

The broad idea of jointly deciding what to preserve and when to update is not
claimed as unique: [PReM](https://arxiv.org/abs/2607.14327),
[RefreshKV](https://aclanthology.org/2025.acl-long.1211/), and
[QEvict](https://arxiv.org/abs/2608.05326) already connect cache selection with
refresh, recovery, or tier movement. StateKV's narrower research hypothesis is
that **compression-history-conditioned physical retained-set interventions can
be propagated to downstream output-distribution risk**, and that a deployable
controller should use one retained-set risk for both its action and the regret
of keeping a stale action. The first part now has controlled and independently
repeated teacher-forced closed-loop evidence; replacing its candidate rollouts
with one low-cost observable remains open after P23 and P24. The structured
prior-art audit is stored in
[`analysis/literature`](analysis/literature/).

## Research question

Static token scores assume that a candidate has the same consequence whenever it
is evaluated. Repeated compression breaks that assumption: earlier evictions
change later queries, attention patterns, residual streams, and KV contents.
StateKV asks:

> Given the compressed trajectory observed now, which retained set produces the
> smallest increase in output risk, and when does state evolution invalidate
> that choice?

Candidate pools can come from baseline selectors or other proposal mechanisms.
StateKV contributes the state-conditioned evaluator shared by the subsequent
decision stages:

1. **Candidate generation** proposes legal retained sets at a fixed budget.
2. **State-conditioned evaluation** estimates each candidate's finite downstream
   effect and incremental output risk under the current trajectory.
3. **Selection and refresh** minimize retained-set risk and compare the old set
   with the new optimum under that same risk.

The training-free direct-policy branch bypasses candidate enumeration: each
policy computes one token score, selects one shared core, and runs only that cache
action. Attention dynamics and lexical rarity are evaluated as separate signals;
they are never executed as a runtime candidate panel.

## Method at a glance

1. **Accumulate state.** Earlier compression decisions move the model away from
   its paired full-cache trajectory.
2. **Intervene at the current state.** Each candidate retained set defines a
   concrete deletion-and-renormalization action.
3. **Measure the downstream response.** The action is transported through the
   remaining network to obtain its effect on output logits.
4. **Rank by local output risk.** A state-conditioned KL approximation converts
   the logit response into a scalar candidate risk.
5. **Refresh by stale-action regret.** As the compressed trajectory evolves,
   compare the old action's current risk with the current minimum.

The teacher makes this pipeline measurable by pairing the compressed run with a
full-cache reference and using deep per-candidate probes. It supplies diagnostic
risk labels for studying selection and refresh; an online policy must replace
those measurements with one cheap retained-set risk and expose it through the
single controller interface.

## Minimal formulation

At boundary $b$, compression history is represented by the displacement between
the observed compressed state and its paired full-cache reference:

```math
\mathbf{s}_{t,b}
=
\mathbf{x}^{\mathrm{hist}}_{t,b}
-
\mathbf{x}^{\mathrm{ref}}_{t,b}.
```

For a retained set $C$, let $\widehat{\Delta\mathbf z}(C)$ denote the transported
finite response in output logits. The teacher computes the exact local
deletion-and-renormalization action and evaluates its second-order increment in
reference KL at the observed state:

```math
\widehat{\mathcal R}_{\mathbf s}(C)
=
\mathbf g_{\mathbf s}^{\top}\widehat{\Delta\mathbf z}(C)
+
\frac{1}{2}
\widehat{\Delta\mathbf z}(C)^{\top}
\mathbf F_{\mathbf s}
\widehat{\Delta\mathbf z}(C).
```

Here $\mathbf g_{\mathbf s}$ and $\mathbf F_{\mathbf s}$ are the local first- and
second-order terms of the reference KL geometry. Selection and refresh use the
same risk object:

```math
C^{\star}_{\mathbf s}=\arg\min_{C\in\mathcal A_{t,\ell}(B)}
\widehat{\mathcal R}_{\mathbf s}(C),
\qquad
G_{\mathbf s}(C_{\mathrm{old}})
=\widehat{\mathcal R}_{\mathbf s}(C_{\mathrm{old}})
-\widehat{\mathcal R}_{\mathbf s}(C^{\star}_{\mathbf s}).
```

The teacher refresh diagnostic is $G_{\mathbf s}(C_{\mathrm{old}})>\tau$. A
training-free token cost $q_{t,i}\ge0$ induces the deployable counterpart

```math
\widetilde{\mathcal R}_t(C)=\sum_{i\notin C}q_{t,i},\qquad
\widetilde G_t(C_{\mathrm{old}})=
\widetilde{\mathcal R}_t(C_{\mathrm{old}})-
\min_{|C|=B}\widetilde{\mathcal R}_t(C).
```

One top-$B$ operation selects $C_t^\star$; the same cost vector decides whether
the old set is stale. The interface is implemented, but the tested $q_{t,i}$ do
not yet support a successful unified deployable controller.

## Evidence

The [frozen experiment registry](experiments/frozen_registry.yaml) links each
claim to its protocol, manifest, and stored result artifacts.

| Finding | Stored evidence and scope |
|---|---|
| The exact set-level deletion identity reaches a maximum FP64 L2 error of $2.26\times10^{-11}$. | [P0 identity rows](experiments/p0_v2_fixed_boundary/results/identity_rows.parquet), at the fixed operating point and stored candidate protocol. |
| Physical boundary replay reaches sequence-first cosine $\approx 1$ and relative L2 $8.09\times10^{-7}$. | [P0 summary](experiments/p0_v2_fixed_boundary/results/p0_v2_summary.json), under controlled fixed-boundary replay. |
| Evaluation at the observed state reaches cosine $0.99974$ and relative L2 $0.02255$ in the P1 operating-point diagnostic. | [P1 summary](experiments/p1_state_conditioned/results/state_operating_point_summary.json), on four stored sequences from the frozen Qwen protocol. |
| The finite-action approximation has a visible trust region: cosine falls from $0.99986$ at amplitude $1/16$ to $0.95463$ at amplitude $1$, with median residual slope $1.983$. | [R1 summary](experiments/p2_recovery/r1_amplitude_trust_region/results/r1_summary.json), from the retrospective amplitude study. |
| The two-midpoint state-local scalar-risk evaluator obtains Spearman $1.0$ and top-1 gain $1.0$ in both evaluation and replication splits, outperforming the stored action-only ranking. | [Evaluation](experiments/p2_recovery/r4_scalar_decision_risk/results/evaluation/analysis_summary.json) and [replication](experiments/p2_recovery/r4_scalar_decision_risk/results/replication/analysis_summary.json), on frozen candidate pools. |
| Dense all-layer mechanistic risk transfers across the limited P3PR model/task study (Spearman $1.0$ formal and $0.9940$ replication; top-1 $1.0$), while the tested relative single-boundary shortcut does not. | [P3PR summary](experiments/p3pr_generalization/results/analysis/analysis_summary.json), across two model families, two task families, and the stored splits. |
| A fixed-decay training-free history sketch does not improve the stored layer-27 action-energy ranking. At the predeclared 64-dimensional, $\rho=0.95$ setting, median Spearman changes by $-0.0157$ on evaluation and $-0.0217$ on replication; pairwise accuracy and normalized regret also worsen. | [StateKV-TF P0 gate](results/temporal_cache_discovery/statekv_tf_sketch_p0_v1/summary.json), a retrospective analysis of 24 real-model frozen trajectories. The negative result holds over 16--128 dimensions and decays 0.5--1.0. |
| Unlabeled diagonal RMS scaling produces small ranking gains but does not pass the metric-repair gate: the primary diagonal-plus-EMA setting improves median Spearman by $0.0096$ on evaluation and $0.0122$ on replication, while evaluation normalized regret worsens by $0.0063$. | [Metric-repair P1 gate](results/temporal_cache_discovery/statekv_tf_metric_repair_p1_v1/summary.json), a retrospective development screen with three projection seeds. The contiguous 12-block variant is a residual-coordinate diagnostic, not a headwise Fisher model. |
| A rank-4 shared pullback refreshed every four steps does not repair the decision metric. Its 32-dimensional randomized Fisher sketch has zero median Spearman gain, pairwise gain $-0.0281$, and normalized-regret gain $-0.0432$ against hidden action energy. | [Shared-pullback P2 pilot](results/temporal_cache_discovery/statekv_shared_jvp_pilot_p2_v1/summary.json), on two real-model development sequences and eight candidates. The installed MLX build could not differentiate `Sum` in forward mode, so this pilot transparently used symmetric finite differences: 64 shared directions, 144 tail-forward evaluations, 21.7 seconds. |
| Reverse-mode output-side VJP is numerically valid, but the predeclared Gaussian 16-direction route fails its gate. A post-hoc Rademacher 8-direction/state variant lowers normalized regret from $0.1945$ to $0.0696$ while missing pairwise accuracy by $0.0026$. | [VJP stress test](results/temporal_cache_discovery/statekv_vjp_routes_p3_stress_v1/summary.json). The promising Rademacher setting is exploratory and requires independent replication. |
| Post-attention VJPs at layers 0, 14, and 27 are valid but do not improve ranking: the predeclared width-8, refresh-4 sum has normalized-regret gain $-0.0669$ and costs about six reverse passes per token. | [Multi-boundary VJP pilot](results/temporal_cache_discovery/statekv_post_multiboundary_vjp_p5_v1/summary.json), with action reconstruction error below $7.2\times10^{-4}$. |
| A direct four-query mean-contribution selector, locked after a four-sequence screen, improves held-out local projected error from $0.0978$ to $0.0709$ and wins $70.8\%$ of matched units. It generates one set and runs zero candidate algorithms. | [Local replication](results/temporal_cache_discovery/statekv_direct_coreset_p4_replication_v1/summary.json), on eight different sequences, three boundaries, and eight future queries. |
| Applying that one shared set to all 28 layers lowers teacher-forced mean exact KL from $0.0485$ to $0.0199$, P95 KL from $0.1793$ to $0.1576$, and maximum KL from $1.6502$ to $0.1692$; six of eight sequences improve. | [Physical replay](results/temporal_cache_discovery/statekv_direct_policy_replay_p6_v1/summary.json) and [paired uncertainty](results/temporal_cache_discovery/statekv_direct_policy_replay_p6_v1/analysis/uncertainty.json). This is held-out development evidence, not free generation. |
| On 12 new sequences and three anchors, the pure contribution policy lowers overall mean KL from $0.3293$ to $0.2569$ and P95 from $1.5670$ to $1.0743$, but slightly worsens the NIAH task mean ($0.15993\rightarrow0.16061$). | [Independent P7 gate](results/temporal_cache_discovery/statekv_direct_policy_independent_multianchor_p7_v1/summary.json) and [paired uncertainty](results/temporal_cache_discovery/statekv_direct_policy_independent_multianchor_p7_v1/analysis/uncertainty.json). The predeclared all-task gate fails. |
| A development-selected 25% contribution shrinkage policy improves mean, P95, maximum, both task means, all three anchor means, and 9/12 sequence means on another 12 new sequences. Mean KL falls from $0.3577$ to $0.3298$, but only 17/36 sample--anchor units improve, below the locked 55% gate. | [Independent P9 gate](results/temporal_cache_discovery/statekv_direct_policy_shrinkage_independent_p9_v1/summary.json) and [paired uncertainty](results/temporal_cache_discovery/statekv_direct_policy_shrinkage_independent_p9_v1/analysis/uncertainty.json). This supports a tail-risk interpretation, not promotion to free generation. |
| The minimal six-layer MLX capture hook preserves logits exactly. At a 128-token cache it adds 0.31 ms to an 18.89 ms decode step; scheduled four-step capture plus CPU scoring amortizes to about 0.43 ms per step at a 16-step refresh interval. Full-cache capture near 1K tokens is materially costlier, so continuous rolling is rejected. | [P10 runtime profile](results/temporal_cache_discovery/statekv_direct_policy_runtime_profile_p10_v3/summary.json). This is an arithmetic/capture microbenchmark, not an end-to-end speedup result. |
| A P8-developed training-free trigger activates shrinkage only when attention/contribution total-variation exceeds $0.24735$. On P9 it lowers mean KL from $0.35766$ to $0.34125$ and P95 to $1.74473$, but activated-unit win rate is only 50% and the NIAH mean is slightly worse. | [Selective P12 gate](results/temporal_cache_discovery/statekv_direct_policy_selective_trigger_independent_p12_v1/summary.json). Selection hashes match the source replay exactly and no extra replay forward is used; the locked selective gate fails. |
| On a third set of 12 new sequences, fixed 25% shrinkage lowers mean KL from $0.52735$ to $0.51766$ and P95 from $3.62435$ to $2.77128$, but CVaR95 rises from $6.57390$ to $6.63419$ and the KL$\ge1$ rate rises from 11.63% to 11.81%. | [Tail-risk P13 gate](results/temporal_cache_discovery/statekv_direct_policy_tail_risk_independent_p13_v1/summary.json), [uncertainty](results/temporal_cache_discovery/statekv_direct_policy_tail_risk_independent_p13_v1/analysis/uncertainty.json), and [tail migration](results/temporal_cache_discovery/statekv_direct_policy_tail_risk_independent_p13_v1/analysis/tail_migration.json). The policy removes four large-loss steps but creates five, so the preregistered tail gate fails. |
| Protecting the top attention tokens and allowing only 4, 8, or 16 contribution-based rescue slots bounds the action change as intended, but no setting passes the development gate. The best-mean $m=8$ variant changes at most eight core tokens and lowers mean KL from $0.29580$ to $0.29367$ and P95 from $1.87740$ to $1.60877$, while CVaR95, maximum KL, and the KL$\ge1$ rate all worsen. | [Protected-rescue P14 screen](results/temporal_cache_discovery/statekv_direct_policy_protected_rescue_screen_p14_v1/summary.json), [selection audit](results/temporal_cache_discovery/statekv_direct_policy_protected_rescue_screen_p14_v1/analysis/protected_rescue_selection.json), and [tail migration](results/temporal_cache_discovery/statekv_direct_policy_protected_rescue_screen_p14_v1/analysis/tail_migration.json). The audit selects no candidate and does not authorize use of the untouched independent split. |
| A fresh development screen compares six fixed signals from access peaks, access dynamics, Key geometry, Value geometry, representation boundaries, and position coverage. Only four-step temporal attention volatility passes all six mean/tail/task constraints, lowering mean KL from $0.36262$ to $0.30730$. | [P15 metrics](results/temporal_cache_discovery/statekv_direct_policy_signal_family_screen_p15_v1/metrics.csv) and [frozen selection](results/temporal_cache_discovery/statekv_direct_policy_signal_family_screen_p15_v1/analysis/signal_family_selection.json). The other five signal families are rejected without within-family parameter searches. |
| On 12 untouched sequences, the frozen temporal-volatility policy passes the aggregate tail-risk gate: mean KL $0.52419\rightarrow0.47115$, P95 $3.02075\rightarrow2.85653$, CVaR95 $5.33994\rightarrow4.98184$, maximum KL $13.19648\rightarrow12.84552$, and KL$\ge1$ rate $13.72\%\rightarrow12.33\%$. Both tasks and all three anchors improve. | [Independent P16 gate](results/temporal_cache_discovery/statekv_direct_policy_temporal_volatility_independent_p16_v1/summary.json), [uncertainty](results/temporal_cache_discovery/statekv_direct_policy_temporal_volatility_independent_p16_v1/analysis/uncertainty.json), and [tail migration](results/temporal_cache_discovery/statekv_direct_policy_temporal_volatility_independent_p16_v1/analysis/tail_migration.json). Sequence wins are 6/12 and the bootstrap interval crosses zero, so the supported claim is aggregate-risk improvement rather than per-sequence superiority. |
| The rolling temporal-volatility state stores four head-averaged attention vectors at six layers: 96 bytes per context token. At 32K tokens, its measured CPU arithmetic costs 0.25 ms per decode-step update and 2.82 ms per refresh. | [P17 runtime profile](results/temporal_cache_discovery/statekv_temporal_volatility_runtime_profile_p17_v1/summary.json). This excludes attention capture and end-to-end decode latency. |
| In matched-budget free generation, temporal volatility improves GovReport ROUGE-L from $7.31$ to $7.95$ and ties RULER at 100, but overall NLL worsens by $0.00374$; the frozen P18 gate fails. Throughput is $95.7\%$ of latest attention and cache length remains 128. | [P18 free-generation gate](results/temporal_cache_discovery/statekv_temporal_volatility_freegen_p18_v1/summary.json). Current 16K attention capture peaks at 6.60 GB, $3.19\times$ the full-cache reference, so the implementation is not a low-memory deployment path. |
| Attention-free KNorm, KeyDiff, and VNormL2 restore full-reference peak memory, but all miss both P19 RULER needles and reach only $35\%$--$41\%$ of the random-control decode throughput. KeyDiff lowers overall NLL by $0.921$ against random and slightly improves GovReport, but is not replication-eligible. | [P19 geometry screen](results/temporal_cache_discovery/statekv_attention_free_geometry_screen_p19_v1/summary.json), on four development samples and three fixed signal families. |
| A static token-rarity selector passes the P20 development screen: GovReport ROUGE-L is $10.88$ versus $9.60$ random and $9.04$ position coverage; RULER recovery is 2/2 versus 0/2 for both controls. It uses no attention capture or K/V rescoring. | [P20 static lexical screen](results/temporal_cache_discovery/statekv_static_lexical_screen_p20_v1/summary.json). This result authorizes replication but is not itself confirmatory evidence. |
| On six untouched P21 samples, token rarity retains all three 16K RULER needles and improves throughput by $25.2\%$ while reducing peak memory to $31.3\%$ of latest attention. It nevertheless fails the cross-task gate: GovReport ROUGE-L is $8.09$ versus $8.92$, and overall NLL is worse by $0.00334$. | [P21 independent replication](results/temporal_cache_discovery/statekv_token_rarity_replication_p21_v1/summary.json). The supported boundary is retrieval-specific feasibility, not a universal selector. |
| In the P22 development audit, latest attention is the only one of seven fixed signals to pass the joint action/refresh screen: action median Spearman is $0.786$, normalized regret is $0.152$, and refresh-benefit Spearman is $0.378$. | [P22 summary](results/temporal_cache_discovery/statekv_risk_consistent_proxy_alignment_p22_v1/summary.json). Candidate actions are an offline label panel; a deployed proxy emits one set and runs zero candidate algorithms. |
| The frozen latest-attention proxy does not replicate its refresh result on six untouched P23 sequences. Action alignment remains strong (median Spearman $0.750$, normalized regret $0.118$), but refresh-benefit Spearman reverses to $-0.350$; the joint gate fails. | [P23 independent audit](results/temporal_cache_discovery/statekv_risk_consistent_proxy_independent_p23b_v1/summary.json) and [new physical source replay](results/temporal_cache_discovery/statekv_risk_consistent_proxy_independent_source_p23a_v1/summary.json). Thresholds and proxy identity were unchanged after P22. |
| Replacing pure access probability with a four-query attention--Value contribution cost does not repair the controller. In P24 its refresh-benefit Spearman is $0.011$ versus $0.160$ for latest attention, and neither signal passes the development gate. | [P24 output-aware audit](results/temporal_cache_discovery/statekv_risk_consistent_output_aware_proxy_p24_v1/summary.json). This reuses P7 sequences and is exploratory negative evidence. |
| The P25 physical-oracle loop completes all eight sample--strategy runs for three control cycles. Every cycle preserves state continuity and the 128-token budget, retains at least four distinct candidate cores, and the loop executes seven post-initial refreshes with seven cold-token recoveries. | [P25 closed-loop summary](results/temporal_cache_discovery/statekv_physical_oracle_closed_loop_p25_v1/summary.json) and [analysis](results/temporal_cache_discovery/statekv_physical_oracle_closed_loop_p25_v1/analysis.json). This is the development closure test. |
| On four untouched P26 sequences, all 16 sample--strategy loops complete four eight-token control periods. The run executes 21 post-initial refreshes and recoveries. Dense mean quadratic risk reaches median Spearman $0.957$ and 93.75% top-1 agreement with exact mean KL; selected exact KL averages $0.3281$ versus $0.3970$ for stale actions. | [P26 independent closed-loop summary](results/temporal_cache_discovery/statekv_physical_oracle_closed_loop_independent_p26_v1/summary.json) and [analysis](results/temporal_cache_discovery/statekv_physical_oracle_closed_loop_independent_p26_v1/analysis.json). These are teacher-forced candidate rollouts, not a low-cost or free-generation result. |
| In a matched teacher-forced comparison, each policy owns its compression history while consuming the same full-cache tokens. On four untouched P28 sequences, StateKV mean exact KL is $0.1482$, versus $0.3046$ for latest attention, $0.7247$ for SnapKV, and $0.5196$ for H2O; StateKV wins every sample-level comparison. | [P28 summary](results/temporal_cache_discovery/statekv_oracle_policy_comparison_independent_p28_v1/summary.json). All methods use per-layer actions, the same 128-token budget, and the same cold recovery capability. |
| Free-generation horizon matters. On P29 development data, only per-token control ($H=1$) passes the joint risk/quality gate; $H=4$ and $H=8$ can lose trajectory KL to SnapKV because teacher-forced lookahead and the realized greedy prefix diverge. | [Horizon analysis](results/temporal_cache_discovery/statekv_oracle_policy_freegen_independent_p30_v1/analysis/horizon_ablation.csv), covering complete $H=1/4/8$ runs with 64 generated tokens. |
| On four new P30 greedy-generation sequences, frozen $H=1$ StateKV lowers mean trajectory KL to $0.1391$, reductions of 66.5%, 33.2%, and 65.4% against attention, SnapKV, and H2O. The task-quality gate fails: GovReport ROUGE-L trails attention and SnapKV, all compressed policies score 0/2 on NIAH, and full cache scores 2/2. | [P30 summary](results/temporal_cache_discovery/statekv_oracle_policy_freegen_independent_p30_v1/summary.json) and [consolidated analysis](results/temporal_cache_discovery/statekv_oracle_policy_freegen_independent_p30_v1/analysis/analysis.json). This supports distribution-risk guidance, not universal generation superiority. |
| On ten Qwen3-8B P31 sequences, the per-token exact-risk teacher reaches mean KL $0.05057$, versus $0.33572$, $0.13250$, and $0.54677$ for Attention, SnapKV, and H2O. It retains all five NIAH needles and is competitive on five GovReport samples. | [P31 summary](results/temporal_cache_discovery/statekv_oracle_policy_freegen_qwen3_8b_n10_p31_v1/summary.json). The teacher evaluates seven candidate rollouts per decision and retains CPU cold recovery. |
| P32 removes candidate model rollouts. Temporal volatility reaches KL $0.09525$ and improves over Attention on 10/10 samples. A direct token-utility controller with dynamic layer budgets reaches KL $0.114995$ and the best observed task-score point estimate without increasing the global KV allocation. | [P32 summary](results/temporal_cache_discovery/statekv_cheap_policy_freegen_qwen3_8b_n10_p32_v1/summary.json) and [compact comparison](results/temporal_cache_discovery/statekv_cheap_policy_freegen_qwen3_8b_n10_p32_v1/comparison_table.csv). Dynamic-versus-static causality, pure eviction, and paper-scale evaluation remain open. |

Together, these results support state-conditioned physical evaluation and a
candidate-specific teacher within the frozen protocols.

## Training-free direct policy

The low-cost branch treats each signal as a candidate definition of the token
cost $q_{t,i}$ above. A chosen signal emits one shared cache set directly; the
seven-signal P22 panel exists only to measure alignment and is never a runtime
ensemble. Latest attention currently demonstrates action-selection feasibility,
not novelty or a validated refresh controller.

For diagnostic boundaries $\mathcal B$, define the latest-query attention score
$a_i^{(b)}$ and the $W$-query value-aware contribution score $c_i^{(b)}$ after
normalization over eligible tokens. One previously explored shrinkage policy is

```math
s_i=\frac{1}{|\mathcal B|}\sum_{b\in\mathcal B}
\left[(1-\lambda)a_i^{(b)}+\lambda c_i^{(b)}\right],
\qquad
c_i^{(b)}=\operatorname{Norm}_b\!\left[
\frac{1}{WH}\sum_{\tau=t-W}^{t-1}\sum_h
\alpha^{(b,h)}_{\tau i}\lVert v_i^{(b,h)}-o_\tau^{(b,h)}\rVert_2
\right].
```

That P8/P9 variant uses $W=4$, $\lambda=0.25$, and
$\mathcal B=\{0,7,14,15,21,27\}$. It selects the 92 highest-scoring eligible
tokens; sink and recent tokens complete the 128-token physical budget. Layer
scores are normalized before aggregation, and the same selected core is applied
to all 28 layers. Runtime candidate enumeration is zero.

P14 tests a bounded alternative rather than another mixture coefficient. For a
92-token eligible core, it protects the top $92-m$ attention tokens and uses the
contribution score only to fill $m\in\{4,8,16\}$ rescue slots:

```math
C_m=\operatorname{Top}_{92-m}(a)\;\cup\;
\operatorname{Top}_{m}\!\left(c\mid i\notin\operatorname{Top}_{92-m}(a)\right).
```

Thus $|C_m\setminus C_{\mathrm{attn}}|\le m$, with one physical retained set and
zero runtime candidate algorithms. The bound holds in every replayed unit, but
the output-risk constraints do not; limiting set displacement alone is not a
reliable surrogate for limiting KL harm.

The next mechanism uses access dynamics rather than contribution magnitude or
static feature geometry. For each diagnostic layer, it computes the standard
deviation of head-averaged attention over the last four queries:

```math
s_i=\frac{1}{|\mathcal B|}\sum_{b\in\mathcal B}
\operatorname{Norm}_b\!\left[
\operatorname{Std}_{\tau=t-4}^{t-1}
\left(\frac{1}{H}\sum_h\alpha_{\tau i}^{(b,h)}\right)
\right].
```

The 92 highest-scoring eligible tokens form one set shared by all layers. The
signal needs no values, labels, learned weights, candidate enumeration, or
training; it retains tokens whose recent access pattern is changing rather than
tokens with the largest average access.

The earlier contribution runtime holds five FP32 vectors per diagnostic layer:
four recent contribution vectors plus the latest attention vector, or 120 bytes
per context token across six layers. The temporal-volatility runtime holds only
four head-averaged attention vectors per layer, or 96 bytes per context token
(3 MiB at 32K tokens), and does not read Value vectors. Both emit one shared
top-$k$ set. The P10 capture study rejects continuous full-cache transfer, so
attention capture remains scheduled for the final four steps before refresh.

The attention-free lexical branch scores the observed token stream directly. If
$f_t(u)$ is the count of token ID $u$ in the observed stream and
$\mathcal N_i=\{i-2,\ldots,i+2\}$ is a clipped local span, the frozen P20/P21
score is

```math
r_i=\frac{1}{|\mathcal N_i|}\sum_{j\in\mathcal N_i}
\log\frac{t+1}{f_t(x_j)+1}.
```

Sink tokens and the 32 most recent tokens are protected; the 92 highest-scoring
eligible positions form the shared core for all layers. Counts are updated once
per observed token, the selector runs no model probes and no candidate
algorithms, and it never materializes an attention matrix. P21 shows why this
simple signal must remain task-scoped: it preserves all tested RULER needles but
does not match latest-attention quality on independent GovReport samples.

The merge diagnostic assigns a deleted value to its nearest retained
representative $\pi(i)$. Its local error satisfies

```math
\left\|\sum_{i\in D}\alpha_i
\left(v_{\pi(i)}-v_i\right)\right\|_2
\le
\sum_{i\in D}\alpha_i
\left\|v_{\pi(i)}-v_i\right\|_2.
```

On the held-out local replication, the primary merge beats hard deletion in all
192 matched units. Keeping the selected hot values at FP16 and quantizing cold
values to 2/3/4 bits uses 23.1%/28.6%/34.0% of FP16 value storage, with mean
local projected errors 5.71%/2.74%/1.25%, respectively.

## Current scope

- **Teacher evaluator:** The supported evaluator uses a paired full-cache
  reference and deep per-candidate probes. Its physical closed loop now carries
  the selected MLX state across periods and can rehydrate selected positions
  from a CPU cold backing store. P25 and untouched P26 both pass the mechanical
  loop gates. Temporal volatility passes an independent teacher-forced
  aggregate-risk gate but fails the later frozen free-generation NLL gate.
  P28 further supports exact-risk teacher superiority on a common token
  trajectory. P30 exposes a small-model task-quality boundary; P31 then shows
  that the per-token teacher substantially lowers Qwen3-8B trajectory KL while
  preserving all tested NIAH needles and remaining competitive on GovReport.
- **Mechanistic boundary:** Natural-amplitude full-vector reconstruction and a
  universal relative single-boundary shortcut did not pass their frozen gates.
- **Training-free boundary:** Direct retained-set search is implemented as
  a one-set, four-query shrinkage policy. It fails independent majority-unit,
  selective-trigger, and CVaR tail-risk gates and is not the default policy.
  A protected-attention rescue mechanism also fails its development tail gate,
  despite respecting its 4/8/16-token action-radius bounds.
  Four-step temporal attention volatility has free-generation quality,
  throughput, and memory evidence, but misses the overall NLL gate and its 16K
  attention-capture implementation has a large absolute memory footprint.
  Static token rarity removes that capture cost and replicates on 3/3 new RULER
  samples, but regresses independent GovReport quality. It is a retrieval-route
  candidate, not the repository default.
  Euclidean history sketches, diagonal/block scaling, the predeclared output-side
  VJP, and multi-boundary VJP also failed their primary gates. A post-hoc
  Rademacher VJP setting remains exploratory.
  P32 supplies the first successful cheap-controller screen. A2 temporal
  volatility is the lowest-KL cheap policy but loses GovReport quality. B2 proves
  that direct retained-set generation is feasible but has task-dependent tails.
  B3 adds state-dependent layer budgets, improves both aggregate KL and quality
  over B2, and is the current main-method candidate. A3 set-output perturbation
  collapses to Attention in all 640 decisions; A4 and the historical-prior B1
  remain diagnostic negative evidence.
- **Storage mechanisms:** Nearest-value merging and a 2/3/4-bit cold-value tier
  have positive local diagnostics and an audited merge bound. Keys, retrieval,
  and end-to-end merged/quantized replay are not yet evaluated.
- **System evaluation:** Matched-budget free-generation quality, decode
  throughput, peak memory, cache length, and hook failures are recorded for
  attention, geometry, position, and lexical signals. The strongest current
  systems result is token rarity versus latest attention: $1.252\times$
  throughput and $0.313\times$ peak memory on P21. Production-kernel parity and
  end-to-end request latency, including model loading and tokenization, remain
  outside the claim.

The current project stage and planned estimator, trigger, and closed-loop work
are tracked in [`configs/ccfa.yaml`](configs/ccfa.yaml).

The next decision-relevant experiment isolates why B3 works. With the token
utility and fixed global budget held constant, compare uniform, static-adaptive,
dynamic, layer-shuffled, and stale layer allocations on new validation data. If
dynamic allocation does not beat the matched static and misaligned controls, the
supported result is a layer-budget prior rather than state-conditioned control.
The following system gate removes the CPU backing store and enforces irreversible
eviction before any larger benchmark claim.

## Reproduce

StateKV targets Python 3.9+. Create an environment and install the canonical
package with:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -e .
```

Run the dependency-light research-path check and repository test suite:

```bash
PYTHONPYCACHEPREFIX=/tmp/statekv-pycache \
  .venv/bin/python scripts/smoke_test.py
.venv/bin/python -m pytest
```

Reproduce the retrospective training-free sketch gate without rerunning the
model:

```bash
.venv/bin/python scripts/analyze_training_free_sketch.py \
  --config configs/stages/training_free_sketch_config.yaml

.venv/bin/python scripts/analyze_metric_repair.py \
  --config configs/stages/training_free_metric_repair_config.yaml
```

The bounded model-backed pullback pilot is reproducible separately:

```bash
.venv/bin/python scripts/run_shared_jvp_pilot.py \
  --config configs/stages/shared_jvp_pilot_config.yaml
```

Run the newer training-free route pilots with cached local model weights:

```bash
HF_HUB_OFFLINE=1 .venv/bin/python scripts/run_vjp_routes_pilot.py \
  --config configs/stages/vjp_routes_stress_config.yaml
HF_HUB_OFFLINE=1 .venv/bin/python scripts/run_multiboundary_vjp_pilot.py \
  --config configs/stages/multiboundary_vjp_pilot_config.yaml
HF_HUB_OFFLINE=1 .venv/bin/python scripts/run_direct_coreset_pilot.py \
  --config configs/stages/direct_coreset_replication_config.yaml
HF_HUB_OFFLINE=1 .venv/bin/python scripts/run_direct_policy_replay.py \
  --config configs/stages/direct_policy_replay_config.yaml
.venv/bin/python scripts/analyze_direct_policy_replay.py \
  --run-dir results/temporal_cache_discovery/statekv_direct_policy_replay_p6_v1 \
  --baseline attention_mean_w1_shared \
  --primary contribution_mean_w4_shared
```

Run the independent multi-anchor gates and the runtime profile:

```bash
HF_HUB_OFFLINE=1 .venv/bin/python scripts/run_direct_policy_replay.py \
  --config configs/stages/direct_policy_independent_multianchor_config.yaml
HF_HUB_OFFLINE=1 .venv/bin/python scripts/run_direct_policy_replay.py \
  --config configs/stages/direct_policy_shrinkage_screen_config.yaml
HF_HUB_OFFLINE=1 .venv/bin/python scripts/run_direct_policy_replay.py \
  --config configs/stages/direct_policy_shrinkage_independent_config.yaml
HF_HUB_OFFLINE=1 .venv/bin/python scripts/profile_direct_policy_runtime.py \
  --config configs/stages/direct_policy_runtime_profile_config.yaml
HF_HUB_OFFLINE=1 .venv/bin/python scripts/run_direct_policy_trigger.py \
  --config configs/stages/direct_policy_trigger_screen_config.yaml
HF_HUB_OFFLINE=1 .venv/bin/python scripts/run_direct_policy_trigger.py \
  --config configs/stages/direct_policy_trigger_independent_config.yaml
HF_HUB_OFFLINE=1 .venv/bin/python scripts/run_direct_policy_replay.py \
  --config configs/stages/direct_policy_tail_risk_independent_config.yaml
HF_HUB_OFFLINE=1 .venv/bin/python scripts/run_direct_policy_replay.py \
  --config configs/stages/direct_policy_protected_rescue_screen_config.yaml
.venv/bin/python scripts/analyze_protected_rescue_screen.py \
  --config configs/stages/direct_policy_protected_rescue_screen_config.yaml
HF_HUB_OFFLINE=1 .venv/bin/python scripts/run_direct_policy_replay.py \
  --config configs/stages/direct_policy_signal_family_screen_config.yaml
.venv/bin/python scripts/analyze_signal_family_screen.py \
  --config configs/stages/direct_policy_signal_family_screen_config.yaml
HF_HUB_OFFLINE=1 .venv/bin/python scripts/run_direct_policy_replay.py \
  --config configs/stages/direct_policy_temporal_volatility_independent_config.yaml
.venv/bin/python scripts/profile_temporal_volatility_runtime.py \
  --config configs/stages/temporal_volatility_runtime_profile_config.yaml
.venv/bin/python scripts/analyze_temporal_volatility_freegen.py \
  --config configs/stages/temporal_volatility_freegen_protocol.yaml
.venv/bin/python scripts/analyze_attention_free_geometry_screen.py \
  --config configs/stages/attention_free_geometry_screen_protocol.yaml
.venv/bin/python scripts/analyze_static_lexical_screen.py \
  --config configs/stages/static_lexical_screen_protocol.yaml
.venv/bin/python scripts/analyze_token_rarity_replication.py \
  --config configs/stages/token_rarity_replication_protocol.yaml
```

Run the risk-consistent proxy audits with cached local weights:

```bash
HF_HUB_OFFLINE=1 .venv/bin/python scripts/run_proxy_alignment.py \
  --config configs/stages/proxy_alignment_protocol.yaml
HF_HUB_OFFLINE=1 .venv/bin/python scripts/run_direct_policy_replay.py \
  --config configs/stages/proxy_alignment_independent_source_protocol.yaml
HF_HUB_OFFLINE=1 .venv/bin/python scripts/run_proxy_alignment.py \
  --config configs/stages/proxy_alignment_independent_protocol.yaml
HF_HUB_OFFLINE=1 .venv/bin/python scripts/run_proxy_alignment.py \
  --config configs/stages/proxy_alignment_output_aware_protocol.yaml
```

Run the expensive physical-oracle closed loop and regenerate its summaries:

```bash
HF_HUB_OFFLINE=1 .venv/bin/python scripts/run_oracle_closed_loop.py \
  --config configs/stages/oracle_closed_loop_protocol.yaml
.venv/bin/python scripts/analyze_oracle_closed_loop.py \
  --run-dir results/temporal_cache_discovery/statekv_physical_oracle_closed_loop_p25_v1

HF_HUB_OFFLINE=1 .venv/bin/python scripts/run_oracle_closed_loop.py \
  --config configs/stages/oracle_closed_loop_independent_protocol.yaml
.venv/bin/python scripts/analyze_oracle_closed_loop.py \
  --run-dir results/temporal_cache_discovery/statekv_physical_oracle_closed_loop_independent_p26_v1
```

These commands intentionally run every proposal under a full-reference teacher;
they are reproducibility commands for the oracle evidence, not deployment
benchmarks.

Run the fixed-policy comparisons and free-generation horizon experiments:

```bash
HF_HUB_OFFLINE=1 .venv/bin/python scripts/run_oracle_policy_comparison.py \
  --config configs/stages/oracle_policy_comparison_independent_protocol.yaml

HF_HUB_OFFLINE=1 .venv/bin/python scripts/run_oracle_policy_freegen.py \
  --config configs/stages/oracle_policy_freegen_h1_protocol.yaml
HF_HUB_OFFLINE=1 .venv/bin/python scripts/run_oracle_policy_freegen.py \
  --config configs/stages/oracle_policy_freegen_h4_protocol.yaml
HF_HUB_OFFLINE=1 .venv/bin/python scripts/run_oracle_policy_freegen.py \
  --config configs/stages/oracle_policy_freegen_protocol.yaml
HF_HUB_OFFLINE=1 .venv/bin/python scripts/run_oracle_policy_freegen.py \
  --config configs/stages/oracle_policy_freegen_independent_protocol.yaml

.venv/bin/python scripts/analyze_oracle_policy_comparison.py
```

The smoke test covers configuration loading, cache budgeting, baseline and
candidate generation, functional measurement, output-risk utilities, refresh
scheduling, the exact retained-set action, and the stable state/risk/decision
API.

Model-scale runs require local model weights, dataset access, and an appropriate
MPS or CUDA environment. Install and test a backend harness from its own project
root:

```bash
.venv/bin/python -m pip install -e benchmarks/torch
.venv/bin/python -m pip install -e benchmarks/mlx

(cd benchmarks/torch && ../../.venv/bin/python -m pytest tests)
(cd benchmarks/mlx && ../../.venv/bin/python -m pytest tests)
```

## Core Python API

The backend-independent core exposes the state, action, risk, and decision
objects used in the formulation:

```python
from statekv import (
    functional_history_state,
    select_lowest_risk,
    set_level_attention_delta,
    state_conditioned_quadratic_risk,
)

state = functional_history_state(history_boundary, reference_boundary)
boundary_delta = set_level_attention_delta(
    attention, values, retained_positions
)
risk = state_conditioned_quadratic_risk(
    reference_logits, state_logits, candidate_delta_logits
)
decision = select_lowest_risk(
    {"candidate-a": float(risk_a), "candidate-b": float(risk_b)}
)
```

`set_level_attention_delta` implements the exact fixed-operating-point action.
Finite downstream transport remains model-aware and is exposed through
`statekv.core.midpoint_path_response`.

## Repository map

| Path | Role |
|---|---|
| [`statekv/core/`](statekv/core/) | Stable backend-independent state, action, risk, and decision contracts. |
| [`statekv/`](statekv/) | Reusable model-aware probes, features, instrumentation, and analysis logic. |
| [`experiments/`](experiments/) | Frozen phase protocols, manifests, and structured results. |
| [`benchmarks/`](benchmarks/) | PyTorch/CUDA and MLX execution harnesses and baselines. |
| [`configs/`](configs/) | Discovery, staged, and frozen experiment configurations. |
| [`analysis/`](analysis/) | Derived tables, figures, and publication analysis. |
| [`tests/`](tests/) | Core contract, protocol, and frozen-evidence tests. |
| [`assets/`](assets/) | Architecture source and rendered deliverables. |

Stable contracts live in `statekv.core`; backend-specific execution and frozen
experiment protocols depend on those contracts, not the reverse.

## Citation and license

The StateKV paper is in preparation and has no archival citation identifier. For
now, cite the repository commit and the relevant frozen experiment manifest.

See [LICENSE](LICENSE) for licensing terms.
