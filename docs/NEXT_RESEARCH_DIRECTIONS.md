# Next research directions — from StateKV's experimental residue

Compiled 2026-08-10. Scope rule: everything below is derived from what this
repository's experiments actually showed. No idea here requires believing
the original StateKV deployability hypothesis — it is falsified (see
`FAILURE_ANALYSIS.md`). Ordered by strength of empirical support.

---

## 1. What is already closed (do not re-enter without new evidence)

- Cheap or expensive *scorers* that beat per-step exact QK routing at
  quality-valid operating points (Gate 0–2, R0, open search, extval).
- V-side routing residuals given QK; page-metadata approximations;
  observation-window scoring; per-head splits; selective refresh triggers
  on high-coverage substrates; dynamic layer budgets; the listed
  training-free estimators (TF-P0/P1, P2, P3, P5, Rademacher VJP).
- Full list with evidence: `analysis/statekv_gate_retrospective_catalog.md`
  (41 entries) and `docs/FINDINGS.md` §D.

## 2. Empirical gaps worth building the next project on

### 2.1 The KL→task-harm bridge (strongest gap)
Two metric families drove every gate dispute and were never calibrated
against each other: distribution fidelity (exact KL / ΔNLL / tail
statistics) and task scores (NIAH / ROUGE-L / containment). The retest
panel shows task scores saturating across a 100× KL range (FINDINGS A10),
and every task collapse was preceded by a KL excursion. **The missing
experiment is a dose-response map**: controlled degradation injections at
measured KL levels, measuring at what KL each task family actually breaks,
on samples large enough to see it. All instrumentation exists
(`oracle_policy_freegen` step_rows carry per-token exact KL and ΔNLL).
This turns the project's most-used proxy into a calibrated instrument —
valuable to any KV-compression paper regardless of method.

### 2.2 Coverage × cadence as a deployment-cost frontier
The cliff (FINDINGS A7) is mapped at three budgets × three cadences × two
context lengths, and its controlling variable is the absolute core budget.
What does not exist is the inverse view: **given a quality target, the
minimal refresh cadence per memory budget** — i.e. a cost model for
exact-routing deployment (transfer volume telemetry already exists in the
R0/extval runs). This is a benchmark-style contribution with immediate
systems relevance: it tells hierarchical-KV systems when cheap refresh is
safe.

### 2.3 The low-coverage quality-valid operating point
Staleness (the refresh phenomenon) demonstrably exists at coverage ≈0.70
(shared-mask, Era-1) and demonstrably vanishes at coverage ≈0.995
(per-layer, Era-2), but every low-coverage point tested so far is
quality-invalid (degeneration). **Is there a substrate — model, mask
semantics, budget — that is simultaneously quality-valid and stale?** If
yes, refresh triggers become testable again on a live premise (R2a/R2b
label machinery is ready). If no such point exists, that is itself a
publishable boundary result. Cheap to explore: the replay + freegen
harnesses both parameterize mask semantics and budget.

### 2.4 Recoverable (CPU-offload) working-set control
The repository's recoverable machinery (KVBackingStore, cold recovery,
`qk_tiered_v` value-tiering) is validated and the retest showed tiered-V
is near-lossless at matched budget (FINDINGS A9). The unexplored question
under recoverable semantics is not *what to keep* (qk_pool settles it at
h1) but **what to fetch and when**: recovery scheduling, transfer-cost
awareness, and tiered-precision policies under a PCIe/memory-bandwidth
cost model. This direction inherits the strongest surviving infrastructure
and has no falsified predecessor inside the repo.

### 2.5 Cross-session / prefix reuse (HF5)
Deferred by the open search for lack of a harness; nothing is known. The
backing-store machinery is the natural substrate. Lowest evidence of any
item here — listed because it is the only completely untouched regime the
search identified.

## 3. Reusable infrastructure (what the next project gets for free)

- **Evaluation stack**: exact same-input KL/ΔNLL per token
  (`trajectory_model.exact_distribution_metrics`), paired full-cache
  reference generation, recoverable closed loop with per-cycle telemetry
  (`oracle_policy_freegen._run_free_policy`), all-pairs paired bootstrap
  (`retest_freegen`).
- **Policy panel**: attention / SnapKV / H2O / uniform / qk_pool /
  quest_like / qk_obswin / qk_tiered_v / token_rarity / A1–B3 cheap
  controllers — one config key each, one process per panel
  (`retest_freegen.py`).
- **Baselines library**: 79-method eviction registry in `benchmarks/mlx`
  (19 exercised by StateKV-era work).
- **Task battery**: synthetic NIAH (any offset/length), multikey NIAH,
  LongBench GovReport with fallback, reasoning-with-distractors —
  `statekv/tasks.py`.
- **Frozen mechanistic phases**: `experiments/` with registries and tests.
- **Methodological assets**: the preregistered-protocol habit, the
  gate-retrospective catalog format, and the retest program's no-gate
  continuous-reporting contract (`analysis/statekv_retest_report.md`).

## 4. Methodological lessons the next project should inherit

1. Development-set positives are hypotheses, not results — the dev→
   independent reversal happened at least four times (P20→P21, P22→P23b,
   Rademacher VJP, arguable P16→P18).
2. Oracles must be audited for forbidden information access before their
   headroom is interpreted (the P31 artifact).
3. Joint gates with veto power need either calibrated metrics or
   non-inferiority framing; uncalibrated KL vetoes produced both false
   rejections (P9, qk_tiered_v) and correct ones (token rarity cross-task).
4. Substrate statements belong in every claim: two eras behaved oppositely
   on staleness; every strong finding above names its substrate.
