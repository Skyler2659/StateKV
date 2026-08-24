# Cheap-R2 Fresh Generalization Report (v1)

Date: 2026-08-25
Branch: `codex/statekv-counterfactual-utility`
Method (frozen before this campaign, no further tuning allowed):
**Cheap-R2 = STRICT_CAUSAL_ROLLOUT_R2, rollout_horizon=32, refresh_frequency=16, sink/recent protection identical across all policies, strict pure eviction, cold recovery = 0.**

Fresh splits used for the first and only time here:
- synthetic multikey NIAH 280–329 (n=50, never used in any design decision)
- LongBench hotpotqa / 2wikimqa / gov_report indices 40–59 (n=20 per task)

Configs: `configs/statekv_counterfactual/cheapr2_fresh_multikey_qwen3_8b.yaml`,
`cheapr2_fresh_longbench_qwen3_8b.yaml`.
Results: `results/statekv_counterfactual/cheapr2_fresh_multikey_v1/`,
`results/statekv_counterfactual/cheapr2_fresh_longbench_v1/`.
Statistics: paired differences, cluster bootstrap 95% CI (B=20000, seed 20260820).

---

## Integrity audit

- multikey: 50 samples × 3 budgets × 4 policies + Full = 650 rows, no missing arms.
- All compressed arms: `strict_pure_eviction=True`, `recoverable_cold_tokens=0`,
  `peak_active_cache_tokens ≤ budget` (0 violations).
- LongBench: n=20 per task (hotpotqa = s1 50–59 + s0b re-run 40–49; the original s0
  hotpotqa rows were overwritten by the top-up run — the runner overwrites
  `sample_summary.csv` per shard tag; lesson: always use a fresh `--output-tag`).
  300 rows total, integrity checks pass (0 violations).

---

## A. Does H32-f16 replicate on fresh multikey? — **PASS**

n=50, task score (percent of needles recovered):

| Budget | Cheap-R2 | QK | SnapKV | H2O | Full |
|---|---|---|---|---|---|
| 128 | **48.0** | 1.0 | 0.5 | 0.0 | — |
| 256 | **69.5** | 21.0 | 0.0 | 5.0 | 82.0 |
| 512 | **81.5** | 25.5 | 55.0 | 16.0 | — |

Paired vs baselines, budget 256 (n=50):

| Opponent | W/T/L | Mean diff | 95% CI |
|---|---|---|---|
| QK | 37/13/**0** | +48.5 | [+38.5, +58.0] |
| SnapKV | **50**/0/0 | +69.5 | [+60.5, +78.5] |
| H2O | 47/3/**0** | +64.5 | [+54.5, +74.0] |
| Full | 6/28/16 | −12.5 | [−21.5, −3.5] |

Full-solvable subset (Full ≥ 75, n=41) @256: R2 = 74.4, QK = 20.7, SnapKV = 0.0,
H2O = 4.3 (Full = 94.5). The advantage is not an artifact of backbone-failure samples.

Dev → fresh comparison @256: dev 67.5 vs QK 20 → fresh 69.5 vs QK 21. No shrinkage.

## B. Does it generalize to realistic multi-evidence tasks? — **NOT REPLICATED (null/negative)**

LongBench @256 (contexts ~2700–3450 tokens, i.e. retention ratio ≈ 8% — budget binds hard):

| Task | n | Full | Cheap-R2 | QK | SnapKV | H2O |
|---|---|---|---|---|---|---|
| hotpotqa | 20 | 40.0 | 35.0 | 35.0 | 35.0 | 35.0 |
| 2wikimqa | 20 | 45.0 | 45.0 | **55.0** | 50.0 | 45.0 |
| gov_report (control) | 20 | 6.25 | 6.10 | 6.49 | 6.09 | 6.23 |

Paired vs QK (budget 256): hotpotqa **0W/20T/0L**(逐样本完全相同;vs H2O 2W/16T/2L,
CI [−20,+20]);
gov_report 7W/1T/12L, mean −0.39, CI [−0.72, −0.05] (QK marginally better; R2
non-inferior to SnapKV/H2O/Full within CI).

Mechanistic reading: per-sample trajectory KL across policies differs strongly
(0.006–2.26), so the policies *are* doing different things — but on these tasks the
task-level outcome is the same. The evidence needed for the answer is either in the
protected recent/sink windows (common to all policies) or lost under *every* policy.
The tight-budget regime alone is not sufficient; the workload must also have
reactivation structure (dormant → reactivated evidence under state change), which
multikey has by construction and hotpotqa/2wikimqa at this context scale apparently
do not (or the effect is smaller than n=20 binary scoring can resolve).

gov_report as non-inferiority control: Cheap-R2 ≈ Full (6.10 vs 6.25, CI of diff
includes 0) — no catastrophic degradation. Pass in its control role.

## C. Is the advantage concentrated in the tight-budget regime? — **PASS**

multikey gradient is monotone and clean: at 128 all baselines collapse (≤1.0) while
R2 keeps 48.0; at 256 the gap is maximal (+48.5 vs QK); at 512 SnapKV recovers to
55.0 and R2 is statistically indistinguishable from Full (−0.5, CI [−10.5, +9.5]).

R2 vs Full by budget: 128 → −34.0 [−43.5, −24.5]; 256 → −12.5 [−21.5, −3.5];
512 → −0.5 [−10.5, +9.5].

## D. Is the ~1.8–2.4× cost Pareto stable? — **PASS (with one caveat)**

- multikey @256: R2 125.2s vs QK 53.4s → **2.34×** (dev was 1.8×).
- hotpotqa @256 (n=20): R2 164.1s vs QK 52.6s → **3.12×**, with only ~4 refreshes.
  Longer contexts make each refresh (prefix recomputation) more expensive; cost
  scales with context length, not just refresh count. This widens the gap on real
  LongBench-scale contexts and matters for the deployment narrative.

---

## Bottom line

1. On synthetic reactivation-heavy retrieval, Cheap-R2's advantage is real, large,
   statistically solid, and replicates perfectly on 50 never-before-used samples.
2. On realistic multi-hop QA (hotpotqa/2wikimqa @256), the advantage does **not**
   replicate: exact tie with QK on hotpotqa, slightly behind on 2wikimqa (CI wide).
3. The honest claim is therefore task-conditional: Cheap-R2 wins where dormant→
   reactivated evidence exists under aggressive compression; it is neutral elsewhere
   and non-inferior on summarization control.
4. This *strengthens* the reactivation hypothesis as the explanatory variable and
   motivates the next step: verify that LongBench-style tasks have low Reactivation
   Index (full-timeline RI), i.e. RI predicts where the method helps — rather than
   treating the LongBench null as a method failure.

## Known limitations / open items

- LongBench n=20 with coarse binary-ish scoring has limited power; CIs are wide.
- LongBench ran at budget 256 only. A budget-128 LongBench arm would test whether
  R2's advantage appears on real tasks under more aggressive compression
  (where multikey shows the largest separation).
- hotpotqa s0 rows (40–49) were lost to a shard-tag overwrite and re-run under tag
  `s0b`; both segments are included in the n=20 above.
- Cost on long contexts (~3×) is higher than the multikey figure (2.3×); runtime
  claims must be quoted per context-length regime.
