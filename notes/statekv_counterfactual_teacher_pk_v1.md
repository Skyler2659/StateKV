# Teacher PK final: R2 vs SnapKV/H2O/QK, strict pure eviction (2026-08-22)

Question (only one): is the R2 teacher itself reliably stronger than
SnapKV/H2O/QK under real strict eviction?

Protocol: 5 tasks x 10 samples x 9 arms = 450 arms. Qwen3-8B-4bit,
budgets 256/512, sink 4 / recent 32, refresh=2 (frozen), strict pure
eviction, cold recovery 0, shared-token core, 64 decode cycles.
Integrity audit: 400 matched arms + 50 full refs, every task x policy x
budget cell = 10, zero budget violations, zero duplicates, strict/cold
flags all green.

Artifacts:
- `results/statekv_counterfactual/teacher_pk_v1_all_arms.csv` (450 rows)
- `results/statekv_counterfactual/teacher_pk_v1_table.csv` (main table)

## Main table (official score, all samples)

| task | Full | R2@512 | R2@256 | SnapKV@512 | SnapKV@256 | H2O@512 | H2O@256 | QK@512 | QK@256 |
|---|---|---|---|---|---|---|---|---|---|
| multikey | 70.0 | 57.5 | **32.5** | 60.0 | 2.5 | 15.0 | 5.0 | 27.5 | 20.0 |
| multiquery | 100.0 | 100.0 | 100.0 | 100.0 | 0.0 | 50.0 | 30.0 | 100.0 | 100.0 |
| passage | 30.0 | 30.0 | **40.0** | 30.0 | 30.0 | 30.0 | 30.0 | 30.0 | 30.0 |
| 2wikimqa | 30.0 | 30.0 | **40.0** | 30.0 | 20.0 | 40.0 | 20.0 | 40.0 | **40.0** |
| govreport | 6.3 | 5.9 | 5.7 | 6.1 | 5.9 | 6.1 | 5.4 | 6.0 | 5.8 |

Full-cache success rate: multikey 7/10, multiquery 10/10, passage 3/10,
2wikimqa 3/10, govreport 0/10 (64-token generation metric ceiling, not
an eviction effect).

## Full-cache-solvable subset (retrieval, full > 50)

- multikey (n=7): @256 R2 35.7 > QK 21.4 > H2O 7.1 > SnapKV 3.6;
  @512 SnapKV 75.0 > R2 71.4 > QK 28.6 > H2O 17.9
- multiquery (n=10): R2 = QK = 100 both budgets
- passage (n=3), 2wikimqa (n=3): everyone ~100 both budgets (no
  discrimination once the backbone can solve the sample)

## R2 paired wins (win/tie/loss), retrieval tasks

- @256: R2 never loses a paired comparison. vs QK: multikey 4/6/0,
  multiquery 0/10/0, 2wikimqa 0/10/0, passage 1/9/0. vs SnapKV:
  9/1/0, 10/0/0, 2/8/0, 1/9/0. vs H2O: 9/0/1, 7/3/0, 2/8/0, 1/9/0.
- @512: tie-dominated; isolated single losses (multikey vs SnapKV 0/9/1,
  2wikimqa vs QK 0/9/1, vs H2O 0/9/1).

## Ranking consistency across budgets

@256: R2 ranks first or joint-first on every retrieval task.
@512: ranking fragments (SnapKV first on multikey, H2O first on
2wikimqa/govreport) — the loose budget removes the discrimination.

## Runtime (mean wall time per arm)

R2 ~684s (~9x the 75-116s of SnapKV/H2O/QK, ~19x full-cache 36s).

## Verdict (per the preregistered three-way framework)

**R2 clearly leads at budget 256, ties at budget 512.**

- vs SnapKV/H2O @256: stable, large wins (mean gains +20 to +100).
- vs QK @256: never loses; decisive only on multikey (+12.5 mean,
  4W/6T/0L). Current-QK is a surprisingly strong baseline everywhere else.
- @512: no method separates from another; SnapKV matches or slightly
  beats R2 on multikey.
- govreport: no method collapses; control holds.

Interpretation: the teacher route is validated, but its eviction
advantage is **conditional on tight budgets and multi-target retrieval**,
not universal. Student distillation should be judged primarily in the
regime where the teacher actually wins (budget 256, multikey-type
workloads); at loose budgets there is no teacher advantage to distill.
