# StateKV Reproducibility Audit

Chain audit — **finding → config → script → raw result** — for every entry in
[`docs/experiments/EXPERIMENT_REGISTRY.md`](experiments/EXPERIMENT_REGISTRY.md).
Compiled 2026-08-10; every path below was existence-checked on disk
(`test -f` / directory listing), not taken on trust.

## Status vocabulary

- `REPRODUCIBLE` — full chain intact on disk and the exact command is documented.
- `PARTIALLY REPRODUCIBLE` — chain intact, but the command is not written down
  (it is uniformly derivable as
  `HF_HUB_OFFLINE=1 .venv/bin/python scripts/<runner>.py --config <config>`),
  or the run additionally needs non-repo local resources (noted which).
- `RESULT EXISTS BUT CONFIG MISSING` — artifacts survive without their config.
- `SCRIPT EXISTS BUT RAW RESULT MISSING` — config/script survive; no run.
- `UNRECOVERABLE` — chain broken beyond reconstruction.
- `REQUIRES RERUN` — the stored artifacts are not usable evidence as-is.

## Environment (applies to every model-scale row)

- Python env: `.venv` (Python 3.9), package installed editable
  (`pip install -e .`); `benchmarks/mlx` and `benchmarks/torch` installed
  editable from their own roots so both sit on `sys.path`.
- Model runs require the local HF cache with `HF_HUB_OFFLINE=1`:
  `mlx-community/Qwen3-8B-4bit` (revision `545dc425…`) for Era 2 and
  `mlx-community/Qwen2.5-1.5B-Instruct-4bit` for Era 0/1, plus the LongBench
  dataset cache. Hardware: Apple-silicon MPS (MLX).
- README "Reproduce" section documents commands only through P28–P30. For
  P31–P35, the gate/closure program, and the retests the commands below are
  reconstructed from run-dir `config.yaml` copies and the runner CLIs; where a
  queue script in `tmp/` records the exact invocation, that is cited.
- Sanity: `PYTHONPYCACHEPREFIX=/tmp/statekv-pycache .venv/bin/python scripts/smoke_test.py`
  and `.venv/bin/python -m pytest` are documented in README and pass
  independently of any run artifact.

Assessment rule used below: a row is `REPRODUCIBLE` when its command is written
down (README, phase manifest, or a `tmp/` queue script); the local-weights /
dataset requirement is part of the documented environment and is noted, not
treated as a chain break. Rows whose commands are only derivable are
`PARTIALLY REPRODUCIBLE` for that reason alone.

---

## 0. Era 0 — frozen phases

Chains: phase manifest + `configs/frozen/*` or phase-local configs +
`experiments/*/scripts/*.py` + stored `results/`. All nine verified intact.

| Entry | Chain | Status | Notes |
|---|---|---|---|
| predictive_closure | `configs/{primary,p0_formal_4bit}.yaml` → `scripts/run_p0_formal.py` → phase results | REPRODUCIBLE | Entry points recorded in `experiments/frozen_registry.yaml` and the phase manifest; needs the frozen Qwen2.5-1.5B MLX weights |
| local_truncated_jacobian | `configs/local_primary.yaml` → `run_l0..l3_formal.py` → results | REPRODUCIBLE | Same |
| p0_v2_fixed_boundary | `configs/frozen/p0_v2_config.yaml` → `run_p0_v2.py` → `results/p0_v2_summary.json` ✓ | REPRODUCIBLE | Historical root config name is provenance-only; runtime uses `configs/frozen/` |
| p1_state_conditioned | `configs/frozen/p1_state_conditioned_config.yaml` → `run_p1.py` → `results/state_operating_point_summary.json` ✓ | REPRODUCIBLE | |
| p2_state_local_risk | `configs/frozen/p2_state_local_config.yaml` → `run_p2.py` → results | REPRODUCIBLE | |
| p2_recovery | `r0/r1/r3/r4 *_config.yaml` → `run_r0_r1.py`, `run_r3.py`, `analyze_r4.py` → `r4_scalar_decision_risk/results/{evaluation,replication}/analysis_summary.json` ✓ | REPRODUCIBLE | R2 preregistered but never run — nothing to reproduce (not a gap) |
| p3_decision_validity | `p3_config.yaml` → `run_p3_trajectory.py` → results | REPRODUCIBLE | |
| p3_physical_recovery | `p3pr_config.yaml` → `run_p3pr.py` → results | REPRODUCIBLE | |
| p3pr_generalization | `p3pr_generalization_config.yaml` → `run_generalization.py` → `results/analysis/analysis_summary.json` ✓ | REPRODUCIBLE | Known metadata omission in `run_summary` models field. Note: `experiments/frozen_registry.yaml` says this is "documented in evidence.md", but no `evidence.md` exists under `experiments/p3pr_generalization/` — a minor documentation break in the frozen registry itself |

## 1. Training-free estimators P0–P5

| Entry | Chain | Status | Notes |
|---|---|---|---|
| P0 tf_sketch | `configs/stages/training_free_sketch_config.yaml` → `scripts/analyze_training_free_sketch.py` → `statekv_tf_sketch_p0_v1/{summary.json,metrics.csv}` ✓ | REPRODUCIBLE | Command in README; retrospective over 24 stored trajectories — no model run needed |
| P1 metric_repair | `training_free_metric_repair_config.yaml` → `analyze_metric_repair.py` → `statekv_tf_metric_repair_p1_v1/` ✓ | REPRODUCIBLE | Command in README; no model run |
| P2 shared_jvp | `shared_jvp_pilot_config.yaml` → `run_shared_jvp_pilot.py` → `statekv_shared_jvp_pilot_p2_v1/` ✓ | REPRODUCIBLE | Command in README; model-backed (local weights) |
| P3 vjp pilot | `vjp_routes_pilot_config.yaml` → `run_vjp_routes_pilot.py` → `statekv_vjp_routes_p3_v1/` ✓ | REPRODUCIBLE | Command pattern in README (stress config shown); local weights |
| P3 vjp stress | `vjp_routes_stress_config.yaml` → `run_vjp_routes_pilot.py` → `statekv_vjp_routes_p3_stress_v1/` ✓ | REPRODUCIBLE | Command in README verbatim |
| P4 coreset pilot | `direct_coreset_pilot_config.yaml` → `run_direct_coreset_pilot.py` → `statekv_direct_coreset_p4_v1/` ✓ | PARTIALLY REPRODUCIBLE | README documents only the replication config; pilot command same runner, one-word config swap |
| P4 coreset replication | `direct_coreset_replication_config.yaml` → `run_direct_coreset_pilot.py` → `statekv_direct_coreset_p4_replication_v1/` ✓ | REPRODUCIBLE | Command in README |
| P5 multiboundary | `multiboundary_vjp_pilot_config.yaml` → `run_multiboundary_vjp_pilot.py` → `statekv_post_multiboundary_vjp_p5_v1/` ✓ | REPRODUCIBLE | Command in README |
| multi-boundary distillation (superseded) | — | — | Never run; nothing to reproduce |

## 2. Direct-policy replay line P6–P24

All configs under `configs/stages/`; all result dirs verified with
`summary.json` + `config.yaml`. Every command in this block except where noted
appears verbatim in the README Reproduce section. All need local
Qwen2.5-1.5B-Instruct-4bit weights.

| Entry | Config → script | Status | Notes |
|---|---|---|---|
| P6 replay | `direct_policy_replay_config.yaml` → `run_direct_policy_replay.py` + `analyze_direct_policy_replay.py` | REPRODUCIBLE | Both commands in README |
| P7 multianchor | `direct_policy_independent_multianchor_config.yaml` → `run_direct_policy_replay.py` | REPRODUCIBLE | |
| P8 shrinkage screen | `direct_policy_shrinkage_screen_config.yaml` → `run_direct_policy_replay.py` | REPRODUCIBLE | |
| P9 shrinkage indep | `direct_policy_shrinkage_independent_config.yaml` → `run_direct_policy_replay.py` | REPRODUCIBLE | |
| P10 runtime profile | `direct_policy_runtime_profile_config.yaml` → `profile_direct_policy_runtime.py` | REPRODUCIBLE | Only the `_v3` run dir exists; the config is the current one. Earlier-profile artifacts are not stored — rerunning produces a fresh profile, not a byte-identical `_v3` |
| P11 trigger screen | `direct_policy_trigger_screen_config.yaml` → `run_direct_policy_trigger.py` | REPRODUCIBLE | |
| P12 trigger indep | `direct_policy_trigger_independent_config.yaml` → `run_direct_policy_trigger.py` | REPRODUCIBLE | Selection hashes match the P9 source replay; requires P9 artifacts present |
| P13 tail risk | `direct_policy_tail_risk_independent_config.yaml` → `run_direct_policy_replay.py` | REPRODUCIBLE | |
| P14 protected rescue | `direct_policy_protected_rescue_screen_config.yaml` → `run_direct_policy_replay.py` + `analyze_protected_rescue_screen.py` | REPRODUCIBLE | Both commands in README |
| P15 signal family | `direct_policy_signal_family_screen_config.yaml` → `run_direct_policy_replay.py` + `analyze_signal_family_screen.py` | REPRODUCIBLE | |
| P16 TV independent | `direct_policy_temporal_volatility_independent_config.yaml` → `run_direct_policy_replay.py` | REPRODUCIBLE | Spot-checked chain intact |
| P17 TV runtime | `temporal_volatility_runtime_profile_config.yaml` → `profile_temporal_volatility_runtime.py` | REPRODUCIBLE | CPU-only arithmetic benchmark |
| P18 TV freegen | `temporal_volatility_freegen_protocol.yaml` → `analyze_temporal_volatility_freegen.py` | PARTIALLY REPRODUCIBLE | Analysis command in README; the underlying harness workloads ran via `benchmarks/mlx/scripts/run_benchmark.py` with `benchmarks/mlx/configs/experiments/statekv/qwen25_15b_*_temporal_volatility_p18{a,b}.yaml` — needs local weights + LongBench/RULER data; the bare `v1` harness dirs' config mapping is not recorded |
| P19 geometry screen | `attention_free_geometry_screen_protocol.yaml` → `analyze_attention_free_geometry_screen.py` | PARTIALLY REPRODUCIBLE | Same harness situation (`..._geometry_screen_p19{a,b}.yaml`; `p19a_v1`/`p19b_v1` dirs exist) |
| P20 lexical screen | `static_lexical_screen_protocol.yaml` → `analyze_static_lexical_screen.py` | PARTIALLY REPRODUCIBLE | Harness configs `..._static_lexical_p20{a,b}.yaml`; `p20a_v1`/`p20b_v1` exist; the two `p20a_integration_failed_*` dirs are REQUIRES RERUN traces (see appendix) |
| P21 token rarity | `token_rarity_replication_protocol.yaml` → `analyze_token_rarity_replication.py` | PARTIALLY REPRODUCIBLE | Harness configs `..._token_rarity_replication_p21{a,b}.yaml`; `p21a_v1`/`p21b_v1` exist; analysis command in README |
| P22 proxy alignment | `proxy_alignment_protocol.yaml` → `run_proxy_alignment.py` | REPRODUCIBLE | |
| P23a source | `proxy_alignment_independent_source_protocol.yaml` → `run_direct_policy_replay.py` | REPRODUCIBLE | |
| P23b independent | `proxy_alignment_independent_protocol.yaml` → `run_proxy_alignment.py` | REPRODUCIBLE | Requires P23a source run present |
| P24 output-aware | `proxy_alignment_output_aware_protocol.yaml` → `run_proxy_alignment.py` | REPRODUCIBLE | |

## 3. Oracle / cheap-controller line P25–P35

| Entry | Config → script | Status | Notes |
|---|---|---|---|
| P25 closed loop | `oracle_closed_loop_protocol.yaml` → `run_oracle_closed_loop.py` + `analyze_oracle_closed_loop.py` | REPRODUCIBLE | Commands in README |
| P26 closed loop indep | `oracle_closed_loop_independent_protocol.yaml` → same runner | REPRODUCIBLE | |
| P27 comparison | `oracle_policy_comparison_protocol.yaml` → `run_oracle_policy_comparison.py` | REPRODUCIBLE | README shows the independent config; dev config is the same runner |
| P28 comparison indep | `oracle_policy_comparison_independent_protocol.yaml` → `run_oracle_policy_comparison.py` + `analyze_oracle_policy_comparison.py` | REPRODUCIBLE | Commands in README; spot-checked chain intact |
| P29 freegen H=8 | `oracle_policy_freegen_protocol.yaml` → `run_oracle_policy_freegen.py` | REPRODUCIBLE | Commands in README (all four horizon configs listed) |
| P29b H=1 | `oracle_policy_freegen_h1_protocol.yaml` → same | REPRODUCIBLE | |
| P29c H=4 | `oracle_policy_freegen_h4_protocol.yaml` → same | REPRODUCIBLE | |
| P30 freegen indep | `oracle_policy_freegen_independent_protocol.yaml` → same | REPRODUCIBLE | Last README-documented command block |
| P31 Qwen3-8B | `oracle_policy_freegen_qwen3_8b_n10_protocol.yaml` → `run_oracle_policy_freegen.py` | PARTIALLY REPRODUCIBLE | Command not in README; derivable. Needs Qwen3-8B-4bit local cache. Spot-checked chain intact |
| P32 cheap | `cheap_policy_freegen_qwen3_8b_n10_protocol.yaml` → `run_cheap_policy_freegen.py` | PARTIALLY REPRODUCIBLE | Command not documented; derivable. Spot-checked chain intact |
| P33 calibration | `statekv_p1_p3_gates_qwen3_8b.yaml` → `run_statekv_gates.py calibration` | PARTIALLY REPRODUCIBLE | Stage mapping verified from the run-dir `config.yaml` copy (`experiment_name: statekv_p1_p3_gates_qwen3_8b`) and the runner's stage choices |
| P34 dynamic budget | same config → `run_statekv_gates.py p1` | PARTIALLY REPRODUCIBLE | Same |
| P35 pure eviction | same config → `run_statekv_gates.py p2` | PARTIALLY REPRODUCIBLE | Same; runtime fragments in `statekv_p1_p3_gates_qwen3_8b_seed20260808_v1_{calibration,p1,p2}` are scratch, not the curated results |
| P36 tail telemetry | — | — | Never run (superseded); nothing to reproduce |

## 4. Gate / closure program

All configs exist under `configs/stages/`; all canonical run dirs verified.
Runners: `scripts/run_statekv_gates.py` (stages `calibration|p1|p2|p2-profile|p3|r2a-labels|r2b-gate|teacher-gate|ladder`),
`scripts/run_oracle_policy_freegen.py` (recoverable / tier / corner / extval
freegen modes), `scripts/run_qkv_decomposition.py` (decomposition battery,
headwise probe, extval decomposition). Exact invocations for the open-search,
corner, and extval runs are recorded in `tmp/open_search_queue.sh`,
`tmp/corner_gate_queue.sh`, and `tmp/extval_queue{,2,3,4}.sh`; the remaining
gate commands are derivable (`run_statekv_gates.py <stage> --config <yaml>`).

| Entry | Config → script | Status | Notes |
|---|---|---|---|
| Teacher gate G0/G1 | `statekv_teacher_gate_g0.yaml` → `run_statekv_gates.py teacher-gate` | PARTIALLY REPRODUCIBLE | Command derivable; analysis builders in `analysis/tables/gate0_*.py`, `gate1_*` |
| Teacher gate substrate B | `statekv_teacher_gate_p23b.yaml` → `run_statekv_gates.py teacher-gate` | PARTIALLY REPRODUCIBLE | Era-1 substrate: needs Qwen2.5-1.5B weights |
| P2 cheap panel (P23b) | `statekv_p2_p23b_cheap.yaml` → `run_statekv_gates.py p2` | PARTIALLY REPRODUCIBLE | |
| Ladder 2B / marginal 2C | `statekv_ladder_2b.yaml` → `run_statekv_gates.py ladder` | PARTIALLY REPRODUCIBLE | The stored run contains the probe-metric defect (post-phase-shift committed KLs are a different-input metric); a rerun with the same config reproduces raw rows, and the corrected analysis is `analysis/tables/ladder_2b_risk_depth.py` + `analysis/statekv_ladder_2b_deep_risk.md` (pre-shift cycles only) |
| Refresh arms 101–105 | `statekv_refresh_arms_qwen3_8b_768_256.yaml` → `run_statekv_gates.py r2b-gate` | PARTIALLY REPRODUCIBLE | A second config `statekv_selective_refresh_r2b.yaml` also exists; the run dir matches the refresh-arms config |
| Refresh-gap R0 | offline: `scripts/analyze_refresh_gap_decomposition.py` over P23b parquets → `analysis/tables/refresh_gap_decomposition_summary.csv` ✓ | PARTIALLY REPRODUCIBLE | No config file (offline analysis); needs the P23b run present |
| R1 prescreen | offline: `analysis/tables/build_trigger_feature_screen.py`, `fit_refresh_trigger.py` → `analysis/tables/trigger_screen_report.md` ✓ | PARTIALLY REPRODUCIBLE | Offline; depends on decomposition records |
| R2a labels v1 | `statekv_selective_refresh_r2a.yaml` → `run_statekv_gates.py r2a-labels` → `statekv_selective_refresh_labels_r2a_v1/` ✓ | PARTIALLY REPRODUCIBLE | |
| R2a labels v2 | `statekv_selective_refresh_r2a_v2.yaml` → same → `statekv_selective_refresh_labels_r2a_v2/partial_*` ✓ | REQUIRES RERUN | Early-stopped; only partial parquets, no `summary.json`/`config.yaml` in the curated dir. Result-as-evidence stands (degenerate operating point), but the run itself is incomplete |
| R2a labels v3 | `statekv_selective_refresh_r2a_v3.yaml` → same → `statekv_selective_refresh_labels_r2a_v3/partial_*` ✓ | REQUIRES RERUN | Same: partial artifacts only |
| R2 trigger verdict | offline analysis over R2a labels → `analysis/tables/selective_refresh_negative_result_r2.md`, `refresh_operating_point_comparison.csv`, `refresh_trigger_no_freeze.json` ✓ | PARTIALLY REPRODUCIBLE | Depends on R2a v1 (+ partial v2/v3) artifacts |
| Recoverable R0 | `statekv_recoverable_r0_qwen3_8b.yaml` → `run_oracle_policy_freegen.py` (recoverable mode) | PARTIALLY REPRODUCIBLE | Mode dispatch verified in `statekv/oracle_policy_freegen.py`; regression test `tests/test_recoverable_r0.py` exists |
| QK–V battery | `statekv_qkv_decomposition_qwen3_8b.yaml` → `run_qkv_decomposition.py` | PARTIALLY REPRODUCIBLE | Derived tables `analysis/tables/qkv_*.csv` regenerable from the run rows |
| qk_tiered_v 256t / 352f / 352t | `statekv_qkvtier_gate_{256t,352f,352t}.yaml` → `run_oracle_policy_freegen.py` (tier mode) | PARTIALLY REPRODUCIBLE | Tier logic in `statekv/oracle_policy_freegen.py`; gate test `tests/test_qkvtier_gate.py`; spot-checked chain intact. Arm memories must match exactly for G5 comparability |
| Open stress 768/128, 768/64 | `statekv_openstress_768_{128,64}.yaml` → `run_oracle_policy_freegen.py` | PARTIALLY REPRODUCIBLE | 768/64 invocation verbatim in `tmp/open_search_queue.sh` |
| Open corner h4 / h16 | `statekv_opencorner_768_64_h{4,16}.yaml` → same | PARTIALLY REPRODUCIBLE | Verbatim in `tmp/open_search_queue.sh` / `tmp/corner_gate_queue.sh` |
| Corner obswin h16 | `statekv_corner_obswin_768_64_h16.yaml` → same | PARTIALLY REPRODUCIBLE | Verbatim in `tmp/corner_gate_queue.sh` |
| Headwise probe HF4 | `statekv_headwise_probe_qwen3_8b.yaml` → `run_qkv_decomposition.py` | PARTIALLY REPRODUCIBLE | Verbatim in `tmp/open_search_queue.sh` |
| Extval freegen arms (3072_256, 3072_64, h4/h16 ×2 budgets, mk, qwen25_7b, reasoning_af) | `statekv_extval_*.yaml` → `run_oracle_policy_freegen.py` | PARTIALLY REPRODUCIBLE | Verbatim in `tmp/extval_queue{,2,3,4}.sh`; qwen25_7b additionally needs that model's local weights; substrate guards regression-tested in `tests/test_external_validity_substrate.py` |
| Extval decomposition | `statekv_extval_decomp_3072_256.yaml` → `run_qkv_decomposition.py` | PARTIALLY REPRODUCIBLE | Verbatim in `tmp/extval_queue.sh` |
| HF1b conditional budgeting | offline probe over decomposition records → `analysis/tables/open_hf1_*` | PARTIALLY REPRODUCIBLE | Deprioritized before any model run (pre-committed probe rule); nothing model-side to rerun |
| `statekv_openstress_3072_256.yaml` | config exists, no run dir | SCRIPT EXISTS BUT RAW RESULT MISSING | The 3072 regime was covered by the extval arms instead |

## 5. Retest program

| Entry | Config → script | Status | Notes |
|---|---|---|---|
| Retest replay era1 n24 | `retest_replay_era1_n24_protocol.yaml` → `run_direct_policy_replay.py` → `statekv_retest_replay_era1_n24_v1/` ✓ | PARTIALLY REPRODUCIBLE | Chain fully intact (config carries `output_run` + `runtime_run_id`), but no queue script records the command: `tmp/retest_queue.sh` covers only the freegen track; `tmp/retest_track_a.log` records the output path only. Command is derivable one-for-one |
| Retest freegen n20 | `retest_freegen_qwen3_8b_n20_protocol.yaml` → `run_retest_freegen.py` → `statekv_retest_freegen_qwen3_8b_n20_v1/` ✓ | REPRODUCIBLE | Exact commands in `tmp/retest_queue.sh` (smoke + full panel + `analysis/build_retest_report.py`); note `tmp/` is being archived into the repo — after that move the path changes, content does not |
| Retest VJP Rademacher | `retest_vjp_rademacher_replication.yaml` (+ `retest_vjp_base_config.yaml`) → `run_vjp_routes_pilot.py` → `statekv_retest_vjp_rademacher_replication_v1/` ✓ | PARTIALLY REPRODUCIBLE | Chain intact; config names the source run `independent_fisher_4bit_24newseq_seed20260726_v1` (present); command recorded only as output path in `tmp/retest_track_d.log` — derivable |

## Appendices

- **Smoke/debug dirs** (registry Appendix A): `REQUIRES RERUN` by design —
  scaffolding, not evidence; most have only `resolved_config.yaml` + fragments.
- **`p20a_integration_failed_*`**: `REQUIRES RERUN` and intentionally not
  rerun-in-place — failure traces superseded by the fixed `p20a_v1` workload
  (config `qwen25_15b_govreport_static_lexical_p20a.yaml`).
- **Discovery-era runs** (registry Appendix C): `PARTIALLY REPRODUCIBLE`.
  Configs for the line live in `configs/discovery/` (`discovery_small.yaml`,
  `functional_probe_*`, `mechanism_targeted_4bit.yaml`, …) and runners in
  `scripts/run_{temporal_discovery,functional_probe,gauge_geometry,
  independent_fisher,mechanism_targeted,output_sensitivity,robust_envelope,
  theory_closing,trajectory_model}.py`. Caveat: `discovery_small` protocols
  v2/v3/v4 coexist as three full result dirs with **no recorded canonical
  pick** — reproducing "the" discovery result requires choosing a protocol
  version that the project itself never pinned down.
- **Seed-suffixed twins / runtime fragments** (registry Appendix D): not
  reproducible targets; they are byproducts of the canonical runs above.
- **`analysis/generate_report.py` pipeline**: `PARTIALLY REPRODUCIBLE` /
  partially dead. `analysis/manifest.json` is stale — it references
  `analysis/README.md`, `analysis/david_update.md`, and
  `analysis/data_schema_report.md`, none of which exist on disk, so the report
  generation/manifest half of the pipeline no longer reflects the tree.

## Known gaps summary

1. README-documented commands stop at P28–P30; everything later relies on
   derivable invocations plus `tmp/*queue*.sh` scripts (being archived).
2. Two retest tracks (replay-era1, VJP Rademacher) have no recorded command —
   configs and results are intact, so this is documentation debt, not data loss.
3. R2a v2/v3 are early-stopped partial runs; their evidence value stands but
   full runs require rerun.
4. All model-scale rows presuppose the local HF model cache and LongBench
   dataset cache; none of that is vendored into the repo.
5. The ladder-2B run artifacts contain the documented probe-metric defect;
   use the corrected pre-shift analysis, not the raw committed probe KLs.
