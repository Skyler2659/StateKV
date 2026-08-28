# Experiment catalog

This catalog maps the research program to its formal reports, primary configuration areas,
and supporting artifacts. It is intended as a compact guide; the detailed protocol records
remain under [`evidence/`](evidence/) and the run artifacts remain under `results/`.

| Program | Research question | Report | Primary configuration area |
|---|---|---|---|
| E0 mechanism foundations | Can state-conditioned cache actions be measured accurately? | [01](experiments/01_frozen_mechanism.md) | `experiments/`, `configs/frozen/` |
| P0–P24 current-state policies | Which live-cache signals support selection or refresh? | [02](experiments/02_training_free_and_direct_policy.md) | `configs/stages/` |
| P25–P35 working-set control | How do physical evaluation, QK routing, coverage, and cadence behave in closed loop? | [03](experiments/03_oracle_gates_and_retests.md) | `configs/stages/` |
| Adaptive temporal utility | Does optimal token importance vary over time? | [04](experiments/04_adaptive_temporal.md) | `configs/adaptive_temporal/` |
| Causal existence / R2 | Can a causal rollout predict future utility? | [05](experiments/05_causal_existence.md) | `configs/statekv_existence/` |
| Counterfactual utility and students | Can a lightweight model retain R2's task-critical ranking? | [06](experiments/06_counterfactual_distillation.md) | `configs/statekv_counterfactual/` |
| Cheap-R2 | What is the smallest rollout schedule that retains the retrieval gain? | [07](experiments/07_cheap_r2.md) | `configs/statekv_counterfactual/` |
| Final benchmark | Where does Cheap-R2 help relative to current-QK and LAQ? | [08](experiments/08_benchmark_results.md) | benchmark result configurations |

## Supporting material

- [`experiments/frozen_registry.yaml`](../experiments/frozen_registry.yaml) lists the frozen
  mechanism-phase manifests.
- [`evidence/`](evidence/) preserves study protocols, detailed result notes, and search records.
- [`evidence/tables/`](evidence/tables/) contains the compact tables used by the earlier programs.
- `results/` contains canonical summaries and run-level artifacts.

For active commands, see [Running StateKV](REPRODUCIBILITY.md).
