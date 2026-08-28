# Running StateKV

## Environment

StateKV requires Python 3.9+. Model-scale runs use Apple-silicon MLX, locally cached
Hugging Face model weights, and the relevant task datasets.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -e .
.venv/bin/python -m pip install -e benchmarks/torch
.venv/bin/python -m pip install -e benchmarks/mlx
```

## Quick verification

```bash
PYTHONPATH=benchmarks/mlx .venv/bin/python -m pytest -q
PYTHONPATH=benchmarks/mlx .venv/bin/python scripts/smoke_test.py
```

## Active experiment entry points

```bash
# Multi-policy recoverable free-generation panel
HF_HUB_OFFLINE=1 .venv/bin/python scripts/run_retest_freegen.py \
  --config configs/stages/retest_freegen_qwen3_8b_n20_protocol.yaml

# QK routing and value-tier comparison
HF_HUB_OFFLINE=1 .venv/bin/python scripts/run_oracle_policy_freegen.py \
  --config configs/stages/statekv_qkvtier_gate_256t.yaml

# Causal future-utility dataset collection
HF_HUB_OFFLINE=1 .venv/bin/python scripts/collect_causal_existence_dataset.py \
  --config configs/statekv_existence/causal_existence_qwen3_8b.yaml

# R2 causal rollout study
HF_HUB_OFFLINE=1 .venv/bin/python scripts/run_causal_rollout_study.py \
  --config configs/statekv_counterfactual/r2_student_qwen3_8b.yaml
```

Each completed study has a frozen configuration and result summary in the repository.
The [experiment catalog](EXPERIMENT_REGISTRY.md) and
[technical appendices](evidence/) map the reports to those artifacts.
