#!/bin/bash
# External-validity gate run queue (MPS serial).
set -x
cd /Users/wangsikai/l1-robust-kv-cache
export HF_HUB_OFFLINE=1
PY=.venv/bin/python
$PY scripts/run_oracle_policy_freegen.py --config configs/stages/statekv_extval_3072_256.yaml
$PY scripts/run_oracle_policy_freegen.py --config configs/stages/statekv_extval_3072_256_h4.yaml
$PY scripts/run_oracle_policy_freegen.py --config configs/stages/statekv_extval_3072_256_h16.yaml
$PY scripts/run_oracle_policy_freegen.py --config configs/stages/statekv_extval_3072_64.yaml
$PY scripts/run_qkv_decomposition.py --config configs/stages/statekv_extval_decomp_3072_256.yaml
echo QUEUE_DONE
