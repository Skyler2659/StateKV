#!/bin/bash
set -x
cd /Users/wangsikai/l1-robust-kv-cache
export HF_HUB_OFFLINE=1
PY=.venv/bin/python
$PY scripts/run_oracle_policy_freegen.py --config configs/stages/statekv_extval_3072_256_reasoning_af.yaml
$PY scripts/run_oracle_policy_freegen.py --config configs/stages/statekv_extval_3072_256_mk.yaml
$PY scripts/run_oracle_policy_freegen.py --config configs/stages/statekv_extval_3072_256_qwen25_7b.yaml
echo QUEUE2_DONE
