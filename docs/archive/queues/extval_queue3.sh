#!/bin/bash
set -x
cd /Users/wangsikai/l1-robust-kv-cache
export HF_HUB_OFFLINE=1
PY=.venv/bin/python
$PY scripts/run_oracle_policy_freegen.py --config configs/stages/statekv_extval_3072_64_h4.yaml
$PY scripts/run_oracle_policy_freegen.py --config configs/stages/statekv_extval_3072_64_h16.yaml
echo QUEUE3_DONE
