#!/bin/bash
set -x
cd /Users/wangsikai/l1-robust-kv-cache
export HF_HUB_OFFLINE=1
.venv/bin/python scripts/run_oracle_policy_freegen.py --config configs/stages/statekv_extval_3072_256_qwen25_7b.yaml
echo QUEUE4_DONE
