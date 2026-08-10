#!/bin/bash
# Open-search run queue: waits for the 768/128 run (pid given as $1), then
# executes the remaining regime/probe runs sequentially on MPS.
set -u
cd /Users/wangsikai/l1-robust-kv-cache
PY=.venv/bin/python
export HF_HUB_OFFLINE=1

if [ -n "${1:-}" ]; then
  while kill -0 "$1" 2>/dev/null; do sleep 20; done
fi

echo "[queue] $(date +%H:%M:%S) starting openstress_768_64"
$PY scripts/run_oracle_policy_freegen.py --config configs/stages/statekv_openstress_768_64.yaml > /tmp/openstress_64.log 2>&1
echo "[queue] $(date +%H:%M:%S) openstress_768_64 exit=$?"

echo "[queue] $(date +%H:%M:%S) starting headwise probe"
$PY scripts/run_qkv_decomposition.py --config configs/stages/statekv_headwise_probe_qwen3_8b.yaml > /tmp/headwise_probe.log 2>&1
echo "[queue] $(date +%H:%M:%S) headwise probe exit=$?"

echo "[queue] $(date +%H:%M:%S) starting opencorner_768_64_h16"
$PY scripts/run_oracle_policy_freegen.py --config configs/stages/statekv_opencorner_768_64_h16.yaml > /tmp/opencorner_h16.log 2>&1
echo "[queue] $(date +%H:%M:%S) opencorner_768_64_h16 exit=$?"

echo "[queue] $(date +%H:%M:%S) ALL DONE"
