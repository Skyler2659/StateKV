#!/bin/bash
set -u
cd /Users/wangsikai/l1-robust-kv-cache
PY=.venv/bin/python
export HF_HUB_OFFLINE=1
while kill -0 "$1" 2>/dev/null; do sleep 20; done
echo "[q2] $(date +%H:%M:%S) starting corner h4"
$PY scripts/run_oracle_policy_freegen.py --config configs/stages/statekv_opencorner_768_64_h4.yaml > /tmp/opencorner_h4.log 2>&1
echo "[q2] $(date +%H:%M:%S) h4 exit=$?"
echo "[q2] $(date +%H:%M:%S) starting obswin h16"
$PY scripts/run_oracle_policy_freegen.py --config configs/stages/statekv_corner_obswin_768_64_h16.yaml > /tmp/corner_obswin.log 2>&1
echo "[q2] $(date +%H:%M:%S) obswin exit=$?"
echo "[q2] $(date +%H:%M:%S) ALL DONE"
