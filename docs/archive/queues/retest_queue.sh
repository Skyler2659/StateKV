#!/bin/bash
# StateKV gate-retest compute queue: smoke -> Track B full panel.
set -x
cd /Users/wangsikai/l1-robust-kv-cache
export HF_HUB_OFFLINE=1
export PYTHONPATH=benchmarks/mlx
PY=.venv/bin/python

$PY scripts/run_retest_freegen.py --config configs/stages/retest_freegen_smoke.yaml
$PY scripts/run_retest_freegen.py --config configs/stages/retest_freegen_qwen3_8b_n20_protocol.yaml
$PY analysis/build_retest_report.py
echo RETEST_QUEUE_DONE
