#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TASKS=${TASKS:-niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,niah_multikey_2,niah_multikey_3,niah_multivalue,niah_multiquery,vt,cwe,fwe,qa_1,qa_2}
METHODS=${METHODS:-full,recency,streamingllm,h2o,snapkv,curdkv,attention,v_leverage,residual_v}
SEEDS=${SEEDS:-42}
CONTEXT_LENGTH=${CONTEXT_LENGTH:-16384}
IFS=',' read -r -a TASK_ARRAY <<< "$TASKS"
IFS=',' read -r -a METHOD_ARRAY <<< "$METHODS"
IFS=',' read -r -a SEED_ARRAY <<< "$SEEDS"

for task in "${TASK_ARRAY[@]}"; do
  if [[ -n "${RULER_DATA_ROOT:-}" ]]; then
    task_data_path="$RULER_DATA_ROOT/$task/$CONTEXT_LENGTH.jsonl"
    [[ -f "$task_data_path" ]] || { echo "Missing $task_data_path"; exit 2; }
  elif [[ -n "${DATA_PATH:-}" && ${#TASK_ARRAY[@]} -eq 1 ]]; then
    task_data_path="$DATA_PATH"
  else
    echo "Set RULER_DATA_ROOT (or DATA_PATH for one task) to official RULER JSONL"
    exit 2
  fi
  for method in "${METHOD_ARRAY[@]}"; do
    for seed in "${SEED_ARRAY[@]}"; do
      TASK="$task" METHOD="$method" SEED="$seed" DATA_PATH="$task_data_path" \
        CONTEXT_LENGTH="$CONTEXT_LENGTH" \
        CONFIG=configs/experiments/ruler_main.yaml \
        "$PROJECT_DIR/scripts/run_single.sh"
    done
  done
done
