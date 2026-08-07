#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BUDGETS=${BUDGETS:-128,256,512}
LENGTHS=${LENGTHS:-8192,16384,32768}
METHODS=${METHODS:-attention,v_leverage,independent_hybrid,score_fusion,product,residual_v}
TASKS=${TASKS:-niah_single_1}
SEEDS=${SEEDS:-42}
IFS=',' read -r -a BUDGET_ARRAY <<< "$BUDGETS"
IFS=',' read -r -a LENGTH_ARRAY <<< "$LENGTHS"
IFS=',' read -r -a METHOD_ARRAY <<< "$METHODS"
IFS=',' read -r -a TASK_ARRAY <<< "$TASKS"
IFS=',' read -r -a SEED_ARRAY <<< "$SEEDS"

for task in "${TASK_ARRAY[@]}"; do
  for budget in "${BUDGET_ARRAY[@]}"; do
    for length in "${LENGTH_ARRAY[@]}"; do
      if [[ -n "${RULER_DATA_ROOT:-}" ]]; then
        task_data_path="$RULER_DATA_ROOT/$task/$length.jsonl"
        [[ -f "$task_data_path" ]] || { echo "Missing $task_data_path"; exit 2; }
      elif [[ -n "${DATA_PATH:-}" && ${#TASK_ARRAY[@]} -eq 1 && ${#LENGTH_ARRAY[@]} -eq 1 ]]; then
        task_data_path="$DATA_PATH"
      else
        echo "Set RULER_DATA_ROOT (or DATA_PATH for one task/length)"
        exit 2
      fi
      for method in "${METHOD_ARRAY[@]}"; do
        for seed in "${SEED_ARRAY[@]}"; do
          TASK="$task" BUDGET="$budget" CONTEXT_LENGTH="$length" \
            METHOD="$method" SEED="$seed" DATA_PATH="$task_data_path" \
            CONFIG=configs/experiments/ruler_main.yaml \
            "$PROJECT_DIR/scripts/run_single.sh"
        done
      done
    done
  done
done
