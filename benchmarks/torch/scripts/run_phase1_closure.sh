#!/usr/bin/env bash
set -euo pipefail

# Minimal research loop from the experiment specification.  It deliberately
# stops at 20 samples/task so diagnostics can be inspected before a full queue.
PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TASKS=${TASKS:-niah_single_1,niah_multikey_1,vt}
METHODS=${METHODS:-full,recency,attention,v_leverage,independent_hybrid,score_fusion,product,residual_v}
BUDGETS=${BUDGETS:-128,256,512}
SEEDS=${SEEDS:-42}
CONTEXT_LENGTH=${CONTEXT_LENGTH:-16384}
NUM_SAMPLES=${NUM_SAMPLES:-20}

IFS=',' read -r -a task_array <<< "$TASKS"
IFS=',' read -r -a method_array <<< "$METHODS"
IFS=',' read -r -a budget_array <<< "$BUDGETS"
IFS=',' read -r -a seed_array <<< "$SEEDS"

for task in "${task_array[@]}"; do
  : "${RULER_DATA_ROOT:?Set RULER_DATA_ROOT; expected <task>/<context>.jsonl}"
  task_data_path="$RULER_DATA_ROOT/$task/$CONTEXT_LENGTH.jsonl"
  [[ -f "$task_data_path" ]] || { echo "Missing $task_data_path"; exit 2; }
  for method in "${method_array[@]}"; do
    for budget in "${budget_array[@]}"; do
      for seed in "${seed_array[@]}"; do
        TASK="$task" METHOD="$method" BUDGET="$budget" SEED="$seed" \
          CONTEXT_LENGTH="$CONTEXT_LENGTH" NUM_SAMPLES="$NUM_SAMPLES" \
          DATA_PATH="$task_data_path" \
          CONFIG=configs/experiments/ruler_main.yaml \
          "$PROJECT_DIR/scripts/run_single.sh" \
          --set diagnostics.overlap=true \
          --set diagnostics.rank_correlation=true \
          --set diagnostics.evidence_recall=true \
          --set diagnostics.quadrants=true
      done
    done
  done
done
