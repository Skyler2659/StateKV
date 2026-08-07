#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
: "${MODEL_PATH:?Set MODEL_PATH to the Hugging Face checkpoint directory}"
: "${RULER_DATA_ROOT:?Set RULER_DATA_ROOT; expected <task>/<context>.jsonl}"
TASKS=${TASKS:-niah_single_1}
METHODS=${METHODS:-full,attention,v_leverage,residual_v}
BUDGETS=${BUDGETS:-256}
CONTEXTS=${CONTEXTS:-16384}
SEEDS=${SEEDS:-42}
mkdir -p "$PROJECT_DIR/logs/slurm" "$PROJECT_DIR/logs/manifests"
MANIFEST="$PROJECT_DIR/logs/manifests/ruler_$(date +%Y%m%d_%H%M%S)_$$.tsv"
touch "$MANIFEST"
IFS=',' read -r -a task_array <<< "$TASKS"
IFS=',' read -r -a method_array <<< "$METHODS"
IFS=',' read -r -a budget_array <<< "$BUDGETS"
IFS=',' read -r -a context_array <<< "$CONTEXTS"
IFS=',' read -r -a seed_array <<< "$SEEDS"
for task in "${task_array[@]}"; do
  for method in "${method_array[@]}"; do
    for budget in "${budget_array[@]}"; do
      for context in "${context_array[@]}"; do
        for seed in "${seed_array[@]}"; do
          data_path="$RULER_DATA_ROOT/$task/$context.jsonl"
          [[ -f "$data_path" ]] || { echo "Missing $data_path"; exit 2; }
          printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$task" "$method" "$budget" "$context" "$seed" "$data_path" >> "$MANIFEST"
        done
      done
    done
  done
done
COUNT=$(wc -l < "$MANIFEST" | tr -d ' ')
cd "$PROJECT_DIR"
sbatch --array="0-$((COUNT - 1))" \
  --export="ALL,MODEL_PATH=$MODEL_PATH,NUM_SAMPLES=${NUM_SAMPLES:-50},MANIFEST=$MANIFEST" \
  scripts/slurm/run_ruler.sh
