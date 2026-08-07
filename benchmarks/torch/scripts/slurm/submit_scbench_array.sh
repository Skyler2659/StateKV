#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
: "${MODEL_PATH:?Set MODEL_PATH to the Hugging Face checkpoint directory}"
: "${SCBENCH_DATA:?Set SCBENCH_DATA to an official raw context/multi_turns JSONL export}"
METHODS=${METHODS:-full,random,recency,v_leverage,curdkv}
BUDGETS=${BUDGETS:-256}
SEEDS=${SEEDS:-42}
mkdir -p "$PROJECT_DIR/logs/slurm" "$PROJECT_DIR/logs/manifests"
MANIFEST="$PROJECT_DIR/logs/manifests/scbench_$(date +%Y%m%d_%H%M%S)_$$.tsv"
touch "$MANIFEST"
IFS=',' read -r -a method_array <<< "$METHODS"
IFS=',' read -r -a budget_array <<< "$BUDGETS"
IFS=',' read -r -a seed_array <<< "$SEEDS"
for method in "${method_array[@]}"; do
  for budget in "${budget_array[@]}"; do
    for seed in "${seed_array[@]}"; do
      printf '%s\t%s\t%s\n' "$method" "$budget" "$seed" >> "$MANIFEST"
    done
  done
done
COUNT=$(wc -l < "$MANIFEST" | tr -d ' ')
cd "$PROJECT_DIR"
sbatch --array="0-$((COUNT - 1))" \
  --export="ALL,MODEL_PATH=$MODEL_PATH,SCBENCH_DATA=$SCBENCH_DATA,TASK=${TASK:-scbench_kv},SCBENCH_CONFIG=${SCBENCH_CONFIG:-configs/experiments/scbench_reuse.yaml},MANIFEST=$MANIFEST" \
  scripts/slurm/run_scbench.sh
