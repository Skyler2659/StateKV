#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
: "${MODEL_PATH:?Set MODEL_PATH to the Hugging Face checkpoint directory}"
TASKS=${TASKS:-narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,gov_report,qmsum,multi_news,trec,triviaqa,samsum,passage_count,passage_retrieval_en,lcc,repobench-p}
METHODS=${METHODS:-full,attention,v_leverage,residual_v}
BUDGETS=${BUDGETS:-256}
SEEDS=${SEEDS:-42}
mkdir -p "$PROJECT_DIR/logs/slurm" "$PROJECT_DIR/logs/manifests"
MANIFEST="$PROJECT_DIR/logs/manifests/longbench_$(date +%Y%m%d_%H%M%S)_$$.tsv"
touch "$MANIFEST"
IFS=',' read -r -a task_array <<< "$TASKS"
IFS=',' read -r -a method_array <<< "$METHODS"
IFS=',' read -r -a budget_array <<< "$BUDGETS"
IFS=',' read -r -a seed_array <<< "$SEEDS"
for task in "${task_array[@]}"; do
  for method in "${method_array[@]}"; do
    for budget in "${budget_array[@]}"; do
      for seed in "${seed_array[@]}"; do
        printf '%s\t%s\t%s\t%s\n' "$task" "$method" "$budget" "$seed" >> "$MANIFEST"
      done
    done
  done
done
COUNT=$(wc -l < "$MANIFEST" | tr -d ' ')
cd "$PROJECT_DIR"
sbatch --array="0-$((COUNT - 1))" \
  --export="ALL,MODEL_PATH=$MODEL_PATH,MANIFEST=$MANIFEST" \
  scripts/slurm/run_longbench.sh

