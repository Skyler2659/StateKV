#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TASKS=${TASKS:-narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,gov_report,qmsum,multi_news,trec,triviaqa,samsum,passage_count,passage_retrieval_en,lcc,repobench-p}
METHODS=${METHODS:-full,attention,v_leverage,residual_v}
IFS=',' read -r -a TASK_ARRAY <<< "$TASKS"
IFS=',' read -r -a METHOD_ARRAY <<< "$METHODS"

for task in "${TASK_ARRAY[@]}"; do
  for method in "${METHOD_ARRAY[@]}"; do
    TASK="$task" METHOD="$method" CONFIG=configs/experiments/longbench_main.yaml \
      "$PROJECT_DIR/scripts/run_single.sh"
  done
done

