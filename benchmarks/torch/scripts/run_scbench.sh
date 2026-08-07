#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
METHODS=${METHODS:-full,random,recency,v_leverage,curdkv}
SCBENCH_MODE=${SCBENCH_MODE:-reuse}
case "$SCBENCH_MODE" in
  reuse) SCBENCH_CONFIG=configs/experiments/scbench_reuse.yaml ;;
  query_agnostic_single) SCBENCH_CONFIG=configs/experiments/scbench_query_agnostic_single.yaml ;;
  query_visible) SCBENCH_CONFIG=configs/experiments/scbench_query_visible.yaml ;;
  *) echo "Unsupported SCBENCH_MODE=$SCBENCH_MODE"; exit 2 ;;
esac
IFS=',' read -r -a METHOD_ARRAY <<< "$METHODS"

for method in "${METHOD_ARRAY[@]}"; do
  if [[ -n "${SCBENCH_DATA:-}" ]]; then
    DATA_PATH="$SCBENCH_DATA" METHOD="$method" \
      CONFIG="$SCBENCH_CONFIG" \
      "$PROJECT_DIR/scripts/run_single.sh"
  else
    METHOD="$method" CONFIG="$SCBENCH_CONFIG" \
      "$PROJECT_DIR/scripts/run_single.sh"
  fi
done
