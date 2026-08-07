#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
METHODS=${METHODS:-attention,v_leverage,independent_hybrid,score_fusion,product,residual_v}
IFS=',' read -r -a METHOD_ARRAY <<< "$METHODS"

for method in "${METHOD_ARRAY[@]}"; do
  METHOD="$method" CONFIG=configs/experiments/ruler_main.yaml \
    "$PROJECT_DIR/scripts/run_single.sh" \
    --set diagnostics.overlap=true \
    --set diagnostics.rank_correlation=true \
    --set diagnostics.evidence_recall=true \
    --set diagnostics.quadrants=true \
    --set diagnostics.reconstruction=true
done

