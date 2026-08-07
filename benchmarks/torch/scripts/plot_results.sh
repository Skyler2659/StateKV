#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
RESULTS_ROOT=${KVBENCH_RESULTS_ROOT:-results}
OUTPUT_DIR=${FIGURE_OUTPUT_DIR:-results/figures}
cd "$PROJECT_DIR"
exec "$PYTHON_BIN" -m kvbench.analysis.plot \
  --results-root "$RESULTS_ROOT" --output-dir "$OUTPUT_DIR"

