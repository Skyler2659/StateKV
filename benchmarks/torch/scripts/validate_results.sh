#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
RESULTS_ROOT=${KVBENCH_RESULTS_ROOT:-results}
cd "$PROJECT_DIR"
exec "$PYTHON_BIN" -m kvbench.analysis.validate --results-root "$RESULTS_ROOT" "$@"

