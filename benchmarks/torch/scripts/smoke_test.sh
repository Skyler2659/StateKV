#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
cd "$PROJECT_DIR"

"$PYTHON_BIN" -m pytest tests -q
"$PYTHON_BIN" -m kvbench.cli.run \
  --config configs/experiments/smoke/ruler_tiny_cpu.yaml \
  --set "output.root=${KVBENCH_RESULTS_ROOT:-results/smoke}"

