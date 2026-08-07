#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
CONFIG=${CONFIG:-configs/experiments/ruler_main.yaml}
MODEL_PATH=${MODEL_PATH:-}
RESULTS_ROOT=${KVBENCH_RESULTS_ROOT:-results}

cd "$PROJECT_DIR"
ARGS=(--config "$CONFIG" --set "output.root=$RESULTS_ROOT")
[[ -n "$MODEL_PATH" ]] && ARGS+=(--set "model.name=$MODEL_PATH")
[[ -n "${TASK:-}" ]] && ARGS+=(--set "benchmark.task=$TASK")
[[ -n "${DATA_PATH:-}" ]] && ARGS+=(--set "benchmark.data_path=$DATA_PATH")
[[ -n "${METHOD:-}" ]] && ARGS+=(--set "method.name=$METHOD")
[[ -n "${ESTIMATOR:-}" ]] && ARGS+=(--set "method.leverage_estimator=$ESTIMATOR")
[[ -n "${BUDGET:-}" ]] && ARGS+=(--set "budget.cache_budget=$BUDGET")
[[ -n "${CONTEXT_LENGTH:-}" ]] && ARGS+=(--set "benchmark.context_length=$CONTEXT_LENGTH")
[[ -n "${NUM_SAMPLES:-}" ]] && ARGS+=(--set "benchmark.num_samples=$NUM_SAMPLES")
[[ -n "${SEED:-}" ]] && ARGS+=(--set "runtime.seed=$SEED")
[[ -n "${DEVICE:-}" ]] && ARGS+=(--set "runtime.device=$DEVICE")
[[ -n "${DTYPE:-}" ]] && ARGS+=(--set "model.dtype=$DTYPE")
[[ -n "${VISIBILITY:-}" ]] && ARGS+=(--set "protocol.visibility=$VISIBILITY")
[[ -n "${CACHE_MODE:-}" ]] && ARGS+=(--set "protocol.cache_mode=$CACHE_MODE")
[[ -n "${UPDATE_POLICY:-}" ]] && ARGS+=(--set "protocol.update_policy=$UPDATE_POLICY")
[[ -n "${UPDATE_INTERVAL:-}" ]] && ARGS+=(--set "protocol.update_interval=$UPDATE_INTERVAL")

exec "$PYTHON_BIN" -m kvbench.cli.run "${ARGS[@]}" "$@"

