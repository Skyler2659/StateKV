#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPOSITORY_ROOT="$(cd "${PROJECT_ROOT}/../.." && pwd)"
PYTHON_BIN_PATH="${PYTHON_BIN_PATH:-${REPOSITORY_ROOT}/.venv/bin/python}"
CONFIG_PATH="${PROJECT_ROOT}/configs/experiments/diagnostics/long_generation/qwen25_15b_govreport_l2_update_sweep.yaml"
LONG_GEN_NUM_SAMPLES="${LONG_GEN_NUM_SAMPLES:-2}"
LONG_GEN_MAX_NEW_TOKENS="${LONG_GEN_MAX_NEW_TOKENS:-256}"
LONG_GEN_MODEL="${LONG_GEN_MODEL:-mlx-community/Qwen2.5-1.5B-Instruct-4bit}"
LONG_GEN_INCLUDE_EVERY_TOKEN="${LONG_GEN_INCLUDE_EVERY_TOKEN:-1}"
LONG_GEN_CACHE_BUDGET="${LONG_GEN_CACHE_BUDGET:-128}"
LONG_GEN_INCLUDE_FULL="${LONG_GEN_INCLUDE_FULL:-1}"
if [[ -z "${LONG_GEN_RESULTS_ROOT:-}" ]]; then
  if [[ "${LONG_GEN_NUM_SAMPLES}" == "2" && "${LONG_GEN_MAX_NEW_TOKENS}" == "256" ]]; then
    LONG_GEN_RESULTS_ROOT="${PROJECT_ROOT}/results/long_generation_leverage"
  else
    LONG_GEN_RESULTS_ROOT="${PROJECT_ROOT}/results/long_generation_leverage_s${LONG_GEN_NUM_SAMPLES}_t${LONG_GEN_MAX_NEW_TOKENS}"
  fi
fi

cd "${PROJECT_ROOT}"

run_condition() {
  local label="$1"
  shift
  echo "[long-generation] starting ${label}"
  "${PYTHON_BIN_PATH}" scripts/run_benchmark.py \
    --config "${CONFIG_PATH}" \
    --model "${LONG_GEN_MODEL}" \
    --num_samples "${LONG_GEN_NUM_SAMPLES}" \
    --max_new_tokens "${LONG_GEN_MAX_NEW_TOKENS}" \
    --budget "${LONG_GEN_CACHE_BUDGET}" \
    --output_dir "${LONG_GEN_RESULTS_ROOT}" \
    --skip_analysis \
    "$@"
}

# Run the inexpensive anchors first.  If the every-token condition is much
# slower than expected, the preceding runs still leave a useful partial curve.
if [[ "${LONG_GEN_INCLUDE_FULL}" == "1" ]]; then
  run_condition "full anchor" --method full
fi
run_condition "L2 prefill-only" --method l2_prefill_only
run_condition "L2 refit every 64 generated tokens" \
  --method l2_leverage --update_policy every_n_steps --update_interval 64
run_condition "L2 refit every 16 generated tokens" \
  --method l2_leverage --update_policy every_n_steps --update_interval 16
if [[ "${LONG_GEN_INCLUDE_EVERY_TOKEN}" == "1" ]]; then
  run_condition "L2 refit every generated token" \
    --method l2_leverage --update_policy every_n_steps --update_interval 1
fi

"${PYTHON_BIN_PATH}" scripts/summarize_long_generation_leverage.py \
  --results-root "${LONG_GEN_RESULTS_ROOT}"
