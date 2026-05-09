#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-gpt2}"
TOKENS="${TOKENS:-512}"
SAMPLES="${SAMPLES:-1}"
OUTDIR="${OUTDIR:-outputs/compare}"
START="${START:-4}"
RECENT="${RECENT:-252}"
CACHE="${CACHE:-256}"

echo "=== plain ==="
python examples/eval_long_ppl.py \
  --model_name_or_path "${MODEL}" \
  --kv_strategy plain \
  --num_samples "${SAMPLES}" \
  --num_eval_tokens "${TOKENS}" \
  --output_dir "${OUTDIR}/plain"

echo "=== streaming ==="
python examples/eval_long_ppl.py \
  --model_name_or_path "${MODEL}" \
  --kv_strategy streaming \
  --start_size "${START}" \
  --recent_size "${RECENT}" \
  --num_samples "${SAMPLES}" \
  --num_eval_tokens "${TOKENS}" \
  --output_dir "${OUTDIR}/streaming"

echo "=== l1_robust ==="
python examples/eval_long_ppl.py \
  --model_name_or_path "${MODEL}" \
  --kv_strategy l1_robust \
  --cache_size "${CACHE}" \
  --start_size "${START}" \
  --sketch_dim 1024 \
  --recompute_interval 32 \
  --seed 0 \
  --num_samples "${SAMPLES}" \
  --num_eval_tokens "${TOKENS}" \
  --output_dir "${OUTDIR}/l1_robust"
