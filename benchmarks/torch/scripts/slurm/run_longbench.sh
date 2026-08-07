#!/usr/bin/env bash
#SBATCH --job-name=kvlongbench
#SBATCH --partition=IAI_SLURM_3090
#SBATCH --gres=gpu:1
#SBATCH --qos=singlegpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/%x-%A_%a.out
#SBATCH --error=logs/slurm/%x-%A_%a.err
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
: "${MODEL_PATH:?Set MODEL_PATH to the Hugging Face checkpoint directory}"
: "${MANIFEST:?MANIFEST is set by submit_longbench_array.sh}"
LINE_NUMBER=$((SLURM_ARRAY_TASK_ID + 1))
IFS=$'\t' read -r task method budget seed < <(sed -n "${LINE_NUMBER}p" "$MANIFEST")
[[ -n "${task:-}" ]] || { echo "Invalid manifest row $LINE_NUMBER"; exit 2; }
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export TOKENIZERS_PARALLELISM=false
cd "$PROJECT_DIR"
"$PYTHON_BIN" -m kvbench.cli.run \
  --config configs/experiments/longbench_main.yaml \
  --set "model.name=$MODEL_PATH" \
  --set "benchmark.task=$task" \
  --set "method.name=$method" \
  --set "budget.cache_budget=$budget" \
  --set "runtime.seed=$seed" \
  --set "output.root=${KVBENCH_RESULTS_ROOT:-results}"

