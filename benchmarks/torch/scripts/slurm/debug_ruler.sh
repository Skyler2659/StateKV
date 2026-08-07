#!/usr/bin/env bash
#SBATCH --job-name=kvdebug
#SBATCH --partition=IAI_SLURM_3090
#SBATCH --gres=gpu:1
#SBATCH --qos=singlegpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
: "${MODEL_PATH:?Set MODEL_PATH to the Hugging Face checkpoint directory}"
: "${RULER_DATA_PATH:?Set RULER_DATA_PATH to one official RULER JSONL file}"
cd "$PROJECT_DIR"

"$PYTHON_BIN" -m kvbench.cli.run \
  --config configs/experiments/ruler_main.yaml \
  --set "model.name=$MODEL_PATH" \
  --set "benchmark.data_path=$RULER_DATA_PATH" \
  --set benchmark.num_samples=1 \
  --set benchmark.context_length=4096 \
  --set budget.cache_budget=128 \
  --set method.name=v_leverage \
  --set generation.max_new_tokens=8 \
  --set generation.compute_teacher_forced_ppl=false \
  --set "output.root=${KVBENCH_RESULTS_ROOT:-results}"

