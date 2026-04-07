#!/bin/bash
#SBATCH --job-name=dino_infer
#SBATCH --account=def-sdaniel
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --export=NONE

set -euo pipefail

PROJECT_ROOT="/home/keyuyao/projects/def-sdaniel/keyuyao/power-grid-detection"
RUN_DIR="${PROJECT_ROOT}/outputs/dino_ft_run2"
DATA_PATH="${PROJECT_ROOT}/data/COCODIR"
RUN_NAME="$(basename "${RUN_DIR}")"
INFER_DIR="${PROJECT_ROOT}/inference/${RUN_NAME}"
OUTPUT_PATH="${INFER_DIR}/predictions.json"

mkdir -p "${PROJECT_ROOT}/logs" "${INFER_DIR}"

module --force purge
module load StdEnv/2023
module load gcc/12.3
module load cuda/12.2

source "${PROJECT_ROOT}/.venv/bin/activate"

cd "${PROJECT_ROOT}"

echo "Starting inference on $(hostname)"
echo "Date: $(date)"
echo "Run dir: ${RUN_DIR}"
echo "Data path: ${DATA_PATH}"
echo "Inference dir: ${INFER_DIR}"
echo "Output path: ${OUTPUT_PATH}"
nvidia-smi

python src/utils/infer_dino.py \
  --run-dir "${RUN_DIR}" \
  --data-path "${DATA_PATH}" \
  --output-path "${OUTPUT_PATH}"
