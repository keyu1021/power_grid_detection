#!/bin/bash
#SBATCH --job-name=dino_val_vis
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

echo "Starting job on $(hostname)"
echo "Date: $(date)"

PROJECT_ROOT="${PROJECT_ROOT:-/home/keyuyao/projects/def-sdaniel/keyuyao/power-grid-detection}"
mkdir -p "${PROJECT_ROOT}/logs" "${PROJECT_ROOT}/outputs"

module --force purge
module load StdEnv/2023
module load gcc/12.3
module load cuda/12.2

source "${PROJECT_ROOT}/.venv/bin/activate"

unset CUDA_MPS_PIPE_DIRECTORY
unset CUDA_MPS_LOG_DIRECTORY
unset CUDA_MPS_ACTIVE_THREAD_PERCENTAGE
unset NVIDIA_MPS_PIPE_DIRECTORY
unset NVIDIA_MPS_LOG_DIRECTORY

cd "${PROJECT_ROOT}"

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi

RUN_DIR="${RUN_DIR:-outputs/dino_ft_run2}"
RUN_NAME="$(basename "${RUN_DIR}")"
COCO_PATH="${COCO_PATH:-data/COCODIR}"
SCORE_THR="${SCORE_THR:-0.3}"
MAX_IMAGES="${MAX_IMAGES:-48}"
INFER_DIR="${INFER_DIR:-inference/${RUN_NAME}}"
PREDICTION_JSON="${PREDICTION_JSON:-${INFER_DIR}/predictions.json}"
OUT_DIR="${OUT_DIR:-${INFER_DIR}/visualization}"

python src/utils/visualize_dino_val.py \
  --run-dir "${RUN_DIR}" \
  --coco-path "${COCO_PATH}" \
  --prediction-json "${PREDICTION_JSON}" \
  --score-thr "${SCORE_THR}" \
  --max-images "${MAX_IMAGES}" \
  --out-dir "${OUT_DIR}"
