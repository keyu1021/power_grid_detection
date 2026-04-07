#!/bin/bash
#SBATCH --job-name=dino_eval
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

PROJECT_ROOT="/home/keyuyao/projects/def-sdaniel/keyuyao/power-grid-detection"
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

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0

cd "${PROJECT_ROOT}"

COCO_PATH="${PROJECT_ROOT}/data/COCODIR_tower"
RUN_DIR="${PROJECT_ROOT}/outputs/dino_ft_run1"
CHECKPOINT="${RUN_DIR}/checkpoint_best_regular.pth"
VAL_ANN="${COCO_PATH}/annotations/instances_val2017.json"
OUTPUT_DIR="${RUN_DIR}/eval"

if [ ! -f "${CHECKPOINT}" ]; then
  echo "Missing checkpoint: ${CHECKPOINT}" >&2
  exit 1
fi

if [ ! -f "${VAL_ANN}" ]; then
  echo "Missing validation annotations: ${VAL_ANN}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo "Evaluating checkpoint: ${CHECKPOINT}"
echo "Using dataset: ${COCO_PATH}"
echo "Writing eval outputs to: ${OUTPUT_DIR}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi

python src/models/DINO/main.py \
  --output_dir "${OUTPUT_DIR}" \
  -c src/models/DINO/config/DINO/DINO_4scale.py \
  --coco_path "${COCO_PATH}" \
  --resume "${CHECKPOINT}" \
  --eval \
  --options \
    num_classes=2 \
    dn_labelbook_size=2 \
    use_ema=False
