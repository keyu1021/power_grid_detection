#!/bin/bash
#SBATCH --job-name=dino_train
#SBATCH --account=def-sdaniel
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
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

# Avoid inheriting stale CUDA MPS client/server variables from login shell.
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

# Allow orchestration layers to pass deterministic output locations.
if [ -n "${OUTPUT_DIR:-}" ]; then
  case "${OUTPUT_DIR}" in
    /*) ;;
    *) OUTPUT_DIR="${PROJECT_ROOT}/${OUTPUT_DIR}" ;;
  esac
else
  run_idx=1
  while [ -e "${PROJECT_ROOT}/outputs/dino_run${run_idx}" ]; do
    run_idx=$((run_idx + 1))
  done
  OUTPUT_DIR="${PROJECT_ROOT}/outputs/dino_run${run_idx}"
fi
mkdir -p "${OUTPUT_DIR}"
echo "Using output directory: ${OUTPUT_DIR}"

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi

# python src/models/DINO/main.py \
#   --config_file src/models/DINO/config/DINO/DINO_4scale.py \
#   --coco_path "${PROJECT_ROOT}/data/COCODIR_tower" \
#   --output_dir "${OUTPUT_DIR}" \
#   --device cuda \
#   --num_workers 0 \
#   --options batch_size=1 num_classes=2 dn_labelbook_size=2

COCO_PATH="${COCO_PATH:-${PROJECT_ROOT}/data/COCODIR}"
CHECKPOINT="${CHECKPOINT:-${PROJECT_ROOT}/src/models/DINO/config/checkpoint0011_4scale.pth}"

# python src/models/DINO/main.py \
#   --config_file src/models/DINO/config/DINO/DINO_4scale.py \
#   --coco_path "${COCO_PATH}" \
#   --output_dir "${OUTPUT_DIR}" \
#   --device cuda \
#   --num_workers 0 \
#   --resume "${checkpoint}" \
#   --options \
#     batch_size=1 \
#     num_classes=1 \
#     dn_labelbook_size=1 \
#     use_ema=False \
#     dn_scalar=100 \
#     embed_init_tgt=TRUE \
#     dn_label_coef=1.0 \
#     dn_bbox_coef=1.0 \
#     dn_box_noise_scale=1.0

python src/models/DINO/main.py \
  --output_dir "${OUTPUT_DIR}" \
  -c src/models/DINO/config/DINO/DINO_4scale.py \
  --coco_path "${COCO_PATH}" \
  --options \
    num_classes=2 \
    dn_labelbook_size=2 \
    dn_scalar=100 \
    embed_init_tgt=TRUE \
    dn_label_coef=1.0 \
    dn_bbox_coef=1.0 \
    use_ema=False \
    dn_box_noise_scale=1.0
