#!/bin/bash
#SBATCH --job-name=dino_finetune
#SBATCH --account=def-sdaniel
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=01:00:00
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

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0

cd "${PROJECT_ROOT}"

# Fine-tune on the tower-only COCO dataset.
COCO_PATH="${COCO_PATH:-${PROJECT_ROOT}/data/COCODIR}"

# Source checkpoint to initialize from.
# Override at submit time, for example:
# sbatch --export=ALL,SOURCE_CKPT=/path/to/checkpoint.pth scripts/finetune_dino.sbatch
SOURCE_CKPT="${SOURCE_CKPT:-${PROJECT_ROOT}/src/models/DINO/config/checkpoint0011_4scale.pth}"
if [ ! -f "${SOURCE_CKPT}" ]; then
  echo "Missing fine-tune source checkpoint: ${SOURCE_CKPT}" >&2
  exit 1
fi

if [ -n "${OUTPUT_DIR:-}" ]; then
  case "${OUTPUT_DIR}" in
    /*) ;;
    *) OUTPUT_DIR="${PROJECT_ROOT}/${OUTPUT_DIR}" ;;
  esac
else
  run_idx=1
  while [ -e "${PROJECT_ROOT}/outputs/dino_ft_run${run_idx}" ]; do
    run_idx=$((run_idx + 1))
  done
  OUTPUT_DIR="${PROJECT_ROOT}/outputs/dino_ft_run${run_idx}"
fi
mkdir -p "${OUTPUT_DIR}"

echo "Using source checkpoint: ${SOURCE_CKPT}"
echo "Using output directory: ${OUTPUT_DIR}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi

python src/models/DINO/main.py \
  --output_dir "${OUTPUT_DIR}" \
  -c src/models/DINO/config/DINO/DINO_4scale.py \
  --coco_path "${COCO_PATH}" \
  --pretrain_model_path "${SOURCE_CKPT}" \
  --finetune_ignore label_enc.weight class_embed \
  --num_workers 2 \
  --options \
    num_classes=4 \
    dn_labelbook_size=4 \
    dn_scalar=100 \
    embed_init_tgt=TRUE \
    dn_label_coef=1.0 \
    dn_bbox_coef=1.0 \
    use_ema=False \
    dn_box_noise_scale=1.0
