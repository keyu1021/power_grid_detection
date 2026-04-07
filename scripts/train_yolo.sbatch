#!/bin/bash
#SBATCH --job-name=yolo_train
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
mkdir -p "${PROJECT_ROOT}/logs" "${PROJECT_ROOT}/models"

module --force purge
module load StdEnv/2023
module load gcc/12.3
module load cuda/12.2
module load opencv/4.13.0

source "${PROJECT_ROOT}/.venv/bin/activate"

unset CUDA_MPS_PIPE_DIRECTORY
unset CUDA_MPS_LOG_DIRECTORY
unset CUDA_MPS_ACTIVE_THREAD_PERCENTAGE
unset NVIDIA_MPS_PIPE_DIRECTORY
unset NVIDIA_MPS_LOG_DIRECTORY

cd "${PROJECT_ROOT}"

YOLO_DATA_ROOT="${YOLO_DATA_ROOT:-${PROJECT_ROOT}/data/yolo_generator_substation}"
YOLO_DATASET_YAML="${YOLO_DATASET_YAML:-${YOLO_DATA_ROOT}/dataset.yaml}"
YOLO_MODEL="${YOLO_MODEL:-${PROJECT_ROOT}/src/models/yolov8n.pt}"
YOLO_EPOCHS="${YOLO_EPOCHS:-100}"
YOLO_IMGSZ="${YOLO_IMGSZ:-1024}"
YOLO_BATCH="${YOLO_BATCH:-8}"
YOLO_DEVICE="${YOLO_DEVICE:-0}"
YOLO_WORKERS="${YOLO_WORKERS:-${SLURM_CPUS_PER_TASK:-2}}"
MODEL_PROJECT_DIR="${MODEL_PROJECT_DIR:-${PROJECT_ROOT}/models}"

if [ -n "${MODEL_PROJECT_DIR:-}" ]; then
  case "${MODEL_PROJECT_DIR}" in
    /*) ;;
    *) MODEL_PROJECT_DIR="${PROJECT_ROOT}/${MODEL_PROJECT_DIR}" ;;
  esac
fi
mkdir -p "${MODEL_PROJECT_DIR}"

if [ -n "${RUN_NAME:-}" ]; then
  YOLO_RUN_NAME="${RUN_NAME}"
else
  dataset_tag="$(basename "${YOLO_DATA_ROOT}")"
  run_idx=1
  while [ -e "${MODEL_PROJECT_DIR}/${dataset_tag}_run_${run_idx}" ]; do
    run_idx=$((run_idx + 1))
  done
  YOLO_RUN_NAME="${dataset_tag}_run_${run_idx}"
fi

if [ ! -f "${YOLO_DATASET_YAML}" ]; then
  echo "Missing dataset yaml: ${YOLO_DATASET_YAML}" >&2
  exit 1
fi

echo "Training YOLO model: ${YOLO_MODEL}"
echo "Dataset yaml: ${YOLO_DATASET_YAML}"
echo "Model project directory: ${MODEL_PROJECT_DIR}"
echo "Run name: ${YOLO_RUN_NAME}"
echo "YOLO workers: ${YOLO_WORKERS}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi

python src/train/yolo_train.py \
  --model "${YOLO_MODEL}" \
  --data "${YOLO_DATASET_YAML}" \
  --epochs "${YOLO_EPOCHS}" \
  --imgsz "${YOLO_IMGSZ}" \
  --batch "${YOLO_BATCH}" \
  --device "${YOLO_DEVICE}" \
  --workers "${YOLO_WORKERS}" \
  --project "${MODEL_PROJECT_DIR}" \
  --name "${YOLO_RUN_NAME}"
