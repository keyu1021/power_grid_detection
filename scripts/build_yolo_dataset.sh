#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/build_yolo_dataset.sh --images-dir DIR --bbox-gpkg FILE [options]

Required:
  --images-dir DIR   Directory of source orthophotos
  --bbox-gpkg FILE   Geopackage with bounding boxes

Optional:
  --splits-root DIR  Output split root (default: data/splits)
  --tiles-root DIR   Output tiled root (default: data/tiles)
  --yolo-root DIR    Output YOLO root (default: data/yolo)
  --train-ratio F    Default: 0.8
  --val-ratio F      Default: 0.2
  --test-ratio F     Default: 0.0
  --tile-size INT    Default: 1024
  --stride INT       Default: 512
  --seed INT         Default: 42
USAGE
}

IMAGES_DIR=""
BBOX_GPKG=""
SPLITS_ROOT="data/splits"
TILES_ROOT="data/tiles"
YOLO_ROOT="data/yolo"
TRAIN_RATIO="0.8"
VAL_RATIO="0.2"
TEST_RATIO="0.0"
TILE_SIZE="1024"
STRIDE="512"
SEED="42"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --images-dir) IMAGES_DIR="$2"; shift 2 ;;
    --bbox-gpkg) BBOX_GPKG="$2"; shift 2 ;;
    --splits-root) SPLITS_ROOT="$2"; shift 2 ;;
    --tiles-root) TILES_ROOT="$2"; shift 2 ;;
    --yolo-root) YOLO_ROOT="$2"; shift 2 ;;
    --train-ratio) TRAIN_RATIO="$2"; shift 2 ;;
    --val-ratio) VAL_RATIO="$2"; shift 2 ;;
    --test-ratio) TEST_RATIO="$2"; shift 2 ;;
    --tile-size) TILE_SIZE="$2"; shift 2 ;;
    --stride) STRIDE="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "$IMAGES_DIR" || -z "$BBOX_GPKG" ]]; then
  usage
  exit 1
fi

if [[ -f ".venv/bin/activate" ]]; then
  source .venv/bin/activate
fi

python -m src.utils.data_utils \
  --photos-dir "$IMAGES_DIR" \
  --output-dir "$SPLITS_ROOT" \
  --train-ratio "$TRAIN_RATIO" \
  --val-ratio "$VAL_RATIO" \
  --test-ratio "$TEST_RATIO" \
  --seed "$SEED"

python -m src.utils.gpkg_to_coco \
  --mode tiled \
  --gpkg "$BBOX_GPKG" \
  --splits-root "$SPLITS_ROOT" \
  --tiles-root "$TILES_ROOT" \
  --class-column class \
  --splits train val test \
  --tile-size "$TILE_SIZE" \
  --stride "$STRIDE" \
  --keep-empty-tiles \
  --empty-tile-fraction 0.01 \
  --seed "$SEED"

python -m src.utils.coco_to_yolo \
  --split-root "$TILES_ROOT" \
  --output-root "$YOLO_ROOT" \
  --splits train val

echo "YOLO dataset ready at $YOLO_ROOT"
