# Power Grid Detection

## YOLO Data Pipeline

Use the shell pipeline to rebuild the YOLO dataset from source orthophotos and a bounding-box geopackage.

```bash
scripts/build_yolo_dataset.sh \
  --images-dir data/labeled \
  --bbox-gpkg data/annotations/bounding_box.gpkg
```

What it does:

1. Splits source orthophotos into `train/val/test` under `data/splits`.
2. Tiles each split and intersects bounding boxes into tiled COCO annotations under `data/tiles`.
3. Converts the tiled COCO splits into YOLO images, labels, and `dataset.yaml` under `data/yolo`.

Useful options:

```bash
scripts/build_yolo_dataset.sh \
  --images-dir data/labeled \
  --bbox-gpkg data/annotations/bounding_box.gpkg \
  --train-ratio 0.8 \
  --val-ratio 0.2 \
  --tile-size 1024 \
  --stride 512
```
