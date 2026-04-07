#!/usr/bin/env python3
"""Georeference predicted bounding boxes from an inference JSON file.

This script reads:
1. A prediction JSON produced by infer_dino.py
2. The matching COCO annotation file for image metadata
3. The original georeferenced tile files

It writes a GeoJSON file with one polygon feature per predicted box.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import rasterio
from rasterio.transform import xy
from shapely.geometry import box


def configure_proj_data_dir() -> None:
    # GeoPandas needs access to proj.db to interpret CRS metadata.
    for key in ("PROJ_DATA", "PROJ_LIB"):
        current = os.environ.get(key)
        if current and (Path(current) / "proj.db").exists():
            return

    candidates = [
        Path("/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcccore/proj/9.4.1/share/proj"),
        Path("/usr/share/proj"),
        Path("/usr/local/share/proj"),
    ]

    for candidate in candidates:
        if (candidate / "proj.db").exists():
            os.environ["PROJ_DATA"] = str(candidate)
            os.environ["PROJ_LIB"] = str(candidate)
            return


def load_coco_images(coco_path: Path) -> dict[int, dict]:
    # Build a quick lookup from image_id to the image metadata in COCO.
    annotation_path = coco_path / "annotations" / "instances_val2017.json"
    payload = json.loads(annotation_path.read_text())
    return {int(image["id"]): image for image in payload["images"]}


def resolve_image_path(image_info: dict, coco_path: Path) -> Path:
    # Prefer the original stored source path. If that path is not available here,
    # fall back to the local val2017 copy under the dataset root.
    source_path = Path(image_info["source_path"])
    if source_path.exists():
        return source_path
    return coco_path / "val2017" / image_info["file_name"]


def pixel_box_to_world_polygon(box_xyxy: list[float], image_path: Path):
    # Convert pixel box corners into map coordinates using the raster transform.
    x0, y0, x1, y1 = box_xyxy
    with rasterio.open(image_path) as src:
        left, top = xy(src.transform, y0, x0, offset="ul")
        right, bottom = xy(src.transform, y1, x1, offset="ul")
        geometry = box(left, bottom, right, top)
        crs = src.crs
    return geometry, crs


def main() -> None:
    parser = argparse.ArgumentParser(description="Georeference DINO predictions into GeoJSON.")
    parser.add_argument("--prediction-json", type=Path, required=True, help="Prediction JSON from infer_dino.py")
    parser.add_argument("--coco-path", type=Path, required=True, help="COCO-format dataset root")
    parser.add_argument("--output-path", type=Path, required=True, help="Output GeoJSON path")
    args = parser.parse_args()

    prediction_json = args.prediction_json.resolve()
    coco_path = args.coco_path.resolve()
    output_path = args.output_path.resolve()

    configure_proj_data_dir()
    import geopandas as gpd

    predictions = json.loads(prediction_json.read_text())
    images_by_id = load_coco_images(coco_path)

    rows = []
    output_crs = None

    for prediction in predictions:
        image_id = int(prediction["image_id"])
        image_info = images_by_id[image_id]
        image_path = resolve_image_path(image_info, coco_path)

        # Write one feature per predicted box so GIS tools can filter and style them easily.
        for box_xyxy, label, score in zip(
            prediction["boxes"],
            prediction["labels"],
            prediction["scores"],
        ):
            geometry, crs = pixel_box_to_world_polygon(box_xyxy, image_path)
            output_crs = crs
            rows.append(
                {
                    "image_id": image_id,
                    "file_name": image_info["file_name"],
                    "label": int(label),
                    "score": float(score),
                    "pixel_xmin": float(box_xyxy[0]),
                    "pixel_ymin": float(box_xyxy[1]),
                    "pixel_xmax": float(box_xyxy[2]),
                    "pixel_ymax": float(box_xyxy[3]),
                    "geometry": geometry,
                }
            )

    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=output_crs)

    # GeoJSON is easy to inspect and load into QGIS.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(output_path, driver="GeoJSON")


if __name__ == "__main__":
    main()
