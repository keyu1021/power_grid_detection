from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import rasterio
from rasterio.windows import Window
from rasterio.windows import bounds as window_bounds
from shapely.geometry import box

try:
    from src.utils.proj_utils import configure_proj_data_dir
except ModuleNotFoundError:
    from proj_utils import configure_proj_data_dir


@dataclass
class LabelsBundle:
    gdf: Any
    class_to_id: dict[str, int]
    categories: list[dict[str, Any]]


def _load_split_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Split annotation file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _image_path_from_entry(image: dict[str, Any], split_dir: Path) -> Path:
    source_path = image.get("source_path")
    if source_path:
        source = Path(source_path)
        if source.exists():
            return source
    file_name = Path(image["file_name"]).name
    return split_dir / "images" / file_name


def _load_labels(gpkg_path: Path, class_column: str, layer: str | None) -> LabelsBundle:
    configure_proj_data_dir()
    import geopandas as gpd

    gdf = gpd.read_file(gpkg_path, layer=layer)
    if gdf.empty:
        raise ValueError(f"No geometries found in {gpkg_path}")
    if class_column not in gdf.columns:
        raise ValueError(f"Class column '{class_column}' not found. Available columns: {list(gdf.columns)}")
    if gdf.crs is None:
        raise ValueError("Input geopackage has no CRS; cannot align annotations with rasters.")

    gdf = gdf.copy()
    gdf[class_column] = gdf[class_column].astype(str).str.strip()
    class_names = sorted(gdf[class_column].dropna().unique().tolist())
    class_to_id = {name: idx + 1 for idx, name in enumerate(class_names)}
    categories = [{"id": cid, "name": name, "supercategory": "object"} for name, cid in class_to_id.items()]

    print(f"Loaded {len(gdf)} features from {gpkg_path}")
    print(f"Classes: {class_to_id}")
    return LabelsBundle(gdf=gdf, class_to_id=class_to_id, categories=categories)


def _bbox_world_to_pixel(
    transform: rasterio.Affine,
    width: int,
    height: int,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
) -> tuple[float, float, float, float]:
    inv = ~transform
    left, top = inv * (xmin, ymax)
    right, bottom = inv * (xmax, ymin)

    x0 = min(left, right)
    y0 = min(top, bottom)
    x1 = max(left, right)
    y1 = max(top, bottom)

    x0 = max(0.0, min(float(width), x0))
    y0 = max(0.0, min(float(height), y0))
    x1 = max(0.0, min(float(width), x1))
    y1 = max(0.0, min(float(height), y1))

    w = x1 - x0
    h = y1 - y0
    return x0, y0, w, h


def convert_gpkg_to_split_coco(
    gpkg_path: Path,
    splits_root: Path,
    class_column: str = "class",
    layer: str | None = None,
    split_names: tuple[str, ...] = ("train", "val", "test"),
) -> None:
    labels_bundle = _load_labels(gpkg_path=gpkg_path, class_column=class_column, layer=layer)
    gdf = labels_bundle.gdf

    for split in split_names:
        split_dir = splits_root / split
        split_json_path = split_dir / "annotations.coco.json"
        if not split_json_path.exists():
            print(f"[skip] Missing split JSON: {split_json_path}")
            continue

        payload = _load_split_json(split_json_path)
        images = payload.get("images", [])
        annotations: list[dict[str, Any]] = []
        ann_id = 1

        for image in images:
            image_id = int(image["id"])
            image_path = _image_path_from_entry(image, split_dir)
            if not image_path.exists():
                print(f"[warn] Missing image file for image_id={image_id}: {image_path}")
                continue

            with rasterio.open(image_path) as src:
                image_poly = box(*src.bounds)
                labels = gdf if str(gdf.crs) == str(src.crs) else gdf.to_crs(src.crs)
                intersects = labels[labels.geometry.intersects(image_poly)]
                if intersects.empty:
                    continue

                for _, row in intersects.iterrows():
                    geom = row.geometry
                    if geom is None or geom.is_empty:
                        continue

                    clipped = geom.intersection(image_poly)
                    if clipped.is_empty:
                        continue

                    xmin, ymin, xmax, ymax = clipped.bounds
                    if xmax <= xmin or ymax <= ymin:
                        continue

                    x, y, w, h = _bbox_world_to_pixel(
                        transform=src.transform,
                        width=src.width,
                        height=src.height,
                        xmin=xmin,
                        ymin=ymin,
                        xmax=xmax,
                        ymax=ymax,
                    )
                    if w <= 0.0 or h <= 0.0:
                        continue

                    cat_id = labels_bundle.class_to_id[str(row[class_column])]
                    segmentation = [[x, y, x + w, y, x + w, y + h, x, y + h]]
                    annotations.append(
                        {
                            "id": ann_id,
                            "image_id": image_id,
                            "category_id": cat_id,
                            "bbox": [x, y, w, h],
                            "area": float(w * h),
                            "iscrowd": 0,
                            "segmentation": segmentation,
                        }
                    )
                    ann_id += 1

        payload["annotations"] = annotations
        payload["categories"] = labels_bundle.categories

        with split_json_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        print(
            f"[done][whole] {split}: images={len(images)}, "
            f"annotations={len(annotations)}, categories={len(labels_bundle.categories)}"
        )


def _iter_tile_origins(width: int, height: int, stride: int) -> list[tuple[int, int]]:
    return [(x, y) for y in range(0, height, stride) for x in range(0, width, stride)]


def _build_tile_window(src_width: int, src_height: int, x: int, y: int, tile_size: int, pad: bool) -> tuple[Window, int, int]:
    if pad:
        return Window(col_off=x, row_off=y, width=tile_size, height=tile_size), tile_size, tile_size

    w = min(tile_size, src_width - x)
    h = min(tile_size, src_height - y)
    return Window(col_off=x, row_off=y, width=w, height=h), int(w), int(h)


def _list_split_images(split_images_dir: Path) -> list[Path]:
    exts = {".tif", ".tiff"}
    return sorted([p for p in split_images_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])


def convert_gpkg_to_tiled_coco(
    gpkg_path: Path,
    splits_root: Path,
    tiles_root: Path,
    class_column: str = "class",
    layer: str | None = None,
    split_names: tuple[str, ...] = ("train", "val", "test"),
    tile_size: int = 1024,
    stride: int = 512,
    pad: bool = False,
    keep_empty_tiles: bool = False,
    empty_tile_fraction: float = 0.01,
    seed: int = 42,
) -> None:
    if tile_size <= 0:
        raise ValueError("tile_size must be > 0")
    if stride <= 0:
        raise ValueError("stride must be > 0")
    if not (0.0 <= empty_tile_fraction <= 1.0):
        raise ValueError("empty_tile_fraction must be in [0, 1]")

    labels_bundle = _load_labels(gpkg_path=gpkg_path, class_column=class_column, layer=layer)
    gdf = labels_bundle.gdf
    rng = random.Random(seed)

    for split in split_names:
        split_images_dir = splits_root / split / "images"
        if not split_images_dir.exists():
            print(f"[skip] Missing split images dir: {split_images_dir}")
            continue

        split_out_dir = tiles_root / split
        split_images_out = split_out_dir / "images"
        split_ann_out = split_out_dir / "annotations.coco.json"

        if split_out_dir.exists():
            shutil.rmtree(split_out_dir)
        split_images_out.mkdir(parents=True, exist_ok=True)

        images: list[dict[str, Any]] = []
        annotations: list[dict[str, Any]] = []
        image_id = 1
        ann_id = 1
        empty_candidates = 0
        empty_kept = 0

        source_images = _list_split_images(split_images_dir)
        if not source_images:
            print(f"[warn] No source images found under: {split_images_dir}")

        for src_path in source_images:
            with rasterio.open(src_path) as src:
                labels = gdf if str(gdf.crs) == str(src.crs) else gdf.to_crs(src.crs)
                src_poly = box(*src.bounds)
                labels_in_src = labels[labels.geometry.intersects(src_poly)]

                profile = src.profile.copy()
                origins = _iter_tile_origins(width=src.width, height=src.height, stride=stride)

                for x, y in origins:
                    window, tile_w, tile_h = _build_tile_window(
                        src_width=src.width,
                        src_height=src.height,
                        x=x,
                        y=y,
                        tile_size=tile_size,
                        pad=pad,
                    )

                    if tile_w <= 0 or tile_h <= 0:
                        continue

                    if pad:
                        tile = src.read(window=window, boundless=True, fill_value=0)
                    else:
                        tile = src.read(window=window)

                    tile_transform = src.window_transform(window)
                    tile_poly = box(*window_bounds(window, src.transform))

                    tile_annotations: list[dict[str, Any]] = []
                    labels_in_tile = labels_in_src[labels_in_src.geometry.intersects(tile_poly)]
                    if not labels_in_tile.empty:
                        for _, row in labels_in_tile.iterrows():
                            geom = row.geometry
                            if geom is None or geom.is_empty:
                                continue

                            clipped = geom.intersection(tile_poly)
                            if clipped.is_empty:
                                continue

                            xmin, ymin, xmax, ymax = clipped.bounds
                            if xmax <= xmin or ymax <= ymin:
                                continue

                            bx, by, bw, bh = _bbox_world_to_pixel(
                                transform=tile_transform,
                                width=tile_w,
                                height=tile_h,
                                xmin=xmin,
                                ymin=ymin,
                                xmax=xmax,
                                ymax=ymax,
                            )
                            if bw <= 0.0 or bh <= 0.0:
                                continue

                            cat_id = labels_bundle.class_to_id[str(row[class_column])]
                            segmentation = [[bx, by, bx + bw, by, bx + bw, by + bh, bx, by + bh]]
                            tile_annotations.append(
                                {
                                    "id": ann_id,
                                    "image_id": image_id,
                                    "category_id": cat_id,
                                    "bbox": [bx, by, bw, bh],
                                    "area": float(bw * bh),
                                    "iscrowd": 0,
                                    "segmentation": segmentation,
                                }
                            )
                            ann_id += 1

                    has_ann = len(tile_annotations) > 0
                    keep_tile = has_ann
                    if not has_ann:
                        empty_candidates += 1
                        if keep_empty_tiles and rng.random() < empty_tile_fraction:
                            keep_tile = True
                            empty_kept += 1

                    if not keep_tile:
                        continue

                    tile_name = f"{src_path.stem}_x{x:05d}_y{y:05d}.tif"
                    tile_path = split_images_out / tile_name
                    tile_profile = profile.copy()
                    tile_profile.update(width=tile_w, height=tile_h, transform=tile_transform)
                    # Normalize output profile so edge tiles are always writable.
                    if str(tile_profile.get("photometric", "")).upper() == "YCBCR":
                        tile_profile.pop("photometric", None)
                        tile_profile["compress"] = "lzw"
                    with rasterio.open(tile_path, "w", **tile_profile) as dst:
                        dst.write(tile)

                    images.append(
                        {
                            "id": image_id,
                            "file_name": f"images/{tile_name}",
                            "width": tile_w,
                            "height": tile_h,
                            "source_path": str(tile_path.resolve()),
                            "split": split,
                        }
                    )
                    annotations.extend(tile_annotations)
                    image_id += 1

        payload = {
            "info": {"description": f"{split} tiled split generated from {gpkg_path.name}"},
            "licenses": [],
            "images": images,
            "annotations": annotations,
            "categories": labels_bundle.categories,
        }
        with split_ann_out.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        print(
            f"[done][tiled] {split}: images={len(images)}, annotations={len(annotations)}, "
            f"categories={len(labels_bundle.categories)}, empty_kept={empty_kept}/{empty_candidates}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert bounding_box.gpkg into COCO (whole-image or tiled mode).")
    parser.add_argument("--mode", choices=["whole", "tiled"], default="whole")
    parser.add_argument("--gpkg", type=Path, default=Path("data/annotations/bounding_box.gpkg"))
    parser.add_argument("--splits-root", type=Path, default=Path("data/splits"))
    parser.add_argument("--tiles-root", type=Path, default=Path("data/tiles"))
    parser.add_argument("--class-column", type=str, default="class")
    parser.add_argument("--layer", type=str, default=None)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])

    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--pad", action="store_true", default=False)
    parser.add_argument("--keep-empty-tiles", action="store_true", default=True)
    parser.add_argument("--no-keep-empty-tiles", action="store_false", dest="keep_empty_tiles")
    parser.add_argument("--empty-tile-fraction", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.mode == "whole":
        convert_gpkg_to_split_coco(
            gpkg_path=args.gpkg,
            splits_root=args.splits_root,
            class_column=args.class_column,
            layer=args.layer,
            split_names=tuple(args.splits),
        )
    else:
        convert_gpkg_to_tiled_coco(
            gpkg_path=args.gpkg,
            splits_root=args.splits_root,
            tiles_root=args.tiles_root,
            class_column=args.class_column,
            layer=args.layer,
            split_names=tuple(args.splits),
            tile_size=args.tile_size,
            stride=args.stride,
            pad=args.pad,
            keep_empty_tiles=args.keep_empty_tiles,
            empty_tile_fraction=args.empty_tile_fraction,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
