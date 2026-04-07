#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import shutil
import tempfile
from pathlib import Path

import rasterio
from shapely.geometry import box

try:
    from src.utils.data_utils import split_photo_dir
    from src.utils.gpkg_to_coco import _bbox_world_to_pixel
    from src.utils.gpkg_to_coco import _build_tile_window
    from src.utils.gpkg_to_coco import _iter_tile_origins
    from src.utils.gpkg_to_coco import _load_labels
except ModuleNotFoundError:
    from data_utils import split_photo_dir
    from gpkg_to_coco import _bbox_world_to_pixel
    from gpkg_to_coco import _build_tile_window
    from gpkg_to_coco import _iter_tile_origins
    from gpkg_to_coco import _load_labels


IMAGE_SUFFIXES = (".tif", ".tiff")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a filtered YOLO dataset directly from labeled orthophotos."
    )
    parser.add_argument("--photos-dir", type=Path, default=Path("data/labeled"))
    parser.add_argument("--gpkg", type=Path, default=Path("data/annotations/bounding_box.gpkg"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--classes",
        nargs="+",
        required=True,
        help="Use source_name or source_name:target_name, for example: tower wind_turbine:generator",
    )
    parser.add_argument("--splits", nargs="+", default=["train", "val"], choices=["train", "val", "test"])
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--max-split-attempts", type=int, default=20)
    parser.add_argument("--split-mode", choices=["photo", "tile"], default="photo")
    return parser.parse_args()


def safe_remove(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def reset_output_root(output_root: Path) -> None:
    safe_remove(output_root / "images")
    safe_remove(output_root / "labels")
    output_root.mkdir(parents=True, exist_ok=True)


def write_dataset_yaml(output_root: Path, class_names: list[str], splits: list[str]) -> None:
    yaml_path = output_root / "dataset.yaml"
    val_split = "val" if "val" in splits else "train"
    lines = [
        "train: images/train",
        f"val: images/{val_split}",
        "",
        "names:",
    ]
    if "test" in splits:
        lines.insert(2, "test: images/test")
    lines.extend(f"  {idx}: {name}" for idx, name in enumerate(class_names))
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_class_mapping(source_names: list[str], class_specs: list[str]) -> tuple[dict[str, int], list[str]]:
    source_to_target: dict[str, int] = {}
    target_names: list[str] = []
    for target_id, spec in enumerate(class_specs):
        source_name, _, target_name = spec.partition(":")
        source_name = source_name.strip()
        target_name = (target_name or source_name).strip()
        if source_name not in source_names:
            raise ValueError(f"Unknown class: {source_name}. Available classes: {source_names}")
        source_to_target[source_name] = target_id
        target_names.append(target_name)
    return source_to_target, target_names


def yolo_line(
    source_class: str,
    bbox_xywh: tuple[float, float, float, float],
    width: int,
    height: int,
    class_map: dict[str, int],
) -> str | None:
    target_class = class_map.get(source_class)
    if target_class is None:
        return None

    x, y, w, h = bbox_xywh
    if w <= 0.0 or h <= 0.0 or width <= 0 or height <= 0:
        return None

    x_center = (x + w / 2.0) / width
    y_center = (y + h / 2.0) / height
    norm_w = w / width
    norm_h = h / height
    values = [min(1.0, max(0.0, value)) for value in (x_center, y_center, norm_w, norm_h)]
    return f"{target_class} {values[0]:.6f} {values[1]:.6f} {values[2]:.6f} {values[3]:.6f}"


def list_split_images(split_images_dir: Path) -> list[Path]:
    return sorted(
        [path for path in split_images_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES]
    )


def split_class_coverage(
    split: str,
    split_root: Path,
    gdf,
    class_column: str,
    class_map: dict[str, int],
    tile_size: int,
    stride: int,
) -> set[int]:
    covered: set[int] = set()
    split_images_dir = split_root / split / "images"
    for src_path in list_split_images(split_images_dir):
        with rasterio.open(src_path) as src:
            labels = gdf if str(gdf.crs) == str(src.crs) else gdf.to_crs(src.crs)
            src_poly = box(*src.bounds)
            labels_in_src = labels[labels.geometry.intersects(src_poly)]
            if labels_in_src.empty:
                continue

            origins = _iter_tile_origins(width=src.width, height=src.height, stride=stride)
            for x, y in origins:
                window, tile_w, tile_h = _build_tile_window(
                    src_width=src.width,
                    src_height=src.height,
                    x=x,
                    y=y,
                    tile_size=tile_size,
                    pad=False,
                )
                if tile_w <= 0 or tile_h <= 0:
                    continue

                tile_transform = src.window_transform(window)
                tile_poly = box(*rasterio.windows.bounds(window, src.transform))
                labels_in_tile = labels_in_src[labels_in_src.geometry.intersects(tile_poly)]
                if labels_in_tile.empty:
                    continue

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

                    bbox_xywh = _bbox_world_to_pixel(
                        transform=tile_transform,
                        width=tile_w,
                        height=tile_h,
                        xmin=xmin,
                        ymin=ymin,
                        xmax=xmax,
                        ymax=ymax,
                    )
                    line = yolo_line(
                        source_class=str(row[class_column]),
                        bbox_xywh=bbox_xywh,
                        width=tile_w,
                        height=tile_h,
                        class_map=class_map,
                    )
                    if line is not None:
                        covered.add(class_map[str(row[class_column])])
                        if len(covered) == len(set(class_map.values())):
                            return covered
    return covered


def find_missing_split_coverage(
    *,
    split_root: Path,
    gdf,
    class_column: str,
    class_map: dict[str, int],
    tile_size: int,
    stride: int,
    splits: list[str],
) -> list[str]:
    required_splits = [split for split in splits if split in {"train", "val"}]
    missing: list[str] = []
    for split in required_splits:
        covered = split_class_coverage(
            split=split,
            split_root=split_root,
            gdf=gdf,
            class_column=class_column,
            class_map=class_map,
            tile_size=tile_size,
            stride=stride,
        )
        if len(covered) != len(set(class_map.values())):
            missing.append(f"{split}:{sorted(covered)}")
    return missing


def create_split_with_boxes(
    photos_dir: Path,
    split_root: Path,
    ratios: tuple[float, float, float],
    base_seed: int,
    max_attempts: int,
    gdf,
    class_column: str,
    class_map: dict[str, int],
    tile_size: int,
    stride: int,
    splits: list[str],
) -> int:
    for attempt in range(max_attempts):
        seed = base_seed + attempt
        safe_remove(split_root)
        split_root.mkdir(parents=True, exist_ok=True)
        split_photo_dir(
            photos_dir=photos_dir,
            output_dir=split_root,
            ratios=ratios,
            seed=seed,
        )
        missing = find_missing_split_coverage(
            split_root=split_root,
            gdf=gdf,
            class_column=class_column,
            class_map=class_map,
            tile_size=tile_size,
            stride=stride,
            splits=splits,
        )
        if not missing:
            if attempt > 0:
                print(f"[done] split retry succeeded with seed={seed}")
            return seed
        print(f"[retry] missing class coverage in {missing} with seed={seed}, reshuffling split")
    raise RuntimeError("Could not create train/val splits with all requested classes.")


def convert_split_to_yolo(
    split: str,
    split_root: Path,
    output_root: Path,
    gdf,
    class_column: str,
    class_map: dict[str, int],
    tile_size: int,
    stride: int,
    output_split: str | None = None,
) -> tuple[int, int]:
    split_images_dir = split_root / split / "images"
    target_split = output_split or split
    images_out = output_root / "images" / target_split
    labels_out = output_root / "labels" / target_split
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    kept_tiles = 0
    kept_boxes = 0
    for src_path in list_split_images(split_images_dir):
        with rasterio.open(src_path) as src:
            labels = gdf if str(gdf.crs) == str(src.crs) else gdf.to_crs(src.crs)
            src_poly = box(*src.bounds)
            labels_in_src = labels[labels.geometry.intersects(src_poly)]
            if labels_in_src.empty:
                continue

            profile = src.profile.copy()
            origins = _iter_tile_origins(width=src.width, height=src.height, stride=stride)

            for x, y in origins:
                window, tile_w, tile_h = _build_tile_window(
                    src_width=src.width,
                    src_height=src.height,
                    x=x,
                    y=y,
                    tile_size=tile_size,
                    pad=False,
                )
                if tile_w <= 0 or tile_h <= 0:
                    continue

                tile_transform = src.window_transform(window)
                tile_poly = box(*rasterio.windows.bounds(window, src.transform))
                labels_in_tile = labels_in_src[labels_in_src.geometry.intersects(tile_poly)]
                if labels_in_tile.empty:
                    continue

                label_lines: list[str] = []
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

                    bbox_xywh = _bbox_world_to_pixel(
                        transform=tile_transform,
                        width=tile_w,
                        height=tile_h,
                        xmin=xmin,
                        ymin=ymin,
                        xmax=xmax,
                        ymax=ymax,
                    )
                    line = yolo_line(
                        source_class=str(row[class_column]),
                        bbox_xywh=bbox_xywh,
                        width=tile_w,
                        height=tile_h,
                        class_map=class_map,
                    )
                    if line is not None:
                        label_lines.append(line)

                if not label_lines:
                    continue

                tile = src.read(window=window)
                tile_name = f"{src_path.stem}_x{x:05d}_y{y:05d}.tif"
                tile_path = images_out / tile_name
                label_path = labels_out / f"{Path(tile_name).stem}.txt"

                tile_profile = profile.copy()
                tile_profile.update(width=tile_w, height=tile_h, transform=tile_transform)
                if str(tile_profile.get("photometric", "")).upper() == "YCBCR":
                    tile_profile.pop("photometric", None)
                    tile_profile["compress"] = "lzw"
                with rasterio.open(tile_path, "w", **tile_profile) as dst:
                    dst.write(tile)

                label_path.write_text("\n".join(label_lines) + "\n", encoding="utf-8")
                kept_tiles += 1
                kept_boxes += len(label_lines)

    print(f"[done] {split}: kept_tiles={kept_tiles} kept_boxes={kept_boxes}")
    return kept_tiles, kept_boxes


def split_yolo_tiles(
    *,
    pool_root: Path,
    output_root: Path,
    splits: list[str],
    ratios: tuple[float, float, float],
    seed: int,
) -> dict[str, int]:
    image_pool = pool_root / "images" / "all"
    label_pool = pool_root / "labels" / "all"
    image_paths = sorted(path for path in image_pool.iterdir() if path.is_file())

    rng = random.Random(seed)
    rng.shuffle(image_paths)

    total = len(image_paths)
    requested = [split for split in ["train", "val", "test"] if split in splits]
    if total == 0:
        return {split: 0 for split in requested}

    train_ratio, val_ratio, test_ratio = ratios
    ratio_map = {"train": train_ratio, "val": val_ratio, "test": test_ratio}
    ratio_total = sum(ratio_map[split] for split in requested)
    if ratio_total <= 0:
        raise ValueError("Split ratios must sum to a positive value for the requested splits.")

    normalized = {split: ratio_map[split] / ratio_total for split in requested}
    counts: dict[str, int] = {}
    assigned = 0
    for split in requested[:-1]:
        count = int(total * normalized[split])
        counts[split] = count
        assigned += count
    counts[requested[-1]] = total - assigned

    cursor = 0
    for split in requested:
        split_images_dir = output_root / "images" / split
        split_labels_dir = output_root / "labels" / split
        split_images_dir.mkdir(parents=True, exist_ok=True)
        split_labels_dir.mkdir(parents=True, exist_ok=True)
        next_cursor = cursor + counts[split]
        for image_path in image_paths[cursor:next_cursor]:
            label_path = label_pool / f"{image_path.stem}.txt"
            shutil.copy2(image_path, split_images_dir / image_path.name)
            shutil.copy2(label_path, split_labels_dir / label_path.name)
        cursor = next_cursor

    return counts


def build_yolo_dataset(
    *,
    split_root: Path,
    output_root: Path,
    gdf,
    class_column: str,
    class_map: dict[str, int],
    tile_size: int,
    stride: int,
    splits: list[str],
) -> tuple[dict[str, int], dict[str, int]]:
    reset_output_root(output_root)

    kept_tiles_by_split: dict[str, int] = {}
    kept_boxes_by_split: dict[str, int] = {}
    for split in splits:
        kept_tiles, kept_boxes = convert_split_to_yolo(
            split=split,
            split_root=split_root,
            output_root=output_root,
            gdf=gdf,
            class_column=class_column,
            class_map=class_map,
            tile_size=tile_size,
            stride=stride,
        )
        kept_tiles_by_split[split] = kept_tiles
        kept_boxes_by_split[split] = kept_boxes
    return kept_tiles_by_split, kept_boxes_by_split


def build_yolo_dataset_by_tile(
    *,
    photos_dir: Path,
    split_root: Path,
    output_root: Path,
    gdf,
    class_column: str,
    class_map: dict[str, int],
    tile_size: int,
    stride: int,
    splits: list[str],
    ratios: tuple[float, float, float],
    seed: int,
) -> tuple[dict[str, int], dict[str, int]]:
    safe_remove(split_root)
    all_images_dir = split_root / "all" / "images"
    all_images_dir.mkdir(parents=True, exist_ok=True)
    for src_path in sorted(path for path in photos_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES):
        pool_path = all_images_dir / src_path.name
        pool_path.symlink_to(src_path.resolve())

    with tempfile.TemporaryDirectory(prefix=f"{output_root.name}_pool_") as pool_dir:
        pool_root = Path(pool_dir)
        convert_split_to_yolo(
            split="all",
            split_root=split_root,
            output_root=pool_root,
            gdf=gdf,
            class_column=class_column,
            class_map=class_map,
            tile_size=tile_size,
            stride=stride,
            output_split="all",
        )
        reset_output_root(output_root)
        kept_tiles_by_split = split_yolo_tiles(
            pool_root=pool_root,
            output_root=output_root,
            splits=splits,
            ratios=ratios,
            seed=seed,
        )
        kept_boxes_by_split = {
            split: sum(
                sum(1 for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip())
                for label_path in (output_root / "labels" / split).glob("*.txt")
            )
            for split in kept_tiles_by_split
        }
        print(
            "[done] tile split counts "
            + " ".join(f"{split}={kept_tiles_by_split.get(split, 0)}" for split in splits if split in kept_tiles_by_split)
        )
        return kept_tiles_by_split, kept_boxes_by_split


def main() -> None:
    args = parse_args()

    output_root = args.output_root.resolve()
    reset_output_root(output_root)

    labels_bundle = _load_labels(gpkg_path=args.gpkg.resolve(), class_column="class", layer=None)
    source_names = [category["name"] for category in labels_bundle.categories]
    class_map, target_names = build_class_mapping(source_names=source_names, class_specs=args.classes)
    with tempfile.TemporaryDirectory(prefix=f"{output_root.name}_splits_") as temp_dir:
        split_root = Path(temp_dir)
        used_seed = None
        total_tiles = 0
        total_boxes = 0
        required_splits = [split for split in args.splits if split in {"train", "val"}]

        if args.split_mode == "tile":
            kept_tiles_by_split, kept_boxes_by_split = build_yolo_dataset_by_tile(
                photos_dir=args.photos_dir.resolve(),
                split_root=split_root,
                output_root=output_root,
                gdf=labels_bundle.gdf,
                class_column="class",
                class_map=class_map,
                tile_size=args.tile_size,
                stride=args.stride,
                splits=args.splits,
                ratios=(args.train_ratio, args.val_ratio, args.test_ratio),
                seed=args.seed,
            )
            empty_required = [split for split in required_splits if kept_tiles_by_split.get(split, 0) == 0]
            if empty_required:
                raise RuntimeError(f"Tile split produced empty required YOLO splits: {empty_required}.")
            used_seed = args.seed
            total_tiles = sum(kept_tiles_by_split.values())
            total_boxes = sum(kept_boxes_by_split.values())
        else:
            for attempt in range(args.max_split_attempts):
                seed = args.seed + attempt
                safe_remove(split_root)
                split_root.mkdir(parents=True, exist_ok=True)
                split_photo_dir(
                    photos_dir=args.photos_dir.resolve(),
                    output_dir=split_root,
                    ratios=(args.train_ratio, args.val_ratio, args.test_ratio),
                    seed=seed,
                )
                missing = find_missing_split_coverage(
                    split_root=split_root,
                    gdf=labels_bundle.gdf,
                    class_column="class",
                    class_map=class_map,
                    tile_size=args.tile_size,
                    stride=args.stride,
                    splits=args.splits,
                )
                if missing:
                    print(f"[retry] missing class coverage in {missing} with seed={seed}, reshuffling split")
                    continue
                kept_tiles_by_split, kept_boxes_by_split = build_yolo_dataset(
                    split_root=split_root,
                    output_root=output_root,
                    gdf=labels_bundle.gdf,
                    class_column="class",
                    class_map=class_map,
                    tile_size=args.tile_size,
                    stride=args.stride,
                    splits=args.splits,
                )

                empty_required = [split for split in required_splits if kept_tiles_by_split.get(split, 0) == 0]
                if not empty_required:
                    used_seed = seed
                    total_tiles = sum(kept_tiles_by_split.values())
                    total_boxes = sum(kept_boxes_by_split.values())
                    break
                print(f"[retry] empty YOLO splits {empty_required} with seed={seed}, reshuffling split")

        if used_seed is None:
            raise RuntimeError("Could not create train/val YOLO splits with at least one tile each.")

        print(f"[done] using split seed={used_seed}")

    write_dataset_yaml(output_root=output_root, class_names=target_names, splits=args.splits)
    print(f"[ready] output={output_root} classes={target_names} total_tiles={total_tiles} total_boxes={total_boxes}")


if __name__ == "__main__":
    main()
