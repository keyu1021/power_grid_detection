from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any


def _safe_remove(path: Path) -> None:
    if path.is_symlink():
        path.unlink()
    elif path.exists():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def _link_or_copy_file(src: Path, dst: Path, copy_images: bool) -> None:
    _safe_remove(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy_images:
        shutil.copy2(src, dst)
    else:
        os.symlink(src, dst)


def _coco_category_map(categories: list[dict[str, Any]]) -> tuple[dict[int, int], list[str]]:
    ordered = sorted(categories, key=lambda category: int(category["id"]))
    cat_id_to_yolo = {int(category["id"]): idx for idx, category in enumerate(ordered)}
    names = [str(category["name"]) for category in ordered]
    return cat_id_to_yolo, names


def _annotation_lines(
    image: dict[str, Any],
    annotations: list[dict[str, Any]],
    cat_id_to_yolo: dict[int, int],
) -> list[str]:
    width = float(image.get("width", 0))
    height = float(image.get("height", 0))
    if width <= 0 or height <= 0:
        return []

    lines: list[str] = []
    for ann in annotations:
        bbox = ann.get("bbox")
        if not bbox or len(bbox) != 4:
            continue

        x, y, w, h = map(float, bbox)
        if w <= 0 or h <= 0:
            continue

        x_center = (x + w / 2.0) / width
        y_center = (y + h / 2.0) / height
        norm_w = w / width
        norm_h = h / height

        values = [x_center, y_center, norm_w, norm_h]
        values = [min(1.0, max(0.0, value)) for value in values]

        cls = cat_id_to_yolo[int(ann["category_id"])]
        lines.append(f"{cls} {values[0]:.6f} {values[1]:.6f} {values[2]:.6f} {values[3]:.6f}")
    return lines


def convert_split(
    images_dir: Path,
    annotations_path: Path,
    output_root: Path,
    split_name: str,
    copy_images: bool,
) -> list[str]:
    if not images_dir.exists() or not annotations_path.exists():
        print(f"[skip] {split_name}: missing images or annotations")
        return []

    payload = json.loads(annotations_path.read_text(encoding="utf-8"))
    images = payload.get("images", [])
    annotations = payload.get("annotations", [])
    categories = payload.get("categories", [])
    cat_id_to_yolo, class_names = _coco_category_map(categories)

    anns_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in annotations:
        anns_by_image[int(ann["image_id"])].append(ann)

    split_images_out = output_root / "images" / split_name
    split_labels_out = output_root / "labels" / split_name
    _safe_remove(split_images_out)
    _safe_remove(split_labels_out)
    split_images_out.mkdir(parents=True, exist_ok=True)
    split_labels_out.mkdir(parents=True, exist_ok=True)

    kept_images = 0
    kept_annotations = 0
    for image in images:
        image_id = int(image["id"])
        image_name = Path(str(image["file_name"])).name
        image_path = images_dir / image_name
        if not image_path.exists():
            continue

        label_lines = _annotation_lines(image, anns_by_image.get(image_id, []), cat_id_to_yolo)
        if not label_lines:
            continue

        _link_or_copy_file(image_path, split_images_out / image_name, copy_images=copy_images)
        label_path = split_labels_out / f"{Path(image_name).stem}.txt"
        label_path.write_text("\n".join(label_lines), encoding="utf-8")

        kept_images += 1
        kept_annotations += len(label_lines)

    print(
        f"[done] {split_name}: kept_images={kept_images} kept_annotations={kept_annotations} "
        f"classes={len(class_names)}"
    )
    return class_names


def write_dataset_yaml(output_root: Path, class_names: list[str]) -> Path:
    yaml_path = output_root / "dataset.yaml"
    lines = [
        "train: images/train",
        "val: images/val",
        "",
        "names:",
    ]
    lines.extend([f"  {idx}: {name}" for idx, name in enumerate(class_names)])
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return yaml_path


def _split_layout(split_root: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    return (
        {
            "train": split_root / "train" / "images",
            "val": split_root / "val" / "images",
            "test": split_root / "test" / "images",
        },
        {
            "train": split_root / "train" / "annotations.coco.json",
            "val": split_root / "val" / "annotations.coco.json",
            "test": split_root / "test" / "annotations.coco.json",
        },
    )


def _coco_layout(coco_root: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    return (
        {
            "train": coco_root / "train2017",
            "val": coco_root / "val2017",
            "test": coco_root / "test2017",
        },
        {
            "train": coco_root / "annotations" / "instances_train2017.json",
            "val": coco_root / "annotations" / "instances_val2017.json",
            "test": coco_root / "annotations" / "instances_test2017.json",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert COCO annotations into YOLO format.")
    parser.add_argument(
        "--coco-root",
        type=Path,
        help="COCO dataset root containing train2017/ val2017/ and annotations/",
    )
    parser.add_argument(
        "--split-root",
        type=Path,
        help="Split dataset root containing train|val|test/images and annotations.coco.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/yolo"),
        help="Destination YOLO dataset root",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        choices=["train", "val", "test"],
        help="Dataset splits to convert",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy images instead of creating symlinks",
    )
    args = parser.parse_args()

    if bool(args.coco_root) == bool(args.split_root):
        raise ValueError("Provide exactly one of --coco-root or --split-root.")

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if args.split_root:
        split_to_images_dir, split_to_annotations = _split_layout(args.split_root.resolve())
    else:
        split_to_images_dir, split_to_annotations = _coco_layout(args.coco_root.resolve())

    class_names: list[str] = []
    for split_name in args.splits:
        current_names = convert_split(
            images_dir=split_to_images_dir[split_name],
            annotations_path=split_to_annotations[split_name],
            output_root=output_root,
            split_name=split_name,
            copy_images=args.copy_images,
        )
        if current_names and not class_names:
            class_names = current_names

    if not class_names:
        print("[skip] no labeled images were converted")
        return

    yaml_path = write_dataset_yaml(output_root=output_root, class_names=class_names)
    print(f"[done] dataset yaml written to {yaml_path}")


if __name__ == "__main__":
    main()
