from pathlib import Path
import shutil
import os
import json
import argparse

# Run from project root:
#   python src/utils/make_cocodir.py --src-root data/splits
#   python src/utils/make_cocodir.py --src-root data/tiles
#
# Expected input structure:
# data/
#   splits/
#     train/
#       images/
#       annotations.coco.json
#     val/
#       images/
#       annotations.coco.json
#     test/                     # optional
#       images/
#       annotations.coco.json
#
# Output:
# data/
#   COCODIR/
#     train2017/               # symlink or copied dir
#     val2017/
#     test2017/                # optional
#     annotations/
#       instances_train.json
#       instances_val2017.json
#       instances_test2017.json  # optional


USE_SYMLINKS = True   # set False if you want to copy image folders instead

PROJECT_ROOT = Path.cwd()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_remove(path: Path) -> None:
    if path.is_symlink():
        path.unlink()
    elif path.exists():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def link_or_copy_dir(src: Path, dst: Path) -> None:
    if not src.exists():
        print(f"[skip] missing source dir: {src}")
        return

    if dst.exists() or dst.is_symlink():
        print(f"[remove] existing destination: {dst}")
        safe_remove(dst)

    if USE_SYMLINKS:
        os.symlink(src, dst, target_is_directory=True)
        print(f"[link] {dst} -> {src}")
    else:
        shutil.copytree(src, dst)
        print(f"[copy] {src} -> {dst}")


def copy_json(src: Path, dst: Path) -> None:
    if not src.exists():
        print(f"[skip] missing annotation file: {src}")
        return
    with src.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    # DINO's COCO loader joins `img_folder` and `file_name`.
    # Our split JSON uses "images/<name>", but COCODIR's img_folder already
    # points at ".../train2017" (the images dir), so keep only basename.
    for image in payload.get("images", []):
        image["file_name"] = Path(image["file_name"]).name

    with dst.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[rewrite] {src} -> {dst}")


def process_split(src_root: Path, dst_root: Path, split_name: str, coco_name: str) -> None:
    split_root = src_root / split_name
    images_src = split_root / "images"
    ann_src = split_root / "annotations.coco.json"

    images_dst = dst_root / coco_name
    ann_dst = (dst_root / "annotations") / f"instances_{coco_name}.json"

    link_or_copy_dir(images_src, images_dst)
    copy_json(ann_src, ann_dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DINO-compatible COCODIR from split folders.")
    parser.add_argument("--src-root", type=Path, default=Path("data/splits"), help="Source split root (e.g. data/splits or data/tiles)")
    parser.add_argument("--dst-root", type=Path, default=Path("data/COCODIR"), help="Destination COCODIR root")
    parser.add_argument("--copy", action="store_true", help="Copy directories instead of creating symlinks")
    args = parser.parse_args()

    src_root = (PROJECT_ROOT / args.src_root).resolve() if not args.src_root.is_absolute() else args.src_root
    dst_root = (PROJECT_ROOT / args.dst_root).resolve() if not args.dst_root.is_absolute() else args.dst_root
    ann_root = dst_root / "annotations"

    global USE_SYMLINKS
    USE_SYMLINKS = not args.copy

    if not src_root.exists():
        raise FileNotFoundError(f"Source root not found: {src_root}")

    ensure_dir(ann_root)

    process_split(src_root, dst_root, "train", "train2017")
    process_split(src_root, dst_root, "val", "val2017")
    process_split(src_root, dst_root, "test", "test2017")  # optional; skipped if missing

    print("\nDone.")
    print(f"Source root: {src_root}")
    print(f"COCO root: {dst_root}")


if __name__ == "__main__":
    main()
