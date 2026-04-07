import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import rasterio


def list_photos(photos_dir: Path, suffixes: Tuple[str, ...] = (".tif", ".tiff")) -> pd.DataFrame:
    photos_dir = Path(photos_dir)
    if not photos_dir.exists():
        raise FileNotFoundError(f"Photos directory not found: {photos_dir}")

    rows = []
    for path in sorted(photos_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in suffixes:
            rows.append({"filename": path.name, "source_path": str(path)})
    return pd.DataFrame(rows)


def read_photo(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read()


def get_photo_size(path: Path) -> Tuple[int, int]:
    with rasterio.open(path) as src:
        return int(src.width), int(src.height)


def is_black_photo(photo: np.ndarray, black_threshold: float = 0.98) -> bool:
    if photo.size == 0:
        return True
    return float(np.mean(photo <= 1)) >= black_threshold


def is_ocean_like_photo(
    photo: np.ndarray,
    min_blue_ratio: float = 1.05,
    max_variance: float = 150.0,
) -> bool:
    if photo.ndim != 3 or photo.shape[0] < 3:
        return False

    rgb = photo[:3].astype(np.float32)
    mean_rgb = rgb.mean(axis=(1, 2))
    red = max(float(mean_rgb[0]), 1e-6)
    green = max(float(mean_rgb[1]), 1e-6)
    blue = float(mean_rgb[2])
    variance = float(rgb.var())

    return blue / red >= min_blue_ratio and blue / green >= min_blue_ratio and variance <= max_variance


def drop_black_photos(
    photos_df: pd.DataFrame,
    source_path_col: str = "source_path",
    black_threshold: float = 0.98,
) -> pd.DataFrame:
    keep_rows = []
    for _, row in photos_df.iterrows():
        if not is_black_photo(read_photo(Path(row[source_path_col])), black_threshold=black_threshold):
            keep_rows.append(row.to_dict())
    return pd.DataFrame(keep_rows)


def drop_ocean_like_photos(
    photos_df: pd.DataFrame,
    source_path_col: str = "source_path",
    min_blue_ratio: float = 1.05,
    max_variance: float = 150.0,
) -> pd.DataFrame:
    keep_rows = []
    for _, row in photos_df.iterrows():
        if not is_ocean_like_photo(
            read_photo(Path(row[source_path_col])),
            min_blue_ratio=min_blue_ratio,
            max_variance=max_variance,
        ):
            keep_rows.append(row.to_dict())
    return pd.DataFrame(keep_rows)


def process_photo_dir(
    photos_dir: Path,
    drop_black: bool = False,
    drop_ocean_like: bool = False,
    black_threshold: float = 0.98,
    min_blue_ratio: float = 1.05,
    max_variance: float = 150.0,
) -> pd.DataFrame:
    photos_df = list_photos(photos_dir)

    if drop_black:
        photos_df = drop_black_photos(
            photos_df,
            black_threshold=black_threshold,
        )

    if drop_ocean_like:
        photos_df = drop_ocean_like_photos(
            photos_df,
            min_blue_ratio=min_blue_ratio,
            max_variance=max_variance,
        )

    if photos_df.empty:
        return pd.DataFrame(columns=["filename", "source_path", "split"])

    return photos_df.sort_values("filename").reset_index(drop=True)


def split_photos(
    photos_df: pd.DataFrame,
    ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> Dict[str, pd.DataFrame]:
    ratios_array = np.asarray(ratios, dtype=float)
    if ratios_array.shape != (3,) or np.any(ratios_array < 0) or ratios_array.sum() <= 0:
        raise ValueError("ratios must be a 3-tuple of non-negative values.")

    include_test = float(ratios_array[2]) > 0.0

    if photos_df.empty:
        empty = pd.DataFrame(columns=["filename", "source_path", "split"])
        splits = {"train": empty.copy(), "val": empty.copy()}
        if include_test:
            splits["test"] = empty.copy()
        return splits

    train_ratio, val_ratio, test_ratio = ratios_array / ratios_array.sum()
    shuffled = photos_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    n_total = len(shuffled)
    n_train = int(round(n_total * train_ratio))
    n_val = int(round(n_total * val_ratio))

    if n_total > 0 and n_train == 0:
        n_train = 1
    if n_train > n_total:
        n_train = n_total
    if n_train + n_val > n_total:
        n_val = max(0, n_total - n_train)

    train_df = shuffled.iloc[:n_train].copy()
    val_df = shuffled.iloc[n_train : n_train + n_val].copy()
    test_df = shuffled.iloc[n_train + n_val :].copy()

    splits = {
        "train": train_df.assign(split="train").sort_values("filename").reset_index(drop=True),
        "val": val_df.assign(split="val").sort_values("filename").reset_index(drop=True),
    }
    if include_test:
        splits["test"] = test_df.assign(split="test").sort_values("filename").reset_index(drop=True)
    return splits


def build_coco_images(split_df: pd.DataFrame) -> list[Dict[str, Any]]:
    images = []
    for image_id, (_, row) in enumerate(split_df.iterrows(), start=1):
        filename = str(row["filename"])
        source_path = Path(row["source_path"])
        width, height = get_photo_size(source_path)
        images.append(
            {
                "id": image_id,
                "file_name": str(Path("images") / filename),
                "width": width,
                "height": height,
                "source_path": str(source_path),
                "split": str(row["split"]),
            }
        )
    return images


def write_split_dataset(split_name: str, split_df: pd.DataFrame, output_dir: Path) -> None:
    split_dir = output_dir / split_name
    if split_dir.exists():
        shutil.rmtree(split_dir)

    images_dir = split_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    for _, row in split_df.iterrows():
        source_path = Path(row["source_path"])
        destination_path = images_dir / str(row["filename"])
        shutil.copy2(source_path, destination_path)

    coco_payload = {
        "info": {"description": f"{split_name} split created by split_photo_dir"},
        "licenses": [],
        "images": build_coco_images(split_df),
        "annotations": [],
        "categories": [],
    }

    annotations_path = split_dir / "annotations.coco.json"
    annotations_path.write_text(json.dumps(coco_payload, indent=2))


def split_photo_dir(
    photos_dir: Path,
    output_dir: Path,
    ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
    drop_black: bool = False,
    drop_ocean_like: bool = False,
    black_threshold: float = 0.98,
    min_blue_ratio: float = 1.05,
    max_variance: float = 150.0,
) -> Dict[str, pd.DataFrame]:
    processed_df = process_photo_dir(
        photos_dir=photos_dir,
        drop_black=drop_black,
        drop_ocean_like=drop_ocean_like,
        black_threshold=black_threshold,
        min_blue_ratio=min_blue_ratio,
        max_variance=max_variance,
    )
    splits = split_photos(processed_df, ratios=ratios, seed=seed)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_df in splits.items():
        write_split_dataset(split_name, split_df, output_dir)

    return splits


def main() -> None:
    parser = argparse.ArgumentParser(description="Split an orthophoto directory into train/val/test folders.")
    parser.add_argument("--photos-dir", type=Path, required=True, help="Input orthophoto directory")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output split root")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--drop-black", action="store_true")
    parser.add_argument("--drop-ocean-like", action="store_true")
    parser.add_argument("--black-threshold", type=float, default=0.98)
    parser.add_argument("--min-blue-ratio", type=float, default=1.05)
    parser.add_argument("--max-variance", type=float, default=150.0)
    args = parser.parse_args()

    ratios = (args.train_ratio, args.val_ratio, args.test_ratio)
    splits = split_photo_dir(
        photos_dir=args.photos_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        ratios=ratios,
        seed=args.seed,
        drop_black=args.drop_black,
        drop_ocean_like=args.drop_ocean_like,
        black_threshold=args.black_threshold,
        min_blue_ratio=args.min_blue_ratio,
        max_variance=args.max_variance,
    )

    for split_name, split_df in splits.items():
        print(f"[done] {split_name}: images={len(split_df)}")


if __name__ == "__main__":
    main()
