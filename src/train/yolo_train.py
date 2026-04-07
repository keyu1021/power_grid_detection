from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YOLO detection model.")
    parser.add_argument("--model", type=Path, required=True, help="Path to a YOLO checkpoint, e.g. yolov8n.pt")
    parser.add_argument("--data", type=Path, required=True, help="Path to the YOLO dataset YAML file")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=1024, help="Training image size")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--device", default="0", help="Device string understood by Ultralytics")
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "2"))),
        help="Number of dataloader workers",
    )
    parser.add_argument("--project", type=Path, required=True, help="Output project directory")
    parser.add_argument("--name", default="train", help="Run name inside the project directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.data.exists():
        raise FileNotFoundError(f"Missing dataset YAML: {args.data}")
    if not args.model.exists():
        raise FileNotFoundError(f"Missing YOLO model checkpoint: {args.model}")

    args.project.mkdir(parents=True, exist_ok=True)

    dataset_yaml = args.data
    with args.data.open("r", encoding="utf-8") as handle:
        dataset_config = yaml.safe_load(handle)

    dataset_root = Path(dataset_config.get("path") or args.data.parent)
    if not dataset_root.is_absolute():
        dataset_root = (args.data.parent / dataset_root).resolve()

    val_path = dataset_config.get("val")
    if isinstance(val_path, str):
        resolved_val = (dataset_root / val_path).resolve()
        if not resolved_val.exists():
            train_path = dataset_config.get("train")
            if not isinstance(train_path, str):
                raise FileNotFoundError(f"Missing validation path and invalid train path in dataset YAML: {args.data}")
            dataset_config["val"] = train_path
            with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as temp_handle:
                yaml.safe_dump(dataset_config, temp_handle, sort_keys=False)
                dataset_yaml = Path(temp_handle.name)
            print(
                f"Validation path {resolved_val} not found; "
                f"falling back to train split for validation using {dataset_yaml}"
            )

    from ultralytics import YOLO

    model = YOLO(str(args.model))
    model.train(
        data=str(dataset_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
