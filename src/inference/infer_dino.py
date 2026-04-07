#!/usr/bin/env python3
"""Run simple DINO inference on a COCO-format validation set.

This script:
1. Loads a trained run from a run directory
2. Builds the validation dataset from a COCO directory
3. Runs inference on every validation sample
4. Saves predictions to a JSON file
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def dino_root() -> Path:
    return repo_root() / "src" / "models" / "DINO"


def load_run_args(run_dir: Path) -> SimpleNamespace:
    # Reuse the saved training config so model creation matches the trained run.
    config_path = run_dir / "config_args_all.json"
    payload = json.loads(config_path.read_text())
    return SimpleNamespace(**payload)


def build_model_and_dataset(run_args: SimpleNamespace):
    # Import DINO from the vendored source tree.
    sys.path.insert(0, str(dino_root()))
    from main import build_model_main  # type: ignore
    from datasets import build_dataset  # type: ignore
    from util.misc import clean_state_dict  # type: ignore

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Keep runtime settings minimal for single-process inference.
    run_args.device = str(device)
    run_args.distributed = False
    run_args.rank = 0
    run_args.world_size = 1
    run_args.local_rank = 0
    run_args.gpu = 0
    run_args.num_workers = 0
    run_args.eval = True
    run_args.test = False

    model, _, postprocessors = build_model_main(run_args)
    model.to(device)
    model.eval()

    dataset_val = build_dataset(image_set="val", args=run_args)
    return model, postprocessors, dataset_val, clean_state_dict, device


def tensor_to_list(tensor: torch.Tensor) -> list[float] | list[list[float]]:
    return tensor.detach().cpu().tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DINO inference on a validation set.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Training run directory")
    parser.add_argument("--data-path", type=Path, required=True, help="COCO-format dataset directory")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=repo_root() / "inference",
        help="Directory or file path for predictions output",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoint_best_regular.pth",
        help="Checkpoint filename inside run-dir",
    )
    parser.add_argument(
        "--score-thr",
        type=float,
        default=0.30,
        help="Keep predictions at or above this score",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    data_path = args.data_path.resolve()
    checkpoint_path = run_dir / args.checkpoint

    # If a directory is given, write a standard predictions filename inside it.
    output_path = args.output_path.resolve()
    if output_path.suffix != ".json":
        output_path = output_path / "predictions.json"

    # Load the saved run configuration, then point it to the requested dataset.
    run_args = load_run_args(run_dir)
    run_args.coco_path = str(data_path)
    run_args.output_dir = str(run_dir)

    model, postprocessors, dataset_val, clean_state_dict, device = build_model_and_dataset(run_args)

    # Restore the trained weights before running inference.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(clean_state_dict(state_dict), strict=False)
    model.eval()

    predictions = []

    with torch.no_grad():
        for image_tensor, target in dataset_val:
            # The model expects a batch dimension, even for a single image.
            inputs = image_tensor.unsqueeze(0).to(device)
            target_sizes = target["size"].unsqueeze(0).to(device)

            outputs = model(inputs)
            result = postprocessors["bbox"](outputs, target_sizes)[0]

            scores = result["scores"].cpu()
            keep = scores >= args.score_thr

            predictions.append(
                {
                    "image_id": int(target["image_id"].item()),
                    "boxes": tensor_to_list(result["boxes"][keep]),
                    "labels": tensor_to_list(result["labels"][keep]),
                    "scores": tensor_to_list(result["scores"][keep]),
                }
            )

    # Save one simple JSON file so downstream scripts can read predictions easily.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(predictions, indent=2))


if __name__ == "__main__":
    main()
