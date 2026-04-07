#!/usr/bin/env python3
"""Visualize saved DINO validation predictions against ground truth tiles.

Creates side-by-side images:
  - Left: ground-truth boxes
  - Right: predicted boxes from an inference JSON file
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image, ImageDraw


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_run_args(run_dir: Path) -> SimpleNamespace:
    cfg_path = run_dir / "config_args_all.json"
    payload = json.loads(cfg_path.read_text())
    return SimpleNamespace(**payload)


def _build_dataset(run_args: SimpleNamespace):
    # Import DINO dataset code by adding its source root to sys.path.
    sys.path.insert(0, str(_repo_root() / "src" / "models" / "DINO"))
    from datasets import build_dataset  # type: ignore

    run_args.device = "cpu"
    run_args.distributed = False
    run_args.rank = 0
    run_args.world_size = 1
    run_args.local_rank = 0
    run_args.gpu = 0
    run_args.num_workers = 0
    run_args.eval = True
    run_args.test = False

    dataset_val = build_dataset(image_set="val", args=run_args)
    return dataset_val


def _denorm_to_pil(img_tensor: torch.Tensor) -> Image.Image:
    # img_tensor is normalized CHW tensor.
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=img_tensor.dtype).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=img_tensor.dtype).view(3, 1, 1)
    img = (img_tensor.cpu() * std + mean).clamp(0.0, 1.0)
    img_u8 = (img.permute(1, 2, 0).numpy() * 255.0).astype("uint8")
    return Image.fromarray(img_u8)


def _cxcywh_to_xyxy_abs(boxes: torch.Tensor, w: int, h: int) -> torch.Tensor:
    scale = torch.tensor([w, h, w, h], dtype=boxes.dtype)
    b = boxes * scale
    cx, cy, bw, bh = b.unbind(dim=1)
    x0 = cx - bw / 2.0
    y0 = cy - bh / 2.0
    x1 = cx + bw / 2.0
    y1 = cy + bh / 2.0
    return torch.stack([x0, y0, x1, y1], dim=1)


def _draw_boxes(
    image: Image.Image,
    boxes: list[list[float]],
    labels: list[int],
    scores: list[float] | None,
    color: str,
    id_to_name: dict[int, str],
) -> Image.Image:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    for i, box in enumerate(boxes):
        x0, y0, x1, y1 = box
        draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
        cls_id = int(labels[i])
        cls_name = id_to_name.get(cls_id, f"id:{cls_id}")
        if scores is None:
            text = cls_name
        else:
            text = f"{cls_name} {scores[i]:.2f}"
        tx, ty = max(0.0, x0), max(0.0, y0 - 11.0)
        draw.text((tx, ty), text, fill=color)
    return canvas


def _concat_lr(left: Image.Image, right: Image.Image) -> Image.Image:
    out = Image.new("RGB", (left.width + right.width, max(left.height, right.height)))
    out.paste(left, (0, 0))
    out.paste(right, (left.width, 0))
    return out


def _load_predictions(prediction_path: Path) -> dict[int, dict]:
    # Index predictions by image_id so dataset samples can look them up quickly.
    payload = json.loads(prediction_path.read_text())
    return {int(item["image_id"]): item for item in payload}


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize saved DINO predictions vs GT.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--coco-path", type=Path, required=True)
    parser.add_argument("--prediction-json", type=Path, required=True)
    parser.add_argument("--score-thr", type=float, default=0.30)
    parser.add_argument("--max-images", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    run_dir = (_repo_root() / args.run_dir).resolve() if not args.run_dir.is_absolute() else args.run_dir
    out_dir = (_repo_root() / args.out_dir).resolve() if not args.out_dir.is_absolute() else args.out_dir
    coco_path = (_repo_root() / args.coco_path).resolve() if not args.coco_path.is_absolute() else args.coco_path
    prediction_json = (
        (_repo_root() / args.prediction_json).resolve()
        if not args.prediction_json.is_absolute()
        else args.prediction_json
    )

    run_args = _load_run_args(run_dir)
    run_args.coco_path = str(coco_path)
    run_args.output_dir = str(run_dir)
    dataset_val = _build_dataset(run_args)
    predictions_by_image = _load_predictions(prediction_json)

    ann_path = coco_path / "annotations" / "instances_val2017.json"
    ann = json.loads(ann_path.read_text())
    id_to_name = {int(c["id"]): str(c["name"]) for c in ann.get("categories", [])}
    id_to_name.setdefault(0, "bg")

    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    indices = list(range(len(dataset_val)))
    rng.shuffle(indices)
    indices = indices[: max(0, args.max_images)]

    print(f"Dataset size: {len(dataset_val)}")
    print(f"Rendering {len(indices)} samples -> {out_dir}")

    for i, idx in enumerate(indices, start=1):
        img_tensor, target = dataset_val[idx]
        image = _denorm_to_pil(img_tensor)
        w, h = image.size

        # Ground truth boxes are stored as normalized cxcywh in the dataset.
        gt_xyxy = _cxcywh_to_xyxy_abs(target["boxes"].cpu(), w=w, h=h)
        gt_boxes = gt_xyxy.tolist()
        gt_labels = target["labels"].cpu().tolist()

        image_id = int(target["image_id"].item())
        pred = predictions_by_image.get(image_id, {})
        pred_boxes = pred.get("boxes", [])
        pred_labels = pred.get("labels", [])
        pred_scores = pred.get("scores", [])

        # The inference JSON may already be filtered, but this keeps visualization flexible.
        kept_boxes = []
        kept_labels = []
        kept_scores = []
        for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
            if score >= args.score_thr:
                kept_boxes.append(box)
                kept_labels.append(int(label))
                kept_scores.append(float(score))

        gt_img = _draw_boxes(
            image=image,
            boxes=gt_boxes,
            labels=gt_labels,
            scores=None,
            color="lime",
            id_to_name=id_to_name,
        )
        pred_img = _draw_boxes(
            image=image,
            boxes=kept_boxes,
            labels=kept_labels,
            scores=kept_scores,
            color="red",
            id_to_name=id_to_name,
        )

        merged = _concat_lr(gt_img, pred_img)
        out_path = out_dir / f"{i:03d}_imageid_{image_id}.png"
        merged.save(out_path)

    print(f"Done. Saved visualizations to: {out_dir}")


if __name__ == "__main__":
    main()
