#!/usr/bin/env python3

from __future__ import annotations

import json
import statistics
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = (
    ROOT
    / "drone"
    / "reference"
    / "official-drone-flyby"
    / "src"
    / "helsinki"
)

IMAGE_DIR = DATA_ROOT / "images"
ANNOTATION_DIR = DATA_ROOT / "annotations"

MODEL_PATH = (
    ROOT
    / "drone"
    / "artifacts"
    / "exp_d007"
    / "runs"
    / "yolo11n_objectness"
    / "weights"
    / "best.pt"
)

ARTIFACT_DIR = (
    ROOT
    / "drone"
    / "artifacts"
    / "exp_d009"
)

RESULT_PATH = ARTIFACT_DIR / "tiled_objectness.json"

VIEW_W = 960
VIEW_H = 540

SOURCE_W = 3840
SOURCE_H = 2160

CONF = 0.01
NMS_IOU = 0.70
MAX_DET = 300

VAL_FRAMES = {22, 23, 24}

CONFIGS = {
    "full": {
        "cols": 1,
        "rows": 1,
        "overlap": 0.0,
    },
    "2x2_o20": {
        "cols": 2,
        "rows": 2,
        "overlap": 0.20,
    },
    "3x2_o20": {
        "cols": 3,
        "rows": 2,
        "overlap": 0.20,
    },
}


def find_image(frame: int) -> Path:
    for suffix in ("png", "jpg", "jpeg"):
        p = IMAGE_DIR / f"frame_{frame:06d}.{suffix}"
        if p.exists():
            return p

    matches = sorted(IMAGE_DIR.glob(f"*{frame:06d}*"))

    if matches:
        return matches[0]

    raise FileNotFoundError(frame)


def source_bbox_to_l0(bbox):
    x1, y1, x2, y2 = map(float, bbox)

    return np.asarray(
        [
            x1 / 4.0,
            y1 / 4.0,
            x2 / 4.0,
            y2 / 4.0,
        ],
        dtype=np.float32,
    )


def iou(a, b):
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)

    area_a = max(0.0, float(a[2] - a[0])) * max(
        0.0, float(a[3] - a[1])
    )

    area_b = max(0.0, float(b[2] - b[0])) * max(
        0.0, float(b[3] - b[1])
    )

    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def make_starts(length, count, overlap):
    if count == 1:
        return [(0, length)]

    nominal = length / count
    tile_size = int(round(nominal * (1.0 + overlap)))

    tile_size = min(length, tile_size)

    starts = np.linspace(
        0,
        length - tile_size,
        count,
    )

    return [
        (
            int(round(start)),
            int(round(start)) + tile_size,
        )
        for start in starts
    ]


def make_tiles(image, config):
    xs = make_starts(
        VIEW_W,
        config["cols"],
        config["overlap"],
    )

    ys = make_starts(
        VIEW_H,
        config["rows"],
        config["overlap"],
    )

    tiles = []

    for y1, y2 in ys:
        for x1, x2 in xs:
            tiles.append(
                {
                    "image": image[y1:y2, x1:x2],
                    "x1": x1,
                    "y1": y1,
                }
            )

    return tiles


def predict_tiles(model, image, config, device):
    tiles = make_tiles(image, config)

    proposals = []

    start = time.perf_counter()

    for tile in tiles:
        result = model.predict(
            source=tile["image"],
            imgsz=960,
            conf=CONF,
            iou=NMS_IOU,
            max_det=MAX_DET,
            device=device,
            verbose=False,
        )[0]

        if result.boxes is None:
            continue

        boxes = (
            result.boxes.xyxy
            .detach()
            .cpu()
            .numpy()
        )

        for box in boxes:
            mapped = np.asarray(
                [
                    box[0] + tile["x1"],
                    box[1] + tile["y1"],
                    box[2] + tile["x1"],
                    box[3] + tile["y1"],
                ],
                dtype=np.float32,
            )

            proposals.append(mapped)

    if device == "mps":
        torch.mps.synchronize()

    latency_ms = (time.perf_counter() - start) * 1000.0

    return proposals, latency_ms, len(tiles)


def evaluate(records):
    total = 0
    hit30 = 0
    hit50 = 0

    class_total = Counter()
    class_hit30 = Counter()
    class_hit50 = Counter()

    for record in records:
        for gt in record["gt"]:
            total += 1

            cls = gt["class"]
            class_total[cls] += 1

            if record["proposals"]:
                best = max(
                    iou(gt["bbox"], proposal)
                    for proposal in record["proposals"]
                )
            else:
                best = 0.0

            if best >= 0.30:
                hit30 += 1
                class_hit30[cls] += 1

            if best >= 0.50:
                hit50 += 1
                class_hit50[cls] += 1

    per_class = {}

    for cls in sorted(class_total):
        per_class[cls] = {
            "r30": class_hit30[cls] / class_total[cls],
            "r50": class_hit50[cls] / class_total[cls],
        }

    return {
        "instances": total,
        "r30": hit30 / total,
        "r50": hit50 / total,
        "per_class": per_class,
    }


def main():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    model = YOLO(str(MODEL_PATH))

    frames = []

    for path in sorted(ANNOTATION_DIR.glob("frame_*.json")):
        data = json.loads(path.read_text())

        frame = int(data["frame"])

        source = cv2.imread(str(find_image(frame)))

        view = cv2.resize(
            source,
            (VIEW_W, VIEW_H),
            interpolation=cv2.INTER_AREA,
        )

        gt = [
            {
                "class": annotation["object_id"],
                "bbox": source_bbox_to_l0(annotation["bbox"]),
            }
            for annotation in data["annotations"]
        ]

        frames.append(
            {
                "frame": frame,
                "image": view,
                "gt": gt,
            }
        )

    # Warm up once.
    for _ in range(3):
        model.predict(
            source=frames[0]["image"],
            imgsz=960,
            conf=CONF,
            device=device,
            verbose=False,
        )

    results = {}

    for name, config in CONFIGS.items():
        print()
        print("=" * 72)
        print(name)
        print("=" * 72)

        records = []
        latencies = []
        counts = []
        tile_counts = []

        for frame in frames:
            proposals, latency, tiles = predict_tiles(
                model,
                frame["image"],
                config,
                device,
            )

            latencies.append(latency)
            counts.append(len(proposals))
            tile_counts.append(tiles)

            records.append(
                {
                    "frame": frame["frame"],
                    "gt": frame["gt"],
                    "proposals": proposals,
                }
            )

        val_records = [
            record
            for record in records
            if record["frame"] in VAL_FRAMES
        ]

        all_eval = evaluate(records)
        val_eval = evaluate(val_records)

        result = {
            "tiles_per_frame": tile_counts[0],
            "mean_proposals": float(statistics.mean(counts)),
            "median_latency_ms": float(statistics.median(latencies)),
            "p95_latency_ms": float(np.percentile(latencies, 95)),
            "all": all_eval,
            "holdout": val_eval,
        }

        results[name] = result

        print(
            f"Tiles/frame:       {result['tiles_per_frame']}"
        )

        print(
            f"Proposals/frame:   {result['mean_proposals']:.1f}"
        )

        print(
            f"Latency median:    {result['median_latency_ms']:.1f} ms"
        )

        print(
            f"Latency p95:       {result['p95_latency_ms']:.1f} ms"
        )

        print(
            f"All recall @0.50:  {all_eval['r50']:.4f}"
        )

        print(
            f"Holdout @0.30:     {val_eval['r30']:.4f}"
        )

        print(
            f"Holdout @0.50:     {val_eval['r50']:.4f}"
        )

    print()
    print("=" * 72)
    print("HOLDOUT PER-CLASS")
    print("=" * 72)

    classes = sorted(
        {
            cls
            for result in results.values()
            for cls in result["holdout"]["per_class"]
        }
    )

    print(
        f"{'class':20s}"
        + "".join(
            f"{name:>14s}"
            for name in CONFIGS
        )
    )

    print("-" * (20 + 14 * len(CONFIGS)))

    for cls in classes:
        line = f"{cls:20s}"

        for name in CONFIGS:
            recall = (
                results[name]
                ["holdout"]
                ["per_class"]
                .get(cls, {})
                .get("r50", 0.0)
            )

            line += f"{recall:14.3f}"

        print(line)

    safe_results = json.loads(
        json.dumps(
            results,
            default=lambda obj:
                obj.item()
                if isinstance(obj, np.generic)
                else obj.tolist()
                if isinstance(obj, np.ndarray)
                else str(obj),
        )
    )

    RESULT_PATH.write_text(
        json.dumps(
            safe_results,
            indent=2,
        )
    )

    print()
    print(f"Saved: {RESULT_PATH}")


if __name__ == "__main__":
    main()
