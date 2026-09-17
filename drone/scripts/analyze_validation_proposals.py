#!/usr/bin/env python3

from __future__ import annotations

import statistics
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]

SEQUENCE = (
    ROOT
    / "drone"
    / "captures"
    / "9204f05e8ffe46f995edd8c093823393"
    / "frames"
)

MODEL_PATH = (
    ROOT
    / "drone"
    / "artifacts"
    / "exp_d010"
    / "runs"
    / "yolo11n_tiled_objectness"
    / "weights"
    / "best.pt"
)

VIEW_W = 960
VIEW_H = 540

YOLO_CONF = 0.025
COLS = 2
ROWS = 2
OVERLAP = 0.20
NMS_IOU = 0.50


def iou(a, b):
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))

    inter = (
        max(0.0, x2 - x1)
        * max(0.0, y2 - y1)
    )

    area_a = (
        max(0.0, float(a[2] - a[0]))
        * max(0.0, float(a[3] - a[1]))
    )

    area_b = (
        max(0.0, float(b[2] - b[0]))
        * max(0.0, float(b[3] - b[1]))
    )

    union = area_a + area_b - inter

    return (
        inter / union
        if union > 0
        else 0.0
    )


def nms(boxes, scores):
    if not boxes:
        return [], []

    order = np.argsort(
        np.asarray(scores)
    )[::-1]

    kept_boxes = []
    kept_scores = []

    while len(order):
        i = int(order[0])

        kept_boxes.append(
            boxes[i]
        )

        kept_scores.append(
            scores[i]
        )

        remaining = []

        for j in order[1:]:
            j = int(j)

            if iou(
                boxes[i],
                boxes[j],
            ) < NMS_IOU:
                remaining.append(
                    j
                )

        order = np.asarray(
            remaining,
            dtype=np.int64,
        )

    return kept_boxes, kept_scores


def make_starts(
    length,
    count,
):
    tile_size = int(
        round(
            (length / count)
            * (1.0 + OVERLAP)
        )
    )

    starts = np.linspace(
        0,
        length - tile_size,
        count,
    )

    return [
        (
            int(round(start)),
            int(round(start))
            + tile_size,
        )
        for start in starts
    ]


def make_tiles(image):
    xs = make_starts(
        VIEW_W,
        COLS,
    )

    ys = make_starts(
        VIEW_H,
        ROWS,
    )

    for y1, y2 in ys:
        for x1, x2 in xs:
            yield (
                image[
                    y1:y2,
                    x1:x2,
                ],
                x1,
                y1,
            )


def main():
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    paths = sorted(
        SEQUENCE.glob("*.png")
    )

    print(
        f"Frames: {len(paths)}"
    )

    print(
        f"Device: {device}"
    )

    model = YOLO(
        str(MODEL_PATH)
    )

    first = cv2.imread(
        str(paths[0])
    )

    # Warm all tile shapes.
    for tile, _, _ in make_tiles(
        first
    ):
        model.predict(
            tile,
            imgsz=960,
            conf=YOLO_CONF,
            iou=0.70,
            max_det=300,
            device=device,
            verbose=False,
        )

    counts = []
    latencies = []

    over_16 = 0
    over_32 = 0
    over_48 = 0

    for number, path in enumerate(
        paths,
        start=1,
    ):
        image = cv2.imread(
            str(path)
        )

        boxes = []
        scores = []

        if device == "mps":
            torch.mps.synchronize()

        start = time.perf_counter()

        for tile, offset_x, offset_y in make_tiles(
            image
        ):
            result = model.predict(
                tile,
                imgsz=960,
                conf=YOLO_CONF,
                iou=0.70,
                max_det=300,
                device=device,
                verbose=False,
            )[0]

            if result.boxes is None:
                continue

            tile_boxes = (
                result.boxes.xyxy
                .detach()
                .cpu()
                .numpy()
            )

            tile_scores = (
                result.boxes.conf
                .detach()
                .cpu()
                .numpy()
            )

            for box, score in zip(
                tile_boxes,
                tile_scores,
            ):
                boxes.append(
                    np.asarray(
                        [
                            float(box[0])
                            + offset_x,
                            float(box[1])
                            + offset_y,
                            float(box[2])
                            + offset_x,
                            float(box[3])
                            + offset_y,
                        ],
                        dtype=np.float32,
                    )
                )

                scores.append(
                    float(score)
                )

        boxes, scores = nms(
            boxes,
            scores,
        )

        if device == "mps":
            torch.mps.synchronize()

        latency = (
            time.perf_counter()
            - start
        ) * 1000.0

        count = len(boxes)

        counts.append(count)
        latencies.append(latency)

        over_16 += count > 16
        over_32 += count > 32
        over_48 += count > 48

        if (
            number <= 10
            or count > 32
        ):
            print(
                f"{path.stem}: "
                f"proposals={count:3d} "
                f"yolo={latency:6.1f} ms"
            )

    print()
    print("=" * 72)
    print("VALIDATION PROPOSAL SUMMARY")
    print("=" * 72)

    print(
        f"Mean proposals:   "
        f"{statistics.mean(counts):.1f}"
    )

    print(
        f"Median proposals: "
        f"{statistics.median(counts):.1f}"
    )

    print(
        f"P95 proposals:    "
        f"{np.percentile(counts, 95):.1f}"
    )

    print(
        f"Max proposals:    "
        f"{max(counts)}"
    )

    print()

    print(
        f"Frames >16 proposals: "
        f"{over_16}/{len(counts)}"
    )

    print(
        f"Frames >32 proposals: "
        f"{over_32}/{len(counts)}"
    )

    print(
        f"Frames >48 proposals: "
        f"{over_48}/{len(counts)}"
    )

    print()

    print(
        f"YOLO median: "
        f"{statistics.median(latencies):.1f} ms"
    )

    print(
        f"YOLO p95: "
        f"{np.percentile(latencies, 95):.1f} ms"
    )


if __name__ == "__main__":
    main()
