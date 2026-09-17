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
    / "exp_d010"
    / "runs"
    / "yolo11n_tiled_objectness"
    / "weights"
    / "best.pt"
)

ARTIFACT_DIR = (
    ROOT
    / "drone"
    / "artifacts"
    / "exp_d011"
)

RESULT_PATH = (
    ARTIFACT_DIR
    / "mapped_tiled_results.json"
)

VIEW_W = 960
VIEW_H = 540

COLS = 2
ROWS = 2
OVERLAP = 0.20

CONF_THRESHOLDS = [
    0.005,
    0.01,
    0.025,
    0.05,
    0.10,
    0.25,
]

VAL_FRAMES = {22, 23, 24}

MAX_DET = 300


def find_image(frame: int) -> Path:
    for suffix in ("png", "jpg", "jpeg"):
        p = IMAGE_DIR / f"frame_{frame:06d}.{suffix}"
        if p.exists():
            return p

    matches = sorted(
        IMAGE_DIR.glob(f"*{frame:06d}*")
    )

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


def make_starts(
    length: int,
    count: int,
    overlap: float,
):
    nominal = length / count

    tile_size = int(
        round(
            nominal * (1.0 + overlap)
        )
    )

    tile_size = min(
        length,
        tile_size,
    )

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


def make_tiles(image):
    xs = make_starts(
        VIEW_W,
        COLS,
        OVERLAP,
    )

    ys = make_starts(
        VIEW_H,
        ROWS,
        OVERLAP,
    )

    tiles = []

    for y1, y2 in ys:
        for x1, x2 in xs:
            tiles.append(
                {
                    "image":
                        image[y1:y2, x1:x2],
                    "x1":
                        x1,
                    "y1":
                        y1,
                }
            )

    return tiles


def iou(a, b):
    x1 = max(
        float(a[0]),
        float(b[0]),
    )

    y1 = max(
        float(a[1]),
        float(b[1]),
    )

    x2 = min(
        float(a[2]),
        float(b[2]),
    )

    y2 = min(
        float(a[3]),
        float(b[3]),
    )

    intersection = (
        max(0.0, x2 - x1)
        * max(0.0, y2 - y1)
    )

    area_a = (
        max(
            0.0,
            float(a[2] - a[0]),
        )
        * max(
            0.0,
            float(a[3] - a[1]),
        )
    )

    area_b = (
        max(
            0.0,
            float(b[2] - b[0]),
        )
        * max(
            0.0,
            float(b[3] - b[1]),
        )
    )

    union = (
        area_a
        + area_b
        - intersection
    )

    return (
        intersection / union
        if union > 0
        else 0.0
    )


def nms(
    boxes: list[np.ndarray],
    scores: list[float],
    threshold: float = 0.50,
):
    if not boxes:
        return [], []

    order = np.argsort(
        np.asarray(scores)
    )[::-1]

    keep_boxes = []
    keep_scores = []

    while len(order):
        i = int(order[0])

        keep_boxes.append(
            boxes[i]
        )

        keep_scores.append(
            scores[i]
        )

        remaining = []

        for j in order[1:]:
            j = int(j)

            if iou(
                boxes[i],
                boxes[j],
            ) < threshold:
                remaining.append(j)

        order = np.asarray(
            remaining,
            dtype=np.int64,
        )

    return (
        keep_boxes,
        keep_scores,
    )


def run_frame(
    model,
    image,
    conf,
    device,
):
    tiles = make_tiles(
        image
    )

    all_boxes = []
    all_scores = []

    if device == "mps":
        torch.mps.synchronize()

    start = time.perf_counter()

    for tile in tiles:
        result = model.predict(
            source=tile["image"],
            imgsz=960,
            conf=conf,
            iou=0.70,
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

        scores = (
            result.boxes.conf
            .detach()
            .cpu()
            .numpy()
        )

        for box, score in zip(
            boxes,
            scores,
        ):
            mapped = np.asarray(
                [
                    float(box[0])
                    + tile["x1"],
                    float(box[1])
                    + tile["y1"],
                    float(box[2])
                    + tile["x1"],
                    float(box[3])
                    + tile["y1"],
                ],
                dtype=np.float32,
            )

            all_boxes.append(
                mapped
            )

            all_scores.append(
                float(score)
            )

    boxes, scores = nms(
        all_boxes,
        all_scores,
        threshold=0.50,
    )

    if device == "mps":
        torch.mps.synchronize()

    latency_ms = (
        time.perf_counter()
        - start
    ) * 1000.0

    return boxes, scores, latency_ms


def evaluate(records):
    total = 0

    hit30 = 0
    hit50 = 0

    class_total = Counter()
    class_hit30 = Counter()
    class_hit50 = Counter()

    for record in records:
        boxes = record["boxes"]

        for gt in record["gt"]:
            total += 1

            cls = gt["class"]

            class_total[
                cls
            ] += 1

            best = (
                max(
                    iou(
                        gt["bbox"],
                        box,
                    )
                    for box in boxes
                )
                if boxes
                else 0.0
            )

            if best >= 0.30:
                hit30 += 1
                class_hit30[
                    cls
                ] += 1

            if best >= 0.50:
                hit50 += 1
                class_hit50[
                    cls
                ] += 1

    per_class = {}

    for cls in sorted(
        class_total
    ):
        per_class[
            cls
        ] = {
            "r30":
                class_hit30[cls]
                / class_total[cls],

            "r50":
                class_hit50[cls]
                / class_total[cls],
        }

    return {
        "instances":
            total,

        "r30":
            hit30 / total,

        "r50":
            hit50 / total,

        "per_class":
            per_class,
    }


def main():
    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print(
        f"Model:  {MODEL_PATH}"
    )

    print(
        f"Device: {device}"
    )

    model = YOLO(
        str(MODEL_PATH)
    )

    frames = []

    for path in sorted(
        ANNOTATION_DIR.glob(
            "frame_*.json"
        )
    ):
        data = json.loads(
            path.read_text()
        )

        frame = int(
            data["frame"]
        )

        source = cv2.imread(
            str(
                find_image(frame)
            )
        )

        view = cv2.resize(
            source,
            (
                VIEW_W,
                VIEW_H,
            ),
            interpolation=cv2.INTER_AREA,
        )

        gt = [
            {
                "class":
                    annotation[
                        "object_id"
                    ],

                "bbox":
                    source_bbox_to_l0(
                        annotation[
                            "bbox"
                        ]
                    ),
            }

            for annotation
            in data["annotations"]
        ]

        frames.append(
            {
                "frame":
                    frame,
                "image":
                    view,
                "gt":
                    gt,
            }
        )

    # Warm-up each tile path.
    warm_tiles = make_tiles(
        frames[0]["image"]
    )

    for tile in warm_tiles:
        model.predict(
            source=tile["image"],
            imgsz=960,
            conf=0.01,
            device=device,
            verbose=False,
        )

    results = {}

    for conf in CONF_THRESHOLDS:
        print()
        print("=" * 72)
        print(
            f"Confidence {conf}"
        )
        print("=" * 72)

        records = []

        latencies = []
        proposal_counts = []

        for frame in frames:
            boxes, scores, latency = run_frame(
                model,
                frame["image"],
                conf,
                device,
            )

            records.append(
                {
                    "frame":
                        frame["frame"],

                    "gt":
                        frame["gt"],

                    "boxes":
                        boxes,
                }
            )

            latencies.append(
                latency
            )

            proposal_counts.append(
                len(boxes)
            )

        all_eval = evaluate(
            records
        )

        holdout_records = [
            r
            for r in records
            if r["frame"]
            in VAL_FRAMES
        ]

        holdout_eval = evaluate(
            holdout_records
        )

        result = {
            "mean_proposals":
                float(
                    statistics.mean(
                        proposal_counts
                    )
                ),

            "median_proposals":
                float(
                    statistics.median(
                        proposal_counts
                    )
                ),

            "median_latency_ms":
                float(
                    statistics.median(
                        latencies
                    )
                ),

            "p95_latency_ms":
                float(
                    np.percentile(
                        latencies,
                        95,
                    )
                ),

            "all":
                all_eval,

            "holdout":
                holdout_eval,
        }

        results[
            str(conf)
        ] = result

        print(
            f"Proposals/frame: "
            f"{result['mean_proposals']:.1f}"
        )

        print(
            f"Latency median: "
            f"{result['median_latency_ms']:.1f} ms"
        )

        print(
            f"Latency p95: "
            f"{result['p95_latency_ms']:.1f} ms"
        )

        print(
            f"All @0.50: "
            f"{all_eval['r50']:.4f}"
        )

        print(
            f"Holdout @0.30: "
            f"{holdout_eval['r30']:.4f}"
        )

        print(
            f"Holdout @0.50: "
            f"{holdout_eval['r50']:.4f}"
        )

    best_conf = max(
        CONF_THRESHOLDS,
        key=lambda c:
            results[
                str(c)
            ][
                "holdout"
            ][
                "r50"
            ],
    )

    best = results[
        str(best_conf)
    ]

    print()
    print("=" * 72)
    print(
        f"BEST CONFIDENCE: {best_conf}"
    )
    print("=" * 72)

    print(
        f"Holdout @0.50: "
        f"{best['holdout']['r50']:.4f}"
    )

    print(
        f"Proposals/frame: "
        f"{best['mean_proposals']:.1f}"
    )

    print(
        f"Median latency: "
        f"{best['median_latency_ms']:.1f} ms"
    )

    print()
    print("PER-CLASS HOLDOUT RECALL")

    print(
        f"{'class':20s} "
        f"{'R@0.30':>8s} "
        f"{'R@0.50':>8s}"
    )

    print("-" * 40)

    for cls, stats in (
        best[
            "holdout"
        ][
            "per_class"
        ].items()
    ):
        print(
            f"{cls:20s} "
            f"{stats['r30']:8.3f} "
            f"{stats['r50']:8.3f}"
        )

    RESULT_PATH.write_text(
        json.dumps(
            results,
            indent=2,
        )
    )

    print()
    print(
        f"Saved: {RESULT_PATH}"
    )


if __name__ == "__main__":
    main()
