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
    / "exp_d017"
)

RESULT_PATH = (
    ARTIFACT_DIR
    / "tracking_upper_bound.json"
)

VIEW_W = 960
VIEW_H = 540

YOLO_CONF = 0.01
YOLO_NMS = 0.50

COLS = 2
ROWS = 2
OVERLAP = 0.20

# Evaluate the whole sequence so tracking has enough temporal context.
FRAMES = list(range(25))

IOU_MATCH = 0.50


def find_image(frame: int) -> Path:
    for suffix in ("png", "jpg", "jpeg"):
        path = (
            IMAGE_DIR
            / f"frame_{frame:06d}.{suffix}"
        )

        if path.exists():
            return path

    matches = sorted(
        IMAGE_DIR.glob(
            f"*{frame:06d}*"
        )
    )

    if matches:
        return matches[0]

    raise FileNotFoundError(frame)


def load_frame(frame: int) -> np.ndarray:
    source = cv2.imread(
        str(
            find_image(frame)
        ),
        cv2.IMREAD_COLOR,
    )

    if source is None:
        raise RuntimeError(
            f"Could not read frame {frame}"
        )

    return cv2.resize(
        source,
        (VIEW_W, VIEW_H),
        interpolation=cv2.INTER_AREA,
    )


def load_gt(frame: int):
    path = (
        ANNOTATION_DIR
        / f"frame_{frame:06d}.json"
    )

    data = json.loads(
        path.read_text()
    )

    result = []

    for annotation in data["annotations"]:
        x1, y1, x2, y2 = map(
            float,
            annotation["bbox"],
        )

        result.append(
            {
                "class":
                    annotation["object_id"],

                "bbox":
                    np.asarray(
                        [
                            x1 / 4.0,
                            y1 / 4.0,
                            x2 / 4.0,
                            y2 / 4.0,
                        ],
                        dtype=np.float32,
                    ),
            }
        )

    return result


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


def make_starts(
    length,
    count,
    overlap,
):
    tile_size = int(
        round(
            (length / count)
            * (1.0 + overlap)
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
        OVERLAP,
    )

    ys = make_starts(
        VIEW_H,
        ROWS,
        OVERLAP,
    )

    result = []

    for y1, y2 in ys:
        for x1, x2 in xs:
            result.append(
                {
                    "image":
                        image[y1:y2, x1:x2],

                    "x1":
                        x1,

                    "y1":
                        y1,
                }
            )

    return result


def nms(
    boxes,
    scores,
    threshold=0.50,
):
    if not boxes:
        return [], []

    order = np.argsort(
        np.asarray(scores)
    )[::-1]

    keep_boxes = []
    keep_scores = []

    while len(order):
        i = int(
            order[0]
        )

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
                remaining.append(
                    j
                )

        order = np.asarray(
            remaining,
            dtype=np.int64,
        )

    return (
        keep_boxes,
        keep_scores,
    )


def run_detector(
    model,
    image,
    device,
):
    boxes = []
    scores = []

    if device == "mps":
        torch.mps.synchronize()

    start = time.perf_counter()

    for tile in make_tiles(
        image
    ):
        result = model.predict(
            source=tile["image"],
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
            )

            scores.append(
                float(score)
            )

    boxes, scores = nms(
        boxes,
        scores,
        YOLO_NMS,
    )

    if device == "mps":
        torch.mps.synchronize()

    latency_ms = (
        time.perf_counter()
        - start
    ) * 1000.0

    return (
        boxes,
        scores,
        latency_ms,
    )


def box_center(box):
    return np.asarray(
        [
            (
                float(box[0])
                + float(box[2])
            )
            / 2.0,

            (
                float(box[1])
                + float(box[3])
            )
            / 2.0,
        ],
        dtype=np.float32,
    )


def propagate_box(
    previous_box,
    velocity,
):
    dx, dy = velocity

    result = previous_box.copy()

    result[
        [0, 2]
    ] += dx

    result[
        [1, 3]
    ] += dy

    return result


def best_detector_match(
    gt_box,
    detector_boxes,
):
    if not detector_boxes:
        return (
            0.0,
            None,
        )

    scores = [
        iou(
            gt_box,
            box,
        )
        for box in detector_boxes
    ]

    best_index = int(
        np.argmax(
            scores
        )
    )

    return (
        float(
            scores[
                best_index
            ]
        ),
        detector_boxes[
            best_index
        ],
    )


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
        f"Device: {device}"
    )

    model = YOLO(
        str(
            MODEL_PATH
        )
    )

    # Warm-up.
    warm_image = load_frame(0)

    run_detector(
        model,
        warm_image,
        device,
    )

    detector_by_frame = {}

    latencies = []

    print(
        "Running detector..."
    )

    for frame in FRAMES:
        image = load_frame(
            frame
        )

        boxes, scores, latency = (
            run_detector(
                model,
                image,
                device,
            )
        )

        detector_by_frame[
            frame
        ] = boxes

        latencies.append(
            latency
        )

        print(
            f"  frame {frame:02d}: "
            f"{len(boxes):3d} proposals, "
            f"{latency:6.1f} ms"
        )

    # -------------------------------------------------------
    # Oracle identity association:
    # for each GT object, if detector hits it, we know
    # which class/identity that proposal belongs to.
    #
    # This is ONLY an upper-bound tracking diagnostic.
    # -------------------------------------------------------

    tracks = {}

    detector_hits = 0
    tracked_hits = 0
    combined_hits = 0
    total_gt = 0

    per_class = {
        "total": Counter(),
        "detector": Counter(),
        "combined": Counter(),
    }

    frame_results = []

    for frame in FRAMES:
        gt_objects = load_gt(
            frame
        )

        detector_boxes = (
            detector_by_frame[
                frame
            ]
        )

        frame_record = {
            "frame":
                frame,
            "objects":
                [],
        }

        for gt in gt_objects:
            class_name = gt[
                "class"
            ]

            gt_box = gt[
                "bbox"
            ]

            total_gt += 1

            per_class[
                "total"
            ][
                class_name
            ] += 1

            detector_iou, detector_box = (
                best_detector_match(
                    gt_box,
                    detector_boxes,
                )
            )

            detector_hit = (
                detector_iou
                >= IOU_MATCH
            )

            if detector_hit:
                detector_hits += 1

                per_class[
                    "detector"
                ][
                    class_name
                ] += 1

            predicted_box = None
            predicted_iou = 0.0

            if class_name in tracks:
                track = tracks[
                    class_name
                ]

                predicted_box = (
                    propagate_box(
                        track[
                            "last_box"
                        ],
                        track[
                            "velocity"
                        ],
                    )
                )

                predicted_iou = iou(
                    gt_box,
                    predicted_box,
                )

                if predicted_iou >= IOU_MATCH:
                    tracked_hits += 1

            combined_hit = (
                detector_hit
                or predicted_iou
                >= IOU_MATCH
            )

            if combined_hit:
                combined_hits += 1

                per_class[
                    "combined"
                ][
                    class_name
                ] += 1

            # Oracle update:
            # if detector hit, use matched detector box.
            # Otherwise if propagation hit, continue using it.
            update_box = None

            if detector_hit:
                update_box = (
                    detector_box
                )

            elif (
                predicted_box
                is not None
                and predicted_iou
                >= IOU_MATCH
            ):
                update_box = (
                    predicted_box
                )

            if update_box is not None:
                center = box_center(
                    update_box
                )

                if class_name in tracks:
                    previous_center = (
                        tracks[
                            class_name
                        ][
                            "center"
                        ]
                    )

                    velocity = (
                        center
                        - previous_center
                    )

                else:
                    # Known global first-order motion:
                    # approx +16.25 px vertically at L0.
                    velocity = np.asarray(
                        [
                            0.25,
                            16.25,
                        ],
                        dtype=np.float32,
                    )

                tracks[
                    class_name
                ] = {
                    "last_box":
                        update_box.copy(),

                    "center":
                        center,

                    "velocity":
                        velocity,
                }

            frame_record[
                "objects"
            ].append(
                {
                    "class":
                        class_name,

                    "detector_iou":
                        float(
                            detector_iou
                        ),

                    "tracked_iou":
                        float(
                            predicted_iou
                        ),

                    "detector_hit":
                        bool(
                            detector_hit
                        ),

                    "combined_hit":
                        bool(
                            combined_hit
                        ),
                }
            )

        frame_results.append(
            frame_record
        )

    detector_recall = (
        detector_hits
        / total_gt
    )

    combined_recall = (
        combined_hits
        / total_gt
    )

    print()
    print("=" * 72)
    print(
        "EXP-D017 TRACKING UPPER BOUND"
    )
    print("=" * 72)

    print(
        f"Total GT instances: "
        f"{total_gt}"
    )

    print(
        f"Detector-only hits: "
        f"{detector_hits}"
    )

    print(
        f"Detector recall @0.50: "
        f"{detector_recall:.4f}"
    )

    print(
        f"Propagation hits: "
        f"{tracked_hits}"
    )

    print(
        f"Combined hits: "
        f"{combined_hits}"
    )

    print(
        f"Combined recall @0.50: "
        f"{combined_recall:.4f}"
    )

    print(
        f"Absolute gain: "
        f"{combined_recall - detector_recall:+.4f}"
    )

    print()
    print(
        "Per-class recall "
        "(detector -> detector+tracking)"
    )

    print(
        f"{'class':20s} "
        f"{'detector':>10s} "
        f"{'combined':>10s}"
    )

    print("-" * 44)

    class_names = sorted(
        per_class[
            "total"
        ]
    )

    per_class_output = {}

    for class_name in class_names:
        total = per_class[
            "total"
        ][
            class_name
        ]

        detector = (
            per_class[
                "detector"
            ][
                class_name
            ]
            / total
        )

        combined = (
            per_class[
                "combined"
            ][
                class_name
            ]
            / total
        )

        per_class_output[
            class_name
        ] = {
            "total":
                total,
            "detector_recall":
                detector,
            "combined_recall":
                combined,
        }

        print(
            f"{class_name:20s} "
            f"{detector:10.3f} "
            f"{combined:10.3f}"
        )

    output = {
        "detector_conf":
            YOLO_CONF,

        "total_gt":
            total_gt,

        "detector_recall":
            detector_recall,

        "combined_recall":
            combined_recall,

        "absolute_gain":
            combined_recall
            - detector_recall,

        "median_detector_latency_ms":
            float(
                statistics.median(
                    latencies
                )
            ),

        "per_class":
            per_class_output,

        "frames":
            frame_results,
    }

    RESULT_PATH.write_text(
        json.dumps(
            output,
            indent=2,
        )
    )

    print()
    print(
        f"Median detector latency: "
        f"{statistics.median(latencies):.1f} ms"
    )

    print(
        f"Saved: {RESULT_PATH}"
    )


if __name__ == "__main__":
    main()
