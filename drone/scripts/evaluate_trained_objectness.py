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
    / "exp_d008"
)

RESULT_PATH = (
    ARTIFACT_DIR
    / "confidence_sweep.json"
)

SOURCE_WIDTH = 3840
SOURCE_HEIGHT = 2160

VIEW_WIDTH = 960
VIEW_HEIGHT = 540

SCALE_X = VIEW_WIDTH / SOURCE_WIDTH
SCALE_Y = VIEW_HEIGHT / SOURCE_HEIGHT

CONF_THRESHOLDS = [
    0.001,
    0.005,
    0.01,
    0.025,
    0.05,
    0.10,
    0.25,
]

IOU_THRESHOLDS = [
    0.30,
    0.50,
]

VAL_FRAMES = {
    22,
    23,
    24,
}


def find_image(frame: int) -> Path:
    candidates = [
        IMAGE_DIR / f"frame_{frame:06d}.png",
        IMAGE_DIR / f"frame_{frame:06d}.jpg",
        IMAGE_DIR / f"frame_{frame:06d}.jpeg",
    ]

    for path in candidates:
        if path.exists():
            return path

    matches = sorted(
        IMAGE_DIR.glob(f"*{frame:06d}*")
    )

    if matches:
        return matches[0]

    raise FileNotFoundError(
        f"No image for frame {frame}"
    )


def source_bbox_to_l0(
    bbox: list[int | float],
) -> np.ndarray:

    x1, y1, x2, y2 = map(
        float,
        bbox,
    )

    return np.asarray(
        [
            x1 * SCALE_X,
            y1 * SCALE_Y,
            x2 * SCALE_X,
            y2 * SCALE_Y,
        ],
        dtype=np.float32,
    )


def iou_xyxy(
    a: np.ndarray,
    b: np.ndarray,
) -> float:

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

    iw = max(
        0.0,
        x2 - x1,
    )

    ih = max(
        0.0,
        y2 - y1,
    )

    intersection = iw * ih

    area_a = max(
        0.0,
        float(a[2] - a[0]),
    ) * max(
        0.0,
        float(a[3] - a[1]),
    )

    area_b = max(
        0.0,
        float(b[2] - b[0]),
    ) * max(
        0.0,
        float(b[3] - b[1]),
    )

    union = (
        area_a
        + area_b
        - intersection
    )

    if union <= 0:
        return 0.0

    return intersection / union


def evaluate_subset(
    records: list[dict],
) -> dict:

    total = 0

    hits = {
        threshold: 0
        for threshold
        in IOU_THRESHOLDS
    }

    class_total = Counter()

    class_hits = {
        threshold: Counter()
        for threshold
        in IOU_THRESHOLDS
    }

    for record in records:

        proposals = record[
            "proposals"
        ]

        for gt in record["gt"]:

            total += 1

            class_name = gt["class"]

            class_total[
                class_name
            ] += 1

            gt_box = gt["bbox"]

            if proposals:

                best_iou = max(
                    iou_xyxy(
                        gt_box,
                        proposal,
                    )
                    for proposal
                    in proposals
                )

            else:
                best_iou = 0.0

            for threshold in (
                IOU_THRESHOLDS
            ):

                if best_iou >= threshold:

                    hits[
                        threshold
                    ] += 1

                    class_hits[
                        threshold
                    ][
                        class_name
                    ] += 1

    result = {
        "instances":
            total,
        "recall":
            {},
        "per_class":
            {},
    }

    for threshold in IOU_THRESHOLDS:

        key = str(threshold)

        result[
            "recall"
        ][key] = (
            hits[threshold]
            / total
            if total
            else 0.0
        )

        per_class = {}

        for class_name in sorted(
            class_total
        ):

            per_class[
                class_name
            ] = (
                class_hits[
                    threshold
                ][class_name]
                / class_total[
                    class_name
                ]
            )

        result[
            "per_class"
        ][key] = per_class

    return result


def main() -> None:

    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            MODEL_PATH
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

    annotation_paths = sorted(
        ANNOTATION_DIR.glob(
            "frame_*.json"
        )
    )

    for annotation_path in (
        annotation_paths
    ):

        with annotation_path.open(
            "r",
            encoding="utf-8",
        ) as f:
            data = json.load(f)

        frame = int(
            data["frame"]
        )

        image = cv2.imread(
            str(
                find_image(frame)
            ),
            cv2.IMREAD_COLOR,
        )

        if image is None:
            raise RuntimeError(
                f"Could not read frame {frame}"
            )

        view = cv2.resize(
            image,
            (
                VIEW_WIDTH,
                VIEW_HEIGHT,
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

    # Warm-up.
    print("Warming up...")

    for _ in range(3):
        model.predict(
            source=frames[0][
                "image"
            ],
            imgsz=960,
            conf=0.001,
            iou=0.70,
            max_det=300,
            device=device,
            verbose=False,
        )

    print("Warm-up complete.")
    print()

    all_results = {}

    for conf in CONF_THRESHOLDS:

        print("=" * 76)

        print(
            f"Confidence: {conf}"
        )

        print("=" * 76)

        records = []

        latencies = []
        proposal_counts = []

        for frame_data in frames:

            if device == "mps":
                torch.mps.synchronize()

            start = time.perf_counter()

            prediction = model.predict(
                source=frame_data[
                    "image"
                ],
                imgsz=960,
                conf=conf,
                iou=0.70,
                max_det=300,
                device=device,
                verbose=False,
            )[0]

            if device == "mps":
                torch.mps.synchronize()

            latency = (
                time.perf_counter()
                - start
            ) * 1000.0

            latencies.append(
                latency
            )

            if prediction.boxes is None:

                proposals = []

            else:

                proposals = [
                    box.astype(
                        np.float32
                    )
                    for box in (
                        prediction
                        .boxes
                        .xyxy
                        .detach()
                        .cpu()
                        .numpy()
                    )
                ]

            proposal_counts.append(
                len(proposals)
            )

            records.append(
                {
                    "frame":
                        frame_data[
                            "frame"
                        ],
                    "gt":
                        frame_data[
                            "gt"
                        ],
                    "proposals":
                        proposals,
                }
            )

        all_eval = evaluate_subset(
            records
        )

        val_records = [
            record
            for record in records
            if record["frame"]
            in VAL_FRAMES
        ]

        val_eval = evaluate_subset(
            val_records
        )

        result = {
            "confidence":
                conf,

            "mean_proposals_per_frame":
                float(
                    statistics.mean(
                        proposal_counts
                    )
                ),

            "median_proposals_per_frame":
                float(
                    statistics.median(
                        proposal_counts
                    )
                ),

            "latency_ms": {
                "median":
                    float(
                        statistics.median(
                            latencies
                        )
                    ),
                "p95":
                    float(
                        np.percentile(
                            latencies,
                            95,
                        )
                    ),
            },

            "all_frames":
                all_eval,

            "temporal_holdout":
                val_eval,
        }

        all_results[
            str(conf)
        ] = result

        print(
            "Proposals/frame: "
            f"mean="
            f"{result['mean_proposals_per_frame']:.1f}, "
            f"median="
            f"{result['median_proposals_per_frame']:.1f}"
        )

        print(
            "Latency: "
            f"median="
            f"{result['latency_ms']['median']:.1f} ms, "
            f"p95="
            f"{result['latency_ms']['p95']:.1f} ms"
        )

        print(
            "All-frame recall:"
        )

        print(
            "  IoU 0.30: "
            f"{all_eval['recall']['0.3']:.4f}"
        )

        print(
            "  IoU 0.50: "
            f"{all_eval['recall']['0.5']:.4f}"
        )

        print(
            "Holdout recall:"
        )

        print(
            "  IoU 0.30: "
            f"{val_eval['recall']['0.3']:.4f}"
        )

        print(
            "  IoU 0.50: "
            f"{val_eval['recall']['0.5']:.4f}"
        )

        print()

    output = {
        "model":
            str(MODEL_PATH),
        "device":
            device,
        "confidence_thresholds":
            CONF_THRESHOLDS,
        "results":
            all_results,
    }

    # Make NumPy proposal arrays JSON safe.
    def make_json_safe(obj):
        if isinstance(
            obj,
            np.generic,
        ):
            return obj.item()

        if isinstance(
            obj,
            np.ndarray,
        ):
            return obj.tolist()

        raise TypeError(
            type(obj).__name__
        )

    with RESULT_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
            default=make_json_safe,
        )

    # Pick operating point based primarily on
    # temporal-holdout IoU@0.50 recall.
    best_conf = max(
        CONF_THRESHOLDS,
        key=lambda conf: (
            all_results[
                str(conf)
            ][
                "temporal_holdout"
            ][
                "recall"
            ][
                "0.5"
            ]
        ),
    )

    best = all_results[
        str(best_conf)
    ]

    print()
    print("=" * 76)

    print(
        "BEST HOLDOUT OPERATING POINT"
    )

    print("=" * 76)

    print(
        f"Confidence: {best_conf}"
    )

    print(
        "Holdout recall @0.30: "
        f"{best['temporal_holdout']['recall']['0.3']:.4f}"
    )

    print(
        "Holdout recall @0.50: "
        f"{best['temporal_holdout']['recall']['0.5']:.4f}"
    )

    print(
        "Proposals/frame: "
        f"{best['mean_proposals_per_frame']:.1f}"
    )

    print()
    print(
        "PER-CLASS HOLDOUT RECALL"
    )

    print(
        f"{'class':20s} "
        f"{'IoU@0.30':>10s} "
        f"{'IoU@0.50':>10s}"
    )

    print("-" * 44)

    classes = sorted(
        best[
            "temporal_holdout"
        ][
            "per_class"
        ][
            "0.3"
        ]
    )

    for class_name in classes:

        r03 = (
            best[
                "temporal_holdout"
            ][
                "per_class"
            ][
                "0.3"
            ][
                class_name
            ]
        )

        r05 = (
            best[
                "temporal_holdout"
            ][
                "per_class"
            ][
                "0.5"
            ][
                class_name
            ]
        )

        print(
            f"{class_name:20s} "
            f"{r03:10.3f} "
            f"{r05:10.3f}"
        )

    print()

    print(
        f"Saved: {RESULT_PATH}"
    )


if __name__ == "__main__":
    main()
