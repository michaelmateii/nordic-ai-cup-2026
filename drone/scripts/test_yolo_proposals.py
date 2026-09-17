#!/usr/bin/env python3

from __future__ import annotations

import json
import statistics
import time
from collections import Counter, defaultdict
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

ARTIFACT_DIR = (
    ROOT
    / "drone"
    / "artifacts"
    / "exp_d004"
)

RESULT_PATH = (
    ARTIFACT_DIR
    / "yolo11n_proposal_recall.json"
)

MODEL_NAME = "yolo11n.pt"

VIEW_WIDTH = 960
VIEW_HEIGHT = 540

SOURCE_WIDTH = 3840
SOURCE_HEIGHT = 2160

SCALE_X = VIEW_WIDTH / SOURCE_WIDTH
SCALE_Y = VIEW_HEIGHT / SOURCE_HEIGHT

# Test deliberately low confidence thresholds because
# we care about proposal recall, not detector precision.
CONF_THRESHOLDS = [
    0.001,
    0.01,
    0.05,
]

IOU_THRESHOLDS = [
    0.30,
    0.50,
]


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


def iou_xyxy(
    a: np.ndarray,
    b: np.ndarray,
) -> float:

    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    intersection_w = max(
        0.0,
        x2 - x1,
    )

    intersection_h = max(
        0.0,
        y2 - y1,
    )

    intersection = (
        intersection_w
        * intersection_h
    )

    area_a = max(
        0.0,
        a[2] - a[0],
    ) * max(
        0.0,
        a[3] - a[1],
    )

    area_b = max(
        0.0,
        b[2] - b[0],
    ) * max(
        0.0,
        b[3] - b[1],
    )

    union = (
        area_a
        + area_b
        - intersection
    )

    if union <= 0:
        return 0.0

    return intersection / union


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


def get_device() -> str:

    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def main() -> None:

    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = get_device()

    print(f"Model:  {MODEL_NAME}")
    print(f"Device: {device}")
    print(
        f"View:   "
        f"{VIEW_WIDTH}x{VIEW_HEIGHT} "
        f"(camera level 0)"
    )

    print()
    print("Loading YOLO...")

    model = YOLO(MODEL_NAME)

    annotation_paths = sorted(
        ANNOTATION_DIR.glob(
            "frame_*.json"
        )
    )

    if not annotation_paths:
        raise RuntimeError(
            "No annotation frames found."
        )

    # ---------------------------------------------------------
    # Load all frames/GT once.
    # ---------------------------------------------------------

    frames = []

    for annotation_path in annotation_paths:

        with annotation_path.open(
            "r",
            encoding="utf-8",
        ) as f:
            annotation_data = json.load(f)

        frame_number = int(
            annotation_data["frame"]
        )

        image_path = find_image(
            frame_number
        )

        source_image = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR,
        )

        if source_image is None:
            raise RuntimeError(
                f"Could not read {image_path}"
            )

        view = cv2.resize(
            source_image,
            (VIEW_WIDTH, VIEW_HEIGHT),
            interpolation=cv2.INTER_AREA,
        )

        gt = []

        for annotation in (
            annotation_data["annotations"]
        ):

            gt.append(
                {
                    "class":
                        annotation["object_id"],
                    "bbox":
                        source_bbox_to_l0(
                            annotation["bbox"]
                        ),
                }
            )

        frames.append(
            {
                "frame": frame_number,
                "view": view,
                "gt": gt,
            }
        )

    # ---------------------------------------------------------
    # Warm-up explicitly so startup doesn't contaminate latency.
    # ---------------------------------------------------------

    print("Warming up model...")

    for _ in range(3):
        model.predict(
            source=frames[0]["view"],
            imgsz=960,
            conf=0.001,
            iou=0.70,
            max_det=300,
            device=device,
            verbose=False,
        )

    print("Warm-up complete.")
    print()

    experiment_results = {}

    for conf_threshold in CONF_THRESHOLDS:

        print(
            "=" * 76
        )

        print(
            f"Testing confidence "
            f"{conf_threshold}"
        )

        print(
            "=" * 76
        )

        per_iou_hits = {
            threshold: 0
            for threshold
            in IOU_THRESHOLDS
        }

        per_class_hits = {
            threshold: Counter()
            for threshold
            in IOU_THRESHOLDS
        }

        per_class_total = Counter()

        latencies_ms = []

        proposal_counts = []

        total_gt = 0

        frame_results = []

        for frame_data in frames:

            if device == "mps":
                torch.mps.synchronize()

            start = time.perf_counter()

            result = model.predict(
                source=frame_data["view"],
                imgsz=960,
                conf=conf_threshold,
                iou=0.70,
                max_det=300,
                device=device,
                verbose=False,
            )[0]

            if device == "mps":
                torch.mps.synchronize()

            latency_ms = (
                time.perf_counter()
                - start
            ) * 1000.0

            latencies_ms.append(
                latency_ms
            )

            if result.boxes is None:
                predicted_boxes = np.empty(
                    (0, 4),
                    dtype=np.float32,
                )
            else:
                predicted_boxes = (
                    result.boxes.xyxy
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )

            proposal_counts.append(
                len(predicted_boxes)
            )

            frame_gt_results = []

            for gt in frame_data["gt"]:

                gt_box = gt["bbox"]
                class_name = gt["class"]

                total_gt += 1
                per_class_total[
                    class_name
                ] += 1

                if len(predicted_boxes):

                    best_iou = max(
                        iou_xyxy(
                            gt_box,
                            proposal,
                        )
                        for proposal
                        in predicted_boxes
                    )

                else:
                    best_iou = 0.0

                hits = {}

                for iou_threshold in (
                    IOU_THRESHOLDS
                ):

                    hit = (
                        best_iou
                        >= iou_threshold
                    )

                    hits[
                        str(iou_threshold)
                    ] = hit

                    if hit:
                        per_iou_hits[
                            iou_threshold
                        ] += 1

                        per_class_hits[
                            iou_threshold
                        ][class_name] += 1

                frame_gt_results.append(
                    {
                        "class":
                            class_name,
                        "best_iou":
                            float(best_iou),
                        "hits":
                            hits,
                    }
                )

            frame_results.append(
                {
                    "frame":
                        frame_data["frame"],
                    "latency_ms":
                        latency_ms,
                    "num_proposals":
                        len(predicted_boxes),
                    "gt":
                        frame_gt_results,
                }
            )

        threshold_result = {
            "confidence_threshold":
                conf_threshold,
            "total_gt":
                total_gt,
            "mean_proposals_per_frame":
                statistics.mean(
                    proposal_counts
                ),
            "median_proposals_per_frame":
                statistics.median(
                    proposal_counts
                ),
            "latency_ms": {
                "mean":
                    statistics.mean(
                        latencies_ms
                    ),
                "median":
                    statistics.median(
                        latencies_ms
                    ),
                "p95":
                    float(
                        np.percentile(
                            latencies_ms,
                            95,
                        )
                    ),
                "max":
                    max(
                        latencies_ms
                    ),
            },
            "recall": {},
            "per_class_recall": {},
            "frames":
                frame_results,
        }

        for iou_threshold in (
            IOU_THRESHOLDS
        ):

            recall = (
                per_iou_hits[
                    iou_threshold
                ]
                / total_gt
            )

            threshold_result[
                "recall"
            ][
                str(iou_threshold)
            ] = recall

            per_class = {}

            for class_name in sorted(
                per_class_total
            ):

                total = (
                    per_class_total[
                        class_name
                    ]
                )

                hit_count = (
                    per_class_hits[
                        iou_threshold
                    ][class_name]
                )

                per_class[
                    class_name
                ] = (
                    hit_count
                    / total
                )

            threshold_result[
                "per_class_recall"
            ][
                str(iou_threshold)
            ] = per_class

        experiment_results[
            str(conf_threshold)
        ] = threshold_result

        print(
            f"Proposals/frame: "
            f"mean="
            f"{threshold_result['mean_proposals_per_frame']:.1f}, "
            f"median="
            f"{threshold_result['median_proposals_per_frame']:.1f}"
        )

        print(
            "Latency: "
            f"median="
            f"{threshold_result['latency_ms']['median']:.1f} ms, "
            f"p95="
            f"{threshold_result['latency_ms']['p95']:.1f} ms"
        )

        for iou_threshold in (
            IOU_THRESHOLDS
        ):

            print(
                f"Proposal recall @ "
                f"IoU {iou_threshold:.2f}: "
                f"{threshold_result['recall'][str(iou_threshold)]:.4f}"
            )

        print()

    result_document = {
        "model":
            MODEL_NAME,
        "device":
            device,
        "view":
            "level_0_960x540",
        "confidence_thresholds":
            CONF_THRESHOLDS,
        "iou_thresholds":
            IOU_THRESHOLDS,
        "results":
            experiment_results,
    }

    with RESULT_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            result_document,
            f,
            indent=2,
            default=lambda obj: (
                obj.item()
                if isinstance(obj, np.generic)
                else obj.tolist()
                if isinstance(obj, np.ndarray)
                else str(obj)
            ),
        )

    # ---------------------------------------------------------
    # Detailed per-class table for most permissive threshold.
    # ---------------------------------------------------------

    best_key = str(
        min(CONF_THRESHOLDS)
    )

    best = experiment_results[
        best_key
    ]

    print()
    print("=" * 76)

    print(
        f"PER-CLASS PROPOSAL RECALL "
        f"(conf={best_key})"
    )

    print("=" * 76)

    print(
        f"{'class':20s} "
        f"{'IoU@0.30':>10s} "
        f"{'IoU@0.50':>10s}"
    )

    print("-" * 44)

    class_names = sorted(
        best[
            "per_class_recall"
        ]["0.3"]
    )

    for class_name in class_names:

        r03 = (
            best[
                "per_class_recall"
            ]["0.3"][class_name]
        )

        r05 = (
            best[
                "per_class_recall"
            ]["0.5"][class_name]
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
