#!/usr/bin/env python3

from __future__ import annotations

import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    ROOT
    / "drone"
    / "artifacts"
    / "exp_d029b"
    / "runs"
    / "yolo11n_validation_domain"
    / "weights"
    / "last.pt"
)

HOLDOUT_PATH = (
    ROOT
    / "drone"
    / "artifacts"
    / "exp_d029a"
    / "manual_holdout.json"
)

CAPTURE = (
    ROOT
    / "drone"
    / "captures"
    / "3224a582bfbf4273a028497662b7aa7c"
    / "frames"
)

OUT = (
    ROOT
    / "drone"
    / "artifacts"
    / "exp_d029c"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

RESULTS_PATH = (
    OUT
    / "holdout_results.json"
)

CONFIDENCES = [
    0.001,
    0.005,
    0.01,
    0.025,
    0.05,
    0.10,
    0.20,
]

IOU_THRESHOLD = 0.50


CLASSES = [
    "condor",
    "hangar",
    "helicopter",
    "jammer",
    "jet_plane",
    "large_launcher",
    "large_tower",
    "medium_launcher",
    "medium_plane",
    "mine_roller",
    "small_launcher",
    "small_plane",
    "small_tower",
    "spacecraft",
    "ta-ta",
    "tank",
]


def iou(a, b):
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))

    inter = (
        max(0.0, x2 - x1)
        * max(0.0, y2 - y1)
    )

    aa = (
        max(0.0, float(a[2] - a[0]))
        * max(0.0, float(a[3] - a[1]))
    )

    ab = (
        max(0.0, float(b[2] - b[0]))
        * max(0.0, float(b[3] - b[1]))
    )

    union = aa + ab - inter

    return inter / union if union > 0 else 0.0


def frame_path(frame_index):
    matches = sorted(
        CAPTURE.glob(
            f"*index_{frame_index:06d}.png"
        )
    )

    if not matches:
        raise FileNotFoundError(
            f"No image for frame_index={frame_index}"
        )

    return matches[0]


def evaluate_conf(
    model,
    holdout,
    device,
    conf,
):
    grouped = defaultdict(list)

    for row in holdout:
        grouped[
            row["frame_index"]
        ].append(row)

    hits = []
    latencies = []
    proposal_counts = []

    for frame_index, gt_rows in sorted(
        grouped.items()
    ):
        path = frame_path(
            frame_index
        )

        image = cv2.imread(
            str(path)
        )

        if image is None:
            raise RuntimeError(
                f"Could not read {path}"
            )

        if device == "mps":
            torch.mps.synchronize()

        start = time.perf_counter()

        result = model.predict(
            image,
            imgsz=960,
            conf=conf,
            iou=0.7,
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

        predictions = []

        if result.boxes is not None:
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

            classes = (
                result.boxes.cls
                .detach()
                .cpu()
                .numpy()
                .astype(int)
            )

            for box, score, cls_idx in zip(
                boxes,
                scores,
                classes,
            ):
                predictions.append(
                    {
                        "bbox": [
                            float(v)
                            for v in box
                        ],
                        "score":
                            float(score),
                        "class":
                            CLASSES[
                                int(cls_idx)
                            ],
                    }
                )

        proposal_counts.append(
            len(predictions)
        )

        for gt in gt_rows:
            same_class = [
                p
                for p in predictions
                if (
                    p["class"]
                    == gt["class"]
                )
            ]

            best_iou = 0.0
            best_score = 0.0

            for pred in same_class:
                overlap = iou(
                    pred["bbox"],
                    gt["bbox"],
                )

                if overlap > best_iou:
                    best_iou = overlap
                    best_score = pred["score"]

            hits.append(
                {
                    "class":
                        gt["class"],
                    "frame_index":
                        gt["frame_index"],
                    "bbox":
                        gt["bbox"],
                    "best_iou":
                        float(best_iou),
                    "best_score":
                        float(best_score),
                    "hit":
                        bool(
                            best_iou
                            >= IOU_THRESHOLD
                        ),
                }
            )

    per_class = defaultdict(list)

    for row in hits:
        per_class[
            row["class"]
        ].append(row)

    class_results = {}

    for cls, rows in sorted(
        per_class.items()
    ):
        class_results[
            cls
        ] = {
            "n":
                len(rows),
            "hits":
                sum(
                    r["hit"]
                    for r in rows
                ),
            "recall_iou50":
                sum(
                    r["hit"]
                    for r in rows
                )
                / len(rows),
            "mean_best_iou":
                float(
                    np.mean(
                        [
                            r["best_iou"]
                            for r in rows
                        ]
                    )
                ),
        }

    total_hits = sum(
        r["hit"]
        for r in hits
    )

    return {
        "confidence":
            conf,
        "objects":
            len(hits),
        "hits":
            total_hits,
        "recall_iou50":
            (
                total_hits
                / len(hits)
                if hits
                else 0.0
            ),
        "mean_proposals_per_frame":
            float(
                np.mean(
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
        "per_class":
            class_results,
        "objects_detail":
            hits,
    }


def main():
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model does not exist yet: {MODEL_PATH}"
        )

    holdout = json.loads(
        HOLDOUT_PATH.read_text()
    )

    model = YOLO(
        str(MODEL_PATH)
    )

    warm = cv2.imread(
        str(
            frame_path(
                holdout[0][
                    "frame_index"
                ]
            )
        )
    )

    model.predict(
        warm,
        imgsz=960,
        conf=0.01,
        device=device,
        verbose=False,
    )

    results = []

    print(
        "Device:",
        device,
    )

    print(
        "Model:",
        MODEL_PATH,
    )

    print(
        "Holdout objects:",
        len(holdout),
    )

    for conf in CONFIDENCES:
        result = evaluate_conf(
            model,
            holdout,
            device,
            conf,
        )

        results.append(
            result
        )

        print()
        print("=" * 72)
        print(
            f"Confidence {conf}"
        )
        print("=" * 72)

        print(
            f"Recall @ IoU0.50: "
            f"{result['recall_iou50']:.3f} "
            f"({result['hits']}/{result['objects']})"
        )

        print(
            f"Proposals/frame: "
            f"{result['mean_proposals_per_frame']:.1f}"
        )

        print(
            f"Latency median: "
            f"{result['median_latency_ms']:.1f} ms"
        )

        print(
            f"Latency p95: "
            f"{result['p95_latency_ms']:.1f} ms"
        )

        print()
        print(
            "Per class:"
        )

        for cls, row in (
            result["per_class"]
            .items()
        ):
            print(
                f"  {cls:20s} "
                f"{row['hits']}/{row['n']} "
                f"R={row['recall_iou50']:.3f} "
                f"meanIoU={row['mean_best_iou']:.3f}"
            )

    best = max(
        results,
        key=lambda r:
            (
                r["recall_iou50"],
                -r[
                    "mean_proposals_per_frame"
                ],
            )
    )

    RESULTS_PATH.write_text(
        json.dumps(
            {
                "best":
                    best,
                "all":
                    results,
            },
            indent=2,
        )
    )

    print()
    print("=" * 72)
    print("BEST OPERATING POINT")
    print("=" * 72)

    print(
        "Confidence:",
        best["confidence"],
    )

    print(
        "Recall @ IoU0.50:",
        f"{best['recall_iou50']:.3f}",
    )

    print(
        "Hits:",
        f"{best['hits']}/{best['objects']}",
    )

    print(
        "Proposals/frame:",
        f"{best['mean_proposals_per_frame']:.1f}",
    )

    print(
        "Median latency:",
        f"{best['median_latency_ms']:.1f} ms",
    )

    print()
    print(
        "Saved:",
        RESULTS_PATH,
    )


if __name__ == "__main__":
    main()
