#!/usr/bin/env python3

from __future__ import annotations

import json
import statistics
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


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
    / "exp_d005"
)

RESULT_PATH = (
    ARTIFACT_DIR
    / "motion_proposal_results.json"
)

VIEW_WIDTH = 960
VIEW_HEIGHT = 540

SOURCE_WIDTH = 3840
SOURCE_HEIGHT = 2160

SCALE_X = VIEW_WIDTH / SOURCE_WIDTH
SCALE_Y = VIEW_HEIGHT / SOURCE_HEIGHT

# Residual thresholds to screen quickly.
DIFF_THRESHOLDS = [12, 20, 30, 45]

IOU_THRESHOLDS = [0.30, 0.50]

# Ignore tiny residual speckles.
MIN_COMPONENT_AREA = 6

# Avoid one giant component swallowing the frame.
MAX_COMPONENT_AREA = 40000

# Expand residual components before evaluating them as proposals.
BOX_PADDING = 6


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
        f"No image found for frame {frame}"
    )


def source_bbox_to_l0(
    bbox: list[int | float],
) -> np.ndarray:
    x1, y1, x2, y2 = map(float, bbox)

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
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))

    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)

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

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def estimate_affine(
    previous_gray: np.ndarray,
    current_gray: np.ndarray,
) -> tuple[np.ndarray, int]:
    """
    Estimate global previous->current image motion using ORB + RANSAC.
    """

    orb = cv2.ORB_create(
        nfeatures=2500,
        scaleFactor=1.2,
        nlevels=8,
        fastThreshold=10,
    )

    kp1, des1 = orb.detectAndCompute(
        previous_gray,
        None,
    )

    kp2, des2 = orb.detectAndCompute(
        current_gray,
        None,
    )

    if (
        des1 is None
        or des2 is None
        or len(kp1) < 8
        or len(kp2) < 8
    ):
        return (
            np.asarray(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ],
                dtype=np.float32,
            ),
            0,
        )

    matcher = cv2.BFMatcher(
        cv2.NORM_HAMMING,
        crossCheck=True,
    )

    matches = matcher.match(
        des1,
        des2,
    )

    matches = sorted(
        matches,
        key=lambda m: m.distance,
    )

    # Use the strongest matches only.
    matches = matches[
        : min(600, len(matches))
    ]

    if len(matches) < 8:
        return (
            np.asarray(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ],
                dtype=np.float32,
            ),
            0,
        )

    src = np.float32(
        [
            kp1[m.queryIdx].pt
            for m in matches
        ]
    ).reshape(-1, 1, 2)

    dst = np.float32(
        [
            kp2[m.trainIdx].pt
            for m in matches
        ]
    ).reshape(-1, 1, 2)

    matrix, inlier_mask = (
        cv2.estimateAffinePartial2D(
            src,
            dst,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
            maxIters=3000,
            confidence=0.995,
            refineIters=20,
        )
    )

    if matrix is None:
        matrix = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )

    inliers = (
        int(inlier_mask.sum())
        if inlier_mask is not None
        else 0
    )

    return (
        matrix.astype(np.float32),
        inliers,
    )


def make_proposals(
    residual: np.ndarray,
    threshold: int,
) -> list[np.ndarray]:
    _, binary = cv2.threshold(
        residual,
        threshold,
        255,
        cv2.THRESH_BINARY,
    )

    # Remove isolated noise but join nearby changed pixels.
    kernel_open = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3),
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel_open,
    )

    kernel_close = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (7, 7),
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel_close,
        iterations=1,
    )

    num_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8,
        )
    )

    proposals = []

    for component_id in range(
        1,
        num_labels,
    ):
        x = int(
            stats[
                component_id,
                cv2.CC_STAT_LEFT,
            ]
        )

        y = int(
            stats[
                component_id,
                cv2.CC_STAT_TOP,
            ]
        )

        w = int(
            stats[
                component_id,
                cv2.CC_STAT_WIDTH,
            ]
        )

        h = int(
            stats[
                component_id,
                cv2.CC_STAT_HEIGHT,
            ]
        )

        area = int(
            stats[
                component_id,
                cv2.CC_STAT_AREA,
            ]
        )

        if area < MIN_COMPONENT_AREA:
            continue

        if area > MAX_COMPONENT_AREA:
            continue

        x1 = max(
            0,
            x - BOX_PADDING,
        )

        y1 = max(
            0,
            y - BOX_PADDING,
        )

        x2 = min(
            VIEW_WIDTH,
            x + w + BOX_PADDING,
        )

        y2 = min(
            VIEW_HEIGHT,
            y + h + BOX_PADDING,
        )

        proposals.append(
            np.asarray(
                [x1, y1, x2, y2],
                dtype=np.float32,
            )
        )

    return proposals


def main() -> None:
    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    annotation_paths = sorted(
        ANNOTATION_DIR.glob(
            "frame_*.json"
        )
    )

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

        source = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR,
        )

        if source is None:
            raise RuntimeError(
                f"Could not read {image_path}"
            )

        view = cv2.resize(
            source,
            (VIEW_WIDTH, VIEW_HEIGHT),
            interpolation=cv2.INTER_AREA,
        )

        gray = cv2.cvtColor(
            view,
            cv2.COLOR_BGR2GRAY,
        )

        gt = []

        for annotation in (
            annotation_data["annotations"]
        ):
            gt.append(
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
            )

        frames.append(
            {
                "frame":
                    frame_number,
                "view":
                    view,
                "gray":
                    gray,
                "gt":
                    gt,
            }
        )

    print(
        f"Frames loaded: {len(frames)}"
    )

    results = {}

    for threshold in DIFF_THRESHOLDS:
        print()
        print("=" * 76)
        print(
            f"Residual threshold: {threshold}"
        )
        print("=" * 76)

        total_gt = 0

        hits = {
            iou: 0
            for iou in IOU_THRESHOLDS
        }

        class_total = Counter()

        class_hits = {
            iou: Counter()
            for iou in IOU_THRESHOLDS
        }

        proposal_counts = []

        latencies = []

        affine_inliers = []

        frame_records = []

        # Frame 0 has no previous frame.
        for i in range(
            1,
            len(frames),
        ):
            previous = frames[i - 1]
            current = frames[i]

            start = time.perf_counter()

            matrix, inliers = (
                estimate_affine(
                    previous["gray"],
                    current["gray"],
                )
            )

            warped_previous = cv2.warpAffine(
                previous["gray"],
                matrix,
                (VIEW_WIDTH, VIEW_HEIGHT),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT,
            )

            residual = cv2.absdiff(
                current["gray"],
                warped_previous,
            )

            # Small blur stabilizes pixel-level interpolation noise.
            residual = cv2.GaussianBlur(
                residual,
                (3, 3),
                0,
            )

            proposals = make_proposals(
                residual,
                threshold,
            )

            latency_ms = (
                time.perf_counter()
                - start
            ) * 1000.0

            proposal_counts.append(
                len(proposals)
            )

            latencies.append(
                latency_ms
            )

            affine_inliers.append(
                inliers
            )

            gt_records = []

            for gt in current["gt"]:
                class_name = gt["class"]
                gt_box = gt["bbox"]

                total_gt += 1
                class_total[
                    class_name
                ] += 1

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

                gt_hit_record = {}

                for iou_threshold in (
                    IOU_THRESHOLDS
                ):
                    matched = bool(
                        best_iou
                        >= iou_threshold
                    )

                    gt_hit_record[
                        str(
                            iou_threshold
                        )
                    ] = matched

                    if matched:
                        hits[
                            iou_threshold
                        ] += 1

                        class_hits[
                            iou_threshold
                        ][
                            class_name
                        ] += 1

                gt_records.append(
                    {
                        "class":
                            class_name,
                        "best_iou":
                            float(best_iou),
                        "hits":
                            gt_hit_record,
                    }
                )

            frame_records.append(
                {
                    "frame":
                        current[
                            "frame"
                        ],
                    "num_proposals":
                        len(proposals),
                    "latency_ms":
                        float(
                            latency_ms
                        ),
                    "affine_inliers":
                        int(inliers),
                    "affine_matrix":
                        matrix.tolist(),
                    "gt":
                        gt_records,
                }
            )

        threshold_result = {
            "threshold":
                threshold,
            "evaluated_gt":
                total_gt,
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
                "mean":
                    float(
                        statistics.mean(
                            latencies
                        )
                    ),
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
                "max":
                    float(
                        max(latencies)
                    ),
            },
            "affine_inliers": {
                "median":
                    float(
                        statistics.median(
                            affine_inliers
                        )
                    ),
                "min":
                    int(
                        min(
                            affine_inliers
                        )
                    ),
            },
            "recall": {},
            "per_class_recall": {},
            "frames":
                frame_records,
        }

        for iou_threshold in (
            IOU_THRESHOLDS
        ):
            threshold_result[
                "recall"
            ][
                str(
                    iou_threshold
                )
            ] = (
                hits[
                    iou_threshold
                ]
                / total_gt
            )

            per_class = {}

            for class_name in sorted(
                class_total
            ):
                per_class[
                    class_name
                ] = (
                    class_hits[
                        iou_threshold
                    ][
                        class_name
                    ]
                    / class_total[
                        class_name
                    ]
                )

            threshold_result[
                "per_class_recall"
            ][
                str(
                    iou_threshold
                )
            ] = per_class

        results[
            str(threshold)
        ] = threshold_result

        print(
            "Proposals/frame: "
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

        print(
            "Affine inliers: "
            f"median="
            f"{threshold_result['affine_inliers']['median']:.0f}, "
            f"min="
            f"{threshold_result['affine_inliers']['min']}"
        )

        for iou_threshold in (
            IOU_THRESHOLDS
        ):
            recall = (
                threshold_result[
                    "recall"
                ][
                    str(
                        iou_threshold
                    )
                ]
            )

            print(
                f"Proposal recall @ "
                f"IoU {iou_threshold:.2f}: "
                f"{recall:.4f}"
            )

    # Pick the threshold with highest IoU@0.30 recall.
    best_threshold = max(
        DIFF_THRESHOLDS,
        key=lambda t: (
            results[
                str(t)
            ][
                "recall"
            ][
                "0.3"
            ]
        ),
    )

    best = results[
        str(
            best_threshold
        )
    ]

    output = {
        "method":
            "ORB affine motion compensation + residual components",
        "view":
            "L0 960x540",
        "thresholds":
            DIFF_THRESHOLDS,
        "best_threshold_by_iou_0.30":
            best_threshold,
        "results":
            results,
    }

    with RESULT_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            output,
            f,
            indent=2,
        )

    print()
    print("=" * 76)
    print(
        "BEST PER-CLASS RECALL "
        f"(threshold={best_threshold})"
    )
    print("=" * 76)

    print(
        f"{'class':20s} "
        f"{'IoU@0.30':>10s} "
        f"{'IoU@0.50':>10s}"
    )

    print("-" * 44)

    classes = sorted(
        best[
            "per_class_recall"
        ][
            "0.3"
        ]
    )

    for class_name in classes:
        r03 = (
            best[
                "per_class_recall"
            ][
                "0.3"
            ][
                class_name
            ]
        )

        r05 = (
            best[
                "per_class_recall"
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
