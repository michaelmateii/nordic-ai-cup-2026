#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]

FRAMES_DIR = (
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
    / "exp_d024"
)

TARGET_LEVEL = 1
TARGET_CENTER = (1920, 1080)

MIN_AREA = 8
MAX_AREA = 5000

THRESHOLDS = [
    20,
    30,
    40,
]


def load_frames():
    records = []

    for meta_path in sorted(
        FRAMES_DIR.glob("*.json")
    ):
        meta = json.loads(
            meta_path.read_text()
        )

        view = meta["view"]

        if (
            view["resolution_level"]
            != TARGET_LEVEL
        ):
            continue

        if (
            view["center_x"],
            view["center_y"],
        ) != TARGET_CENTER:
            continue

        image_path = (
            meta_path.with_suffix(
                ".png"
            )
        )

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            continue

        records.append(
            {
                "frame_index":
                    int(
                        meta["frame_index"]
                    ),
                "path":
                    image_path,
                "image":
                    image,
            }
        )

    records.sort(
        key=lambda r:
            r["frame_index"]
    )

    return records


def build_background(records):
    stack = np.stack(
        [
            cv2.cvtColor(
                r["image"],
                cv2.COLOR_BGR2GRAY,
            )
            for r in records
        ],
        axis=0,
    )

    median = np.median(
        stack,
        axis=0,
    ).astype(
        np.uint8
    )

    return median


def candidates(
    image,
    background,
    threshold,
):
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    diff = cv2.absdiff(
        gray,
        background,
    )

    _, mask = cv2.threshold(
        diff,
        threshold,
        255,
        cv2.THRESH_BINARY,
    )

    kernel = np.ones(
        (3, 3),
        dtype=np.uint8,
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
    )

    mask = cv2.dilate(
        mask,
        kernel,
        iterations=1,
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    boxes = []

    for contour in contours:
        x, y, w, h = (
            cv2.boundingRect(
                contour
            )
        )

        area = w * h

        if not (
            MIN_AREA
            <= area
            <= MAX_AREA
        ):
            continue

        boxes.append(
            (
                x,
                y,
                x + w,
                y + h,
            )
        )

    return mask, boxes


def make_montage(
    records,
    background,
    threshold,
):
    thumbs = []

    for record in records:
        image = record[
            "image"
        ].copy()

        _, boxes = candidates(
            image,
            background,
            threshold,
        )

        for x1, y1, x2, y2 in boxes:
            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                1,
            )

        image = cv2.resize(
            image,
            (480, 270),
            interpolation=
                cv2.INTER_AREA,
        )

        cv2.putText(
            image,
            (
                f"idx="
                f"{record['frame_index']:03d} "
                f"boxes={len(boxes)}"
            ),
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        thumbs.append(
            image
        )

    cols = 4

    rows = []

    for start in range(
        0,
        len(thumbs),
        cols,
    ):
        row = thumbs[
            start:
            start + cols
        ]

        while len(row) < cols:
            row.append(
                np.zeros_like(
                    thumbs[0]
                )
            )

        rows.append(
            np.hstack(
                row
            )
        )

    return np.vstack(
        rows
    )


def main():
    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = load_frames()

    print(
        "Matching L1 frames:",
        len(records),
    )

    print(
        "Frame indices:",
        [
            r["frame_index"]
            for r in records
        ],
    )

    if len(records) < 5:
        raise RuntimeError(
            "Not enough matching views."
        )

    background = (
        build_background(
            records
        )
    )

    cv2.imwrite(
        str(
            OUT
            / "median_background.png"
        ),
        background,
    )

    for threshold in THRESHOLDS:
        counts = []

        for record in records:
            _, boxes = candidates(
                record["image"],
                background,
                threshold,
            )

            counts.append(
                len(boxes)
            )

        montage = make_montage(
            records,
            background,
            threshold,
        )

        path = (
            OUT
            / (
                "temporal_candidates"
                f"_thr{threshold}.jpg"
            )
        )

        cv2.imwrite(
            str(path),
            montage,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                92,
            ],
        )

        print()
        print(
            f"Threshold {threshold}"
        )

        print(
            f"  mean boxes: "
            f"{np.mean(counts):.1f}"
        )

        print(
            f"  median:     "
            f"{np.median(counts):.1f}"
        )

        print(
            f"  max:        "
            f"{max(counts)}"
        )

        print(
            f"  montage:    "
            f"{path}"
        )


if __name__ == "__main__":
    main()
