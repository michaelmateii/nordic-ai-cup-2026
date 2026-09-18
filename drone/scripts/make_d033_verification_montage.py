#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]

CANDIDATES = (
    ROOT
    / "drone"
    / "artifacts"
    / "exp_d033"
    / "missing_class_candidates"
    / "candidates.json"
)

OUT = (
    ROOT
    / "drone"
    / "artifacts"
    / "exp_d033"
    / "verification"
)

OUT.mkdir(parents=True, exist_ok=True)

WANTED = {
    "spacecraft": {
        65, 67, 71, 72, 74,
        75, 77, 78, 79, 81, 82,
    },
    "condor": {
        67, 72,
    },
}


def make_card(row, cls):
    image = cv2.imread(row["image_path"])

    if image is None:
        return None

    h, w = image.shape[:2]

    x1, y1, x2, y2 = [
        int(round(v))
        for v in row["bbox"]
    ]

    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(x1 + 1, min(w, x2))
    y2 = max(y1 + 1, min(h, y2))

    bw = x2 - x1
    bh = y2 - y1

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    pad_x = max(120, int(bw * 4))
    pad_y = max(120, int(bh * 4))

    rx1 = max(0, cx - bw // 2 - pad_x)
    ry1 = max(0, cy - bh // 2 - pad_y)
    rx2 = min(w, cx + (bw + 1) // 2 + pad_x)
    ry2 = min(h, cy + (bh + 1) // 2 + pad_y)

    crop = image[ry1:ry2, rx1:rx2].copy()

    if crop.size == 0:
        return None

    cv2.rectangle(
        crop,
        (x1 - rx1, y1 - ry1),
        (x2 - rx1, y2 - ry1),
        (0, 0, 255),
        3,
    )

    canvas_w = 520
    image_h = 380
    header_h = 90

    resized = cv2.resize(
        crop,
        (canvas_w, image_h),
        interpolation=cv2.INTER_CUBIC,
    )

    card = np.zeros(
        (header_h + image_h, canvas_w, 3),
        dtype=np.uint8,
    )

    card[header_h:, :] = resized

    line1 = (
        f"{cls} idx={row['frame_index']} "
        f"P={row['class_probability']:.3f} "
        f"O={row['detector_score']:.3f}"
    )

    line2 = (
        f"L{row['resolution_level']} "
        f"center=({row['center_x']},{row['center_y']})"
    )

    line3 = f"bbox={row['bbox']}"

    cv2.putText(
        card, line1, (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56, (255, 255, 255), 1, cv2.LINE_AA,
    )

    cv2.putText(
        card, line2, (8, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48, (255, 255, 255), 1, cv2.LINE_AA,
    )

    cv2.putText(
        card, line3, (8, 76),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42, (255, 255, 255), 1, cv2.LINE_AA,
    )

    return card


def main():
    data = json.loads(
        CANDIDATES.read_text()
    )["candidates"]

    for cls, wanted_indices in WANTED.items():
        rows = []

        for row in data[cls]:
            if row["frame_index"] not in wanted_indices:
                continue

            rows.append(row)

        # Keep highest probability candidate per
        # frame_index + camera center.
        best = {}

        for row in rows:
            key = (
                row["frame_index"],
                row["center_x"],
                row["center_y"],
            )

            if (
                key not in best
                or row["class_probability"]
                > best[key]["class_probability"]
            ):
                best[key] = row

        rows = sorted(
            best.values(),
            key=lambda r: (
                r["frame_index"],
                r["center_y"],
                r["center_x"],
            ),
        )

        cards = []

        for row in rows:
            card = make_card(row, cls)

            if card is not None:
                cards.append(card)

        if not cards:
            continue

        cols = 2

        blank = np.zeros_like(cards[0])

        while len(cards) % cols:
            cards.append(blank.copy())

        sheet_rows = []

        for i in range(0, len(cards), cols):
            sheet_rows.append(
                np.hstack(cards[i:i + cols])
            )

        sheet = np.vstack(sheet_rows)

        out_path = OUT / f"{cls}_verification.jpg"

        cv2.imwrite(
            str(out_path),
            sheet,
            [cv2.IMWRITE_JPEG_QUALITY, 96],
        )

        print(
            f"{cls}: {len(rows)} candidates"
        )
        print(
            f"  saved: {out_path}"
        )


if __name__ == "__main__":
    main()
