#!/usr/bin/env python3

import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]

BASE = (
    ROOT
    / "drone"
    / "reference"
    / "official-drone-flyby"
    / "src"
    / "helsinki"
)

ANN = BASE / "annotations"
IMAGES = BASE / "images"

OUT = (
    ROOT
    / "drone"
    / "artifacts"
    / "exp_d027"
    / "class_reference_sheet.jpg"
)

CELL_W = 320
CELL_H = 260
COLS = 4


def find_image(frame):
    for suffix in [".png", ".jpg", ".jpeg"]:
        p = IMAGES / f"frame_{frame:06d}{suffix}"
        if p.exists():
            return p

    raise FileNotFoundError(frame)


def main():
    samples = defaultdict(list)

    for p in sorted(ANN.glob("*.json")):
        data = json.loads(p.read_text())

        for ann in data["annotations"]:
            x1, y1, x2, y2 = ann["bbox"]

            samples[ann["object_id"]].append({
                "frame": data["frame"],
                "bbox": ann["bbox"],
                "area": (x2 - x1) * (y2 - y1),
            })

    cards = []

    for class_name in sorted(samples):
        options = sorted(
            samples[class_name],
            key=lambda x: x["area"],
            reverse=True,
        )

        sample = options[0]

        image = cv2.imread(
            str(find_image(sample["frame"]))
        )

        x1, y1, x2, y2 = sample["bbox"]

        w = x2 - x1
        h = y2 - y1

        pad_x = max(20, int(w * 0.6))
        pad_y = max(20, int(h * 0.6))

        ih, iw = image.shape[:2]

        cx1 = max(0, x1 - pad_x)
        cy1 = max(0, y1 - pad_y)
        cx2 = min(iw, x2 + pad_x)
        cy2 = min(ih, y2 + pad_y)

        crop = image[cy1:cy2, cx1:cx2].copy()

        cv2.rectangle(
            crop,
            (x1 - cx1, y1 - cy1),
            (x2 - cx1, y2 - cy1),
            (0, 0, 255),
            3,
        )

        scale = min(
            (CELL_W - 20) / crop.shape[1],
            (CELL_H - 45) / crop.shape[0],
        )

        scale = min(scale, 6.0)

        crop = cv2.resize(
            crop,
            (
                max(1, int(crop.shape[1] * scale)),
                max(1, int(crop.shape[0] * scale)),
            ),
            interpolation=(
                cv2.INTER_NEAREST
                if scale > 1
                else cv2.INTER_AREA
            ),
        )

        card = np.zeros(
            (CELL_H, CELL_W, 3),
            dtype=np.uint8,
        )

        x = (CELL_W - crop.shape[1]) // 2
        y = 38

        card[
            y:y + crop.shape[0],
            x:x + crop.shape[1],
        ] = crop

        cv2.putText(
            card,
            class_name,
            (8, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cards.append(card)

    rows = []

    for i in range(0, len(cards), COLS):
        row = cards[i:i + COLS]

        while len(row) < COLS:
            row.append(np.zeros_like(cards[0]))

        rows.append(np.hstack(row))

    sheet = np.vstack(rows)

    OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cv2.imwrite(
        str(OUT),
        sheet,
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )

    print("Classes:", len(cards))
    print("Saved:", OUT)


if __name__ == "__main__":
    main()
