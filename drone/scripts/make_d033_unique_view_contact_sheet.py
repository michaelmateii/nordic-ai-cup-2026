#!/usr/bin/env python3

import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]

MANIFEST = (
    ROOT
    / "drone"
    / "artifacts"
    / "exp_d033"
    / "unique_views_manifest.json"
)

OUT = (
    ROOT
    / "drone"
    / "artifacts"
    / "exp_d033"
    / "unique_views_contact_sheet.jpg"
)

rows = json.loads(
    MANIFEST.read_text()
)

thumb_w = 320
thumb_h = 180
header_h = 28
cols = 4

cards = []

for row in rows:
    image = cv2.imread(
        row["image"]
    )

    if image is None:
        continue

    thumb = cv2.resize(
        image,
        (thumb_w, thumb_h),
        interpolation=cv2.INTER_AREA,
    )

    card = np.zeros(
        (
            thumb_h + header_h,
            thumb_w,
            3,
        ),
        dtype=np.uint8,
    )

    card[
        header_h:,
        :
    ] = thumb

    label = (
        f"i{row['frame_index']} "
        f"({row['center_x']},"
        f"{row['center_y']})"
    )

    cv2.putText(
        card,
        label,
        (5, 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    cards.append(card)

while len(cards) % cols:
    cards.append(
        np.zeros_like(
            cards[0]
        )
    )

sheet_rows = []

for i in range(
    0,
    len(cards),
    cols,
):
    sheet_rows.append(
        np.hstack(
            cards[i:i + cols]
        )
    )

sheet = np.vstack(
    sheet_rows
)

cv2.imwrite(
    str(OUT),
    sheet,
    [
        cv2.IMWRITE_JPEG_QUALITY,
        90,
    ],
)

print("Saved:", OUT)
