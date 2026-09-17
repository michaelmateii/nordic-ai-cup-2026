#!/usr/bin/env python3

from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]

SEQUENCE = (
    ROOT
    / "drone"
    / "captures"
    / "9204f05e8ffe46f995edd8c093823393"
    / "frames"
)

OUTPUT = (
    ROOT
    / "drone"
    / "artifacts"
    / "exp_d021"
    / "validation_contact_sheet.jpg"
)

N_SAMPLES = 30
COLS = 5


def main():
    paths = sorted(
        SEQUENCE.glob("*.png")
    )

    if not paths:
        raise RuntimeError(
            f"No PNG files found in {SEQUENCE}"
        )

    sample_count = min(
        N_SAMPLES,
        len(paths),
    )

    indices = np.linspace(
        0,
        len(paths) - 1,
        sample_count,
        dtype=int,
    )

    thumbs = []

    for position in indices:
        path = paths[position]

        image = cv2.imread(
            str(path)
        )

        if image is None:
            continue

        image = cv2.resize(
            image,
            (480, 270),
            interpolation=cv2.INTER_AREA,
        )

        cv2.rectangle(
            image,
            (0, 0),
            (480, 30),
            (0, 0, 0),
            -1,
        )

        cv2.putText(
            image,
            path.stem,
            (7, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        thumbs.append(
            image
        )

    if not thumbs:
        raise RuntimeError(
            "No images could be read."
        )

    rows = []

    for start in range(
        0,
        len(thumbs),
        COLS,
    ):
        row = thumbs[
            start:start + COLS
        ]

        while len(row) < COLS:
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

    sheet = np.vstack(
        rows
    )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ok = cv2.imwrite(
        str(OUTPUT),
        sheet,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            92,
        ],
    )

    if not ok:
        raise RuntimeError(
            f"Could not save {OUTPUT}"
        )

    print(
        f"Frames found: {len(paths)}"
    )

    print(
        f"Samples shown: {len(thumbs)}"
    )

    print(
        f"Saved: {OUTPUT}"
    )


if __name__ == "__main__":
    main()
