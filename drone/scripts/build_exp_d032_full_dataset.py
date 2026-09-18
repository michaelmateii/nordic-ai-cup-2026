#!/usr/bin/env python3

from __future__ import annotations

import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[2]

CAPTURE = (
    ROOT
    / "drone"
    / "captures"
    / "3224a582bfbf4273a028497662b7aa7c"
    / "frames"
)

CANONICAL = (
    ROOT
    / "drone"
    / "artifacts"
    / "exp_d029"
    / "canonical_annotations.json"
)

OUT = (
    ROOT
    / "drone"
    / "artifacts"
    / "exp_d032"
)

DATASET = OUT / "dataset"


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

CLASS_TO_IDX = {
    name: i
    for i, name in enumerate(CLASSES)
}


def frame_path(frame_index: int) -> Path:
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


def write_yolo(
    path: Path,
    rows,
    width: int,
    height: int,
):
    lines = []

    for row in rows:
        x1, y1, x2, y2 = (
            map(float, row["bbox"])
        )

        cx = (
            (x1 + x2) / 2
            / width
        )

        cy = (
            (y1 + y2) / 2
            / height
        )

        bw = (
            (x2 - x1)
            / width
        )

        bh = (
            (y2 - y1)
            / height
        )

        cls = CLASS_TO_IDX[
            row["class"]
        ]

        lines.append(
            f"{cls} "
            f"{cx:.8f} "
            f"{cy:.8f} "
            f"{bw:.8f} "
            f"{bh:.8f}"
        )

    path.write_text(
        "\n".join(lines)
        + ("\n" if lines else "")
    )


def main():
    rows = json.loads(
        CANONICAL.read_text()
    )

    if DATASET.exists():
        shutil.rmtree(
            DATASET
        )

    image_dir = (
        DATASET
        / "images"
        / "train"
    )

    label_dir = (
        DATASET
        / "labels"
        / "train"
    )

    image_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    label_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    by_frame = defaultdict(list)

    for row in rows:
        by_frame[
            int(row["frame_index"])
        ].append(row)

    for frame_index, frame_rows in sorted(
        by_frame.items()
    ):
        src = frame_path(
            frame_index
        )

        dst = (
            image_dir
            / src.name
        )

        shutil.copy2(
            src,
            dst,
        )

        image = cv2.imread(
            str(src)
        )

        if image is None:
            raise RuntimeError(
                f"Could not read {src}"
            )

        h, w = image.shape[:2]

        write_yolo(
            label_dir
            / f"{src.stem}.txt",
            frame_rows,
            w,
            h,
        )

    yaml_path = (
        DATASET
        / "dataset.yaml"
    )

    names = "\n".join(
        f"  {i}: {name}"
        for i, name in enumerate(
            CLASSES
        )
    )

    # val points at train only because Ultralytics
    # requires a val key. Training scripts use val=False.
    yaml_path.write_text(
        f"""path: {DATASET}
train: images/train
val: images/train

names:
{names}
"""
    )

    class_counts = Counter(
        row["class"]
        for row in rows
    )

    source_counts = Counter(
        row["source"]
        for row in rows
    )

    print("=" * 80)
    print("EXP-D032 FULL PRODUCTION DATASET")
    print("=" * 80)

    print(
        "Frames:",
        len(by_frame),
    )

    print(
        "Boxes:",
        len(rows),
    )

    print(
        "Manual:",
        source_counts["manual"],
    )

    print(
        "Pseudo:",
        source_counts["pseudo"],
    )

    print()
    print("CLASS COUNTS")

    for cls in CLASSES:
        n = class_counts[cls]

        if n:
            print(
                f"  {cls:20s} {n}"
            )

    print()
    print(
        "Missing classes:"
    )

    for cls in CLASSES:
        if class_counts[cls] == 0:
            print(
                f"  {cls}"
            )

    print()
    print(
        "Dataset:",
        DATASET,
    )

    print(
        "YAML:",
        yaml_path,
    )


if __name__ == "__main__":
    main()
