#!/usr/bin/env python3

from __future__ import annotations

import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]

CAPTURE = (
    ROOT
    / "drone"
    / "captures"
    / "3224a582bfbf4273a028497662b7aa7c"
    / "frames"
)

CANONICAL_PATH = (
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
    / "exp_d029a"
)

DATASET = OUT / "dataset"
HOLDOUT_PATH = OUT / "manual_holdout.json"
TRAIN_PATH = OUT / "train_annotations.json"

EXCLUSION_GAP = 10


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


HOLDOUT_SPECS = [
    # frame 125: three classes on one held-out image
    {
        "class": "hangar",
        "frame_index": 125,
        "bbox": [533, 39, 591, 78],
    },
    {
        "class": "jet_plane",
        "frame_index": 125,
        "bbox": [554, 80, 576, 106],
    },
    {
        "class": "small_plane",
        "frame_index": 125,
        "bbox": [627, 61, 647, 89],
    },
    {
        "class": "small_plane",
        "frame_index": 125,
        "bbox": [597, 59, 622, 85],
    },

    # frame 107: helicopter + large tower
    {
        "class": "helicopter",
        "frame_index": 107,
        "bbox": [351, 506, 396, 539],
    },
    {
        "class": "large_tower",
        "frame_index": 107,
        "bbox": [438, 204, 455, 230],
    },

    # frame 205: launcher + tank
    {
        "class": "large_launcher",
        "frame_index": 205,
        "bbox": [539, 26, 594, 89],
    },
    {
        "class": "tank",
        "frame_index": 205,
        "bbox": [244, 474, 273, 497],
    },

    # frame 82: medium plane + second tank
    {
        "class": "medium_plane",
        "frame_index": 82,
        "bbox": [625, 490, 644, 525],
    },
    {
        "class": "tank",
        "frame_index": 82,
        "bbox": [640, 216, 671, 237],
    },

    # mine roller
    {
        "class": "mine_roller",
        "frame_index": 14,
        "bbox": [417, 439, 461, 468],
    },

    # two small-tower holdouts
    {
        "class": "small_tower",
        "frame_index": 182,
        "bbox": [553, 344, 573, 364],
    },
    {
        "class": "small_tower",
        "frame_index": 186,
        "bbox": [542, 504, 561, 528],
    },

    # second large-tower holdout
    {
        "class": "large_tower",
        "frame_index": 222,
        "bbox": [263, 158, 293, 183],
    },
]


def image_path(frame_index: int) -> Path:
    matches = sorted(
        CAPTURE.glob(
            f"*index_{frame_index:06d}.png"
        )
    )

    if not matches:
        raise FileNotFoundError(
            f"No frame_index={frame_index}"
        )

    return matches[0]


def choose_holdout(manual_rows):
    holdout = []

    for spec in HOLDOUT_SPECS:
        matches = [
            row
            for row in manual_rows
            if (
                row["class"] == spec["class"]
                and row["frame_index"] == spec["frame_index"]
                and row["bbox"] == spec["bbox"]
            )
        ]

        if len(matches) != 1:
            raise RuntimeError(
                "Could not uniquely resolve holdout: "
                f"{spec} matches={len(matches)}"
            )

        holdout.append(matches[0])

    return holdout


def same_annotation(a, b):
    return (
        a["source"] == "manual"
        and b["source"] == "manual"
        and a["class"] == b["class"]
        and a["frame_index"] == b["frame_index"]
        and a["bbox"] == b["bbox"]
    )


def write_yolo(path, rows, width, height):
    lines = []

    for row in rows:
        x1, y1, x2, y2 = row["bbox"]

        cx = ((x1 + x2) / 2) / width
        cy = ((y1 + y2) / 2) / height
        bw = (x2 - x1) / width
        bh = (y2 - y1) / height

        cls = CLASS_TO_IDX[row["class"]]

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
        CANONICAL_PATH.read_text()
    )

    manual = [
        r
        for r in rows
        if r["source"] == "manual"
    ]

    holdout = choose_holdout(
        manual
    )

    holdout_frames = {
        r["frame_index"]
        for r in holdout
    }

    # Remove:
    # 1. exact held-out manual annotations
    # 2. ALL annotations on held-out frames
    # 3. propagated annotations within ±10 frames of a
    #    same-class held-out manual object
    train = []

    removed_near_holdout = []

    for row in rows:
        if row["frame_index"] in holdout_frames:
            continue

        is_exact_holdout = any(
            same_annotation(
                row,
                h,
            )
            for h in holdout
        )

        if is_exact_holdout:
            continue

        leak = False

        if row["source"] == "pseudo":
            for h in holdout:
                if row["class"] != h["class"]:
                    continue

                if (
                    abs(
                        row["frame_index"]
                        - h["frame_index"]
                    )
                    <= EXCLUSION_GAP
                ):
                    leak = True
                    break

        if leak:
            removed_near_holdout.append(row)
            continue

        train.append(row)

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    HOLDOUT_PATH.write_text(
        json.dumps(
            holdout,
            indent=2,
        )
    )

    TRAIN_PATH.write_text(
        json.dumps(
            train,
            indent=2,
        )
    )

    if DATASET.exists():
        shutil.rmtree(DATASET)

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

    for row in train:
        by_frame[
            row["frame_index"]
        ].append(row)

    for frame_index, frame_rows in sorted(
        by_frame.items()
    ):
        src = image_path(
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
        for i, name in enumerate(CLASSES)
    )

    yaml_path.write_text(
        f"""path: {DATASET}
train: images/train
# Required by Ultralytics' dataset schema.
# EXP-D029 does NOT use this for model selection; train(val=False)
# and the independent manual holdout evaluator is authoritative.
val: images/train

names:
{names}
"""
    )

    train_counts = Counter(
        r["class"]
        for r in train
    )

    holdout_counts = Counter(
        r["class"]
        for r in holdout
    )

    print("=" * 80)
    print("EXP-D029A LEAKAGE-AWARE SPLIT")
    print("=" * 80)

    print(
        "Canonical boxes:",
        len(rows),
    )

    print(
        "Training boxes:",
        len(train),
    )

    print(
        "Training frames:",
        len(by_frame),
    )

    print(
        "Holdout manual boxes:",
        len(holdout),
    )

    print(
        "Holdout frames:",
        len(holdout_frames),
    )

    print(
        "Pseudo boxes excluded near holdout:",
        len(removed_near_holdout),
    )

    print()

    print("TRAIN COUNTS")
    for cls in CLASSES:
        n = train_counts[cls]

        if n:
            print(
                f"  {cls:20s} {n}"
            )

    print()

    print("MANUAL HOLDOUT COUNTS")
    for cls in CLASSES:
        n = holdout_counts[cls]

        if n:
            print(
                f"  {cls:20s} {n}"
            )

    print()

    print("Holdout objects:")
    for row in sorted(
        holdout,
        key=lambda r: (
            r["class"],
            r["frame_index"],
        ),
    ):
        print(
            f"  {row['class']:20s} "
            f"frame={row['frame_index']:3d} "
            f"bbox={row['bbox']}"
        )

    print()
    print("Dataset:", DATASET)
    print("Holdout:", HOLDOUT_PATH)
    print("Train:", TRAIN_PATH)


if __name__ == "__main__":
    main()
