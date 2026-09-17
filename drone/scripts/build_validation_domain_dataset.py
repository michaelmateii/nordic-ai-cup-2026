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

MANUAL = (
    ROOT
    / "drone"
    / "artifacts"
    / "exp_d027"
    / "validation_seed_annotations.json"
)

PSEUDO = (
    ROOT
    / "drone"
    / "artifacts"
    / "exp_d028d"
    / "accepted_pseudolabels.json"
)

OUT = (
    ROOT
    / "drone"
    / "artifacts"
    / "exp_d029"
)

DATASET = OUT / "dataset"

CANONICAL = OUT / "canonical_annotations.json"

# Late temporal holdout.
HOLDOUT_START = 190
HOLDOUT_END = 249

IOU_DEDUP = 0.70


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


def iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    inter = (
        max(0, x2 - x1)
        * max(0, y2 - y1)
    )

    aa = (
        max(0, a[2] - a[0])
        * max(0, a[3] - a[1])
    )

    ab = (
        max(0, b[2] - b[0])
        * max(0, b[3] - b[1])
    )

    union = aa + ab - inter

    return (
        inter / union
        if union > 0
        else 0.0
    )


def frame_image_path(
    frame_index,
):
    matches = sorted(
        CAPTURE.glob(
            f"*index_{frame_index:06d}.png"
        )
    )

    if not matches:
        raise FileNotFoundError(
            frame_index
        )

    return matches[0]


def load_manual():
    raw = json.loads(
        MANUAL.read_text()
    )

    rows = []

    for frame_key, annotations in raw.items():
        for row in annotations:
            rows.append(
                {
                    "class":
                        row["class"],
                    "frame":
                        int(row["frame"]),
                    "frame_index":
                        int(row["frame_index"]),
                    "bbox":
                        [
                            int(v)
                            for v in row["bbox"]
                        ],
                    "source":
                        "manual",
                }
            )

    return rows


def load_pseudo():
    raw = json.loads(
        PSEUDO.read_text()
    )

    rows = []

    for row in raw:
        rows.append(
            {
                "class":
                    row["class"],
                "frame":
                    int(
                        row.get(
                            "frame",
                            row["target_frame_index"] + 1,
                        )
                    ),
                "frame_index":
                    int(
                        row["target_frame_index"]
                    ),
                "bbox":
                    [
                        int(v)
                        for v in row["bbox"]
                    ],
                "source":
                    "pseudo",
            }
        )

    return rows


def merge_rows():
    manual = load_manual()
    pseudo = load_pseudo()

    grouped = defaultdict(list)

    # Manual goes first and wins conflicts.
    for row in manual:
        grouped[
            row["frame_index"]
        ].append(row)

    for row in pseudo:
        existing = grouped[
            row["frame_index"]
        ]

        duplicate = False

        for old in existing:
            if old["class"] != row["class"]:
                continue

            if (
                iou(
                    old["bbox"],
                    row["bbox"],
                )
                >= IOU_DEDUP
            ):
                duplicate = True
                break

        if not duplicate:
            existing.append(row)

    rows = []

    for frame_index in sorted(grouped):
        rows.extend(
            grouped[frame_index]
        )

    return rows


def write_yolo_label(
    path,
    boxes,
    image_width,
    image_height,
):
    lines = []

    for row in boxes:
        x1, y1, x2, y2 = (
            row["bbox"]
        )

        cx = (
            (x1 + x2) / 2
            / image_width
        )

        cy = (
            (y1 + y2) / 2
            / image_height
        )

        w = (
            (x2 - x1)
            / image_width
        )

        h = (
            (y2 - y1)
            / image_height
        )

        cls = CLASS_TO_IDX[
            row["class"]
        ]

        lines.append(
            f"{cls} "
            f"{cx:.8f} "
            f"{cy:.8f} "
            f"{w:.8f} "
            f"{h:.8f}"
        )

    path.write_text(
        "\n".join(lines)
        + ("\n" if lines else "")
    )


def main():
    if DATASET.exists():
        shutil.rmtree(
            DATASET
        )

    for split in [
        "train",
        "val",
    ]:
        (
            DATASET
            / "images"
            / split
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

        (
            DATASET
            / "labels"
            / split
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

    rows = merge_rows()

    CANONICAL.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    CANONICAL.write_text(
        json.dumps(
            rows,
            indent=2,
        )
    )

    by_frame = defaultdict(list)

    for row in rows:
        by_frame[
            row["frame_index"]
        ].append(row)

    split_counts = Counter()
    class_counts = {
        "train": Counter(),
        "val": Counter(),
    }

    source_counts = Counter()

    for frame_index, boxes in sorted(
        by_frame.items()
    ):
        image_path = frame_image_path(
            frame_index
        )

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            continue

        h, w = image.shape[:2]

        split = (
            "val"
            if (
                HOLDOUT_START
                <= frame_index
                < HOLDOUT_END
            )
            else "train"
        )

        dst_image = (
            DATASET
            / "images"
            / split
            / image_path.name
        )

        shutil.copy2(
            image_path,
            dst_image,
        )

        dst_label = (
            DATASET
            / "labels"
            / split
            / (
                image_path.stem
                + ".txt"
            )
        )

        write_yolo_label(
            dst_label,
            boxes,
            w,
            h,
        )

        split_counts[
            split
        ] += 1

        for row in boxes:
            class_counts[
                split
            ][
                row["class"]
            ] += 1

            source_counts[
                row["source"]
            ] += 1

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

    yaml_path.write_text(
        f"""path: {DATASET}
train: images/train
val: images/val

names:
{names}
"""
    )

    print("=" * 80)
    print("EXP-D029 CANONICAL DATASET")
    print("=" * 80)

    print(
        "Canonical boxes:",
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

    print(
        "Train frames:",
        split_counts["train"],
    )

    print(
        "Val frames:",
        split_counts["val"],
    )

    for split in [
        "train",
        "val",
    ]:
        print()
        print(
            split.upper(),
            "CLASS COUNTS",
        )

        for cls in CLASSES:
            n = class_counts[
                split
            ][cls]

            if n:
                print(
                    f"  {cls:20s} "
                    f"{n}"
                )

    print()
    print(
        "Canonical:",
        CANONICAL,
    )

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
