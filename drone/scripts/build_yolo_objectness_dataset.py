#!/usr/bin/env python3

from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path

import cv2


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

OUTPUT_ROOT = (
    ROOT
    / "drone"
    / "artifacts"
    / "exp_d007"
    / "dataset"
)

VIEW_WIDTH = 960
VIEW_HEIGHT = 540

SOURCE_WIDTH = 3840
SOURCE_HEIGHT = 2160

TRAIN_MAX_FRAME = 20
VAL_MIN_FRAME = 22


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


def xyxy_to_yolo(
    bbox: list[int | float],
) -> tuple[float, float, float, float]:

    x1, y1, x2, y2 = map(float, bbox)

    cx = ((x1 + x2) / 2.0) / SOURCE_WIDTH
    cy = ((y1 + y2) / 2.0) / SOURCE_HEIGHT

    w = (x2 - x1) / SOURCE_WIDTH
    h = (y2 - y1) / SOURCE_HEIGHT

    return cx, cy, w, h


def split_for_frame(frame: int) -> str | None:

    if frame <= TRAIN_MAX_FRAME:
        return "train"

    if frame >= VAL_MIN_FRAME:
        return "val"

    return None


def main() -> None:

    if OUTPUT_ROOT.exists():
        shutil.rmtree(
            OUTPUT_ROOT
        )

    for split in ("train", "val"):
        (
            OUTPUT_ROOT
            / "images"
            / split
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

        (
            OUTPUT_ROOT
            / "labels"
            / split
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

    split_frames = Counter()
    split_boxes = Counter()
    split_classes = {
        "train": Counter(),
        "val": Counter(),
    }

    annotation_paths = sorted(
        ANNOTATION_DIR.glob(
            "frame_*.json"
        )
    )

    for annotation_path in annotation_paths:

        with annotation_path.open(
            "r",
            encoding="utf-8",
        ) as f:
            data = json.load(f)

        frame = int(
            data["frame"]
        )

        split = split_for_frame(
            frame
        )

        if split is None:
            continue

        source_path = find_image(
            frame
        )

        image = cv2.imread(
            str(source_path),
            cv2.IMREAD_COLOR,
        )

        if image is None:
            raise RuntimeError(
                f"Failed reading {source_path}"
            )

        image_l0 = cv2.resize(
            image,
            (
                VIEW_WIDTH,
                VIEW_HEIGHT,
            ),
            interpolation=cv2.INTER_AREA,
        )

        image_name = (
            f"frame_{frame:06d}.jpg"
        )

        output_image = (
            OUTPUT_ROOT
            / "images"
            / split
            / image_name
        )

        cv2.imwrite(
            str(output_image),
            image_l0,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                95,
            ],
        )

        labels = []

        for annotation in (
            data["annotations"]
        ):

            cx, cy, w, h = (
                xyxy_to_yolo(
                    annotation["bbox"]
                )
            )

            # One class only: target = class 0.
            labels.append(
                "0 "
                f"{cx:.8f} "
                f"{cy:.8f} "
                f"{w:.8f} "
                f"{h:.8f}"
            )

            split_boxes[
                split
            ] += 1

            split_classes[
                split
            ][
                annotation[
                    "object_id"
                ]
            ] += 1

        output_label = (
            OUTPUT_ROOT
            / "labels"
            / split
            / image_name.replace(
                ".jpg",
                ".txt",
            )
        )

        output_label.write_text(
            "\n".join(labels)
            + "\n",
            encoding="utf-8",
        )

        split_frames[
            split
        ] += 1

    yaml_path = (
        OUTPUT_ROOT
        / "dataset.yaml"
    )

    yaml_path.write_text(
        f"""path: {OUTPUT_ROOT}
train: images/train
val: images/val

names:
  0: target
""",
        encoding="utf-8",
    )

    print("=" * 72)
    print("EXP-D007 DATASET")
    print("=" * 72)

    print(
        f"Train frames: "
        f"{split_frames['train']}"
    )

    print(
        f"Train boxes:  "
        f"{split_boxes['train']}"
    )

    print(
        f"Val frames:   "
        f"{split_frames['val']}"
    )

    print(
        f"Val boxes:    "
        f"{split_boxes['val']}"
    )

    print()
    print(
        "Train class appearances:"
    )

    for class_name, count in sorted(
        split_classes["train"].items()
    ):
        print(
            f"  {class_name:20s} "
            f"{count}"
        )

    print()
    print(
        "Val class appearances:"
    )

    for class_name, count in sorted(
        split_classes["val"].items()
    ):
        print(
            f"  {class_name:20s} "
            f"{count}"
        )

    print()
    print(
        f"Dataset: {OUTPUT_ROOT}"
    )

    print(
        f"YAML:    {yaml_path}"
    )


if __name__ == "__main__":
    main()
