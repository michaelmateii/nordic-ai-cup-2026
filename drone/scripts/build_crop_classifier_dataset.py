#!/usr/bin/env python3

from __future__ import annotations

import json
import shutil
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
    / "exp_d013"
    / "dataset"
)

TRAIN_MAX_FRAME = 20
VAL_MIN_FRAME = 22

VIEW_W = 960
VIEW_H = 540

PADDING = 0.30


def find_image(frame: int) -> Path:
    for suffix in ("png", "jpg", "jpeg"):
        path = IMAGE_DIR / f"frame_{frame:06d}.{suffix}"

        if path.exists():
            return path

    matches = sorted(
        IMAGE_DIR.glob(f"*{frame:06d}*")
    )

    if matches:
        return matches[0]

    raise FileNotFoundError(frame)


def split_for_frame(frame: int) -> str | None:
    if frame <= TRAIN_MAX_FRAME:
        return "train"

    if frame >= VAL_MIN_FRAME:
        return "val"

    return None


def bbox_l0(bbox):
    x1, y1, x2, y2 = map(float, bbox)

    return (
        x1 / 4.0,
        y1 / 4.0,
        x2 / 4.0,
        y2 / 4.0,
    )


def crop_object(image, bbox):
    x1, y1, x2, y2 = bbox

    w = x2 - x1
    h = y2 - y1

    px = w * PADDING
    py = h * PADDING

    image_h, image_w = image.shape[:2]

    x1 = max(0, int(round(x1 - px)))
    y1 = max(0, int(round(y1 - py)))
    x2 = min(image_w, int(round(x2 + px)))
    y2 = min(image_h, int(round(y2 + py)))

    return image[y1:y2, x1:x2]


def main() -> None:
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)

    counts = {
        "train": {},
        "val": {},
    }

    for annotation_path in sorted(
        ANNOTATION_DIR.glob("frame_*.json")
    ):
        data = json.loads(
            annotation_path.read_text()
        )

        frame = int(data["frame"])

        split = split_for_frame(frame)

        if split is None:
            continue

        source = cv2.imread(
            str(find_image(frame)),
            cv2.IMREAD_COLOR,
        )

        view = cv2.resize(
            source,
            (VIEW_W, VIEW_H),
            interpolation=cv2.INTER_AREA,
        )

        for index, annotation in enumerate(
            data["annotations"]
        ):
            class_name = annotation["object_id"]

            crop = crop_object(
                view,
                bbox_l0(annotation["bbox"]),
            )

            if crop.size == 0:
                continue

            class_dir = (
                OUTPUT_ROOT
                / split
                / class_name
            )

            class_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            output = (
                class_dir
                / f"frame_{frame:06d}_{index:02d}.jpg"
            )

            cv2.imwrite(
                str(output),
                crop,
                [cv2.IMWRITE_JPEG_QUALITY, 98],
            )

            counts[split][class_name] = (
                counts[split].get(
                    class_name,
                    0,
                )
                + 1
            )

    print("=" * 72)
    print("EXP-D013 CROP DATASET")
    print("=" * 72)

    for split in ("train", "val"):
        print()
        print(split.upper())

        total = 0

        for class_name, count in sorted(
            counts[split].items()
        ):
            print(
                f"  {class_name:20s} {count}"
            )

            total += count

        print(
            f"  TOTAL                {total}"
        )

    print()
    print(
        f"Dataset: {OUTPUT_ROOT}"
    )


if __name__ == "__main__":
    main()
