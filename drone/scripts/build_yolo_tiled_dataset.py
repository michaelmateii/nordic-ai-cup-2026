#!/usr/bin/env python3

from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


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
    / "exp_d010"
    / "dataset"
)

VIEW_W = 960
VIEW_H = 540

COLS = 2
ROWS = 2
OVERLAP = 0.20

TRAIN_MAX_FRAME = 20
VAL_MIN_FRAME = 22


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

    raise FileNotFoundError(
        f"No image found for frame {frame}"
    )


def split_for_frame(
    frame: int,
) -> str | None:
    if frame <= TRAIN_MAX_FRAME:
        return "train"

    if frame >= VAL_MIN_FRAME:
        return "val"

    return None


def make_starts(
    length: int,
    count: int,
    overlap: float,
) -> list[tuple[int, int]]:

    nominal = length / count

    tile_size = int(
        round(
            nominal
            * (1.0 + overlap)
        )
    )

    tile_size = min(
        tile_size,
        length,
    )

    starts = np.linspace(
        0,
        length - tile_size,
        count,
    )

    return [
        (
            int(round(start)),
            int(round(start))
            + tile_size,
        )
        for start in starts
    ]


def source_bbox_to_l0(
    bbox,
) -> tuple[float, float, float, float]:

    x1, y1, x2, y2 = map(
        float,
        bbox,
    )

    return (
        x1 / 4.0,
        y1 / 4.0,
        x2 / 4.0,
        y2 / 4.0,
    )


def bbox_center(
    bbox,
) -> tuple[float, float]:

    x1, y1, x2, y2 = bbox

    return (
        (x1 + x2) / 2.0,
        (y1 + y2) / 2.0,
    )


def clip_to_tile(
    bbox,
    tile,
):
    x1, y1, x2, y2 = bbox

    clipped_x1 = max(
        x1,
        tile["x1"],
    )

    clipped_y1 = max(
        y1,
        tile["y1"],
    )

    clipped_x2 = min(
        x2,
        tile["x2"],
    )

    clipped_y2 = min(
        y2,
        tile["y2"],
    )

    if (
        clipped_x2 <= clipped_x1
        or clipped_y2 <= clipped_y1
    ):
        return None

    return (
        clipped_x1 - tile["x1"],
        clipped_y1 - tile["y1"],
        clipped_x2 - tile["x1"],
        clipped_y2 - tile["y1"],
    )


def to_yolo(
    bbox,
    tile_w: int,
    tile_h: int,
):
    x1, y1, x2, y2 = bbox

    cx = (
        (x1 + x2)
        / 2.0
        / tile_w
    )

    cy = (
        (y1 + y2)
        / 2.0
        / tile_h
    )

    w = (
        x2 - x1
    ) / tile_w

    h = (
        y2 - y1
    ) / tile_h

    return cx, cy, w, h


def main() -> None:

    if OUTPUT_ROOT.exists():
        shutil.rmtree(
            OUTPUT_ROOT
        )

    for split in (
        "train",
        "val",
    ):
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

    xs = make_starts(
        VIEW_W,
        COLS,
        OVERLAP,
    )

    ys = make_starts(
        VIEW_H,
        ROWS,
        OVERLAP,
    )

    counters = Counter()

    class_counts = {
        "train": Counter(),
        "val": Counter(),
    }

    annotation_paths = sorted(
        ANNOTATION_DIR.glob(
            "frame_*.json"
        )
    )

    for annotation_path in (
        annotation_paths
    ):
        data = json.loads(
            annotation_path.read_text()
        )

        frame = int(
            data["frame"]
        )

        split = split_for_frame(
            frame
        )

        if split is None:
            continue

        source = cv2.imread(
            str(
                find_image(
                    frame
                )
            )
        )

        if source is None:
            raise RuntimeError(
                f"Could not read frame {frame}"
            )

        view = cv2.resize(
            source,
            (
                VIEW_W,
                VIEW_H,
            ),
            interpolation=cv2.INTER_AREA,
        )

        tile_specs = []

        tile_index = 0

        for y1, y2 in ys:
            for x1, x2 in xs:
                tile_index += 1

                tile_specs.append(
                    {
                        "index": tile_index,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "center_x":
                            (x1 + x2) / 2.0,
                        "center_y":
                            (y1 + y2) / 2.0,
                        "objects": [],
                    }
                )

        objects = []

        for annotation in (
            data["annotations"]
        ):
            objects.append(
                {
                    "class":
                        annotation[
                            "object_id"
                        ],
                    "bbox":
                        source_bbox_to_l0(
                            annotation[
                                "bbox"
                            ]
                        ),
                }
            )

        # Every GT object must be assigned exactly once.
        for obj in objects:
            cx, cy = bbox_center(
                obj["bbox"]
            )

            candidates = [
                tile
                for tile in tile_specs
                if (
                    tile["x1"]
                    <= cx
                    < tile["x2"]
                    and tile["y1"]
                    <= cy
                    < tile["y2"]
                )
            ]

            if not candidates:
                raise RuntimeError(
                    f"No tile contains center "
                    f"frame={frame}, object={obj}"
                )

            selected = min(
                candidates,
                key=lambda tile: (
                    (cx - tile["center_x"]) ** 2
                    + (cy - tile["center_y"]) ** 2
                ),
            )

            selected[
                "objects"
            ].append(
                obj
            )

        assigned_count = sum(
            len(
                tile["objects"]
            )
            for tile in tile_specs
        )

        if assigned_count != len(objects):
            raise RuntimeError(
                f"Assignment mismatch frame {frame}: "
                f"{assigned_count} assigned vs "
                f"{len(objects)} GT"
            )

        for tile in tile_specs:
            x1 = tile["x1"]
            y1 = tile["y1"]
            x2 = tile["x2"]
            y2 = tile["y2"]

            crop = view[
                y1:y2,
                x1:x2,
            ]

            tile_h, tile_w = (
                crop.shape[:2]
            )

            labels = []

            for obj in tile["objects"]:
                clipped = clip_to_tile(
                    obj["bbox"],
                    tile,
                )

                if clipped is None:
                    raise RuntimeError(
                        f"Assigned GT does not "
                        f"intersect tile: {obj}"
                    )

                yolo_box = to_yolo(
                    clipped,
                    tile_w,
                    tile_h,
                )

                labels.append(
                    "0 "
                    + " ".join(
                        f"{value:.8f}"
                        for value
                        in yolo_box
                    )
                )

                counters[
                    f"{split}_boxes"
                ] += 1

                class_counts[
                    split
                ][
                    obj["class"]
                ] += 1

            basename = (
                f"frame_{frame:06d}"
                f"_tile_{tile['index']:02d}"
            )

            image_path = (
                OUTPUT_ROOT
                / "images"
                / split
                / f"{basename}.jpg"
            )

            label_path = (
                OUTPUT_ROOT
                / "labels"
                / split
                / f"{basename}.txt"
            )

            cv2.imwrite(
                str(image_path),
                crop,
                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    95,
                ],
            )

            label_path.write_text(
                (
                    "\n".join(
                        labels
                    )
                    + "\n"
                )
                if labels
                else "",
                encoding="utf-8",
            )

            counters[
                f"{split}_tiles"
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
    print("EXP-D010 TILED DATASET")
    print("=" * 72)

    print(
        "Train tiles:",
        counters["train_tiles"],
    )

    print(
        "Train boxes:",
        counters["train_boxes"],
    )

    print(
        "Val tiles:  ",
        counters["val_tiles"],
    )

    print(
        "Val boxes:  ",
        counters["val_boxes"],
    )

    print()
    print("Train class appearances:")

    for cls, count in sorted(
        class_counts["train"].items()
    ):
        print(
            f"  {cls:20s} {count}"
        )

    print()
    print("Val class appearances:")

    for cls, count in sorted(
        class_counts["val"].items()
    ):
        print(
            f"  {cls:20s} {count}"
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
