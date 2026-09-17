#!/usr/bin/env python3

from __future__ import annotations

import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]

OFFICIAL = (
    ROOT
    / "drone"
    / "reference"
    / "official-drone-flyby"
)

ANNOTATION_DIR = (
    OFFICIAL
    / "src"
    / "helsinki"
    / "annotations"
)

OUTPUT_DIR = ROOT / "drone" / "artifacts"
OUTPUT_JSON = OUTPUT_DIR / "dataset_report.json"


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def load_annotation(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    annotation_files = sorted(
        ANNOTATION_DIR.glob("frame_*.json")
    )

    print(f"Official root:      {OFFICIAL}")
    print(f"Annotation dir:     {ANNOTATION_DIR}")
    print(f"Annotation frames:  {len(annotation_files)}")
    print()

    if not annotation_files:
        raise RuntimeError(
            f"No frame annotations found in {ANNOTATION_DIR}"
        )

    class_counts: Counter[str] = Counter()

    widths: defaultdict[str, list[float]] = defaultdict(list)
    heights: defaultdict[str, list[float]] = defaultdict(list)
    areas: defaultdict[str, list[float]] = defaultdict(list)

    # Each class corresponds to one physical object identity
    # in the supplied Helsinki sequence.
    centers: defaultdict[
        str,
        list[tuple[int, float, float]]
    ] = defaultdict(list)

    parsed_frames: list[dict[str, Any]] = []

    total_boxes = 0

    for path in annotation_files:
        data = load_annotation(path)

        frame = int(data["frame"])
        annotations = data["annotations"]

        normalized_objects = []

        for annotation in annotations:
            cls = str(annotation["object_id"])

            bbox = annotation["bbox"]

            if len(bbox) != 4:
                raise ValueError(
                    f"Invalid bbox in {path}: {bbox}"
                )

            # Official Drone Flyby annotation format is xyxy.
            x1, y1, x2, y2 = map(float, bbox)

            width = x2 - x1
            height = y2 - y1

            if width <= 0 or height <= 0:
                raise ValueError(
                    f"Invalid bbox dimensions in {path}: {bbox}"
                )

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            class_counts[cls] += 1

            widths[cls].append(width)
            heights[cls].append(height)
            areas[cls].append(width * height)

            centers[cls].append(
                (frame, cx, cy)
            )

            total_boxes += 1

            normalized_objects.append(
                {
                    "class": cls,
                    "bbox_xyxy": [
                        x1,
                        y1,
                        x2,
                        y2,
                    ],
                    "width": width,
                    "height": height,
                    "center": [
                        cx,
                        cy,
                    ],
                }
            )

        parsed_frames.append(
            {
                "frame": frame,
                "file": path.name,
                "objects": normalized_objects,
            }
        )

    # ---------------------------------------------------------
    # Temporal motion
    # ---------------------------------------------------------

    motion_by_class: dict[str, dict[str, Any]] = {}

    pooled_dx: list[float] = []
    pooled_dy: list[float] = []
    pooled_distance: list[float] = []

    for cls, observations in centers.items():

        observations = sorted(
            observations,
            key=lambda row: row[0],
        )

        dxs: list[float] = []
        dys: list[float] = []
        distances: list[float] = []

        for previous, current in zip(
            observations,
            observations[1:],
        ):

            frame_a, x_a, y_a = previous
            frame_b, x_b, y_b = current

            # Compare only genuinely adjacent source frames.
            if frame_b - frame_a != 1:
                continue

            dx = x_b - x_a
            dy = y_b - y_a

            distance = math.hypot(dx, dy)

            dxs.append(dx)
            dys.append(dy)
            distances.append(distance)

            pooled_dx.append(dx)
            pooled_dy.append(dy)
            pooled_distance.append(distance)

        motion_by_class[cls] = {
            "consecutive_comparisons": len(dxs),
            "median_dx": median(dxs),
            "median_dy": median(dys),
            "median_distance": median(distances),
        }

    # ---------------------------------------------------------
    # Class report
    # ---------------------------------------------------------

    class_names = sorted(class_counts)

    class_report = {}

    for cls in class_names:

        median_width = median(widths[cls])
        median_height = median(heights[cls])

        class_report[cls] = {
            "appearances": class_counts[cls],

            "median_source_width": median_width,
            "median_source_height": median_height,
            "median_source_area": median(
                areas[cls]
            ),

            # Camera level 0:
            # 3840x2160 source -> 960x540 image.
            "median_level0_width": (
                median_width / 4
                if median_width is not None
                else None
            ),
            "median_level0_height": (
                median_height / 4
                if median_height is not None
                else None
            ),

            # Camera level 1:
            # 1920x1080 source region -> 960x540.
            "median_level1_width": (
                median_width / 2
                if median_width is not None
                else None
            ),
            "median_level1_height": (
                median_height / 2
                if median_height is not None
                else None
            ),

            # Camera level 2:
            # 960x540 source region -> 960x540.
            "median_level2_width": median_width,
            "median_level2_height": median_height,

            "motion": motion_by_class[cls],
        }

    report = {
        "annotation_frames": len(
            annotation_files
        ),
        "parsed_frames": len(
            parsed_frames
        ),
        "total_boxes": total_boxes,
        "classes_found": len(
            class_names
        ),
        "class_names": class_names,

        "global_motion": {
            "consecutive_pairs": len(
                pooled_dx
            ),
            "median_dx": median(
                pooled_dx
            ),
            "median_dy": median(
                pooled_dy
            ),
            "median_distance": median(
                pooled_distance
            ),
        },

        "class_report": class_report,
        "frames": parsed_frames,
    }

    with OUTPUT_JSON.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            report,
            f,
            indent=2,
        )

    # ---------------------------------------------------------
    # Console summary
    # ---------------------------------------------------------

    print("=" * 100)
    print("DATASET SUMMARY")
    print("=" * 100)

    print(
        f"Annotated frames: {len(annotation_files)}"
    )
    print(
        f"Total boxes:      {total_boxes}"
    )
    print(
        f"Classes found:    {len(class_names)}"
    )

    print()

    header = (
        f"{'class':20s} "
        f"{'n':>4s} "
        f"{'source median':>15s} "
        f"{'L0':>13s} "
        f"{'L1':>13s} "
        f"{'L2':>13s}"
    )

    print(header)
    print("-" * 100)

    for cls in class_names:

        r = class_report[cls]

        sw = r["median_source_width"]
        sh = r["median_source_height"]

        print(
            f"{cls:20s} "
            f"{r['appearances']:4d} "
            f"{sw:6.1f}x{sh:<6.1f} "
            f"{sw/4:5.1f}x{sh/4:<5.1f} "
            f"{sw/2:5.1f}x{sh/2:<5.1f} "
            f"{sw:5.1f}x{sh:<5.1f}"
        )

    print()
    print("Global consecutive-frame motion")
    print("--------------------------------")

    print(
        "Comparisons:",
        report["global_motion"][
            "consecutive_pairs"
        ],
    )

    print(
        "Median dx:",
        report["global_motion"][
            "median_dx"
        ],
    )

    print(
        "Median dy:",
        report["global_motion"][
            "median_dy"
        ],
    )

    print(
        "Median distance:",
        report["global_motion"][
            "median_distance"
        ],
    )

    print()
    print(
        f"Saved report: {OUTPUT_JSON}"
    )


if __name__ == "__main__":
    main()