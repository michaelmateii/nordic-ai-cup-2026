#!/usr/bin/env python3

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]

OFFICIAL = (
    ROOT
    / "drone"
    / "reference"
    / "official-drone-flyby"
)

HELSINKI_IMAGES = (
    OFFICIAL
    / "src"
    / "helsinki"
    / "images"
)

HELSINKI_ANN = (
    OFFICIAL
    / "src"
    / "helsinki"
    / "annotations"
)

VALIDATION = (
    ROOT
    / "drone"
    / "captures"
    / "3224a582bfbf4273a028497662b7aa7c"
    / "frames"
)

OUT = (
    ROOT
    / "drone"
    / "artifacts"
    / "exp_d025"
)

TEMPLATES_DIR = OUT / "templates"

# L1 maps source pixels to transmitted pixels at 0.5x.
BASE_L1_SCALE = 0.50

# Cheap robustness screen around expected scale.
RELATIVE_SCALES = [
    0.75,
    0.90,
    1.00,
    1.10,
    1.25,
]

TOP_K_PER_CLASS = 8

# Add a little surrounding appearance, but not enough to let
# Helsinki background dominate the match.
CONTEXT = 0.10


def find_image(frame_number: int) -> Path:
    candidates = [
        HELSINKI_IMAGES
        / f"frame_{frame_number:06d}.png",

        HELSINKI_IMAGES
        / f"frame_{frame_number:06d}.jpg",

        HELSINKI_IMAGES
        / f"frame_{frame_number:06d}.jpeg",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"No image found for Helsinki frame {frame_number}"
    )


def crop_box_with_context(
    image: np.ndarray,
    bbox,
):
    x1, y1, x2, y2 = map(
        float,
        bbox,
    )

    w = x2 - x1
    h = y2 - y1

    px = w * CONTEXT
    py = h * CONTEXT

    ih, iw = image.shape[:2]

    x1 = max(
        0,
        int(round(x1 - px)),
    )

    y1 = max(
        0,
        int(round(y1 - py)),
    )

    x2 = min(
        iw,
        int(round(x2 + px)),
    )

    y2 = min(
        ih,
        int(round(y2 + py)),
    )

    if x2 <= x1 or y2 <= y1:
        return None

    return image[
        y1:y2,
        x1:x2,
    ]


def collect_annotations():
    by_class = defaultdict(list)

    for ann_path in sorted(
        HELSINKI_ANN.glob("*.json")
    ):
        data = json.loads(
            ann_path.read_text()
        )

        frame = int(
            data["frame"]
        )

        for ann in data[
            "annotations"
        ]:
            bbox = ann["bbox"]

            width = (
                bbox[2]
                - bbox[0]
            )

            height = (
                bbox[3]
                - bbox[1]
            )

            area = (
                width
                * height
            )

            by_class[
                ann["object_id"]
            ].append(
                {
                    "frame":
                        frame,
                    "bbox":
                        bbox,
                    "area":
                        area,
                    "width":
                        width,
                    "height":
                        height,
                }
            )

    return by_class


def choose_median_template(
    samples,
):
    ordered = sorted(
        samples,
        key=lambda x:
            x["area"],
    )

    return ordered[
        len(ordered) // 2
    ]


def build_templates():
    TEMPLATES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    annotations = (
        collect_annotations()
    )

    templates = {}

    print("Building templates...")

    for class_name in sorted(
        annotations
    ):
        sample = (
            choose_median_template(
                annotations[
                    class_name
                ]
            )
        )

        image_path = find_image(
            sample["frame"]
        )

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            raise RuntimeError(
                f"Could not read {image_path}"
            )

        crop = crop_box_with_context(
            image,
            sample["bbox"],
        )

        if crop is None:
            continue

        # Nominal conversion from source resolution to L1 received view.
        l1_width = max(
            3,
            int(
                round(
                    crop.shape[1]
                    * BASE_L1_SCALE
                )
            ),
        )

        l1_height = max(
            3,
            int(
                round(
                    crop.shape[0]
                    * BASE_L1_SCALE
                )
            ),
        )

        nominal = cv2.resize(
            crop,
            (
                l1_width,
                l1_height,
            ),
            interpolation=
                cv2.INTER_AREA,
        )

        template_path = (
            TEMPLATES_DIR
            / f"{class_name}.png"
        )

        cv2.imwrite(
            str(template_path),
            nominal,
        )

        gray = cv2.cvtColor(
            nominal,
            cv2.COLOR_BGR2GRAY,
        )

        templates[
            class_name
        ] = {
            "gray":
                gray,
            "source_frame":
                sample["frame"],
            "source_bbox":
                sample["bbox"],
            "nominal_size":
                [
                    l1_width,
                    l1_height,
                ],
        }

        print(
            f"  {class_name:20s} "
            f"source={sample['width']}x{sample['height']} "
            f"L1≈{l1_width}x{l1_height}"
        )

    return templates


def load_validation_frames():
    records = []

    for meta_path in sorted(
        VALIDATION.glob("*.json")
    ):
        data = json.loads(
            meta_path.read_text()
        )

        if (
            data["view"][
                "resolution_level"
            ]
            != 1
        ):
            continue

        image_path = (
            meta_path.with_suffix(
                ".png"
            )
        )

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            continue

        records.append(
            {
                "frame":
                    int(data["frame"]),
                "frame_index":
                    int(
                        data[
                            "frame_index"
                        ]
                    ),
                "center_x":
                    int(
                        data["view"][
                            "center_x"
                        ]
                    ),
                "center_y":
                    int(
                        data["view"][
                            "center_y"
                        ]
                    ),
                "image":
                    image,
                "path":
                    image_path,
            }
        )

    return records


def keep_top(
    results,
    candidate,
):
    results.append(
        candidate
    )

    results.sort(
        key=lambda x:
            x["score"],
        reverse=True,
    )

    del results[
        TOP_K_PER_CLASS:
    ]


def scan_template(
    frame_gray,
    base_template,
):
    best = None

    for relative_scale in (
        RELATIVE_SCALES
    ):
        tw = max(
            3,
            int(
                round(
                    base_template.shape[1]
                    * relative_scale
                )
            ),
        )

        th = max(
            3,
            int(
                round(
                    base_template.shape[0]
                    * relative_scale
                )
            ),
        )

        if (
            tw >= frame_gray.shape[1]
            or th >= frame_gray.shape[0]
        ):
            continue

        template = cv2.resize(
            base_template,
            (
                tw,
                th,
            ),
            interpolation=(
                cv2.INTER_CUBIC
                if relative_scale > 1
                else cv2.INTER_AREA
            ),
        )

        response = (
            cv2.matchTemplate(
                frame_gray,
                template,
                cv2.TM_CCOEFF_NORMED,
            )
        )

        _, max_score, _, max_loc = (
            cv2.minMaxLoc(
                response
            )
        )

        candidate = {
            "score":
                float(max_score),
            "relative_scale":
                float(relative_scale),
            "x1":
                int(max_loc[0]),
            "y1":
                int(max_loc[1]),
            "x2":
                int(
                    max_loc[0]
                    + tw
                ),
            "y2":
                int(
                    max_loc[1]
                    + th
                ),
        }

        if (
            best is None
            or candidate["score"]
            > best["score"]
        ):
            best = candidate

    return best


def save_match_crop(
    class_name,
    rank,
    result,
):
    image = cv2.imread(
        result["image_path"]
    )

    x1 = result["x1"]
    y1 = result["y1"]
    x2 = result["x2"]
    y2 = result["y2"]

    w = x2 - x1
    h = y2 - y1

    pad_x = max(
        15,
        int(round(w * 1.5)),
    )

    pad_y = max(
        15,
        int(round(h * 1.5)),
    )

    ih, iw = image.shape[:2]

    cx1 = max(
        0,
        x1 - pad_x,
    )

    cy1 = max(
        0,
        y1 - pad_y,
    )

    cx2 = min(
        iw,
        x2 + pad_x,
    )

    cy2 = min(
        ih,
        y2 + pad_y,
    )

    crop = image[
        cy1:cy2,
        cx1:cx2,
    ].copy()

    cv2.rectangle(
        crop,
        (
            x1 - cx1,
            y1 - cy1,
        ),
        (
            x2 - cx1,
            y2 - cy1,
        ),
        (0, 0, 255),
        2,
    )

    cv2.putText(
        crop,
        (
            f"{class_name} "
            f"{result['score']:.3f}"
        ),
        (5, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    path = (
        OUT
        / "matches"
        / (
            f"{class_name}"
            f"_rank{rank:02d}"
            f"_frame{result['frame_index']:03d}.png"
        )
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cv2.imwrite(
        str(path),
        crop,
    )

    return path


def build_montage(
    ranked_results,
):
    cards = []

    for class_name in sorted(
        ranked_results
    ):
        results = ranked_results[
            class_name
        ]

        for rank, result in enumerate(
            results[:3],
            start=1,
        ):
            path = save_match_crop(
                class_name,
                rank,
                result,
            )

            image = cv2.imread(
                str(path)
            )

            if image is None:
                continue

            cell_w = 300
            cell_h = 210

            scale = min(
                (cell_w - 10)
                / image.shape[1],
                (cell_h - 30)
                / image.shape[0],
            )

            scale = min(
                scale,
                5.0,
            )

            image = cv2.resize(
                image,
                (
                    max(
                        1,
                        int(
                            image.shape[1]
                            * scale
                        ),
                    ),
                    max(
                        1,
                        int(
                            image.shape[0]
                            * scale
                        ),
                    ),
                ),
                interpolation=(
                    cv2.INTER_NEAREST
                    if scale > 1
                    else cv2.INTER_AREA
                ),
            )

            card = np.zeros(
                (
                    cell_h,
                    cell_w,
                    3,
                ),
                dtype=np.uint8,
            )

            x = (
                cell_w
                - image.shape[1]
            ) // 2

            y = 28

            card[
                y:y + image.shape[0],
                x:x + image.shape[1],
            ] = image

            label = (
                f"{class_name} "
                f"#{rank} "
                f"{result['score']:.3f}"
            )

            cv2.putText(
                card,
                label,
                (5, 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            cards.append(
                card
            )

    cols = 3

    rows = []

    for start in range(
        0,
        len(cards),
        cols,
    ):
        row = cards[
            start:start + cols
        ]

        while len(row) < cols:
            row.append(
                np.zeros_like(
                    cards[0]
                )
            )

        rows.append(
            np.hstack(
                row
            )
        )

    montage = np.vstack(
        rows
    )

    path = (
        OUT
        / "template_match_montage.jpg"
    )

    cv2.imwrite(
        str(path),
        montage,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            94,
        ],
    )

    return path


def main():
    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    templates = (
        build_templates()
    )

    frames = (
        load_validation_frames()
    )

    print()
    print(
        "Validation L1 frames:",
        len(frames),
    )

    ranked_results = {
        name: []
        for name in templates
    }

    for i, frame in enumerate(
        frames,
        start=1,
    ):
        gray = cv2.cvtColor(
            frame["image"],
            cv2.COLOR_BGR2GRAY,
        )

        for class_name, template_info in (
            templates.items()
        ):
            match = scan_template(
                gray,
                template_info[
                    "gray"
                ],
            )

            if match is None:
                continue

            match.update(
                {
                    "class":
                        class_name,
                    "frame":
                        frame["frame"],
                    "frame_index":
                        frame[
                            "frame_index"
                        ],
                    "center_x":
                        frame["center_x"],
                    "center_y":
                        frame["center_y"],
                    "image_path":
                        str(
                            frame["path"]
                        ),
                }
            )

            keep_top(
                ranked_results[
                    class_name
                ],
                match,
            )

        if (
            i % 25 == 0
            or i == len(frames)
        ):
            print(
                f"{i:3d}/{len(frames)}"
            )

    output = {
        "base_l1_scale":
            BASE_L1_SCALE,
        "relative_scales":
            RELATIVE_SCALES,
        "results":
            ranked_results,
    }

    results_path = (
        OUT
        / "template_results.json"
    )

    results_path.write_text(
        json.dumps(
            output,
            indent=2,
        )
    )

    montage = build_montage(
        ranked_results
    )

    print()
    print("=" * 80)
    print("EXP-D025 TEMPLATE SCREEN")
    print("=" * 80)

    for class_name in sorted(
        ranked_results
    ):
        scores = [
            x["score"]
            for x in ranked_results[
                class_name
            ]
        ]

        print(
            f"{class_name:20s} "
            f"best={scores[0]:.4f} "
            f"top3="
            f"{np.mean(scores[:3]):.4f} "
            f"top8="
            f"{np.mean(scores):.4f}"
        )

    print()
    print(
        "Results:",
        results_path,
    )

    print(
        "Montage:",
        montage,
    )


if __name__ == "__main__":
    main()
