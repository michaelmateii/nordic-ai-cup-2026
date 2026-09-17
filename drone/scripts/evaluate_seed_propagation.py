#!/usr/bin/env python3

from __future__ import annotations

import json
from collections import defaultdict
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

SEEDS_PATH = (
    ROOT
    / "drone"
    / "artifacts"
    / "exp_d027"
    / "validation_seed_annotations.json"
)

OUT = (
    ROOT
    / "drone"
    / "artifacts"
    / "exp_d027b"
)

RESULTS_PATH = (
    OUT
    / "propagation_benchmark.json"
)

MONTAGE_PATH = (
    OUT
    / "propagation_montage.jpg"
)


SCALES = [
    0.70,
    0.80,
    0.90,
    1.00,
    1.10,
    1.20,
    1.30,
]

# Test whether local temporal propagation is easier
# than long-range re-identification.
GAP_LIMITS = [
    15,
    30,
    60,
    None,
]

IOU_SUCCESS = 0.50


def iou(
    a,
    b,
):
    x1 = max(
        float(a[0]),
        float(b[0]),
    )

    y1 = max(
        float(a[1]),
        float(b[1]),
    )

    x2 = min(
        float(a[2]),
        float(b[2]),
    )

    y2 = min(
        float(a[3]),
        float(b[3]),
    )

    inter = (
        max(0.0, x2 - x1)
        * max(0.0, y2 - y1)
    )

    area_a = (
        max(0.0, float(a[2] - a[0]))
        * max(0.0, float(a[3] - a[1]))
    )

    area_b = (
        max(0.0, float(b[2] - b[0]))
        * max(0.0, float(b[3] - b[1]))
    )

    union = (
        area_a
        + area_b
        - inter
    )

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
            f"No image for frame_index={frame_index}"
        )

    return matches[0]


def load_seeds():
    raw = json.loads(
        SEEDS_PATH.read_text()
    )

    seeds = []

    seed_id = 0

    for frame_key, rows in raw.items():
        for row in rows:
            seeds.append(
                {
                    "seed_id":
                        seed_id,
                    "class":
                        row["class"],
                    "frame":
                        int(row["frame"]),
                    "frame_index":
                        int(
                            row["frame_index"]
                        ),
                    "bbox":
                        [
                            int(v)
                            for v in row["bbox"]
                        ],
                    "image_path":
                        str(
                            frame_image_path(
                                int(
                                    row[
                                        "frame_index"
                                    ]
                                )
                            )
                        ),
                }
            )

            seed_id += 1

    seeds.sort(
        key=lambda r:
            (
                r["class"],
                r["frame_index"],
            )
    )

    return seeds


def extract_template(
    image,
    bbox,
):
    x1, y1, x2, y2 = (
        map(int, bbox)
    )

    x1 = max(
        0,
        x1,
    )

    y1 = max(
        0,
        y1,
    )

    x2 = min(
        image.shape[1],
        x2,
    )

    y2 = min(
        image.shape[0],
        y2,
    )

    if (
        x2 <= x1
        or y2 <= y1
    ):
        return None

    return image[
        y1:y2,
        x1:x2,
    ].copy()


def preprocess_gray(
    image,
):
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0,
    )

    return gray


def match_template(
    template_bgr,
    target_bgr,
):
    target_gray = (
        preprocess_gray(
            target_bgr
        )
    )

    base_gray = (
        preprocess_gray(
            template_bgr
        )
    )

    best = None

    for scale in SCALES:
        width = max(
            3,
            int(
                round(
                    base_gray.shape[1]
                    * scale
                )
            ),
        )

        height = max(
            3,
            int(
                round(
                    base_gray.shape[0]
                    * scale
                )
            ),
        )

        if (
            width >= target_gray.shape[1]
            or height >= target_gray.shape[0]
        ):
            continue

        resized = cv2.resize(
            base_gray,
            (
                width,
                height,
            ),
            interpolation=(
                cv2.INTER_CUBIC
                if scale > 1.0
                else cv2.INTER_AREA
            ),
        )

        response = cv2.matchTemplate(
            target_gray,
            resized,
            cv2.TM_CCOEFF_NORMED,
        )

        _, score, _, location = (
            cv2.minMaxLoc(
                response
            )
        )

        candidate = {
            "score":
                float(score),
            "scale":
                float(scale),
            "bbox":
                [
                    int(location[0]),
                    int(location[1]),
                    int(
                        location[0]
                        + width
                    ),
                    int(
                        location[1]
                        + height
                    ),
                ],
        }

        if (
            best is None
            or candidate["score"]
            > best["score"]
        ):
            best = candidate

    return best


def evaluate_pair(
    query,
    target,
):
    query_image = cv2.imread(
        query["image_path"]
    )

    target_image = cv2.imread(
        target["image_path"]
    )

    if (
        query_image is None
        or target_image is None
    ):
        return None

    template = extract_template(
        query_image,
        query["bbox"],
    )

    if template is None:
        return None

    match = match_template(
        template,
        target_image,
    )

    if match is None:
        return None

    overlap = iou(
        match["bbox"],
        target["bbox"],
    )

    return {
        "query_seed_id":
            query["seed_id"],
        "target_seed_id":
            target["seed_id"],
        "class":
            query["class"],
        "query_frame_index":
            query["frame_index"],
        "target_frame_index":
            target["frame_index"],
        "temporal_gap":
            abs(
                query["frame_index"]
                - target["frame_index"]
            ),
        "score":
            match["score"],
        "scale":
            match["scale"],
        "pred_bbox":
            match["bbox"],
        "gt_bbox":
            target["bbox"],
        "iou":
            float(overlap),
        "success":
            bool(
                overlap
                >= IOU_SUCCESS
            ),
        "target_image_path":
            target["image_path"],
    }


def summarize(
    rows,
    max_gap,
):
    selected = []

    for row in rows:
        if (
            max_gap is None
            or row["temporal_gap"]
            <= max_gap
        ):
            selected.append(
                row
            )

    if not selected:
        return {
            "pairs": 0,
        }

    ious = np.asarray(
        [
            r["iou"]
            for r in selected
        ],
        dtype=float,
    )

    scores = np.asarray(
        [
            r["score"]
            for r in selected
        ],
        dtype=float,
    )

    success = np.asarray(
        [
            r["success"]
            for r in selected
        ],
        dtype=bool,
    )

    return {
        "pairs":
            int(
                len(selected)
            ),
        "recall_iou50":
            float(
                success.mean()
            ),
        "mean_iou":
            float(
                ious.mean()
            ),
        "median_iou":
            float(
                np.median(
                    ious
                )
            ),
        "mean_match_score":
            float(
                scores.mean()
            ),
        "median_match_score":
            float(
                np.median(
                    scores
                )
            ),
    }


def class_summary(
    rows,
):
    grouped = defaultdict(list)

    for row in rows:
        grouped[
            row["class"]
        ].append(
            row
        )

    output = {}

    for class_name, class_rows in sorted(
        grouped.items()
    ):
        output[
            class_name
        ] = {}

        for gap in GAP_LIMITS:
            label = (
                "all"
                if gap is None
                else f"gap_le_{gap}"
            )

            output[
                class_name
            ][label] = summarize(
                class_rows,
                gap,
            )

    return output


def make_montage(
    rows,
):
    # Show the most informative examples:
    # successful high-IoU pairs and failures.
    successes = sorted(
        [
            r
            for r in rows
            if r["success"]
        ],
        key=lambda r:
            r["iou"],
        reverse=True,
    )[:20]

    failures = sorted(
        [
            r
            for r in rows
            if not r["success"]
        ],
        key=lambda r:
            r["score"],
        reverse=True,
    )[:20]

    selected = (
        successes
        + failures
    )

    if not selected:
        return

    cell_w = 300
    cell_h = 230
    cols = 4

    cards = []

    for row in selected:
        image = cv2.imread(
            row[
                "target_image_path"
            ]
        )

        if image is None:
            continue

        px1, py1, px2, py2 = (
            row["pred_bbox"]
        )

        gx1, gy1, gx2, gy2 = (
            row["gt_bbox"]
        )

        x1 = min(
            px1,
            gx1,
        )

        y1 = min(
            py1,
            gy1,
        )

        x2 = max(
            px2,
            gx2,
        )

        y2 = max(
            py2,
            gy2,
        )

        width = max(
            1,
            x2 - x1,
        )

        height = max(
            1,
            y2 - y1,
        )

        pad_x = max(
            30,
            int(
                round(
                    width
                    * 1.5
                )
            ),
        )

        pad_y = max(
            30,
            int(
                round(
                    height
                    * 1.5
                )
            ),
        )

        cx1 = max(
            0,
            x1 - pad_x,
        )

        cy1 = max(
            0,
            y1 - pad_y,
        )

        cx2 = min(
            image.shape[1],
            x2 + pad_x,
        )

        cy2 = min(
            image.shape[0],
            y2 + pad_y,
        )

        crop = image[
            cy1:cy2,
            cx1:cx2,
        ].copy()

        # Green = manual GT.
        cv2.rectangle(
            crop,
            (
                gx1 - cx1,
                gy1 - cy1,
            ),
            (
                gx2 - cx1,
                gy2 - cy1,
            ),
            (0, 255, 0),
            2,
        )

        # Red = propagated prediction.
        cv2.rectangle(
            crop,
            (
                px1 - cx1,
                py1 - cy1,
            ),
            (
                px2 - cx1,
                py2 - cy1,
            ),
            (0, 0, 255),
            2,
        )

        usable_w = (
            cell_w - 10
        )

        usable_h = (
            cell_h - 42
        )

        scale = min(
            usable_w
            / crop.shape[1],
            usable_h
            / crop.shape[0],
        )

        scale = min(
            scale,
            5.0,
        )

        crop = cv2.resize(
            crop,
            (
                max(
                    1,
                    int(
                        crop.shape[1]
                        * scale
                    ),
                ),
                max(
                    1,
                    int(
                        crop.shape[0]
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

        ox = (
            cell_w
            - crop.shape[1]
        ) // 2

        oy = 38

        card[
            oy:oy + crop.shape[0],
            ox:ox + crop.shape[1],
        ] = crop

        text = (
            f"{row['class']} "
            f"gap={row['temporal_gap']} "
            f"IoU={row['iou']:.2f} "
            f"NCC={row['score']:.2f}"
        )

        cv2.putText(
            card,
            text,
            (5, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.37,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        cards.append(
            card
        )

    montage_rows = []

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

        montage_rows.append(
            np.hstack(
                row
            )
        )

    montage = np.vstack(
        montage_rows
    )

    cv2.imwrite(
        str(
            MONTAGE_PATH
        ),
        montage,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            94,
        ],
    )


def main():
    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    seeds = load_seeds()

    print(
        "Manual seeds:",
        len(seeds),
    )

    by_class = defaultdict(list)

    for seed in seeds:
        by_class[
            seed["class"]
        ].append(
            seed
        )

    print()
    print(
        "Classes with >=2 seeds:"
    )

    for class_name, rows in sorted(
        by_class.items()
    ):
        if len(rows) >= 2:
            print(
                f"  {class_name:20s} "
                f"{len(rows)}"
            )

    results = []

    for class_name, rows in sorted(
        by_class.items()
    ):
        if len(rows) < 2:
            continue

        for query in rows:
            for target in rows:
                if (
                    query["seed_id"]
                    == target["seed_id"]
                ):
                    continue

                result = evaluate_pair(
                    query,
                    target,
                )

                if result is not None:
                    results.append(
                        result
                    )

    overall = {}

    for gap in GAP_LIMITS:
        label = (
            "all"
            if gap is None
            else f"gap_le_{gap}"
        )

        overall[
            label
        ] = summarize(
            results,
            gap,
        )

    output = {
        "manual_seed_count":
            len(seeds),
        "evaluated_pairs":
            len(results),
        "iou_success_threshold":
            IOU_SUCCESS,
        "scales":
            SCALES,
        "overall":
            overall,
        "by_class":
            class_summary(
                results
            ),
        "pairs":
            results,
    }

    RESULTS_PATH.write_text(
        json.dumps(
            output,
            indent=2,
        )
    )

    make_montage(
        results
    )

    print()
    print("=" * 80)
    print(
        "EXP-D027B PROPAGATION BENCHMARK"
    )
    print("=" * 80)

    for label in [
        "gap_le_15",
        "gap_le_30",
        "gap_le_60",
        "all",
    ]:
        row = overall[
            label
        ]

        if row["pairs"] == 0:
            print(
                f"{label:12s}: "
                f"no pairs"
            )
            continue

        print(
            f"{label:12s}: "
            f"pairs={row['pairs']:3d} "
            f"R@IoU.50="
            f"{row['recall_iou50']:.3f} "
            f"medianIoU="
            f"{row['median_iou']:.3f} "
            f"medianNCC="
            f"{row['median_match_score']:.3f}"
        )

    print()
    print("PER CLASS — ALL PAIRS")
    print("-" * 80)

    summaries = (
        class_summary(
            results
        )
    )

    for class_name in sorted(
        summaries
    ):
        row = (
            summaries[
                class_name
            ]["all"]
        )

        print(
            f"{class_name:20s} "
            f"pairs={row['pairs']:3d} "
            f"R@.50="
            f"{row['recall_iou50']:.3f} "
            f"medIoU="
            f"{row['median_iou']:.3f}"
        )

    print()
    print(
        "Results:",
        RESULTS_PATH,
    )

    print(
        "Montage:",
        MONTAGE_PATH,
    )


if __name__ == "__main__":
    main()
