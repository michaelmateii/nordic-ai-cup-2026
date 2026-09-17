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
    / "exp_d027d"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

RESULTS_PATH = (
    OUT
    / "cycle_pseudolabels.json"
)

MONTAGE_PATH = (
    OUT
    / "cycle_pseudolabel_montage.jpg"
)


MAX_GAP = 10
MIN_NCC = 0.90
MIN_CYCLE_IOU = 0.50

SCALES = [
    0.80,
    0.90,
    1.00,
    1.10,
    1.20,
]

TRUSTED_CLASSES = {
    "tank",
    "large_tower",
}


def iou(a, b):
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


def load_frame_records():
    records = []

    for meta_path in sorted(
        CAPTURE.glob("*.json")
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

        if not image_path.exists():
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

                "source_region_xyxy":
                    data["view"][
                        "source_region_xyxy"
                    ],

                "image_path":
                    str(image_path),
            }
        )

    records.sort(
        key=lambda r:
            r["frame_index"]
    )

    return records


def load_seeds(
    frame_records,
):
    raw = json.loads(
        SEEDS_PATH.read_text()
    )

    by_index = {
        r["frame_index"]: r
        for r in frame_records
    }

    seeds = []

    seed_id = 0

    for rows in raw.values():
        for row in rows:
            frame_index = int(
                row["frame_index"]
            )

            frame_record = (
                by_index[
                    frame_index
                ]
            )

            seeds.append(
                {
                    "seed_id":
                        seed_id,

                    "class":
                        row["class"],

                    "frame":
                        int(
                            row["frame"]
                        ),

                    "frame_index":
                        frame_index,

                    "bbox":
                        [
                            int(v)
                            for v in row["bbox"]
                        ],

                    "center_x":
                        frame_record[
                            "center_x"
                        ],

                    "center_y":
                        frame_record[
                            "center_y"
                        ],

                    "image_path":
                        frame_record[
                            "image_path"
                        ],
                }
            )

            seed_id += 1

    return seeds


def crop_box(
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


def gray(image):
    g = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    return cv2.GaussianBlur(
        g,
        (3, 3),
        0,
    )


def match_template(
    template_bgr,
    target_bgr,
):
    template_base = (
        gray(
            template_bgr
        )
    )

    target_gray = (
        gray(
            target_bgr
        )
    )

    best = None

    for scale in SCALES:
        width = max(
            3,
            int(
                round(
                    template_base.shape[1]
                    * scale
                )
            ),
        )

        height = max(
            3,
            int(
                round(
                    template_base.shape[0]
                    * scale
                )
            ),
        )

        if (
            width >= target_gray.shape[1]
            or height >= target_gray.shape[0]
        ):
            continue

        template = cv2.resize(
            template_base,
            (
                width,
                height,
            ),
            interpolation=(
                cv2.INTER_CUBIC
                if scale > 1
                else cv2.INTER_AREA
            ),
        )

        response = cv2.matchTemplate(
            target_gray,
            template,
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


def candidate_frames(
    seed,
    frame_records,
):
    output = []

    for frame in frame_records:
        if (
            frame["frame_index"]
            == seed["frame_index"]
        ):
            continue

        gap = abs(
            frame["frame_index"]
            - seed["frame_index"]
        )

        if gap > MAX_GAP:
            continue

        # Same actual camera region only.
        if (
            frame["center_x"]
            != seed["center_x"]
            or frame["center_y"]
            != seed["center_y"]
        ):
            continue

        output.append(
            frame
        )

    return output


def propagate_one(
    seed,
    target,
):
    seed_image = cv2.imread(
        seed["image_path"]
    )

    target_image = cv2.imread(
        target["image_path"]
    )

    if (
        seed_image is None
        or target_image is None
    ):
        return None

    seed_template = crop_box(
        seed_image,
        seed["bbox"],
    )

    if seed_template is None:
        return None

    forward = match_template(
        seed_template,
        target_image,
    )

    if (
        forward is None
        or forward["score"]
        < MIN_NCC
    ):
        return None

    target_template = crop_box(
        target_image,
        forward["bbox"],
    )

    if target_template is None:
        return None

    backward = match_template(
        target_template,
        seed_image,
    )

    if backward is None:
        return None

    cycle_iou = iou(
        backward["bbox"],
        seed["bbox"],
    )

    accepted = (
        cycle_iou
        >= MIN_CYCLE_IOU
    )

    return {
        "seed_id":
            seed["seed_id"],

        "class":
            seed["class"],

        "trusted_class":
            seed["class"]
            in TRUSTED_CLASSES,

        "seed_frame_index":
            seed["frame_index"],

        "target_frame_index":
            target["frame_index"],

        "temporal_gap":
            abs(
                target["frame_index"]
                - seed["frame_index"]
            ),

        "bbox":
            forward["bbox"],

        "forward_ncc":
            float(
                forward["score"]
            ),

        "backward_ncc":
            float(
                backward["score"]
            ),

        "cycle_iou":
            float(
                cycle_iou
            ),

        "accepted":
            bool(
                accepted
            ),

        "image_path":
            target["image_path"],

        "center_x":
            target["center_x"],

        "center_y":
            target["center_y"],

        "source_region_xyxy":
            target[
                "source_region_xyxy"
            ],
    }


def deduplicate(
    rows,
):
    grouped = defaultdict(
        list
    )

    for row in rows:
        if not row[
            "accepted"
        ]:
            continue

        grouped[
            (
                row["target_frame_index"],
                row["class"],
            )
        ].append(
            row
        )

    output = []

    for key, candidates in grouped.items():
        candidates.sort(
            key=lambda r:
                (
                    r["cycle_iou"],
                    r["forward_ncc"],
                ),
            reverse=True,
        )

        kept = []

        for candidate in candidates:
            duplicate = False

            for existing in kept:
                if (
                    iou(
                        candidate["bbox"],
                        existing["bbox"],
                    )
                    >= 0.50
                ):
                    duplicate = True
                    break

            if not duplicate:
                kept.append(
                    candidate
                )

        output.extend(
            kept
        )

    output.sort(
        key=lambda r:
            (
                r["target_frame_index"],
                r["class"],
            )
    )

    return output


def make_montage(
    rows,
):
    rows = sorted(
        rows,
        key=lambda r:
            (
                not r[
                    "trusted_class"
                ],
                -r[
                    "cycle_iou"
                ],
                -r[
                    "forward_ncc"
                ],
            )
    )

    cell_w = 280
    cell_h = 220
    cols = 4

    cards = []

    for row in rows[:80]:
        image = cv2.imread(
            row["image_path"]
        )

        if image is None:
            continue

        x1, y1, x2, y2 = (
            row["bbox"]
        )

        w = max(
            1,
            x2 - x1,
        )

        h = max(
            1,
            y2 - y1,
        )

        pad_x = max(
            35,
            int(
                round(
                    w * 2
                )
            ),
        )

        pad_y = max(
            35,
            int(
                round(
                    h * 2
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

        scale = min(
            (cell_w - 10)
            / crop.shape[1],

            (cell_h - 45)
            / crop.shape[0],
        )

        scale = min(
            scale,
            6.0,
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

        oy = 40

        card[
            oy:oy + crop.shape[0],
            ox:ox + crop.shape[1],
        ] = crop

        status = (
            "TRUST"
            if row[
                "trusted_class"
            ]
            else "AUDIT"
        )

        label = (
            f"{status} "
            f"{row['class']} "
            f"f{row['target_frame_index']} "
            f"N={row['forward_ncc']:.2f} "
            f"C={row['cycle_iou']:.2f}"
        )

        cv2.putText(
            card,
            label,
            (5, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        cards.append(
            card
        )

    if not cards:
        return

    montage_rows = []

    for start in range(
        0,
        len(cards),
        cols,
    ):
        row = cards[
            start:
            start + cols
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
    frames = (
        load_frame_records()
    )

    seeds = load_seeds(
        frames
    )

    print(
        "L1 frames:",
        len(frames),
    )

    print(
        "Manual seeds:",
        len(seeds),
    )

    raw_results = []

    for seed in seeds:
        candidates = candidate_frames(
            seed,
            frames,
        )

        for target in candidates:
            result = propagate_one(
                seed,
                target,
            )

            if result is not None:
                raw_results.append(
                    result
                )

    accepted = [
        r
        for r in raw_results
        if r["accepted"]
    ]

    deduped = deduplicate(
        raw_results
    )

    trusted = [
        r
        for r in deduped
        if r["trusted_class"]
    ]

    audit = [
        r
        for r in deduped
        if not r["trusted_class"]
    ]

    by_class = defaultdict(
        int
    )

    for row in deduped:
        by_class[
            row["class"]
        ] += 1

    output = {
        "max_gap":
            MAX_GAP,

        "min_ncc":
            MIN_NCC,

        "min_cycle_iou":
            MIN_CYCLE_IOU,

        "trusted_classes":
            sorted(
                TRUSTED_CLASSES
            ),

        "raw_matches":
            len(raw_results),

        "accepted_cycle_matches":
            len(accepted),

        "deduplicated_matches":
            len(deduped),

        "trusted_matches":
            len(trusted),

        "audit_only_matches":
            len(audit),

        "by_class":
            dict(
                sorted(
                    by_class.items()
                )
            ),

        "pseudolabels":
            deduped,
    }

    RESULTS_PATH.write_text(
        json.dumps(
            output,
            indent=2,
        )
    )

    make_montage(
        deduped
    )

    print()
    print("=" * 80)
    print(
        "EXP-D027D CYCLE-CONSISTENT PSEUDOLABELS"
    )
    print("=" * 80)

    print(
        "Raw forward matches:",
        len(raw_results),
    )

    print(
        "Cycle accepted:",
        len(accepted),
    )

    print(
        "After dedup:",
        len(deduped),
    )

    print(
        "Trusted:",
        len(trusted),
    )

    print(
        "Audit-only:",
        len(audit),
    )

    print()

    print(
        "Per class:"
    )

    for cls, count in sorted(
        by_class.items()
    ):
        marker = (
            "TRUST"
            if cls in TRUSTED_CLASSES
            else "AUDIT"
        )

        print(
            f"  {cls:20s} "
            f"{count:3d} "
            f"{marker}"
        )

    print()
    print(
        "Saved:",
        RESULTS_PATH,
    )

    print(
        "Montage:",
        MONTAGE_PATH,
    )


if __name__ == "__main__":
    main()
