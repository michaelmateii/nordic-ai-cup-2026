#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import math
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]

CAPTURE_ROOT = (
    ROOT
    / "drone"
    / "captures"
)

PIPELINE_PATH = (
    ROOT
    / "drone"
    / "scripts"
    / "evaluate_yolo_mobilenet_pipeline.py"
)

OUT = (
    ROOT
    / "drone"
    / "artifacts"
    / "exp_d033"
    / "missing_class_candidates"
)

JSON_OUT = OUT / "candidates.json"


TARGET_CLASSES = [
    "condor",
    "jammer",
    "small_launcher",
    "spacecraft",
    "ta-ta",
]

TOP_K = 30

# Do not throw away weaker MobileNet candidates during
# retrieval. Human inspection will make the final decision.
MIN_CLASS_PROB = 0.01


def load_pipeline_module():
    spec = importlib.util.spec_from_file_location(
        "d033_old_pipeline",
        PIPELINE_PATH,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not import {PIPELINE_PATH}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    sys.modules[
        "d033_old_pipeline"
    ] = module

    spec.loader.exec_module(
        module
    )

    return module


def discover_unique_requests():
    """
    Deduplicate the same validation request captured in
    several competition attempts.

    The validation scene is deterministic enough that
    frame_index + resolution level + source region
    identifies the same visual request for our purposes.
    """

    chosen = {}

    sequence_counts = defaultdict(int)

    for sequence in sorted(
        CAPTURE_ROOT.iterdir()
    ):
        if (
            not sequence.is_dir()
            or sequence.name == "local"
        ):
            continue

        frames_dir = sequence / "frames"

        if not frames_dir.exists():
            continue

        for meta_path in sorted(
            frames_dir.glob("*.json")
        ):
            try:
                meta = json.loads(
                    meta_path.read_text()
                )
            except Exception:
                continue

            image_path = (
                meta_path.with_suffix(
                    ".png"
                )
            )

            if not image_path.exists():
                continue

            view = meta["view"]

            region = tuple(
                int(v)
                for v in view[
                    "source_region_xyxy"
                ]
            )

            key = (
                int(meta["frame_index"]),
                int(
                    view[
                        "resolution_level"
                    ]
                ),
                region,
            )

            candidate = {
                "sequence_id":
                    sequence.name,

                "frame":
                    int(meta["frame"]),

                "frame_index":
                    int(
                        meta[
                            "frame_index"
                        ]
                    ),

                "resolution_level":
                    int(
                        view[
                            "resolution_level"
                        ]
                    ),

                "center_x":
                    int(
                        view["center_x"]
                    ),

                "center_y":
                    int(
                        view["center_y"]
                    ),

                "source_region_xyxy":
                    list(region),

                "image_path":
                    str(image_path),
            }

            # First copy of an exact request is enough.
            if key not in chosen:
                chosen[key] = candidate
                sequence_counts[
                    sequence.name
                ] += 1

    rows = list(
        chosen.values()
    )

    rows.sort(
        key=lambda r: (
            r["frame_index"],
            r["resolution_level"],
            r["center_y"],
            r["center_x"],
        )
    )

    return rows, sequence_counts


def make_context_card(
    image,
    bbox,
    class_name,
    class_probability,
    detector_score,
    frame_index,
    level,
    center_x,
    center_y,
):
    h, w = image.shape[:2]

    x1, y1, x2, y2 = [
        int(round(v))
        for v in bbox
    ]

    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(x1 + 1, min(w, x2))
    y2 = max(y1 + 1, min(h, y2))

    bw = x2 - x1
    bh = y2 - y1

    # Show context around the proposal, not only the
    # classifier crop. This makes human verification
    # substantially easier.
    pad_x = max(
        40,
        int(bw * 2.0),
    )

    pad_y = max(
        40,
        int(bh * 2.0),
    )

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    rx1 = max(
        0,
        cx - bw // 2 - pad_x,
    )

    ry1 = max(
        0,
        cy - bh // 2 - pad_y,
    )

    rx2 = min(
        w,
        cx + (bw + 1) // 2 + pad_x,
    )

    ry2 = min(
        h,
        cy + (bh + 1) // 2 + pad_y,
    )

    context = image[
        ry1:ry2,
        rx1:rx2,
    ].copy()

    if context.size == 0:
        return None

    local_x1 = x1 - rx1
    local_y1 = y1 - ry1
    local_x2 = x2 - rx1
    local_y2 = y2 - ry1

    cv2.rectangle(
        context,
        (
            local_x1,
            local_y1,
        ),
        (
            local_x2,
            local_y2,
        ),
        (0, 0, 255),
        2,
    )

    canvas_w = 300
    image_h = 220
    header_h = 62

    resized = cv2.resize(
        context,
        (
            canvas_w,
            image_h,
        ),
        interpolation=
            cv2.INTER_AREA,
    )

    card = np.zeros(
        (
            header_h + image_h,
            canvas_w,
            3,
        ),
        dtype=np.uint8,
    )

    card[
        header_h:,
        :
    ] = resized

    line1 = (
        f"{class_name} "
        f"P={class_probability:.3f} "
        f"O={detector_score:.3f}"
    )

    line2 = (
        f"idx={frame_index} "
        f"L{level} "
        f"c=({center_x},{center_y})"
    )

    cv2.putText(
        card,
        line1,
        (5, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.47,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        card,
        line2,
        (5, 47),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return card


def write_montage(
    class_name,
    candidates,
):
    cards = []

    for row in candidates[:TOP_K]:
        image = cv2.imread(
            row["image_path"]
        )

        if image is None:
            continue

        card = make_context_card(
            image=image,
            bbox=row["bbox"],
            class_name=class_name,
            class_probability=
                row[
                    "class_probability"
                ],
            detector_score=
                row[
                    "detector_score"
                ],
            frame_index=
                row["frame_index"],
            level=
                row[
                    "resolution_level"
                ],
            center_x=
                row["center_x"],
            center_y=
                row["center_y"],
        )

        if card is not None:
            cards.append(card)

    if not cards:
        return None

    cols = 4

    blank = np.zeros_like(
        cards[0]
    )

    while len(cards) % cols:
        cards.append(
            blank.copy()
        )

    montage_rows = []

    for i in range(
        0,
        len(cards),
        cols,
    ):
        montage_rows.append(
            np.hstack(
                cards[i:i + cols]
            )
        )

    montage = np.vstack(
        montage_rows
    )

    out_path = (
        OUT
        / f"{class_name}.jpg"
    )

    cv2.imwrite(
        str(out_path),
        montage,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            94,
        ],
    )

    return out_path


def main():
    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    pipe = load_pipeline_module()

    device_name = (
        "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    device = torch.device(
        device_name
    )

    print(
        "Device:",
        device,
    )

    print(
        "YOLO:",
        pipe.YOLO_MODEL,
    )

    print(
        "Classifier:",
        pipe.CLASSIFIER_MODEL,
    )

    requests, sequence_counts = (
        discover_unique_requests()
    )

    print()
    print(
        "Unique captured requests:",
        len(requests),
    )

    print()
    print(
        "Unique requests contributed "
        "by sequence:"
    )

    for seq, count in sorted(
        sequence_counts.items(),
        key=lambda x: (
            -x[1],
            x[0],
        ),
    ):
        print(
            f"  {seq}  {count}"
        )

    print()
    print(
        "Loading objectness model..."
    )

    yolo = YOLO(
        str(
            pipe.YOLO_MODEL
        )
    )

    print(
        "Loading MobileNet..."
    )

    (
        classifier,
        classes,
        transform,
    ) = pipe.build_classifier(
        device
    )

    class_to_idx = {
        name: i
        for i, name in enumerate(
            classes
        )
    }

    print()
    print(
        "Classifier classes:"
    )

    for i, name in enumerate(
        classes
    ):
        print(
            f"  {i:2d} {name}"
        )

    for class_name in TARGET_CLASSES:
        if class_name not in class_to_idx:
            raise RuntimeError(
                f"{class_name} absent from "
                "classifier metadata"
            )

    # Warm both models.
    if requests:
        warm = cv2.imread(
            requests[0][
                "image_path"
            ]
        )

        if warm is not None:
            pipe.run_yolo(
                yolo,
                warm,
                device_name,
            )

            warm_crop = pipe.crop_rgb(
                warm,
                [
                    100,
                    100,
                    180,
                    180,
                ],
            )

            if warm_crop is not None:
                pipe.classify_batch(
                    [warm_crop],
                    classifier,
                    transform,
                    device,
                )

    candidates = {
        class_name: []
        for class_name
        in TARGET_CLASSES
    }

    total_proposals = 0

    for request_index, request in enumerate(
        requests,
        start=1,
    ):
        image = cv2.imread(
            request[
                "image_path"
            ]
        )

        if image is None:
            continue

        (
            boxes,
            detector_scores,
            _,
        ) = pipe.run_yolo(
            yolo,
            image,
            device_name,
        )

        crops = []
        valid_boxes = []
        valid_detector_scores = []

        for box, detector_score in zip(
            boxes,
            detector_scores,
        ):
            crop = pipe.crop_rgb(
                image,
                box,
            )

            if crop is None:
                continue

            crops.append(
                crop
            )

            valid_boxes.append(
                [
                    float(v)
                    for v in box
                ]
            )

            valid_detector_scores.append(
                float(
                    detector_score
                )
            )

        total_proposals += len(
            valid_boxes
        )

        if not crops:
            if (
                request_index % 25
                == 0
            ):
                print(
                    f"{request_index:4d}/"
                    f"{len(requests)} "
                    f"proposals="
                    f"{total_proposals}"
                )
            continue

        (
            probabilities,
            _,
        ) = pipe.classify_batch(
            crops,
            classifier,
            transform,
            device,
        )

        for proposal_index, (
            box,
            detector_score,
        ) in enumerate(
            zip(
                valid_boxes,
                valid_detector_scores,
            )
        ):
            for class_name in TARGET_CLASSES:
                class_index = class_to_idx[
                    class_name
                ]

                class_probability = float(
                    probabilities[
                        proposal_index,
                        class_index,
                    ]
                )

                if (
                    class_probability
                    < MIN_CLASS_PROB
                ):
                    continue

                # Save both measures. Ranking for
                # retrieval is primarily classifier
                # probability because objectness already
                # filtered the candidate set.
                combined_score = (
                    class_probability
                    * math.sqrt(
                        max(
                            detector_score,
                            1e-6,
                        )
                    )
                )

                candidates[
                    class_name
                ].append(
                    {
                        "class":
                            class_name,

                        "class_probability":
                            class_probability,

                        "detector_score":
                            detector_score,

                        "combined_score":
                            combined_score,

                        "bbox":
                            box,

                        **request,
                    }
                )

        if (
            request_index % 25
            == 0
            or request_index
            == len(requests)
        ):
            print(
                f"{request_index:4d}/"
                f"{len(requests)} "
                f"cumulative proposals="
                f"{total_proposals}"
            )

    # Rank and retain a larger JSON list than the montage
    # so we can inspect deeper if necessary.
    JSON_TOP_K = 100

    saved = {}

    print()
    print("=" * 80)
    print(
        "EXP-D033 MISSING CLASS RETRIEVAL"
    )
    print("=" * 80)

    for class_name in TARGET_CLASSES:
        rows = candidates[
            class_name
        ]

        rows.sort(
            key=lambda r: (
                r[
                    "class_probability"
                ],
                r[
                    "detector_score"
                ],
            ),
            reverse=True,
        )

        rows = rows[
            :JSON_TOP_K
        ]

        saved[
            class_name
        ] = rows

        montage = write_montage(
            class_name,
            rows,
        )

        print()
        print(
            class_name
        )

        if rows:
            print(
                f"  candidates retained: "
                f"{len(rows)}"
            )

            print(
                f"  best class P: "
                f"{rows[0]['class_probability']:.4f}"
            )

            print(
                f"  best objectness: "
                f"{rows[0]['detector_score']:.4f}"
            )

            print(
                f"  top frame: "
                f"{rows[0]['frame_index']}"
            )

        else:
            print(
                "  no candidates"
            )

        print(
            f"  montage: {montage}"
        )

    JSON_OUT.write_text(
        json.dumps(
            {
                "target_classes":
                    TARGET_CLASSES,

                "unique_requests":
                    len(requests),

                "total_proposals":
                    total_proposals,

                "ranking":
                    "class_probability_then_objectness",

                "candidates":
                    saved,
            },
            indent=2,
        )
    )

    print()
    print(
        "Total proposals:",
        total_proposals,
    )

    print(
        "Saved:",
        JSON_OUT,
    )


if __name__ == "__main__":
    main()
