#!/usr/bin/env python3

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]

SEQUENCE = (
    ROOT
    / "drone"
    / "captures"
    / "3224a582bfbf4273a028497662b7aa7c"
    / "frames"
)

MODEL_PATH = (
    ROOT
    / "drone"
    / "artifacts"
    / "exp_d010"
    / "runs"
    / "yolo11n_tiled_objectness"
    / "weights"
    / "best.pt"
)

OUT = (
    ROOT
    / "drone"
    / "artifacts"
    / "exp_d023"
)

CROPS = OUT / "crops"

YOLO_CONF = 0.025
NMS_IOU = 0.50

# The received L1 image is always 960x540.
VIEW_W = 960
VIEW_H = 540

COLS = 2
ROWS = 2
OVERLAP = 0.20

CROP_CONTEXT = 0.70

# Keep the audit manageable.
MAX_PROPOSALS_PER_FRAME = 20
MAX_MONTAGE_CROPS = 400


def iou(a, b):
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))

    inter = (
        max(0.0, x2 - x1)
        * max(0.0, y2 - y1)
    )

    aa = (
        max(0.0, float(a[2] - a[0]))
        * max(0.0, float(a[3] - a[1]))
    )

    ab = (
        max(0.0, float(b[2] - b[0]))
        * max(0.0, float(b[3] - b[1]))
    )

    union = aa + ab - inter

    return inter / union if union > 0 else 0.0


def nms(boxes, scores):
    if not boxes:
        return [], []

    order = np.argsort(scores)[::-1]

    kept_boxes = []
    kept_scores = []

    while len(order):
        i = int(order[0])

        kept_boxes.append(boxes[i])
        kept_scores.append(float(scores[i]))

        remaining = []

        for j in order[1:]:
            j = int(j)

            if iou(boxes[i], boxes[j]) < NMS_IOU:
                remaining.append(j)

        order = np.asarray(
            remaining,
            dtype=np.int64,
        )

    return kept_boxes, kept_scores


def make_starts(length, count):
    tile_size = int(
        round(
            (length / count)
            * (1.0 + OVERLAP)
        )
    )

    starts = np.linspace(
        0,
        length - tile_size,
        count,
    )

    return [
        (
            int(round(start)),
            int(round(start)) + tile_size,
        )
        for start in starts
    ]


def tiles(image):
    xs = make_starts(VIEW_W, COLS)
    ys = make_starts(VIEW_H, ROWS)

    for y1, y2 in ys:
        for x1, x2 in xs:
            yield (
                image[y1:y2, x1:x2],
                x1,
                y1,
            )


def crop_with_context(image, box):
    x1, y1, x2, y2 = map(float, box)

    w = x2 - x1
    h = y2 - y1

    px = w * CROP_CONTEXT
    py = h * CROP_CONTEXT

    ih, iw = image.shape[:2]

    cx1 = max(0, int(round(x1 - px)))
    cy1 = max(0, int(round(y1 - py)))
    cx2 = min(iw, int(round(x2 + px)))
    cy2 = min(ih, int(round(y2 + py)))

    if cx2 <= cx1 or cy2 <= cy1:
        return None

    return image[cy1:cy2, cx1:cx2]


def make_montage(records):
    records = records[:MAX_MONTAGE_CROPS]

    cell_w = 240
    cell_h = 190
    cols = 5

    rows = math.ceil(
        len(records) / cols
    )

    canvas = np.zeros(
        (
            rows * cell_h,
            cols * cell_w,
            3,
        ),
        dtype=np.uint8,
    )

    for pos, record in enumerate(records):
        crop = cv2.imread(
            record["crop_path"]
        )

        if crop is None:
            continue

        max_w = cell_w - 10
        max_h = cell_h - 38

        scale = min(
            max_w / crop.shape[1],
            max_h / crop.shape[0],
        )

        scale = min(scale, 4.0)

        resized = cv2.resize(
            crop,
            (
                max(1, int(crop.shape[1] * scale)),
                max(1, int(crop.shape[0] * scale)),
            ),
            interpolation=(
                cv2.INTER_NEAREST
                if scale > 1
                else cv2.INTER_AREA
            ),
        )

        row = pos // cols
        col = pos % cols

        ox = col * cell_w
        oy = row * cell_h

        y = oy + 30
        x = (
            ox
            + (cell_w - resized.shape[1]) // 2
        )

        canvas[
            y:y + resized.shape[0],
            x:x + resized.shape[1],
        ] = resized

        label = (
            f"#{record['id']:04d} "
            f"f{record['frame_index']:03d} "
            f"{record['score']:.2f}"
        )

        cv2.putText(
            canvas,
            label,
            (ox + 5, oy + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    path = OUT / "proposal_montage.jpg"

    cv2.imwrite(
        str(path),
        canvas,
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

    CROPS.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    model = YOLO(
        str(MODEL_PATH)
    )

    metadata_paths = sorted(
        SEQUENCE.glob("*.json")
    )

    records = []
    proposal_id = 0

    print("Frames:", len(metadata_paths))
    print("Device:", device)

    for n, meta_path in enumerate(
        metadata_paths,
        start=1,
    ):
        meta = json.loads(
            meta_path.read_text()
        )

        if (
            meta["view"]["resolution_level"]
            != 1
        ):
            continue

        image_path = meta_path.with_suffix(
            ".png"
        )

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            continue

        boxes = []
        scores = []

        for tile, offset_x, offset_y in tiles(
            image
        ):
            result = model.predict(
                tile,
                imgsz=960,
                conf=YOLO_CONF,
                iou=0.70,
                max_det=300,
                device=device,
                verbose=False,
            )[0]

            if result.boxes is None:
                continue

            tb = (
                result.boxes.xyxy
                .detach()
                .cpu()
                .numpy()
            )

            ts = (
                result.boxes.conf
                .detach()
                .cpu()
                .numpy()
            )

            for box, score in zip(tb, ts):
                boxes.append(
                    np.asarray(
                        [
                            float(box[0]) + offset_x,
                            float(box[1]) + offset_y,
                            float(box[2]) + offset_x,
                            float(box[3]) + offset_y,
                        ],
                        dtype=np.float32,
                    )
                )

                scores.append(
                    float(score)
                )

        boxes, scores = nms(
            boxes,
            np.asarray(scores),
        )

        ranked = sorted(
            zip(boxes, scores),
            key=lambda x: x[1],
            reverse=True,
        )[
            :MAX_PROPOSALS_PER_FRAME
        ]

        for box, score in ranked:
            crop = crop_with_context(
                image,
                box,
            )

            if crop is None:
                continue

            crop_name = (
                f"proposal_{proposal_id:04d}"
                f"_frame_{meta['frame_index']:03d}.png"
            )

            crop_path = (
                CROPS
                / crop_name
            )

            cv2.imwrite(
                str(crop_path),
                crop,
            )

            records.append(
                {
                    "id": proposal_id,
                    "frame":
                        int(meta["frame"]),
                    "frame_index":
                        int(meta["frame_index"]),
                    "score":
                        float(score),
                    "bbox_view_xyxy":
                        [
                            float(v)
                            for v in box
                        ],
                    "view_center":
                        [
                            int(
                                meta["view"]["center_x"]
                            ),
                            int(
                                meta["view"]["center_y"]
                            ),
                        ],
                    "source_region_xyxy":
                        meta["view"][
                            "source_region_xyxy"
                        ],
                    "crop_path":
                        str(crop_path),
                }
            )

            proposal_id += 1

        if (
            n % 25 == 0
            or n == len(metadata_paths)
        ):
            print(
                f"{n:3d}/{len(metadata_paths)} "
                f"cumulative proposals="
                f"{len(records)}"
            )

    records.sort(
        key=lambda r: r["score"],
        reverse=True,
    )

    manifest = (
        OUT
        / "proposal_manifest.json"
    )

    manifest.write_text(
        json.dumps(
            records,
            indent=2,
        )
    )

    montage = make_montage(
        records
    )

    print()
    print("=" * 72)
    print("EXP-D023 PROPOSAL AUDIT")
    print("=" * 72)
    print(
        "L1 frames audited:",
        sum(
            1
            for p in metadata_paths
            if json.loads(
                p.read_text()
            )["view"]["resolution_level"]
            == 1
        ),
    )
    print(
        "Proposal crops:",
        len(records),
    )
    print(
        "Manifest:",
        manifest,
    )
    print(
        "Montage:",
        montage,
    )


if __name__ == "__main__":
    main()
