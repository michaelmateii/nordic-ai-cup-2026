#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]

OFFICIAL = (
    ROOT
    / "drone"
    / "reference"
    / "official-drone-flyby"
)

if str(OFFICIAL) not in sys.path:
    sys.path.insert(
        0,
        str(OFFICIAL),
    )


from dtos import (
    OBJECT_CLASSES,
    DroneFlybyPredictionDto,
)

from utils import (
    clip_bbox_to_frame,
    view_bbox_to_global,
)


MODEL_PATH_21 = (
    ROOT
    / "drone"
    / "artifacts"
    / "exp_d040"
    / "seed_21"
    / "weights"
    / "best.pt"
)

MODEL_PATH_7 = (
    ROOT
    / "drone"
    / "artifacts"
    / "exp_d040"
    / "seed_7"
    / "weights"
    / "best.pt"
)


CONFIDENCE = 0.001
MODEL_NMS_IOU = 0.70
ENSEMBLE_NMS_IOU = 0.65
IMAGE_SIZE = 960
MAX_DETECTIONS = 300


if torch.cuda.is_available():
    DEVICE_NAME = "cuda:0"
elif torch.backends.mps.is_available():
    DEVICE_NAME = "mps"
else:
    DEVICE_NAME = "cpu"


if DEVICE_NAME == "cpu":
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)


for path in (
    MODEL_PATH_21,
    MODEL_PATH_7,
):
    if not path.exists():
        raise FileNotFoundError(
            f"Ensemble model not found: {path}"
        )


print(
    "[D041 ensemble] loading seed21 + seed7 "
    f"on {DEVICE_NAME}..."
)

_MODEL_21 = YOLO(
    str(MODEL_PATH_21)
)

_MODEL_7 = YOLO(
    str(MODEL_PATH_7)
)


def _class_name(
    names,
    class_index: int,
) -> str | None:

    if isinstance(
        names,
        dict,
    ):
        value = names.get(
            class_index
        )

    else:
        if not (
            0
            <= class_index
            < len(names)
        ):
            return None

        value = names[
            class_index
        ]

    if value is None:
        return None

    value = str(value)

    if value not in OBJECT_CLASSES:
        return None

    return value


def _iou(
    a,
    b,
) -> float:

    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(
        0.0,
        ix2 - ix1,
    )

    ih = max(
        0.0,
        iy2 - iy1,
    )

    inter = iw * ih

    if inter <= 0:
        return 0.0

    area_a = max(
        0.0,
        ax2 - ax1,
    ) * max(
        0.0,
        ay2 - ay1,
    )

    area_b = max(
        0.0,
        bx2 - bx1,
    ) * max(
        0.0,
        by2 - by1,
    )

    union = (
        area_a
        + area_b
        - inter
    )

    if union <= 0:
        return 0.0

    return inter / union


def _class_aware_nms(
    rows,
):
    """
    Merge duplicate predictions produced by the two seeds.

    Each row:
      {
        "class_name": str,
        "bbox": [x1,y1,x2,y2] in local pixel coords,
        "score": float,
      }
    """

    by_class = {}

    for row in rows:
        by_class.setdefault(
            row["class_name"],
            [],
        ).append(
            row
        )

    kept = []

    for class_name, class_rows in by_class.items():

        pending = sorted(
            class_rows,
            key=lambda r:
                r["score"],
            reverse=True,
        )

        while pending:

            best = pending.pop(0)

            kept.append(
                best
            )

            remaining = []

            for row in pending:

                overlap = _iou(
                    best["bbox"],
                    row["bbox"],
                )

                if overlap < ENSEMBLE_NMS_IOU:
                    remaining.append(
                        row
                    )

            pending = remaining

    return kept


def _run_model(
    model,
    image,
):
    result = model.predict(
        source=image,
        imgsz=IMAGE_SIZE,
        conf=CONFIDENCE,
        iou=MODEL_NMS_IOU,
        max_det=MAX_DETECTIONS,
        device=DEVICE_NAME,
        verbose=False,
    )[0]

    if result.boxes is None:
        return []

    boxes = (
        result.boxes.xyxy
        .detach()
        .cpu()
        .numpy()
    )

    scores = (
        result.boxes.conf
        .detach()
        .cpu()
        .numpy()
    )

    classes = (
        result.boxes.cls
        .detach()
        .cpu()
        .numpy()
        .astype(int)
    )

    rows = []

    for box, score, class_index in zip(
        boxes,
        scores,
        classes,
    ):

        class_name = _class_name(
            result.names,
            int(class_index),
        )

        if class_name is None:
            continue

        rows.append(
            {
                "class_name":
                    class_name,

                "bbox":
                    [
                        float(v)
                        for v in box
                    ],

                "score":
                    float(score),
            }
        )

    return rows


def _warm_model() -> None:

    print(
        "[D041 ensemble] warming both YOLO models..."
    )

    dummy = np.zeros(
        (
            540,
            960,
            3,
        ),
        dtype=np.uint8,
    )

    for _ in range(2):
        _run_model(
            _MODEL_21,
            dummy,
        )

        _run_model(
            _MODEL_7,
            dummy,
        )

    if DEVICE_NAME == "mps":
        torch.mps.synchronize()

    print(
        "[D041 ensemble] warmup complete"
    )


print(
    "[D041 runtime] "
    f"seed21={MODEL_PATH_21.name} "
    f"seed7={MODEL_PATH_7.name} "
    f"conf={CONFIDENCE} "
    f"model_nms={MODEL_NMS_IOU} "
    f"ensemble_nms={ENSEMBLE_NMS_IOU} "
    f"imgsz={IMAGE_SIZE}"
)

_warm_model()


@torch.inference_mode()
def detect(
    image: np.ndarray,
    source_region_xyxy,
    original_width: int,
    original_height: int,
) -> list[DroneFlybyPredictionDto]:

    if image is None:
        return []

    image_height, image_width = (
        image.shape[:2]
    )

    if (
        image_width <= 0
        or image_height <= 0
    ):
        return []

    rows = []

    rows.extend(
        _run_model(
            _MODEL_21,
            image,
        )
    )

    rows.extend(
        _run_model(
            _MODEL_7,
            image,
        )
    )

    rows = _class_aware_nms(
        rows
    )

    # Keep global ordering deterministic.
    rows.sort(
        key=lambda r:
            r["score"],
        reverse=True,
    )

    annotations = []

    for row in rows:

        x1, y1, x2, y2 = (
            row["bbox"]
        )

        local_bbox = (
            max(
                0.0,
                min(
                    1.0,
                    x1 / image_width,
                ),
            ),
            max(
                0.0,
                min(
                    1.0,
                    y1 / image_height,
                ),
            ),
            max(
                0.0,
                min(
                    1.0,
                    x2 / image_width,
                ),
            ),
            max(
                0.0,
                min(
                    1.0,
                    y2 / image_height,
                ),
            ),
        )

        if (
            local_bbox[2]
            <= local_bbox[0]
            or local_bbox[3]
            <= local_bbox[1]
        ):
            continue

        global_bbox = (
            view_bbox_to_global(
                local_bbox,
                source_region_xyxy,
                original_width,
                original_height,
            )
        )

        global_bbox = (
            clip_bbox_to_frame(
                global_bbox
            )
        )

        if global_bbox is None:
            continue

        annotations.append(
            DroneFlybyPredictionDto(
                object_id=
                    row["class_name"],

                bbox=
                    global_bbox,

                confidence=float(
                    max(
                        0.0,
                        min(
                            1.0,
                            row["score"],
                        ),
                    )
                ),
            )
        )

        if len(annotations) >= 500:
            break

    return annotations
