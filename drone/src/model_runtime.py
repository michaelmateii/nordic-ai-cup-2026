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


MODEL_PATH = (
    ROOT
    / "drone"
    / "artifacts"
    / "exp_d032"
    / "runs"
    / "yolo11n_full_real_domain"
    / "weights"
    / "last.pt"
)

CONFIDENCE = 0.001
NMS_IOU = 0.70
IMAGE_SIZE = 960
MAX_DETECTIONS = 300

if torch.cuda.is_available():
    DEVICE_NAME = "cuda:0"
elif torch.backends.mps.is_available():
    DEVICE_NAME = "mps"
else:
    DEVICE_NAME = "cpu"


if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"D030 model not found: {MODEL_PATH}"
    )


print(
    f"[Drone D030] loading "
    f"{MODEL_PATH.name} "
    f"on {DEVICE_NAME}..."
)

_MODEL = YOLO(
    str(MODEL_PATH)
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

def _warm_model() -> None:
    print(
        "[Drone D030] warming YOLO..."
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
        _MODEL.predict(
            source=dummy,
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE,
            iou=NMS_IOU,
            max_det=MAX_DETECTIONS,
            device=DEVICE_NAME,
            verbose=False,
        )

    if DEVICE_NAME == "mps":
        torch.mps.synchronize()

    print(
        "[Drone D030] warmup complete"
    )
    

print(
    "[Drone runtime] "
    f"model={MODEL_PATH} "
    f"conf={CONFIDENCE} "
    f"nms={NMS_IOU} "
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

    result = _MODEL.predict(
        source=image,
        imgsz=IMAGE_SIZE,
        conf=CONFIDENCE,
        iou=NMS_IOU,
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

    annotations = []

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

        x1, y1, x2, y2 = (
            float(v)
            for v in box
        )

        # YOLO returns pixel coordinates relative
        # to the received view. Convert to normalized
        # coordinates relative to that view first.
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

        # Official helper:
        #
        # view-normalized
        # -> source pixels
        # -> full-frame normalized coordinates.
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
                object_id=class_name,
                bbox=global_bbox,
                confidence=float(
                    max(
                        0.0,
                        min(
                            1.0,
                            float(score),
                        ),
                    )
                ),
            )
        )

        if (
            len(annotations)
            >= 500
        ):
            break

    return annotations
