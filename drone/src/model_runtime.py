#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import timm
import torch
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]

OFFICIAL = (
    ROOT
    / "drone"
    / "reference"
    / "official-drone-flyby"
)

if str(OFFICIAL) not in sys.path:
    sys.path.insert(0, str(OFFICIAL))

from dtos import DroneFlybyPredictionDto


YOLO_MODEL = (
    ROOT
    / "drone"
    / "artifacts"
    / "exp_d010"
    / "runs"
    / "yolo11n_tiled_objectness"
    / "weights"
    / "best.pt"
)

CLASSIFIER_MODEL = (
    ROOT
    / "drone"
    / "artifacts"
    / "exp_d013"
    / "classifier"
    / "mobilenetv3_small_best.pt"
)

CLASSIFIER_META = (
    ROOT
    / "drone"
    / "artifacts"
    / "exp_d013"
    / "classifier"
    / "metadata.json"
)


YOLO_CONF = 0.025
YOLO_NMS = 0.50

CLASSIFIER_BATCH_SIZE = 16

VIEW_W = 960
VIEW_H = 540

COLS = 2
ROWS = 2
OVERLAP = 0.20

CROP_PADDING = 0.30


DEVICE_NAME = (
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

DEVICE = torch.device(DEVICE_NAME)


def _iou(a, b) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))

    intersection = (
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

    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def _make_starts(length, count, overlap):
    tile_size = int(
        round(
            (length / count)
            * (1.0 + overlap)
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


def _make_tiles(image):
    xs = _make_starts(
        VIEW_W,
        COLS,
        OVERLAP,
    )

    ys = _make_starts(
        VIEW_H,
        ROWS,
        OVERLAP,
    )

    tiles = []

    for y1, y2 in ys:
        for x1, x2 in xs:
            tiles.append(
                {
                    "image": image[y1:y2, x1:x2],
                    "x1": x1,
                    "y1": y1,
                }
            )

    return tiles


def _nms(boxes, scores):
    if not boxes:
        return [], []

    order = np.argsort(
        np.asarray(scores)
    )[::-1]

    keep_boxes = []
    keep_scores = []

    while len(order):
        i = int(order[0])

        keep_boxes.append(boxes[i])
        keep_scores.append(scores[i])

        remaining = []

        for j in order[1:]:
            j = int(j)

            if _iou(
                boxes[i],
                boxes[j],
            ) < YOLO_NMS:
                remaining.append(j)

        order = np.asarray(
            remaining,
            dtype=np.int64,
        )

    return keep_boxes, keep_scores


def _crop_rgb(image, bbox):
    x1, y1, x2, y2 = map(float, bbox)

    width = x2 - x1
    height = y2 - y1

    px = width * CROP_PADDING
    py = height * CROP_PADDING

    image_h, image_w = image.shape[:2]

    x1 = max(0, int(round(x1 - px)))
    y1 = max(0, int(round(y1 - py)))
    x2 = min(image_w, int(round(x2 + px)))
    y2 = min(image_h, int(round(y2 + py)))

    crop = image[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    return cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2RGB,
    )


print(
    f"[Drone] loading YOLO on {DEVICE_NAME}..."
)

_YOLO = YOLO(
    str(YOLO_MODEL)
)


_METADATA = json.loads(
    CLASSIFIER_META.read_text()
)

_CLASSES = _METADATA["classes"]


print(
    f"[Drone] loading MobileNet on {DEVICE_NAME}..."
)

_CLASSIFIER = timm.create_model(
    "mobilenetv3_small_100",
    pretrained=False,
    num_classes=len(_CLASSES),
)

_CLASSIFIER.load_state_dict(
    torch.load(
        CLASSIFIER_MODEL,
        map_location="cpu",
    )
)

_CLASSIFIER.eval()
_CLASSIFIER.to(DEVICE)


_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(
            (224, 224)
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406,
            ],
            std=[
                0.229,
                0.224,
                0.225,
            ],
        ),
    ]
)

def _warm_models() -> None:
    print(
        "[Drone] warming YOLO + MobileNet..."
    )

    dummy_frame = np.zeros(
        (
            VIEW_H,
            VIEW_W,
            3,
        ),
        dtype=np.uint8,
    )

    # Warm all four YOLO tile calls.
    for tile in _make_tiles(
        dummy_frame
    ):
        _YOLO.predict(
            source=tile["image"],
            imgsz=960,
            conf=YOLO_CONF,
            iou=0.70,
            max_det=300,
            device=DEVICE_NAME,
            verbose=False,
        )

    # Warm the exact fixed classifier batch shape.
    dummy_batch = torch.zeros(
        (
            CLASSIFIER_BATCH_SIZE,
            3,
            224,
            224,
        ),
        dtype=torch.float32,
        device=DEVICE,
    )

    with torch.inference_mode():
        for _ in range(3):
            _CLASSIFIER(
                dummy_batch
            )

    if DEVICE.type == "mps":
        torch.mps.synchronize()

    print(
        "[Drone] warmup complete"
    )


_warm_models()


@torch.inference_mode()
def detect(
    image: np.ndarray,
    original_width: int,
    original_height: int,
) -> list[DroneFlybyPredictionDto]:

    boxes = []
    detector_scores = []

    for tile in _make_tiles(image):

        result = _YOLO.predict(
            source=tile["image"],
            imgsz=960,
            conf=YOLO_CONF,
            iou=0.70,
            max_det=300,
            device=DEVICE_NAME,
            verbose=False,
        )[0]

        if result.boxes is None:
            continue

        tile_boxes = (
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

        for box, score in zip(
            tile_boxes,
            scores,
        ):

            boxes.append(
                np.asarray(
                    [
                        float(box[0])
                        + tile["x1"],

                        float(box[1])
                        + tile["y1"],

                        float(box[2])
                        + tile["x1"],

                        float(box[3])
                        + tile["y1"],
                    ],
                    dtype=np.float32,
                )
            )

            detector_scores.append(
                float(score)
            )

    boxes, detector_scores = _nms(
        boxes,
        detector_scores,
    )

    crops = []
    valid_boxes = []
    valid_detector_scores = []

    for box, score in zip(
        boxes,
        detector_scores,
    ):
        crop = _crop_rgb(
            image,
            box,
        )

        if crop is None:
            continue

        crops.append(crop)
        valid_boxes.append(box)
        valid_detector_scores.append(score)

    if not crops:
        return []

    all_probabilities = []

    for start in range(
        0,
        len(crops),
        CLASSIFIER_BATCH_SIZE,
    ):
        chunk = crops[
            start:
            start + CLASSIFIER_BATCH_SIZE
        ]

        tensors = [
            _TRANSFORM(
                Image.fromarray(
                    crop
                )
            )
            for crop in chunk
        ]
        
        

        # Pad every classifier invocation to the same shape.
        #
        # This is intentional: stable MPS tensor shapes should avoid
        # expensive graph recompilation when proposal count changes.
        while len(tensors) < CLASSIFIER_BATCH_SIZE:
            tensors.append(
                torch.zeros_like(
                    tensors[0]
                )
            )

        batch = torch.stack(
            tensors
        ).to(DEVICE)

        chunk_probabilities = torch.softmax(
            _CLASSIFIER(batch),
            dim=1,
        )

        # Remove predictions corresponding to padding.
        chunk_probabilities = (
            chunk_probabilities[
                :len(chunk)
            ]
            .float()
            .cpu()
            .numpy()
        )

        all_probabilities.append(
            chunk_probabilities
        )

    probabilities = np.concatenate(
        all_probabilities,
        axis=0,
    )

    annotations = []

    for index, box in enumerate(
        valid_boxes
    ):

        class_index = int(
            np.argmax(
                probabilities[index]
            )
        )

        class_name = _CLASSES[
            class_index
        ]

        class_probability = float(
            probabilities[
                index,
                class_index,
            ]
        )

        confidence = float(
            class_probability
            * max(
                valid_detector_scores[index],
                1e-6,
            )
        )

        # L0 view pixel -> source pixel -> normalized global.
        #
        # At L0 the received 960x540 covers the entire
        # 3840x2160 source frame.
        x1 = float(box[0]) / VIEW_W
        y1 = float(box[1]) / VIEW_H
        x2 = float(box[2]) / VIEW_W
        y2 = float(box[3]) / VIEW_H

        # Defensive clipping.
        x1 = min(max(x1, 0.0), 1.0)
        y1 = min(max(y1, 0.0), 1.0)
        x2 = min(max(x2, 0.0), 1.0)
        y2 = min(max(y2, 0.0), 1.0)

        if x2 <= x1 or y2 <= y1:
            continue

        annotations.append(
            DroneFlybyPredictionDto(
                object_id=
                    class_name,

                bbox=[
                    x1,
                    y1,
                    x2,
                    y2,
                ],

                confidence=
                    confidence,
            )
        )

    return annotations
