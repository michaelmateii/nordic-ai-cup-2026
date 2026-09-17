#!/usr/bin/env python3

from __future__ import annotations

import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import timm
import torch
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO


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

ARTIFACT_DIR = (
    ROOT
    / "drone"
    / "artifacts"
    / "exp_d014"
)

RESULT_PATH = (
    ARTIFACT_DIR
    / "yolo_mobilenet_results.json"
)

VIEW_W = 960
VIEW_H = 540

HOLDOUT_FRAMES = {
    22,
    23,
    24,
}

YOLO_CONF = 0.005
YOLO_NMS = 0.50

COLS = 2
ROWS = 2
OVERLAP = 0.20

CROP_PADDING = 0.30

MODEL_NAME = "mobilenetv3_small_100"


def find_image(frame: int) -> Path:
    for suffix in ("png", "jpg", "jpeg"):
        path = (
            IMAGE_DIR
            / f"frame_{frame:06d}.{suffix}"
        )

        if path.exists():
            return path

    matches = sorted(
        IMAGE_DIR.glob(
            f"*{frame:06d}*"
        )
    )

    if matches:
        return matches[0]

    raise FileNotFoundError(frame)


def load_frame(frame: int) -> np.ndarray:
    source = cv2.imread(
        str(
            find_image(
                frame
            )
        ),
        cv2.IMREAD_COLOR,
    )

    if source is None:
        raise RuntimeError(
            f"Could not read frame {frame}"
        )

    return cv2.resize(
        source,
        (
            VIEW_W,
            VIEW_H,
        ),
        interpolation=cv2.INTER_AREA,
    )


def bbox_l0(bbox):
    x1, y1, x2, y2 = map(
        float,
        bbox,
    )

    return np.asarray(
        [
            x1 / 4.0,
            y1 / 4.0,
            x2 / 4.0,
            y2 / 4.0,
        ],
        dtype=np.float32,
    )


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

    intersection = (
        max(
            0.0,
            x2 - x1,
        )
        * max(
            0.0,
            y2 - y1,
        )
    )

    area_a = (
        max(
            0.0,
            float(
                a[2]
                - a[0]
            ),
        )
        * max(
            0.0,
            float(
                a[3]
                - a[1]
            ),
        )
    )

    area_b = (
        max(
            0.0,
            float(
                b[2]
                - b[0]
            ),
        )
        * max(
            0.0,
            float(
                b[3]
                - b[1]
            ),
        )
    )

    union = (
        area_a
        + area_b
        - intersection
    )

    return (
        intersection / union
        if union > 0
        else 0.0
    )


def make_starts(
    length,
    count,
    overlap,
):
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
            int(
                round(
                    start
                )
            ),
            int(
                round(
                    start
                )
            )
            + tile_size,
        )
        for start in starts
    ]


def make_tiles(image):
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

    result = []

    for y1, y2 in ys:
        for x1, x2 in xs:
            result.append(
                {
                    "image":
                        image[
                            y1:y2,
                            x1:x2,
                        ],
                    "x1":
                        x1,
                    "y1":
                        y1,
                }
            )

    return result


def nms(
    boxes,
    scores,
    threshold=0.50,
):
    if not boxes:
        return [], []

    order = np.argsort(
        np.asarray(
            scores
        )
    )[::-1]

    keep_boxes = []
    keep_scores = []

    while len(order):
        i = int(
            order[0]
        )

        keep_boxes.append(
            boxes[i]
        )

        keep_scores.append(
            scores[i]
        )

        remaining = []

        for j in order[1:]:
            j = int(j)

            if iou(
                boxes[i],
                boxes[j],
            ) < threshold:
                remaining.append(
                    j
                )

        order = np.asarray(
            remaining,
            dtype=np.int64,
        )

    return (
        keep_boxes,
        keep_scores,
    )


def run_yolo(
    model,
    image,
    device,
):
    boxes = []
    scores = []

    if device == "mps":
        torch.mps.synchronize()

    start = time.perf_counter()

    for tile in make_tiles(
        image
    ):
        result = model.predict(
            source=tile[
                "image"
            ],
            imgsz=960,
            conf=YOLO_CONF,
            iou=0.70,
            max_det=300,
            device=device,
            verbose=False,
        )[0]

        if result.boxes is None:
            continue

        tile_boxes = (
            result.boxes
            .xyxy
            .detach()
            .cpu()
            .numpy()
        )

        tile_scores = (
            result.boxes
            .conf
            .detach()
            .cpu()
            .numpy()
        )

        for box, score in zip(
            tile_boxes,
            tile_scores,
        ):
            boxes.append(
                np.asarray(
                    [
                        float(
                            box[0]
                        )
                        + tile["x1"],
                        float(
                            box[1]
                        )
                        + tile["y1"],
                        float(
                            box[2]
                        )
                        + tile["x1"],
                        float(
                            box[3]
                        )
                        + tile["y1"],
                    ],
                    dtype=np.float32,
                )
            )

            scores.append(
                float(
                    score
                )
            )

    boxes, scores = nms(
        boxes,
        scores,
        YOLO_NMS,
    )

    if device == "mps":
        torch.mps.synchronize()

    latency = (
        time.perf_counter()
        - start
    ) * 1000.0

    return (
        boxes,
        scores,
        latency,
    )


def crop_rgb(
    image,
    bbox,
):
    x1, y1, x2, y2 = map(
        float,
        bbox,
    )

    w = x2 - x1
    h = y2 - y1

    px = w * CROP_PADDING
    py = h * CROP_PADDING

    image_h, image_w = (
        image.shape[:2]
    )

    x1 = max(
        0,
        int(
            round(
                x1 - px
            )
        ),
    )

    y1 = max(
        0,
        int(
            round(
                y1 - py
            )
        ),
    )

    x2 = min(
        image_w,
        int(
            round(
                x2 + px
            )
        ),
    )

    y2 = min(
        image_h,
        int(
            round(
                y2 + py
            )
        ),
    )

    crop = image[
        y1:y2,
        x1:x2,
    ]

    if crop.size == 0:
        return None

    return cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2RGB,
    )


def build_classifier(
    device,
):
    metadata = json.loads(
        CLASSIFIER_META.read_text()
    )

    classes = metadata[
        "classes"
    ]

    model = timm.create_model(
        MODEL_NAME,
        pretrained=False,
        num_classes=len(
            classes
        ),
    )

    state_dict = torch.load(
        CLASSIFIER_MODEL,
        map_location="cpu",
    )

    model.load_state_dict(
        state_dict
    )

    model.eval()
    model.to(
        device
    )

    transform = transforms.Compose(
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

    return (
        model,
        classes,
        transform,
    )


@torch.inference_mode()
def classify_batch(
    crops,
    model,
    transform,
    device,
):
    if not crops:
        return (
            np.empty(
                (0, 0),
                dtype=np.float32,
            ),
            0.0,
        )

    tensors = [
        transform(
            Image.fromarray(
                crop
            )
        )
        for crop in crops
    ]

    batch = torch.stack(
        tensors
    ).to(
        device
    )

    if device.type == "mps":
        torch.mps.synchronize()

    start = time.perf_counter()

    logits = model(
        batch
    )

    probabilities = torch.softmax(
        logits,
        dim=1,
    )

    if device.type == "mps":
        torch.mps.synchronize()

    latency = (
        time.perf_counter()
        - start
    ) * 1000.0

    return (
        probabilities
        .float()
        .cpu()
        .numpy(),
        latency,
    )


def average_precision(
    predictions,
    gt_by_frame,
):
    total_gt = sum(
        len(
            boxes
        )
        for boxes
        in gt_by_frame.values()
    )

    if total_gt == 0:
        return None

    predictions = sorted(
        predictions,
        key=lambda row:
            row[
                "score"
            ],
        reverse=True,
    )

    matched = {
        frame:
            np.zeros(
                len(
                    boxes
                ),
                dtype=bool,
            )
        for frame, boxes
        in gt_by_frame.items()
    }

    tp = []
    fp = []

    for prediction in predictions:
        frame = prediction[
            "frame"
        ]

        box = prediction[
            "bbox"
        ]

        gt_boxes = (
            gt_by_frame.get(
                frame,
                [],
            )
        )

        best_iou = 0.0
        best_index = -1

        for idx, gt_box in enumerate(
            gt_boxes
        ):
            if matched[
                frame
            ][idx]:
                continue

            current_iou = iou(
                box,
                gt_box,
            )

            if current_iou > best_iou:
                best_iou = current_iou
                best_index = idx

        if (
            best_index >= 0
            and best_iou >= 0.50
        ):
            matched[
                frame
            ][
                best_index
            ] = True

            tp.append(
                1.0
            )

            fp.append(
                0.0
            )

        else:
            tp.append(
                0.0
            )

            fp.append(
                1.0
            )

    if not predictions:
        return 0.0

    tp = np.cumsum(
        np.asarray(
            tp
        )
    )

    fp = np.cumsum(
        np.asarray(
            fp
        )
    )

    recall = (
        tp / total_gt
    )

    precision = (
        tp
        / np.maximum(
            tp + fp,
            1e-12,
        )
    )

    recall_points = np.linspace(
        0,
        1,
        101,
    )

    interpolated = []

    for recall_point in (
        recall_points
    ):
        valid = precision[
            recall
            >= recall_point
        ]

        interpolated.append(
            np.max(
                valid
            )
            if len(valid)
            else 0.0
        )

    return float(
        np.mean(
            interpolated
        )
    )


def main() -> None:
    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    device_name = (
        "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    device = torch.device(
        device_name
    )

    print(
        f"Device: {device}"
    )

    yolo = YOLO(
        str(
            YOLO_MODEL
        )
    )

    classifier, classes, transform = (
        build_classifier(
            device
        )
    )

    predictions_by_class = {
        class_name: []
        for class_name
        in classes
    }

    gt_by_class = {
        class_name:
            defaultdict(
                list
            )
        for class_name
        in classes
    }

    yolo_latencies = []
    classifier_latencies = []
    total_latencies = []
    proposal_counts = []

    # Warm up.
    warm_image = load_frame(
        22
    )

    run_yolo(
        yolo,
        warm_image,
        device_name,
    )

    warm_crop = crop_rgb(
        warm_image,
        [
            100,
            100,
            150,
            150,
        ],
    )

    classify_batch(
        [warm_crop],
        classifier,
        transform,
        device,
    )

    for frame in sorted(
        HOLDOUT_FRAMES
    ):
        image = load_frame(
            frame
        )

        annotation_path = (
            ANNOTATION_DIR
            / f"frame_{frame:06d}.json"
        )

        data = json.loads(
            annotation_path.read_text()
        )

        for annotation in data[
            "annotations"
        ]:
            class_name = annotation[
                "object_id"
            ]

            gt_by_class[
                class_name
            ][
                frame
            ].append(
                bbox_l0(
                    annotation[
                        "bbox"
                    ]
                )
            )

        start = time.perf_counter()

        (
            boxes,
            detector_scores,
            yolo_latency,
        ) = run_yolo(
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
            crop = crop_rgb(
                image,
                box,
            )

            if crop is None:
                continue

            crops.append(
                crop
            )

            valid_boxes.append(
                box
            )

            valid_detector_scores.append(
                detector_score
            )

        (
            probabilities,
            classifier_latency,
        ) = classify_batch(
            crops,
            classifier,
            transform,
            device,
        )

        proposal_counts.append(
            len(
                valid_boxes
            )
        )

        yolo_latencies.append(
            yolo_latency
        )

        classifier_latencies.append(
            classifier_latency
        )

        for index, box in enumerate(
            valid_boxes
        ):
            class_index = int(
                np.argmax(
                    probabilities[
                        index
                    ]
                )
            )

            class_name = classes[
                class_index
            ]

            class_probability = float(
                probabilities[
                    index,
                    class_index,
                ]
            )

            score = (
                class_probability
                * max(
                    valid_detector_scores[
                        index
                    ],
                    1e-6,
                )
            )

            predictions_by_class[
                class_name
            ].append(
                {
                    "frame":
                        frame,
                    "bbox":
                        box,
                    "score":
                        score,
                    "class_probability":
                        class_probability,
                }
            )

        total_latency = (
            time.perf_counter()
            - start
        ) * 1000.0

        total_latencies.append(
            total_latency
        )

        print(
            f"Frame {frame}: "
            f"proposals="
            f"{len(valid_boxes):2d}, "
            f"YOLO="
            f"{yolo_latency:6.1f} ms, "
            f"MobileNet="
            f"{classifier_latency:6.1f} ms, "
            f"total="
            f"{total_latency:6.1f} ms"
        )

    ap = {}

    for class_name in classes:
        ap[
            class_name
        ] = average_precision(
            predictions_by_class[
                class_name
            ],
            gt_by_class[
                class_name
            ],
        )

    scored_aps = [
        value
        for value in ap.values()
        if value is not None
    ]

    macro_map = float(
        np.mean(
            scored_aps
        )
    )

    print()
    print("=" * 72)
    print(
        "EXP-D014 HOLDOUT RESULTS"
    )
    print("=" * 72)

    print(
        f"Macro mAP@0.50: "
        f"{macro_map:.4f}"
    )

    print()
    print(
        "AP@0.50 by class"
    )

    for class_name in classes:
        value = ap[
            class_name
        ]

        if value is None:
            continue

        print(
            f"  "
            f"{class_name:20s} "
            f"{value:.4f}"
        )

    print()
    print("Latency")

    print(
        f"  mean proposals/frame: "
        f"{statistics.mean(proposal_counts):.1f}"
    )

    print(
        f"  YOLO median: "
        f"{statistics.median(yolo_latencies):.1f} ms"
    )

    print(
        f"  MobileNet batch median: "
        f"{statistics.median(classifier_latencies):.1f} ms"
    )

    print(
        f"  total median: "
        f"{statistics.median(total_latencies):.1f} ms"
    )

    print(
        f"  total max: "
        f"{max(total_latencies):.1f} ms"
    )

    output = {
        "macro_map_50":
            macro_map,

        "ap_50":
            ap,

        "mean_proposals":
            float(
                statistics.mean(
                    proposal_counts
                )
            ),

        "latency_ms": {
            "yolo_median":
                float(
                    statistics.median(
                        yolo_latencies
                    )
                ),

            "classifier_median":
                float(
                    statistics.median(
                        classifier_latencies
                    )
                ),

            "total_median":
                float(
                    statistics.median(
                        total_latencies
                    )
                ),

            "total_max":
                float(
                    max(
                        total_latencies
                    )
                ),
        },
    }

    RESULT_PATH.write_text(
        json.dumps(
            output,
            indent=2,
        )
    )

    print()
    print(
        f"Saved: {RESULT_PATH}"
    )


if __name__ == "__main__":
    main()
