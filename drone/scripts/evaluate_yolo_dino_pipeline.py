#!/usr/bin/env python3

from __future__ import annotations

import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel
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

ARTIFACT_DIR = (
    ROOT
    / "drone"
    / "artifacts"
    / "exp_d012"
)

RESULT_PATH = (
    ARTIFACT_DIR
    / "yolo_dino_results.json"
)

DINO_MODEL = "facebook/dinov2-small"

VIEW_W = 960
VIEW_H = 540

TRAIN_MAX_FRAME = 20
HOLDOUT_FRAMES = {22, 23, 24}

YOLO_CONF = 0.005
YOLO_NMS = 0.50

COLS = 2
ROWS = 2
OVERLAP = 0.20

CROP_PADDING = 0.20


def find_image(frame: int) -> Path:
    for suffix in ("png", "jpg", "jpeg"):
        p = IMAGE_DIR / f"frame_{frame:06d}.{suffix}"
        if p.exists():
            return p

    matches = sorted(
        IMAGE_DIR.glob(f"*{frame:06d}*")
    )

    if matches:
        return matches[0]

    raise FileNotFoundError(frame)


def load_frame(frame: int) -> np.ndarray:
    source = cv2.imread(
        str(find_image(frame)),
        cv2.IMREAD_COLOR,
    )

    if source is None:
        raise RuntimeError(
            f"Could not read frame {frame}"
        )

    return cv2.resize(
        source,
        (VIEW_W, VIEW_H),
        interpolation=cv2.INTER_AREA,
    )


def bbox_l0(bbox):
    x1, y1, x2, y2 = map(float, bbox)

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
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))

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

    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def make_starts(length, count, overlap):
    if count == 1:
        return [(0, length)]

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
            int(round(start))
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
                        image[y1:y2, x1:x2],
                    "x1": x1,
                    "y1": y1,
                }
            )

    return result


def nms(boxes, scores, threshold=0.50):
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

            if iou(
                boxes[i],
                boxes[j],
            ) < threshold:
                remaining.append(j)

        order = np.asarray(
            remaining,
            dtype=np.int64,
        )

    return keep_boxes, keep_scores


def run_yolo(
    model,
    image,
    device,
):
    boxes = []
    scores = []

    start = time.perf_counter()

    for tile in make_tiles(image):
        result = model.predict(
            source=tile["image"],
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
            result.boxes.xyxy
            .detach()
            .cpu()
            .numpy()
        )

        tile_scores = (
            result.boxes.conf
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

            scores.append(float(score))

    boxes, scores = nms(
        boxes,
        scores,
        YOLO_NMS,
    )

    if device == "mps":
        torch.mps.synchronize()

    latency = (
        time.perf_counter() - start
    ) * 1000.0

    return boxes, scores, latency


def padded_crop(image, bbox):
    x1, y1, x2, y2 = map(
        float,
        bbox,
    )

    w = x2 - x1
    h = y2 - y1

    px = w * CROP_PADDING
    py = h * CROP_PADDING

    image_h, image_w = image.shape[:2]

    x1 = max(
        0,
        int(round(x1 - px)),
    )

    y1 = max(
        0,
        int(round(y1 - py)),
    )

    x2 = min(
        image_w,
        int(round(x2 + px)),
    )

    y2 = min(
        image_h,
        int(round(y2 + py)),
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


def normalize_rows(array):
    norms = np.linalg.norm(
        array,
        axis=1,
        keepdims=True,
    )

    return array / np.maximum(
        norms,
        1e-12,
    )


@torch.inference_mode()
def embed_batch(
    crops,
    processor,
    model,
    device,
):
    if not crops:
        return (
            np.empty(
                (0, model.config.hidden_size),
                dtype=np.float32,
            ),
            0.0,
        )

    inputs = processor(
        images=crops,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(device)
        for key, value
        in inputs.items()
    }

    if device.type == "mps":
        torch.mps.synchronize()

    start = time.perf_counter()

    output = model(**inputs)

    if device.type == "mps":
        torch.mps.synchronize()

    latency = (
        time.perf_counter() - start
    ) * 1000.0

    embeddings = (
        output.last_hidden_state[:, 0, :]
        .float()
        .cpu()
        .numpy()
    )

    return (
        normalize_rows(
            embeddings
        ),
        latency,
    )


def build_prototypes(
    processor,
    dino,
    device,
):
    grouped = defaultdict(list)

    annotation_paths = sorted(
        ANNOTATION_DIR.glob(
            "frame_*.json"
        )
    )

    for annotation_path in annotation_paths:
        data = json.loads(
            annotation_path.read_text()
        )

        frame = int(
            data["frame"]
        )

        if frame > TRAIN_MAX_FRAME:
            continue

        image = load_frame(frame)

        for annotation in data[
            "annotations"
        ]:
            crop = padded_crop(
                image,
                bbox_l0(
                    annotation["bbox"]
                ),
            )

            if crop is None:
                continue

            grouped[
                annotation["object_id"]
            ].append(crop)

    class_names = sorted(
        grouped
    )

    prototypes = []

    print("Building DINO prototypes...")

    for class_name in class_names:
        embeddings, _ = embed_batch(
            grouped[class_name],
            processor,
            dino,
            device,
        )

        prototype = np.mean(
            embeddings,
            axis=0,
        )

        prototype /= max(
            np.linalg.norm(
                prototype
            ),
            1e-12,
        )

        prototypes.append(
            prototype
        )

        print(
            f"  {class_name:20s} "
            f"{len(grouped[class_name])}"
        )

    return (
        class_names,
        np.stack(
            prototypes
        ),
    )


def average_precision(
    predictions,
    gt_by_frame,
):
    total_gt = sum(
        len(v)
        for v in gt_by_frame.values()
    )

    if total_gt == 0:
        return None

    predictions = sorted(
        predictions,
        key=lambda row:
            row["score"],
        reverse=True,
    )

    matched = {
        frame:
            np.zeros(
                len(boxes),
                dtype=bool,
            )
        for frame, boxes
        in gt_by_frame.items()
    }

    tp = []
    fp = []

    for prediction in predictions:
        frame = prediction["frame"]
        box = prediction["bbox"]

        gt_boxes = gt_by_frame.get(
            frame,
            [],
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

            score = iou(
                box,
                gt_box,
            )

            if score > best_iou:
                best_iou = score
                best_index = idx

        if (
            best_index >= 0
            and best_iou >= 0.50
        ):
            matched[
                frame
            ][best_index] = True

            tp.append(1.0)
            fp.append(0.0)

        else:
            tp.append(0.0)
            fp.append(1.0)

    if not predictions:
        return 0.0

    tp = np.cumsum(
        np.asarray(tp)
    )

    fp = np.cumsum(
        np.asarray(fp)
    )

    recall = (
        tp / total_gt
    )

    precision = (
        tp / np.maximum(
            tp + fp,
            1e-12,
        )
    )

    # COCO-style 101-point interpolation at one IoU threshold.
    recall_points = np.linspace(
        0,
        1,
        101,
    )

    interpolated = []

    for recall_point in recall_points:
        valid = precision[
            recall >= recall_point
        ]

        interpolated.append(
            np.max(valid)
            if len(valid)
            else 0.0
        )

    return float(
        np.mean(
            interpolated
        )
    )


def main():
    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    device_name = (
        "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    torch_device = torch.device(
        device_name
    )

    print(
        f"Device: {device_name}"
    )

    yolo = YOLO(
        str(YOLO_MODEL)
    )

    processor = (
        AutoImageProcessor
        .from_pretrained(
            DINO_MODEL
        )
    )

    dino = (
        AutoModel
        .from_pretrained(
            DINO_MODEL
        )
    )

    dino.eval()
    dino.to(
        torch_device
    )

    class_names, prototypes = (
        build_prototypes(
            processor,
            dino,
            torch_device,
        )
    )

    class_index = {
        name: index
        for index, name
        in enumerate(class_names)
    }

    # Warm YOLO.
    warm_image = load_frame(
        min(HOLDOUT_FRAMES)
    )

    run_yolo(
        yolo,
        warm_image,
        device_name,
    )

    predictions_by_class = {
        cls: []
        for cls in class_names
    }

    gt_by_class = {
        cls: defaultdict(list)
        for cls in class_names
    }

    yolo_latencies = []
    dino_latencies = []
    total_latencies = []
    proposal_counts = []

    for frame in sorted(
        HOLDOUT_FRAMES
    ):
        image = load_frame(frame)

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
            cls = annotation[
                "object_id"
            ]

            gt_by_class[
                cls
            ][frame].append(
                bbox_l0(
                    annotation["bbox"]
                )
            )

        total_start = (
            time.perf_counter()
        )

        boxes, yolo_scores, yolo_latency = (
            run_yolo(
                yolo,
                image,
                device_name,
            )
        )

        crops = []
        valid_boxes = []
        valid_yolo_scores = []

        for box, score in zip(
            boxes,
            yolo_scores,
        ):
            crop = padded_crop(
                image,
                box,
            )

            if crop is None:
                continue

            crops.append(crop)
            valid_boxes.append(box)
            valid_yolo_scores.append(
                score
            )

        embeddings, dino_latency = (
            embed_batch(
                crops,
                processor,
                dino,
                torch_device,
            )
        )

        proposal_counts.append(
            len(valid_boxes)
        )

        yolo_latencies.append(
            yolo_latency
        )

        dino_latencies.append(
            dino_latency
        )

        if len(embeddings):
            similarities = (
                embeddings
                @ prototypes.T
            )

            for idx, box in enumerate(
                valid_boxes
            ):
                ranked = np.argsort(
                    similarities[idx]
                )[::-1]

                best_index = int(
                    ranked[0]
                )

                second_index = int(
                    ranked[1]
                )

                predicted_class = (
                    class_names[
                        best_index
                    ]
                )

                similarity = float(
                    similarities[
                        idx,
                        best_index
                    ]
                )

                margin = float(
                    similarities[
                        idx,
                        best_index
                    ]
                    - similarities[
                        idx,
                        second_index
                    ]
                )

                # Combined ranking confidence.
                score = (
                    similarity
                    * max(
                        valid_yolo_scores[
                            idx
                        ],
                        1e-6,
                    )
                )

                predictions_by_class[
                    predicted_class
                ].append(
                    {
                        "frame":
                            frame,
                        "bbox":
                            box,
                        "score":
                            score,
                        "similarity":
                            similarity,
                        "margin":
                            margin,
                    }
                )

        total_latency = (
            time.perf_counter()
            - total_start
        ) * 1000.0

        total_latencies.append(
            total_latency
        )

        print(
            f"Frame {frame}: "
            f"proposals={len(valid_boxes):2d}, "
            f"YOLO={yolo_latency:6.1f} ms, "
            f"DINO={dino_latency:6.1f} ms, "
            f"total={total_latency:6.1f} ms"
        )

    ap = {}

    for class_name in class_names:
        ap[class_name] = (
            average_precision(
                predictions_by_class[
                    class_name
                ],
                gt_by_class[
                    class_name
                ],
            )
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
    print("EXP-D012 HOLDOUT RESULTS")
    print("=" * 72)

    print(
        f"Macro mAP@0.50: "
        f"{macro_map:.4f}"
    )

    print()
    print("AP@0.50 by class")

    for class_name in class_names:
        value = ap[
            class_name
        ]

        if value is None:
            continue

        print(
            f"  {class_name:20s} "
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
        f"  DINO batch median: "
        f"{statistics.median(dino_latencies):.1f} ms"
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

        "latency_ms": {
            "yolo_median":
                float(
                    statistics.median(
                        yolo_latencies
                    )
                ),

            "dino_median":
                float(
                    statistics.median(
                        dino_latencies
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

        "mean_proposals":
            float(
                statistics.mean(
                    proposal_counts
                )
            ),
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
