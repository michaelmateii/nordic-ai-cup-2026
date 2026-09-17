#!/usr/bin/env python3

from __future__ import annotations

import json
import statistics
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel


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

ARTIFACT_DIR = (
    ROOT
    / "drone"
    / "artifacts"
    / "exp_d006"
)

RESULT_PATH = (
    ARTIFACT_DIR
    / "embedding_scale_results.json"
)

MODEL_NAME = "facebook/dinov2-small"

# Effective source->camera-image scaling.
LEVEL_SCALES = {
    "L0": 0.25,
    "L1": 0.50,
    "L2": 1.00,
}

PADDING_FRACTION = 0.20


def device_for_torch() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def find_image(frame: int) -> Path:
    candidates = [
        IMAGE_DIR / f"frame_{frame:06d}.png",
        IMAGE_DIR / f"frame_{frame:06d}.jpg",
        IMAGE_DIR / f"frame_{frame:06d}.jpeg",
    ]

    for path in candidates:
        if path.exists():
            return path

    matches = sorted(
        IMAGE_DIR.glob(f"*{frame:06d}*")
    )

    if matches:
        return matches[0]

    raise FileNotFoundError(frame)


def resize_for_level(
    image: np.ndarray,
    scale: float,
) -> np.ndarray:

    if scale == 1.0:
        return image

    height, width = image.shape[:2]

    return cv2.resize(
        image,
        (
            int(round(width * scale)),
            int(round(height * scale)),
        ),
        interpolation=cv2.INTER_AREA,
    )


def scale_bbox(
    bbox: list[int | float],
    scale: float,
) -> list[float]:

    return [
        float(value) * scale
        for value in bbox
    ]


def padded_crop(
    image: np.ndarray,
    bbox: list[float],
) -> np.ndarray:

    x1, y1, x2, y2 = bbox

    width = x2 - x1
    height = y2 - y1

    pad_x = width * PADDING_FRACTION
    pad_y = height * PADDING_FRACTION

    image_h, image_w = image.shape[:2]

    x1 = max(
        0,
        int(round(x1 - pad_x)),
    )
    y1 = max(
        0,
        int(round(y1 - pad_y)),
    )
    x2 = min(
        image_w,
        int(round(x2 + pad_x)),
    )
    y2 = min(
        image_h,
        int(round(y2 + pad_y)),
    )

    crop = image[
        y1:y2,
        x1:x2,
    ]

    if crop.size == 0:
        raise RuntimeError(
            f"Empty crop: {bbox}"
        )

    return cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2RGB,
    )


def normalize(
    vector: np.ndarray,
) -> np.ndarray:

    denominator = np.linalg.norm(
        vector
    )

    if denominator == 0:
        return vector

    return vector / denominator


@torch.inference_mode()
def embed(
    image_rgb: np.ndarray,
    processor,
    model,
    device: torch.device,
) -> tuple[np.ndarray, float]:

    inputs = processor(
        images=image_rgb,
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

    latency_ms = (
        time.perf_counter()
        - start
    ) * 1000.0

    vector = (
        output.last_hidden_state[
            :,
            0,
            :,
        ]
        .float()
        .cpu()
        .numpy()[0]
    )

    return (
        normalize(vector),
        latency_ms,
    )


def cosine(
    a: np.ndarray,
    b: np.ndarray,
) -> float:

    return float(
        np.dot(a, b)
        / (
            np.linalg.norm(a)
            * np.linalg.norm(b)
            + 1e-12
        )
    )


def classify_leave_one_out(
    samples: list[dict],
) -> dict:

    class_names = sorted(
        {
            sample["class"]
            for sample in samples
        }
    )

    correct = 0

    per_class_total = Counter()
    per_class_correct = Counter()

    margins = []

    predictions = []

    for query_index, query in enumerate(
        samples
    ):

        prototypes = {}

        for class_name in class_names:

            embeddings = [
                candidate["embedding"]
                for index, candidate
                in enumerate(samples)
                if index != query_index
                and candidate["class"]
                == class_name
            ]

            if not embeddings:
                continue

            prototype = np.mean(
                np.stack(embeddings),
                axis=0,
            )

            prototypes[
                class_name
            ] = normalize(
                prototype
            )

        scores = {
            class_name: cosine(
                query["embedding"],
                prototype,
            )
            for class_name, prototype
            in prototypes.items()
        }

        ranked = sorted(
            scores.items(),
            key=lambda item:
                item[1],
            reverse=True,
        )

        predicted = ranked[0][0]

        margin = (
            ranked[0][1]
            - ranked[1][1]
        )

        true_class = query["class"]

        success = (
            predicted
            == true_class
        )

        per_class_total[
            true_class
        ] += 1

        if success:
            correct += 1
            per_class_correct[
                true_class
            ] += 1

        margins.append(
            margin
        )

        predictions.append(
            {
                "frame":
                    query["frame"],
                "true":
                    true_class,
                "predicted":
                    predicted,
                "correct":
                    bool(success),
                "margin":
                    float(margin),
            }
        )

    accuracy = (
        correct
        / len(samples)
    )

    per_class = {}

    for class_name in sorted(
        per_class_total
    ):

        total = per_class_total[
            class_name
        ]

        class_correct = (
            per_class_correct[
                class_name
            ]
        )

        per_class[
            class_name
        ] = {
            "correct":
                class_correct,
            "total":
                total,
            "accuracy":
                class_correct
                / total,
        }

    return {
        "accuracy":
            accuracy,
        "median_margin":
            float(
                statistics.median(
                    margins
                )
            ),
        "per_class":
            per_class,
        "predictions":
            predictions,
    }


def main() -> None:

    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = device_for_torch()

    print(
        f"Device: {device}"
    )
    print(
        f"Model:  {MODEL_NAME}"
    )

    print()
    print(
        "Loading model..."
    )

    processor = (
        AutoImageProcessor
        .from_pretrained(
            MODEL_NAME
        )
    )

    model = (
        AutoModel
        .from_pretrained(
            MODEL_NAME
        )
    )

    model.eval()
    model.to(device)

    annotation_paths = sorted(
        ANNOTATION_DIR.glob(
            "frame_*.json"
        )
    )

    level_results = {}

    for level_name, scale in (
        LEVEL_SCALES.items()
    ):

        print()
        print("=" * 72)
        print(
            f"{level_name} "
            f"(scale={scale})"
        )
        print("=" * 72)

        samples = []
        latencies = []

        for annotation_path in (
            annotation_paths
        ):

            with annotation_path.open(
                "r",
                encoding="utf-8",
            ) as f:
                data = json.load(f)

            frame = int(
                data["frame"]
            )

            original = cv2.imread(
                str(
                    find_image(
                        frame
                    )
                ),
                cv2.IMREAD_COLOR,
            )

            scaled = resize_for_level(
                original,
                scale,
            )

            for annotation in (
                data["annotations"]
            ):

                bbox = scale_bbox(
                    annotation["bbox"],
                    scale,
                )

                crop = padded_crop(
                    scaled,
                    bbox,
                )

                embedding, latency = embed(
                    crop,
                    processor,
                    model,
                    device,
                )

                latencies.append(
                    latency
                )

                samples.append(
                    {
                        "frame":
                            frame,
                        "class":
                            annotation[
                                "object_id"
                            ],
                        "embedding":
                            embedding,
                    }
                )

        result = (
            classify_leave_one_out(
                samples
            )
        )

        result[
            "latency_ms"
        ] = {
            "median":
                float(
                    np.median(
                        latencies
                    )
                ),
            "p95":
                float(
                    np.percentile(
                        latencies,
                        95,
                    )
                ),
        }

        level_results[
            level_name
        ] = result

        print(
            f"Accuracy:      "
            f"{result['accuracy']:.4f}"
        )

        print(
            f"Median margin: "
            f"{result['median_margin']:.4f}"
        )

        print(
            f"Latency median:"
            f" {result['latency_ms']['median']:.1f} ms"
        )

        print()
        print(
            "Per-class accuracy"
        )

        for class_name, stats in (
            result[
                "per_class"
            ].items()
        ):

            print(
                f"  {class_name:20s} "
                f"{stats['correct']:3d}/"
                f"{stats['total']:<3d} "
                f"{stats['accuracy']:.3f}"
            )

    output = {
        "model":
            MODEL_NAME,
        "device":
            str(device),
        "levels":
            level_results,
    }

    # Remove arrays before JSON.
    for level in output[
        "levels"
    ].values():
        for prediction in level[
            "predictions"
        ]:
            pass

    with RESULT_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            output,
            f,
            indent=2,
        )

    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)

    for level_name in (
        LEVEL_SCALES
    ):
        result = level_results[
            level_name
        ]

        print(
            f"{level_name}: "
            f"accuracy="
            f"{result['accuracy']:.4f}, "
            f"margin="
            f"{result['median_margin']:.4f}"
        )

    print()
    print(
        f"Saved: {RESULT_PATH}"
    )


if __name__ == "__main__":
    main()
