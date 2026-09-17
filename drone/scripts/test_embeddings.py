#!/usr/bin/env python3

from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
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

ARTIFACT_DIR = ROOT / "drone" / "artifacts" / "exp_d003"
RESULT_PATH = ARTIFACT_DIR / "embedding_results.json"

MODEL_NAME = "facebook/dinov2-small"

PADDING_FRACTION = 0.20


def pick_device() -> torch.device:
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

    raise FileNotFoundError(
        f"No image found for frame {frame}"
    )


def padded_crop(
    image: np.ndarray,
    bbox: list[int | float],
    padding_fraction: float = PADDING_FRACTION,
) -> np.ndarray:

    x1, y1, x2, y2 = map(float, bbox)

    width = x2 - x1
    height = y2 - y1

    pad_x = width * padding_fraction
    pad_y = height * padding_fraction

    h, w = image.shape[:2]

    x1 = max(0, int(round(x1 - pad_x)))
    y1 = max(0, int(round(y1 - pad_y)))
    x2 = min(w, int(round(x2 + pad_x)))
    y2 = min(h, int(round(y2 + pad_y)))

    crop = image[y1:y2, x1:x2]

    if crop.size == 0:
        raise ValueError(
            f"Empty crop for bbox {bbox}"
        )

    # OpenCV -> RGB
    return cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2RGB,
    )


def l2_normalize(
    vector: np.ndarray,
) -> np.ndarray:

    norm = np.linalg.norm(vector)

    if norm == 0:
        return vector

    return vector / norm


@torch.inference_mode()
def embed_crop(
    crop_rgb: np.ndarray,
    processor,
    model,
    device: torch.device,
) -> tuple[np.ndarray, float]:

    inputs = processor(
        images=crop_rgb,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    if device.type == "mps":
        torch.mps.synchronize()

    start = time.perf_counter()

    outputs = model(**inputs)

    if device.type == "mps":
        torch.mps.synchronize()

    elapsed_ms = (
        time.perf_counter() - start
    ) * 1000.0

    # DINOv2 CLS token.
    embedding = (
        outputs.last_hidden_state[:, 0, :]
        .float()
        .cpu()
        .numpy()[0]
    )

    return (
        l2_normalize(embedding),
        elapsed_ms,
    )


def cosine_similarity(
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


def main() -> None:

    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = pick_device()

    print(f"Device: {device}")
    print(f"Model:  {MODEL_NAME}")
    print()

    print("Loading pretrained model...")

    processor = AutoImageProcessor.from_pretrained(
        MODEL_NAME
    )

    model = AutoModel.from_pretrained(
        MODEL_NAME
    )

    model.eval()
    model.to(device)

    samples = []

    annotation_paths = sorted(
        ANNOTATION_DIR.glob(
            "frame_*.json"
        )
    )

    print(
        f"Annotation frames: "
        f"{len(annotation_paths)}"
    )

    print("Extracting embeddings...")

    latencies = []

    for annotation_path in annotation_paths:

        with annotation_path.open(
            "r",
            encoding="utf-8",
        ) as f:
            data = json.load(f)

        frame = int(data["frame"])

        image_path = find_image(frame)

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR,
        )

        if image is None:
            raise RuntimeError(
                f"Failed to read {image_path}"
            )

        for annotation in data["annotations"]:

            class_name = annotation[
                "object_id"
            ]

            bbox = annotation["bbox"]

            crop = padded_crop(
                image,
                bbox,
            )

            embedding, latency_ms = (
                embed_crop(
                    crop,
                    processor,
                    model,
                    device,
                )
            )

            latencies.append(
                latency_ms
            )

            samples.append(
                {
                    "frame": frame,
                    "class": class_name,
                    "bbox": bbox,
                    "embedding": embedding,
                }
            )

    print(
        f"Embeddings: {len(samples)}"
    )

    # -----------------------------------------------------
    # Leave-one-observation-out nearest class prototype
    # -----------------------------------------------------

    predictions = []

    correct = 0

    per_class_total = Counter()
    per_class_correct = Counter()

    margins = []

    for index, query in enumerate(samples):

        prototypes = {}

        for class_name in sorted(
            {
                sample["class"]
                for sample in samples
            }
        ):

            candidates = [
                sample["embedding"]
                for j, sample in enumerate(samples)
                if j != index
                and sample["class"] == class_name
            ]

            if not candidates:
                continue

            prototype = np.mean(
                np.stack(candidates),
                axis=0,
            )

            prototypes[class_name] = (
                l2_normalize(prototype)
            )

        similarities = {
            class_name: cosine_similarity(
                query["embedding"],
                prototype,
            )
            for class_name, prototype
            in prototypes.items()
        }

        ranked = sorted(
            similarities.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        predicted_class = ranked[0][0]

        best_score = ranked[0][1]

        second_score = (
            ranked[1][1]
            if len(ranked) > 1
            else -1.0
        )

        margin = (
            best_score - second_score
        )

        margins.append(margin)

        true_class = query["class"]

        is_correct = (
            predicted_class == true_class
        )

        per_class_total[true_class] += 1

        if is_correct:
            correct += 1
            per_class_correct[
                true_class
            ] += 1

        predictions.append(
            {
                "frame": query["frame"],
                "true_class": true_class,
                "predicted_class":
                    predicted_class,
                "correct": is_correct,
                "best_similarity":
                    best_score,
                "second_similarity":
                    second_score,
                "margin":
                    margin,
            }
        )

    accuracy = (
        correct / len(predictions)
        if predictions
        else 0.0
    )

    per_class_accuracy = {}

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

        per_class_accuracy[
            class_name
        ] = {
            "correct": class_correct,
            "total": total,
            "accuracy": (
                class_correct / total
            ),
        }

    latency_array = np.asarray(
        latencies,
        dtype=np.float64,
    )

    result = {
        "model": MODEL_NAME,
        "device": str(device),
        "samples": len(samples),
        "overall_accuracy": accuracy,
        "per_class_accuracy":
            per_class_accuracy,
        "embedding_latency_ms": {
            "mean": float(
                np.mean(latency_array)
            ),
            "median": float(
                np.median(latency_array)
            ),
            "p95": float(
                np.percentile(
                    latency_array,
                    95,
                )
            ),
            "max": float(
                np.max(latency_array)
            ),
        },
        "margin": {
            "median": float(
                np.median(margins)
            ),
            "p10": float(
                np.percentile(
                    margins,
                    10,
                )
            ),
        },
        "predictions":
            predictions,
    }

    with RESULT_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            result,
            f,
            indent=2,
        )

    print()
    print("=" * 72)
    print("EXP-D003 RESULTS")
    print("=" * 72)

    print(
        f"Samples:          "
        f"{len(samples)}"
    )

    print(
        f"Overall accuracy: "
        f"{accuracy:.4f}"
    )

    print()
    print("Per-class accuracy")

    for class_name, stats in (
        per_class_accuracy.items()
    ):
        print(
            f"  {class_name:20s} "
            f"{stats['correct']:3d}/"
            f"{stats['total']:<3d} "
            f"{stats['accuracy']:.3f}"
        )

    print()
    print("Embedding inference latency")

    print(
        f"  mean:   "
        f"{result['embedding_latency_ms']['mean']:.1f} ms"
    )

    print(
        f"  median: "
        f"{result['embedding_latency_ms']['median']:.1f} ms"
    )

    print(
        f"  p95:    "
        f"{result['embedding_latency_ms']['p95']:.1f} ms"
    )

    print(
        f"  max:    "
        f"{result['embedding_latency_ms']['max']:.1f} ms"
    )

    print()
    print(
        f"Median top-1 margin: "
        f"{result['margin']['median']:.4f}"
    )

    print(
        f"P10 top-1 margin:    "
        f"{result['margin']['p10']:.4f}"
    )

    print()
    print(
        f"Saved: {RESULT_PATH}"
    )


if __name__ == "__main__":
    main()
