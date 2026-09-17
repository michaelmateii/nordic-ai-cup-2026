#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]

BASE_SCRIPT = (
    ROOT
    / "drone"
    / "scripts"
    / "evaluate_yolo_mobilenet_pipeline.py"
)

MODEL_PATH = (
    ROOT
    / "drone"
    / "artifacts"
    / "exp_d013"
    / "classifier"
    / "mobilenetv3_small_best.pt"
)

META_PATH = (
    ROOT
    / "drone"
    / "artifacts"
    / "exp_d013"
    / "classifier"
    / "metadata.json"
)

YOLO_PATH = (
    ROOT
    / "drone"
    / "artifacts"
    / "exp_d010"
    / "runs"
    / "yolo11n_tiled_objectness"
    / "weights"
    / "best.pt"
)

OUTPUT_DIR = (
    ROOT
    / "drone"
    / "artifacts"
    / "exp_d015"
)

OUTPUT_JSON = (
    OUTPUT_DIR
    / "runtime_sweep.json"
)

CONFIGS = [
    ("mps", 224),
    ("mps", 160),
    ("mps", 128),
    ("cpu", 224),
    ("cpu", 160),
    ("cpu", 128),
]


def load_base_module():
    spec = importlib.util.spec_from_file_location(
        "d014",
        BASE_SCRIPT,
    )

    module = importlib.util.module_from_spec(
        spec
    )

    assert spec.loader is not None
    spec.loader.exec_module(module)

    return module


def build_transform(size: int):
    return transforms.Compose(
        [
            transforms.Resize(
                (size, size)
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


def build_classifier(
    num_classes: int,
    device: torch.device,
):
    model = timm.create_model(
        "mobilenetv3_small_100",
        pretrained=False,
        num_classes=num_classes,
    )

    state = torch.load(
        MODEL_PATH,
        map_location="cpu",
    )

    model.load_state_dict(
        state
    )

    model.eval()
    model.to(device)

    return model


@torch.inference_mode()
def classify(
    crops,
    model,
    transform,
    device,
):
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
    ).to(device)

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

    latency_ms = (
        time.perf_counter()
        - start
    ) * 1000.0

    return (
        probabilities
        .float()
        .cpu()
        .numpy(),
        latency_ms,
    )


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    d014 = load_base_module()

    metadata = json.loads(
        META_PATH.read_text()
    )

    classes = metadata[
        "classes"
    ]

    print(
        "Generating fixed YOLO proposals once..."
    )

    yolo_device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    yolo = YOLO(
        str(YOLO_PATH)
    )

    fixed_frames = []

    yolo_latencies = []

    # Warm YOLO.
    warm_image = d014.load_frame(
        22
    )

    d014.run_yolo(
        yolo,
        warm_image,
        yolo_device,
    )

    for frame in sorted(
        d014.HOLDOUT_FRAMES
    ):
        image = d014.load_frame(
            frame
        )

        boxes, scores, latency = (
            d014.run_yolo(
                yolo,
                image,
                yolo_device,
            )
        )

        crops = []
        valid_boxes = []
        valid_scores = []

        for box, score in zip(
            boxes,
            scores,
        ):
            crop = d014.crop_rgb(
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

            valid_scores.append(
                score
            )

        fixed_frames.append(
            {
                "frame":
                    frame,
                "crops":
                    crops,
                "boxes":
                    valid_boxes,
                "detector_scores":
                    valid_scores,
            }
        )

        yolo_latencies.append(
            latency
        )

    gt_by_class = {
        cls: defaultdict(list)
        for cls in classes
    }

    for frame in sorted(
        d014.HOLDOUT_FRAMES
    ):
        annotation_path = (
            d014.ANNOTATION_DIR
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
                d014.bbox_l0(
                    annotation[
                        "bbox"
                    ]
                )
            )

    results = {}

    for device_name, size in CONFIGS:
        if (
            device_name == "mps"
            and not torch.backends.mps.is_available()
        ):
            continue

        print()
        print("=" * 72)

        print(
            f"{device_name.upper()} "
            f"{size}x{size}"
        )

        print("=" * 72)

        device = torch.device(
            device_name
        )

        model = build_classifier(
            len(classes),
            device,
        )

        transform = build_transform(
            size
        )

        # Warm classifier with realistic batch.
        classify(
            fixed_frames[0][
                "crops"
            ],
            model,
            transform,
            device,
        )

        predictions = {
            cls: []
            for cls in classes
        }

        latencies = []

        for frame_data in (
            fixed_frames
        ):
            probabilities, latency = classify(
                frame_data[
                    "crops"
                ],
                model,
                transform,
                device,
            )

            latencies.append(
                latency
            )

            for index, box in enumerate(
                frame_data[
                    "boxes"
                ]
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
                        frame_data[
                            "detector_scores"
                        ][index],
                        1e-6,
                    )
                )

                predictions[
                    class_name
                ].append(
                    {
                        "frame":
                            frame_data[
                                "frame"
                            ],
                        "bbox":
                            box,
                        "score":
                            score,
                    }
                )

        aps = {}

        for cls in classes:
            aps[cls] = d014.average_precision(
                predictions[
                    cls
                ],
                gt_by_class[
                    cls
                ],
            )

        scored = [
            value
            for value in aps.values()
            if value is not None
        ]

        macro_map = float(
            np.mean(
                scored
            )
        )

        median_classifier = float(
            statistics.median(
                latencies
            )
        )

        max_classifier = float(
            max(
                latencies
            )
        )

        estimated_total = (
            statistics.median(
                yolo_latencies
            )
            + median_classifier
        )

        result = {
            "device":
                device_name,

            "input_size":
                size,

            "macro_map_50":
                macro_map,

            "classifier_latency_ms": {
                "median":
                    median_classifier,
                "max":
                    max_classifier,
            },

            "estimated_total_median_ms":
                float(
                    estimated_total
                ),

            "ap_50":
                aps,
        }

        key = (
            f"{device_name}_{size}"
        )

        results[key] = result

        print(
            f"Macro mAP@0.50: "
            f"{macro_map:.4f}"
        )

        print(
            f"Classifier median: "
            f"{median_classifier:.1f} ms"
        )

        print(
            f"Classifier max: "
            f"{max_classifier:.1f} ms"
        )

        print(
            f"Estimated total median: "
            f"{estimated_total:.1f} ms"
        )

        # Free device memory before next config.
        del model

        if device_name == "mps":
            torch.mps.empty_cache()

    OUTPUT_JSON.write_text(
        json.dumps(
            results,
            indent=2,
        )
    )

    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)

    for key, result in results.items():
        print(
            f"{key:10s} "
            f"mAP={result['macro_map_50']:.4f} "
            f"classifier="
            f"{result['classifier_latency_ms']['median']:.1f} ms "
            f"estimated_total="
            f"{result['estimated_total_median_ms']:.1f} ms"
        )

    print()
    print(
        f"Saved: {OUTPUT_JSON}"
    )


if __name__ == "__main__":
    main()
