#!/usr/bin/env python3

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel


ROOT = Path(__file__).resolve().parents[2]

OFFICIAL = (
    ROOT
    / "drone"
    / "reference"
    / "official-drone-flyby"
)

HELSINKI_IMAGES = (
    OFFICIAL
    / "src"
    / "helsinki"
    / "images"
)

HELSINKI_ANN = (
    OFFICIAL
    / "src"
    / "helsinki"
    / "annotations"
)

VALIDATION = (
    ROOT
    / "drone"
    / "captures"
    / "3224a582bfbf4273a028497662b7aa7c"
    / "frames"
)

OUT = (
    ROOT
    / "drone"
    / "artifacts"
    / "exp_d026a"
)

MODEL_NAME = "facebook/dinov2-small"

# Original received L1 view.
VIEW_W = 960
VIEW_H = 540

# Same 16:9 aspect ratio and divisible by DINO patch size 14.
DINO_W = 896
DINO_H = 504

# Cheap screening set.
N_VALIDATION_FRAMES = 16

TOP_K_PER_CLASS = 5

# Prototype crop context.
PROTOTYPE_CONTEXT = 0.15

# Display crop context around predicted box.
DISPLAY_CONTEXT = 2.0


IMAGENET_MEAN = torch.tensor(
    [0.485, 0.456, 0.406]
).view(1, 3, 1, 1)

IMAGENET_STD = torch.tensor(
    [0.229, 0.224, 0.225]
).view(1, 3, 1, 1)


def find_image(
    frame_number: int,
) -> Path:
    for suffix in (
        ".png",
        ".jpg",
        ".jpeg",
    ):
        path = (
            HELSINKI_IMAGES
            / (
                f"frame_{frame_number:06d}"
                f"{suffix}"
            )
        )

        if path.exists():
            return path

    raise FileNotFoundError(
        f"No image for Helsinki frame "
        f"{frame_number}"
    )


def preprocess(
    image_bgr: np.ndarray,
    width: int | None = None,
    height: int | None = None,
):
    image = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB,
    )

    if (
        width is not None
        and height is not None
    ):
        image = cv2.resize(
            image,
            (width, height),
            interpolation=cv2.INTER_AREA,
        )

    array = (
        image.astype(np.float32)
        / 255.0
    )

    tensor = torch.from_numpy(
        array
    ).permute(
        2, 0, 1
    ).unsqueeze(0)

    return tensor


def normalize_input(
    tensor,
    device,
):
    mean = IMAGENET_MEAN.to(device)
    std = IMAGENET_STD.to(device)

    tensor = tensor.to(device)

    return (
        tensor - mean
    ) / std


def crop_with_context(
    image,
    bbox,
    context,
):
    x1, y1, x2, y2 = map(
        float,
        bbox,
    )

    w = x2 - x1
    h = y2 - y1

    px = w * context
    py = h * context

    ih, iw = image.shape[:2]

    x1 = max(
        0,
        int(round(x1 - px)),
    )

    y1 = max(
        0,
        int(round(y1 - py)),
    )

    x2 = min(
        iw,
        int(round(x2 + px)),
    )

    y2 = min(
        ih,
        int(round(y2 + py)),
    )

    if x2 <= x1 or y2 <= y1:
        return None

    return image[
        y1:y2,
        x1:x2,
    ]


def resize_crop_for_dino(
    crop,
    patch_size,
):
    h, w = crop.shape[:2]

    # Keep aspect ratio.
    scale = min(
        224.0 / max(w, h),
        1.0 if max(w, h) >= 112 else 224.0 / max(w, h),
    )

    target_w = max(
        patch_size,
        int(round(w * scale)),
    )

    target_h = max(
        patch_size,
        int(round(h * scale)),
    )

    # DINO needs spatial dimensions divisible by patch size.
    target_w = max(
        patch_size,
        int(
            round(
                target_w / patch_size
            )
            * patch_size
        ),
    )

    target_h = max(
        patch_size,
        int(
            round(
                target_h / patch_size
            )
            * patch_size
        ),
    )

    target_w = min(
        224,
        target_w,
    )

    target_h = min(
        224,
        target_h,
    )

    target_w = max(
        patch_size,
        target_w,
    )

    target_h = max(
        patch_size,
        target_h,
    )

    return cv2.resize(
        crop,
        (
            target_w,
            target_h,
        ),
        interpolation=(
            cv2.INTER_CUBIC
            if scale > 1
            else cv2.INTER_AREA
        ),
    )


def patch_mean_embedding(
    model,
    crop,
    device,
    patch_size,
):
    crop = resize_crop_for_dino(
        crop,
        patch_size,
    )

    tensor = preprocess(
        crop
    )

    tensor = normalize_input(
        tensor,
        device,
    )

    with torch.inference_mode():
        output = model(
            pixel_values=tensor
        )

    # Remove CLS token.
    features = (
        output.last_hidden_state[
            :,
            1:,
            :,
        ]
    )

    feature = features.mean(
        dim=1
    )

    feature = F.normalize(
        feature,
        dim=-1,
    )

    return feature[0]


def load_helsinki_samples():
    by_class = defaultdict(list)

    for path in sorted(
        HELSINKI_ANN.glob("*.json")
    ):
        data = json.loads(
            path.read_text()
        )

        frame = int(
            data["frame"]
        )

        for ann in data[
            "annotations"
        ]:
            bbox = ann["bbox"]

            by_class[
                ann["object_id"]
            ].append(
                {
                    "frame":
                        frame,
                    "bbox":
                        bbox,
                    "width":
                        float(
                            bbox[2]
                            - bbox[0]
                        ),
                    "height":
                        float(
                            bbox[3]
                            - bbox[1]
                        ),
                }
            )

    return by_class


def build_prototypes(
    model,
    device,
    patch_size,
):
    samples = (
        load_helsinki_samples()
    )

    prototypes = {}
    dimensions = {}

    print(
        "Building Helsinki DINO prototypes..."
    )

    for class_name in sorted(
        samples
    ):
        embeddings = []

        widths = []
        heights = []

        for sample in samples[
            class_name
        ]:
            image = cv2.imread(
                str(
                    find_image(
                        sample["frame"]
                    )
                )
            )

            crop = crop_with_context(
                image,
                sample["bbox"],
                PROTOTYPE_CONTEXT,
            )

            if crop is None:
                continue

            feature = (
                patch_mean_embedding(
                    model,
                    crop,
                    device,
                    patch_size,
                )
            )

            embeddings.append(
                feature
            )

            widths.append(
                sample["width"]
            )

            heights.append(
                sample["height"]
            )

        matrix = torch.stack(
            embeddings
        )

        prototype = matrix.mean(
            dim=0
        )

        prototype = F.normalize(
            prototype,
            dim=0,
        )

        prototypes[
            class_name
        ] = prototype

        # Source pixels -> L1 transmitted pixels is 0.5.
        dimensions[
            class_name
        ] = (
            float(
                np.median(widths)
                * 0.5
            ),
            float(
                np.median(heights)
                * 0.5
            ),
        )

        print(
            f"  {class_name:20s} "
            f"n={len(embeddings):2d} "
            f"L1≈"
            f"{dimensions[class_name][0]:.1f}"
            f"x"
            f"{dimensions[class_name][1]:.1f}"
        )

    return (
        prototypes,
        dimensions,
    )


def load_validation_frames():
    records = []

    for meta_path in sorted(
        VALIDATION.glob("*.json")
    ):
        meta = json.loads(
            meta_path.read_text()
        )

        if (
            meta["view"][
                "resolution_level"
            ]
            != 1
        ):
            continue

        image_path = (
            meta_path.with_suffix(
                ".png"
            )
        )

        if not image_path.exists():
            continue

        records.append(
            {
                "frame":
                    int(meta["frame"]),
                "frame_index":
                    int(
                        meta[
                            "frame_index"
                        ]
                    ),
                "image_path":
                    image_path,
            }
        )

    if len(records) <= N_VALIDATION_FRAMES:
        return records

    positions = np.linspace(
        0,
        len(records) - 1,
        N_VALIDATION_FRAMES,
        dtype=int,
    )

    return [
        records[int(i)]
        for i in positions
    ]


def dense_features(
    model,
    image,
    device,
    patch_size,
):
    tensor = preprocess(
        image,
        DINO_W,
        DINO_H,
    )

    tensor = normalize_input(
        tensor,
        device,
    )

    with torch.inference_mode():
        output = model(
            pixel_values=tensor
        )

    tokens = (
        output.last_hidden_state[
            :,
            1:,
            :,
        ]
    )

    grid_h = (
        DINO_H
        // patch_size
    )

    grid_w = (
        DINO_W
        // patch_size
    )

    dim = tokens.shape[-1]

    features = tokens.reshape(
        1,
        grid_h,
        grid_w,
        dim,
    ).permute(
        0,
        3,
        1,
        2,
    )

    return features


def class_similarity_map(
    feature_map,
    prototype,
    expected_width,
    expected_height,
    patch_size,
):
    # Convert expected L1 pixels to the resized DINO frame.
    scale_x = (
        DINO_W
        / VIEW_W
    )

    scale_y = (
        DINO_H
        / VIEW_H
    )

    expected_w_dino = (
        expected_width
        * scale_x
    )

    expected_h_dino = (
        expected_height
        * scale_y
    )

    pool_w = max(
        1,
        int(
            round(
                expected_w_dino
                / patch_size
            )
        ),
    )

    pool_h = max(
        1,
        int(
            round(
                expected_h_dino
                / patch_size
            )
        ),
    )

    # Give larger objects enough representation.
    pool_w = min(
        pool_w,
        feature_map.shape[-1],
    )

    pool_h = min(
        pool_h,
        feature_map.shape[-2],
    )

    pooled = F.avg_pool2d(
        feature_map,
        kernel_size=(
            pool_h,
            pool_w,
        ),
        stride=1,
    )

    pooled = F.normalize(
        pooled,
        dim=1,
    )

    similarity = (
        pooled
        * prototype.view(
            1,
            -1,
            1,
            1,
        )
    ).sum(
        dim=1
    )[0]

    return (
        similarity,
        pool_w,
        pool_h,
    )


def top_locations(
    similarity,
    k=3,
):
    flat = similarity.flatten()

    count = min(
        k,
        flat.numel(),
    )

    values, indices = (
        torch.topk(
            flat,
            k=count,
        )
    )

    width = (
        similarity.shape[1]
    )

    results = []

    for value, index in zip(
        values,
        indices,
    ):
        index = int(
            index.item()
        )

        y = index // width
        x = index % width

        results.append(
            (
                float(
                    value.item()
                ),
                x,
                y,
            )
        )

    return results


def patch_location_to_view_box(
    x,
    y,
    pool_w,
    pool_h,
    patch_size,
):
    # Coordinates in resized DINO frame.
    x1_dino = (
        x
        * patch_size
    )

    y1_dino = (
        y
        * patch_size
    )

    x2_dino = (
        (x + pool_w)
        * patch_size
    )

    y2_dino = (
        (y + pool_h)
        * patch_size
    )

    # Convert back to the received 960x540 L1 image.
    x1 = (
        x1_dino
        * VIEW_W
        / DINO_W
    )

    y1 = (
        y1_dino
        * VIEW_H
        / DINO_H
    )

    x2 = (
        x2_dino
        * VIEW_W
        / DINO_W
    )

    y2 = (
        y2_dino
        * VIEW_H
        / DINO_H
    )

    return [
        float(x1),
        float(y1),
        float(x2),
        float(y2),
    ]


def retain(
    results,
    candidate,
):
    results.append(
        candidate
    )

    results.sort(
        key=lambda r:
            r["similarity"],
        reverse=True,
    )

    del results[
        TOP_K_PER_CLASS:
    ]


def save_match_card(
    class_name,
    rank,
    result,
):
    image = cv2.imread(
        result["image_path"]
    )

    x1, y1, x2, y2 = (
        result["bbox"]
    )

    w = max(
        1.0,
        x2 - x1,
    )

    h = max(
        1.0,
        y2 - y1,
    )

    cx = (
        x1 + x2
    ) / 2

    cy = (
        y1 + y2
    ) / 2

    display_w = max(
        50,
        w * DISPLAY_CONTEXT,
    )

    display_h = max(
        50,
        h * DISPLAY_CONTEXT,
    )

    ih, iw = image.shape[:2]

    dx1 = max(
        0,
        int(
            round(
                cx
                - display_w / 2
            )
        ),
    )

    dy1 = max(
        0,
        int(
            round(
                cy
                - display_h / 2
            )
        ),
    )

    dx2 = min(
        iw,
        int(
            round(
                cx
                + display_w / 2
            )
        ),
    )

    dy2 = min(
        ih,
        int(
            round(
                cy
                + display_h / 2
            )
        ),
    )

    crop = image[
        dy1:dy2,
        dx1:dx2,
    ].copy()

    cv2.rectangle(
        crop,
        (
            int(round(x1)) - dx1,
            int(round(y1)) - dy1,
        ),
        (
            int(round(x2)) - dx1,
            int(round(y2)) - dy1,
        ),
        (0, 0, 255),
        2,
    )

    label = (
        f"{class_name} "
        f"{result['similarity']:.3f}"
    )

    cv2.putText(
        crop,
        label,
        (5, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    path = (
        OUT
        / "matches"
        / (
            f"{class_name}"
            f"_rank{rank:02d}"
            f"_frame"
            f"{result['frame_index']:03d}.png"
        )
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cv2.imwrite(
        str(path),
        crop,
    )

    return path


def build_montage(
    results,
):
    cards = []

    cell_w = 300
    cell_h = 210

    for class_name in sorted(
        results
    ):
        for rank, result in enumerate(
            results[class_name][:3],
            start=1,
        ):
            path = save_match_card(
                class_name,
                rank,
                result,
            )

            image = cv2.imread(
                str(path)
            )

            if image is None:
                continue

            usable_w = (
                cell_w - 10
            )

            usable_h = (
                cell_h - 32
            )

            scale = min(
                usable_w
                / image.shape[1],
                usable_h
                / image.shape[0],
            )

            scale = min(
                scale,
                6.0,
            )

            resized = cv2.resize(
                image,
                (
                    max(
                        1,
                        int(
                            image.shape[1]
                            * scale
                        ),
                    ),
                    max(
                        1,
                        int(
                            image.shape[0]
                            * scale
                        ),
                    ),
                ),
                interpolation=(
                    cv2.INTER_NEAREST
                    if scale > 1
                    else cv2.INTER_AREA
                ),
            )

            card = np.zeros(
                (
                    cell_h,
                    cell_w,
                    3,
                ),
                dtype=np.uint8,
            )

            x = (
                cell_w
                - resized.shape[1]
            ) // 2

            y = 30

            card[
                y:y + resized.shape[0],
                x:x + resized.shape[1],
            ] = resized

            cv2.putText(
                card,
                (
                    f"{class_name} "
                    f"#{rank} "
                    f"{result['similarity']:.3f}"
                ),
                (5, 19),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            cards.append(
                card
            )

    cols = 3

    rows = []

    for start in range(
        0,
        len(cards),
        cols,
    ):
        row = cards[
            start:start + cols
        ]

        while len(row) < cols:
            row.append(
                np.zeros_like(
                    cards[0]
                )
            )

        rows.append(
            np.hstack(row)
        )

    montage = np.vstack(
        rows
    )

    path = (
        OUT
        / "dense_dino_montage.jpg"
    )

    cv2.imwrite(
        str(path),
        montage,
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

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device,
    )

    print(
        "Model:",
        MODEL_NAME,
    )

    print(
        "Loading DINO..."
    )

    model = AutoModel.from_pretrained(
        MODEL_NAME
    )

    model.eval()
    model.to(device)

    patch_size = int(
        model.config.patch_size
    )

    print(
        "Patch size:",
        patch_size,
    )

    (
        prototypes,
        dimensions,
    ) = build_prototypes(
        model,
        device,
        patch_size,
    )

    frames = (
        load_validation_frames()
    )

    print()
    print(
        "Validation frames sampled:",
        len(frames),
    )

    print(
        "Frame indices:",
        [
            f["frame_index"]
            for f in frames
        ],
    )

    results = {
        class_name: []
        for class_name
        in prototypes
    }

    for number, frame in enumerate(
        frames,
        start=1,
    ):
        image = cv2.imread(
            str(
                frame[
                    "image_path"
                ]
            )
        )

        features = dense_features(
            model,
            image,
            device,
            patch_size,
        )

        for class_name in sorted(
            prototypes
        ):
            expected_w, expected_h = (
                dimensions[
                    class_name
                ]
            )

            (
                similarity,
                pool_w,
                pool_h,
            ) = class_similarity_map(
                features,
                prototypes[
                    class_name
                ],
                expected_w,
                expected_h,
                patch_size,
            )

            for (
                score,
                x,
                y,
            ) in top_locations(
                similarity,
                k=3,
            ):
                bbox = (
                    patch_location_to_view_box(
                        x,
                        y,
                        pool_w,
                        pool_h,
                        patch_size,
                    )
                )

                retain(
                    results[
                        class_name
                    ],
                    {
                        "class":
                            class_name,
                        "similarity":
                            score,
                        "frame":
                            frame["frame"],
                        "frame_index":
                            frame[
                                "frame_index"
                            ],
                        "image_path":
                            str(
                                frame[
                                    "image_path"
                                ]
                            ),
                        "bbox":
                            bbox,
                        "pool_tokens":
                            [
                                int(pool_w),
                                int(pool_h),
                            ],
                    },
                )

        print(
            f"{number:02d}/"
            f"{len(frames)} "
            f"frame_index="
            f"{frame['frame_index']:03d}"
        )

    serializable = {
        class_name:
            rows
        for class_name, rows
        in results.items()
    }

    result_path = (
        OUT
        / "dense_dino_results.json"
    )

    result_path.write_text(
        json.dumps(
            serializable,
            indent=2,
        )
    )

    montage = build_montage(
        results
    )

    print()
    print("=" * 80)
    print(
        "EXP-D026A DENSE DINO SCREEN"
    )
    print("=" * 80)

    for class_name in sorted(
        results
    ):
        scores = [
            r["similarity"]
            for r in results[
                class_name
            ]
        ]

        print(
            f"{class_name:20s} "
            f"best={scores[0]:.4f} "
            f"top3="
            f"{np.mean(scores[:3]):.4f} "
            f"top5="
            f"{np.mean(scores):.4f}"
        )

    print()
    print(
        "Results:",
        result_path,
    )

    print(
        "Montage:",
        montage,
    )


if __name__ == "__main__":
    main()
