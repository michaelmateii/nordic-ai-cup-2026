#!/usr/bin/env python3

from pathlib import Path

import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]

DATASET = (
    ROOT
    / "drone"
    / "artifacts"
    / "exp_d032"
    / "dataset"
    / "dataset.yaml"
)

RUNS = (
    ROOT
    / "drone"
    / "artifacts"
    / "exp_d032"
    / "runs"
)


def main():
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print("Device:", device)
    print("Dataset:", DATASET)

    model = YOLO(
        "yolo11s.pt"
    )

    model.train(
        data=str(DATASET),

        epochs=100,
        imgsz=960,
        batch=4,

        device=device,
        workers=0,

        project=str(RUNS),
        name="yolo11s_full_real_domain",
        exist_ok=True,

        hsv_h=0.01,
        hsv_s=0.20,
        hsv_v=0.20,

        degrees=3.0,
        translate=0.05,
        scale=0.15,
        shear=0.0,
        perspective=0.0,

        fliplr=0.5,
        flipud=0.0,

        mosaic=0.25,
        mixup=0.0,
        copy_paste=0.0,

        seed=42,
        deterministic=True,

        pretrained=True,

        patience=0,
        val=False,

        verbose=True,
    )


if __name__ == "__main__":
    main()
