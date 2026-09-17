#!/usr/bin/env python3

from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]

DATA_YAML = (
    ROOT
    / "drone"
    / "artifacts"
    / "exp_d007"
    / "dataset"
    / "dataset.yaml"
)

RUN_DIR = (
    ROOT
    / "drone"
    / "artifacts"
    / "exp_d007"
    / "runs"
)


def main() -> None:
    print(f"Dataset: {DATA_YAML}")
    print(f"Runs:    {RUN_DIR}")

    model = YOLO("yolo11n.pt")

    model.train(
        data=str(DATA_YAML),

        # Preserve more detail for tiny L0 targets.
        imgsz=960,

        epochs=80,
        patience=15,

        batch=4,

        device="mps",

        workers=2,

        pretrained=True,

        # Dataset already represents the exact L0 geometry.
        rect=True,

        # Conservative augmentation for the first screen.
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.3,

        degrees=3.0,
        translate=0.05,
        scale=0.20,

        fliplr=0.5,
        flipud=0.0,

        mosaic=0.5,
        mixup=0.0,

        # Avoid excessive geometric distortion of tiny targets.
        shear=0.0,
        perspective=0.0,

        project=str(RUN_DIR),
        name="yolo11n_objectness",

        exist_ok=True,

        plots=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()
