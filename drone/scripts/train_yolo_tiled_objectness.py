#!/usr/bin/env python3

from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]

DATA = (
    ROOT
    / "drone"
    / "artifacts"
    / "exp_d010"
    / "dataset"
    / "dataset.yaml"
)

RUNS = (
    ROOT
    / "drone"
    / "artifacts"
    / "exp_d010"
    / "runs"
)


def main() -> None:
    print(f"Dataset: {DATA}")
    print(f"Runs:    {RUNS}")

    model = YOLO("yolo11n.pt")

    model.train(
        data=str(DATA),
        imgsz=960,
        epochs=80,
        patience=15,
        batch=4,
        device="mps",
        workers=2,
        pretrained=True,
        rect=True,

        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.3,

        degrees=3.0,
        translate=0.05,
        scale=0.20,

        fliplr=0.5,
        flipud=0.0,

        mosaic=0.0,
        mixup=0.0,

        shear=0.0,
        perspective=0.0,

        project=str(RUNS),
        name="yolo11n_tiled_objectness",
        exist_ok=True,

        plots=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()
