#!/usr/bin/env python3

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]

RESULTS = (
    ROOT
    / "drone"
    / "artifacts"
    / "exp_d027b"
    / "propagation_benchmark.json"
)

OUT = (
    ROOT
    / "drone"
    / "artifacts"
    / "exp_d027c"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


GAPS = [
    10,
    15,
    20,
    30,
]

THRESHOLDS = [
    0.80,
    0.85,
    0.90,
    0.92,
    0.94,
    0.95,
    0.96,
    0.97,
    0.98,
    0.99,
]


def summarize(rows):
    if not rows:
        return None

    success = np.asarray(
        [
            r["iou"] >= 0.50
            for r in rows
        ],
        dtype=bool,
    )

    ious = np.asarray(
        [
            r["iou"]
            for r in rows
        ],
        dtype=float,
    )

    return {
        "n":
            len(rows),

        "precision_iou50":
            float(
                success.mean()
            ),

        "median_iou":
            float(
                np.median(ious)
            ),

        "mean_iou":
            float(
                ious.mean()
            ),
    }


def main():
    data = json.loads(
        RESULTS.read_text()
    )

    pairs = data["pairs"]

    output = {}

    print(
        "=" * 84
    )

    print(
        "EXP-D027C PROPAGATION CALIBRATION"
    )

    print(
        "=" * 84
    )

    for gap in GAPS:
        print()
        print(
            f"MAX GAP = {gap}"
        )

        print(
            "-" * 84
        )

        print(
            f"{'NCC':>6} "
            f"{'N':>5} "
            f"{'P@IoU.50':>10} "
            f"{'medianIoU':>10}"
        )

        print(
            "-" * 84
        )

        gap_output = {}

        for threshold in THRESHOLDS:
            selected = [
                r
                for r in pairs
                if (
                    r["temporal_gap"]
                    <= gap
                    and r["score"]
                    >= threshold
                )
            ]

            result = summarize(
                selected
            )

            gap_output[
                str(threshold)
            ] = result

            if result is None:
                print(
                    f"{threshold:6.2f} "
                    f"{0:5d} "
                    f"{'-':>10} "
                    f"{'-':>10}"
                )

            else:
                print(
                    f"{threshold:6.2f} "
                    f"{result['n']:5d} "
                    f"{result['precision_iou50']:10.3f} "
                    f"{result['median_iou']:10.3f}"
                )

        output[
            str(gap)
        ] = gap_output

    print()
    print(
        "=" * 84
    )

    print(
        "PER-CLASS — GAP <= 15"
    )

    print(
        "=" * 84
    )

    classes = sorted(
        set(
            r["class"]
            for r in pairs
        )
    )

    for class_name in classes:
        rows = [
            r
            for r in pairs
            if (
                r["class"]
                == class_name
                and r["temporal_gap"]
                <= 15
            )
        ]

        if not rows:
            continue

        print()
        print(
            class_name
        )

        for threshold in [
            0.90,
            0.94,
            0.96,
            0.97,
            0.98,
        ]:
            selected = [
                r
                for r in rows
                if r["score"]
                >= threshold
            ]

            result = summarize(
                selected
            )

            if result is None:
                continue

            print(
                f"  NCC>={threshold:.2f}: "
                f"n={result['n']:2d} "
                f"P@.50="
                f"{result['precision_iou50']:.3f} "
                f"medIoU="
                f"{result['median_iou']:.3f}"
            )

    path = (
        OUT
        / "calibration.json"
    )

    path.write_text(
        json.dumps(
            output,
            indent=2,
        )
    )

    print()
    print(
        "Saved:",
        path,
    )


if __name__ == "__main__":
    main()
