#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]

SOURCE = (
    ROOT
    / "drone"
    / "artifacts"
    / "exp_d028c"
    / "cycle_pseudolabels.json"
)

OUT = (
    ROOT
    / "drone"
    / "artifacts"
    / "exp_d028d"
)

STATE_PATH = (
    OUT
    / "audit_state.json"
)

ACCEPTED_PATH = (
    OUT
    / "accepted_pseudolabels.json"
)

WINDOW = "EXP-D028D pseudo-label audit"


def load_rows():
    data = json.loads(
        SOURCE.read_text()
    )

    rows = data[
        "pseudolabels"
    ]

    # Trusted first, then strongest cycle consistency.
    rows.sort(
        key=lambda r: (
            not r["trusted_class"],
            -r["cycle_iou"],
            -r["forward_ncc"],
        )
    )

    return rows


def load_state():
    if not STATE_PATH.exists():
        return {}

    return json.loads(
        STATE_PATH.read_text()
    )


def save_state(state):
    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    STATE_PATH.write_text(
        json.dumps(
            state,
            indent=2,
        )
    )


def decision_key(row):
    return (
        f"{row['class']}:"
        f"{row['target_frame_index']}:"
        f"{row['bbox']}"
    )


def crop_display(row):
    image = cv2.imread(
        row["image_path"]
    )

    if image is None:
        raise RuntimeError(
            row["image_path"]
        )

    x1, y1, x2, y2 = (
        map(int, row["bbox"])
    )

    w = max(
        1,
        x2 - x1,
    )

    h = max(
        1,
        y2 - y1,
    )

    pad_x = max(
        100,
        int(round(w * 5)),
    )

    pad_y = max(
        100,
        int(round(h * 5)),
    )

    cx = (
        x1 + x2
    ) // 2

    cy = (
        y1 + y2
    ) // 2

    cx1 = max(
        0,
        cx - pad_x,
    )

    cy1 = max(
        0,
        cy - pad_y,
    )

    cx2 = min(
        image.shape[1],
        cx + pad_x,
    )

    cy2 = min(
        image.shape[0],
        cy + pad_y,
    )

    crop = image[
        cy1:cy2,
        cx1:cx2,
    ].copy()

    cv2.rectangle(
        crop,
        (
            x1 - cx1,
            y1 - cy1,
        ),
        (
            x2 - cx1,
            y2 - cy1,
        ),
        (0, 0, 255),
        3,
    )

    return crop


def main():
    rows = load_rows()

    state = load_state()

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 72)
    print("EXP-D028D PSEUDO-LABEL AUDIT")
    print("=" * 72)
    print()
    print("Controls:")
    print("  y = ACCEPT")
    print("  n = REJECT")
    print("  u = undo previous decision")
    print("  q / ESC = save and quit")
    print()
    print(
        "Accept only if BOTH class and box look correct."
    )
    print()

    cv2.namedWindow(
        WINDOW,
        cv2.WINDOW_NORMAL,
    )

    cv2.resizeWindow(
        WINDOW,
        1200,
        900,
    )

    index = 0

    while index < len(rows):
        row = rows[index]

        key = decision_key(
            row
        )

        # Skip already reviewed labels.
        if key in state:
            index += 1
            continue

        crop = crop_display(
            row
        )

        canvas = np.zeros(
            (
                crop.shape[0] + 80,
                max(
                    crop.shape[1],
                    700,
                ),
                3,
            ),
            dtype=np.uint8,
        )

        canvas[
            80:
            80 + crop.shape[0],
            0:
            crop.shape[1],
        ] = crop

        trust = (
            "TRUST"
            if row["trusted_class"]
            else "AUDIT"
        )

        title = (
            f"{index + 1}/{len(rows)} | "
            f"{trust} | "
            f"{row['class']} | "
            f"frame={row['target_frame_index']} | "
            f"NCC={row['forward_ncc']:.3f} | "
            f"cycle={row['cycle_iou']:.3f}"
        )

        cv2.putText(
            canvas,
            title,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            canvas,
            "Y = accept    N = reject    U = undo    Q = quit",
            (10, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        cv2.imshow(
            WINDOW,
            canvas,
        )

        code = (
            cv2.waitKey(0)
            & 0xFF
        )

        if code in (
            27,
            ord("q"),
        ):
            break

        if code == ord("y"):
            state[key] = {
                "decision":
                    "accept",
                "row":
                    row,
            }

            save_state(
                state
            )

            print(
                f"ACCEPT "
                f"{row['class']} "
                f"frame={row['target_frame_index']}"
            )

            index += 1
            continue

        if code == ord("n"):
            state[key] = {
                "decision":
                    "reject",
                "row":
                    row,
            }

            save_state(
                state
            )

            print(
                f"REJECT "
                f"{row['class']} "
                f"frame={row['target_frame_index']}"
            )

            index += 1
            continue

        if code == ord("u"):
            reviewed = list(
                state.keys()
            )

            if reviewed:
                last_key = (
                    reviewed[-1]
                )

                del state[
                    last_key
                ]

                save_state(
                    state
                )

                index = max(
                    0,
                    index - 1,
                )

            continue

    save_state(
        state
    )

    accepted = [
        value["row"]
        for value in state.values()
        if (
            value["decision"]
            == "accept"
        )
    ]

    rejected = [
        value["row"]
        for value in state.values()
        if (
            value["decision"]
            == "reject"
        )
    ]

    ACCEPTED_PATH.write_text(
        json.dumps(
            accepted,
            indent=2,
        )
    )

    by_class = {}

    for row in accepted:
        cls = row["class"]

        by_class[cls] = (
            by_class.get(
                cls,
                0,
            )
            + 1
        )

    cv2.destroyAllWindows()

    print()
    print("=" * 72)
    print("AUDIT SUMMARY")
    print("=" * 72)

    print(
        "Reviewed:",
        len(state),
        "/",
        len(rows),
    )

    print(
        "Accepted:",
        len(accepted),
    )

    print(
        "Rejected:",
        len(rejected),
    )

    if state:
        print(
            "Acceptance rate:",
            f"{len(accepted) / len(state):.1%}",
        )

    print()
    print("Accepted per class:")

    for cls, n in sorted(
        by_class.items()
    ):
        print(
            f"  {cls:20s} {n}"
        )

    print()
    print(
        "Saved:",
        ACCEPTED_PATH,
    )


if __name__ == "__main__":
    main()
