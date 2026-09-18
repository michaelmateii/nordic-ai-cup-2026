#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]

CAPTURE = (
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
    / "exp_d027"
)

ANNOTATIONS_PATH = (
    OUT
    / "validation_seed_annotations.json"
)

SELECTION_MANIFEST = (
    OUT
    / "selected_seed_frames.json"
)

N_FRAMES = 80


CLASSES = [
    "condor",
    "hangar",
    "jammer",
    "jet_plane",
    "small_launcher",
    "small_plane",
    "small_tower",
    "spacecraft",
    "ta-ta",
]


KEYS = [
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
]


KEY_TO_CLASS = dict(
    zip(
        KEYS,
        CLASSES,
    )
)


WINDOW = "EXP-D027 validation seed annotator"

drawing = False
start_point = None
current_point = None
pending_box = None


def load_all_l1():
    records = []

    for meta_path in sorted(
        CAPTURE.glob("*.json")
    ):
        data = json.loads(
            meta_path.read_text()
        )

        if (
            data["view"]["resolution_level"]
            != 1
        ):
            continue

        image_path = (
            meta_path.with_suffix(".png")
        )

        if not image_path.exists():
            continue

        records.append(
            {
                "frame":
                    int(data["frame"]),
                "frame_index":
                    int(data["frame_index"]),
                "center_x":
                    int(
                        data["view"]["center_x"]
                    ),
                "center_y":
                    int(
                        data["view"]["center_y"]
                    ),
                "source_region_xyxy":
                    data["view"][
                        "source_region_xyxy"
                    ],
                "image_path":
                    str(image_path),
            }
        )

    return records


def choose_seed_frames(
    records,
):
    if len(records) <= N_FRAMES:
        return records

    positions = np.linspace(
        0,
        len(records) - 1,
        N_FRAMES,
        dtype=int,
    )

    selected = [
        records[int(i)]
        for i in positions
    ]

    return selected


def load_or_create_selection():
    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    if SELECTION_MANIFEST.exists():
        return json.loads(
            SELECTION_MANIFEST.read_text()
        )

    selected = choose_seed_frames(
        load_all_l1()
    )

    SELECTION_MANIFEST.write_text(
        json.dumps(
            selected,
            indent=2,
        )
    )

    return selected


def load_annotations():
    if not ANNOTATIONS_PATH.exists():
        return {}

    return json.loads(
        ANNOTATIONS_PATH.read_text()
    )


def save_annotations(
    annotations,
):
    ANNOTATIONS_PATH.write_text(
        json.dumps(
            annotations,
            indent=2,
        )
    )


def mouse_callback(
    event,
    x,
    y,
    flags,
    param,
):
    global drawing
    global start_point
    global current_point
    global pending_box

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True

        start_point = (
            x,
            y,
        )

        current_point = (
            x,
            y,
        )

        pending_box = None

    elif (
        event == cv2.EVENT_MOUSEMOVE
        and drawing
    ):
        current_point = (
            x,
            y,
        )

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

        x1 = min(
            start_point[0],
            x,
        )

        y1 = min(
            start_point[1],
            y,
        )

        x2 = max(
            start_point[0],
            x,
        )

        y2 = max(
            start_point[1],
            y,
        )

        if (
            x2 - x1 >= 3
            and y2 - y1 >= 3
        ):
            pending_box = [
                int(x1),
                int(y1),
                int(x2),
                int(y2),
            ]

        current_point = (
            x,
            y,
        )


def draw_legend(
    image,
):
    x = 8
    y = 50

    for key, class_name in zip(
        KEYS,
        CLASSES,
    ):
        cv2.putText(
            image,
            f"{key}: {class_name}",
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        y += 18

        if y > 510:
            x += 170
            y = 50


def draw_annotations(
    image,
    rows,
):
    for row in rows:
        x1, y1, x2, y2 = (
            row["bbox"]
        )

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            image,
            row["class"],
            (
                x1,
                max(15, y1 - 5),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )


def main():
    global pending_box

    frames = (
        load_or_create_selection()
    )

    annotations = (
        load_annotations()
    )

    print()
    print("=" * 72)
    print("EXP-D027 VALIDATION SEED ANNOTATOR")
    print("=" * 72)

    for key, class_name in zip(
        KEYS,
        CLASSES,
    ):
        print(
            f"  {key} = {class_name}"
        )

    print()
    print("Controls:")
    print("  drag mouse = create pending box")
    print("  class key  = assign pending box")
    print("  n / space  = next frame")
    print("  p          = previous frame")
    print("  u          = undo last annotation")
    print("  x          = clear pending box")
    print("  q / ESC    = quit")
    print()
    print(
        "ONLY label objects you are highly confident about."
    )
    print()

    cv2.namedWindow(
        WINDOW,
        cv2.WINDOW_NORMAL,
    )

    cv2.resizeWindow(
        WINDOW,
        1440,
        810,
    )

    cv2.setMouseCallback(
        WINDOW,
        mouse_callback,
    )

    index = 0

    while True:
        record = frames[index]

        image = cv2.imread(
            record["image_path"]
        )

        if image is None:
            raise RuntimeError(
                record["image_path"]
            )

        key_id = str(
            record["frame_index"]
        )

        frame_annotations = (
            annotations.get(
                key_id,
                [],
            )
        )

        display = image.copy()

        draw_annotations(
            display,
            frame_annotations,
        )

        if drawing:
            cv2.rectangle(
                display,
                start_point,
                current_point,
                (0, 255, 255),
                2,
            )

        elif pending_box is not None:
            x1, y1, x2, y2 = (
                pending_box
            )

            cv2.rectangle(
                display,
                (x1, y1),
                (x2, y2),
                (0, 255, 255),
                2,
            )

        cv2.rectangle(
            display,
            (0, 0),
            (960, 36),
            (0, 0, 0),
            -1,
        )

        title = (
            f"seed {index + 1}/{len(frames)} | "
            f"frame_index={record['frame_index']} | "
            f"L1 center=({record['center_x']},"
            f"{record['center_y']}) | "
            f"labels={len(frame_annotations)}"
        )

        cv2.putText(
            display,
            title,
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        draw_legend(
            display
        )

        cv2.imshow(
            WINDOW,
            display,
        )

        code = (
            cv2.waitKey(20)
            & 0xFF
        )

        if code == 255:
            continue

        if code in (
            27,
            ord("q"),
        ):
            break

        if code in (
            ord("n"),
            ord(" "),
        ):
            pending_box = None

            index = min(
                len(frames) - 1,
                index + 1,
            )

            continue

        if code == ord("p"):
            pending_box = None

            index = max(
                0,
                index - 1,
            )

            continue

        if code == ord("x"):
            pending_box = None
            continue

        if code == ord("u"):
            if frame_annotations:
                frame_annotations.pop()

                annotations[
                    key_id
                ] = frame_annotations

                save_annotations(
                    annotations
                )

                print(
                    f"Undo frame "
                    f"{record['frame_index']}"
                )

            continue

        char = chr(code)

        if (
            char in KEY_TO_CLASS
            and pending_box is not None
        ):
            class_name = (
                KEY_TO_CLASS[char]
            )

            row = {
                "class":
                    class_name,
                "bbox":
                    pending_box,
                "frame":
                    record["frame"],
                "frame_index":
                    record["frame_index"],
                "center_x":
                    record["center_x"],
                "center_y":
                    record["center_y"],
                "source_region_xyxy":
                    record[
                        "source_region_xyxy"
                    ],
            }

            frame_annotations.append(
                row
            )

            annotations[
                key_id
            ] = frame_annotations

            save_annotations(
                annotations
            )

            print(
                f"Saved "
                f"frame={record['frame_index']:03d} "
                f"class={class_name:20s} "
                f"bbox={pending_box}"
            )

            pending_box = None

    save_annotations(
        annotations
    )

    cv2.destroyAllWindows()

    total = sum(
        len(v)
        for v in annotations.values()
    )

    classes = {}

    for rows in annotations.values():
        for row in rows:
            cls = row["class"]

            classes[cls] = (
                classes.get(cls, 0)
                + 1
            )

    print()
    print("=" * 72)
    print("CURRENT SEED INVENTORY")
    print("=" * 72)

    print(
        "Annotations:",
        total,
    )

    print(
        "Frames labeled:",
        sum(
            bool(v)
            for v in annotations.values()
        ),
    )

    print()

    for class_name in CLASSES:
        if class_name in classes:
            print(
                f"{class_name:20s} "
                f"{classes[class_name]}"
            )

    print()
    print(
        "Saved:",
        ANNOTATIONS_PATH,
    )


if __name__ == "__main__":
    main()
