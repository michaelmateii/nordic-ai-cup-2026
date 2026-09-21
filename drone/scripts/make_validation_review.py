import json
import math
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]

CAPTURE_ROOT = ROOT / "drone" / "captures"

THRESHOLDS = [
    0.01,
    0.02,
    0.03,
    0.05,
    0.08,
    0.10,
]

sequences = [
    p for p in CAPTURE_ROOT.iterdir()
    if p.is_dir()
    and (p / "predictions").exists()
]

if not sequences:
    raise RuntimeError(
        "No captured sequence with predictions found."
    )

SEQ = max(
    sequences,
    key=lambda p: p.stat().st_mtime,
)

FRAMES = SEQ / "frames"
PREDS = SEQ / "predictions"

BASE_OUT = (
    ROOT
    / "drone"
    / "artifacts"
    / "validation_review"
    / SEQ.name
)

print("Sequence:", SEQ.name)

prediction_files = sorted(
    PREDS.glob("*.json")
)


def intersect(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


for threshold in THRESHOLDS:

    threshold_name = (
        f"conf_{threshold:.3f}"
        .replace(".", "_")
    )

    OUT = BASE_OUT / threshold_name
    OVERLAYS = OUT / "overlays"
    SHEETS = OUT / "contact_sheets"

    OVERLAYS.mkdir(
        parents=True,
        exist_ok=True,
    )

    SHEETS.mkdir(
        parents=True,
        exist_ok=True,
    )

    overlay_paths = []
    total_visible = 0
    total_kept = 0

    for pred_path in prediction_files:

        pred = json.loads(
            pred_path.read_text(
                encoding="utf-8"
            )
        )

        stem = pred_path.stem

        image_path = (
            FRAMES
            / f"{stem}.png"
        )

        metadata_path = (
            FRAMES
            / f"{stem}.json"
        )

        if not image_path.exists():
            print(
                "Missing image:",
                image_path,
            )
            continue

        if not metadata_path.exists():
            print(
                "Missing metadata:",
                metadata_path,
            )
            continue

        img = Image.open(
            image_path
        ).convert("RGB")

        draw = ImageDraw.Draw(img)

        iw, ih = img.size

        view = pred["view"]

        rx1, ry1, rx2, ry2 = (
            view["source_region_xyxy"]
        )

        region_w = rx2 - rx1
        region_h = ry2 - ry1

        original_w = (
            pred["original_width"]
        )

        original_h = (
            pred["original_height"]
        )

        crop_region = (
            rx1,
            ry1,
            rx2,
            ry2,
        )

        visible_count = 0

        for ann in pred["annotations"]:

            confidence = float(
                ann["confidence"]
            )

            if confidence < threshold:
                continue

            total_kept += 1

            gx1 = (
                ann["bbox"][0]
                * original_w
            )

            gy1 = (
                ann["bbox"][1]
                * original_h
            )

            gx2 = (
                ann["bbox"][2]
                * original_w
            )

            gy2 = (
                ann["bbox"][3]
                * original_h
            )

            global_box = (
                gx1,
                gy1,
                gx2,
                gy2,
            )

            visible = intersect(
                global_box,
                crop_region,
            )

            if visible is None:
                continue

            visible_count += 1
            total_visible += 1

            vx1, vy1, vx2, vy2 = (
                visible
            )

            lx1 = (
                (vx1 - rx1)
                / region_w
            ) * iw

            ly1 = (
                (vy1 - ry1)
                / region_h
            ) * ih

            lx2 = (
                (vx2 - rx1)
                / region_w
            ) * iw

            ly2 = (
                (vy2 - ry1)
                / region_h
            ) * ih

            draw.rectangle(
                [
                    lx1,
                    ly1,
                    lx2,
                    ly2,
                ],
                outline="lime",
                width=2,
            )

            label = (
                f'{ann["object_id"]} '
                f'{confidence:.3f}'
            )

            draw.text(
                (
                    int(lx1),
                    max(
                        24,
                        int(ly1) - 14,
                    ),
                ),
                label,
                fill="lime",
            )

        draw.rectangle(
            [0, 0, 900, 24],
            fill="black",
        )

        draw.text(
            (5, 4),
            (
                f'{stem} '
                f'L{view["resolution_level"]} '
                f'center=('
                f'{view["center_x"]},'
                f'{view["center_y"]}) '
                f'conf>={threshold:.3f} '
                f'visible={visible_count}'
            ),
            fill="white",
        )

        out_path = (
            OVERLAYS
            / f"{stem}.jpg"
        )

        img.save(
            out_path,
            quality=94,
        )

        overlay_paths.append(
            out_path
        )

    COLS = 2
    ROWS = 2
    PER_SHEET = 4

    tile_w = 800
    tile_h = 480

    for start in range(
        0,
        len(overlay_paths),
        PER_SHEET,
    ):

        chunk = overlay_paths[
            start:
            start + PER_SHEET
        ]

        sheet = Image.new(
            "RGB",
            (
                COLS * tile_w,
                ROWS * tile_h,
            ),
            (230, 230, 230),
        )

        for i, path in enumerate(
            chunk
        ):

            img = Image.open(
                path
            ).convert("RGB")

            img.thumbnail(
                (
                    tile_w - 10,
                    tile_h - 10,
                )
            )

            x = (
                i % COLS
            ) * tile_w

            y = (
                i // COLS
            ) * tile_h

            px = (
                x
                + (
                    tile_w
                    - img.width
                )
                // 2
            )

            py = (
                y
                + (
                    tile_h
                    - img.height
                )
                // 2
            )

            sheet.paste(
                img,
                (px, py),
            )

        sheet_path = (
            SHEETS
            / f"review_{start:04d}.jpg"
        )

        sheet.save(
            sheet_path,
            quality=94,
        )

    print()
    print(
        f"Threshold: {threshold:.3f}"
    )

    print(
        "Frames:",
        len(overlay_paths),
    )

    print(
        "Visible predictions:",
        total_visible,
    )

    print(
        "All kept predictions:",
        total_kept,
    )

    print(
        "Avg visible/frame:",
        round(
            total_visible
            / max(
                1,
                len(overlay_paths),
            ),
            2,
        ),
    )

    print(
        "Sheets:",
        math.ceil(
            len(overlay_paths)
            / PER_SHEET
        ),
    )

    print(
        "Output:",
        OUT,
    )