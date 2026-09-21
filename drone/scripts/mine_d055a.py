import json
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]

SEQ = ROOT / "drone" / "captures" / "ed590596961d4390b6919a7633824f6b"
PREDS = SEQ / "predictions"
FRAMES = SEQ / "frames"

OUT = ROOT / "drone" / "artifacts" / "exp_d055a"
CROPS = OUT / "crops"
SHEETS = OUT / "contact_sheets"

CROPS.mkdir(parents=True, exist_ok=True)
SHEETS.mkdir(parents=True, exist_ok=True)

MIN_CONF = 0.03
MERGE_IOU = 0.50
MAX_PER_FRAME = 12


def iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih

    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])

    return inter / (aa + bb - inter + 1e-9)


candidates = []

for pred_file in sorted(PREDS.glob("*.json")):

    data = json.loads(
        pred_file.read_text(encoding="utf-8")
    )

    stem = pred_file.stem
    img_path = FRAMES / f"{stem}.png"

    if not img_path.exists():
        continue

    img = Image.open(img_path).convert("RGB")
    iw, ih = img.size

    rows = []

    for ann in data["annotations"]:

        conf = float(ann["confidence"])

        if conf < MIN_CONF:
            continue

        # predictions are already frame-global normalized
        ow = int(data["original_width"])
        oh = int(data["original_height"])

        gx1 = float(ann["bbox"][0]) * ow
        gy1 = float(ann["bbox"][1]) * oh
        gx2 = float(ann["bbox"][2]) * ow
        gy2 = float(ann["bbox"][3]) * oh

        rx1, ry1, rx2, ry2 = data["view"]["source_region_xyxy"]

        # convert global source coordinates into received crop pixels
        lx1 = (gx1 - rx1) / (rx2 - rx1) * iw
        ly1 = (gy1 - ry1) / (ry2 - ry1) * ih
        lx2 = (gx2 - rx1) / (rx2 - rx1) * iw
        ly2 = (gy2 - ry1) / (ry2 - ry1) * ih

        # skip boxes not actually visible in current crop
        if lx2 <= 0 or ly2 <= 0 or lx1 >= iw or ly1 >= ih:
            continue

        box = [
            max(0.0, lx1),
            max(0.0, ly1),
            min(float(iw), lx2),
            min(float(ih), ly2),
        ]

        if box[2] <= box[0] or box[3] <= box[1]:
            continue

        rows.append(
            {
                "box": box,
                "conf": conf,
                "class_name": ann["object_id"],
            }
        )

    rows.sort(
        key=lambda r: r["conf"],
        reverse=True,
    )

    # class-agnostic merge
    kept = []

    for row in rows:
        if all(
            iou(row["box"], k["box"]) < MERGE_IOU
            for k in kept
        ):
            kept.append(row)

    kept = kept[:MAX_PER_FRAME]

    for row in kept:

        x1, y1, x2, y2 = row["box"]

        w = x2 - x1
        h = y2 - y1
        margin = 0.35 * max(w, h) + 8

        cx1 = int(max(0, x1 - margin))
        cy1 = int(max(0, y1 - margin))
        cx2 = int(min(iw, x2 + margin))
        cy2 = int(min(ih, y2 + margin))

        crop = img.crop(
            (cx1, cy1, cx2, cy2)
        )

        candidate_id = len(candidates)

        crop_path = (
            CROPS
            / f"{candidate_id:04d}.jpg"
        )

        crop.save(
            crop_path,
            quality=95,
        )

        candidates.append(
            {
                "id": candidate_id,
                "frame_file": str(img_path),
                "frame_stem": stem,
                "bbox_crop_pixels": row["box"],
                "class_name": row["class_name"],
                "confidence": row["conf"],
                "crop_path": str(crop_path),
            }
        )


(OUT / "candidates.json").write_text(
    json.dumps(
        candidates,
        indent=2,
    ),
    encoding="utf-8",
)

# make contact sheets
TILE_W = 160
TILE_H = 190
COLS = 8
ROWS = 6
PER_SHEET = COLS * ROWS

for start in range(
    0,
    len(candidates),
    PER_SHEET,
):

    chunk = candidates[
        start:start + PER_SHEET
    ]

    sheet = Image.new(
        "RGB",
        (
            COLS * TILE_W,
            ROWS * TILE_H,
        ),
        "black",
    )

    draw = ImageDraw.Draw(sheet)

    for i, c in enumerate(chunk):

        crop = Image.open(
            c["crop_path"]
        ).convert("RGB")

        crop.thumbnail(
            (TILE_W - 8, TILE_H - 35)
        )

        x = (i % COLS) * TILE_W
        y = (i // COLS) * TILE_H

        px = x + (TILE_W - crop.width) // 2
        py = y + 2

        sheet.paste(
            crop,
            (px, py),
        )

        draw.text(
            (x + 4, y + TILE_H - 29),
            f'{c["id"]} {c["class_name"][:11]}',
            fill="white",
        )

        draw.text(
            (x + 4, y + TILE_H - 15),
            f'{c["confidence"]:.3f}',
            fill="white",
        )

    out = (
        SHEETS
        / f"sheet_{start // PER_SHEET:03d}.jpg"
    )

    sheet.save(
        out,
        quality=95,
    )


print("Candidates:", len(candidates))
print("Output:", OUT)
