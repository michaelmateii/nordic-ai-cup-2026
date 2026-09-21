from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]

CAPTURE_DIR = (
    ROOT
    / "drone"
    / "captures"
    / "3224a582bfbf4273a028497662b7aa7c"
    / "frames"
)

MODEL_21 = (
    ROOT
    / "drone"
    / "artifacts"
    / "exp_d040"
    / "seed_21"
    / "weights"
    / "best.pt"
)

MODEL_7 = (
    ROOT
    / "drone"
    / "artifacts"
    / "exp_d040"
    / "seed_7"
    / "weights"
    / "best.pt"
)

OUT = (
    ROOT
    / "drone"
    / "artifacts"
    / "exp_d047"
)

CROPS_AGREE = OUT / "agreement_crops"
CROPS_DISAGREE = OUT / "disagreement_crops"
SHEETS = OUT / "contact_sheets"

for p in (OUT, CROPS_AGREE, CROPS_DISAGREE, SHEETS):
    p.mkdir(parents=True, exist_ok=True)


IMGSZ = 960
CONF = 0.001
NMS_IOU = 0.70
MAX_DET = 300

MATCH_IOU = 0.50

STRONG_AGREE_IOU = 0.60
STRONG_AGREE_MIN_CONF = 0.02

MAX_AGREEMENTS = 400
MAX_DISAGREEMENTS = 300


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def find_region(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            if (
                str(key).lower() == "source_region_xyxy"
                and isinstance(value, list)
                and len(value) == 4
            ):
                return value

            result = find_region(value)
            if result is not None:
                return result

    elif isinstance(obj, list):
        for value in obj:
            result = find_region(value)
            if result is not None:
                return result

    return None


def is_l1(meta):
    region = find_region(meta)

    if region is None:
        return False

    x1, y1, x2, y2 = region

    return (
        int(round(x2 - x1)) == 1920
        and int(round(y2 - y1)) == 1080
    )


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)

    inter = iw * ih

    if inter <= 0:
        return 0.0

    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ba = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = aa + ba - inter

    if union <= 0:
        return 0.0

    return inter / union


def detections(model, image):
    result = model.predict(
        source=image,
        imgsz=IMGSZ,
        conf=CONF,
        iou=NMS_IOU,
        max_det=MAX_DET,
        device=0,
        verbose=False,
    )[0]

    rows = []

    if result.boxes is None:
        return rows

    boxes = result.boxes.xyxy.detach().cpu().numpy()
    confs = result.boxes.conf.detach().cpu().numpy()
    classes = result.boxes.cls.detach().cpu().numpy().astype(int)

    for box, score, cls_id in zip(boxes, confs, classes):

        if isinstance(result.names, dict):
            name = result.names.get(int(cls_id))
        else:
            name = result.names[int(cls_id)]

        if name is None:
            continue

        rows.append({
            "class_name": str(name),
            "class_id": int(cls_id),
            "bbox": [float(v) for v in box],
            "score": float(score),
        })

    return rows


def match_models(a_rows, b_rows):
    matches = []
    used_b = set()

    order_a = sorted(
        range(len(a_rows)),
        key=lambda i: a_rows[i]["score"],
        reverse=True,
    )

    for ai in order_a:
        a = a_rows[ai]

        best = None

        for bi, b in enumerate(b_rows):
            if bi in used_b:
                continue

            if a["class_name"] != b["class_name"]:
                continue

            ov = iou(a["bbox"], b["bbox"])

            if best is None or ov > best[0]:
                best = (ov, bi)

        if best is not None and best[0] >= MATCH_IOU:
            ov, bi = best
            used_b.add(bi)

            matches.append(
                (ai, bi, ov)
            )

    matched_a = {x[0] for x in matches}
    matched_b = {x[1] for x in matches}

    only_a = [
        (i, row)
        for i, row in enumerate(a_rows)
        if i not in matched_a
    ]

    only_b = [
        (i, row)
        for i, row in enumerate(b_rows)
        if i not in matched_b
    ]

    return matches, only_a, only_b


def expanded_crop(image, bbox, scale=2.2):
    h, w = image.shape[:2]

    x1, y1, x2, y2 = bbox

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    bw = max(16.0, x2 - x1)
    bh = max(16.0, y2 - y1)

    nw = bw * scale
    nh = bh * scale

    xx1 = max(0, int(cx - nw / 2))
    yy1 = max(0, int(cy - nh / 2))
    xx2 = min(w, int(cx + nw / 2))
    yy2 = min(h, int(cy + nh / 2))

    return image[yy1:yy2, xx1:xx2].copy(), (xx1, yy1)


def draw_candidate(image, bbox, label):
    crop, (ox, oy) = expanded_crop(
        image,
        bbox,
        scale=2.4,
    )

    if crop.size == 0:
        return None

    x1, y1, x2, y2 = bbox

    p1 = (
        int(x1 - ox),
        int(y1 - oy),
    )

    p2 = (
        int(x2 - ox),
        int(y2 - oy),
    )

    cv2.rectangle(
        crop,
        p1,
        p2,
        (0, 255, 0),
        2,
    )

    cv2.putText(
        crop,
        label,
        (5, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )

    return crop


def make_sheet(paths, destination, cols=5):
    if not paths:
        return

    tiles = []

    for path in paths:
        img = Image.open(path).convert("RGB")
        img.thumbnail((300, 220))

        tile = Image.new(
            "RGB",
            (320, 250),
            "white",
        )

        x = (320 - img.width) // 2
        y = 5

        tile.paste(img, (x, y))

        draw = ImageDraw.Draw(tile)

        text = path.stem[:44]

        draw.text(
            (5, 225),
            text,
            fill="black",
        )

        tiles.append(tile)

    rows = math.ceil(len(tiles) / cols)

    sheet = Image.new(
        "RGB",
        (cols * 320, rows * 250),
        (230, 230, 230),
    )

    for i, tile in enumerate(tiles):
        x = (i % cols) * 320
        y = (i // cols) * 250

        sheet.paste(
            tile,
            (x, y),
        )

    sheet.save(destination, quality=92)


print("Loading models...")

m21 = YOLO(str(MODEL_21))
m7 = YOLO(str(MODEL_7))

dummy = np.zeros(
    (540, 960, 3),
    dtype=np.uint8,
)

detections(m21, dummy)
detections(m7, dummy)

print("Models warmed.")


frames = []

for meta_path in sorted(
    CAPTURE_DIR.glob("frame_*_index_*.json")
):
    meta = load_json(meta_path)

    if not is_l1(meta):
        continue

    png = meta_path.with_suffix(".png")

    if not png.exists():
        continue

    frames.append(
        (png, meta_path, meta)
    )


print("L1 frames:", len(frames))


agreements = []
disagreements = []


for n, (png, meta_path, meta) in enumerate(frames, 1):

    image = cv2.imread(str(png))

    if image is None:
        continue

    r21 = detections(
        m21,
        image,
    )

    r7 = detections(
        m7,
        image,
    )

    matches, only21, only7 = match_models(
        r21,
        r7,
    )

    for ai, bi, ov in matches:

        a = r21[ai]
        b = r7[bi]

        min_conf = min(
            a["score"],
            b["score"],
        )

        mean_conf = (
            a["score"]
            + b["score"]
        ) / 2

        rank_score = (
            ov
            * math.sqrt(
                max(
                    1e-12,
                    a["score"]
                    * b["score"],
                )
            )
        )

        bbox = [
            (
                a["bbox"][j]
                + b["bbox"][j]
            ) / 2
            for j in range(4)
        ]

        agreements.append({
            "frame_png": str(png),
            "frame_meta": str(meta_path),
            "frame_name": png.stem,
            "class_name": a["class_name"],
            "class_id": a["class_id"],
            "bbox": bbox,
            "seed21_score": a["score"],
            "seed7_score": b["score"],
            "min_conf": min_conf,
            "mean_conf": mean_conf,
            "iou": ov,
            "rank_score": rank_score,
        })

    for source, items in (
        ("seed21_only", only21),
        ("seed7_only", only7),
    ):
        for _, row in items:

            disagreements.append({
                "frame_png": str(png),
                "frame_meta": str(meta_path),
                "frame_name": png.stem,
                "source": source,
                "class_name": row["class_name"],
                "class_id": row["class_id"],
                "bbox": row["bbox"],
                "score": row["score"],
            })

    if (
        n % 20 == 0
        or n == len(frames)
    ):
        print(
            f"{n}/{len(frames)} "
            f"agreements={len(agreements)} "
            f"disagreements={len(disagreements)}"
        )


agreements.sort(
    key=lambda r: r["rank_score"],
    reverse=True,
)

disagreements.sort(
    key=lambda r: r["score"],
    reverse=True,
)


strong_agreements = [
    r
    for r in agreements
    if (
        r["iou"] >= STRONG_AGREE_IOU
        and r["min_conf"] >= STRONG_AGREE_MIN_CONF
    )
][:MAX_AGREEMENTS]

top_disagreements = (
    disagreements[:MAX_DISAGREEMENTS]
)


agreement_crop_paths = []

for idx, row in enumerate(
    strong_agreements
):

    image = cv2.imread(
        row["frame_png"]
    )

    label = (
        f'{row["class_name"]} '
        f'21={row["seed21_score"]:.3f} '
        f'7={row["seed7_score"]:.3f} '
        f'IoU={row["iou"]:.2f}'
    )

    crop = draw_candidate(
        image,
        row["bbox"],
        label,
    )

    if crop is None:
        continue

    name = (
        f"A_{idx:04d}_"
        f'{row["class_name"]}_'
        f'{row["frame_name"]}.jpg'
    )

    path = CROPS_AGREE / name

    cv2.imwrite(
        str(path),
        crop,
    )

    row["candidate_id"] = (
        f"A{idx:04d}"
    )

    row["crop_path"] = str(path)

    agreement_crop_paths.append(
        path
    )


disagreement_crop_paths = []

for idx, row in enumerate(
    top_disagreements
):

    image = cv2.imread(
        row["frame_png"]
    )

    label = (
        f'{row["class_name"]} '
        f'{row["source"]} '
        f'{row["score"]:.3f}'
    )

    crop = draw_candidate(
        image,
        row["bbox"],
        label,
    )

    if crop is None:
        continue

    name = (
        f"D_{idx:04d}_"
        f'{row["class_name"]}_'
        f'{row["frame_name"]}.jpg'
    )

    path = CROPS_DISAGREE / name

    cv2.imwrite(
        str(path),
        crop,
    )

    row["candidate_id"] = (
        f"D{idx:04d}"
    )

    row["crop_path"] = str(path)

    disagreement_crop_paths.append(
        path
    )


(OUT / "agreements.json").write_text(
    json.dumps(
        strong_agreements,
        indent=2,
    ),
    encoding="utf-8",
)

(OUT / "disagreements.json").write_text(
    json.dumps(
        top_disagreements,
        indent=2,
    ),
    encoding="utf-8",
)


with (
    OUT / "review.csv"
).open(
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "candidate_id",
        "type",
        "class_name",
        "frame_name",
        "score_a",
        "score_b",
        "iou",
        "decision",
        "correct_class",
        "correct_box",
        "notes",
        "crop_path",
    ])

    for row in strong_agreements:
        writer.writerow([
            row.get(
                "candidate_id",
                "",
            ),
            "agreement",
            row["class_name"],
            row["frame_name"],
            row["seed21_score"],
            row["seed7_score"],
            row["iou"],
            "",
            "",
            "",
            "",
            row.get(
                "crop_path",
                "",
            ),
        ])

    for row in top_disagreements:
        writer.writerow([
            row.get(
                "candidate_id",
                "",
            ),
            row["source"],
            row["class_name"],
            row["frame_name"],
            row["score"],
            "",
            "",
            "",
            "",
            "",
            "",
            row.get(
                "crop_path",
                "",
            ),
        ])


for start in range(
    0,
    len(agreement_crop_paths),
    50,
):
    make_sheet(
        agreement_crop_paths[
            start:start + 50
        ],
        SHEETS
        / f"agreements_{start:04d}.jpg",
    )


for start in range(
    0,
    len(disagreement_crop_paths),
    50,
):
    make_sheet(
        disagreement_crop_paths[
            start:start + 50
        ],
        SHEETS
        / f"disagreements_{start:04d}.jpg",
    )


print()
print("DONE")
print("total agreements:", len(agreements))
print(
    "strong agreements:",
    len(strong_agreements),
)
print(
    "disagreements:",
    len(disagreements),
)
print(
    "review candidates:",
    len(strong_agreements)
    + len(top_disagreements),
)
print("output:", OUT)
