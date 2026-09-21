import json
import shutil
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]

BASE = ROOT / "drone" / "artifacts" / "exp_d047b" / "dataset"
SELECTED = ROOT / "drone" / "artifacts" / "exp_d048a" / "selected_disagreements.json"
ACCEPTED = ROOT / "drone" / "artifacts" / "exp_d048a" / "accepted_ids.json"
OUT = ROOT / "drone" / "artifacts" / "exp_d048b" / "dataset"

IMG_OUT = OUT / "images" / "train"
LBL_OUT = OUT / "labels" / "train"

if OUT.exists():
    shutil.rmtree(OUT)

shutil.copytree(BASE, OUT)

rows = json.loads(SELECTED.read_text(encoding="utf-8"))
accepted_ids = set(json.loads(ACCEPTED.read_text(encoding="utf-8-sig")))

accepted = [
    r for r in rows
    if r["selection_id"] in accepted_ids
]

def yolo_to_xyxy(line, width, height):
    parts = line.split()
    cls = int(parts[0])
    cx, cy, bw, bh = map(float, parts[1:5])

    cx *= width
    cy *= height
    bw *= width
    bh *= height

    return cls, [
        cx - bw / 2,
        cy - bh / 2,
        cx + bw / 2,
        cy + bh / 2,
    ]

def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)

    inter = iw * ih
    if inter <= 0:
        return 0.0

    aa = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    bb = max(0, bx2 - bx1) * max(0, by2 - by1)

    return inter / max(1e-9, aa + bb - inter)

added = 0
duplicates = 0
new_images = 0

for r in accepted:
    src_img = Path(r["frame_png"])
    if not src_img.exists():
        print("MISSING:", src_img)
        continue

    stem = "d047_" + src_img.stem
    dst_img = IMG_OUT / f"{stem}.png"
    dst_lbl = LBL_OUT / f"{stem}.txt"

    if not dst_img.exists():
        shutil.copy2(src_img, dst_img)
        new_images += 1

    with Image.open(src_img) as im:
        width, height = im.size

    cls = int(r["class_id"])
    x1, y1, x2, y2 = map(float, r["bbox"])

    existing_lines = []
    existing_boxes = []

    if dst_lbl.exists():
        existing_lines = [
            x for x in dst_lbl.read_text(encoding="utf-8").splitlines()
            if x.strip()
        ]

        for line in existing_lines:
            ecls, ebox = yolo_to_xyxy(line, width, height)
            existing_boxes.append((ecls, ebox))

    is_duplicate = any(
        ecls == cls and iou([x1, y1, x2, y2], ebox) >= 0.60
        for ecls, ebox in existing_boxes
    )

    if is_duplicate:
        duplicates += 1
        continue

    cx = ((x1 + x2) / 2) / width
    cy = ((y1 + y2) / 2) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height

    existing_lines.append(
        f"{cls} {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f}"
    )

    dst_lbl.write_text(
        "\n".join(existing_lines) + "\n",
        encoding="utf-8",
    )

    added += 1

print("Accepted D048 candidates:", len(accepted))
print("Actually added:", added)
print("Skipped duplicates:", duplicates)
print("New frames copied:", new_images)
print("Total images:", len(list(IMG_OUT.glob("*.png"))))

total_boxes = 0
for p in LBL_OUT.glob("*.txt"):
    total_boxes += sum(
        1 for x in p.read_text(encoding="utf-8").splitlines()
        if x.strip()
    )

print("Total boxes:", total_boxes)
print("Output:", OUT)
