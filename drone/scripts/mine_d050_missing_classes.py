from pathlib import Path
from collections import defaultdict
import json
import math

from PIL import Image, ImageDraw
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]

MODEL = (
    ROOT / "drone" / "artifacts" / "exp_d045"
    / "runs" / "yolo11n_synth_missing_v1"
    / "weights" / "best.pt"
)

FRAMES = (
    ROOT / "drone" / "captures"
    / "3224a582bfbf4273a028497662b7aa7c"
    / "frames"
)

OUT = ROOT / "drone" / "artifacts" / "exp_d050"
CROPS = OUT / "crops"
SHEETS = OUT / "contact_sheets"

CROPS.mkdir(parents=True, exist_ok=True)
SHEETS.mkdir(parents=True, exist_ok=True)

TARGETS = {
    "condor",
    "jammer",
    "small_launcher",
    "spacecraft",
    "ta-ta",
}

model = YOLO(str(MODEL))

pngs = sorted(FRAMES.glob("frame_*_index_*.png"))

print("frames:", len(pngs))
print("model:", MODEL)

proposals = []

for i, path in enumerate(pngs):

    result = model.predict(
        source=str(path),
        imgsz=1280,
        conf=0.001,
        iou=0.70,
        max_det=100,
        device=0,
        verbose=False,
    )[0]

    if result.boxes is None:
        continue

    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    clss = result.boxes.cls.cpu().numpy().astype(int)

    for box, conf, cls in zip(boxes, confs, clss):

        name = result.names[int(cls)]

        if name not in TARGETS:
            continue

        proposals.append({
            "frame_path": str(path),
            "frame_name": path.stem,
            "class_name": name,
            "score": float(conf),
            "bbox": [float(x) for x in box],
        })

    if (i + 1) % 20 == 0:
        print(
            f"{i+1}/{len(pngs)} "
            f"proposals={len(proposals)}"
        )


# -------------------------------------------------------
# Rank + temporal dedupe.
# Keep strongest examples, but don't fill sheets with
# adjacent frames containing the same physical object.
# -------------------------------------------------------

by_class = defaultdict(list)

for p in proposals:
    by_class[p["class_name"]].append(p)


selected = []

for cls in sorted(TARGETS):

    items = sorted(
        by_class[cls],
        key=lambda x: x["score"],
        reverse=True,
    )

    kept = []
    used_frames = []

    for p in items:

        try:
            frame_idx = int(
                p["frame_name"].split("_index_")[-1]
            )
        except Exception:
            frame_idx = None

        if frame_idx is not None:
            if any(
                abs(frame_idx - old) <= 2
                for old in used_frames
            ):
                continue

        kept.append(p)

        if frame_idx is not None:
            used_frames.append(frame_idx)

        # 20 per class is enough for manual review.
        if len(kept) >= 20:
            break

    selected.extend(kept)


selected.sort(
    key=lambda x: (
        x["class_name"],
        -x["score"],
    )
)


# -------------------------------------------------------
# Crop with context.
# -------------------------------------------------------

for i, p in enumerate(selected):

    p["selection_id"] = f"D50_{i:03d}"

    img = Image.open(p["frame_path"]).convert("RGB")
    w, h = img.size

    x1, y1, x2, y2 = p["bbox"]

    bw = x2 - x1
    bh = y2 - y1

    # Give lots of context around tiny targets.
    padx = max(50, bw * 2.5)
    pady = max(50, bh * 2.5)

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    xx1 = max(0, int(cx - bw/2 - padx))
    yy1 = max(0, int(cy - bh/2 - pady))
    xx2 = min(w, int(cx + bw/2 + padx))
    yy2 = min(h, int(cy + bh/2 + pady))

    crop = img.crop((xx1, yy1, xx2, yy2))

    draw = ImageDraw.Draw(crop)

    draw.rectangle(
        [
            x1 - xx1,
            y1 - yy1,
            x2 - xx1,
            y2 - yy1,
        ],
        outline="lime",
        width=3,
    )

    crop_path = CROPS / f'{p["selection_id"]}.jpg'
    crop.save(crop_path, quality=95)

    p["crop_path"] = str(crop_path)


(OUT / "proposals.json").write_text(
    json.dumps(selected, indent=2),
    encoding="utf-8",
)


# -------------------------------------------------------
# Contact sheets.
# -------------------------------------------------------

tiles = []

for p in selected:

    img = Image.open(p["crop_path"]).convert("RGB")
    img.thumbnail((320, 220))

    tile = Image.new(
        "RGB",
        (340, 270),
        "white",
    )

    px = (340 - img.width) // 2
    tile.paste(img, (px, 5))

    draw = ImageDraw.Draw(tile)

    draw.text(
        (5, 230),
        f'{p["selection_id"]} {p["class_name"]}',
        fill="black",
    )

    draw.text(
        (5, 248),
        f'conf={p["score"]:.4f} {p["frame_name"]}',
        fill="black",
    )

    tiles.append(tile)


COLS = 5
PER_SHEET = 50

for start in range(0, len(tiles), PER_SHEET):

    chunk = tiles[start:start+PER_SHEET]
    rows = math.ceil(len(chunk) / COLS)

    sheet = Image.new(
        "RGB",
        (COLS * 340, rows * 270),
        (230, 230, 230),
    )

    for j, tile in enumerate(chunk):
        x = (j % COLS) * 340
        y = (j // COLS) * 270
        sheet.paste(tile, (x, y))

    sheet.save(
        SHEETS / f"d050_{start:03d}.jpg",
        quality=94,
    )


print()
print("ALL raw proposals:", len(proposals))
print("selected:", len(selected))

for cls in sorted(TARGETS):
    vals = [x for x in selected if x["class_name"] == cls]

    if vals:
        print(
            f"{cls:18s} "
            f"{len(vals):2d} "
            f"best={vals[0]['score']:.4f} "
            f"worst={vals[-1]['score']:.4f}"
        )
    else:
        print(f"{cls:18s} 0")

print("output:", OUT)
