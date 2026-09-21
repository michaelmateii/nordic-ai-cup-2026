from pathlib import Path
from PIL import Image, ImageDraw
import math

ROOT = Path(__file__).resolve().parents[2]

FRAMES = (
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
    / "exp_d049"
    / "l1_contact_sheets"
)

OUT.mkdir(parents=True, exist_ok=True)

pngs = sorted(FRAMES.glob("frame_*_index_*.png"))

# Only keep frames whose paired JSON indicates L1 by 1920x1080 source region.
import json

selected = []

for png in pngs:
    meta = png.with_suffix(".json")
    if not meta.exists():
        continue

    try:
        d = json.loads(meta.read_text(encoding="utf-8"))
    except Exception:
        continue

    region = None

    def find_region(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if k == "source_region_xyxy" and isinstance(v, list) and len(v) == 4:
                    return v
                r = find_region(v)
                if r is not None:
                    return r
        elif isinstance(x, list):
            for v in x:
                r = find_region(v)
                if r is not None:
                    return r
        return None

    region = find_region(d)

    if region is None:
        continue

    x1, y1, x2, y2 = region

    if round(x2 - x1) == 1920 and round(y2 - y1) == 1080:
        selected.append(png)

print("L1 frames:", len(selected))

COLS = 3
ROWS = 3
PER_SHEET = COLS * ROWS

for start in range(0, len(selected), PER_SHEET):
    chunk = selected[start:start + PER_SHEET]

    tile_w = 640
    tile_h = 390

    sheet = Image.new(
        "RGB",
        (COLS * tile_w, ROWS * tile_h),
        "white",
    )

    draw = ImageDraw.Draw(sheet)

    for j, path in enumerate(chunk):
        img = Image.open(path).convert("RGB")
        img.thumbnail((620, 350))

        x = (j % COLS) * tile_w
        y = (j // COLS) * tile_h

        px = x + (tile_w - img.width) // 2
        py = y + 5

        sheet.paste(img, (px, py))

        draw.text(
            (x + 8, y + 360),
            path.stem,
            fill="black",
        )

    out = OUT / f"sheet_{start:03d}.jpg"
    sheet.save(out, quality=92)

print("Sheets:", math.ceil(len(selected) / PER_SHEET))
print("Output:", OUT)
