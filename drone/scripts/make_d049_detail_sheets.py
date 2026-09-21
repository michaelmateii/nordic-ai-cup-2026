import json
from pathlib import Path
from PIL import Image, ImageDraw
import math

ROOT = Path(__file__).resolve().parents[2]

FRAMES = (
    ROOT / "drone" / "captures"
    / "3224a582bfbf4273a028497662b7aa7c"
    / "frames"
)

OUT = (
    ROOT / "drone" / "artifacts"
    / "exp_d049" / "quadrant_sheets"
)
OUT.mkdir(parents=True, exist_ok=True)


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


frames = []

for png in sorted(FRAMES.glob("frame_*_index_*.png")):
    meta = png.with_suffix(".json")
    if not meta.exists():
        continue

    try:
        d = json.loads(meta.read_text(encoding="utf-8"))
    except Exception:
        continue

    r = find_region(d)
    if r is None:
        continue

    x1, y1, x2, y2 = r

    if round(x2-x1) == 1920 and round(y2-y1) == 1080:
        frames.append(png)


tiles = []

for png in frames:
    im = Image.open(png).convert("RGB")
    w, h = im.size

    # overlapping 2x2 crops
    boxes = [
        (0, 0, w//2 + 120, h//2 + 80),
        (w//2 - 120, 0, w, h//2 + 80),
        (0, h//2 - 80, w//2 + 120, h),
        (w//2 - 120, h//2 - 80, w, h),
    ]

    for q, box in enumerate(boxes):
        crop = im.crop(box)

        # Keep much more detail than the full-frame sheets.
        crop.thumbnail((800, 500))

        tile = Image.new("RGB", (820, 540), "white")
        x = (820 - crop.width)//2
        tile.paste(crop, (x, 5))

        draw = ImageDraw.Draw(tile)
        draw.text(
            (8, 512),
            f"{png.stem} Q{q}",
            fill="black",
        )

        tiles.append(tile)


COLS = 2
ROWS = 2
PER_SHEET = 4

for start in range(0, len(tiles), PER_SHEET):
    chunk = tiles[start:start+PER_SHEET]

    sheet = Image.new(
        "RGB",
        (COLS*820, ROWS*540),
        (225,225,225),
    )

    for j, tile in enumerate(chunk):
        x = (j % COLS)*820
        y = (j // COLS)*540
        sheet.paste(tile, (x,y))

    sheet.save(
        OUT / f"detail_{start:04d}.jpg",
        quality=95,
    )

print("L1 frames:", len(frames))
print("detail tiles:", len(tiles))
print("sheets:", math.ceil(len(tiles)/4))
print("output:", OUT)
