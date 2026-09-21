import json
import math
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]

SRC = (
    ROOT
    / "drone"
    / "artifacts"
    / "exp_d048a"
    / "selected_disagreements.json"
)

OUT = (
    ROOT
    / "drone"
    / "artifacts"
    / "exp_d048a"
    / "contact_sheets"
)

OUT.mkdir(parents=True, exist_ok=True)

rows = json.loads(
    SRC.read_text(encoding="utf-8")
)

tiles = []

for r in rows:
    crop = Path(r["crop_path"])

    if not crop.exists():
        print("Missing crop:", crop)
        continue

    img = Image.open(crop).convert("RGB")
    img.thumbnail((320, 220))

    tile = Image.new(
        "RGB",
        (340, 265),
        "white",
    )

    x = (340 - img.width) // 2
    tile.paste(img, (x, 5))

    draw = ImageDraw.Draw(tile)

    draw.text(
        (5, 228),
        f'{r["selection_id"]} {r["class_name"]}',
        fill="black",
    )

    draw.text(
        (5, 245),
        f'{r["source"]} conf={r["score"]:.3f}',
        fill="black",
    )

    tiles.append(tile)

COLS = 5
PER_SHEET = 50

for start in range(0, len(tiles), PER_SHEET):
    chunk = tiles[start:start + PER_SHEET]
    rows_n = math.ceil(len(chunk) / COLS)

    sheet = Image.new(
        "RGB",
        (COLS * 340, rows_n * 265),
        (230, 230, 230),
    )

    for j, tile in enumerate(chunk):
        sx = (j % COLS) * 340
        sy = (j // COLS) * 265
        sheet.paste(tile, (sx, sy))

    sheet.save(
        OUT / f"d048_{start:03d}.jpg",
        quality=92,
    )

print("Candidates:", len(tiles))
print("Sheets:", math.ceil(len(tiles) / PER_SHEET))
print("Output:", OUT)
