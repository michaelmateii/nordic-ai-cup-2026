import json
import math
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[2]

SRC = (
    ROOT
    / "drone"
    / "artifacts"
    / "exp_d047a"
    / "selected_candidates.json"
)

OUT = (
    ROOT
    / "drone"
    / "artifacts"
    / "exp_d047a"
    / "contact_sheets"
)

OUT.mkdir(parents=True, exist_ok=True)

rows = json.loads(
    SRC.read_text(encoding="utf-8")
)

tiles = []

for i, row in enumerate(rows):

    crop_path = Path(row["crop_path"])

    img = Image.open(crop_path).convert("RGB")
    img.thumbnail((320, 220))

    tile = Image.new(
        "RGB",
        (340, 265),
        "white",
    )

    x = (340 - img.width) // 2
    tile.paste(img, (x, 5))

    draw = ImageDraw.Draw(tile)

    label = (
        f'{i:03d} '
        f'{row["class_name"]} '
        f'c={row["min_conf"]:.3f} '
        f'iou={row["iou"]:.2f}'
    )

    draw.text(
        (5, 228),
        label,
        fill="black",
    )

    tile_id = f"S{i:03d}"

    row["selection_id"] = tile_id

    draw.text(
        (5, 245),
        tile_id,
        fill="black",
    )

    tiles.append(tile)

SRC.write_text(
    json.dumps(rows, indent=2),
    encoding="utf-8",
)

COLS = 5
PER_SHEET = 50

for start in range(
    0,
    len(tiles),
    PER_SHEET,
):

    chunk = tiles[start:start + PER_SHEET]

    rows_n = math.ceil(
        len(chunk) / COLS
    )

    sheet = Image.new(
        "RGB",
        (
            COLS * 340,
            rows_n * 265,
        ),
        (230, 230, 230),
    )

    for j, tile in enumerate(chunk):

        sx = (j % COLS) * 340
        sy = (j // COLS) * 265

        sheet.paste(
            tile,
            (sx, sy),
        )

    destination = (
        OUT
        / f"selected_{start:03d}.jpg"
    )

    sheet.save(
        destination,
        quality=92,
    )

print("DONE")
print("selected candidates:", len(rows))
print("sheets:", math.ceil(len(rows) / PER_SHEET))
print("output:", OUT)
