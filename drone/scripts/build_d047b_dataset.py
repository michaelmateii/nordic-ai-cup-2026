import json
import shutil
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parents[2]

BASE = ROOT / "drone" / "artifacts" / "exp_d032" / "dataset"
SELECTED = (
    ROOT / "drone" / "artifacts" / "exp_d047a"
    / "selected_candidates.json"
)

OUT = ROOT / "drone" / "artifacts" / "exp_d047b" / "dataset"

IMG_OUT = OUT / "images" / "train"
LBL_OUT = OUT / "labels" / "train"

if OUT.exists():
    shutil.rmtree(OUT)

IMG_OUT.mkdir(parents=True)
LBL_OUT.mkdir(parents=True)

# ---------------------------------------------------------
# 1. Preserve all original D032 real-domain supervision.
# ---------------------------------------------------------

for img in (BASE / "images" / "train").glob("*.png"):
    shutil.copy2(img, IMG_OUT / img.name)

for lbl in (BASE / "labels" / "train").glob("*.txt"):
    shutil.copy2(lbl, LBL_OUT / lbl.name)


# ---------------------------------------------------------
# 2. Group selected D047 pseudo-labels by source frame.
#    Multiple objects from one frame must share one label file.
# ---------------------------------------------------------

rows = json.loads(
    SELECTED.read_text(encoding="utf-8")
)

by_frame = defaultdict(list)

for row in rows:
    by_frame[row["frame_png"]].append(row)


new_boxes = 0
new_images = 0


for frame_png_str, frame_rows in by_frame.items():

    src_img = Path(frame_png_str)

    if not src_img.exists():
        print("MISSING IMAGE:", src_img)
        continue

    # Prefix avoids collision with D032 filenames.
    out_stem = "d047_" + src_img.stem

    dst_img = IMG_OUT / f"{out_stem}.png"
    dst_lbl = LBL_OUT / f"{out_stem}.txt"

    shutil.copy2(
        src_img,
        dst_img,
    )

    # Captured L1 request image geometry.
    from PIL import Image

    with Image.open(src_img) as im:
        width, height = im.size

    label_lines = []

    for row in frame_rows:

        cls = int(row["class_id"])

        x1, y1, x2, y2 = row["bbox"]

        # Clip just in case.
        x1 = max(0.0, min(float(width), x1))
        x2 = max(0.0, min(float(width), x2))
        y1 = max(0.0, min(float(height), y1))
        y2 = max(0.0, min(float(height), y2))

        bw = x2 - x1
        bh = y2 - y1

        if bw <= 1 or bh <= 1:
            continue

        cx = (x1 + x2) / 2.0 / width
        cy = (y1 + y2) / 2.0 / height
        nw = bw / width
        nh = bh / height

        label_lines.append(
            f"{cls} "
            f"{cx:.8f} "
            f"{cy:.8f} "
            f"{nw:.8f} "
            f"{nh:.8f}"
        )

        new_boxes += 1

    if label_lines:
        dst_lbl.write_text(
            "\n".join(label_lines) + "\n",
            encoding="utf-8",
        )

        new_images += 1
    else:
        dst_img.unlink(missing_ok=True)


# ---------------------------------------------------------
# 3. Dataset YAML
# ---------------------------------------------------------

names = [
    "condor",
    "hangar",
    "helicopter",
    "jammer",
    "jet_plane",
    "large_launcher",
    "large_tower",
    "medium_launcher",
    "medium_plane",
    "mine_roller",
    "small_launcher",
    "small_plane",
    "small_tower",
    "spacecraft",
    "ta-ta",
    "tank",
]

yaml_lines = [
    f"path: {OUT.as_posix()}",
    "train: images/train",
    "val: images/train",
    "names:",
]

for i, name in enumerate(names):
    yaml_lines.append(f"  {i}: {name}")

(OUT / "data.yaml").write_text(
    "\n".join(yaml_lines) + "\n",
    encoding="utf-8",
)


# ---------------------------------------------------------
# 4. Final sanity report.
# ---------------------------------------------------------

counts = Counter()

label_files = list(LBL_OUT.glob("*.txt"))

for p in label_files:
    for line in p.read_text(
        encoding="utf-8"
    ).splitlines():

        if line.strip():
            counts[int(line.split()[0])] += 1


print()
print("DONE")
print("new D047 images:", new_images)
print("new D047 boxes:", new_boxes)
print("total images:", len(list(IMG_OUT.glob('*.png'))))
print("total label files:", len(label_files))
print("total boxes:", sum(counts.values()))

for cls, n in sorted(counts.items()):
    print(f"{cls:2d} {names[cls]:18s} {n}")
