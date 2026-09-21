from pathlib import Path
import json
import random
import shutil

ROOT = Path(__file__).resolve().parents[2]

OUT = (
    ROOT
    / "drone"
    / "artifacts"
    / "exp_d056a"
    / "dataset"
)

IMG_OUT = OUT / "images" / "train"
LBL_OUT = OUT / "labels" / "train"

IMG_OUT.mkdir(parents=True, exist_ok=True)
LBL_OUT.mkdir(parents=True, exist_ok=True)

random.seed(56)

# --------------------------------------------------
# Sources
# --------------------------------------------------

D032 = (
    ROOT
    / "drone"
    / "artifacts"
    / "exp_d032"
    / "dataset"
)

D045 = (
    ROOT
    / "drone"
    / "artifacts"
    / "exp_d045"
    / "dataset"
)

D048 = (
    ROOT
    / "drone"
    / "artifacts"
    / "exp_d048b"
    / "dataset"
)

CAPTURE = (
    ROOT
    / "drone"
    / "captures"
    / "ed590596961d4390b6919a7633824f6b"
)

CAPTURE_FRAMES = CAPTURE / "frames"
CAPTURE_PREDS = CAPTURE / "predictions"

# --------------------------------------------------
# Utility
# --------------------------------------------------

counter = 0

def copy_pair(image_path, label_path, prefix):
    global counter

    suffix = image_path.suffix.lower()

    name = f"{prefix}_{counter:06d}"

    shutil.copy2(
        image_path,
        IMG_OUT / f"{name}{suffix}",
    )

    if label_path is not None and label_path.exists():
        shutil.copy2(
            label_path,
            LBL_OUT / f"{name}.txt",
        )
    else:
        (
            LBL_OUT / f"{name}.txt"
        ).write_text("", encoding="utf-8")

    counter += 1


def add_yolo_dataset(dataset, prefix, limit=None):
    images = sorted(
        (dataset / "images" / "train").glob("*")
    )

    images = [
        p for p in images
        if p.suffix.lower()
        in {".png", ".jpg", ".jpeg"}
    ]

    if limit is not None and len(images) > limit:
        images = random.sample(
            images,
            limit,
        )

    for image_path in images:
        label_path = (
            dataset
            / "labels"
            / "train"
            / f"{image_path.stem}.txt"
        )

        copy_pair(
            image_path,
            label_path,
            prefix,
        )

    print(
        prefix,
        "added:",
        len(images),
    )


# --------------------------------------------------
# Positive data
# --------------------------------------------------

# Keep all D032 real/reference positives.
add_yolo_dataset(
    D032,
    "d032",
)

# Keep all manually reviewed D048 real-domain images.
add_yolo_dataset(
    D048,
    "d048",
)

# D045 has 2200 highly repetitive rare-class synthetic images.
# First experiment: use a balanced subset rather than letting it
# dominate the dataset.
add_yolo_dataset(
    D045,
    "d045",
    limit=600,
)

# --------------------------------------------------
# Hard negatives from captured validation domain.
#
# Conservative rule:
# only use frames where the old detector produced NO
# reasonably confident detections (>= 0.03).
# --------------------------------------------------

negative_candidates = []

for pred_path in sorted(
    CAPTURE_PREDS.glob("*.json")
):
    data = json.loads(
        pred_path.read_text(
            encoding="utf-8"
        )
    )

    strong = [
        ann
        for ann in data["annotations"]
        if float(
            ann["confidence"]
        ) >= 0.03
    ]

    if strong:
        continue

    img_path = (
        CAPTURE_FRAMES
        / f"{pred_path.stem}.png"
    )

    if img_path.exists():
        negative_candidates.append(
            img_path
        )

print(
    "hard-negative candidates:",
    len(negative_candidates),
)

# Avoid overwhelming the positive set.
NEGATIVE_LIMIT = 120

if (
    len(negative_candidates)
    > NEGATIVE_LIMIT
):
    negative_candidates = random.sample(
        negative_candidates,
        NEGATIVE_LIMIT,
    )

for img_path in negative_candidates:
    copy_pair(
        img_path,
        None,
        "hardneg",
    )

print(
    "hard negatives added:",
    len(negative_candidates),
)

# --------------------------------------------------
# YAML
# --------------------------------------------------

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
    yaml_lines.append(
        f"  {i}: {name}"
    )

(
    OUT / "data.yaml"
).write_text(
    "\n".join(yaml_lines) + "\n",
    encoding="utf-8",
)

print()
print("DONE")
print("Total images:", counter)
print("Output:", OUT)
