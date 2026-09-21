from pathlib import Path
import random
import shutil

ROOT = Path(__file__).resolve().parents[2]

DATA = (
    ROOT
    / "drone"
    / "artifacts"
    / "exp_d056a"
    / "dataset"
)

IMG_TRAIN = DATA / "images" / "train"
LBL_TRAIN = DATA / "labels" / "train"

IMG_VAL = DATA / "images" / "val"
LBL_VAL = DATA / "labels" / "val"

IMG_VAL.mkdir(parents=True, exist_ok=True)
LBL_VAL.mkdir(parents=True, exist_ok=True)

random.seed(56)

images = sorted(
    p for p in IMG_TRAIN.iterdir()
    if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
)

# Keep roughly 15% for validation.
n_val = max(
    1,
    round(len(images) * 0.15),
)

chosen = set(
    random.sample(images, n_val)
)

for img in chosen:
    label = LBL_TRAIN / f"{img.stem}.txt"

    shutil.move(
        str(img),
        IMG_VAL / img.name,
    )

    shutil.move(
        str(label),
        LBL_VAL / label.name,
    )

yaml_path = DATA / "data.yaml"

text = yaml_path.read_text(
    encoding="utf-8"
)

text = text.replace(
    "val: images/train",
    "val: images/val",
)

yaml_path.write_text(
    text,
    encoding="utf-8",
)

print("train:", len(images) - n_val)
print("val:", n_val)
print("total:", len(images))
