import json
import random
import shutil
from pathlib import Path
from collections import Counter

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

HELSINKI = (
    ROOT
    / "drone"
    / "reference"
    / "official-drone-flyby"
    / "src"
    / "helsinki"
)

H_IMAGES = HELSINKI / "images"
H_ANN = HELSINKI / "annotations"

CAPTURE = (
    ROOT
    / "drone"
    / "captures"
    / "ed590596961d4390b6919a7633824f6b"
)

CAP_FRAMES = CAPTURE / "frames"
CAP_PREDS = CAPTURE / "predictions"

OUT = (
    ROOT
    / "drone"
    / "artifacts"
    / "exp_d057a"
    / "dataset"
)

D048 = (
    ROOT
    / "drone"
    / "artifacts"
    / "exp_d048b"
    / "dataset"
)

N_SYNTH = 3000
VAL_FRAC = 0.08
NEG_FRAC = 0.15
REAL_BG_FRAC = 0.65

RNG = random.Random(57)

VW = 960
VH = 540

CLASS_NAMES = [
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

CLASS_INDEX = {
    name: i
    for i, name in enumerate(CLASS_NAMES)
}

SOURCE_SIZE = {
    0: (3840, 2160),
    1: (1920, 1080),
    2: (960, 540),
}

LEVEL_SCALE = {
    0: 0.25,
    1: 0.50,
    2: 1.00,
}


def load_ann(path):
    obj = json.loads(
        path.read_text(encoding="utf-8")
    )

    if isinstance(obj, list):
        rows = obj

    elif isinstance(obj, dict):
        for key in (
            "annotations",
            "objects",
            "detections",
            "labels",
        ):
            if key in obj and isinstance(obj[key], list):
                rows = obj[key]
                break
        else:
            raise RuntimeError(
                f"Unknown annotation JSON structure: {path}"
            )
    else:
        raise RuntimeError(
            f"Unsupported annotation structure: {path}"
        )

    out = []

    for a in rows:
        name = (
            a.get("object_id")
            or a.get("class")
            or a.get("label")
        )

        bbox = a.get("bbox")

        if name not in CLASS_INDEX:
            continue

        if not bbox or len(bbox) != 4:
            continue

        x1, y1, x2, y2 = map(float, bbox)

        out.append(
            {
                "object_id": name,
                "bbox": (
                    x1,
                    y1,
                    x2,
                    y2,
                ),
            }
        )

    return out


def frame_number(path):
    digits = "".join(
        c for c in path.stem
        if c.isdigit()
    )
    return int(digits[-6:])


def extract_patches():
    patches = {
        name: []
        for name in CLASS_NAMES
    }

    image_files = sorted(
        H_IMAGES.glob("*.png")
    )

    ann_files = sorted(
        H_ANN.glob("*.json")
    )

    anns_by_num = {
        frame_number(p): p
        for p in ann_files
    }

    for img_path in image_files:
        n = frame_number(img_path)

        if n not in anns_by_num:
            continue

        img = cv2.imread(str(img_path))

        if img is None:
            continue

        anns = load_ann(
            anns_by_num[n]
        )

        for a in anns:
            x1, y1, x2, y2 = a["bbox"]

            # Only use fully visible objects.
            if (
                x1 <= 2
                or y1 <= 2
                or x2 >= 3838
                or y2 >= 2158
            ):
                continue

            margin = 3

            X1 = max(
                0,
                int(np.floor(x1)) - margin,
            )
            Y1 = max(
                0,
                int(np.floor(y1)) - margin,
            )
            X2 = min(
                3840,
                int(np.ceil(x2)) + margin,
            )
            Y2 = min(
                2160,
                int(np.ceil(y2)) + margin,
            )

            patch = img[
                Y1:Y2,
                X1:X2,
            ].copy()

            local_box = (
                x1 - X1,
                y1 - Y1,
                x2 - X1,
                y2 - Y1,
            )

            patches[
                a["object_id"]
            ].append(
                (
                    patch,
                    local_box,
                )
            )

    return patches


def feather_mask(h, w, r):
    m = np.ones(
        (h, w),
        np.float32,
    )

    r = max(
        1,
        min(
            r,
            h // 3,
            w // 3,
        ),
    )

    ramp = np.linspace(
        0,
        1,
        r + 1,
    )[1:]

    m[:r, :] *= ramp[:, None]
    m[-r:, :] *= ramp[::-1][:, None]
    m[:, :r] *= ramp[None, :]
    m[:, -r:] *= ramp[::-1][None, :]

    return cv2.GaussianBlur(
        m,
        (0, 0),
        0.6,
    )


def orient(patch, box):
    x1, y1, x2, y2 = box

    if RNG.random() < 0.5:
        w = patch.shape[1]

        patch = patch[:, ::-1].copy()

        x1, x2 = (
            w - x2,
            w - x1,
        )

    k = RNG.randint(0, 3)

    for _ in range(k):
        old_h, old_w = patch.shape[:2]

        patch = np.ascontiguousarray(
            np.rot90(patch)
        )

        x1, y1, x2, y2 = (
            y1,
            old_w - x2,
            y2,
            old_w - x1,
        )

    return (
        patch,
        (x1, y1, x2, y2),
    )


def paste_object(
    bg,
    patch,
    box,
    level,
):
    scale = (
        LEVEL_SCALE[level]
        * RNG.uniform(0.75, 1.30)
    )

    ph, pw = patch.shape[:2]

    nh = max(
        2,
        int(round(ph * scale)),
    )

    nw = max(
        2,
        int(round(pw * scale)),
    )

    if (
        nh >= VH - 4
        or nw >= VW - 4
    ):
        return None

    interpolation = (
        cv2.INTER_AREA
        if scale < 1
        else cv2.INTER_LINEAR
    )

    p = cv2.resize(
        patch,
        (nw, nh),
        interpolation=interpolation,
    )

    p = np.clip(
        p.astype(np.float32)
        * RNG.uniform(0.80, 1.20)
        + RNG.uniform(-15, 15),
        0,
        255,
    ).astype(np.uint8)

    x0 = RNG.randint(
        0,
        VW - nw - 1,
    )

    y0 = RNG.randint(
        0,
        VH - nh - 1,
    )

    mask = feather_mask(
        nh,
        nw,
        max(
            2,
            int(
                0.22
                * min(nh, nw)
            ),
        ),
    )[..., None]

    roi = bg[
        y0:y0 + nh,
        x0:x0 + nw,
    ].astype(np.float32)

    bg[
        y0:y0 + nh,
        x0:x0 + nw,
    ] = (
        roi * (1.0 - mask)
        + p.astype(np.float32) * mask
    ).astype(np.uint8)

    bx1, by1, bx2, by2 = box

    return (
        x0 + bx1 * scale,
        y0 + by1 * scale,
        x0 + bx2 * scale,
        y0 + by2 * scale,
    )


def make_helsinki_background(level):
    img_path = RNG.choice(
        list(H_IMAGES.glob("*.png"))
    )

    img = cv2.imread(
        str(img_path)
    )

    sw, sh = SOURCE_SIZE[level]

    x = RNG.randint(
        0,
        3840 - sw,
    )

    y = RNG.randint(
        0,
        2160 - sh,
    )

    crop = img[
        y:y + sh,
        x:x + sw,
    ]

    return cv2.resize(
        crop,
        (VW, VH),
        interpolation=cv2.INTER_AREA,
    )


def build_hard_negative_pool():
    pool = []

    for pred_path in sorted(
        CAP_PREDS.glob("*.json")
    ):
        data = json.loads(
            pred_path.read_text(
                encoding="utf-8"
            )
        )

        strong = [
            a
            for a in data.get(
                "annotations",
                [],
            )
            if float(
                a.get(
                    "confidence",
                    0.0,
                )
            ) >= 0.03
        ]

        if strong:
            continue

        img_path = (
            CAP_FRAMES
            / f"{pred_path.stem}.png"
        )

        meta_path = (
            CAP_FRAMES
            / f"{pred_path.stem}.json"
        )

        if (
            img_path.exists()
            and meta_path.exists()
        ):
            meta = json.loads(
                meta_path.read_text(
                    encoding="utf-8"
                )
            )

            # Metadata structures differed between
            # experiments, so try several forms.
            level = None

            if isinstance(
                meta.get("view"),
                dict,
            ):
                level = meta["view"].get(
                    "resolution_level"
                )

            if level is None:
                level = meta.get(
                    "resolution_level"
                )

            if level is None:
                continue

            pool.append(
                (
                    int(level),
                    img_path,
                )
            )

    return pool


def copy_d048():
    added = 0

    for split in ("train", "val"):
        src_i = (
            D048
            / "images"
            / split
        )

        src_l = (
            D048
            / "labels"
            / split
        )

        if not src_i.exists():
            continue

        for image in src_i.iterdir():
            if image.suffix.lower() not in {
                ".png",
                ".jpg",
                ".jpeg",
            }:
                continue

            label = (
                src_l
                / f"{image.stem}.txt"
            )

            if not label.exists():
                continue

            target_split = (
                "val"
                if RNG.random()
                < VAL_FRAC
                else "train"
            )

            name = (
                f"d048_{added:05d}"
                + image.suffix.lower()
            )

            shutil.copy2(
                image,
                OUT
                / "images"
                / target_split
                / name,
            )

            shutil.copy2(
                label,
                OUT
                / "labels"
                / target_split
                / f"{Path(name).stem}.txt",
            )

            added += 1

    return added


for split in ("train", "val"):
    (
        OUT
        / "images"
        / split
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        OUT
        / "labels"
        / split
    ).mkdir(
        parents=True,
        exist_ok=True,
    )


patches = extract_patches()

print(
    "patches per class:"
)

for name in CLASS_NAMES:
    print(
        f"{name:18s}",
        len(patches[name]),
    )

missing = [
    name
    for name in CLASS_NAMES
    if not patches[name]
]

if missing:
    raise RuntimeError(
        f"Missing object patches for classes: {missing}"
    )


hard_negatives = (
    build_hard_negative_pool()
)

print(
    "hard-negative backgrounds:",
    len(hard_negatives),
)

counts = Counter()


for i in range(N_SYNTH):
    split = (
        "val"
        if RNG.random() < VAL_FRAC
        else "train"
    )

    # L1-heavy because our deployed policy spends
    # most of its time there.
    r = RNG.random()

    if r < 0.10:
        wanted_level = 0
    elif r < 0.70:
        wanted_level = 1
    else:
        wanted_level = 2

    use_real = (
        hard_negatives
        and RNG.random()
        < REAL_BG_FRAC
    )

    if use_real:
        matches = [
            item
            for item in hard_negatives
            if item[0] == wanted_level
        ]

        if not matches:
            matches = hard_negatives

        level, path = RNG.choice(
            matches
        )

        bg = cv2.imread(
            str(path)
        )

        if bg is None:
            continue

        if bg.shape[:2] != (
            VH,
            VW,
        ):
            bg = cv2.resize(
                bg,
                (VW, VH),
            )

    else:
        level = wanted_level

        bg = (
            make_helsinki_background(
                level
            )
        )

    bg = np.ascontiguousarray(
        bg
    )

    # Whole-image domain jitter.
    if RNG.random() < 0.5:
        hsv = cv2.cvtColor(
            bg,
            cv2.COLOR_BGR2HSV,
        ).astype(np.float32)

        hsv[..., 1] *= RNG.uniform(
            0.70,
            1.30,
        )

        hsv[..., 2] *= RNG.uniform(
            0.75,
            1.25,
        )

        bg = cv2.cvtColor(
            np.clip(
                hsv,
                0,
                255,
            ).astype(np.uint8),
            cv2.COLOR_HSV2BGR,
        )

    labels = []

    is_negative = (
        RNG.random() < NEG_FRAC
    )

    if not is_negative:
        n_obj = RNG.randint(
            1,
            6,
        )

        for _ in range(n_obj):
            cls_name = RNG.choice(
                CLASS_NAMES
            )

            patch, box = RNG.choice(
                patches[cls_name]
            )

            patch = patch.copy()

            patch, box = orient(
                patch,
                box,
            )

            pasted = paste_object(
                bg,
                patch,
                box,
                level,
            )

            if pasted is None:
                continue

            x1, y1, x2, y2 = pasted

            if (
                x2 - x1 < 4
                or y2 - y1 < 4
            ):
                continue

            labels.append(
                (
                    CLASS_INDEX[
                        cls_name
                    ],
                    (x1 + x2)
                    / 2
                    / VW,
                    (y1 + y2)
                    / 2
                    / VH,
                    (x2 - x1)
                    / VW,
                    (y2 - y1)
                    / VH,
                )
            )

            counts[
                cls_name
            ] += 1

    name = (
        f"s{i:05d}_L{level}"
    )

    cv2.imwrite(
        str(
            OUT
            / "images"
            / split
            / f"{name}.jpg"
        ),
        bg,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            92,
        ],
    )

    with open(
        OUT
        / "labels"
        / split
        / f"{name}.txt",
        "w",
        encoding="utf-8",
    ) as f:
        for (
            cls,
            x,
            y,
            w,
            h,
        ) in labels:
            f.write(
                f"{cls} "
                f"{x:.6f} "
                f"{y:.6f} "
                f"{w:.6f} "
                f"{h:.6f}\n"
            )

    if i % 250 == 0:
        print(
            "generated",
            i,
        )


d048_added = copy_d048()


yaml = [
    f"path: {OUT.as_posix()}",
    "train: images/train",
    "val: images/val",
    "names:",
]

for i, name in enumerate(
    CLASS_NAMES
):
    yaml.append(
        f"  {i}: {name}"
    )

(
    OUT
    / "data.yaml"
).write_text(
    "\n".join(yaml)
    + "\n",
    encoding="utf-8",
)


print()
print("DONE")
print(
    "Synthetic:",
    N_SYNTH,
)
print(
    "D048 added:",
    d048_added,
)
print(
    "hard-negative pool:",
    len(hard_negatives),
)

print()
print(
    "synthetic pasted objects:"
)

for name in CLASS_NAMES:
    print(
        f"{name:18s}",
        counts[name],
    )

print(
    "Output:",
    OUT,
)
