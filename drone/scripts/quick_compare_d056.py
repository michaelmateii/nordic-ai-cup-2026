from pathlib import Path
from collections import Counter
import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]

CAP = (
    ROOT / "drone" / "captures"
    / "ed590596961d4390b6919a7633824f6b"
    / "frames"
)

OLD = (
    ROOT / "drone" / "artifacts"
    / "exp_d040" / "seed_21"
    / "weights" / "best.pt"
)

NEW = (
    ROOT / "drone" / "artifacts"
    / "exp_d056a" / "best.pt"
)

old = YOLO(str(OLD))
new = YOLO(str(NEW))

imgs = sorted(CAP.glob("*.png"))

# Sample every ~5th captured frame.
imgs = imgs[::5]

thresholds = [0.001, 0.01, 0.03, 0.10]

def run(model):
    stats = {t: Counter() for t in thresholds}
    totals = {t: 0 for t in thresholds}

    for p in imgs:
        im = cv2.imread(str(p))

        r = model.predict(
            im,
            imgsz=960,
            conf=0.001,
            iou=0.70,
            max_det=300,
            device=0,
            verbose=False,
        )[0]

        if r.boxes is None:
            continue

        for cls, conf in zip(
            r.boxes.cls.cpu().tolist(),
            r.boxes.conf.cpu().tolist(),
        ):
            cls = int(cls)

            for t in thresholds:
                if conf >= t:
                    totals[t] += 1
                    stats[t][cls] += 1

    return totals, stats

old_totals, old_stats = run(old)
new_totals, new_stats = run(new)

print("frames:", len(imgs))

for t in thresholds:
    print()
    print("THRESHOLD", t)
    print("D040 total:", old_totals[t])
    print("D056 total:", new_totals[t])
    print("ratio:", round(
        new_totals[t] / max(1, old_totals[t]),
        3
    ))

    print("D040:", sorted(old_stats[t].items()))
    print("D056:", sorted(new_stats[t].items()))
