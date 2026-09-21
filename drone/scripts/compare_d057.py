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
    / "exp_d057a" / "best.pt"
)

old = YOLO(str(OLD))
new = YOLO(str(NEW))

imgs = sorted(CAP.glob("*.png"))

thresholds = [0.03, 0.10, 0.30]

def run(model):
    totals = {t: 0 for t in thresholds}
    cls_counts = {t: Counter() for t in thresholds}

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

        for cls, conf in zip(
            r.boxes.cls.cpu().tolist(),
            r.boxes.conf.cpu().tolist(),
        ):
            cls = int(cls)
            for t in thresholds:
                if conf >= t:
                    totals[t] += 1
                    cls_counts[t][cls] += 1

    return totals, cls_counts

old_t, old_c = run(old)
new_t, new_c = run(new)

print("frames:", len(imgs))

for t in thresholds:
    print()
    print("THRESHOLD", t)
    print("D040 total:", old_t[t])
    print("D057 total:", new_t[t])
    print("D040 classes:", sorted(old_c[t].items()))
    print("D057 classes:", sorted(new_c[t].items()))
