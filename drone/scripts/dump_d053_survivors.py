import json
from pathlib import Path

P = Path(
    r"drone/captures/ed590596961d4390b6919a7633824f6b/predictions"
)

for f in sorted(P.glob("*.json")):
    data = json.loads(
        f.read_text(encoding="utf-8")
    )

    for ann in data["annotations"]:
        conf = float(ann["confidence"])

        if conf >= 0.020:
            print(
                f"{f.stem} "
                f"{ann['object_id']:20s} "
                f"{conf:.4f}"
            )
