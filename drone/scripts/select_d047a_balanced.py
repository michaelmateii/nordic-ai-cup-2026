import json
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "drone" / "artifacts" / "exp_d047" / "agreements.json"
OUT = ROOT / "drone" / "artifacts" / "exp_d047a"

OUT.mkdir(parents=True, exist_ok=True)

CAPS = {
    "hangar": 999,
    "helicopter": 10,
    "jet_plane": 999,
    "large_launcher": 999,
    "large_tower": 12,
    "medium_launcher": 999,
    "medium_plane": 999,
    "mine_roller": 999,
    "small_plane": 12,
    "small_tower": 999,
    "tank": 20,
}

rows = json.loads(SRC.read_text(encoding="utf-8"))

rows = [
    r for r in rows
    if r["min_conf"] >= 0.70
    and r["iou"] >= 0.80
]

by_class = defaultdict(list)

for r in rows:
    by_class[r["class_name"]].append(r)

selected = []

for cls, items in sorted(by_class.items()):
    items.sort(
        key=lambda r: (
            r["min_conf"],
            r["iou"],
            r["rank_score"],
        ),
        reverse=True,
    )

    cap = CAPS.get(cls, 999)

    selected.extend(items[:cap])

selected.sort(
    key=lambda r: (
        r["class_name"],
        -r["min_conf"],
        -r["iou"],
    )
)

(OUT / "selected_candidates.json").write_text(
    json.dumps(selected, indent=2),
    encoding="utf-8",
)

print("Selected:", len(selected))

from collections import Counter
c = Counter(r["class_name"] for r in selected)

for k, v in sorted(c.items()):
    print(f"{k:18s} {v}")
