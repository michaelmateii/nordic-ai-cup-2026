import json
from pathlib import Path
from collections import defaultdict, Counter

ROOT = Path(__file__).resolve().parents[2]

SRC = (
    ROOT
    / "drone"
    / "artifacts"
    / "exp_d047"
    / "disagreements.json"
)

OUT = (
    ROOT
    / "drone"
    / "artifacts"
    / "exp_d048a"
)

OUT.mkdir(parents=True, exist_ok=True)

rows = json.loads(
    SRC.read_text(encoding="utf-8")
)

# We want genuinely strong one-model-only predictions.
MIN_SCORE = 0.03

# Keep the manual workload reasonable.
PER_CLASS_PER_SOURCE = 6

rows = [
    r for r in rows
    if r["score"] >= MIN_SCORE
]

groups = defaultdict(list)

for r in rows:
    groups[
        (
            r["source"],
            r["class_name"],
        )
    ].append(r)

selected = []

for key, items in groups.items():

    items.sort(
        key=lambda r: r["score"],
        reverse=True,
    )

    # Avoid taking many nearly adjacent copies
    # of the same physical object.
    kept = []
    used_indices = []

    for r in items:

        name = r["frame_name"]

        try:
            frame_index = int(
                name.split("_index_")[-1]
            )
        except Exception:
            frame_index = None

        if frame_index is not None:
            too_close = any(
                abs(frame_index - old) <= 2
                for old in used_indices
            )

            if too_close:
                continue

        kept.append(r)

        if frame_index is not None:
            used_indices.append(
                frame_index
            )

        if len(kept) >= PER_CLASS_PER_SOURCE:
            break

    selected.extend(kept)

selected.sort(
    key=lambda r: (
        r["class_name"],
        r["source"],
        -r["score"],
    )
)

for i, r in enumerate(selected):
    r["selection_id"] = f"D48_{i:03d}"

(OUT / "selected_disagreements.json").write_text(
    json.dumps(selected, indent=2),
    encoding="utf-8",
)

print("Selected:", len(selected))

print("\nBy class:")
c = Counter(r["class_name"] for r in selected)

for k, v in sorted(c.items()):
    print(f"{k:18s} {v}")

print("\nBy source:")
s = Counter(r["source"] for r in selected)

for k, v in sorted(s.items()):
    print(f"{k:18s} {v}")

