import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SRC = ROOT / "drone" / "src"
REF = ROOT / "drone" / "reference" / "official-drone-flyby"

sys.path.insert(0, str(SRC))
sys.path.insert(0, str(REF))

from dtos import DroneFlybyPredictionDto
from tracker_d055 import SequenceTracker, merge_current_with_memory

SEQ = (
    ROOT
    / "drone"
    / "captures"
    / "ed590596961d4390b6919a7633824f6b"
)

PREDS = SEQ / "predictions"

tracker = SequenceTracker()

total_current = 0
total_memory = 0
total_final = 0

frames_with_memory = 0

max_memory = 0
max_final = 0

print(
    "frame  current  memory  final  classes_from_memory"
)

for pred_path in sorted(PREDS.glob("*.json")):

    data = json.loads(
        pred_path.read_text(
            encoding="utf-8"
        )
    )

    frame = int(
        data.get(
            "frame",
            data.get(
                "frame_index",
                0,
            ),
        )
    )

    current = []

    for ann in data["annotations"]:

        # IMPORTANT:
        # simulate the current D055 detector threshold.
        conf = float(
            ann["confidence"]
        )

        if conf < 0.03:
            continue

        current.append(
            DroneFlybyPredictionDto(
                object_id=
                    ann["object_id"],

                bbox=[
                    float(v)
                    for v in ann["bbox"]
                ],

                confidence=conf,
            )
        )

    tracker.update(
        frame=frame,
        detections=current,
    )

    memory = (
        tracker.memory_predictions()
    )

    final = merge_current_with_memory(
        current_annotations=list(current),
        memory_annotations=list(memory),
    )

    total_current += len(current)
    total_memory += len(memory)
    total_final += len(final)

    max_memory = max(
        max_memory,
        len(memory),
    )

    max_final = max(
        max_final,
        len(final),
    )

    if memory:
        frames_with_memory += 1

        classes = ",".join(
            sorted(
                {
                    x.object_id
                    for x in memory
                }
            )
        )

        print(
            f"{frame:5d}  "
            f"{len(current):7d}  "
            f"{len(memory):6d}  "
            f"{len(final):5d}  "
            f"{classes}"
        )


count = len(
    list(PREDS.glob("*.json"))
)

print()
print("Frames:", count)

print(
    "Current detections:",
    total_current,
)

print(
    "Memory detections:",
    total_memory,
)

print(
    "Final detections:",
    total_final,
)

print(
    "Frames with memory:",
    frames_with_memory,
)

print(
    "Average current/frame:",
    round(
        total_current
        / max(1, count),
        3,
    ),
)

print(
    "Average memory/frame:",
    round(
        total_memory
        / max(1, count),
        3,
    ),
)

print(
    "Average final/frame:",
    round(
        total_final
        / max(1, count),
        3,
    ),
)

print(
    "Max memory in one frame:",
    max_memory,
)

print(
    "Max final in one frame:",
    max_final,
)
