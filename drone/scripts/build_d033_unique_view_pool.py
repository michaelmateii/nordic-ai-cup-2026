#!/usr/bin/env python3

from __future__ import annotations

import json
import shutil
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

CAPTURE_ROOT = ROOT / "drone" / "captures"

OUT = (
    ROOT
    / "drone"
    / "artifacts"
    / "exp_d033"
    / "unique_regions"
)

MANIFEST = (
    ROOT
    / "drone"
    / "artifacts"
    / "exp_d033"
    / "unique_regions_manifest.json"
)


def main():
    if OUT.exists():
        shutil.rmtree(OUT)

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Group all captured frames by actual spatial crop.
    groups = defaultdict(list)

    for seq in sorted(CAPTURE_ROOT.iterdir()):
        if (
            not seq.is_dir()
            or seq.name == "local"
            or not (seq / "frames").exists()
        ):
            continue

        for meta_path in sorted(
            (seq / "frames").glob("*.json")
        ):
            d = json.loads(
                meta_path.read_text()
            )

            view = d["view"]

            if view["resolution_level"] != 1:
                continue

            image_path = meta_path.with_suffix(
                ".png"
            )

            if not image_path.exists():
                continue

            region = tuple(
                int(v)
                for v in view["source_region_xyxy"]
            )

            # Spatial identity only.
            key = (
                int(view["resolution_level"]),
                region,
            )

            groups[key].append(
                {
                    "image_path":
                        image_path,
                    "sequence_id":
                        seq.name,
                    "frame":
                        int(d["frame"]),
                    "frame_index":
                        int(d["frame_index"]),
                    "center_x":
                        int(view["center_x"]),
                    "center_y":
                        int(view["center_y"]),
                    "source_region_xyxy":
                        list(region),
                }
            )

    rows = []

    # Pick one representative from each spatial region.
    for i, (key, candidates) in enumerate(
        sorted(
            groups.items(),
            key=lambda item:
                (
                    item[0][1][1],
                    item[0][1][0],
                    item[0][1][3],
                    item[0][1][2],
                ),
        )
    ):
        # Middle temporal example avoids always taking
        # the first frame of a sequence.
        candidates = sorted(
            candidates,
            key=lambda r:
                (
                    r["frame_index"],
                    r["sequence_id"],
                ),
        )

        chosen = candidates[
            len(candidates) // 2
        ]

        name = (
            f"region_{i:03d}"
            f"_cx_{chosen['center_x']:04d}"
            f"_cy_{chosen['center_y']:04d}"
            f".png"
        )

        dst = OUT / name

        shutil.copy2(
            chosen["image_path"],
            dst,
        )

        row = dict(chosen)

        # Convert Path objects before JSON serialization.
        row["image_path"] = str(
            row["image_path"]
        )

        row["image"] = str(dst)

        row["captures_of_region"] = len(
            candidates
        )

        rows.append(row)

    MANIFEST.write_text(
        json.dumps(
            rows,
            indent=2,
        )
    )

    print("=" * 72)
    print("EXP-D033 UNIQUE SPATIAL REGIONS")
    print("=" * 72)

    print(
        "Unique L1 spatial regions:",
        len(rows),
    )

    print()
    for row in rows:
        print(
            f"center=({row['center_x']:4d},"
            f"{row['center_y']:4d}) "
            f"seen={row['captures_of_region']:3d} "
            f"region={row['source_region_xyxy']}"
        )

    print()
    print("Directory:", OUT)
    print("Manifest:", MANIFEST)


if __name__ == "__main__":
    main()
