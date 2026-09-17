#!/usr/bin/env python3

from __future__ import annotations

import base64
import json
import threading
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]

CAPTURE_ROOT = (
    ROOT
    / "drone"
    / "captures"
)

_lock = threading.Lock()


def _safe_name(value: str) -> str:
    return "".join(
        ch
        if ch.isalnum() or ch in ("-", "_", ".")
        else "_"
        for ch in value
    )


def capture_request(request: Any) -> Path:
    """
    Persist exactly what the evaluation server sent us.

    Layout:

    drone/captures/<sequence_id>/
        sequence_metadata.json
        frames/
            frame_000000_index_000000.png
            frame_000000_index_000000.json
            ...
    """

    sequence_id = _safe_name(
        request.sequence_id
    )

    sequence_dir = (
        CAPTURE_ROOT
        / sequence_id
    )

    frames_dir = (
        sequence_dir
        / "frames"
    )

    frames_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    stem = (
        f"frame_{request.frame:06d}"
        f"_index_{request.frame_index:06d}"
    )

    image_path = (
        frames_dir
        / f"{stem}.png"
    )

    metadata_path = (
        frames_dir
        / f"{stem}.json"
    )

    image_bytes = base64.b64decode(
        request.view.image
    )

    metadata = {
        "sequence_id":
            request.sequence_id,

        "frame":
            int(request.frame),

        "frame_index":
            int(request.frame_index),

        "request_id":
            request.request_id,

        "frame_interval_ms":
            int(
                request.frame_interval_ms
            ),

        "response_timeout_ms":
            int(
                request.response_timeout_ms
            ),

        "original_width":
            int(
                request.original_width
            ),

        "original_height":
            int(
                request.original_height
            ),

        "view": {
            "resolution_level":
                int(
                    request.view.resolution_level
                ),

            "center_x":
                int(
                    request.view.center_x
                ),

            "center_y":
                int(
                    request.view.center_y
                ),

            "view_id":
                request.view.view_id,

            "image_media_type":
                request.view.image_media_type,

            "width":
                int(
                    request.view.width
                ),

            "height":
                int(
                    request.view.height
                ),

            "source_region_xyxy":
                [
                    int(value)
                    for value
                    in request.view.source_region_xyxy
                ],
        },

        "camera_constraints": {
            "maximum_center_delta":
                float(
                    request.camera_constraints
                    .maximum_center_delta
                ),

            "allowed_resolution_levels":
                [
                    int(value)
                    for value
                    in request.camera_constraints
                    .allowed_resolution_levels
                ],

            "full_view_reset_exempt_from_delta":
                bool(
                    request.camera_constraints
                    .full_view_reset_exempt_from_delta
                ),

            "center_bounds":
                [
                    {
                        "resolution_level":
                            int(
                                bound.resolution_level
                            ),

                        "width":
                            int(
                                bound.width
                            ),

                        "height":
                            int(
                                bound.height
                            ),

                        "minimum_center_x":
                            int(
                                bound.minimum_center_x
                            ),

                        "maximum_center_x":
                            int(
                                bound.maximum_center_x
                            ),

                        "minimum_center_y":
                            int(
                                bound.minimum_center_y
                            ),

                        "maximum_center_y":
                            int(
                                bound.maximum_center_y
                            ),
                    }

                    for bound
                    in request.camera_constraints
                    .center_bounds
                ],
        },

        "camera_command_feedback":
            (
                request.camera_command_feedback
                .model_dump()
                if request.camera_command_feedback
                is not None
                else None
            ),
    }

    with _lock:
        image_path.write_bytes(
            image_bytes
        )

        metadata_path.write_text(
            json.dumps(
                metadata,
                indent=2,
            ),
            encoding="utf-8",
        )

        sequence_metadata_path = (
            sequence_dir
            / "sequence_metadata.json"
        )

        if not sequence_metadata_path.exists():
            sequence_metadata_path.write_text(
                json.dumps(
                    {
                        "sequence_id":
                            request.sequence_id,

                        "original_width":
                            int(
                                request.original_width
                            ),

                        "original_height":
                            int(
                                request.original_height
                            ),

                        "frame_interval_ms":
                            int(
                                request.frame_interval_ms
                            ),

                        "response_timeout_ms":
                            int(
                                request.response_timeout_ms
                            ),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

    return image_path
