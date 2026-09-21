#!/usr/bin/env python3

from __future__ import annotations

import logging
import math
import sys
from pathlib import Path
import time

_SEEN_FRAME_INDICES = set()
_REQUEST_LATENCIES_MS = []

ROOT = Path(__file__).resolve().parents[2]

OFFICIAL = (
    ROOT
    / "drone"
    / "reference"
    / "official-drone-flyby"
)

if str(OFFICIAL) not in sys.path:
    sys.path.insert(
        0,
        str(OFFICIAL),
    )


from dtos import (
    DroneFlybyPredictRequestDto,
    DroneFlybyPredictResponseDto,
    RequestedViewDto,
)

from capture import capture_request
from model_runtime import detect
from tracker_d055 import (
    SequenceTracker,
    merge_current_with_memory,
)
from utils import decode_view


logger = logging.getLogger(__name__)

# EXP-D055A temporal memory.
_TRACKERS = {}


L1_TARGETS = [
    (960, 540),
    (2880, 540),
    (2880, 1620),
    (960, 1620),
]


def clamp(
    value: float,
    low: float,
    high: float,
):
    return max(
        low,
        min(
            high,
            value,
        ),
    )


def get_l1_bounds(
    request: DroneFlybyPredictRequestDto,
):
    for bound in (
        request
        .camera_constraints
        .center_bounds
    ):
        if (
            bound.resolution_level
            == 1
        ):
            return bound

    return None


def choose_next_view(
    request: DroneFlybyPredictRequestDto,
):
    """
    EXP-D031 stable-camera policy.

    Enter L1 at the center once, then issue no further
    camera commands. This avoids stale/out-of-order L1
    movement commands in the remote evaluator.
    """

    allowed = (
        request
        .camera_constraints
        .allowed_resolution_levels
    )

    if 1 not in allowed:
        return None

    target_x = 1920
    target_y = 1080

    # Already at the desired L1 view:
    # leave the camera alone.
    if (
        request.view.resolution_level == 1
        and request.view.center_x == target_x
        and request.view.center_y == target_y
    ):
        return None

    # Only transition into L1 from the full L0 view.
    #
    # Once we are anywhere in L1, do NOT issue another
    # movement command. This intentionally eliminates
    # L1->L1 camera races for D031.
    if request.view.resolution_level != 0:
        return None

    return RequestedViewDto(
        resolution_level=1,
        center_x=target_x,
        center_y=target_y,
    )

def predict(
    request: DroneFlybyPredictRequestDto,
) -> DroneFlybyPredictResponseDto:

    request_start = time.perf_counter()

    # Use frame_index if the DTO exposes it.
    # Otherwise fall back to frame.
    frame_index = getattr(
        request,
        "frame_index",
        request.frame,
    )

    _SEEN_FRAME_INDICES.add(
        int(frame_index)
    )

    # EXP-D031:
    # capture disabled during benchmark to minimize request latency.

    try:
        image = decode_view(
            request.view
        )

        annotations = detect(
            image,
            source_region_xyxy=
                request.view.source_region_xyxy,
            original_width=
                request.original_width,
            original_height=
                request.original_height,
        )

        # -------------------------------------------------
        # EXP-D055A temporal object memory.
        # -------------------------------------------------

        sequence_id = str(
            request.sequence_id
        )

        # A new sequence always starts at frame_index zero.
        if int(frame_index) == 0:
            _TRACKERS.pop(
                sequence_id,
                None,
            )

        tracker = _TRACKERS.setdefault(
            sequence_id,
            SequenceTracker(),
        )

        tracker.update(
            frame=int(request.frame),
            detections=annotations,
        )

        memory_annotations = (
            tracker.memory_predictions()
        )

        # Current detector output always remains.
        # Memory only supplements detections that disappeared.
        annotations = merge_current_with_memory(
            current_annotations=list(annotations),
            memory_annotations=memory_annotations,
        )

    except Exception:
        logger.exception(
            "D030 model failed frame=%s",
            request.frame,
        )

        annotations = []

    request_latency_ms = (
        time.perf_counter()
        - request_start
    ) * 1000.0

    _REQUEST_LATENCIES_MS.append(
        request_latency_ms
    )

    if int(frame_index) >= 248:
        indices = sorted(
            _SEEN_FRAME_INDICES
        )

        missing = sorted(
            set(range(249))
            - _SEEN_FRAME_INDICES
        )

        latencies = sorted(
            _REQUEST_LATENCIES_MS
        )

        median_ms = (
            latencies[
                len(latencies) // 2
            ]
            if latencies
            else 0.0
        )

        max_ms = (
            max(latencies)
            if latencies
            else 0.0
        )

        print(
            "[Remote telemetry] "
            f"received={len(indices)}/249 "
            f"missing={len(missing)} "
            f"median_ms={median_ms:.1f} "
            f"max_ms={max_ms:.1f}"
        )

        print(
            "[Remote telemetry] "
            f"missing_indices={missing}"
        )
        
        _SEEN_FRAME_INDICES.clear()
        _REQUEST_LATENCIES_MS.clear()

    return DroneFlybyPredictResponseDto(
        request_id=
            request.request_id,
        frame=
            request.frame,
        annotations=
            annotations,
        requested_view=
            choose_next_view(
                request
            ),
    )