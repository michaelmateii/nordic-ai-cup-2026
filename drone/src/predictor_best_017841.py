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
from utils import decode_view


logger = logging.getLogger(__name__)


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
    annotations,
):
    """
    EXP-D051A active L1 target-following.

    - Enter centered L1 from L0.
    - From centered L1, target one small/high-confidence detection.
    - From an off-center L1 view, return to centered L1.
    """

    constraints = request.camera_constraints

    allowed = constraints.allowed_resolution_levels

    if 1 not in allowed:
        return None

    bounds = constraints.bounds_for_level(1)

    if bounds is None:
        return None

    CENTER_X = 1920
    CENTER_Y = 1080

    current_level = int(
        request.view.resolution_level
    )

    current_x = int(
        request.view.center_x
    )

    current_y = int(
        request.view.center_y
    )

    # From L0, establish the known-good centered L1 view.
    if current_level == 0:
        return RequestedViewDto(
            resolution_level=1,
            center_x=CENTER_X,
            center_y=CENTER_Y,
        )

    if current_level != 1:
        return None

    # If we are currently off-center, return to discovery view.
    if (
        current_x != CENTER_X
        or current_y != CENTER_Y
    ):
        return RequestedViewDto(
            resolution_level=1,
            center_x=CENTER_X,
            center_y=CENTER_Y,
        )

    # We are at centered L1.
    # Find a small, credible target that can benefit from a closer view.
    candidates = []

    for ann in annotations:

        conf = float(
            ann.confidence
        )

        if conf < 0.05:
            continue

        x1, y1, x2, y2 = [
            float(v)
            for v in ann.bbox
        ]

        bw = (
            x2 - x1
        ) * request.original_width

        bh = (
            y2 - y1
        ) * request.original_height

        area = bw * bh

        # Skip huge/easy objects.
        if area > 12000:
            continue

        global_x = (
            (x1 + x2) / 2.0
            * request.original_width
        )

        global_y = (
            (y1 + y2) / 2.0
            * request.original_height
        )

        # Prefer high confidence and small targets.
        priority = (
            conf
            / max(
                1.0,
                area ** 0.5,
            )
        )

        candidates.append(
            (
                priority,
                conf,
                area,
                global_x,
                global_y,
                ann.object_id,
            )
        )

    if not candidates:
        return None

    candidates.sort(
        reverse=True
    )

    (
        _priority,
        _conf,
        _area,
        target_x,
        target_y,
        _object_id,
    ) = candidates[0]

    target_x = int(
        round(
            clamp(
                target_x,
                bounds.minimum_center_x,
                bounds.maximum_center_x,
            )
        )
    )

    target_y = int(
        round(
            clamp(
                target_y,
                bounds.minimum_center_y,
                bounds.maximum_center_y,
            )
        )
    )

    # Avoid pointless tiny movements.
    dx = target_x - current_x
    dy = target_y - current_y

    distance = math.sqrt(
        dx * dx
        + dy * dy
    )

    if distance < 150:
        return None

    max_delta = float(
        constraints.maximum_center_delta
    )

    if distance > max_delta:
        scale = (
            max_delta
            / distance
        )

        target_x = int(
            round(
                current_x
                + dx * scale
            )
        )

        target_y = int(
            round(
                current_y
                + dy * scale
            )
        )

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
                request,
                annotations,
            ),
    )