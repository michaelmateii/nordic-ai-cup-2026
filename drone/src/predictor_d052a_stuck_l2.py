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
    EXP-D052A — true L1 -> L2 target-inspection policy.

    Policy:
    - If at L0: go to centered L1.
    - If at L1 but not centered: go to centered L1.
    - If at centered L1: choose one small/high-confidence target and zoom to L2.
    - If at L2: reset to L0 (safe full-view reset), then repeat.
    """

    constraints = request.camera_constraints
    current_view = request.view

    current_level = int(current_view.resolution_level)
    current_x = int(current_view.center_x)
    current_y = int(current_view.center_y)

    FULL_W = 3840
    FULL_H = 2160

    CENTER_X = FULL_W // 2
    CENTER_Y = FULL_H // 2

    def bounds_for_level(level: int):
        for b in constraints.center_bounds:
            if int(b.resolution_level) == level:
                return b
        return None

    def legal_request(level: int, center_x: int, center_y: int):
        # Full-view reset is explicitly exempt from delta if enabled
        if (
            level == 0
            and getattr(constraints, "full_view_reset_exempt_from_delta", False)
        ):
            b = bounds_for_level(0)
            if b is None:
                return None
            return RequestedViewDto(
                resolution_level=0,
                center_x=int(max(b.minimum_center_x, min(b.maximum_center_x, center_x))),
                center_y=int(max(b.minimum_center_y, min(b.maximum_center_y, center_y))),
            )

        b = bounds_for_level(level)
        if b is None:
            return None

        tx = int(max(b.minimum_center_x, min(b.maximum_center_x, center_x)))
        ty = int(max(b.minimum_center_y, min(b.maximum_center_y, center_y)))

        max_delta = float(constraints.maximum_center_delta)
        dx = tx - current_x
        dy = ty - current_y
        distance = (dx * dx + dy * dy) ** 0.5

        if distance > max_delta and distance > 0:
            scale = max_delta / distance
            tx = int(round(current_x + dx * scale))
            ty = int(round(current_y + dy * scale))
            tx = int(max(b.minimum_center_x, min(b.maximum_center_x, tx)))
            ty = int(max(b.minimum_center_y, min(b.maximum_center_y, ty)))

        return RequestedViewDto(
            resolution_level=int(level),
            center_x=int(tx),
            center_y=int(ty),
        )

    # 1) If currently L0 -> go to centered L1
    if current_level == 0:
        if 1 in constraints.allowed_resolution_levels:
            return legal_request(1, CENTER_X, CENTER_Y)
        return None

    # 2) If currently L2 -> safe reset to L0
    if current_level == 2:
        if 0 in constraints.allowed_resolution_levels:
            return legal_request(0, CENTER_X, CENTER_Y)
        return None

    # 3) We are at L1
    if current_level != 1:
        return None

    # If not centered L1, re-center first
    if abs(current_x - CENTER_X) > 10 or abs(current_y - CENTER_Y) > 10:
        return legal_request(1, CENTER_X, CENTER_Y)

    # From centered L1, pick a target for L2 zoom
    if 2 not in constraints.allowed_resolution_levels:
        return None

    candidates = []

    for ann in annotations:
        try:
            x1, y1, x2, y2 = [float(v) for v in ann.bbox]
            conf = float(ann.confidence)
        except Exception:
            continue

        if conf < 0.20:
            continue

        gx1 = x1 * FULL_W
        gy1 = y1 * FULL_H
        gx2 = x2 * FULL_W
        gy2 = y2 * FULL_H

        w = max(0.0, gx2 - gx1)
        h = max(0.0, gy2 - gy1)
        area = w * h

        # We want smaller/harder objects, not giant obvious ones
        if area < 40:
            continue
        if area > 12000:
            continue

        cx = (gx1 + gx2) / 2.0
        cy = (gy1 + gy2) / 2.0

        dx = cx - CENTER_X
        dy = cy - CENTER_Y
        dist = (dx * dx + dy * dy) ** 0.5

        # Don't zoom if it's already basically centered
        if dist < 120:
            continue

        # Prefer higher confidence and smaller objects
        priority = conf - 0.00003 * area

        candidates.append(
            (
                priority,
                conf,
                area,
                int(round(cx)),
                int(round(cy)),
                ann.object_id,
            )
        )

    if not candidates:
        return None

    candidates.sort(reverse=True)
    _, conf, area, target_x, target_y, object_id = candidates[0]

    # Optional debug print
    print(
        f"[D052A] L1->L2 target class={object_id} "
        f"conf={conf:.3f} area={area:.1f} "
        f"center=({target_x},{target_y})"
    )

    return legal_request(2, target_x, target_y)

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