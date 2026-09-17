#!/usr/bin/env python3

from __future__ import annotations

import logging
import math
import sys
from pathlib import Path


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
):
    allowed = (
        request
        .camera_constraints
        .allowed_resolution_levels
    )

    if 1 not in allowed:
        return None

    bounds = get_l1_bounds(
        request
    )

    if bounds is None:
        return None

    target_index = (
        request.frame_index
        % len(L1_TARGETS)
    )

    target_x, target_y = (
        L1_TARGETS[
            target_index
        ]
    )

    target_x = clamp(
        target_x,
        bounds.minimum_center_x,
        bounds.maximum_center_x,
    )

    target_y = clamp(
        target_y,
        bounds.minimum_center_y,
        bounds.maximum_center_y,
    )

    current_x = (
        request.view.center_x
    )

    current_y = (
        request.view.center_y
    )

    max_delta = max(
        0.0,
        float(
            request
            .camera_constraints
            .maximum_center_delta
        )
        - 1.0
    )

    dx = (
        target_x
        - current_x
    )

    dy = (
        target_y
        - current_y
    )

    distance = math.hypot(
        dx,
        dy,
    )

    if (
        distance > max_delta
        and distance > 0
    ):
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

    target_x = int(
        clamp(
            target_x,
            bounds.minimum_center_x,
            bounds.maximum_center_x,
        )
    )

    target_y = int(
        clamp(
            target_y,
            bounds.minimum_center_y,
            bounds.maximum_center_y,
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

    try:
        capture_request(
            request
        )

    except Exception:
        logger.exception(
            "Capture failed frame=%s",
            request.frame,
        )

    return DroneFlybyPredictResponseDto(
        request_id=
            request.request_id,

        frame=
            request.frame,

        annotations=[],

        requested_view=
            choose_next_view(
                request
            ),
    )
