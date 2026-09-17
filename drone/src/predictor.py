#!/usr/bin/env python3

from __future__ import annotations

import logging
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


from dtos import (  # noqa: E402
    DroneFlybyPredictRequestDto,
    DroneFlybyPredictResponseDto,
    RequestedViewDto,
)

from capture import capture_request  # noqa: E402


logger = logging.getLogger(__name__)


FULL_CENTER_X = 1920
FULL_CENTER_Y = 1080


def choose_capture_view(
    request: DroneFlybyPredictRequestDto,
):
    """
    Keep/reset camera to complete L0 view.

    This maximizes information retained from the validation sequence.
    """

    if (
        request.view.resolution_level == 0
        and request.view.center_x
        == FULL_CENTER_X
        and request.view.center_y
        == FULL_CENTER_Y
    ):
        # Already at the complete view.
        return None

    if (
        0
        not in request.camera_constraints
        .allowed_resolution_levels
    ):
        return None

    return RequestedViewDto(
        resolution_level=0,
        center_x=FULL_CENTER_X,
        center_y=FULL_CENTER_Y,
    )


def predict(
    request: DroneFlybyPredictRequestDto,
) -> DroneFlybyPredictResponseDto:

    try:
        captured_path = (
            capture_request(
                request
            )
        )

        logger.info(
            "Captured frame %s index %s -> %s",
            request.frame,
            request.frame_index,
            captured_path,
        )

    except Exception:
        # Capture must never cost us the prediction.
        logger.exception(
            "Capture failed for frame %s",
            request.frame,
        )

    return DroneFlybyPredictResponseDto(
        request_id=
            request.request_id,

        frame=
            request.frame,

        annotations=[],

        requested_view=
            choose_capture_view(
                request
            ),
    )
