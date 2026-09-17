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


from dtos import (
    DroneFlybyPredictRequestDto,
    DroneFlybyPredictResponseDto,
    RequestedViewDto,
)

from utils import decode_view

from capture import capture_request
from model_runtime import detect


logger = logging.getLogger(__name__)


FULL_CENTER_X = 1920
FULL_CENTER_Y = 1080


def choose_next_view(
    request: DroneFlybyPredictRequestDto,
):
    # For D019 keep full L0 coverage.
    if (
        request.view.resolution_level == 0
        and request.view.center_x
        == FULL_CENTER_X
        and request.view.center_y
        == FULL_CENTER_Y
    ):
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
        capture_request(
            request
        )

    except Exception:
        logger.exception(
            "Capture failed on frame %s",
            request.frame,
        )

    try:
        image = decode_view(
            request.view
        )

        annotations = detect(
            image,
            original_width=
                request.original_width,

            original_height=
                request.original_height,
        )

    except Exception:
        logger.exception(
            "Model failed on frame %s",
            request.frame,
        )

        annotations = []

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