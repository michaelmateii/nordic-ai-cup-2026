#!/usr/bin/env python3

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI


ROOT = Path(__file__).resolve().parents[2]

OFFICIAL = (
    ROOT
    / "drone"
    / "reference"
    / "official-drone-flyby"
)

SOURCE_DIR = (
    ROOT
    / "drone"
    / "src"
)

for path in (
    SOURCE_DIR,
    OFFICIAL,
):
    if str(path) not in sys.path:
        sys.path.insert(
            0,
            str(path),
        )


from dtos import (  # noqa: E402
    DroneFlybyPredictRequestDto,
    DroneFlybyPredictResponseDto,
)

from predictor import predict  # noqa: E402
from utils import validate_response  # noqa: E402


HOST = "0.0.0.0"
PORT = 9053


logging.basicConfig(
    level=logging.INFO
)

logger = logging.getLogger(
    __name__
)

app = FastAPI()


@app.post(
    "/predict",
    response_model=
        DroneFlybyPredictResponseDto,
)
def predict_endpoint(
    request:
        DroneFlybyPredictRequestDto,
):
    start = time.perf_counter()

    response = predict(
        request
    )

    validate_response(
        response
    )

    elapsed_ms = (
        time.perf_counter()
        - start
    ) * 1000.0

    logger.info(
        (
            "frame=%s index=%s "
            "level=%s "
            "center=(%s,%s) "
            "detections=%s "
            "total=%.1fms"
        ),
        request.frame,
        request.frame_index,
        request.view.resolution_level,
        request.view.center_x,
        request.view.center_y,
        len(
            response.annotations
        ),
        elapsed_ms,
    )

    return response


@app.get("/")
def index():
    return {
        "service":
            "nordic-ai-cup-drone",

        "status":
            "ok",
    }


if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host=HOST,
        port=PORT,
        reload=False,
    )
