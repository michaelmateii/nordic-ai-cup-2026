from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field


REPO_ROOT = Path(__file__).resolve().parents[1]

RUNTIME_DIR = (
    REPO_ROOT
    / "medical"
    / "src"
    / "runtime"
)

if str(RUNTIME_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(RUNTIME_DIR),
    )

from predictor import MedicalPredictor


class PredictRequest(BaseModel):
    audio_base64: str
    audio_filename: str
    questions: list[str]


class PredictResponse(BaseModel):
    answers: list[bool]
    evidence_start: list[float | None]
    evidence_end: list[float | None]


app = FastAPI(
    title="Nordic AI Cup 2026 - Medical Appointment",
    version="0.1.0",
)


print("[medical-api] initializing predictor...")

predictor = MedicalPredictor()

print("[medical-api] predictor ready")


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
    }


@app.post(
    "/predict",
    response_model=PredictResponse,
)
def predict(
    request: PredictRequest,
) -> PredictResponse:
    question_count = len(
        request.questions
    )

    result = predictor.predict_request(
        audio_base64=(
            request.audio_base64
        ),
        audio_filename=(
            request.audio_filename
        ),
        questions=request.questions,
    )

    answers = result[
        "answers"
    ]

    starts = result[
        "evidence_start"
    ]

    ends = result[
        "evidence_end"
    ]

    if not (
        len(answers)
        == len(starts)
        == len(ends)
        == question_count
    ):
        print(
            "[medical-api] invalid internal "
            "response lengths; returning "
            "safe fallback",
            file=sys.stderr,
        )

        return PredictResponse(
            answers=[
                False
                for _ in range(
                    question_count
                )
            ],
            evidence_start=[
                None
                for _ in range(
                    question_count
                )
            ],
            evidence_end=[
                None
                for _ in range(
                    question_count
                )
            ],
        )

    # Enforce the competition contract:
    # FALSE -> null evidence.
    for index, answer in enumerate(
        answers
    ):
        if not answer:
            starts[index] = None
            ends[index] = None

    return PredictResponse(
        answers=answers,
        evidence_start=starts,
        evidence_end=ends,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )