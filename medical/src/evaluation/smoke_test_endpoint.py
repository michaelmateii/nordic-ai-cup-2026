from __future__ import annotations

import base64
import json
import time
import urllib.request
from pathlib import Path

import pandas as pd


OFFICIAL_ROOT = (
    Path(__file__).resolve().parents[3].parent
    / "Nordic-AI-Cup-2026-official"
    / "medical-appointment"
)

QUESTION_CSV = (
    OFFICIAL_ROOT
    / "data"
    / "question_train.csv"
)

AUDIO_PATH = (
    OFFICIAL_ROOT
    / "data"
    / "audio"
    / "conversation_sample_10.mp3"
)

ENDPOINT = (
    "http://127.0.0.1:8000/predict"
)

SAMPLE_ID = "sample_10"


def normalize_id(
    value: object,
) -> str:
    text = Path(
        str(value).strip()
    ).stem

    if text.startswith(
        "conversation_"
    ):
        text = text[
            len("conversation_"):
        ]

    return text


def main() -> None:
    df = pd.read_csv(
        QUESTION_CSV
    )

    mask = (
        df["transcript_id"]
        .map(normalize_id)
        == SAMPLE_ID
    )

    rows = df[
        mask
    ].copy()

    if len(rows) != 10:
        raise RuntimeError(
            "Expected exactly 10 "
            f"questions for {SAMPLE_ID}; "
            f"got {len(rows)}"
        )

    audio_bytes = (
        AUDIO_PATH.read_bytes()
    )

    payload = {
        "audio_base64": (
            base64.b64encode(
                audio_bytes
            ).decode("ascii")
        ),
        "audio_filename": (
            AUDIO_PATH.name
        ),
        "questions": (
            rows["question"]
            .astype(str)
            .tolist()
        ),
    }

    request = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(
            payload
        ).encode("utf-8"),
        headers={
            "Content-Type": (
                "application/json"
            ),
        },
        method="POST",
    )

    print("=" * 78)
    print(
        "Nordic AI Cup 2026 — "
        "Medical endpoint smoke test"
    )
    print("=" * 78)

    print(
        f"Audio:     {AUDIO_PATH.name}"
    )

    print(
        f"Questions: {len(rows)}"
    )

    start = time.perf_counter()

    with urllib.request.urlopen(
        request,
        timeout=60,
    ) as response:
        raw = response.read()

        status = response.status

    elapsed = (
        time.perf_counter()
        - start
    )

    result = json.loads(
        raw.decode("utf-8")
    )

    print(
        f"HTTP:      {status}"
    )

    print(
        f"Latency:   {elapsed:.2f} s"
    )

    required = {
        "answers",
        "evidence_start",
        "evidence_end",
    }

    if set(result) != required:
        raise RuntimeError(
            "Unexpected response keys: "
            f"{result.keys()}"
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
        == 10
    ):
        raise RuntimeError(
            "Invalid response lengths: "
            f"{len(answers)}, "
            f"{len(starts)}, "
            f"{len(ends)}"
        )

    for index, answer in enumerate(
        answers
    ):
        if not isinstance(
            answer,
            bool,
        ):
            raise RuntimeError(
                f"Answer {index} is "
                "not bool."
            )

        if not answer:
            if (
                starts[index]
                is not None
                or ends[index]
                is not None
            ):
                raise RuntimeError(
                    "FALSE answer contains "
                    f"evidence at {index}."
                )

        else:
            if (
                starts[index]
                is None
                or ends[index]
                is None
            ):
                raise RuntimeError(
                    "TRUE answer is missing "
                    f"evidence at {index}."
                )

            if not (
                0
                <= float(starts[index])
                < float(ends[index])
            ):
                raise RuntimeError(
                    "Invalid evidence span "
                    f"at {index}: "
                    f"{starts[index]} - "
                    f"{ends[index]}"
                )

    print()
    print(
        "STRUCTURE: PASS"
    )

    print()
    print(
        f"Predicted YES: "
        f"{sum(answers)}/10"
    )

    print()
    print("PREDICTIONS")
    print("-" * 78)

    gold_answers = (
        rows["answer"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(
            ["yes", "true", "1"]
        )
        .tolist()
    )

    correct = 0

    for index, (
        question,
        gold,
        predicted,
        start_time,
        end_time,
    ) in enumerate(
        zip(
            rows["question"],
            gold_answers,
            answers,
            starts,
            ends,
        ),
        start=1,
    ):
        if gold == predicted:
            correct += 1

        print()
        print(
            f"{index:02d}. "
            f"{question}"
        )

        print(
            f"    gold={gold} "
            f"pred={predicted}"
        )

        print(
            "    evidence="
            f"{start_time} - "
            f"{end_time}"
        )

    print()
    print("-" * 78)

    print(
        f"Smoke-test accuracy: "
        f"{correct}/10 "
        f"({correct / 10:.3f})"
    )

    print(
        f"Request latency: "
        f"{elapsed:.2f} s"
    )

    print()
    print("RESPONSE JSON")
    print("-" * 78)

    print(
        json.dumps(
            result,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
