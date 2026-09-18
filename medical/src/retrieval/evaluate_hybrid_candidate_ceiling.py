from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


QUESTIONS = Path(
    r"C:\Users\Calle\Projects\Nordic-AI-Cup-2026-official"
    r"\medical-appointment\data\question_train.csv"
)

ASR_DIR = Path(
    r"medical\artifacts\asr\faster_distil_large_v3_int8f32_b1"
)

OUTPUT = Path(
    r"medical\artifacts\retrieval\hybrid_candidate_ceiling.csv"
)

WORD_WINDOW_SIZES = (
    6,
    10,
    14,
    18,
    24,
    32,
)

WORD_STRIDE = 3

MAX_SENTENCES = 3


def normalize_transcript_id(
    value: object,
) -> str:
    text = Path(
        str(value).strip()
    ).stem

    if text.startswith(
        "conversation_sample_"
    ):
        return text

    if text.startswith("sample_"):
        return f"conversation_{text}"

    if text.isdigit():
        return (
            f"conversation_sample_"
            f"{int(text)}"
        )

    return text


def temporal_iou(
    start_a: float,
    end_a: float,
    start_b: float,
    end_b: float,
) -> float:
    intersection = max(
        0.0,
        min(end_a, end_b)
        - max(start_a, start_b),
    )

    union = (
        max(end_a, end_b)
        - min(start_a, start_b)
    )

    if union <= 0:
        return 0.0

    return intersection / union


def load_words(
    transcript_id: object,
) -> list[dict]:
    stem = normalize_transcript_id(
        transcript_id
    )

    path = ASR_DIR / f"{stem}.json"

    data = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    words = []

    for item in data["words"]:
        start = item.get("start")
        end = item.get("end")

        if (
            start is None
            or end is None
        ):
            continue

        words.append(
            {
                "start": float(start),
                "end": float(end),
                "word": str(item["word"]),
            }
        )

    return words


def is_sentence_end(
    text: str,
) -> bool:
    return bool(
        re.search(
            r'[.!?]["\']?$',
            text.strip(),
        )
    )


def make_sentences(
    words: list[dict],
) -> list[dict]:
    sentences = []

    if not words:
        return sentences

    start_index = 0

    for index, word in enumerate(words):
        if not is_sentence_end(
            word["word"]
        ):
            continue

        text = "".join(
            item["word"]
            for item in words[
                start_index:
                index + 1
            ]
        ).strip()

        if text:
            sentences.append(
                {
                    "start": words[
                        start_index
                    ]["start"],
                    "end": words[
                        index
                    ]["end"],
                    "text": text,
                }
            )

        start_index = index + 1

    if start_index < len(words):
        text = "".join(
            item["word"]
            for item in words[
                start_index:
            ]
        ).strip()

        if text:
            sentences.append(
                {
                    "start": words[
                        start_index
                    ]["start"],
                    "end": words[-1]["end"],
                    "text": text,
                }
            )

    return sentences


def sentence_candidates(
    sentences: list[dict],
) -> list[dict]:
    output = []

    for size in range(
        1,
        MAX_SENTENCES + 1,
    ):
        for start in range(
            len(sentences)
            - size
            + 1
        ):
            group = sentences[
                start:
                start + size
            ]

            output.append(
                {
                    "kind": (
                        f"sentence_{size}"
                    ),
                    "start": float(
                        group[0]["start"]
                    ),
                    "end": float(
                        group[-1]["end"]
                    ),
                }
            )

    return output


def word_candidates(
    words: list[dict],
) -> list[dict]:
    output = []

    for size in WORD_WINDOW_SIZES:
        if len(words) < size:
            continue

        starts = list(
            range(
                0,
                len(words) - size + 1,
                WORD_STRIDE,
            )
        )

        final_start = (
            len(words)
            - size
        )

        if final_start not in starts:
            starts.append(
                final_start
            )

        for start_index in starts:
            end_index = (
                start_index
                + size
                - 1
            )

            output.append(
                {
                    "kind": (
                        f"word_{size}"
                    ),
                    "start": float(
                        words[
                            start_index
                        ]["start"]
                    ),
                    "end": float(
                        words[
                            end_index
                        ]["end"]
                    ),
                }
            )

    return output


def deduplicate(
    candidates: list[dict],
) -> list[dict]:
    seen = set()
    output = []

    for candidate in candidates:
        key = (
            round(
                candidate["start"],
                3,
            ),
            round(
                candidate["end"],
                3,
            ),
        )

        if key in seen:
            continue

        seen.add(key)
        output.append(candidate)

    return output


def main() -> None:
    questions = pd.read_csv(
        QUESTIONS
    )

    positives = questions[
        questions["question_type"]
        == "positive"
    ].copy()

    cache = {}

    rows = []

    candidate_counts = []

    for number, (_, row) in enumerate(
        positives.iterrows(),
        start=1,
    ):
        transcript_id = row[
            "transcript_id"
        ]

        stem = normalize_transcript_id(
            transcript_id
        )

        if stem not in cache:
            words = load_words(
                transcript_id
            )

            sentences = make_sentences(
                words
            )

            candidates = (
                sentence_candidates(
                    sentences
                )
                + word_candidates(
                    words
                )
            )

            cache[stem] = deduplicate(
                candidates
            )

        candidates = cache[stem]

        candidate_counts.append(
            len(candidates)
        )

        gold_start = float(
            row["evidence_start"]
        )

        gold_end = float(
            row["evidence_end"]
        )

        best = None
        best_tiou = -1.0

        for candidate in candidates:
            tiou = temporal_iou(
                candidate["start"],
                candidate["end"],
                gold_start,
                gold_end,
            )

            if tiou > best_tiou:
                best_tiou = tiou
                best = candidate

        assert best is not None

        rows.append(
            {
                "question_id": (
                    row["question_id"]
                ),
                "transcript_id": (
                    transcript_id
                ),
                "gold_start": (
                    gold_start
                ),
                "gold_end": (
                    gold_end
                ),
                "oracle_start": (
                    best["start"]
                ),
                "oracle_end": (
                    best["end"]
                ),
                "oracle_kind": (
                    best["kind"]
                ),
                "oracle_tiou": (
                    best_tiou
                ),
                "candidate_count": (
                    len(candidates)
                ),
            }
        )

        if (
            number % 25 == 0
            or number
            == len(positives)
        ):
            print(
                f"[{number:03d}/"
                f"{len(positives):03d}] "
                "processed"
            )

    result = pd.DataFrame(rows)

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT,
        index=False,
    )

    scores = result[
        "oracle_tiou"
    ]

    kinds = (
        result["oracle_kind"]
        .value_counts()
    )

    print()
    print("=" * 78)
    print(
        "Nordic AI Cup 2026 — "
        "M034 Hybrid Candidate Ceiling"
    )
    print("=" * 78)

    print()
    print("ORACLE LOCALIZATION")
    print("-" * 78)

    print(
        f"Mean tIoU:       "
        f"{scores.mean():.4f}"
    )

    print(
        f"Median tIoU:     "
        f"{scores.median():.4f}"
    )

    print(
        f"Minimum:         "
        f"{scores.min():.4f}"
    )

    print(
        f"tIoU >= 0.50:    "
        f"{(scores >= 0.50).mean():.4f}"
    )

    print(
        f"tIoU >= 0.75:    "
        f"{(scores >= 0.75).mean():.4f}"
    )

    print(
        f"tIoU >= 0.90:    "
        f"{(scores >= 0.90).mean():.4f}"
    )

    print()
    print("CANDIDATES")
    print("-" * 78)

    print(
        f"Mean/conversation: "
        f"{np.mean(candidate_counts):.1f}"
    )

    print(
        f"Max/conversation:  "
        f"{np.max(candidate_counts)}"
    )

    print()
    print("ORACLE WINNER TYPE")
    print("-" * 78)

    for kind, count in kinds.items():
        print(
            f"{kind:<16} "
            f"{count:>4}"
        )

    print()
    print("REFERENCE")
    print("-" * 78)

    print(
        "Sentence candidate oracle: 0.7997"
    )

    print(
        "Unrestricted word oracle:  0.9294"
    )

    print()
    print(
        f"Detailed results: {OUTPUT}"
    )


if __name__ == "__main__":
    main()