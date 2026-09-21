from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


QUESTIONS = Path(
    Path(__file__).resolve().parents[3].parent
    / "Nordic-AI-Cup-2026-official"
    / "medical-appointment"
    / "data"
    / "question_train.csv"
)

ASR_DIR = Path(
    r"medical\artifacts\asr\faster_distil_large_v3_int8f32_b1"
)

OUTPUT = Path(
    r"medical\artifacts\retrieval\sentence_candidate_ceiling.csv"
)


def normalize_transcript_id(value: object) -> str:
    text = Path(str(value).strip()).stem

    if text.startswith("conversation_sample_"):
        return text

    if text.startswith("sample_"):
        return f"conversation_{text}"

    if text.isdigit():
        return f"conversation_sample_{int(text)}"

    return text


def temporal_iou(
    pred_start: float,
    pred_end: float,
    gold_start: float,
    gold_end: float,
) -> float:
    intersection = max(
        0.0,
        min(pred_end, gold_end)
        - max(pred_start, gold_start),
    )

    union = (
        max(pred_end, gold_end)
        - min(pred_start, gold_start)
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
        if (
            item.get("start") is None
            or item.get("end") is None
        ):
            continue

        words.append(
            {
                "start": float(
                    item["start"]
                ),
                "end": float(
                    item["end"]
                ),
                "word": str(
                    item["word"]
                ),
            }
        )

    return words


def is_sentence_end(
    text: str,
) -> bool:
    stripped = text.strip()

    return bool(
        re.search(
            r'[.!?]["\']?$',
            stripped,
        )
    )


def make_sentences(
    words: list[dict],
) -> list[dict]:
    sentences = []

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
                    "start_index": (
                        start_index
                    ),
                    "end_index": index,
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
                    "start_index": (
                        start_index
                    ),
                    "end_index": (
                        len(words) - 1
                    ),
                    "start": words[
                        start_index
                    ]["start"],
                    "end": words[-1][
                        "end"
                    ],
                    "text": text,
                }
            )

    return sentences


def make_sentence_groups(
    sentences: list[dict],
    max_group_size: int,
) -> list[dict]:
    groups = []

    for size in range(
        1,
        max_group_size + 1,
    ):
        for start in range(
            0,
            len(sentences) - size + 1,
        ):
            selected = sentences[
                start:
                start + size
            ]

            groups.append(
                {
                    "sentence_count": size,
                    "start": selected[
                        0
                    ]["start"],
                    "end": selected[
                        -1
                    ]["end"],
                    "text": " ".join(
                        item["text"]
                        for item in selected
                    ),
                }
            )

    return groups


def main() -> None:
    df = pd.read_csv(
        QUESTIONS
    )

    positives = df[
        df["question_type"]
        == "positive"
    ].copy()

    cache = {}
    rows = []

    for _, row in positives.iterrows():
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

            cache[stem] = {
                1: make_sentence_groups(
                    sentences,
                    1,
                ),
                2: make_sentence_groups(
                    sentences,
                    2,
                ),
                3: make_sentence_groups(
                    sentences,
                    3,
                ),
            }

        gold_start = float(
            row["evidence_start"]
        )

        gold_end = float(
            row["evidence_end"]
        )

        result = {
            "question_id": row[
                "question_id"
            ],
            "transcript_id": (
                transcript_id
            ),
            "gold_start": gold_start,
            "gold_end": gold_end,
        }

        for max_group in (
            1,
            2,
            3,
        ):
            candidates = cache[
                stem
            ][max_group]

            ious = [
                temporal_iou(
                    candidate["start"],
                    candidate["end"],
                    gold_start,
                    gold_end,
                )
                for candidate
                in candidates
            ]

            best_index = int(
                np.argmax(ious)
            )

            best = candidates[
                best_index
            ]

            result[
                f"oracle_tiou_s{max_group}"
            ] = float(
                ious[best_index]
            )

            result[
                f"oracle_text_s{max_group}"
            ] = best["text"]

            result[
                f"oracle_start_s{max_group}"
            ] = best["start"]

            result[
                f"oracle_end_s{max_group}"
            ] = best["end"]

        rows.append(result)

    result_df = pd.DataFrame(
        rows
    )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_df.to_csv(
        OUTPUT,
        index=False,
    )

    print("=" * 78)
    print(
        "Nordic AI Cup 2026 — "
        "Sentence Candidate Ceiling"
    )
    print("=" * 78)

    for group_size in (
        1,
        2,
        3,
    ):
        scores = result_df[
            f"oracle_tiou_s{group_size}"
        ]

        print()
        print(
            f"UP TO {group_size} SENTENCE(S)"
        )
        print("-" * 78)

        print(
            f"Mean oracle tIoU:   "
            f"{scores.mean():.4f}"
        )

        print(
            f"Median oracle tIoU: "
            f"{scores.median():.4f}"
        )

        print(
            f"Minimum:            "
            f"{scores.min():.4f}"
        )

        print(
            f"tIoU >= 0.50:       "
            f"{(scores >= 0.50).mean():.4f}"
        )

        print(
            f"tIoU >= 0.75:       "
            f"{(scores >= 0.75).mean():.4f}"
        )

        print(
            f"tIoU >= 0.90:       "
            f"{(scores >= 0.90).mean():.4f}"
        )

    print()
    print("REFERENCE")
    print("-" * 78)

    print(
        "M007 unrestricted word oracle: 0.9294"
    )

    print(
        "M010 segment mean:             0.4347"
    )

    print()
    print(
        f"Detailed results: {OUTPUT}"
    )


if __name__ == "__main__":
    main()