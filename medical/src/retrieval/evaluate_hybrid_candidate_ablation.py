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
    r"medical\artifacts\retrieval\hybrid_candidate_ablation.csv"
)

WORD_STRIDE = 3
MAX_SENTENCES = 3


CONFIGS = {
    "sentences_only": (),
    "sent_plus_w6": (
        6,
    ),
    "sent_plus_w6_w10": (
        6,
        10,
    ),
    "sent_plus_w6_w10_w14": (
        6,
        10,
        14,
    ),
    "sent_plus_w6_w10_w14_w18": (
        6,
        10,
        14,
        18,
    ),
    "all_word_windows": (
        6,
        10,
        14,
        18,
        24,
        32,
    ),
}


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

    if text.startswith(
        "sample_"
    ):
        return (
            f"conversation_{text}"
        )

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

    path = (
        ASR_DIR
        / f"{stem}.json"
    )

    data = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    output = []

    for word in data["words"]:
        start = word.get(
            "start"
        )

        end = word.get(
            "end"
        )

        if (
            start is None
            or end is None
        ):
            continue

        output.append(
            {
                "start": float(
                    start
                ),
                "end": float(
                    end
                ),
                "word": str(
                    word["word"]
                ),
            }
        )

    return output


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
    if not words:
        return []

    output = []
    start_index = 0

    for index, word in enumerate(
        words
    ):
        if not is_sentence_end(
            word["word"]
        ):
            continue

        output.append(
            {
                "start": float(
                    words[
                        start_index
                    ]["start"]
                ),
                "end": float(
                    words[index]["end"]
                ),
            }
        )

        start_index = (
            index + 1
        )

    if start_index < len(words):
        output.append(
            {
                "start": float(
                    words[
                        start_index
                    ]["start"]
                ),
                "end": float(
                    words[-1]["end"]
                ),
            }
        )

    return output


def make_sentence_candidates(
    sentences: list[dict],
) -> list[dict]:
    output = []

    for size in range(
        1,
        MAX_SENTENCES + 1,
    ):
        for start_index in range(
            len(sentences)
            - size
            + 1
        ):
            group = sentences[
                start_index:
                start_index + size
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


def make_word_candidates(
    words: list[dict],
    sizes: tuple[int, ...],
) -> list[dict]:
    output = []

    for size in sizes:
        if len(words) < size:
            continue

        start_indices = list(
            range(
                0,
                len(words)
                - size
                + 1,
                WORD_STRIDE,
            )
        )

        final_start = (
            len(words)
            - size
        )

        if final_start not in (
            start_indices
        ):
            start_indices.append(
                final_start
            )

        for start_index in (
            start_indices
        ):
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

        output.append(
            candidate
        )

    return output


def main() -> None:
    questions = pd.read_csv(
        QUESTIONS
    )

    positives = questions[
        questions["question_type"]
        == "positive"
    ].copy()

    transcript_cache = {}

    rows = []

    for config_name, word_sizes in (
        CONFIGS.items()
    ):
        oracle_scores = []
        candidate_counts = []
        winner_types = []

        for _, row in (
            positives.iterrows()
        ):
            transcript_id = (
                row[
                    "transcript_id"
                ]
            )

            stem = (
                normalize_transcript_id(
                    transcript_id
                )
            )

            if stem not in (
                transcript_cache
            ):
                words = load_words(
                    transcript_id
                )

                sentences = (
                    make_sentences(
                        words
                    )
                )

                transcript_cache[
                    stem
                ] = {
                    "words": words,
                    "sentences": sentences,
                }

            cached = (
                transcript_cache[
                    stem
                ]
            )

            candidates = (
                make_sentence_candidates(
                    cached[
                        "sentences"
                    ]
                )
                + make_word_candidates(
                    cached["words"],
                    word_sizes,
                )
            )

            candidates = (
                deduplicate(
                    candidates
                )
            )

            candidate_counts.append(
                len(candidates)
            )

            gold_start = float(
                row[
                    "evidence_start"
                ]
            )

            gold_end = float(
                row[
                    "evidence_end"
                ]
            )

            best_tiou = -1.0
            best_kind = None

            for candidate in (
                candidates
            ):
                tiou = temporal_iou(
                    candidate[
                        "start"
                    ],
                    candidate[
                        "end"
                    ],
                    gold_start,
                    gold_end,
                )

                if (
                    tiou
                    > best_tiou
                ):
                    best_tiou = tiou
                    best_kind = (
                        candidate[
                            "kind"
                        ]
                    )

            oracle_scores.append(
                best_tiou
            )

            winner_types.append(
                best_kind
            )

        scores = np.asarray(
            oracle_scores,
            dtype=float,
        )

        rows.append(
            {
                "config": (
                    config_name
                ),
                "word_sizes": (
                    ",".join(
                        str(value)
                        for value
                        in word_sizes
                    )
                ),
                "mean_oracle_tiou": (
                    scores.mean()
                ),
                "median_oracle_tiou": (
                    np.median(
                        scores
                    )
                ),
                "tiou_ge_050": (
                    (
                        scores >= 0.50
                    ).mean()
                ),
                "tiou_ge_075": (
                    (
                        scores >= 0.75
                    ).mean()
                ),
                "tiou_ge_090": (
                    (
                        scores >= 0.90
                    ).mean()
                ),
                "mean_candidates": (
                    np.mean(
                        candidate_counts
                    )
                ),
                "max_candidates": (
                    np.max(
                        candidate_counts
                    )
                ),
            }
        )

    result = pd.DataFrame(
        rows
    )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT,
        index=False,
    )

    print("=" * 100)
    print(
        "Nordic AI Cup 2026 — "
        "M035 Hybrid Candidate Ablation"
    )
    print("=" * 100)

    print()

    for _, row in (
        result.iterrows()
    ):
        print(
            f"{row['config']}"
        )

        print(
            "-" * 78
        )

        print(
            f"Word sizes:        "
            f"{row['word_sizes']}"
        )

        print(
            f"Mean oracle tIoU:  "
            f"{row['mean_oracle_tiou']:.4f}"
        )

        print(
            f"Median:            "
            f"{row['median_oracle_tiou']:.4f}"
        )

        print(
            f"tIoU >= 0.50:      "
            f"{row['tiou_ge_050']:.4f}"
        )

        print(
            f"tIoU >= 0.75:      "
            f"{row['tiou_ge_075']:.4f}"
        )

        print(
            f"tIoU >= 0.90:      "
            f"{row['tiou_ge_090']:.4f}"
        )

        print(
            f"Mean candidates:   "
            f"{row['mean_candidates']:.1f}"
        )

        print(
            f"Max candidates:    "
            f"{int(row['max_candidates'])}"
        )

        print()

    print("REFERENCE")
    print("-" * 78)

    print(
        "Full M034 oracle:       "
        "0.8541"
    )

    print(
        "M030 sentence OOF:      "
        "0.5089"
    )

    print()

    print(
        f"Detailed results: {OUTPUT}"
    )


if __name__ == "__main__":
    main()