from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DEFAULT_QUESTION_CSV = Path(
    Path(__file__).resolve().parents[3].parent
    / "Nordic-AI-Cup-2026-official"
    / "medical-appointment"
    / "data"
    / "question_train.csv"
)

DEFAULT_ASR_DIR = Path(
    r"medical\artifacts\asr\faster_distil_large_v3_int8f32_b1"
)

DEFAULT_OUTPUT = Path(
    r"medical\artifacts\retrieval\word_window_tfidf_results.csv"
)

WINDOW_SIZES = (6, 10, 14, 18, 24, 32)
STRIDE = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--questions",
        type=Path,
        default=DEFAULT_QUESTION_CSV,
    )

    parser.add_argument(
        "--asr-dir",
        type=Path,
        default=DEFAULT_ASR_DIR,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    return parser.parse_args()


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


def normalize_transcript_id(value: object) -> str:
    text = Path(str(value).strip()).stem

    if text.startswith("conversation_sample_"):
        return text

    if text.startswith("sample_"):
        return f"conversation_{text}"

    if text.isdigit():
        return f"conversation_sample_{int(text)}"

    return text


def load_words(
    asr_dir: Path,
    transcript_id: object,
) -> list[dict]:
    stem = normalize_transcript_id(transcript_id)
    path = asr_dir / f"{stem}.json"

    if not path.exists():
        raise FileNotFoundError(path)

    data = json.loads(
        path.read_text(encoding="utf-8")
    )

    words: list[dict] = []

    for item in data["words"]:
        start = item.get("start")
        end = item.get("end")

        if start is None or end is None:
            continue

        words.append(
            {
                "start": float(start),
                "end": float(end),
                "word": str(item["word"]),
            }
        )

    if not words:
        raise RuntimeError(
            f"No timestamped words in {path}"
        )

    return words


def make_windows(
    words: list[dict],
) -> list[dict]:
    candidates: list[dict] = []
    n = len(words)

    seen: set[tuple[int, int]] = set()

    for size in WINDOW_SIZES:
        if size > n:
            continue

        starts = list(
            range(
                0,
                n - size + 1,
                STRIDE,
            )
        )

        final_start = n - size

        if final_start not in starts:
            starts.append(final_start)

        for start_index in starts:
            end_index = start_index + size - 1

            key = (
                start_index,
                end_index,
            )

            if key in seen:
                continue

            seen.add(key)

            text = "".join(
                item["word"]
                for item in words[
                    start_index : end_index + 1
                ]
            ).strip()

            candidates.append(
                {
                    "start_index": start_index,
                    "end_index": end_index,
                    "start": words[
                        start_index
                    ]["start"],
                    "end": words[
                        end_index
                    ]["end"],
                    "word_count": size,
                    "text": text,
                }
            )

    return candidates


def score_candidates(
    question: str,
    candidates: list[dict],
) -> np.ndarray:
    texts = [
        candidate["text"]
        for candidate in candidates
    ]

    documents = texts + [question]

    word_vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        sublinear_tf=True,
        token_pattern=r"(?u)\b\w+\b",
    )

    word_matrix = word_vectorizer.fit_transform(
        documents
    )

    word_scores = cosine_similarity(
        word_matrix[-1],
        word_matrix[:-1],
    )[0]

    char_vectorizer = TfidfVectorizer(
        lowercase=True,
        analyzer="char_wb",
        ngram_range=(3, 5),
        sublinear_tf=True,
    )

    char_matrix = char_vectorizer.fit_transform(
        documents
    )

    char_scores = cosine_similarity(
        char_matrix[-1],
        char_matrix[:-1],
    )[0]

    return (
        0.55 * word_scores
        + 0.45 * char_scores
    )


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.questions)

    positives = df[
        df["question_type"] == "positive"
    ].copy()

    word_cache: dict[
        str,
        list[dict],
    ] = {}

    window_cache: dict[
        str,
        list[dict],
    ] = {}

    results: list[dict] = []

    for _, row in positives.iterrows():
        transcript_id = row[
            "transcript_id"
        ]

        stem = normalize_transcript_id(
            transcript_id
        )

        if stem not in word_cache:
            words = load_words(
                args.asr_dir,
                transcript_id,
            )

            word_cache[stem] = words
            window_cache[stem] = make_windows(
                words
            )

        candidates = window_cache[stem]

        question = str(row["question"])

        gold_start = float(
            row["evidence_start"]
        )

        gold_end = float(
            row["evidence_end"]
        )

        scores = score_candidates(
            question,
            candidates,
        )

        ranking = np.argsort(
            scores
        )[::-1]

        candidate_ious = np.array(
            [
                temporal_iou(
                    candidate["start"],
                    candidate["end"],
                    gold_start,
                    gold_end,
                )
                for candidate in candidates
            ],
            dtype=float,
        )

        top_index = int(
            ranking[0]
        )

        top = candidates[
            top_index
        ]

        top_iou = float(
            candidate_ious[
                top_index
            ]
        )

        oracle_index = int(
            np.argmax(candidate_ious)
        )

        oracle_iou = float(
            candidate_ious[
                oracle_index
            ]
        )

        def recall_at(k: int) -> bool:
            selected = ranking[:k]

            return bool(
                np.any(
                    candidate_ious[
                        selected
                    ] > 0
                )
            )

        results.append(
            {
                "question_id": row[
                    "question_id"
                ],
                "transcript_id": (
                    transcript_id
                ),
                "question": question,
                "gold_start": gold_start,
                "gold_end": gold_end,
                "pred_start": top[
                    "start"
                ],
                "pred_end": top[
                    "end"
                ],
                "pred_word_count": (
                    top["word_count"]
                ),
                "pred_text": top[
                    "text"
                ],
                "retrieval_score": (
                    float(
                        scores[
                            top_index
                        ]
                    )
                ),
                "tiou": top_iou,
                "candidate_oracle_tiou": (
                    oracle_iou
                ),
                "top1_overlap": (
                    recall_at(1)
                ),
                "top3_overlap": (
                    recall_at(3)
                ),
                "top5_overlap": (
                    recall_at(5)
                ),
                "top10_overlap": (
                    recall_at(10)
                ),
            }
        )

    result_df = pd.DataFrame(
        results
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_df.to_csv(
        args.output,
        index=False,
        quoting=csv.QUOTE_MINIMAL,
    )

    scores = result_df["tiou"]

    print("=" * 78)
    print(
        "Nordic AI Cup 2026 — "
        "Word-Window Evidence Retrieval"
    )
    print("=" * 78)

    print(
        f"Gold-positive questions: "
        f"{len(result_df)}"
    )

    print()
    print("TOP-1 LOCALIZATION")
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
        f"Any overlap:     "
        f"{(scores > 0).mean():.4f}"
    )

    print(
        f"tIoU >= 0.25:    "
        f"{(scores >= 0.25).mean():.4f}"
    )

    print(
        f"tIoU >= 0.50:    "
        f"{(scores >= 0.50).mean():.4f}"
    )

    print(
        f"tIoU >= 0.75:    "
        f"{(scores >= 0.75).mean():.4f}"
    )

    print()

    print("RETRIEVAL RECALL")
    print("-" * 78)

    print(
        "R@1 any overlap:  "
        f"{result_df['top1_overlap'].mean():.4f}"
    )

    print(
        "R@3 any overlap:  "
        f"{result_df['top3_overlap'].mean():.4f}"
    )

    print(
        "R@5 any overlap:  "
        f"{result_df['top5_overlap'].mean():.4f}"
    )

    print(
        "R@10 any overlap: "
        f"{result_df['top10_overlap'].mean():.4f}"
    )

    print()
    print("CANDIDATE-WINDOW CEILING")
    print("-" * 78)

    oracle = result_df[
        "candidate_oracle_tiou"
    ]

    print(
        "Oracle candidate-window "
        f"mean tIoU:   {oracle.mean():.4f}"
    )

    print(
        "Oracle candidate-window "
        f"median tIoU: {oracle.median():.4f}"
    )

    print()

    failures = result_df[
        scores == 0
    ]

    print(
        "Top-1 zero-overlap failures: "
        f"{len(failures)}"
    )

    print()
    print("PREVIOUS BASELINES")
    print("-" * 78)
    print(
        "Segment TF-IDF top-1 mean tIoU: "
        "0.4016"
    )
    print(
        "Segment TF-IDF R@3 overlap:      "
        "0.8513"
    )
    print(
        "Word timestamp oracle mean:      "
        "0.9294"
    )

    print()
    print(
        f"Detailed results: "
        f"{args.output}"
    )


if __name__ == "__main__":
    main()