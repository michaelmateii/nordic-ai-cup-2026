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
    r"C:\Users\Calle\Projects\Nordic-AI-Cup-2026-official"
    r"\medical-appointment\data\question_train.csv"
)

DEFAULT_ASR_DIR = Path(
    r"medical\artifacts\asr\faster_distil_large_v3_int8f32_b1"
)

DEFAULT_OUTPUT = Path(
    r"medical\artifacts\retrieval\coarse_to_fine_results.csv"
)

TOP_SEGMENTS = 3

WINDOW_SIZES = (
    4,
    6,
    8,
    10,
    12,
    16,
    20,
)

LOCAL_WORD_MARGIN = 10


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
            f"conversation_sample_{int(text)}"
        )

    return text


def load_asr(
    asr_dir: Path,
    transcript_id: object,
) -> dict:
    stem = normalize_transcript_id(
        transcript_id
    )

    path = asr_dir / f"{stem}.json"

    if not path.exists():
        raise FileNotFoundError(path)

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def get_segments(
    data: dict,
) -> list[dict]:
    result = []

    for segment in data["segments"]:
        text = str(
            segment["text"]
        ).strip()

        if not text:
            continue

        result.append(
            {
                "start": float(
                    segment["start"]
                ),
                "end": float(
                    segment["end"]
                ),
                "text": text,
            }
        )

    return result


def get_words(
    data: dict,
) -> list[dict]:
    result = []

    for item in data["words"]:
        start = item.get("start")
        end = item.get("end")

        if start is None or end is None:
            continue

        result.append(
            {
                "start": float(start),
                "end": float(end),
                "word": str(
                    item["word"]
                ),
            }
        )

    return result


def combined_tfidf_scores(
    query: str,
    texts: list[str],
) -> np.ndarray:
    documents = texts + [query]

    word_vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        sublinear_tf=True,
        token_pattern=r"(?u)\b\w+\b",
    )

    word_matrix = (
        word_vectorizer.fit_transform(
            documents
        )
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

    char_matrix = (
        char_vectorizer.fit_transform(
            documents
        )
    )

    char_scores = cosine_similarity(
        char_matrix[-1],
        char_matrix[:-1],
    )[0]

    return (
        0.55 * word_scores
        + 0.45 * char_scores
    )


def words_for_neighborhood(
    words: list[dict],
    segment_start: float,
    segment_end: float,
) -> tuple[int, int]:
    inside = [
        i
        for i, word in enumerate(words)
        if (
            word["end"] >= segment_start
            and word["start"] <= segment_end
        )
    ]

    if not inside:
        raise RuntimeError(
            "No words found for segment "
            f"{segment_start}-{segment_end}"
        )

    start_index = max(
        0,
        min(inside) - LOCAL_WORD_MARGIN,
    )

    end_index = min(
        len(words) - 1,
        max(inside) + LOCAL_WORD_MARGIN,
    )

    return start_index, end_index


def make_local_windows(
    words: list[dict],
    start_index: int,
    end_index: int,
) -> list[dict]:
    candidates = []
    seen = set()

    for size in WINDOW_SIZES:
        for i in range(
            start_index,
            end_index - size + 2,
        ):
            j = i + size - 1

            if j > end_index:
                break

            key = (i, j)

            if key in seen:
                continue

            seen.add(key)

            text = "".join(
                item["word"]
                for item in words[
                    i : j + 1
                ]
            ).strip()

            candidates.append(
                {
                    "start_index": i,
                    "end_index": j,
                    "start": words[i][
                        "start"
                    ],
                    "end": words[j]["end"],
                    "word_count": size,
                    "text": text,
                }
            )

    return candidates


def main() -> None:
    args = parse_args()

    df = pd.read_csv(
        args.questions
    )

    positives = df[
        df["question_type"]
        == "positive"
    ].copy()

    cache = {}
    results = []

    for _, row in positives.iterrows():
        transcript_id = row[
            "transcript_id"
        ]

        stem = normalize_transcript_id(
            transcript_id
        )

        if stem not in cache:
            data = load_asr(
                args.asr_dir,
                transcript_id,
            )

            cache[stem] = {
                "segments": get_segments(
                    data
                ),
                "words": get_words(
                    data
                ),
            }

        segments = cache[stem][
            "segments"
        ]

        words = cache[stem]["words"]

        question = str(
            row["question"]
        )

        gold_start = float(
            row["evidence_start"]
        )

        gold_end = float(
            row["evidence_end"]
        )

        segment_texts = [
            item["text"]
            for item in segments
        ]

        segment_scores = (
            combined_tfidf_scores(
                question,
                segment_texts,
            )
        )

        segment_ranking = (
            np.argsort(
                segment_scores
            )[::-1]
        )

        selected_segments = (
            segment_ranking[
                :TOP_SEGMENTS
            ]
        )

        candidates = []
        candidate_keys = set()

        for segment_index in (
            selected_segments
        ):
            segment = segments[
                int(segment_index)
            ]

            local_start, local_end = (
                words_for_neighborhood(
                    words,
                    segment["start"],
                    segment["end"],
                )
            )

            local_candidates = (
                make_local_windows(
                    words,
                    local_start,
                    local_end,
                )
            )

            for candidate in (
                local_candidates
            ):
                key = (
                    candidate[
                        "start_index"
                    ],
                    candidate[
                        "end_index"
                    ],
                )

                if key in candidate_keys:
                    continue

                candidate_keys.add(key)
                candidates.append(
                    candidate
                )

        candidate_texts = [
            item["text"]
            for item in candidates
        ]

        candidate_scores = (
            combined_tfidf_scores(
                question,
                candidate_texts,
            )
        )

        ranking = np.argsort(
            candidate_scores
        )[::-1]

        candidate_ious = np.array(
            [
                temporal_iou(
                    item["start"],
                    item["end"],
                    gold_start,
                    gold_end,
                )
                for item in candidates
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

        oracle_iou = float(
            candidate_ious.max()
        )

        def overlap_at(
            k: int,
        ) -> bool:
            indices = ranking[:k]

            return bool(
                np.any(
                    candidate_ious[
                        indices
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
                "gold_start": (
                    gold_start
                ),
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
                "score": float(
                    candidate_scores[
                        top_index
                    ]
                ),
                "tiou": top_iou,
                "candidate_oracle_tiou": (
                    oracle_iou
                ),
                "top1_overlap": (
                    overlap_at(1)
                ),
                "top3_overlap": (
                    overlap_at(3)
                ),
                "top5_overlap": (
                    overlap_at(5)
                ),
                "top10_overlap": (
                    overlap_at(10)
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

    tiou = result_df["tiou"]
    oracle = result_df[
        "candidate_oracle_tiou"
    ]

    print("=" * 78)
    print(
        "Nordic AI Cup 2026 — "
        "Coarse-to-Fine Evidence Retrieval"
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
        f"{tiou.mean():.4f}"
    )

    print(
        f"Median tIoU:     "
        f"{tiou.median():.4f}"
    )

    print(
        f"Any overlap:     "
        f"{(tiou > 0).mean():.4f}"
    )

    print(
        f"tIoU >= 0.25:    "
        f"{(tiou >= 0.25).mean():.4f}"
    )

    print(
        f"tIoU >= 0.50:    "
        f"{(tiou >= 0.50).mean():.4f}"
    )

    print(
        f"tIoU >= 0.75:    "
        f"{(tiou >= 0.75).mean():.4f}"
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
    print("LOCAL-CANDIDATE CEILING")
    print("-" * 78)

    print(
        "Oracle local-window mean tIoU:   "
        f"{oracle.mean():.4f}"
    )

    print(
        "Oracle local-window median tIoU: "
        f"{oracle.median():.4f}"
    )

    print()

    print(
        "Top-1 zero-overlap failures: "
        f"{(tiou == 0).sum()}"
    )

    print()
    print("REFERENCE")
    print("-" * 78)

    print(
        "Segment top-1 mean tIoU:       "
        "0.4016"
    )

    print(
        "Segment R@3 overlap:            "
        "0.8513"
    )

    print(
        "Global word-window mean tIoU:   "
        "0.3522"
    )

    print(
        "Word-timestamp oracle mean:     "
        "0.9294"
    )

    print()
    print(
        f"Detailed results: "
        f"{args.output}"
    )


if __name__ == "__main__":
    main()