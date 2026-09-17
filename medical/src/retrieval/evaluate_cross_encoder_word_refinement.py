from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import CrossEncoder
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
    r"medical\artifacts\retrieval\cross_encoder_word_refinement.csv"
)

MODEL_ID = "cross-encoder/ms-marco-MiniLM-L6-v2"

LEXICAL_TOP_K = 8
SEMANTIC_TOP_SEGMENTS = 3

WINDOW_SIZES = (
    4,
    6,
    8,
    10,
    12,
    14,
    16,
    20,
    24,
)

LOCAL_WORD_MARGIN = 8


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
    segments = []

    for item in data["segments"]:
        text = str(
            item["text"]
        ).strip()

        if not text:
            continue

        segments.append(
            {
                "start": float(
                    item["start"]
                ),
                "end": float(
                    item["end"]
                ),
                "text": text,
            }
        )

    return segments


def get_words(
    data: dict,
) -> list[dict]:
    words = []

    for item in data["words"]:
        start = item.get("start")
        end = item.get("end")

        if start is None or end is None:
            continue

        words.append(
            {
                "start": float(start),
                "end": float(end),
                "word": str(
                    item["word"]
                ),
            }
        )

    return words


def lexical_scores(
    question: str,
    texts: list[str],
) -> np.ndarray:
    documents = texts + [question]

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


def segment_word_bounds(
    words: list[dict],
    segment: dict,
) -> tuple[int, int]:
    indices = [
        index
        for index, word in enumerate(words)
        if (
            word["end"]
            >= segment["start"]
            and word["start"]
            <= segment["end"]
        )
    ]

    if not indices:
        raise RuntimeError(
            "Could not map segment to words: "
            f"{segment}"
        )

    start_index = max(
        0,
        min(indices) - LOCAL_WORD_MARGIN,
    )

    end_index = min(
        len(words) - 1,
        max(indices) + LOCAL_WORD_MARGIN,
    )

    return start_index, end_index


def make_windows(
    words: list[dict],
    neighborhoods: list[
        tuple[int, int]
    ],
) -> list[dict]:
    candidates = []
    seen = set()

    for neighborhood_start, neighborhood_end in neighborhoods:
        for size in WINDOW_SIZES:
            if size > (
                neighborhood_end
                - neighborhood_start
                + 1
            ):
                continue

            final_start = (
                neighborhood_end
                - size
                + 1
            )

            for start_index in range(
                neighborhood_start,
                final_start + 1,
            ):
                end_index = (
                    start_index
                    + size
                    - 1
                )

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
                        start_index:
                        end_index + 1
                    ]
                ).strip()

                candidates.append(
                    {
                        "start_index": (
                            start_index
                        ),
                        "end_index": (
                            end_index
                        ),
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


def main() -> None:
    args = parse_args()

    df = pd.read_csv(
        args.questions
    )

    positives = df[
        df["question_type"]
        == "positive"
    ].copy()

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 78)
    print(
        "Loading semantic reranker"
    )
    print("=" * 78)
    print(f"Model:  {MODEL_ID}")
    print(f"Device: {device}")

    model = CrossEncoder(
        MODEL_ID,
        device=device,
        max_length=256,
    )

    cache = {}
    results = []

    total_rerank_seconds = 0.0
    total_window_pairs = 0

    for row_index, (
        _,
        row,
    ) in enumerate(
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
            data = load_asr(
                args.asr_dir,
                transcript_id,
            )

            cache[stem] = {
                "segments": (
                    get_segments(data)
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

        # -------------------------------------------------
        # Stage 1: lexical segment retrieval
        # -------------------------------------------------

        segment_texts = [
            segment["text"]
            for segment in segments
        ]

        lex_scores = lexical_scores(
            question,
            segment_texts,
        )

        lexical_ranking = np.argsort(
            lex_scores
        )[::-1]

        lexical_candidates = (
            lexical_ranking[
                :LEXICAL_TOP_K
            ]
        )

        # -------------------------------------------------
        # Stage 2: semantic segment reranking
        # -------------------------------------------------

        segment_pairs = [
            (
                question,
                segments[
                    int(index)
                ]["text"],
            )
            for index in (
                lexical_candidates
            )
        ]

        segment_scores = (
            np.asarray(
                model.predict(
                    segment_pairs,
                    batch_size=32,
                    show_progress_bar=False,
                )
            ).reshape(-1)
        )

        semantic_order = (
            np.argsort(
                segment_scores
            )[::-1]
        )

        top_segment_indices = [
            int(
                lexical_candidates[
                    index
                ]
            )
            for index in (
                semantic_order[
                    :SEMANTIC_TOP_SEGMENTS
                ]
            )
        ]

        # -------------------------------------------------
        # Stage 3: local word candidate generation
        # -------------------------------------------------

        neighborhoods = []

        for segment_index in (
            top_segment_indices
        ):
            neighborhoods.append(
                segment_word_bounds(
                    words,
                    segments[
                        segment_index
                    ],
                )
            )

        windows = make_windows(
            words,
            neighborhoods,
        )

        if not windows:
            raise RuntimeError(
                "No word windows generated "
                f"for {row['question_id']}"
            )

        # -------------------------------------------------
        # Stage 4: semantic word-window reranking
        # -------------------------------------------------

        window_pairs = [
            (
                question,
                window["text"],
            )
            for window in windows
        ]

        rerank_start = (
            time.perf_counter()
        )

        window_scores = np.asarray(
            model.predict(
                window_pairs,
                batch_size=64,
                show_progress_bar=False,
            )
        ).reshape(-1)

        total_rerank_seconds += (
            time.perf_counter()
            - rerank_start
        )

        total_window_pairs += len(
            window_pairs
        )

        ranking = np.argsort(
            window_scores
        )[::-1]

        candidate_ious = np.array(
            [
                temporal_iou(
                    window["start"],
                    window["end"],
                    gold_start,
                    gold_end,
                )
                for window in windows
            ],
            dtype=float,
        )

        top_index = int(
            ranking[0]
        )

        top = windows[top_index]

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
                    top[
                        "word_count"
                    ]
                ),
                "pred_text": top[
                    "text"
                ],
                "semantic_score": (
                    float(
                        window_scores[
                            top_index
                        ]
                    )
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
            }
        )

        if (
            row_index % 25 == 0
            or row_index
            == len(positives)
        ):
            print(
                f"[{row_index:03d}/"
                f"{len(positives):03d}] "
                f"processed"
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

    print()
    print("=" * 78)
    print(
        "Nordic AI Cup 2026 — "
        "Semantic Word Refinement"
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

    print("RERANKED WINDOW RECALL")
    print("-" * 78)

    print(
        "R@1 any overlap: "
        f"{result_df['top1_overlap'].mean():.4f}"
    )

    print(
        "R@3 any overlap: "
        f"{result_df['top3_overlap'].mean():.4f}"
    )

    print(
        "R@5 any overlap: "
        f"{result_df['top5_overlap'].mean():.4f}"
    )

    print()
    print("LOCAL CANDIDATE CEILING")
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
    print("LATENCY")
    print("-" * 78)

    print(
        f"Word-window pairs scored: "
        f"{total_window_pairs}"
    )

    print(
        f"Window rerank time: "
        f"{total_rerank_seconds:.2f} s"
    )

    print(
        f"Mean per question: "
        f"{total_rerank_seconds / len(result_df):.4f} s"
    )

    print(
        f"Mean per 10-question request: "
        f"{10 * total_rerank_seconds / len(result_df):.4f} s"
    )

    print()
    print("REFERENCE")
    print("-" * 78)

    print(
        "Segment TF-IDF mean tIoU:       "
        "0.4016"
    )

    print(
        "Cross-encoder segment mean:      "
        "0.4347"
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