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
    r"medical\artifacts\retrieval\cross_encoder_segment_results.csv"
)

MODEL_ID = "cross-encoder/ms-marco-MiniLM-L6-v2"
LEXICAL_TOP_K = 8


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


def load_segments(
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

    segments = []

    for segment in data["segments"]:
        text = str(segment["text"]).strip()

        if not text:
            continue

        segments.append(
            {
                "start": float(segment["start"]),
                "end": float(segment["end"]),
                "text": text,
            }
        )

    return segments


def lexical_scores(
    question: str,
    segments: list[dict],
) -> np.ndarray:
    texts = [segment["text"] for segment in segments]
    documents = texts + [question]

    word_vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        sublinear_tf=True,
        token_pattern=r"(?u)\b\w+\b",
    )

    word_matrix = word_vectorizer.fit_transform(documents)

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

    char_matrix = char_vectorizer.fit_transform(documents)

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

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 78)
    print("Loading semantic reranker")
    print("=" * 78)
    print(f"Model:  {MODEL_ID}")
    print(f"Device: {device}")

    load_start = time.perf_counter()

    model = CrossEncoder(
        MODEL_ID,
        device=device,
        max_length=256,
    )

    print(
        f"Load:   "
        f"{time.perf_counter() - load_start:.2f} s"
    )
    print()

    cache: dict[str, list[dict]] = {}
    results = []

    rerank_start = time.perf_counter()

    for _, row in positives.iterrows():
        transcript_id = row["transcript_id"]
        stem = normalize_transcript_id(transcript_id)

        if stem not in cache:
            cache[stem] = load_segments(
                args.asr_dir,
                transcript_id,
            )

        segments = cache[stem]
        question = str(row["question"])

        gold_start = float(row["evidence_start"])
        gold_end = float(row["evidence_end"])

        first_stage_scores = lexical_scores(
            question,
            segments,
        )

        first_stage_ranking = np.argsort(
            first_stage_scores
        )[::-1]

        candidate_indices = first_stage_ranking[
            :LEXICAL_TOP_K
        ]

        pairs = [
            (
                question,
                segments[int(index)]["text"],
            )
            for index in candidate_indices
        ]

        semantic_scores = np.asarray(
            model.predict(
                pairs,
                batch_size=32,
                show_progress_bar=False,
            )
        ).reshape(-1)

        reranked_order = np.argsort(
            semantic_scores
        )[::-1]

        ranked_indices = [
            int(candidate_indices[i])
            for i in reranked_order
        ]

        candidate_ious = np.array(
            [
                temporal_iou(
                    segment["start"],
                    segment["end"],
                    gold_start,
                    gold_end,
                )
                for segment in segments
            ],
            dtype=float,
        )

        top_index = ranked_indices[0]
        top = segments[top_index]
        top_iou = float(candidate_ious[top_index])

        def overlap_at(k: int) -> bool:
            indices = ranked_indices[:k]

            return bool(
                np.any(
                    candidate_ious[indices] > 0
                )
            )

        oracle_top_k_iou = float(
            max(
                candidate_ious[index]
                for index in candidate_indices
            )
        )

        results.append(
            {
                "question_id": row["question_id"],
                "transcript_id": transcript_id,
                "question": question,
                "gold_start": gold_start,
                "gold_end": gold_end,
                "pred_start": top["start"],
                "pred_end": top["end"],
                "pred_text": top["text"],
                "semantic_score": float(
                    semantic_scores[
                        int(reranked_order[0])
                    ]
                ),
                "tiou": top_iou,
                "top1_overlap": overlap_at(1),
                "top3_overlap": overlap_at(3),
                "top5_overlap": overlap_at(5),
                "oracle_lexical_top8_tiou": (
                    oracle_top_k_iou
                ),
            }
        )

    elapsed = time.perf_counter() - rerank_start

    result_df = pd.DataFrame(results)

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

    print("=" * 78)
    print(
        "Nordic AI Cup 2026 — "
        "Cross-Encoder Segment Reranking"
    )
    print("=" * 78)
    print(
        f"Gold-positive questions: "
        f"{len(result_df)}"
    )
    print()

    print("TOP-1 LOCALIZATION")
    print("-" * 78)
    print(f"Mean tIoU:       {tiou.mean():.4f}")
    print(f"Median tIoU:     {tiou.median():.4f}")
    print(f"Any overlap:     {(tiou > 0).mean():.4f}")
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
    print("RERANKED RECALL")
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
    print("FIRST-STAGE CEILING")
    print("-" * 78)

    oracle = result_df[
        "oracle_lexical_top8_tiou"
    ]

    print(
        "Oracle lexical-top8 mean tIoU: "
        f"{oracle.mean():.4f}"
    )

    print()
    print("LATENCY")
    print("-" * 78)
    print(
        f"Total 195-question rerank time: "
        f"{elapsed:.2f} s"
    )
    print(
        f"Mean per question: "
        f"{elapsed / len(result_df):.4f} s"
    )

    print()
    print("REFERENCE")
    print("-" * 78)
    print(
        "Segment TF-IDF mean tIoU: 0.4016"
    )
    print(
        "Segment TF-IDF R@1:       0.6718"
    )
    print(
        "Segment TF-IDF R@3:       0.8513"
    )

    print()
    print(
        f"Detailed results: {args.output}"
    )


if __name__ == "__main__":
    main()