from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


import numpy as np
import pandas as pd
import torch
from sentence_transformers import CrossEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[3]

CLASSIFICATION_DIR = (
    REPO_ROOT
    / "medical"
    / "src"
    / "classification"
)

if str(CLASSIFICATION_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(CLASSIFICATION_DIR),
    )

from evaluate_nli_segmentwise import question_to_claim


DEFAULT_QUESTION_CSV = Path(
    r"C:\Users\Calle\Projects\Nordic-AI-Cup-2026-official"
    r"\medical-appointment\data\question_train.csv"
)

DEFAULT_ASR_DIR = Path(
    r"medical\artifacts\asr\faster_distil_large_v3_int8f32_b1"
)

DEFAULT_OUTPUT = Path(
    r"medical\artifacts\retrieval\nli_word_refinement.csv"
)

RETRIEVER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
NLI_MODEL = "cross-encoder/nli-deberta-v3-small"

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
    28,
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


def normalize_transcript_id(value: object) -> str:
    text = Path(str(value).strip()).stem

    if text.startswith("conversation_sample_"):
        return text

    if text.startswith("sample_"):
        return f"conversation_{text}"

    if text.isdigit():
        return f"conversation_sample_{int(text)}"

    return text


def load_asr(
    asr_dir: Path,
    transcript_id: object,
) -> dict:
    stem = normalize_transcript_id(transcript_id)
    path = asr_dir / f"{stem}.json"

    if not path.exists():
        raise FileNotFoundError(path)

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def get_segments(data: dict) -> list[dict]:
    return [
        {
            "start": float(item["start"]),
            "end": float(item["end"]),
            "text": str(item["text"]).strip(),
        }
        for item in data["segments"]
        if str(item["text"]).strip()
    ]


def get_words(data: dict) -> list[dict]:
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
                "word": str(item["word"]),
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


def segment_word_bounds(
    words: list[dict],
    segment: dict,
) -> tuple[int, int]:
    indices = [
        i
        for i, word in enumerate(words)
        if (
            word["end"] >= segment["start"]
            and word["start"] <= segment["end"]
        )
    ]

    if not indices:
        raise RuntimeError(
            f"No words found for segment {segment}"
        )

    start = max(
        0,
        min(indices) - LOCAL_WORD_MARGIN,
    )

    end = min(
        len(words) - 1,
        max(indices) + LOCAL_WORD_MARGIN,
    )

    return start, end


def make_windows(
    words: list[dict],
    neighborhoods: list[tuple[int, int]],
) -> list[dict]:
    windows = []
    seen = set()

    for local_start, local_end in neighborhoods:
        available = (
            local_end
            - local_start
            + 1
        )

        for size in WINDOW_SIZES:
            if size > available:
                continue

            last_start = (
                local_end
                - size
                + 1
            )

            for start_index in range(
                local_start,
                last_start + 1,
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
                    word["word"]
                    for word in words[
                        start_index:
                        end_index + 1
                    ]
                ).strip()

                windows.append(
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

    return windows


def get_label_indices(model) -> dict[str, int]:
    mapping = {
        int(key): str(value).lower()
        for key, value
        in model.config.id2label.items()
    }

    result = {}

    for index, label in mapping.items():
        for target in (
            "contradiction",
            "entailment",
            "neutral",
        ):
            if target in label:
                result[target] = index

    if len(result) != 3:
        result = {
            "contradiction": 0,
            "entailment": 1,
            "neutral": 2,
        }

    return result


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.questions)

    positives = df[
        df["question_type"] == "positive"
    ].copy()

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 78)
    print("Loading M022 retrieval models")
    print("=" * 78)
    print(f"Retriever: {RETRIEVER_MODEL}")
    print(f"NLI:       {NLI_MODEL}")
    print(f"Device:    {device}")

    retriever = CrossEncoder(
        RETRIEVER_MODEL,
        device=device,
        max_length=256,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        NLI_MODEL
    )

    nli_model = (
        AutoModelForSequenceClassification
        .from_pretrained(NLI_MODEL)
        .to(device)
        .eval()
    )

    labels = get_label_indices(
        nli_model
    )

    cache = {}
    results = []

    total_nli_pairs = 0
    total_nli_seconds = 0.0

    for row_number, (_, row) in enumerate(
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

        claim = question_to_claim(
            question
        )

        gold_start = float(
            row["evidence_start"]
        )

        gold_end = float(
            row["evidence_end"]
        )

        # ---------------------------------------------
        # Coarse segment retrieval
        # ---------------------------------------------

        segment_texts = [
            segment["text"]
            for segment in segments
        ]

        lex_scores = lexical_scores(
            question,
            segment_texts,
        )

        lexical_indices = np.argsort(
            lex_scores
        )[::-1][:LEXICAL_TOP_K]

        retrieval_pairs = [
            (
                question,
                segments[
                    int(index)
                ]["text"],
            )
            for index in lexical_indices
        ]

        semantic_scores = np.asarray(
            retriever.predict(
                retrieval_pairs,
                batch_size=32,
                show_progress_bar=False,
            )
        ).reshape(-1)

        order = np.argsort(
            semantic_scores
        )[::-1]

        top_segments = [
            int(
                lexical_indices[
                    position
                ]
            )
            for position in order[
                :SEMANTIC_TOP_SEGMENTS
            ]
        ]

        neighborhoods = [
            segment_word_bounds(
                words,
                segments[index],
            )
            for index in top_segments
        ]

        windows = make_windows(
            words,
            neighborhoods,
        )

        if not windows:
            raise RuntimeError(
                f"No windows for "
                f"{row['question_id']}"
            )

        # ---------------------------------------------
        # NLI reranking of local word spans
        # ---------------------------------------------

        premises = [
            window["text"]
            for window in windows
        ]

        hypotheses = [
            claim
            for _ in windows
        ]

        start_time = (
            time.perf_counter()
        )

        all_margins = []

        batch_size = 128

        for start in range(
            0,
            len(windows),
            batch_size,
        ):
            batch_premises = premises[
                start:
                start + batch_size
            ]

            batch_hypotheses = hypotheses[
                start:
                start + batch_size
            ]

            encoded = tokenizer(
                batch_premises,
                batch_hypotheses,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )

            encoded = {
                key: value.to(device)
                for key, value
                in encoded.items()
            }

            with torch.inference_mode():
                logits = nli_model(
                    **encoded
                ).logits

            probs = torch.softmax(
                logits,
                dim=-1,
            )

            entailment = probs[
                :,
                labels["entailment"],
            ]

            contradiction = probs[
                :,
                labels["contradiction"],
            ]

            margins = (
                entailment
                - contradiction
            )

            all_margins.extend(
                margins.detach()
                .cpu()
                .numpy()
                .tolist()
            )

        total_nli_seconds += (
            time.perf_counter()
            - start_time
        )

        total_nli_pairs += len(
            windows
        )

        margins = np.asarray(
            all_margins,
            dtype=float,
        )

        ranking = np.argsort(
            margins
        )[::-1]

        candidate_ious = np.asarray(
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

        best_index = int(
            ranking[0]
        )

        best = windows[
            best_index
        ]

        best_iou = float(
            candidate_ious[
                best_index
            ]
        )

        oracle_iou = float(
            candidate_ious.max()
        )

        def overlap_at(k: int) -> bool:
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
                "hypothesis": claim,
                "gold_start": gold_start,
                "gold_end": gold_end,
                "pred_start": best[
                    "start"
                ],
                "pred_end": best[
                    "end"
                ],
                "pred_word_count": (
                    best[
                        "word_count"
                    ]
                ),
                "pred_text": best[
                    "text"
                ],
                "nli_margin": float(
                    margins[
                        best_index
                    ]
                ),
                "tiou": best_iou,
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
            row_number % 25 == 0
            or row_number
            == len(positives)
        ):
            print(
                f"[{row_number:03d}/"
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
    )

    tiou = result_df["tiou"]

    oracle = result_df[
        "candidate_oracle_tiou"
    ]

    print()
    print("=" * 78)
    print(
        "Nordic AI Cup 2026 — "
        "NLI Word-Span Refinement"
    )
    print("=" * 78)

    print(
        f"Positive questions: "
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
    print("RERANKED RECALL")
    print("-" * 78)

    print(
        f"R@1 overlap: "
        f"{result_df['top1_overlap'].mean():.4f}"
    )

    print(
        f"R@3 overlap: "
        f"{result_df['top3_overlap'].mean():.4f}"
    )

    print(
        f"R@5 overlap: "
        f"{result_df['top5_overlap'].mean():.4f}"
    )

    print()
    print("CANDIDATE CEILING")
    print("-" * 78)

    print(
        f"Oracle mean tIoU:   "
        f"{oracle.mean():.4f}"
    )

    print(
        f"Oracle median tIoU: "
        f"{oracle.median():.4f}"
    )

    print()
    print("LATENCY")
    print("-" * 78)

    print(
        f"NLI window pairs: "
        f"{total_nli_pairs}"
    )

    print(
        f"NLI rerank time: "
        f"{total_nli_seconds:.2f} s"
    )

    print(
        f"Mean/question: "
        f"{total_nli_seconds / len(result_df):.4f} s"
    )

    print(
        f"Estimated/10 questions: "
        f"{10 * total_nli_seconds / len(result_df):.4f} s"
    )

    print()
    print("REFERENCE")
    print("-" * 78)
    print(
        "M010 segment mean tIoU:   0.4347"
    )
    print(
        "M011 MS-MARCO word mean:  0.3612"
    )
    print(
        "M007 word oracle mean:    0.9294"
    )

    print()
    print(
        f"Detailed results: "
        f"{args.output}"
    )


if __name__ == "__main__":
    main()