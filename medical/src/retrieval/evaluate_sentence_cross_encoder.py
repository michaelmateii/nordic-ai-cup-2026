from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import CrossEncoder


QUESTIONS = Path(
    r"C:\Users\Calle\Projects\Nordic-AI-Cup-2026-official"
    r"\medical-appointment\data\question_train.csv"
)

ASR_DIR = Path(
    r"medical\artifacts\asr\faster_distil_large_v3_int8f32_b1"
)

OUTPUT = Path(
    r"medical\artifacts\retrieval\sentence_cross_encoder_results.csv"
)

MODEL_ID = "cross-encoder/ms-marco-MiniLM-L6-v2"

MAX_SENTENCES = 3


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

    if not path.exists():
        raise FileNotFoundError(path)

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
                "start": float(item["start"]),
                "end": float(item["end"]),
                "word": str(item["word"]),
            }
        )

    return words


def is_sentence_end(text: str) -> bool:
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


def make_candidates(
    sentences: list[dict],
) -> list[dict]:
    candidates = []

    for size in range(
        1,
        MAX_SENTENCES + 1,
    ):
        for start in range(
            0,
            len(sentences) - size + 1,
        ):
            group = sentences[
                start:
                start + size
            ]

            candidates.append(
                {
                    "sentence_count": size,
                    "start": group[0]["start"],
                    "end": group[-1]["end"],
                    "text": " ".join(
                        item["text"]
                        for item in group
                    ),
                }
            )

    return candidates


def main() -> None:
    df = pd.read_csv(
        QUESTIONS
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
        "Nordic AI Cup 2026 — "
        "Sentence Cross-Encoder Retrieval"
    )
    print("=" * 78)
    print(f"Model:  {MODEL_ID}")
    print(f"Device: {device}")
    print()

    model = CrossEncoder(
        MODEL_ID,
        device=device,
        max_length=256,
    )

    cache = {}
    rows = []

    total_pairs = 0
    total_seconds = 0.0

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
            words = load_words(
                transcript_id
            )

            sentences = make_sentences(
                words
            )

            cache[stem] = make_candidates(
                sentences
            )

        candidates = cache[stem]

        question = str(
            row["question"]
        )

        pairs = [
            (
                question,
                candidate["text"],
            )
            for candidate in candidates
        ]

        start_time = time.perf_counter()

        scores = np.asarray(
            model.predict(
                pairs,
                batch_size=64,
                show_progress_bar=False,
            )
        ).reshape(-1)

        total_seconds += (
            time.perf_counter()
            - start_time
        )

        total_pairs += len(pairs)

        ranking = np.argsort(
            scores
        )[::-1]

        gold_start = float(
            row["evidence_start"]
        )

        gold_end = float(
            row["evidence_end"]
        )

        ious = np.asarray(
            [
                temporal_iou(
                    candidate["start"],
                    candidate["end"],
                    gold_start,
                    gold_end,
                )
                for candidate
                in candidates
            ],
            dtype=float,
        )

        best_index = int(
            ranking[0]
        )

        best = candidates[
            best_index
        ]

        oracle_index = int(
            np.argmax(ious)
        )

        def overlap_at(k: int) -> bool:
            return bool(
                np.any(
                    ious[
                        ranking[:k]
                    ] > 0
                )
            )

        rows.append(
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
                "pred_start": best[
                    "start"
                ],
                "pred_end": best[
                    "end"
                ],
                "pred_text": best[
                    "text"
                ],
                "pred_sentence_count": (
                    best[
                        "sentence_count"
                    ]
                ),
                "score": float(
                    scores[
                        best_index
                    ]
                ),
                "tiou": float(
                    ious[
                        best_index
                    ]
                ),
                "candidate_oracle_tiou": (
                    float(
                        ious[
                            oracle_index
                        ]
                    )
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

    tiou = result_df["tiou"]

    oracle = result_df[
        "candidate_oracle_tiou"
    ]

    print()
    print("=" * 78)
    print("RESULT")
    print("=" * 78)

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
        f"Pairs scored:      "
        f"{total_pairs}"
    )

    print(
        f"Total rerank time: "
        f"{total_seconds:.2f} s"
    )

    print(
        f"Mean/question:     "
        f"{total_seconds / len(result_df):.4f} s"
    )

    print(
        f"Estimated/10 q:    "
        f"{10 * total_seconds / len(result_df):.4f} s"
    )

    print()
    print("REFERENCE")
    print("-" * 78)
    print(
        "M010 segment mean tIoU: 0.4347"
    )
    print(
        "M023 candidate oracle:  0.7997"
    )

    print()
    print(
        f"Detailed results: {OUTPUT}"
    )


if __name__ == "__main__":
    main()