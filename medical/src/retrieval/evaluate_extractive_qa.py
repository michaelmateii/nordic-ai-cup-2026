from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import CrossEncoder
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

DEFAULT_QUESTION_CSV = Path(
    r"C:\Users\Calle\Projects\Nordic-AI-Cup-2026-official"
    r"\medical-appointment\data\question_train.csv"
)

DEFAULT_ASR_DIR = Path(
    r"medical\artifacts\asr\faster_distil_large_v3_int8f32_b1"
)

DEFAULT_OUTPUT = Path(
    r"medical\artifacts\retrieval\extractive_qa_results.csv"
)

RETRIEVER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
QA_MODEL = "deepset/minilm-uncased-squad2"

LEXICAL_TOP_K = 8
SEMANTIC_TOP_K = 3

PADDINGS = (0, 1, 2, 4, 6)


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


def lexical_scores(
    question: str,
    segments: list[dict],
) -> np.ndarray:
    texts = [
        segment["text"]
        for segment in segments
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


def words_in_segment(
    words: list[dict],
    segment: dict,
) -> list[int]:
    return [
        index
        for index, word in enumerate(words)
        if (
            word["end"] >= segment["start"]
            and word["start"] <= segment["end"]
        )
    ]


def build_context(
    words: list[dict],
    indices: list[int],
) -> tuple[str, list[tuple[int, int, int]]]:
    """
    Build exact context text from ASR words.

    Returns:
      context
      [(char_start, char_end, global_word_index), ...]
    """

    pieces = []
    mapping = []

    cursor = 0

    for word_index in indices:
        text = words[word_index]["word"]

        char_start = cursor
        pieces.append(text)
        cursor += len(text)
        char_end = cursor

        mapping.append(
            (
                char_start,
                char_end,
                word_index,
            )
        )

    return "".join(pieces).strip(), mapping


def answer_to_word_bounds(
    answer_start: int,
    answer_end: int,
    mapping: list[tuple[int, int, int]],
) -> tuple[int, int] | None:
    selected = []

    for (
        char_start,
        char_end,
        word_index,
    ) in mapping:
        if (
            char_end > answer_start
            and char_start < answer_end
        ):
            selected.append(word_index)

    if not selected:
        return None

    return min(selected), max(selected)

def run_extractive_qa(
    question: str,
    context: str,
    tokenizer,
    model,
    device: str,
) -> dict:
    encoded = tokenizer(
        question,
        context,
        return_tensors="pt",
        truncation="only_second",
        max_length=384,
        return_offsets_mapping=True,
    )

    offset_mapping = encoded.pop(
        "offset_mapping"
    )[0]

    sequence_ids = encoded.sequence_ids(
        0
    )

    encoded = {
        key: value.to(device)
        for key, value in encoded.items()
    }

    with torch.inference_mode():
        output = model(
            **encoded
        )

    start_logits = output.start_logits[0]
    end_logits = output.end_logits[0]

    # Only context tokens are valid answer tokens.
    valid_indices = [
        index
        for index, sequence_id
        in enumerate(sequence_ids)
        if sequence_id == 1
    ]

    if not valid_indices:
        return {
            "answer": "",
            "start": 0,
            "end": 0,
            "score": 0.0,
        }

    best_score = float("-inf")
    best_start_token = None
    best_end_token = None

    # Keep answers reasonably short.
    max_answer_tokens = 40

    for start_token in valid_indices:
        max_end_token = min(
            start_token
            + max_answer_tokens
            - 1,
            valid_indices[-1],
        )

        for end_token in range(
            start_token,
            max_end_token + 1,
        ):
            if sequence_ids[end_token] != 1:
                break

            score = float(
                start_logits[start_token]
                + end_logits[end_token]
            )

            if score > best_score:
                best_score = score
                best_start_token = (
                    start_token
                )
                best_end_token = (
                    end_token
                )

    if (
        best_start_token is None
        or best_end_token is None
    ):
        return {
            "answer": "",
            "start": 0,
            "end": 0,
            "score": 0.0,
        }

    char_start = int(
        offset_mapping[
            best_start_token
        ][0]
    )

    char_end = int(
        offset_mapping[
            best_end_token
        ][1]
    )

    answer = context[
        char_start:char_end
    ]

    # The absolute logit sum is sufficient for choosing
    # the best QA span among the retrieved segments.
    return {
        "answer": answer,
        "start": char_start,
        "end": char_end,
        "score": best_score,
    }

def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.questions)

    positives = df[
        df["question_type"] == "positive"
    ].copy()

    device_name = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 78)
    print("Loading retrieval + extractive QA models")
    print("=" * 78)
    print(f"Retriever: {RETRIEVER_MODEL}")
    print(f"QA model:  {QA_MODEL}")
    print(f"Device:    {device_name}")

    retriever = CrossEncoder(
        RETRIEVER_MODEL,
        device=device_name,
        max_length=256,
    )

    qa_tokenizer = AutoTokenizer.from_pretrained(
        QA_MODEL
    )

    qa_model = (
        AutoModelForQuestionAnswering
        .from_pretrained(QA_MODEL)
        .to(device_name)
        .eval()
    )

    cache = {}
    rows = []

    start_time = time.perf_counter()

    for row_number, (_, row) in enumerate(
        positives.iterrows(),
        start=1,
    ):
        transcript_id = row["transcript_id"]
        stem = normalize_transcript_id(
            transcript_id
        )

        if stem not in cache:
            data = load_asr(
                args.asr_dir,
                transcript_id,
            )

            cache[stem] = {
                "segments": get_segments(data),
                "words": get_words(data),
            }

        segments = cache[stem]["segments"]
        words = cache[stem]["words"]

        question = str(row["question"])

        gold_start = float(
            row["evidence_start"]
        )

        gold_end = float(
            row["evidence_end"]
        )

        # Stage 1: lexical segment candidates.
        first_scores = lexical_scores(
            question,
            segments,
        )

        lexical_indices = np.argsort(
            first_scores
        )[::-1][:LEXICAL_TOP_K]

        # Stage 2: semantic segment reranking.
        pairs = [
            (
                question,
                segments[int(index)]["text"],
            )
            for index in lexical_indices
        ]

        semantic_scores = np.asarray(
            retriever.predict(
                pairs,
                batch_size=32,
                show_progress_bar=False,
            )
        ).reshape(-1)

        semantic_order = np.argsort(
            semantic_scores
        )[::-1]

        top_segment_indices = [
            int(lexical_indices[position])
            for position in semantic_order[
                :SEMANTIC_TOP_K
            ]
        ]

        qa_candidates = []

        for segment_index in top_segment_indices:
            segment = segments[
                segment_index
            ]

            word_indices = words_in_segment(
                words,
                segment,
            )

            if not word_indices:
                continue

            context, mapping = build_context(
                words,
                word_indices,
            )

            if not context:
                continue

            result = run_extractive_qa(
                question=question,
                context=context,
                tokenizer=qa_tokenizer,
                model=qa_model,
                device=device_name,
            )

            answer_text = str(
                result.get("answer", "")
            )

            answer_start = int(
                result.get("start", 0)
            )

            answer_end = int(
                result.get("end", 0)
            )

            qa_score = float(
                result.get("score", 0.0)
            )

            bounds = answer_to_word_bounds(
                answer_start,
                answer_end,
                mapping,
            )

            if bounds is None:
                continue

            qa_candidates.append(
                {
                    "qa_score": qa_score,
                    "answer": answer_text,
                    "start_index": bounds[0],
                    "end_index": bounds[1],
                    "segment_index": segment_index,
                }
            )

        if not qa_candidates:
            raise RuntimeError(
                "No QA span produced for "
                f"{row['question_id']}"
            )

        best = max(
            qa_candidates,
            key=lambda item: item["qa_score"],
        )

        output = {
            "question_id": row["question_id"],
            "transcript_id": transcript_id,
            "question": question,
            "gold_start": gold_start,
            "gold_end": gold_end,
            "qa_score": best["qa_score"],
            "qa_answer": best["answer"],
        }

        for padding in PADDINGS:
            start_index = max(
                0,
                best["start_index"] - padding,
            )

            end_index = min(
                len(words) - 1,
                best["end_index"] + padding,
            )

            pred_start = words[
                start_index
            ]["start"]

            pred_end = words[
                end_index
            ]["end"]

            tiou = temporal_iou(
                pred_start,
                pred_end,
                gold_start,
                gold_end,
            )

            output[
                f"start_p{padding}"
            ] = pred_start

            output[
                f"end_p{padding}"
            ] = pred_end

            output[
                f"tiou_p{padding}"
            ] = tiou

        rows.append(output)

        if (
            row_number % 25 == 0
            or row_number == len(positives)
        ):
            print(
                f"[{row_number:03d}/"
                f"{len(positives):03d}] processed"
            )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    result_df = pd.DataFrame(rows)

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_df.to_csv(
        args.output,
        index=False,
    )

    print()
    print("=" * 78)
    print(
        "Nordic AI Cup 2026 — "
        "Extractive QA Evidence Anchor"
    )
    print("=" * 78)

    for padding in PADDINGS:
        scores = result_df[
            f"tiou_p{padding}"
        ]

        print()
        print(
            f"PADDING ±{padding} WORDS"
        )
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
            f"tIoU >= 0.50:    "
            f"{(scores >= 0.50).mean():.4f}"
        )

        print(
            f"tIoU >= 0.75:    "
            f"{(scores >= 0.75).mean():.4f}"
        )

    print()
    print("LATENCY")
    print("-" * 78)

    print(
        f"Total:            "
        f"{elapsed:.2f} s"
    )

    print(
        f"Mean/question:    "
        f"{elapsed / len(result_df):.4f} s"
    )

    print(
        f"Estimated/10 q:   "
        f"{10 * elapsed / len(result_df):.4f} s"
    )

    print()
    print("REFERENCE")
    print("-" * 78)
    print(
        "M010 segment mean tIoU:        0.4347"
    )
    print(
        "M007 word oracle mean tIoU:    0.9294"
    )

    print()
    print(
        f"Detailed results: {args.output}"
    )


if __name__ == "__main__":
    main()