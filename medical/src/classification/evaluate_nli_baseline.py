from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import CrossEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForSequenceClassification, AutoTokenizer


DEFAULT_QUESTION_CSV = Path(
    r"C:\Users\Calle\Projects\Nordic-AI-Cup-2026-official"
    r"\medical-appointment\data\question_train.csv"
)

DEFAULT_ASR_DIR = Path(
    r"medical\artifacts\asr\faster_distil_large_v3_int8f32_b1"
)

DEFAULT_OUTPUT = Path(
    r"medical\artifacts\classification\nli_baseline_results.csv"
)

RETRIEVER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
NLI_MODEL = "cross-encoder/nli-deberta-v3-small"

LEXICAL_TOP_K = 8
NLI_TOP_K = 5


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

    return [
        {
            "start": float(item["start"]),
            "end": float(item["end"]),
            "text": str(item["text"]).strip(),
        }
        for item in data["segments"]
        if str(item["text"]).strip()
    ]


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


def get_label_indices(model) -> dict[str, int]:
    mapping = {
        int(key): str(value).lower()
        for key, value in model.config.id2label.items()
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

    # Model card specifies:
    # 0 contradiction, 1 entailment, 2 neutral.
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

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 78)
    print("Nordic AI Cup 2026 — NLI Classification Baseline")
    print("=" * 78)
    print(f"Retriever: {RETRIEVER_MODEL}")
    print(f"NLI:       {NLI_MODEL}")
    print(f"Device:    {device}")
    print()

    print("Loading retrieval reranker...")

    retriever = CrossEncoder(
        RETRIEVER_MODEL,
        device=device,
        max_length=256,
    )

    print("Loading NLI model...")

    tokenizer = AutoTokenizer.from_pretrained(
        NLI_MODEL
    )

    nli_model = (
        AutoModelForSequenceClassification
        .from_pretrained(NLI_MODEL)
        .to(device)
        .eval()
    )

    label_indices = get_label_indices(
        nli_model
    )

    print(
        "NLI labels:",
        label_indices,
    )
    print()

    cache: dict[str, list[dict]] = {}
    results = []

    start_time = time.perf_counter()

    for row_number, (_, row) in enumerate(
        df.iterrows(),
        start=1,
    ):
        transcript_id = row["transcript_id"]
        stem = normalize_transcript_id(
            transcript_id
        )

        if stem not in cache:
            cache[stem] = load_segments(
                args.asr_dir,
                transcript_id,
            )

        segments = cache[stem]
        question = str(row["question"])

        # ---------------------------------------------
        # Stage 1: lexical candidate generation
        # ---------------------------------------------

        lex_scores = lexical_scores(
            question,
            segments,
        )

        lexical_ranking = np.argsort(
            lex_scores
        )[::-1]

        candidate_indices = lexical_ranking[
            :LEXICAL_TOP_K
        ]

        # ---------------------------------------------
        # Stage 2: semantic passage reranking
        # ---------------------------------------------

        pairs = [
            (
                question,
                segments[int(index)]["text"],
            )
            for index in candidate_indices
        ]

        passage_scores = np.asarray(
            retriever.predict(
                pairs,
                batch_size=32,
                show_progress_bar=False,
            )
        ).reshape(-1)

        semantic_order = np.argsort(
            passage_scores
        )[::-1]

        nli_segment_indices = [
            int(candidate_indices[index])
            for index in semantic_order[
                :NLI_TOP_K
            ]
        ]

        # ---------------------------------------------
        # Stage 3: NLI
        #
        # premise   = retrieved transcript segment
        # hypothesis = raw yes/no question
        #
        # Raw question-as-hypothesis is intentional
        # for this baseline and will be measured.
        # ---------------------------------------------

        premises = [
            segments[index]["text"]
            for index in nli_segment_indices
        ]

        hypotheses = [
            question
            for _ in premises
        ]

        encoded = tokenizer(
            premises,
            hypotheses,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )

        encoded = {
            key: value.to(device)
            for key, value in encoded.items()
        }

        with torch.inference_mode():
            logits = nli_model(
                **encoded
            ).logits

        probabilities = torch.softmax(
            logits,
            dim=-1,
        ).detach().cpu().numpy()

        entailment = probabilities[
            :,
            label_indices["entailment"],
        ]

        contradiction = probabilities[
            :,
            label_indices["contradiction"],
        ]

        neutral = probabilities[
            :,
            label_indices["neutral"],
        ]

        best_entailment_position = int(
            np.argmax(entailment)
        )

        max_entailment = float(
            entailment[
                best_entailment_position
            ]
        )

        max_contradiction = float(
            contradiction.max()
        )

        max_neutral = float(
            neutral.max()
        )

        # Untuned zero-shot baseline:
        # YES only if one retrieved passage has
        # entailment as its dominant NLI outcome.
        selected_probs = probabilities[
            best_entailment_position
        ]

        predicted_yes = (
            int(np.argmax(selected_probs))
            == label_indices["entailment"]
        )

        selected_segment_index = (
            nli_segment_indices[
                best_entailment_position
            ]
        )

        selected_segment = segments[
            selected_segment_index
        ]

        answer_value = str(row["answer"]).strip().lower()

        if answer_value in {"yes", "true", "1"}:
            gold_yes = True
        elif answer_value in {"no", "false", "0"}:
            gold_yes = False
        else:
            raise ValueError(
                f"Unexpected answer value: {row['answer']!r}"
            )
            
        

        results.append(
            {
                "question_id": row[
                    "question_id"
                ],
                "transcript_id": transcript_id,
                "question": question,
                "question_type": row[
                    "question_type"
                ],
                "gold_yes": gold_yes,
                "predicted_yes": (
                    predicted_yes
                ),
                "correct": (
                    predicted_yes
                    == gold_yes
                ),
                "max_entailment": (
                    max_entailment
                ),
                "max_contradiction": (
                    max_contradiction
                ),
                "max_neutral": (
                    max_neutral
                ),
                "selected_start": (
                    selected_segment["start"]
                ),
                "selected_end": (
                    selected_segment["end"]
                ),
                "selected_text": (
                    selected_segment["text"]
                ),
            }
        )

        if (
            row_number % 50 == 0
            or row_number == len(df)
        ):
            print(
                f"[{row_number:03d}/"
                f"{len(df):03d}] processed"
            )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    result_df = pd.DataFrame(
        results
    )
    
    gold_yes_count = int(
        result_df["gold_yes"].sum()
    )

    if gold_yes_count != 195:
        raise RuntimeError(
            f"Expected 195 gold YES labels, got {gold_yes_count}"
        )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_df.to_csv(
        args.output,
        index=False,
    )

    accuracy = result_df[
        "correct"
    ].mean()

    print()
    print("=" * 78)
    print("RESULT")
    print("=" * 78)

    print(
        f"Accuracy: "
        f"{accuracy:.4f}"
    )

    print()
    print("QUESTION TYPE ACCURACY")
    print("-" * 78)

    for question_type in (
        "positive",
        "hard_negative",
        "off_topic",
    ):
        subset = result_df[
            result_df["question_type"]
            == question_type
        ]

        print(
            f"{question_type:15s} "
            f"{subset['correct'].mean():.4f} "
            f"({subset['correct'].sum()}/"
            f"{len(subset)})"
        )

    print()
    print("PREDICTION DISTRIBUTION")
    print("-" * 78)

    print(
        "Predicted YES: "
        f"{result_df['predicted_yes'].mean():.4f}"
    )

    print(
        "Gold YES:      "
        f"{result_df['gold_yes'].mean():.4f}"
    )

    print()
    print("CONFUSION MATRIX")
    print("-" * 78)

    matrix = confusion_matrix(
        result_df["gold_yes"],
        result_df["predicted_yes"],
        labels=[False, True],
    )

    print(
        "rows=gold [NO, YES], "
        "cols=pred [NO, YES]"
    )
    print(matrix)

    print()
    print("LATENCY")
    print("-" * 78)

    print(
        f"390-question total: "
        f"{elapsed:.2f} s"
    )

    print(
        f"Mean/question: "
        f"{elapsed / len(df):.4f} s"
    )

    print(
        f"Estimated/10 questions: "
        f"{10 * elapsed / len(df):.4f} s"
    )

    print()
    print(
        f"Detailed results: "
        f"{args.output}"
    )


if __name__ == "__main__":
    main()