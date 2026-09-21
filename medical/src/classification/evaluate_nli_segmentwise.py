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
    r"medical\artifacts\classification\nli_segmentwise_results.csv"
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

def question_to_claim(question: str) -> str:
    """
    Convert common yes/no question forms into declarative NLI hypotheses.

    The conversion deliberately preserves auxiliaries such as did/does
    rather than attempting verb inflection.
    """

    text = question.strip()

    if text.endswith("?"):
        text = text[:-1].strip()

    # Handle introductory phrases:
    # "At any point, did the patient..."
    # "Alongside asthma, was..."
    # "According to the doctor, is..."
    prefix = ""

    if "," in text:
        first_part, remainder = text.split(",", 1)

        remainder = remainder.strip()

        auxiliary_starters = (
            "does ",
            "do ",
            "did ",
            "is ",
            "are ",
            "was ",
            "were ",
            "has ",
            "have ",
            "will ",
            "should ",
            "can ",
        )

        if remainder.lower().startswith(auxiliary_starters):
            prefix = first_part.strip() + ", "
            text = remainder

    words = text.split()

    if len(words) < 2:
        return question.rstrip("?") + "."

    auxiliary = words[0].lower()

    # Existential forms:
    # Is there X? -> There is X.
    # Are there X? -> There are X.
    if (
        auxiliary in {"is", "are", "was", "were"}
        and words[1].lower() == "there"
    ):
        remainder = " ".join(words[2:])

        claim = (
            f"There {words[0].lower()} "
            f"{remainder}"
        )

        return prefix + claim + "."

    # Standard subject-auxiliary inversion.
    #
    # We need to identify the subject boundary. The dataset is dominated
    # by noun phrases such as:
    #
    #   the patient
    #   the doctor
    #   the treatment
    #   blood pressure
    #   both medications
    #
    # For determinism, locate the first likely predicate boundary by
    # treating the first noun phrase conservatively.

    if auxiliary in {
        "does",
        "do",
        "did",
        "is",
        "are",
        "was",
        "were",
        "has",
        "have",
        "will",
        "should",
        "can",
    }:
        remainder = words[1:]

        # Most generated questions use "the <noun>" subjects.
        if (
            len(remainder) >= 2
            and remainder[0].lower() == "the"
        ):
            subject_length = 2

            # Common multi-token subjects.
            if len(remainder) >= 3:
                three_token_subjects = {
                    "the blood pressure",
                    "the heart rate",
                    "the treatment plan",
                    "the kidney function",
                    "the liver function",
                    "the diabetes medication",
                    "the current medication",
                    "the prescribed medication",
                }

                candidate = " ".join(
                    token.lower()
                    for token in remainder[:3]
                )

                if candidate in three_token_subjects:
                    subject_length = 3

            subject = " ".join(
                remainder[:subject_length]
            )

            predicate = " ".join(
                remainder[subject_length:]
            )

        # Determiners such as "both medications".
        elif (
            len(remainder) >= 2
            and remainder[0].lower()
            in {"both", "any", "all", "either"}
        ):
            subject = " ".join(
                remainder[:2]
            )

            predicate = " ".join(
                remainder[2:]
            )

        else:
            # Fallback: first token as subject.
            subject = remainder[0]

            predicate = " ".join(
                remainder[1:]
            )

        claim = (
            f"{subject} {words[0].lower()} "
            f"{predicate}"
        ).strip()

        # Declarative sentence should start with a capital.
        claim = claim[0].upper() + claim[1:]

        return prefix + claim + "."

    # Questions already written in declarative form, e.g.
    # "The patient was advised to..."
    claim = text

    if claim:
        claim = claim[0].upper() + claim[1:]

    return prefix + claim + "."

def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.questions)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 78)
    print(
        "Nordic AI Cup 2026 — "
        "Segmentwise Declarative NLI"
    )
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
        claim = question_to_claim(question)

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
        # premise    = retrieved transcript segment
        # hypothesis = declarative claim derived from the yes/no question
        # ---------------------------------------------

        premises = [
            segments[index]["text"]
            for index in nli_segment_indices
        ]

        hypotheses = [
            claim
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

        # -------------------------------------------------
        # Coherent segment-level scores.
        #
        # Unlike EXP-M014, entailment and contradiction
        # are combined only when they come from the SAME
        # retrieved transcript segment.
        # -------------------------------------------------

        segment_ratio = (
            entailment
            / (
                entailment
                + contradiction
                + 1e-9
            )
        )

        segment_entailment_margin = (
            entailment
            - contradiction
        )

        segment_entailment_vs_other = (
            entailment
            - np.maximum(
                contradiction,
                neutral,
            )
        )

        best_ratio_position = int(
            np.argmax(segment_ratio)
        )

        best_entailment_position = int(
            np.argmax(entailment)
        )

        best_margin_position = int(
            np.argmax(
                segment_entailment_margin
            )
        )
        
        margin_segment_index = (
            nli_segment_indices[
                best_margin_position
            ]
        )

        margin_segment = segments[
            margin_segment_index
        ]

        best_vs_other_position = int(
            np.argmax(
                segment_entailment_vs_other
            )
        )

        max_segment_ratio = float(
            segment_ratio[
                best_ratio_position
            ]
        )

        max_segment_entailment = float(
            entailment[
                best_entailment_position
            ]
        )

        max_segment_margin = float(
            segment_entailment_margin[
                best_margin_position
            ]
        )

        max_segment_vs_other = float(
            segment_entailment_vs_other[
                best_vs_other_position
            ]
        )

        # Keep diagnostic probabilities for the segment
        # chosen by the ratio score.
        selected_entailment = float(
            entailment[
                best_ratio_position
            ]
        )

        selected_contradiction = float(
            contradiction[
                best_ratio_position
            ]
        )

        selected_neutral = float(
            neutral[
                best_ratio_position
            ]
        )

        selected_segment_index = (
            nli_segment_indices[
                best_ratio_position
            ]
        )

        selected_segment = segments[
            selected_segment_index
        ]

        # Raw prediction is diagnostic only.
        # Proper thresholding happens conversation-disjoint
        # in the calibration experiment.
        predicted_yes = (
            max_segment_ratio >= 0.5
        )

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
                "hypothesis": claim,
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
                "max_segment_ratio": (
                    max_segment_ratio
                ),
                "max_segment_entailment": (
                    max_segment_entailment
                ),
                "max_segment_margin": (
                    max_segment_margin
                ),
                "max_segment_vs_other": (
                    max_segment_vs_other
                ),
                "selected_entailment": (
                    selected_entailment
                ),
                "selected_contradiction": (
                    selected_contradiction
                ),
                "selected_neutral": (
                    selected_neutral
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
                "margin_selected_start": (
                    margin_segment["start"]
                ),
                "margin_selected_end": (
                    margin_segment["end"]
                ),
                "margin_selected_text": (
                    margin_segment["text"]
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