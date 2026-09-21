from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from sentence_transformers import CrossEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GroupKFold
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


# ============================================================================
# Paths / configuration
# ============================================================================

INPUT = Path(
    r"medical\artifacts\retrieval"
    r"\m047_all_questions_candidates_oof.csv"
)

OUTPUT = Path(
    r"medical\artifacts\retrieval"
    r"\m051_top5_reranker_oof.csv"
)

TOPK_FEATURE_OUTPUT = Path(
    r"medical\artifacts\retrieval"
    r"\m051_top5_features.csv"
)

TOP_K = 5
N_SPLITS = 5

RETRIEVER_MODEL_ID = (
    "cross-encoder/ms-marco-MiniLM-L6-v2"
)

NLI_MODEL_ID = (
    "cross-encoder/nli-deberta-v3-small"
)

NLI_MAX_LENGTH = 256
NLI_BATCH_SIZE = 64

SEED = 42


# ============================================================================
# Question -> declarative claim
# Mirrors the runtime conversion used by the classifier.
# ============================================================================

def question_to_claim(
    question: str,
) -> str:
    text = question.strip()

    if text.endswith("?"):
        text = text[:-1].strip()

    prefix = ""

    if "," in text:
        first_part, remainder = (
            text.split(
                ",",
                1,
            )
        )

        remainder = (
            remainder.strip()
        )

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

        if remainder.lower().startswith(
            auxiliary_starters
        ):
            prefix = (
                first_part.strip()
                + ", "
            )

            text = remainder

    words = text.split()

    if len(words) < 2:
        return (
            question.rstrip("?")
            + "."
        )

    auxiliary = (
        words[0].lower()
    )

    if (
        auxiliary
        in {
            "is",
            "are",
            "was",
            "were",
        }
        and words[1].lower()
        == "there"
    ):
        remainder = " ".join(
            words[2:]
        )

        claim = (
            f"There "
            f"{words[0].lower()} "
            f"{remainder}"
        )

        return (
            prefix
            + claim
            + "."
        )

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
        remainder = (
            words[1:]
        )

        if (
            len(remainder) >= 2
            and remainder[0].lower()
            == "the"
        ):
            subject_length = 2

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
                    for token
                    in remainder[:3]
                )

                if (
                    candidate
                    in three_token_subjects
                ):
                    subject_length = 3

            subject = " ".join(
                remainder[
                    :subject_length
                ]
            )

            predicate = " ".join(
                remainder[
                    subject_length:
                ]
            )

        elif (
            len(remainder) >= 2
            and remainder[0].lower()
            in {
                "both",
                "any",
                "all",
                "either",
            }
        ):
            subject = " ".join(
                remainder[:2]
            )

            predicate = " ".join(
                remainder[2:]
            )

        else:
            subject = (
                remainder[0]
            )

            predicate = " ".join(
                remainder[1:]
            )

        claim = (
            f"{subject} "
            f"{words[0].lower()} "
            f"{predicate}"
        ).strip()

        if claim:
            claim = (
                claim[0].upper()
                + claim[1:]
            )

        return (
            prefix
            + claim
            + "."
        )

    claim = text

    if claim:
        claim = (
            claim[0].upper()
            + claim[1:]
        )

    return (
        prefix
        + claim
        + "."
    )


# ============================================================================
# Text features
# ============================================================================

def lexical_scores(
    question: str,
    texts: list[str],
) -> np.ndarray:
    if not texts:
        return np.zeros(
            0,
            dtype=float,
        )

    documents = (
        texts
        + [question]
    )

    word_vectorizer = (
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            sublinear_tf=True,
            token_pattern=(
                r"(?u)\b\w+\b"
            ),
        )
    )

    word_matrix = (
        word_vectorizer
        .fit_transform(
            documents
        )
    )

    word_scores = (
        cosine_similarity(
            word_matrix[-1],
            word_matrix[:-1],
        )[0]
    )

    char_vectorizer = (
        TfidfVectorizer(
            lowercase=True,
            analyzer="char_wb",
            ngram_range=(3, 5),
            sublinear_tf=True,
        )
    )

    char_matrix = (
        char_vectorizer
        .fit_transform(
            documents
        )
    )

    char_scores = (
        cosine_similarity(
            char_matrix[-1],
            char_matrix[:-1],
        )[0]
    )

    return (
        0.55 * word_scores
        + 0.45 * char_scores
    )


def token_overlap_features(
    question: str,
    candidate: str,
) -> dict[str, float]:
    q_tokens = set(
        re.findall(
            r"\b\w+\b",
            question.lower(),
        )
    )

    c_tokens = set(
        re.findall(
            r"\b\w+\b",
            candidate.lower(),
        )
    )

    if not q_tokens:
        q_coverage = 0.0
    else:
        q_coverage = (
            len(
                q_tokens
                & c_tokens
            )
            / len(
                q_tokens
            )
        )

    union = (
        q_tokens
        | c_tokens
    )

    if not union:
        jaccard = 0.0
    else:
        jaccard = (
            len(
                q_tokens
                & c_tokens
            )
            / len(union)
        )

    return {
        "question_token_coverage": (
            float(q_coverage)
        ),
        "token_jaccard": (
            float(jaccard)
        ),
    }


# ============================================================================
# NLI helpers
# ============================================================================

def get_label_indices(
    model,
) -> dict[str, int]:
    mapping = {
        int(key): str(value).lower()
        for key, value
        in model.config.id2label.items()
    }

    result = {}

    for index, label in (
        mapping.items()
    ):
        for target in (
            "contradiction",
            "entailment",
            "neutral",
        ):
            if target in label:
                result[target] = (
                    index
                )

    if len(result) != 3:
        return {
            "contradiction": 0,
            "entailment": 1,
            "neutral": 2,
        }

    return result


def score_nli(
    dataframe: pd.DataFrame,
    tokenizer,
    model,
    labels: dict[str, int],
    device: str,
) -> pd.DataFrame:
    df = dataframe.copy()

    premises = (
        df["candidate_text"]
        .astype(str)
        .tolist()
    )

    hypotheses = [
        question_to_claim(
            question
        )
        for question
        in df["question"]
        .astype(str)
        .tolist()
    ]

    entailment_values = []
    contradiction_values = []
    neutral_values = []

    for start in range(
        0,
        len(df),
        NLI_BATCH_SIZE,
    ):
        end = min(
            start
            + NLI_BATCH_SIZE,
            len(df),
        )

        encoded = tokenizer(
            premises[
                start:end
            ],
            hypotheses[
                start:end
            ],
            padding=True,
            truncation=True,
            max_length=(
                NLI_MAX_LENGTH
            ),
            return_tensors="pt",
        )

        encoded = {
            key: value.to(
                device
            )
            for key, value
            in encoded.items()
        }

        with torch.inference_mode():
            logits = (
                model(
                    **encoded
                ).logits
            )

            probabilities = (
                torch.softmax(
                    logits,
                    dim=-1,
                )
            )

        probabilities = (
            probabilities
            .detach()
            .cpu()
            .numpy()
        )

        entailment_values.extend(
            probabilities[
                :,
                labels[
                    "entailment"
                ],
            ].tolist()
        )

        contradiction_values.extend(
            probabilities[
                :,
                labels[
                    "contradiction"
                ],
            ].tolist()
        )

        neutral_values.extend(
            probabilities[
                :,
                labels[
                    "neutral"
                ],
            ].tolist()
        )

    df[
        "nli_entailment"
    ] = np.asarray(
        entailment_values,
        dtype=float,
    )

    df[
        "nli_contradiction"
    ] = np.asarray(
        contradiction_values,
        dtype=float,
    )

    df[
        "nli_neutral"
    ] = np.asarray(
        neutral_values,
        dtype=float,
    )

    df[
        "nli_margin"
    ] = (
        df["nli_entailment"]
        - df[
            "nli_contradiction"
        ]
    )

    df[
        "nli_ratio"
    ] = (
        df["nli_entailment"]
        / (
            df[
                "nli_entailment"
            ]
            + df[
                "nli_contradiction"
            ]
            + 1e-12
        )
    )

    return df


# ============================================================================
# Feature generation
# ============================================================================

CANDIDATE_KINDS = [
    "sentence_1",
    "sentence_2",
    "sentence_3",
    "word_6",
    "word_10",
]


def build_topk_features(
    dataframe: pd.DataFrame,
    retriever,
    nli_tokenizer,
    nli_model,
    nli_labels,
    device: str,
) -> pd.DataFrame:
    rows = []

    positive = dataframe[
        dataframe[
            "question_type"
        ]
        == "positive"
    ].copy()

    print(
        f"Positive questions: "
        f"{positive['question_id'].nunique()}"
    )

    for number, (
        question_id,
        group,
    ) in enumerate(
        positive.groupby(
            "question_id",
            sort=False,
        ),
        start=1,
    ):
        top = (
            group.nlargest(
                TOP_K,
                "oof_score",
            )
            .copy()
            .reset_index(
                drop=True
            )
        )

        top[
            "first_stage_rank"
        ] = np.arange(
            1,
            len(top) + 1,
            dtype=float,
        )

        top_score = float(
            top[
                "oof_score"
            ].iloc[0]
        )

        second_score = (
            float(
                top[
                    "oof_score"
                ].iloc[1]
            )
            if len(top) > 1
            else top_score
        )

        texts = (
            top[
                "candidate_text"
            ]
            .astype(str)
            .tolist()
        )

        question = str(
            top[
                "question"
            ].iloc[0]
        )

        lexical = (
            lexical_scores(
                question,
                texts,
            )
        )

        semantic_pairs = [
            (
                question,
                text,
            )
            for text in texts
        ]

        semantic = np.asarray(
            retriever.predict(
                semantic_pairs,
                batch_size=32,
                show_progress_bar=False,
            )
        ).reshape(-1)

        for index, row in (
            top.iterrows()
        ):
            candidate_text = str(
                row[
                    "candidate_text"
                ]
            )

            overlap = (
                token_overlap_features(
                    question,
                    candidate_text,
                )
            )

            start = float(
                row[
                    "candidate_start"
                ]
            )

            end = float(
                row[
                    "candidate_end"
                ]
            )

            score = float(
                row[
                    "oof_score"
                ]
            )

            feature_row = {
                "question_id": (
                    question_id
                ),
                "transcript_id": (
                    row[
                        "transcript_id"
                    ]
                ),
                "question": (
                    question
                ),
                "candidate_index": int(
                    row[
                        "candidate_index"
                    ]
                ),
                "candidate_start": (
                    start
                ),
                "candidate_end": (
                    end
                ),
                "candidate_kind": (
                    row[
                        "candidate_kind"
                    ]
                ),
                "candidate_text": (
                    candidate_text
                ),
                "sentence_count": float(
                    row[
                        "sentence_count"
                    ]
                ),
                "target_tiou": float(
                    row[
                        "target_tiou"
                    ]
                ),

                # First-stage ranker.
                "first_stage_score": (
                    score
                ),
                "first_stage_rank": float(
                    row[
                        "first_stage_rank"
                    ]
                ),
                "score_gap_from_top1": (
                    top_score
                    - score
                ),
                "top1_top2_gap": (
                    top_score
                    - second_score
                ),

                # Candidate geometry.
                "duration": (
                    end
                    - start
                ),
                "word_count": float(
                    len(
                        candidate_text.split()
                    )
                ),

                # Retrieval / lexical signals.
                "lexical_score": float(
                    lexical[index]
                ),
                "semantic_score": float(
                    semantic[index]
                ),

                **overlap,
            }

            for kind in (
                CANDIDATE_KINDS
            ):
                feature_row[
                    f"kind_{kind}"
                ] = float(
                    row[
                        "candidate_kind"
                    ]
                    == kind
                )

            rows.append(
                feature_row
            )

        if (
            number % 25 == 0
            or number
            == positive[
                "question_id"
            ].nunique()
        ):
            print(
                f"[{number:03d}/195] "
                "top-5 features built"
            )

    features = pd.DataFrame(
        rows
    )

    print()
    print(
        "Scoring top-5 candidates "
        "with evidence NLI..."
    )

    features = score_nli(
        features,
        nli_tokenizer,
        nli_model,
        nli_labels,
        device,
    )

    return features


# ============================================================================
# Second-stage HGB model
# ============================================================================

FEATURE_COLUMNS = [
    "first_stage_score",
    "first_stage_rank",
    "score_gap_from_top1",
    "top1_top2_gap",

    "duration",
    "word_count",
    "sentence_count",

    "lexical_score",
    "semantic_score",
    "question_token_coverage",
    "token_jaccard",

    "nli_entailment",
    "nli_contradiction",
    "nli_neutral",
    "nli_margin",
    "nli_ratio",

    "kind_sentence_1",
    "kind_sentence_2",
    "kind_sentence_3",
    "kind_word_6",
    "kind_word_10",
]


def make_model():
    return (
        HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.05,
            max_iter=150,
            max_leaf_nodes=7,
            min_samples_leaf=10,
            l2_regularization=1.0,
            random_state=SEED,
        )
    )


def summarize(
    name: str,
    dataframe: pd.DataFrame,
    score_column: str,
) -> dict[str, float]:
    selected_indices = (
        dataframe.groupby(
            "question_id"
        )[score_column]
        .idxmax()
    )

    selected = (
        dataframe.loc[
            selected_indices
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    scores = (
        selected[
            "target_tiou"
        ]
        .astype(float)
    )

    metrics = {
        "mean_tiou": float(
            scores.mean()
        ),
        "median_tiou": float(
            scores.median()
        ),
        "any_overlap": float(
            (scores > 0).mean()
        ),
        "tiou_ge_025": float(
            (scores >= 0.25).mean()
        ),
        "tiou_ge_050": float(
            (scores >= 0.50).mean()
        ),
        "tiou_ge_075": float(
            (scores >= 0.75).mean()
        ),
    }

    print()
    print(name)
    print("-" * 78)

    print(
        f"Mean tIoU:       "
        f"{metrics['mean_tiou']:.4f}"
    )

    print(
        f"Median tIoU:     "
        f"{metrics['median_tiou']:.4f}"
    )

    print(
        f"Any overlap:     "
        f"{metrics['any_overlap']:.4f}"
    )

    print(
        f"tIoU >= 0.25:    "
        f"{metrics['tiou_ge_025']:.4f}"
    )

    print(
        f"tIoU >= 0.50:    "
        f"{metrics['tiou_ge_050']:.4f}"
    )

    print(
        f"tIoU >= 0.75:    "
        f"{metrics['tiou_ge_075']:.4f}"
    )

    return metrics


# ============================================================================
# Main
# ============================================================================

def main():
    wall_start = (
        time.perf_counter()
    )

    if not INPUT.exists():
        raise FileNotFoundError(
            INPUT
        )

    df = pd.read_csv(
        INPUT
    )

    required = {
        "question_id",
        "transcript_id",
        "question",
        "question_type",
        "candidate_index",
        "candidate_start",
        "candidate_end",
        "candidate_kind",
        "candidate_text",
        "sentence_count",
        "target_tiou",
        "oof_score",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise RuntimeError(
            "Missing columns: "
            f"{sorted(missing)}"
        )

    positive_questions = (
        df[
            df[
                "question_type"
            ]
            == "positive"
        ][
            "question_id"
        ]
        .nunique()
    )

    if positive_questions != 195:
        raise RuntimeError(
            "Expected 195 positives; "
            f"got {positive_questions}"
        )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 78)
    print(
        "Nordic AI Cup 2026 — "
        "M051 Top-5 Evidence Reranker"
    )
    print("=" * 78)

    print(
        f"Input:      {INPUT}"
    )

    print(
        f"Top K:      {TOP_K}"
    )

    print(
        f"Device:     {device}"
    )

    print()
    print(
        "Loading MS MARCO "
        "semantic scorer..."
    )

    retriever = CrossEncoder(
        RETRIEVER_MODEL_ID,
        device=device,
        max_length=256,
    )

    print(
        "Loading evidence NLI..."
    )

    nli_tokenizer = (
        AutoTokenizer.from_pretrained(
            NLI_MODEL_ID
        )
    )

    nli_model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            NLI_MODEL_ID
        )
        .to(device)
        .eval()
    )

    nli_labels = (
        get_label_indices(
            nli_model
        )
    )

    features = (
        build_topk_features(
            df,
            retriever,
            nli_tokenizer,
            nli_model,
            nli_labels,
            device,
        )
    )

    TOPK_FEATURE_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    features.to_csv(
        TOPK_FEATURE_OUTPUT,
        index=False,
    )

    print()
    print(
        f"Feature table: "
        f"{TOPK_FEATURE_OUTPUT}"
    )

    print(
        f"Rows:          "
        f"{len(features)}"
    )

    print(
        f"Features:      "
        f"{len(FEATURE_COLUMNS)}"
    )

    # --------------------------------------------------------
    # Baseline top-1 from M047.
    # --------------------------------------------------------

    summarize(
        "M047 FIRST-STAGE TOP-1",
        features,
        "first_stage_score",
    )

    # --------------------------------------------------------
    # Top-5 oracle.
    # --------------------------------------------------------

    features[
        "oracle_score"
    ] = (
        features[
            "target_tiou"
        ]
    )

    summarize(
        "TOP-5 ORACLE",
        features,
        "oracle_score",
    )

    # --------------------------------------------------------
    # Conversation-disjoint OOF M051.
    # --------------------------------------------------------

    question_table = (
        features[
            [
                "question_id",
                "transcript_id",
            ]
        ]
        .drop_duplicates(
            "question_id"
        )
        .reset_index(
            drop=True
        )
    )

    splitter = GroupKFold(
        n_splits=N_SPLITS
    )

    groups = (
        question_table[
            "transcript_id"
        ]
        .astype(str)
        .to_numpy()
    )

    features[
        "m051_oof_score"
    ] = np.nan

    print()
    print("=" * 78)
    print(
        "M051 CONVERSATION-DISJOINT CV"
    )
    print("=" * 78)

    for fold, (
        train_q_index,
        test_q_index,
    ) in enumerate(
        splitter.split(
            question_table,
            groups=groups,
        ),
        start=1,
    ):
        train_ids = set(
            question_table.iloc[
                train_q_index
            ][
                "question_id"
            ]
        )

        test_ids = set(
            question_table.iloc[
                test_q_index
            ][
                "question_id"
            ]
        )

        train = features[
            features[
                "question_id"
            ].isin(
                train_ids
            )
        ].copy()

        test = features[
            features[
                "question_id"
            ].isin(
                test_ids
            )
        ].copy()

        model = make_model()

        model.fit(
            train[
                FEATURE_COLUMNS
            ],
            train[
                "target_tiou"
            ].astype(float),
        )

        predictions = (
            model.predict(
                test[
                    FEATURE_COLUMNS
                ]
            )
        )

        features.loc[
            test.index,
            "m051_oof_score",
        ] = predictions

        test = test.copy()

        test[
            "m051_score"
        ] = predictions

        selected_indices = (
            test.groupby(
                "question_id"
            )[
                "m051_score"
            ]
            .idxmax()
        )

        fold_selected = (
            test.loc[
                selected_indices
            ]
        )

        fold_mean = float(
            fold_selected[
                "target_tiou"
            ].mean()
        )

        baseline_indices = (
            test.groupby(
                "question_id"
            )[
                "first_stage_score"
            ]
            .idxmax()
        )

        baseline_mean = float(
            test.loc[
                baseline_indices,
                "target_tiou",
            ].mean()
        )

        print(
            f"Fold {fold}: "
            f"questions="
            f"{len(test_ids)}, "
            f"M047="
            f"{baseline_mean:.4f}, "
            f"M051="
            f"{fold_mean:.4f}, "
            f"delta="
            f"{fold_mean - baseline_mean:+.4f}"
        )

    if (
        features[
            "m051_oof_score"
        ]
        .isna()
        .any()
    ):
        raise RuntimeError(
            "Some candidates did not "
            "receive an OOF score."
        )

    metrics = summarize(
        "M051 TOP-5 HGB OOF",
        features,
        "m051_oof_score",
    )

    # --------------------------------------------------------
    # Save selected result rows.
    # --------------------------------------------------------

    selected_indices = (
        features.groupby(
            "question_id"
        )[
            "m051_oof_score"
        ]
        .idxmax()
    )

    result = (
        features.loc[
            selected_indices
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT,
        index=False,
    )

    elapsed = (
        time.perf_counter()
        - wall_start
    )

    print()
    print("=" * 78)
    print("M051 RESULT")
    print("=" * 78)

    print(
        f"M047 top-1:        "
        f"0.5158"
    )

    print(
        f"M051 top-5 HGB:    "
        f"{metrics['mean_tiou']:.4f}"
    )

    print(
        f"Top-5 oracle:      "
        f"0.6893"
    )

    print(
        f"Remaining gap:     "
        f"{0.6893 - metrics['mean_tiou']:+.4f}"
    )

    print(
        f"Wall time:         "
        f"{elapsed:.1f} s"
    )

    print()
    print(
        f"Detailed results: "
        f"{OUTPUT}"
    )


if __name__ == "__main__":
    main()