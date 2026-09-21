from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


QUESTION_PATH = Path(
    Path(__file__).resolve().parents[3].parent
    / "Nordic-AI-Cup-2026-official"
    / "medical-appointment"
    / "data"
    / "question_train.csv"
)

NLI_PATH = Path(
    r"medical\artifacts\classification\nli_segmentwise_results.csv"
)

M047_CANDIDATES = Path(
    r"medical\artifacts\retrieval\m047_all_questions_candidates_oof.csv"
)

OUTPUT = Path(
    r"medical\artifacts\analysis\m053_challenge_structure.csv"
)

N_SPLITS = 5
SEED = 42


def parse_bool(value):
    if isinstance(value, bool):
        return value

    text = str(value).strip().lower()

    if text in {"true", "1", "yes"}:
        return True

    if text in {"false", "0", "no"}:
        return False

    raise ValueError(value)


def normalize_question(text: str) -> str:
    text = str(text).lower()

    text = re.sub(
        r"\d+(?:\.\d+)?",
        "<NUM>",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def first_word(text: str) -> str:
    match = re.match(
        r"\s*([A-Za-z]+)",
        str(text),
    )

    return (
        match.group(1).lower()
        if match
        else ""
    )


def evaluate_predictions(
    name,
    y,
    pred,
):
    acc = accuracy_score(
        y,
        pred,
    )

    cm = confusion_matrix(
        y,
        pred,
        labels=[False, True],
    )

    print()
    print(name)
    print("-" * 78)

    print(
        f"Accuracy: {acc:.4f}"
    )

    print(
        "Confusion matrix "
        "[[TN, FP], [FN, TP]]:"
    )

    print(cm)

    return acc


def group_cv_text(
    df,
    column,
):
    splitter = GroupKFold(
        n_splits=N_SPLITS
    )

    y = df[
        "gold_bool"
    ].to_numpy()

    groups = (
        df[
            "transcript_id"
        ]
        .astype(str)
        .to_numpy()
    )

    oof = np.zeros(
        len(df),
        dtype=bool,
    )

    for train_idx, test_idx in (
        splitter.split(
            df,
            y,
            groups,
        )
    ):
        model = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=(3, 5),
                        min_df=2,
                        sublinear_tf=True,
                        max_features=20000,
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        C=2.0,
                        max_iter=5000,
                        class_weight=None,
                        random_state=SEED,
                    ),
                ),
            ]
        )

        model.fit(
            df.iloc[
                train_idx
            ][column],
            y[
                train_idx
            ],
        )

        oof[
            test_idx
        ] = model.predict(
            df.iloc[
                test_idx
            ][column]
        )

    return oof


def random_cv_text(
    df,
    column,
):
    y = df[
        "gold_bool"
    ].to_numpy()

    splitter = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=SEED,
    )

    oof = np.zeros(
        len(df),
        dtype=bool,
    )

    for train_idx, test_idx in (
        splitter.split(
            df,
            y,
        )
    ):
        model = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=(3, 5),
                        min_df=2,
                        sublinear_tf=True,
                        max_features=20000,
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        C=2.0,
                        max_iter=5000,
                        random_state=SEED,
                    ),
                ),
            ]
        )

        model.fit(
            df.iloc[
                train_idx
            ][column],
            y[
                train_idx
            ],
        )

        oof[
            test_idx
        ] = model.predict(
            df.iloc[
                test_idx
            ][column]
        )

    return oof


def group_cv_numeric(
    df,
    columns,
):
    splitter = GroupKFold(
        n_splits=N_SPLITS
    )

    y = df[
        "gold_bool"
    ].to_numpy()

    groups = (
        df[
            "transcript_id"
        ]
        .astype(str)
        .to_numpy()
    )

    oof = np.zeros(
        len(df),
        dtype=bool,
    )

    for train_idx, test_idx in (
        splitter.split(
            df,
            y,
            groups,
        )
    ):
        model = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=200,
            max_leaf_nodes=7,
            min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=SEED,
        )

        model.fit(
            df.iloc[
                train_idx
            ][columns],
            y[
                train_idx
            ],
        )

        oof[
            test_idx
        ] = model.predict(
            df.iloc[
                test_idx
            ][columns]
        )

    return oof


def main():
    questions = pd.read_csv(
        QUESTION_PATH
    )

    print("=" * 78)
    print(
        "Nordic AI Cup 2026 — "
        "M053 Challenge Structure Audit"
    )
    print("=" * 78)

    print()
    print("RAW QUESTION COLUMNS")
    print("-" * 78)

    for column in questions.columns:
        print(column)

    if len(questions) != 390:
        raise RuntimeError(
            f"Expected 390 rows, got "
            f"{len(questions)}"
        )

    questions["gold_bool"] = [
        parse_bool(value)
        for value
        in questions["answer"]
        if False
    ] if False else [
        parse_bool(value)
        for value
        in (
            questions["gold_yes"]
            if "gold_yes" in questions.columns
            else questions["answer"]
        )
    ]

    # Preserve original within-conversation order.
    questions[
        "question_position"
    ] = (
        questions.groupby(
            "transcript_id",
            sort=False,
        ).cumcount()
        + 1
    )

    questions[
        "normalized_question"
    ] = (
        questions["question"]
        .astype(str)
        .map(
            normalize_question
        )
    )

    questions[
        "first_word"
    ] = (
        questions["question"]
        .astype(str)
        .map(
            first_word
        )
    )

    print()
    print("BASIC COUNTS")
    print("-" * 78)

    print(
        "Questions:",
        len(questions),
    )

    print(
        "Conversations:",
        questions[
            "transcript_id"
        ].nunique(),
    )

    print(
        "YES:",
        int(
            questions[
                "gold_bool"
            ].sum()
        ),
    )

    print(
        "NO:",
        int(
            (
                ~questions[
                    "gold_bool"
                ]
            ).sum()
        ),
    )

    if (
        "question_type"
        in questions.columns
    ):
        print()
        print(
            questions[
                "question_type"
            ]
            .value_counts()
            .to_string()
        )

    # ============================================================
    # Per-conversation schema
    # ============================================================

    print()
    print("=" * 78)
    print("PER-CONVERSATION STRUCTURE")
    print("=" * 78)

    counts = (
        questions.groupby(
            "transcript_id"
        )
        .size()
    )

    print()
    print(
        "Questions per conversation:"
    )

    print(
        counts.value_counts()
        .sort_index()
        .to_string()
    )

    yes_counts = (
        questions.groupby(
            "transcript_id"
        )[
            "gold_bool"
        ]
        .sum()
    )

    print()
    print(
        "YES count per conversation:"
    )

    print(
        yes_counts.value_counts()
        .sort_index()
        .to_string()
    )

    if (
        "question_type"
        in questions.columns
    ):
        composition = (
            questions.groupby(
                "transcript_id"
            )[
                "question_type"
            ]
            .apply(
                lambda x: tuple(
                    x.tolist()
                )
            )
        )

        print()
        print(
            "Distinct question-type "
            "orderings:"
        )

        print(
            composition.value_counts()
            .head(20)
            .to_string()
        )

    # ============================================================
    # Position leakage
    # ============================================================

    print()
    print("=" * 78)
    print("QUESTION POSITION")
    print("=" * 78)

    position_table = (
        questions.groupby(
            "question_position"
        )
        .agg(
            n=(
                "gold_bool",
                "size",
            ),
            yes_rate=(
                "gold_bool",
                "mean",
            ),
        )
    )

    print(
        position_table.to_string()
    )

    position_majority = (
        questions.groupby(
            "question_position"
        )[
            "gold_bool"
        ]
        .mean()
        >= 0.5
    )

    position_pred = (
        questions[
            "question_position"
        ]
        .map(
            position_majority
        )
        .to_numpy(
            dtype=bool
        )
    )

    evaluate_predictions(
        "POSITION-ONLY IN-SAMPLE DIAGNOSTIC",
        questions[
            "gold_bool"
        ].to_numpy(),
        position_pred,
    )

    # ============================================================
    # Auxiliary verb / wording
    # ============================================================

    print()
    print("=" * 78)
    print("QUESTION OPENING")
    print("=" * 78)

    first_word_table = (
        questions.groupby(
            "first_word"
        )
        .agg(
            n=(
                "gold_bool",
                "size",
            ),
            yes_rate=(
                "gold_bool",
                "mean",
            ),
        )
        .sort_values(
            "n",
            ascending=False,
        )
    )

    print(
        first_word_table
        .head(30)
        .to_string()
    )

    # ============================================================
    # Duplicate / template structure
    # ============================================================

    print()
    print("=" * 78)
    print("QUESTION DUPLICATION")
    print("=" * 78)

    exact_counts = (
        questions[
            "question"
        ]
        .astype(str)
        .value_counts()
    )

    normalized_counts = (
        questions[
            "normalized_question"
        ]
        .value_counts()
    )

    print(
        "Exact duplicate text groups:",
        int(
            (
                exact_counts > 1
            ).sum()
        ),
    )

    print(
        "Normalized duplicate groups:",
        int(
            (
                normalized_counts > 1
            ).sum()
        ),
    )

    duplicated = questions[
        questions[
            "normalized_question"
        ].duplicated(
            keep=False
        )
    ]

    if len(duplicated):
        mixed = (
            duplicated.groupby(
                "normalized_question"
            )[
                "gold_bool"
            ]
            .nunique()
        )

        print(
            "Normalized duplicate groups "
            "with mixed labels:",
            int(
                (mixed > 1).sum()
            ),
        )

        print()
        print(
            "Largest normalized templates:"
        )

        print(
            normalized_counts[
                normalized_counts > 1
            ]
            .head(30)
            .to_string()
        )

    # ============================================================
    # Question-only ML leakage diagnostic
    # ============================================================

    print()
    print("=" * 78)
    print("QUESTION-ONLY CLASSIFICATION")
    print("=" * 78)

    group_pred = group_cv_text(
        questions,
        "question",
    )

    evaluate_predictions(
        "CHAR TF-IDF + LOGREG "
        "(conversation-disjoint)",
        questions[
            "gold_bool"
        ].to_numpy(),
        group_pred,
    )

    random_pred = random_cv_text(
        questions,
        "question",
    )

    evaluate_predictions(
        "CHAR TF-IDF + LOGREG "
        "(random question CV)",
        questions[
            "gold_bool"
        ].to_numpy(),
        random_pred,
    )

    # ============================================================
    # Existing runtime-safe signals
    # ============================================================

    print()
    print("=" * 78)
    print(
        "EXISTING MODEL SIGNAL AUDIT"
    )
    print("=" * 78)

    nli = pd.read_csv(
        NLI_PATH
    )

    candidate = pd.read_csv(
        M047_CANDIDATES
    )

    candidate_summary = (
        candidate.groupby(
            "question_id"
        )
        .agg(
            evidence_max_score=(
                "oof_score",
                "max",
            ),
            evidence_mean_score=(
                "oof_score",
                "mean",
            ),
            evidence_std_score=(
                "oof_score",
                "std",
            ),
        )
        .reset_index()
    )

    sorted_scores = (
        candidate[
            [
                "question_id",
                "oof_score",
            ]
        ]
        .sort_values(
            [
                "question_id",
                "oof_score",
            ],
            ascending=[
                True,
                False,
            ],
        )
    )

    sorted_scores[
        "rank"
    ] = (
        sorted_scores.groupby(
            "question_id"
        )
        .cumcount()
        + 1
    )

    top2 = (
        sorted_scores[
            sorted_scores[
                "rank"
            ]
            <= 2
        ]
        .pivot(
            index="question_id",
            columns="rank",
            values="oof_score",
        )
        .reset_index()
    )

    top2.columns = [
        "question_id",
        "evidence_top1",
        "evidence_top2",
    ]

    candidate_summary = (
        candidate_summary.merge(
            top2,
            on="question_id",
            how="left",
        )
    )

    candidate_summary[
        "evidence_top_gap"
    ] = (
        candidate_summary[
            "evidence_top1"
        ]
        - candidate_summary[
            "evidence_top2"
        ]
    )

    merged = (
        questions.merge(
            nli[
                [
                    "question_id",
                    "max_segment_ratio",
                    "max_segment_entailment",
                    "max_segment_margin",
                    "max_segment_vs_other",
                    "selected_entailment",
                    "selected_contradiction",
                    "selected_neutral",
                ]
            ],
            on="question_id",
            validate="one_to_one",
        )
        .merge(
            candidate_summary,
            on="question_id",
            validate="one_to_one",
        )
    )

    numeric_features = [
        "question_position",
        "max_segment_ratio",
        "max_segment_entailment",
        "max_segment_margin",
        "max_segment_vs_other",
        "selected_entailment",
        "selected_contradiction",
        "selected_neutral",
        "evidence_max_score",
        "evidence_mean_score",
        "evidence_std_score",
        "evidence_top_gap",
    ]

    numeric_pred = group_cv_numeric(
        merged,
        numeric_features,
    )

    evaluate_predictions(
        "RUNTIME-SAFE NUMERIC SIGNALS "
        "(conversation-disjoint)",
        merged[
            "gold_bool"
        ].to_numpy(),
        numeric_pred,
    )

    # ============================================================
    # Combined text + numeric model
    # ============================================================

    print()
    print("=" * 78)
    print(
        "QUESTION TEXT + RUNTIME SIGNALS"
    )
    print("=" * 78)

    y = (
        merged[
            "gold_bool"
        ]
        .to_numpy()
    )

    groups = (
        merged[
            "transcript_id"
        ]
        .astype(str)
        .to_numpy()
    )

    splitter = GroupKFold(
        n_splits=N_SPLITS
    )

    combined_oof = np.zeros(
        len(merged),
        dtype=bool,
    )

    for train_idx, test_idx in (
        splitter.split(
            merged,
            y,
            groups,
        )
    ):
        transformer = (
            ColumnTransformer(
                [
                    (
                        "text",
                        TfidfVectorizer(
                            analyzer="char_wb",
                            ngram_range=(
                                3,
                                5,
                            ),
                            min_df=2,
                            max_features=20000,
                            sublinear_tf=True,
                        ),
                        "question",
                    ),
                    (
                        "numeric",
                        StandardScaler(),
                        numeric_features,
                    ),
                ]
            )
        )

        model = Pipeline(
            [
                (
                    "features",
                    transformer,
                ),
                (
                    "clf",
                    LogisticRegression(
                        C=1.0,
                        max_iter=5000,
                        random_state=SEED,
                    ),
                ),
            ]
        )

        model.fit(
            merged.iloc[
                train_idx
            ],
            y[
                train_idx
            ],
        )

        combined_oof[
            test_idx
        ] = model.predict(
            merged.iloc[
                test_idx
            ]
        )

    evaluate_predictions(
        "COMBINED QUESTION + SIGNALS "
        "(conversation-disjoint)",
        y,
        combined_oof,
    )

    merged[
        "m053_question_text_pred"
    ] = group_pred

    merged[
        "m053_numeric_pred"
    ] = numeric_pred

    merged[
        "m053_combined_pred"
    ] = combined_oof

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    merged.to_csv(
        OUTPUT,
        index=False,
    )

    print()
    print("=" * 78)
    print("M053 AUDIT COMPLETE")
    print("=" * 78)

    print(
        f"Saved: {OUTPUT}"
    )


if __name__ == "__main__":
    main()