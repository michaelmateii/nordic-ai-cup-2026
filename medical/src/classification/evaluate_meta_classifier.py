from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


NLI_PATH = Path(
    r"medical\artifacts\classification\nli_segmentwise_results.csv"
)

EVIDENCE_PATH = Path(
    r"medical\artifacts\retrieval\finetuned_hybrid_ranker_oof.csv"
)

OUTPUT = Path(
    r"medical\artifacts\classification\m042_meta_classifier_oof.csv"
)

N_SPLITS = 5

NUMERIC_FEATURES = [
    "max_segment_ratio",
    "max_segment_entailment",
    "max_segment_margin",
    "max_segment_vs_other",
    "selected_entailment",
    "selected_contradiction",
    "selected_neutral",
    "question_word_count",
    "question_char_count",
    "has_number",
    "has_unit",
    "has_negation",
    "starts_did",
    "starts_does",
    "starts_is",
    "starts_are",
    "starts_was",
    "starts_were",
    "starts_has",
    "starts_have",
    "starts_will",
    "starts_should",
]


NEGATION_TERMS = {
    "no",
    "not",
    "never",
    "none",
    "without",
    "unchanged",
    "discontinued",
    "stop",
    "stopped",
}

UNIT_PATTERN = re.compile(
    r"\b("
    r"mg|mcg|g|kg|ml|l|"
    r"mmol|mmol/l|mg/dl|"
    r"cm|mm|%"
    r")\b",
    flags=re.IGNORECASE,
)

NUMBER_PATTERN = re.compile(
    r"\b\d+(?:\.\d+)?\b"
)


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value

    text = str(value).strip().lower()

    if text in {
        "true",
        "1",
        "yes",
    }:
        return True

    if text in {
        "false",
        "0",
        "no",
    }:
        return False

    raise ValueError(
        f"Cannot parse boolean: {value!r}"
    )


def add_question_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    df = df.copy()

    questions = (
        df["question"]
        .fillna("")
        .astype(str)
    )

    lower = questions.str.lower()

    df["question_word_count"] = (
        questions.str.split().str.len()
    )

    df["question_char_count"] = (
        questions.str.len()
    )

    df["has_number"] = (
        questions.str.contains(
            NUMBER_PATTERN,
            regex=True,
        )
        .astype(int)
    )

    df["has_unit"] = (
        questions.str.contains(
            UNIT_PATTERN,
            regex=True,
        )
        .astype(int)
    )

    df["has_negation"] = lower.map(
        lambda text: int(
            any(
                re.search(
                    rf"\b{re.escape(term)}\b",
                    text,
                )
                for term in NEGATION_TERMS
            )
        )
    )

    for auxiliary in [
        "did",
        "does",
        "is",
        "are",
        "was",
        "were",
        "has",
        "have",
        "will",
        "should",
    ]:
        df[
            f"starts_{auxiliary}"
        ] = (
            lower.str.startswith(
                auxiliary + " "
            )
            .astype(int)
        )

    return df


def competition_score(
    *,
    gold: np.ndarray,
    predicted: np.ndarray,
    evidence_tiou: np.ndarray,
) -> tuple[
    float,
    float,
    float,
]:
    accuracy = float(
        np.mean(
            gold == predicted
        )
    )

    positive_mask = gold

    if not np.any(
        positive_mask
    ):
        mean_tiou = 0.0

    else:
        positive_evidence = np.where(
            predicted[
                positive_mask
            ],
            evidence_tiou[
                positive_mask
            ],
            0.0,
        )

        mean_tiou = float(
            positive_evidence.mean()
        )

    composite = (
        0.4 * accuracy
        + 0.6 * mean_tiou
    )

    return (
        accuracy,
        mean_tiou,
        composite,
    )


def find_best_threshold(
    *,
    probabilities: np.ndarray,
    gold: np.ndarray,
    evidence_tiou: np.ndarray,
) -> tuple[
    float,
    float,
]:
    unique = np.unique(
        probabilities
    )

    if len(unique) == 1:
        candidates = np.array(
            [
                unique[0] - 1e-9,
                unique[0] + 1e-9,
            ]
        )

    else:
        midpoints = (
            unique[:-1]
            + unique[1:]
        ) / 2.0

        candidates = np.concatenate(
            [
                np.array(
                    [
                        unique[0]
                        - 1e-9
                    ]
                ),
                midpoints,
                np.array(
                    [
                        unique[-1]
                        + 1e-9
                    ]
                ),
            ]
        )

    best_threshold = 0.5
    best_composite = -1.0

    for threshold in candidates:
        predicted = (
            probabilities
            >= threshold
        )

        _, _, composite = (
            competition_score(
                gold=gold,
                predicted=predicted,
                evidence_tiou=(
                    evidence_tiou
                ),
            )
        )

        if (
            composite
            > best_composite
        ):
            best_composite = (
                composite
            )

            best_threshold = float(
                threshold
            )

    return (
        best_threshold,
        best_composite,
    )


def make_logistic() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scale",
                StandardScaler(),
            ),
            (
                "model",
                LogisticRegression(
                    C=1.0,
                    max_iter=5000,
                    class_weight=None,
                    random_state=42,
                ),
            ),
        ]
    )


def make_hgb() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "model",
                HistGradientBoostingClassifier(
                    learning_rate=0.05,
                    max_iter=150,
                    max_leaf_nodes=7,
                    min_samples_leaf=20,
                    l2_regularization=1.0,
                    random_state=42,
                ),
            ),
        ]
    )


def run_model(
    *,
    name: str,
    factory,
    df: pd.DataFrame,
) -> pd.DataFrame:
    splitter = GroupKFold(
        n_splits=N_SPLITS
    )

    X = df[
        NUMERIC_FEATURES
    ]

    y = (
        df["gold_bool"]
        .to_numpy(
            dtype=bool
        )
    )

    groups = (
        df["transcript_id"]
        .astype(str)
        .to_numpy()
    )

    evidence = (
        df["evidence_tiou"]
        .to_numpy(
            dtype=float
        )
    )

    oof_probability = np.zeros(
        len(df),
        dtype=float,
    )

    oof_prediction = np.zeros(
        len(df),
        dtype=bool,
    )

    fold_threshold = np.zeros(
        len(df),
        dtype=float,
    )

    thresholds = []

    print()
    print("=" * 78)
    print(name)
    print("=" * 78)

    for fold, (
        train_index,
        test_index,
    ) in enumerate(
        splitter.split(
            X,
            y,
            groups,
        ),
        start=1,
    ):
        model = factory()

        model.fit(
            X.iloc[
                train_index
            ],
            y[
                train_index
            ],
        )

        train_probability = (
            model.predict_proba(
                X.iloc[
                    train_index
                ]
            )[:, 1]
        )

        threshold, train_composite = (
            find_best_threshold(
                probabilities=(
                    train_probability
                ),
                gold=y[
                    train_index
                ],
                evidence_tiou=(
                    evidence[
                        train_index
                    ]
                ),
            )
        )

        test_probability = (
            model.predict_proba(
                X.iloc[
                    test_index
                ]
            )[:, 1]
        )

        test_prediction = (
            test_probability
            >= threshold
        )

        oof_probability[
            test_index
        ] = test_probability

        oof_prediction[
            test_index
        ] = test_prediction

        fold_threshold[
            test_index
        ] = threshold

        thresholds.append(
            threshold
        )

        (
            fold_accuracy,
            fold_tiou,
            fold_composite,
        ) = competition_score(
            gold=y[
                test_index
            ],
            predicted=(
                test_prediction
            ),
            evidence_tiou=(
                evidence[
                    test_index
                ]
            ),
        )

        print(
            f"Fold {fold}: "
            f"threshold={threshold:.4f}, "
            f"train_comp="
            f"{train_composite:.4f}, "
            f"test_acc="
            f"{fold_accuracy:.4f}, "
            f"test_tIoU="
            f"{fold_tiou:.4f}, "
            f"test_comp="
            f"{fold_composite:.4f}"
        )

    (
        accuracy,
        mean_tiou,
        composite,
    ) = competition_score(
        gold=y,
        predicted=(
            oof_prediction
        ),
        evidence_tiou=evidence,
    )

    positive_mask = y

    negative_mask = ~y

    true_positive = int(
        np.sum(
            oof_prediction
            & positive_mask
        )
    )

    false_negative = int(
        np.sum(
            (~oof_prediction)
            & positive_mask
        )
    )

    true_negative = int(
        np.sum(
            (~oof_prediction)
            & negative_mask
        )
    )

    false_positive = int(
        np.sum(
            oof_prediction
            & negative_mask
        )
    )

    print()
    print("OOF RESULT")
    print("-" * 78)

    print(
        f"Accuracy:           "
        f"{accuracy:.4f}"
    )

    print(
        f"Mean scored tIoU:   "
        f"{mean_tiou:.4f}"
    )

    print(
        f"Composite:          "
        f"{composite:.4f}"
    )

    print(
        f"Predicted YES:      "
        f"{oof_prediction.mean():.4f}"
    )

    print(
        f"TP / FN:            "
        f"{true_positive} / "
        f"{false_negative}"
    )

    print(
        f"TN / FP:            "
        f"{true_negative} / "
        f"{false_positive}"
    )

    print(
        "Fold thresholds:    "
        + ", ".join(
            f"{value:.4f}"
            for value
            in thresholds
        )
    )

    print(
        f"Mean threshold:     "
        f"{np.mean(thresholds):.4f}"
    )

    result = df[
        [
            "question_id",
            "transcript_id",
            "question",
            "gold_bool",
            "evidence_tiou",
        ]
    ].copy()

    result["model"] = name

    result[
        "oof_probability"
    ] = oof_probability

    result[
        "oof_predicted_yes"
    ] = oof_prediction

    result[
        "fold_threshold"
    ] = fold_threshold

    result[
        "correct"
    ] = (
        result[
            "gold_bool"
        ]
        == result[
            "oof_predicted_yes"
        ]
    )

    result[
        "accuracy"
    ] = accuracy

    result[
        "mean_scored_tiou"
    ] = mean_tiou

    result[
        "composite"
    ] = composite

    return result


def main() -> None:
    df = pd.read_csv(
        NLI_PATH
    )

    if len(df) != 390:
        raise RuntimeError(
            f"Expected 390 rows; "
            f"got {len(df)}"
        )

    df = add_question_features(
        df
    )

    df["gold_bool"] = [
        parse_bool(value)
        for value
        in df["gold_yes"]
    ]

    # Gold-negative questions receive zero
    # evidence contribution by definition.
    df["evidence_tiou"] = 0.0

    evidence = pd.read_csv(
        EVIDENCE_PATH
    )

    evidence_by_id = (
        evidence.set_index(
            "question_id"
        )[
            "target_tiou"
        ]
        .astype(float)
    )

    positive_mask = (
        df["gold_bool"]
    )

    for index in df.index[
        positive_mask
    ]:
        question_id = (
            df.at[
                index,
                "question_id",
            ]
        )

        if (
            question_id
            not in evidence_by_id.index
        ):
            raise RuntimeError(
                "Missing M036 evidence "
                f"for {question_id}"
            )

        df.at[
            index,
            "evidence_tiou",
        ] = float(
            evidence_by_id.loc[
                question_id
            ]
        )

    # Explicit leakage guard.
    forbidden = {
        "question_type",
        "gold_yes",
        "gold_bool",
        "correct",
        "evidence_tiou",
    }

    overlap = (
        forbidden
        & set(
            NUMERIC_FEATURES
        )
    )

    if overlap:
        raise RuntimeError(
            "LEAKAGE: forbidden features "
            f"present: {sorted(overlap)}"
        )

    print("=" * 78)
    print(
        "Nordic AI Cup 2026 — "
        "M042 Composite-Aware Meta-Classifier"
    )
    print("=" * 78)

    print(
        f"Rows:     {len(df)}"
    )

    print(
        f"Features: "
        f"{len(NUMERIC_FEATURES)}"
    )

    print()
    print("FEATURES")
    print("-" * 78)

    for feature in (
        NUMERIC_FEATURES
    ):
        print(feature)

    logistic = run_model(
        name="logistic",
        factory=make_logistic,
        df=df,
    )

    hgb = run_model(
        name="hgb",
        factory=make_hgb,
        df=df,
    )

    result = pd.concat(
        [
            logistic,
            hgb,
        ],
        ignore_index=True,
    )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT,
        index=False,
    )

    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)

    summary = (
        result.groupby(
            "model"
        )[
            [
                "accuracy",
                "mean_scored_tiou",
                "composite",
            ]
        ]
        .first()
    )

    print(
        summary.to_string()
    )

    print()
    print("REFERENCE")
    print("-" * 78)

    print(
        "M018 accuracy:       0.8821"
    )

    print(
        "M037 scored tIoU:    0.4436"
    )

    print(
        "M037 composite:      0.6190"
    )

    print()
    print(
        f"Detailed results: "
        f"{OUTPUT}"
    )


if __name__ == "__main__":
    main()