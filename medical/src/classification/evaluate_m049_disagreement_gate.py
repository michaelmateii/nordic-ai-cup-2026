from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


INPUT = Path(
    r"medical\artifacts\analysis\m048_all_disagreement_evidence.csv"
)

OUTPUT = Path(
    r"medical\artifacts\classification\m049_disagreement_gate_oof.csv"
)

N_SPLITS = 5


def competition_score(
    gold: np.ndarray,
    predicted: np.ndarray,
    evidence_tiou: np.ndarray,
) -> tuple[float, float, float]:
    accuracy = float(
        np.mean(
            gold == predicted
        )
    )

    positive = gold

    scored_tiou = float(
        np.mean(
            np.where(
                predicted[positive],
                evidence_tiou[positive],
                0.0,
            )
        )
    )

    composite = (
        0.4 * accuracy
        + 0.6 * scored_tiou
    )

    return (
        accuracy,
        scored_tiou,
        composite,
    )


def apply_gate(
    frame: pd.DataFrame,
    threshold: float,
) -> np.ndarray:
    """
    M018 is the conservative base classifier.

    M042 may override M018 only when:
      - the systems disagree, and
      - the final evidence ranker has sufficiently
        high confidence.

    Otherwise retain M018.
    """

    m018 = (
        frame["m018_pred"]
        .astype(bool)
        .to_numpy()
    )

    m042 = (
        frame["m042_pred"]
        .astype(bool)
        .to_numpy()
    )

    evidence_score = (
        frame[
            "evidence_ranker_score"
        ]
        .astype(float)
        .to_numpy()
    )

    disagreement = (
        m018 != m042
    )

    trust_m042 = (
        disagreement
        & (
            evidence_score
            >= threshold
        )
    )

    prediction = (
        m018.copy()
    )

    prediction[
        trust_m042
    ] = m042[
        trust_m042
    ]

    return prediction


def candidate_thresholds(
    scores: np.ndarray,
) -> np.ndarray:
    values = np.unique(
        scores
    )

    if len(values) == 0:
        return np.asarray(
            [0.5],
            dtype=float,
        )

    if len(values) == 1:
        return np.asarray(
            [
                values[0] - 1e-9,
                values[0] + 1e-9,
            ],
            dtype=float,
        )

    midpoints = (
        values[:-1]
        + values[1:]
    ) / 2.0

    return np.concatenate(
        [
            np.asarray(
                [
                    values[0]
                    - 1e-9
                ]
            ),
            midpoints,
            np.asarray(
                [
                    values[-1]
                    + 1e-9
                ]
            ),
        ]
    )


def find_best_threshold(
    train: pd.DataFrame,
) -> tuple[
    float,
    float,
]:
    disagreements = train[
        train["m018_pred"]
        != train["m042_pred"]
    ]

    candidates = (
        candidate_thresholds(
            disagreements[
                "evidence_ranker_score"
            ]
            .astype(float)
            .to_numpy()
        )
    )

    gold = (
        train["gold_bool"]
        .astype(bool)
        .to_numpy()
    )

    evidence = (
        train[
            "evidence_gold_tiou"
        ]
        .fillna(0.0)
        .astype(float)
        .to_numpy()
    )

    best_threshold = 0.5
    best_composite = -1.0

    for threshold in candidates:
        predicted = apply_gate(
            train,
            float(threshold),
        )

        _, _, composite = (
            competition_score(
                gold,
                predicted,
                evidence,
            )
        )

        if composite > best_composite:
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


def main() -> None:
    df = pd.read_csv(
        INPUT
    )

    if len(df) != 390:
        raise RuntimeError(
            f"Expected 390 rows, got {len(df)}"
        )

    required = {
        "question_id",
        "transcript_id",
        "gold_bool",
        "m018_pred",
        "m042_pred",
        "evidence_ranker_score",
        "evidence_gold_tiou",
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

    for column in [
        "gold_bool",
        "m018_pred",
        "m042_pred",
    ]:
        df[column] = (
            df[column]
            .astype(str)
            .str.lower()
            .map(
                {
                    "true": True,
                    "false": False,
                }
            )
        )

        if df[column].isna().any():
            raise RuntimeError(
                f"Could not parse {column}"
            )

    groups = (
        df["transcript_id"]
        .astype(str)
        .to_numpy()
    )

    splitter = GroupKFold(
        n_splits=N_SPLITS
    )

    oof_prediction = np.zeros(
        len(df),
        dtype=bool,
    )

    oof_threshold = np.zeros(
        len(df),
        dtype=float,
    )

    fold_thresholds = []

    print("=" * 78)
    print(
        "Nordic AI Cup 2026 — "
        "M049 Evidence-Gated M018/M042 Classifier"
    )
    print("=" * 78)

    for fold, (
        train_index,
        test_index,
    ) in enumerate(
        splitter.split(
            df,
            groups=groups,
        ),
        start=1,
    ):
        train = df.iloc[
            train_index
        ].copy()

        test = df.iloc[
            test_index
        ].copy()

        threshold, train_composite = (
            find_best_threshold(
                train
            )
        )

        prediction = apply_gate(
            test,
            threshold,
        )

        oof_prediction[
            test_index
        ] = prediction

        oof_threshold[
            test_index
        ] = threshold

        fold_thresholds.append(
            threshold
        )

        gold = (
            test["gold_bool"]
            .astype(bool)
            .to_numpy()
        )

        evidence = (
            test[
                "evidence_gold_tiou"
            ]
            .fillna(0.0)
            .astype(float)
            .to_numpy()
        )

        (
            accuracy,
            scored_tiou,
            composite,
        ) = competition_score(
            gold,
            prediction,
            evidence,
        )

        disagreements = int(
            (
                test["m018_pred"]
                != test["m042_pred"]
            ).sum()
        )

        used_m042 = int(
            np.sum(
                prediction
                != test[
                    "m018_pred"
                ]
                .astype(bool)
                .to_numpy()
            )
        )

        print(
            f"Fold {fold}: "
            f"threshold={threshold:.4f}, "
            f"train_comp={train_composite:.4f}, "
            f"test_disagree={disagreements}, "
            f"accepted_m042={used_m042}, "
            f"acc={accuracy:.4f}, "
            f"tIoU={scored_tiou:.4f}, "
            f"comp={composite:.4f}"
        )

    gold = (
        df["gold_bool"]
        .astype(bool)
        .to_numpy()
    )

    evidence = (
        df[
            "evidence_gold_tiou"
        ]
        .fillna(0.0)
        .astype(float)
        .to_numpy()
    )

    (
        accuracy,
        scored_tiou,
        composite,
    ) = competition_score(
        gold,
        oof_prediction,
        evidence,
    )

    tp = int(
        np.sum(
            oof_prediction
            & gold
        )
    )

    fn = int(
        np.sum(
            (~oof_prediction)
            & gold
        )
    )

    tn = int(
        np.sum(
            (~oof_prediction)
            & (~gold)
        )
    )

    fp = int(
        np.sum(
            oof_prediction
            & (~gold)
        )
    )

    m018 = (
        df["m018_pred"]
        .astype(bool)
        .to_numpy()
    )

    m042 = (
        df["m042_pred"]
        .astype(bool)
        .to_numpy()
    )

    disagreement = (
        m018 != m042
    )

    accepted_override = (
        disagreement
        & (
            oof_prediction
            == m042
        )
    )

    rejected_override = (
        disagreement
        & (
            oof_prediction
            == m018
        )
    )

    df[
        "m049_predicted_yes"
    ] = oof_prediction

    df[
        "m049_threshold"
    ] = oof_threshold

    df[
        "m049_accepted_m042"
    ] = accepted_override

    df[
        "m049_rejected_m042"
    ] = rejected_override

    df[
        "m049_correct"
    ] = (
        oof_prediction
        == gold
    )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT,
        index=False,
    )

    print()
    print("=" * 78)
    print("M049 OOF RESULT")
    print("=" * 78)

    print(
        f"Accuracy:           "
        f"{accuracy:.4f}"
    )

    print(
        f"Mean scored tIoU:   "
        f"{scored_tiou:.4f}"
    )

    print(
        f"Composite:          "
        f"{composite:.4f}"
    )

    print(
        f"TP / FN:            "
        f"{tp} / {fn}"
    )

    print(
        f"TN / FP:            "
        f"{tn} / {fp}"
    )

    print(
        f"Disagreements:      "
        f"{int(disagreement.sum())}"
    )

    print(
        f"Accepted M042:      "
        f"{int(accepted_override.sum())}"
    )

    print(
        f"Rejected M042:      "
        f"{int(rejected_override.sum())}"
    )

    print(
        "Fold thresholds:    "
        + ", ".join(
            f"{value:.4f}"
            for value
            in fold_thresholds
        )
    )

    print(
        f"Mean threshold:     "
        f"{np.mean(fold_thresholds):.4f}"
    )

    print()
    print("DISAGREEMENT OUTCOMES")
    print("-" * 78)

    disagreement_df = df[
        disagreement
    ].copy()

    columns = [
        "question_id",
        "question_type",
        "question",
        "gold_bool",
        "m018_pred",
        "m042_pred",
        "evidence_ranker_score",
        "m049_threshold",
        "m049_accepted_m042",
        "m049_predicted_yes",
        "m049_correct",
        "evidence_text",
    ]

    print(
        disagreement_df[
            columns
        ]
        .sort_values(
            "evidence_ranker_score",
            ascending=False,
        )
        .to_string(
            index=False
        )
    )

    print()
    print("REFERENCE")
    print("-" * 78)

    print(
        "M037 / M018 composite: 0.6190"
    )

    print(
        "M042 composite:        0.6297"
    )

    print(
        "M018 errors:           46"
    )

    print(
        "M042 errors:           49"
    )

    print(
        "M018/M042 oracle:      38 errors"
    )

    print()
    print(
        f"Saved: {OUTPUT}"
    )


if __name__ == "__main__":
    main()