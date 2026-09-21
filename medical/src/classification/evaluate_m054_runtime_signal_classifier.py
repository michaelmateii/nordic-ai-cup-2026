from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
)
from sklearn.model_selection import (
    GroupKFold,
)


SOURCE = Path(
    r"medical\artifacts\analysis"
    r"\m053_challenge_structure.csv"
)

EVIDENCE = Path(
    r"medical\artifacts\retrieval"
    r"\m047_all_questions_hybrid_oof.csv"
)

OUTPUT = Path(
    r"medical\artifacts\classification"
    r"\m054_runtime_signal_classifier_oof.csv"
)

N_SPLITS = 5
SEED = 42


FEATURES = [
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


def make_model():
    return (
        HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=200,
            max_leaf_nodes=7,
            min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=SEED,
        )
    )


def competition_score(
    gold: np.ndarray,
    pred: np.ndarray,
    evidence: np.ndarray,
):
    accuracy = float(
        np.mean(
            gold == pred
        )
    )

    positive = gold

    scored_tiou = float(
        np.mean(
            np.where(
                pred[positive],
                evidence[positive],
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


def threshold_candidates(
    probabilities: np.ndarray,
):
    values = np.unique(
        probabilities
    )

    if len(values) == 1:
        return np.asarray(
            [
                values[0] - 1e-9,
                values[0] + 1e-9,
            ]
        )

    mids = (
        values[:-1]
        + values[1:]
    ) / 2.0

    return np.concatenate(
        [
            [
                values[0]
                - 1e-9
            ],
            mids,
            [
                values[-1]
                + 1e-9
            ],
        ]
    )


def best_threshold(
    probabilities,
    gold,
    evidence,
):
    best_t = 0.5
    best_score = -1.0

    for threshold in (
        threshold_candidates(
            probabilities
        )
    ):
        pred = (
            probabilities
            >= threshold
        )

        _, _, composite = (
            competition_score(
                gold,
                pred,
                evidence,
            )
        )

        if (
            composite
            > best_score
        ):
            best_score = (
                composite
            )

            best_t = float(
                threshold
            )

    return (
        best_t,
        best_score,
    )


def main():
    df = pd.read_csv(
        SOURCE
    )

    evidence = pd.read_csv(
        EVIDENCE
    )[
        [
            "question_id",
            "target_tiou",
        ]
    ].rename(
        columns={
            "target_tiou":
                "evidence_tiou"
        }
    )

    df = df.merge(
        evidence,
        on="question_id",
        how="left",
        validate="one_to_one",
    )

    if len(df) != 390:
        raise RuntimeError(
            f"Expected 390 rows, "
            f"got {len(df)}"
        )

    # For negative questions the evidence
    # tIoU is not scored.
    df[
        "evidence_tiou"
    ] = (
        df[
            "evidence_tiou"
        ]
        .fillna(0.0)
    )

    gold = (
        df[
            "gold_bool"
        ]
        .astype(bool)
        .to_numpy()
    )

    groups = (
        df[
            "transcript_id"
        ]
        .astype(str)
        .to_numpy()
    )

    X = df[
        FEATURES
    ]

    splitter = GroupKFold(
        n_splits=N_SPLITS
    )

    oof_probability = np.zeros(
        len(df),
        dtype=float,
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
        "M054 Runtime-Signal Classifier"
    )
    print("=" * 78)

    for fold, (
        train_idx,
        test_idx,
    ) in enumerate(
        splitter.split(
            X,
            gold,
            groups,
        ),
        start=1,
    ):
        model = make_model()

        model.fit(
            X.iloc[
                train_idx
            ],
            gold[
                train_idx
            ],
        )

        # Same calibration convention used
        # in earlier meta-classifier work.
        train_probability = (
            model.predict_proba(
                X.iloc[
                    train_idx
                ]
            )[:, 1]
        )

        threshold, train_comp = (
            best_threshold(
                train_probability,
                gold[
                    train_idx
                ],
                df.iloc[
                    train_idx
                ][
                    "evidence_tiou"
                ]
                .to_numpy(
                    dtype=float
                ),
            )
        )

        probability = (
            model.predict_proba(
                X.iloc[
                    test_idx
                ]
            )[:, 1]
        )

        prediction = (
            probability
            >= threshold
        )

        oof_probability[
            test_idx
        ] = probability

        oof_prediction[
            test_idx
        ] = prediction

        oof_threshold[
            test_idx
        ] = threshold

        fold_thresholds.append(
            threshold
        )

        (
            accuracy,
            tiou,
            composite,
        ) = competition_score(
            gold[
                test_idx
            ],
            prediction,
            df.iloc[
                test_idx
            ][
                "evidence_tiou"
            ]
            .to_numpy(
                dtype=float
            ),
        )

        print(
            f"Fold {fold}: "
            f"threshold="
            f"{threshold:.4f}, "
            f"train_comp="
            f"{train_comp:.4f}, "
            f"acc="
            f"{accuracy:.4f}, "
            f"tIoU="
            f"{tiou:.4f}, "
            f"comp="
            f"{composite:.4f}"
        )

    (
        accuracy,
        tiou,
        composite,
    ) = competition_score(
        gold,
        oof_prediction,
        df[
            "evidence_tiou"
        ].to_numpy(
            dtype=float
        ),
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

    df[
        "m054_probability"
    ] = oof_probability

    df[
        "m054_predicted_yes"
    ] = oof_prediction

    df[
        "m054_threshold"
    ] = oof_threshold

    df[
        "m054_correct"
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
    print("M054 OOF RESULT")
    print("=" * 78)

    print(
        f"Accuracy:         "
        f"{accuracy:.4f}"
    )

    print(
        f"Scored tIoU:      "
        f"{tiou:.4f}"
    )

    print(
        f"Composite:        "
        f"{composite:.4f}"
    )

    print(
        f"TP / FN:          "
        f"{tp} / {fn}"
    )

    print(
        f"TN / FP:          "
        f"{tn} / {fp}"
    )

    print(
        "Fold thresholds:  "
        + ", ".join(
            f"{x:.4f}"
            for x
            in fold_thresholds
        )
    )

    print(
        f"Mean threshold:   "
        f"{np.mean(fold_thresholds):.4f}"
    )

    print()
    print("REFERENCE")
    print("-" * 78)

    print(
        "M053 default numeric "
        "accuracy: 0.9026"
    )

    print(
        "M049 OOF composite:      "
        "0.6354"
    )

    print(
        "M049 hidden validation:  "
        "0.6178"
    )

    print()
    print(
        f"Saved: {OUTPUT}"
    )


if __name__ == "__main__":
    main()