from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold


M054_PATH = Path(
    r"medical\artifacts\classification"
    r"\m054_runtime_signal_classifier_oof.csv"
)

M055_PATH = Path(
    r"medical\artifacts\classification"
    r"\m055_base_nli_runtime_signal_oof.csv"
)

OUTPUT = Path(
    r"medical\artifacts\classification"
    r"\m056_dual_nli_fusion_oof.csv"
)

N_SPLITS = 5
SEED = 42


SMALL_NLI = [
    "max_segment_ratio",
    "max_segment_entailment",
    "max_segment_margin",
    "max_segment_vs_other",
    "selected_entailment",
    "selected_contradiction",
    "selected_neutral",
]

BASE_NLI = [
    "max_segment_ratio_base",
    "max_segment_entailment_base",
    "max_segment_margin_base",
    "max_segment_vs_other_base",
    "selected_entailment_base",
    "selected_contradiction_base",
    "selected_neutral_base",
]

OTHER_FEATURES = [
    "question_position",
    "evidence_max_score",
    "evidence_mean_score",
    "evidence_std_score",
    "evidence_top_gap",

    # Existing model probabilities are also
    # legitimate runtime signals.
    "m054_probability",
    "m055_probability",

    # Distances from calibrated thresholds.
    "m054_signed_margin",
    "m055_signed_margin",

    # Cross-model agreement/disagreement.
    "probability_difference",
    "absolute_probability_difference",
]

FEATURES = (
    SMALL_NLI
    + BASE_NLI
    + OTHER_FEATURES
)


def parse_bool(value):
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

    raise ValueError(value)


def make_model():
    return HistGradientBoostingClassifier(
        learning_rate=0.04,
        max_iter=250,
        max_leaf_nodes=7,
        min_samples_leaf=20,
        l2_regularization=1.5,
        random_state=SEED,
    )


def competition_score(
    gold,
    pred,
    evidence,
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
    probabilities,
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
    best_threshold = 0.5
    best_composite = -1.0

    for threshold in (
        threshold_candidates(
            probabilities
        )
    ):
        prediction = (
            probabilities
            >= threshold
        )

        _, _, composite = (
            competition_score(
                gold,
                prediction,
                evidence,
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


def main():
    small = pd.read_csv(
        M054_PATH
    )

    base = pd.read_csv(
        M055_PATH
    )

    if len(small) != 390:
        raise RuntimeError(
            f"M054 rows={len(small)}"
        )

    if len(base) != 390:
        raise RuntimeError(
            f"M055 rows={len(base)}"
        )

    base_columns = [
        "question_id",

        "max_segment_ratio",
        "max_segment_entailment",
        "max_segment_margin",
        "max_segment_vs_other",

        "selected_entailment",
        "selected_contradiction",
        "selected_neutral",

        "m055_probability",
        "m055_predicted_yes",
        "m055_threshold",
    ]

    base = base[
        base_columns
    ].rename(
        columns={
            "max_segment_ratio":
                "max_segment_ratio_base",

            "max_segment_entailment":
                "max_segment_entailment_base",

            "max_segment_margin":
                "max_segment_margin_base",

            "max_segment_vs_other":
                "max_segment_vs_other_base",

            "selected_entailment":
                "selected_entailment_base",

            "selected_contradiction":
                "selected_contradiction_base",

            "selected_neutral":
                "selected_neutral_base",
        }
    )

    df = small.merge(
        base,
        on="question_id",
        validate="one_to_one",
    )

    if len(df) != 390:
        raise RuntimeError(
            f"Merged rows={len(df)}"
        )

    # Verify labels.
    df[
        "gold_bool"
    ] = [
        parse_bool(value)
        for value
        in df["gold_bool"]
    ]

    # Runtime-safe cross-model features.
    df[
        "m054_signed_margin"
    ] = (
        df[
            "m054_probability"
        ]
        - df[
            "m054_threshold"
        ]
    )

    df[
        "m055_signed_margin"
    ] = (
        df[
            "m055_probability"
        ]
        - df[
            "m055_threshold"
        ]
    )

    df[
        "probability_difference"
    ] = (
        df[
            "m055_probability"
        ]
        - df[
            "m054_probability"
        ]
    )

    df[
        "absolute_probability_difference"
    ] = (
        df[
            "probability_difference"
        ].abs()
    )

    missing = (
        set(FEATURES)
        - set(df.columns)
    )

    if missing:
        raise RuntimeError(
            "Missing features: "
            f"{sorted(missing)}"
        )

    forbidden = {
        "gold_bool",
        "question_type",
        "m054_correct",
        "m055_correct",
        "evidence_tiou",
    }

    leakage = (
        forbidden
        & set(FEATURES)
    )

    if leakage:
        raise RuntimeError(
            "Leakage features: "
            f"{sorted(leakage)}"
        )

    gold = (
        df[
            "gold_bool"
        ]
        .astype(bool)
        .to_numpy()
    )

    evidence = (
        df[
            "evidence_tiou"
        ]
        .fillna(0.0)
        .astype(float)
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

    thresholds = []

    print("=" * 78)
    print(
        "Nordic AI Cup 2026 — "
        "M056 Dual-NLI Runtime-Signal Fusion"
    )
    print("=" * 78)

    print(
        f"Features: {len(FEATURES)}"
    )

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

        train_prob = (
            model.predict_proba(
                X.iloc[
                    train_idx
                ]
            )[:, 1]
        )

        threshold, train_comp = (
            best_threshold(
                train_prob,
                gold[
                    train_idx
                ],
                evidence[
                    train_idx
                ],
            )
        )

        test_prob = (
            model.predict_proba(
                X.iloc[
                    test_idx
                ]
            )[:, 1]
        )

        test_pred = (
            test_prob
            >= threshold
        )

        oof_probability[
            test_idx
        ] = test_prob

        oof_prediction[
            test_idx
        ] = test_pred

        oof_threshold[
            test_idx
        ] = threshold

        thresholds.append(
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
            test_pred,
            evidence[
                test_idx
            ],
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
        evidence,
    )

    tp = int(
        (
            oof_prediction
            & gold
        ).sum()
    )

    fn = int(
        (
            (~oof_prediction)
            & gold
        ).sum()
    )

    tn = int(
        (
            (~oof_prediction)
            & (~gold)
        ).sum()
    )

    fp = int(
        (
            oof_prediction
            & (~gold)
        ).sum()
    )

    df[
        "m056_probability"
    ] = oof_probability

    df[
        "m056_predicted_yes"
    ] = oof_prediction

    df[
        "m056_threshold"
    ] = oof_threshold

    df[
        "m056_correct"
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
    print(
        "M056 OOF RESULT"
    )
    print("=" * 78)

    print(
        f"Accuracy:       "
        f"{accuracy:.4f}"
    )

    print(
        f"Scored tIoU:    "
        f"{tiou:.4f}"
    )

    print(
        f"Composite:      "
        f"{composite:.4f}"
    )

    print(
        f"TP / FN:        "
        f"{tp} / {fn}"
    )

    print(
        f"TN / FP:        "
        f"{tn} / {fp}"
    )

    print(
        "Thresholds:     "
        + ", ".join(
            f"{x:.4f}"
            for x
            in thresholds
        )
    )

    print(
        f"Mean threshold: "
        f"{np.mean(thresholds):.4f}"
    )

    print()
    print("REFERENCE")
    print("-" * 78)

    print(
        "M054 composite:      "
        "0.6484"
    )

    print(
        "M055 composite:      "
        "0.6534"
    )

    print(
        "M054/M055 oracle "
        "accuracy: 0.9641"
    )

    print()

    print(
        f"Saved: {OUTPUT}"
    )


if __name__ == "__main__":
    main()
    