from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline


SOURCE = Path(
    r"medical\artifacts\classification\nli_segmentwise_results.csv"
)

EVIDENCE = Path(
    r"medical\artifacts\retrieval\finetuned_hybrid_ranker_oof.csv"
)

OUTPUT = Path(
    r"medical\artifacts\classification\m044_nested_meta_classifier.csv"
)

OUTER_SPLITS = 5
INNER_SPLITS = 4


FEATURES = [
    "max_segment_ratio",
    "max_segment_entailment",
    "max_segment_margin",
    "max_segment_vs_other",
    "selected_entailment",
    "selected_contradiction",
    "selected_neutral",
]


def parse_bool(value):
    if isinstance(value, bool):
        return value

    text = str(value).strip().lower()

    if text in {"true", "1", "yes"}:
        return True

    if text in {"false", "0", "no"}:
        return False

    raise ValueError(value)


def make_model():
    return Pipeline(
        [
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


def competition_score(
    gold,
    pred,
    evidence,
):
    accuracy = float(
        np.mean(gold == pred)
    )

    mask = gold

    scored = np.where(
        pred[mask],
        evidence[mask],
        0.0,
    )

    tiou = float(
        np.mean(scored)
    )

    composite = (
        0.4 * accuracy
        + 0.6 * tiou
    )

    return (
        accuracy,
        tiou,
        composite,
    )


def best_threshold(
    probabilities,
    gold,
    evidence,
):
    values = np.unique(
        probabilities
    )

    mids = (
        values[:-1]
        + values[1:]
    ) / 2.0

    candidates = np.concatenate(
        [
            [values[0] - 1e-9],
            mids,
            [values[-1] + 1e-9],
        ]
    )

    best_t = 0.5
    best_score = -1.0

    for threshold in candidates:
        pred = (
            probabilities
            >= threshold
        )

        _, _, score = (
            competition_score(
                gold,
                pred,
                evidence,
            )
        )

        if score > best_score:
            best_score = score
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

    df["gold_bool"] = [
        parse_bool(value)
        for value
        in df["gold_yes"]
    ]

    evidence = pd.read_csv(
        EVIDENCE
    ).set_index(
        "question_id"
    )

    df["evidence_tiou"] = 0.0

    for index, row in df.iterrows():
        if not row["gold_bool"]:
            continue

        df.at[
            index,
            "evidence_tiou",
        ] = float(
            evidence.loc[
                row["question_id"],
                "target_tiou",
            ]
        )

    X = df[
        FEATURES
    ]

    y = df[
        "gold_bool"
    ].to_numpy(
        dtype=bool
    )

    groups = df[
        "transcript_id"
    ].astype(str).to_numpy()

    e = df[
        "evidence_tiou"
    ].to_numpy(
        dtype=float
    )

    outer = GroupKFold(
        n_splits=OUTER_SPLITS
    )

    final_prob = np.zeros(
        len(df)
    )

    final_pred = np.zeros(
        len(df),
        dtype=bool,
    )

    thresholds = np.zeros(
        len(df)
    )

    print("=" * 78)
    print(
        "Nordic AI Cup 2026 — "
        "M044 Nested Meta-Classifier"
    )
    print("=" * 78)

    for outer_fold, (
        train_idx,
        test_idx,
    ) in enumerate(
        outer.split(
            X,
            y,
            groups,
        ),
        start=1,
    ):
        inner_X = X.iloc[
            train_idx
        ]

        inner_y = y[
            train_idx
        ]

        inner_groups = groups[
            train_idx
        ]

        inner_e = e[
            train_idx
        ]

        inner = GroupKFold(
            n_splits=INNER_SPLITS
        )

        inner_oof = np.zeros(
            len(train_idx)
        )

        for (
            inner_train,
            inner_test,
        ) in inner.split(
            inner_X,
            inner_y,
            inner_groups,
        ):
            model = make_model()

            model.fit(
                inner_X.iloc[
                    inner_train
                ],
                inner_y[
                    inner_train
                ],
            )

            inner_oof[
                inner_test
            ] = model.predict_proba(
                inner_X.iloc[
                    inner_test
                ]
            )[:, 1]

        threshold, inner_comp = (
            best_threshold(
                inner_oof,
                inner_y,
                inner_e,
            )
        )

        model = make_model()

        model.fit(
            X.iloc[
                train_idx
            ],
            y[
                train_idx
            ],
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

        final_prob[
            test_idx
        ] = probability

        final_pred[
            test_idx
        ] = prediction

        thresholds[
            test_idx
        ] = threshold

        acc, tiou, comp = (
            competition_score(
                y[test_idx],
                prediction,
                e[test_idx],
            )
        )

        print(
            f"Fold {outer_fold}: "
            f"threshold={threshold:.4f}, "
            f"inner_comp={inner_comp:.4f}, "
            f"test_acc={acc:.4f}, "
            f"test_tIoU={tiou:.4f}, "
            f"test_comp={comp:.4f}"
        )

    accuracy, tiou, composite = (
        competition_score(
            y,
            final_pred,
            e,
        )
    )

    tp = int(
        np.sum(
            final_pred & y
        )
    )

    fn = int(
        np.sum(
            (~final_pred) & y
        )
    )

    tn = int(
        np.sum(
            (~final_pred) & (~y)
        )
    )

    fp = int(
        np.sum(
            final_pred & (~y)
        )
    )

    print()
    print("OOF RESULT")
    print("-" * 78)

    print(
        f"Accuracy:         {accuracy:.4f}"
    )

    print(
        f"Scored tIoU:      {tiou:.4f}"
    )

    print(
        f"Composite:        {composite:.4f}"
    )

    print(
        f"TP / FN:          {tp} / {fn}"
    )

    print(
        f"TN / FP:          {tn} / {fp}"
    )

    unique_thresholds = []

    for value in thresholds:
        if value not in unique_thresholds:
            unique_thresholds.append(
                value
            )

    print(
        "Thresholds:       "
        + ", ".join(
            f"{value:.4f}"
            for value
            in unique_thresholds
        )
    )

    print()
    print("REFERENCE")
    print("-" * 78)

    print(
        "M037 composite:    0.6190"
    )

    print(
        "M042 full:         0.6297"
    )

    print(
        "M043 NLI-only:     0.6212"
    )

    result = df[
        [
            "question_id",
            "transcript_id",
            "gold_bool",
            "evidence_tiou",
        ]
    ].copy()

    result[
        "oof_probability"
    ] = final_prob

    result[
        "oof_predicted_yes"
    ] = final_pred

    result[
        "threshold"
    ] = thresholds

    result.to_csv(
        OUTPUT,
        index=False,
    )

    print()
    print(
        f"Saved: {OUTPUT}"
    )


if __name__ == "__main__":
    main()