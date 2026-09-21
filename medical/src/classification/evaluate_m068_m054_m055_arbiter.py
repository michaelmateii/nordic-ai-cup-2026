from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
)
from sklearn.linear_model import (
    LogisticRegression,
)
from sklearn.model_selection import (
    GroupKFold,
)
from sklearn.preprocessing import (
    StandardScaler,
)
from sklearn.pipeline import (
    Pipeline,
)


M054 = Path(
    r"medical\artifacts\classification"
    r"\m054_runtime_signal_classifier_oof.csv"
)

M055 = Path(
    r"medical\artifacts\classification"
    r"\m055_base_nli_runtime_signal_oof.csv"
)

OUTPUT = Path(
    r"medical\artifacts\classification"
    r"\m068_m054_m055_arbiter_oof.csv"
)

N_SPLITS = 5
SEED = 42


# ----------------------------------------------------------------------
# Features available at inference.
#
# 054_* = DeBERTa-v3-small NLI-derived signals.
# 055_* = DeBERTa-v3-base NLI-derived signals.
#
# Shared evidence statistics are included once.
# ----------------------------------------------------------------------

NLI_NAMES = [
    "max_segment_ratio",
    "max_segment_entailment",
    "max_segment_margin",
    "max_segment_vs_other",
    "selected_entailment",
    "selected_contradiction",
    "selected_neutral",
]

SHARED_NAMES = [
    "question_position",
    "evidence_max_score",
    "evidence_mean_score",
    "evidence_std_score",
    "evidence_top_gap",
]

FEATURES = [
    "m054_probability",
    "m055_probability",

    "m054_signed",
    "m055_signed",

    "m054_abs_margin",
    "m055_abs_margin",

    "probability_difference",
    "signed_difference",

    "models_agree",
]

FEATURES += [
    f"m054_{name}"
    for name in NLI_NAMES
]

FEATURES += [
    f"m055_{name}"
    for name in NLI_NAMES
]

FEATURES += SHARED_NAMES


def parse_bool(
    value: object,
) -> bool:
    if isinstance(
        value,
        (bool, np.bool_),
    ):
        return bool(value)

    return (
        str(value)
        .strip()
        .lower()
        in {
            "true",
            "1",
            "yes",
        }
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
            ],
            dtype=float,
        )

    mids = (
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
            mids,
            np.asarray(
                [
                    values[-1]
                    + 1e-9
                ]
            ),
        ]
    )


def best_threshold(
    probability: np.ndarray,
    gold: np.ndarray,
    evidence: np.ndarray,
):
    best_t = 0.5
    best_score = -1.0

    for threshold in (
        threshold_candidates(
            probability
        )
    ):
        pred = (
            probability
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


def build_table():
    a = pd.read_csv(
        M054
    )

    b = pd.read_csv(
        M055
    )

    # Keep only required M054 information.
    left_columns = [
        "question_id",
        "transcript_id",
        "question",
        "question_type",
        "gold_bool",
        "evidence_tiou",
        "question_position",

        "m054_probability",
        "m054_predicted_yes",
        "m054_threshold",
    ] + NLI_NAMES + [
        "evidence_max_score",
        "evidence_mean_score",
        "evidence_std_score",
        "evidence_top_gap",
    ]

    left = a[
        left_columns
    ].copy()

    # Rename M054 NLI features.
    left = left.rename(
        columns={
            name:
                f"m054_{name}"
            for name
            in NLI_NAMES
        }
    )

    right_columns = [
        "question_id",

        "m055_probability",
        "m055_predicted_yes",
        "m055_threshold",
    ] + NLI_NAMES

    right = b[
        right_columns
    ].copy()

    # Rename M055 NLI features.
    right = right.rename(
        columns={
            name:
                f"m055_{name}"
            for name
            in NLI_NAMES
        }
    )

    df = left.merge(
        right,
        on="question_id",
        validate="one_to_one",
    )

    if len(df) != 390:
        raise RuntimeError(
            f"Expected 390 rows, "
            f"got {len(df)}"
        )

    df[
        "gold_bool"
    ] = [
        parse_bool(x)
        for x
        in df["gold_bool"]
    ]

    for column in [
        "m054_predicted_yes",
        "m055_predicted_yes",
    ]:
        df[column] = [
            parse_bool(x)
            for x
            in df[column]
        ]

    # Signed distance from each fold's
    # calibrated threshold.
    df[
        "m054_signed"
    ] = (
        df["m054_probability"]
        - df["m054_threshold"]
    )

    df[
        "m055_signed"
    ] = (
        df["m055_probability"]
        - df["m055_threshold"]
    )

    df[
        "m054_abs_margin"
    ] = (
        df["m054_signed"]
        .abs()
    )

    df[
        "m055_abs_margin"
    ] = (
        df["m055_signed"]
        .abs()
    )

    df[
        "probability_difference"
    ] = (
        df["m055_probability"]
        - df["m054_probability"]
    )

    df[
        "signed_difference"
    ] = (
        df["m055_signed"]
        - df["m054_signed"]
    )

    df[
        "models_agree"
    ] = (
        df["m054_predicted_yes"]
        == df["m055_predicted_yes"]
    ).astype(float)

    return df


def make_hgb():
    return (
        HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=150,
            max_leaf_nodes=5,
            min_samples_leaf=12,
            l2_regularization=2.0,
            random_state=SEED,
        )
    )


def make_logistic():
    return Pipeline(
        [
            (
                "scale",
                StandardScaler(),
            ),
            (
                "model",
                LogisticRegression(
                    C=0.25,
                    max_iter=5000,
                    random_state=SEED,
                ),
            ),
        ]
    )


def run_model(
    df: pd.DataFrame,
    model_name: str,
):
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
        .to_numpy(
            dtype=float
        )
    )

    groups = (
        df[
            "transcript_id"
        ]
        .astype(str)
        .to_numpy()
    )

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

    print()
    print("=" * 78)
    print(
        f"M068 — {model_name.upper()}"
    )
    print("=" * 78)

    for fold, (
        train_idx,
        test_idx,
    ) in enumerate(
        splitter.split(
            df,
            gold,
            groups,
        ),
        start=1,
    ):
        if model_name == "hgb":
            model = make_hgb()
        elif model_name == "logistic":
            model = make_logistic()
        else:
            raise ValueError(
                model_name
            )

        model.fit(
            df.iloc[
                train_idx
            ][FEATURES],
            gold[
                train_idx
            ],
        )

        train_probability = (
            model.predict_proba(
                df.iloc[
                    train_idx
                ][FEATURES]
            )[:, 1]
        )

        threshold, train_comp = (
            best_threshold(
                train_probability,
                gold[
                    train_idx
                ],
                evidence[
                    train_idx
                ],
            )
        )

        test_probability = (
            model.predict_proba(
                df.iloc[
                    test_idx
                ][FEATURES]
            )[:, 1]
        )

        test_prediction = (
            test_probability
            >= threshold
        )

        oof_probability[
            test_idx
        ] = test_probability

        oof_prediction[
            test_idx
        ] = test_prediction

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
            test_prediction,
            evidence[
                test_idx
            ],
        )

        print(
            f"Fold {fold}: "
            f"threshold={threshold:.4f}, "
            f"train_comp={train_comp:.4f}, "
            f"acc={accuracy:.4f}, "
            f"tIoU={tiou:.4f}, "
            f"comp={composite:.4f}"
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

    return {
        "name":
            model_name,

        "probability":
            oof_probability,

        "prediction":
            oof_prediction,

        "threshold":
            oof_threshold,

        "thresholds":
            thresholds,

        "accuracy":
            accuracy,

        "tiou":
            tiou,

        "composite":
            composite,

        "tp":
            tp,

        "fn":
            fn,

        "tn":
            tn,

        "fp":
            fp,
    }


def main():
    df = build_table()

    print("=" * 78)
    print(
        "Nordic AI Cup 2026 — "
        "M068 M054/M055 Classification Fusion"
    )
    print("=" * 78)

    print(
        f"Rows:      {len(df)}"
    )

    print(
        f"Features:  {len(FEATURES)}"
    )

    disagree = (
        df[
            "m054_predicted_yes"
        ]
        != df[
            "m055_predicted_yes"
        ]
    )

    print(
        f"Disagreements: "
        f"{int(disagree.sum())}"
    )

    # --------------------------------------------------------------
    # References.
    # --------------------------------------------------------------

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
        .to_numpy(
            dtype=float
        )
    )

    p54 = (
        df[
            "m054_predicted_yes"
        ]
        .astype(bool)
        .to_numpy()
    )

    p55 = (
        df[
            "m055_predicted_yes"
        ]
        .astype(bool)
        .to_numpy()
    )

    ref54 = competition_score(
        gold,
        p54,
        evidence,
    )

    ref55 = competition_score(
        gold,
        p55,
        evidence,
    )

    oracle = np.where(
        p55 == gold,
        p55,
        p54,
    )

    # This is the true prediction oracle only
    # where either model is correct.
    both_wrong = (
        (p54 != gold)
        & (p55 != gold)
    )

    oracle = gold.copy()
    oracle[
        both_wrong
    ] = p55[
        both_wrong
    ]

    oracle_metrics = (
        competition_score(
            gold,
            oracle,
            evidence,
        )
    )

    print()
    print("REFERENCE")
    print("-" * 78)

    print(
        f"M054: acc={ref54[0]:.4f}, "
        f"tIoU={ref54[1]:.4f}, "
        f"comp={ref54[2]:.4f}"
    )

    print(
        f"M055: acc={ref55[0]:.4f}, "
        f"tIoU={ref55[1]:.4f}, "
        f"comp={ref55[2]:.4f}"
    )

    print(
        f"Two-model oracle: "
        f"acc={oracle_metrics[0]:.4f}, "
        f"tIoU={oracle_metrics[1]:.4f}, "
        f"comp={oracle_metrics[2]:.4f}"
    )

    results = [
        run_model(
            df,
            "hgb",
        ),
        run_model(
            df,
            "logistic",
        ),
    ]

    for result in results:
        print()
        print(
            f"{result['name'].upper()} "
            "OOF RESULT"
        )
        print("-" * 78)

        print(
            f"Accuracy:       "
            f"{result['accuracy']:.4f}"
        )

        print(
            f"Scored tIoU:    "
            f"{result['tiou']:.4f}"
        )

        print(
            f"Composite:      "
            f"{result['composite']:.4f}"
        )

        print(
            f"TP / FN:        "
            f"{result['tp']} / "
            f"{result['fn']}"
        )

        print(
            f"TN / FP:        "
            f"{result['tn']} / "
            f"{result['fp']}"
        )

        print(
            "Thresholds:     "
            + ", ".join(
                f"{x:.4f}"
                for x
                in result[
                    "thresholds"
                ]
            )
        )

    best = max(
        results,
        key=lambda x:
            x["composite"],
    )

    df[
        "m068_probability"
    ] = best[
        "probability"
    ]

    df[
        "m068_predicted_yes"
    ] = best[
        "prediction"
    ]

    df[
        "m068_threshold"
    ] = best[
        "threshold"
    ]

    df[
        "m068_correct"
    ] = (
        best[
            "prediction"
        ]
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
    print("M068 BEST RESULT")
    print("=" * 78)

    print(
        f"Model:       "
        f"{best['name']}"
    )

    print(
        f"M055 comp:   "
        f"{ref55[2]:.4f}"
    )

    print(
        f"M068 comp:   "
        f"{best['composite']:.4f}"
    )

    print(
        f"Gain:        "
        f"{best['composite'] - ref55[2]:+.4f}"
    )

    print(
        f"Accuracy:    "
        f"{best['accuracy']:.4f}"
    )

    print(
        f"Scored tIoU: "
        f"{best['tiou']:.4f}"
    )

    print()
    print(
        f"Saved: {OUTPUT}"
    )


if __name__ == "__main__":
    main()