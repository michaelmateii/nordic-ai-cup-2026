from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SOURCE = Path(
    r"medical\artifacts\classification"
    r"\m070_task_specific_nli_oof.csv"
)

OUTPUT = Path(
    r"medical\artifacts\classification"
    r"\m075_m055_m070_meta_fusion_oof.csv"
)

SEED = 42
N_SPLITS = 5


BASE_FEATURES = [
    "m055_probability",
    "m070_max",
    "m070_second",
    "m070_gap",
    "m070_top3_mean",
    "m070_mean",
]

OPTIONAL_FEATURES = [
    "question_position",
    "evidence_max_score",
    "evidence_mean_score",
    "evidence_std_score",
    "evidence_top_gap",
]


def parse_bool(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    return (
        str(value)
        .strip()
        .lower()
        in {"true", "1", "yes"}
    )


def competition_score(
    gold,
    pred,
    evidence,
):
    accuracy = float(
        np.mean(gold == pred)
    )

    positive = gold

    tiou = float(
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
        + 0.6 * tiou
    )

    return accuracy, tiou, composite


def threshold_candidates(values):
    unique = np.unique(values)

    if len(unique) == 1:
        return np.asarray(
            [
                unique[0] - 1e-9,
                unique[0] + 1e-9,
            ]
        )

    mids = (
        unique[:-1]
        + unique[1:]
    ) / 2.0

    return np.concatenate(
        [
            [unique[0] - 1e-9],
            mids,
            [unique[-1] + 1e-9],
        ]
    )


def best_threshold(
    probability,
    gold,
    evidence,
):
    best_t = 0.5
    best_score = -1.0

    for threshold in threshold_candidates(
        probability
    ):
        pred = (
            probability
            >= threshold
        )

        _, _, score = competition_score(
            gold,
            pred,
            evidence,
        )

        if score > best_score:
            best_score = score
            best_t = float(
                threshold
            )

    return best_t, best_score


def make_hgb():
    return HistGradientBoostingClassifier(
        learning_rate=0.04,
        max_iter=200,
        max_leaf_nodes=5,
        min_samples_leaf=12,
        l2_regularization=3.0,
        random_state=SEED,
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
    df,
    features,
    name,
):
    gold = (
        df["gold_bool"]
        .astype(bool)
        .to_numpy()
    )

    evidence = (
        df["evidence_tiou"]
        .fillna(0.0)
        .to_numpy(dtype=float)
    )

    groups = (
        df["transcript_id"]
        .astype(str)
        .to_numpy()
    )

    splitter = GroupKFold(
        n_splits=N_SPLITS
    )

    oof_prob = np.zeros(
        len(df),
        dtype=float,
    )

    oof_pred = np.zeros(
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
    print(name.upper())
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
        if name.startswith("hgb"):
            model = make_hgb()
        else:
            model = make_logistic()

        model.fit(
            df.iloc[
                train_idx
            ][features],
            gold[
                train_idx
            ],
        )

        train_probability = (
            model.predict_proba(
                df.iloc[
                    train_idx
                ][features]
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
                ][features]
            )[:, 1]
        )

        prediction = (
            test_probability
            >= threshold
        )

        oof_prob[
            test_idx
        ] = test_probability

        oof_pred[
            test_idx
        ] = prediction

        oof_threshold[
            test_idx
        ] = threshold

        thresholds.append(
            threshold
        )

        metrics = competition_score(
            gold[
                test_idx
            ],
            prediction,
            evidence[
                test_idx
            ],
        )

        print(
            f"Fold {fold}: "
            f"threshold={threshold:.4f}, "
            f"train={train_comp:.4f}, "
            f"acc={metrics[0]:.4f}, "
            f"tIoU={metrics[1]:.4f}, "
            f"comp={metrics[2]:.4f}"
        )

    metrics = competition_score(
        gold,
        oof_pred,
        evidence,
    )

    return {
        "name": name,
        "probability": oof_prob,
        "prediction": oof_pred,
        "threshold": oof_threshold,
        "thresholds": thresholds,
        "accuracy": metrics[0],
        "tiou": metrics[1],
        "composite": metrics[2],
    }


def main():
    df = pd.read_csv(
        SOURCE
    )

    df[
        "gold_bool"
    ] = [
        parse_bool(x)
        for x
        in df["gold_bool"]
    ]

    features = [
        column
        for column in (
            BASE_FEATURES
            + OPTIONAL_FEATURES
        )
        if column in df.columns
    ]

    missing_required = (
        set(BASE_FEATURES)
        - set(features)
    )

    if missing_required:
        raise RuntimeError(
            "Missing required columns: "
            f"{sorted(missing_required)}"
        )

    # Extra interaction signals.
    df[
        "m055_minus_m070"
    ] = (
        df["m055_probability"]
        - df["m070_max"]
    )

    df[
        "m055_times_m070"
    ] = (
        df["m055_probability"]
        * df["m070_max"]
    )

    df[
        "m070_peakiness"
    ] = (
        df["m070_max"]
        - df["m070_mean"]
    )

    df[
        "m070_top3_peakiness"
    ] = (
        df["m070_max"]
        - df["m070_top3_mean"]
    )

    features += [
        "m055_minus_m070",
        "m055_times_m070",
        "m070_peakiness",
        "m070_top3_peakiness",
    ]

    print("=" * 78)
    print(
        "Nordic AI Cup 2026 — "
        "M075 Learned M055/M070 Meta-Fusion"
    )
    print("=" * 78)

    print(
        f"Rows:     {len(df)}"
    )

    print(
        f"Features: {len(features)}"
    )

    print()
    print(
        "\n".join(features)
    )

    gold = (
        df["gold_bool"]
        .astype(bool)
        .to_numpy()
    )

    evidence = (
        df["evidence_tiou"]
        .fillna(0.0)
        .to_numpy(dtype=float)
    )

    # ------------------------------------------------------------
    # M074 reference.
    # ------------------------------------------------------------

    m074_score = (
        0.68
        * df["m055_probability"]
        .to_numpy()
        +
        0.32
        * df["m070_max"]
        .to_numpy()
    )

    m074_pred = (
        m074_score
        >= 0.3226589362393797
    )

    m074 = competition_score(
        gold,
        m074_pred,
        evidence,
    )

    print()
    print("REFERENCE M074")
    print("-" * 78)

    print(
        f"Accuracy:    {m074[0]:.4f}"
    )

    print(
        f"Scored tIoU: {m074[1]:.4f}"
    )

    print(
        f"Composite:   {m074[2]:.4f}"
    )

    results = [
        run_model(
            df,
            features,
            "hgb_full",
        ),
        run_model(
            df,
            features,
            "logistic_full",
        ),
    ]

    for result in results:
        print()
        print(
            result["name"].upper()
            + " OOF RESULT"
        )
        print("-" * 78)

        print(
            f"Accuracy:    "
            f"{result['accuracy']:.4f}"
        )

        print(
            f"Scored tIoU: "
            f"{result['tiou']:.4f}"
        )

        print(
            f"Composite:   "
            f"{result['composite']:.4f}"
        )

        print(
            "Thresholds:  "
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
        "m075_probability"
    ] = best[
        "probability"
    ]

    df[
        "m075_prediction"
    ] = best[
        "prediction"
    ]

    df[
        "m075_threshold"
    ] = best[
        "threshold"
    ]

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
    print("M075 BEST")
    print("=" * 78)

    print(
        f"Model:       "
        f"{best['name']}"
    )

    print(
        f"M074:        "
        f"{m074[2]:.4f}"
    )

    print(
        f"M075:        "
        f"{best['composite']:.4f}"
    )

    print(
        f"Gain:        "
        f"{best['composite'] - m074[2]:+.4f}"
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