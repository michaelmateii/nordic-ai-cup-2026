from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold


QUESTIONS = Path(
    Path(__file__).resolve().parents[3].parent
    / "Nordic-AI-Cup-2026-official"
    / "medical-appointment"
    / "data"
    / "question_train.csv"
)

NLI = Path(
    r"medical\artifacts\classification"
    r"\m055_nli_base_segmentwise_results.csv"
)

CANDIDATES = Path(
    r"medical\artifacts\retrieval"
    r"\m047_all_questions_candidates_oof.csv"
)

SELECTED_EVIDENCE = Path(
    r"medical\artifacts\retrieval"
    r"\m047_all_questions_hybrid_oof.csv"
)

OUTPUT = Path(
    r"medical\artifacts\classification"
    r"\m055_base_nli_runtime_signal_oof.csv"
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
        f"Cannot parse bool: {value!r}"
    )


def make_model():
    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=200,
        max_leaf_nodes=7,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=SEED,
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
                    values[0] - 1e-9
                ]
            ),
            mids,
            np.asarray(
                [
                    values[-1] + 1e-9
                ]
            ),
        ]
    )


def best_threshold(
    probabilities: np.ndarray,
    gold: np.ndarray,
    evidence: np.ndarray,
):
    best_t = 0.5
    best_score = -1.0

    for threshold in threshold_candidates(
        probabilities
    ):
        pred = (
            probabilities
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

    return (
        best_t,
        best_score,
    )


def main() -> None:
    questions = pd.read_csv(
        QUESTIONS
    )

    if len(questions) != 390:
        raise RuntimeError(
            f"Expected 390 questions; "
            f"got {len(questions)}"
        )

    questions[
        "gold_bool"
    ] = [
        parse_bool(value)
        for value
        in questions["answer"]
    ]

    questions[
        "question_position"
    ] = (
        questions.groupby(
            "transcript_id",
            sort=False,
        )
        .cumcount()
        + 1
    )

    nli = pd.read_csv(
        NLI
    )

    if len(nli) != 390:
        raise RuntimeError(
            f"Expected 390 NLI rows; "
            f"got {len(nli)}"
        )

    candidates = pd.read_csv(
        CANDIDATES
    )

    candidate_summary = (
        candidates.groupby(
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

    ranked = (
        candidates[
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

    ranked[
        "rank"
    ] = (
        ranked.groupby(
            "question_id"
        )
        .cumcount()
        + 1
    )

    top2 = (
        ranked[
            ranked["rank"] <= 2
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
            validate="one_to_one",
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

    evidence = pd.read_csv(
        SELECTED_EVIDENCE
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

    df = (
        questions[
            [
                "question_id",
                "transcript_id",
                "question",
                "question_type",
                "gold_bool",
                "question_position",
            ]
        ]
        .merge(
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
        .merge(
            evidence,
            on="question_id",
            how="left",
            validate="one_to_one",
        )
    )

    if len(df) != 390:
        raise RuntimeError(
            f"Merged rows={len(df)}"
        )

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
        "M055 DeBERTa-v3-base Runtime-Signal Classifier"
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
        model = make_model()

        model.fit(
            df.iloc[
                train_idx
            ][FEATURES],
            gold[
                train_idx
            ],
        )

        train_prob = (
            model.predict_proba(
                df.iloc[
                    train_idx
                ][FEATURES]
            )[:, 1]
        )

        threshold, train_comp = (
            best_threshold(
                train_prob,
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

        test_prob = (
            model.predict_proba(
                df.iloc[
                    test_idx
                ][FEATURES]
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
        "m055_probability"
    ] = oof_probability

    df[
        "m055_predicted_yes"
    ] = oof_prediction

    df[
        "m055_threshold"
    ] = oof_threshold

    df[
        "m055_correct"
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
    print("M055 OOF RESULT")
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
            f"{value:.4f}"
            for value
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
        "M054 composite: 0.6484"
    )

    print(
        "M049 hidden:    0.6178"
    )

    print()
    print(
        f"Saved: {OUTPUT}"
    )


if __name__ == "__main__":
    main()
    