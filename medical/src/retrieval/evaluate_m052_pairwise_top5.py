from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


INPUT = Path(
    r"medical\artifacts\retrieval\m051_top5_features.csv"
)

OUTPUT = Path(
    r"medical\artifacts\retrieval\m052_pairwise_top5_oof.csv"
)

N_SPLITS = 5

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
    return Pipeline(
        [
            (
                "scale",
                StandardScaler(),
            ),
            (
                "model",
                LogisticRegression(
                    C=1.0,
                    max_iter=5000,
                    random_state=42,
                ),
            ),
        ]
    )


def build_pairs(
    dataframe: pd.DataFrame,
):
    X_rows = []
    y_rows = []

    for _, group in dataframe.groupby(
        "question_id",
        sort=False,
    ):
        group = group.reset_index(
            drop=True
        )

        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                tiou_i = float(
                    group.loc[
                        i,
                        "target_tiou",
                    ]
                )

                tiou_j = float(
                    group.loc[
                        j,
                        "target_tiou",
                    ]
                )

                if np.isclose(
                    tiou_i,
                    tiou_j,
                    atol=1e-8,
                ):
                    continue

                a = (
                    group.loc[
                        i,
                        FEATURE_COLUMNS,
                    ]
                    .astype(float)
                    .to_numpy()
                )

                b = (
                    group.loc[
                        j,
                        FEATURE_COLUMNS,
                    ]
                    .astype(float)
                    .to_numpy()
                )

                difference = (
                    a - b
                )

                label = int(
                    tiou_i
                    > tiou_j
                )

                X_rows.append(
                    difference
                )

                y_rows.append(
                    label
                )

                # Add reversed pair so the
                # training distribution is symmetric.
                X_rows.append(
                    -difference
                )

                y_rows.append(
                    1 - label
                )

    return (
        np.asarray(
            X_rows,
            dtype=float,
        ),
        np.asarray(
            y_rows,
            dtype=int,
        ),
    )


def rank_question(
    model,
    group: pd.DataFrame,
) -> int:
    group = group.reset_index()

    wins = np.zeros(
        len(group),
        dtype=float,
    )

    confidence = np.zeros(
        len(group),
        dtype=float,
    )

    for i in range(len(group)):
        for j in range(
            i + 1,
            len(group),
        ):
            a = (
                group.loc[
                    i,
                    FEATURE_COLUMNS,
                ]
                .astype(float)
                .to_numpy()
            )

            b = (
                group.loc[
                    j,
                    FEATURE_COLUMNS,
                ]
                .astype(float)
                .to_numpy()
            )

            difference = (
                a - b
            ).reshape(
                1,
                -1,
            )

            probability = float(
                model.predict_proba(
                    difference
                )[0, 1]
            )

            if probability >= 0.5:
                wins[i] += 1.0
            else:
                wins[j] += 1.0

            confidence[i] += (
                probability
            )

            confidence[j] += (
                1.0
                - probability
            )

    # Primary ranking = pairwise wins.
    # Confidence breaks ties.
    order = np.lexsort(
        (
            -confidence,
            -wins,
        )
    )

    best_local_index = int(
        order[0]
    )

    return int(
        group.loc[
            best_local_index,
            "index",
        ]
    )


def summarize(
    name: str,
    selected: pd.DataFrame,
):
    scores = (
        selected[
            "target_tiou"
        ].astype(float)
    )

    print()
    print(name)
    print("-" * 78)

    print(
        f"Mean tIoU:       "
        f"{scores.mean():.4f}"
    )

    print(
        f"Median tIoU:     "
        f"{scores.median():.4f}"
    )

    print(
        f"Any overlap:     "
        f"{(scores > 0).mean():.4f}"
    )

    print(
        f"tIoU >= 0.25:    "
        f"{(scores >= 0.25).mean():.4f}"
    )

    print(
        f"tIoU >= 0.50:    "
        f"{(scores >= 0.50).mean():.4f}"
    )

    print(
        f"tIoU >= 0.75:    "
        f"{(scores >= 0.75).mean():.4f}"
    )

    return float(
        scores.mean()
    )


def main():
    df = pd.read_csv(
        INPUT
    )

    if (
        df["question_id"]
        .nunique()
        != 195
    ):
        raise RuntimeError(
            "Expected 195 questions."
        )

    question_table = (
        df[
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

    df[
        "m052_selected"
    ] = False

    print("=" * 78)
    print(
        "Nordic AI Cup 2026 — "
        "M052 Pairwise Top-5 Ranker"
    )
    print("=" * 78)

    for fold, (
        train_q,
        test_q,
    ) in enumerate(
        splitter.split(
            question_table,
            groups=groups,
        ),
        start=1,
    ):
        train_ids = set(
            question_table.iloc[
                train_q
            ][
                "question_id"
            ]
        )

        test_ids = set(
            question_table.iloc[
                test_q
            ][
                "question_id"
            ]
        )

        train = df[
            df[
                "question_id"
            ].isin(
                train_ids
            )
        ].copy()

        test = df[
            df[
                "question_id"
            ].isin(
                test_ids
            )
        ].copy()

        X_train, y_train = (
            build_pairs(
                train
            )
        )

        model = (
            make_model()
        )

        model.fit(
            X_train,
            y_train,
        )

        selected_indices = []

        for _, group in (
            test.groupby(
                "question_id",
                sort=False,
            )
        ):
            best_index = (
                rank_question(
                    model,
                    group,
                )
            )

            selected_indices.append(
                best_index
            )

        selected = (
            df.loc[
                selected_indices
            ]
        )

        baseline_indices = (
            test.groupby(
                "question_id"
            )[
                "first_stage_score"
            ]
            .idxmax()
        )

        baseline = (
            test.loc[
                baseline_indices
            ]
        )

        baseline_mean = float(
            baseline[
                "target_tiou"
            ].mean()
        )

        pairwise_mean = float(
            selected[
                "target_tiou"
            ].mean()
        )

        print(
            f"Fold {fold}: "
            f"questions="
            f"{len(test_ids)}, "
            f"pairs="
            f"{len(X_train)}, "
            f"M047="
            f"{baseline_mean:.4f}, "
            f"M052="
            f"{pairwise_mean:.4f}, "
            f"delta="
            f"{pairwise_mean - baseline_mean:+.4f}"
        )

        df.loc[
            selected_indices,
            "m052_selected",
        ] = True

    selected = df[
        df[
            "m052_selected"
        ]
    ].copy()

    mean_tiou = summarize(
        "M052 PAIRWISE TOP-5 OOF",
        selected,
    )

    oracle_indices = (
        df.groupby(
            "question_id"
        )[
            "target_tiou"
        ]
        .idxmax()
    )

    oracle = (
        df.loc[
            oracle_indices
        ]
    )

    summarize(
        "TOP-5 ORACLE",
        oracle,
    )

    selected.to_csv(
        OUTPUT,
        index=False,
    )

    print()
    print("=" * 78)
    print("M052 RESULT")
    print("=" * 78)

    print(
        "M047 top-1:       0.5158"
    )

    print(
        f"M052 pairwise:    "
        f"{mean_tiou:.4f}"
    )

    print(
        "Top-5 oracle:     0.6893"
    )

    print(
        f"Gain vs M047:     "
        f"{mean_tiou - 0.5158:+.4f}"
    )

    print()
    print(
        f"Detailed results: "
        f"{OUTPUT}"
    )


if __name__ == "__main__":
    main()