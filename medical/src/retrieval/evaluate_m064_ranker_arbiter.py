from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


INPUT = Path(
    r"medical\artifacts\retrieval"
    r"\m063_all_candidate_oof_scores.csv"
)

OUTPUT = Path(
    r"medical\artifacts\retrieval"
    r"\m064_ranker_arbiter_oof.csv"
)

N_SPLITS = 5
SEED = 42


NUMERIC_FEATURES = [
    "rank63_in_47",

    "m063_margin",
    "m047_top_gap",
    "m047_gap_to_m063",

    "score47",
    "score63_in_47",

    "m047_duration",
    "m063_duration",
    "duration_difference",
    "duration_ratio",

    "m047_word_count",
    "m063_word_count",
    "word_count_difference",

    "start_difference",
    "end_difference",
    "center_difference",

    "text_token_jaccard",
    "m063_is_earlier",
]

CATEGORICAL_FEATURES = [
    "kind47",
    "kind63",
    "kind_transition",
]


def token_set(text: str) -> set[str]:
    return set(
        str(text)
        .lower()
        .split()
    )


def build_question_table(
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    df = candidates.copy()

    # ----------------------------------------------------------
    # M047 ranking
    # ----------------------------------------------------------

    df = df.sort_values(
        [
            "question_id",
            "oof_score",
        ],
        ascending=[
            True,
            False,
        ],
    )

    df[
        "m047_rank"
    ] = (
        df.groupby(
            "question_id"
        )
        .cumcount()
        + 1
    )

    # ----------------------------------------------------------
    # M063 ranking
    # ----------------------------------------------------------

    df = df.sort_values(
        [
            "question_id",
            "m063_oof_score",
        ],
        ascending=[
            True,
            False,
        ],
    )

    df[
        "m063_rank"
    ] = (
        df.groupby(
            "question_id"
        )
        .cumcount()
        + 1
    )

    top47 = (
        df[
            df["m047_rank"]
            == 1
        ][
            [
                "question_id",
                "transcript_id",

                "candidate_index",
                "candidate_start",
                "candidate_end",
                "candidate_kind",
                "candidate_text",

                "target_tiou",
                "oof_score",
            ]
        ]
        .rename(
            columns={
                "candidate_index":
                    "m047_index",

                "candidate_start":
                    "m047_start",

                "candidate_end":
                    "m047_end",

                "candidate_kind":
                    "kind47",

                "candidate_text":
                    "m047_text",

                "target_tiou":
                    "m047_tiou",

                "oof_score":
                    "score47",
            }
        )
    )

    second47 = (
        df[
            df["m047_rank"]
            == 2
        ][
            [
                "question_id",
                "oof_score",
            ]
        ]
        .rename(
            columns={
                "oof_score":
                    "m047_second",
            }
        )
    )

    top63 = (
        df[
            df["m063_rank"]
            == 1
        ][
            [
                "question_id",

                "candidate_index",
                "candidate_start",
                "candidate_end",
                "candidate_kind",
                "candidate_text",

                "target_tiou",
                "m063_oof_score",

                "m047_rank",
                "oof_score",
            ]
        ]
        .rename(
            columns={
                "candidate_index":
                    "m063_index",

                "candidate_start":
                    "m063_start",

                "candidate_end":
                    "m063_end",

                "candidate_kind":
                    "kind63",

                "candidate_text":
                    "m063_text",

                "target_tiou":
                    "m063_tiou",

                "m063_oof_score":
                    "score63",

                "m047_rank":
                    "rank63_in_47",

                "oof_score":
                    "score63_in_47",
            }
        )
    )

    second63 = (
        df[
            df["m063_rank"]
            == 2
        ][
            [
                "question_id",
                "m063_oof_score",
            ]
        ]
        .rename(
            columns={
                "m063_oof_score":
                    "m063_second",
            }
        )
    )

    q = (
        top47
        .merge(
            second47,
            on="question_id",
            validate="one_to_one",
        )
        .merge(
            top63,
            on="question_id",
            validate="one_to_one",
        )
        .merge(
            second63,
            on="question_id",
            validate="one_to_one",
        )
    )

    # ----------------------------------------------------------
    # Derived inference-safe features
    # ----------------------------------------------------------

    q[
        "changed"
    ] = (
        q["m047_index"]
        != q["m063_index"]
    )

    q[
        "m063_margin"
    ] = (
        q["score63"]
        - q["m063_second"]
    )

    q[
        "m047_top_gap"
    ] = (
        q["score47"]
        - q["m047_second"]
    )

    q[
        "m047_gap_to_m063"
    ] = (
        q["score47"]
        - q["score63_in_47"]
    )

    q[
        "m047_duration"
    ] = (
        q["m047_end"]
        - q["m047_start"]
    )

    q[
        "m063_duration"
    ] = (
        q["m063_end"]
        - q["m063_start"]
    )

    q[
        "duration_difference"
    ] = (
        q["m063_duration"]
        - q["m047_duration"]
    )

    q[
        "duration_ratio"
    ] = (
        q["m063_duration"]
        / (
            q["m047_duration"]
            + 1e-6
        )
    )

    q[
        "m047_word_count"
    ] = (
        q["m047_text"]
        .astype(str)
        .str.split()
        .str.len()
        .astype(float)
    )

    q[
        "m063_word_count"
    ] = (
        q["m063_text"]
        .astype(str)
        .str.split()
        .str.len()
        .astype(float)
    )

    q[
        "word_count_difference"
    ] = (
        q["m063_word_count"]
        - q["m047_word_count"]
    )

    q[
        "start_difference"
    ] = (
        q["m063_start"]
        - q["m047_start"]
    )

    q[
        "end_difference"
    ] = (
        q["m063_end"]
        - q["m047_end"]
    )

    center47 = (
        q["m047_start"]
        + q["m047_end"]
    ) / 2.0

    center63 = (
        q["m063_start"]
        + q["m063_end"]
    ) / 2.0

    q[
        "center_difference"
    ] = (
        center63
        - center47
    )

    q[
        "m063_is_earlier"
    ] = (
        center63
        < center47
    ).astype(float)

    jaccards = []

    for a, b in zip(
        q["m047_text"],
        q["m063_text"],
    ):
        ta = token_set(a)
        tb = token_set(b)

        union = (
            ta | tb
        )

        if not union:
            value = 0.0
        else:
            value = (
                len(
                    ta & tb
                )
                / len(union)
            )

        jaccards.append(
            value
        )

    q[
        "text_token_jaccard"
    ] = jaccards

    q[
        "kind_transition"
    ] = (
        q["kind47"].astype(str)
        + "->"
        + q["kind63"].astype(str)
    )

    # Gold utility is training-only.
    q[
        "utility"
    ] = (
        q["m063_tiou"]
        - q["m047_tiou"]
    )

    q[
        "m063_better"
    ] = (
        q["utility"]
        > 1e-9
    )

    q[
        "m047_better"
    ] = (
        q["utility"]
        < -1e-9
    )

    return q


def make_ridge():
    transformer = (
        ColumnTransformer(
            [
                (
                    "num",
                    StandardScaler(),
                    NUMERIC_FEATURES,
                ),
                (
                    "cat",
                    OneHotEncoder(
                        handle_unknown="ignore",
                    ),
                    CATEGORICAL_FEATURES,
                ),
            ]
        )
    )

    return Pipeline(
        [
            (
                "features",
                transformer,
            ),
            (
                "model",
                Ridge(
                    alpha=10.0,
                ),
            ),
        ]
    )


def make_logistic():
    transformer = (
        ColumnTransformer(
            [
                (
                    "num",
                    StandardScaler(),
                    NUMERIC_FEATURES,
                ),
                (
                    "cat",
                    OneHotEncoder(
                        handle_unknown="ignore",
                    ),
                    CATEGORICAL_FEATURES,
                ),
            ]
        )
    )

    return Pipeline(
        [
            (
                "features",
                transformer,
            ),
            (
                "model",
                LogisticRegression(
                    C=0.25,
                    max_iter=5000,
                    class_weight=None,
                    random_state=SEED,
                ),
            ),
        ]
    )


def make_hgb():
    # HGB cannot directly consume strings.
    # Use only numeric features here.
    return HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=100,
        max_leaf_nodes=5,
        min_samples_leaf=8,
        l2_regularization=2.0,
        random_state=SEED,
    )


def selected_mean(
    frame: pd.DataFrame,
    prediction: np.ndarray,
) -> float:
    return float(
        np.mean(
            np.where(
                prediction,
                frame["m063_tiou"],
                frame["m047_tiou"],
            )
        )
    )


def tune_threshold(
    frame: pd.DataFrame,
    scores: np.ndarray,
):
    # Candidate cutoffs from actual training scores.
    values = np.unique(
        scores
    )

    if len(values) == 1:
        thresholds = [
            float(
                values[0]
            )
        ]
    else:
        mids = (
            values[:-1]
            + values[1:]
        ) / 2.0

        thresholds = np.concatenate(
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

    best_threshold = None
    best_mean = -1.0

    for threshold in thresholds:
        use63 = (
            scores
            >= threshold
        )

        mean = selected_mean(
            frame,
            use63,
        )

        if mean > best_mean:
            best_mean = mean
            best_threshold = float(
                threshold
            )

    return (
        best_threshold,
        best_mean,
    )


def evaluate_model(
    name: str,
    question_table: pd.DataFrame,
):
    groups = (
        question_table[
            "transcript_id"
        ]
        .astype(str)
        .to_numpy()
    )

    splitter = GroupKFold(
        n_splits=N_SPLITS
    )

    question_table = (
        question_table.copy()
    )

    score_column = (
        f"{name}_score"
    )

    decision_column = (
        f"{name}_choose_m063"
    )

    question_table[
        score_column
    ] = np.nan

    question_table[
        decision_column
    ] = False

    fold_thresholds = []

    print()
    print("=" * 78)
    print(
        f"M064 — {name.upper()}"
    )
    print("=" * 78)

    for fold, (
        train_idx,
        test_idx,
    ) in enumerate(
        splitter.split(
            question_table,
            groups=groups,
        ),
        start=1,
    ):
        train_all = (
            question_table.iloc[
                train_idx
            ].copy()
        )

        test_all = (
            question_table.iloc[
                test_idx
            ].copy()
        )

        # Arbiter only needs examples where
        # M047 and M063 actually disagree.
        train = train_all[
            train_all["changed"]
        ].copy()

        test = test_all[
            test_all["changed"]
        ].copy()

        # Ties provide no preference signal.
        train_non_tie = train[
            ~np.isclose(
                train["utility"],
                0.0,
            )
        ].copy()

        if name == "ridge":
            model = make_ridge()

            model.fit(
                train_non_tie[
                    NUMERIC_FEATURES
                    + CATEGORICAL_FEATURES
                ],
                train_non_tie[
                    "utility"
                ],
            )

            train_scores = (
                model.predict(
                    train[
                        NUMERIC_FEATURES
                        + CATEGORICAL_FEATURES
                    ]
                )
            )

            test_scores = (
                model.predict(
                    test[
                        NUMERIC_FEATURES
                        + CATEGORICAL_FEATURES
                    ]
                )
            )

        elif name == "logistic":
            model = make_logistic()

            model.fit(
                train_non_tie[
                    NUMERIC_FEATURES
                    + CATEGORICAL_FEATURES
                ],
                train_non_tie[
                    "m063_better"
                ].astype(int),
            )

            train_scores = (
                model.predict_proba(
                    train[
                        NUMERIC_FEATURES
                        + CATEGORICAL_FEATURES
                    ]
                )[:, 1]
            )

            test_scores = (
                model.predict_proba(
                    test[
                        NUMERIC_FEATURES
                        + CATEGORICAL_FEATURES
                    ]
                )[:, 1]
            )

        elif name == "hgb":
            model = make_hgb()

            model.fit(
                train_non_tie[
                    NUMERIC_FEATURES
                ],
                train_non_tie[
                    "utility"
                ],
            )

            train_scores = (
                model.predict(
                    train[
                        NUMERIC_FEATURES
                    ]
                )
            )

            test_scores = (
                model.predict(
                    test[
                        NUMERIC_FEATURES
                    ]
                )
            )

        else:
            raise ValueError(
                name
            )

        threshold, train_mean = (
            tune_threshold(
                train,
                train_scores,
            )
        )

        test_decision = (
            test_scores
            >= threshold
        )

        # Same candidates require no arbitration.
        full_test_decision = np.zeros(
            len(test_all),
            dtype=bool,
        )

        full_test_scores = np.zeros(
            len(test_all),
            dtype=float,
        )

        changed_positions = (
            np.where(
                test_all[
                    "changed"
                ].to_numpy()
            )[0]
        )

        full_test_decision[
            changed_positions
        ] = test_decision

        full_test_scores[
            changed_positions
        ] = test_scores

        question_table.loc[
            test_all.index,
            score_column,
        ] = full_test_scores

        question_table.loc[
            test_all.index,
            decision_column,
        ] = full_test_decision

        baseline = float(
            test_all[
                "m047_tiou"
            ].mean()
        )

        pure63 = float(
            test_all[
                "m063_tiou"
            ].mean()
        )

        gated = selected_mean(
            test_all,
            full_test_decision,
        )

        accepted = int(
            full_test_decision.sum()
        )

        fold_thresholds.append(
            threshold
        )

        print(
            f"Fold {fold}: "
            f"train_changed="
            f"{len(train)}, "
            f"test_changed="
            f"{len(test)}, "
            f"threshold="
            f"{threshold:.4f}, "
            f"accepted="
            f"{accepted}, "
            f"M047="
            f"{baseline:.4f}, "
            f"M063="
            f"{pure63:.4f}, "
            f"M064="
            f"{gated:.4f}, "
            f"delta="
            f"{gated-baseline:+.4f}"
        )

    if (
        question_table[
            score_column
        ]
        .isna()
        .any()
    ):
        raise RuntimeError(
            f"Missing OOF outputs "
            f"for {name}"
        )

    decisions = (
        question_table[
            decision_column
        ]
        .to_numpy(
            dtype=bool
        )
    )

    overall = selected_mean(
        question_table,
        decisions,
    )

    oracle = float(
        question_table[
            [
                "m047_tiou",
                "m063_tiou",
            ]
        ]
        .max(axis=1)
        .mean()
    )

    accepted = int(
        decisions.sum()
    )

    useful_switches = int(
        (
            decisions
            & (
                question_table[
                    "utility"
                ]
                > 0
            )
        ).sum()
    )

    harmful_switches = int(
        (
            decisions
            & (
                question_table[
                    "utility"
                ]
                < 0
            )
        ).sum()
    )

    neutral_switches = int(
        (
            decisions
            & np.isclose(
                question_table[
                    "utility"
                ],
                0.0,
            )
        ).sum()
    )

    print()
    print(
        f"{name.upper()} OOF RESULT"
    )

    print("-" * 78)

    print(
        f"Mean tIoU:       "
        f"{overall:.4f}"
    )

    print(
        f"Gain vs M047:    "
        f"{overall - 0.5158235448:+.4f}"
    )

    print(
        f"Accepted M063:   "
        f"{accepted}"
    )

    print(
        f"Useful switches: "
        f"{useful_switches}"
    )

    print(
        f"Harmful switches:"
        f" {harmful_switches}"
    )

    print(
        f"Neutral switches:"
        f" {neutral_switches}"
    )

    print(
        f"Oracle:          "
        f"{oracle:.4f}"
    )

    return (
        question_table,
        overall,
        name,
    )


def main():
    raw = pd.read_csv(
        INPUT
    )

    if (
        raw["question_id"]
        .nunique()
        != 195
    ):
        raise RuntimeError(
            "Expected 195 questions."
        )

    q = build_question_table(
        raw
    )

    print("=" * 78)
    print(
        "Nordic AI Cup 2026 — "
        "M064 M047/M063 Evidence Arbiter"
    )
    print("=" * 78)

    print(
        f"Questions:    {len(q)}"
    )

    print(
        f"Changed:      "
        f"{int(q.changed.sum())}"
    )

    print(
        f"M063 wins:    "
        f"{int(q.m063_better.sum())}"
    )

    print(
        f"M047 wins:    "
        f"{int(q.m047_better.sum())}"
    )

    print(
        f"Ties:         "
        f"{int(np.isclose(q.utility, 0).sum())}"
    )

    print(
        f"M047 mean:    "
        f"{q.m047_tiou.mean():.4f}"
    )

    print(
        f"M063 mean:    "
        f"{q.m063_tiou.mean():.4f}"
    )

    oracle = float(
        q[
            [
                "m047_tiou",
                "m063_tiou",
            ]
        ]
        .max(axis=1)
        .mean()
    )

    print(
        f"Oracle:       "
        f"{oracle:.4f}"
    )

    results = []

    for model_name in (
        "ridge",
        "logistic",
        "hgb",
    ):
        result = (
            evaluate_model(
                model_name,
                q,
            )
        )

        results.append(
            result
        )

    best = max(
        results,
        key=lambda item:
            item[1],
    )

    best_table, best_mean, best_name = (
        best
    )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_table.to_csv(
        OUTPUT,
        index=False,
    )

    print()
    print("=" * 78)
    print(
        "M064 BEST RESULT"
    )
    print("=" * 78)

    print(
        f"Model:              "
        f"{best_name}"
    )

    print(
        f"M047:               "
        f"{q.m047_tiou.mean():.4f}"
    )

    print(
        f"M063:               "
        f"{q.m063_tiou.mean():.4f}"
    )

    print(
        f"M064:               "
        f"{best_mean:.4f}"
    )

    print(
        f"Gain vs M047:       "
        f"{best_mean - q.m047_tiou.mean():+.4f}"
    )

    print(
        f"Two-model oracle:   "
        f"{oracle:.4f}"
    )

    print(
        f"Oracle captured:    "
        f"{(
            (best_mean - q.m047_tiou.mean())
            /
            max(
                oracle
                - q.m047_tiou.mean(),
                1e-12,
            )
        ):.1%}"
    )

    print()
    print(
        f"Saved: {OUTPUT}"
    )


if __name__ == "__main__":
    main()