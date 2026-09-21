from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold


INPUT = Path(
    r"medical\artifacts\retrieval"
    r"\m047_all_questions_candidates_oof.csv"
)

OUTPUT = Path(
    r"medical\artifacts\retrieval"
    r"\m062_temporal_prior_oof.csv"
)

N_SPLITS = 5


# Search ranges.
LAMBDA_GRID = np.asarray(
    [
        0.00,
        0.01,
        0.02,
        0.03,
        0.04,
        0.05,
        0.075,
        0.10,
        0.125,
        0.15,
        0.20,
        0.25,
        0.30,
    ],
    dtype=float,
)

DELTA_GRID = np.asarray(
    [
        0.00,
        0.01,
        0.02,
        0.03,
        0.05,
        0.075,
        0.10,
        0.125,
        0.15,
        0.20,
    ],
    dtype=float,
)


def minmax_per_question(
    df: pd.DataFrame,
    column: str,
) -> pd.Series:
    minimum = (
        df.groupby("question_id")[column]
        .transform("min")
    )

    maximum = (
        df.groupby("question_id")[column]
        .transform("max")
    )

    denominator = (
        maximum - minimum
    ).replace(0.0, 1.0)

    return (
        df[column] - minimum
    ) / denominator


def select_by_score(
    df: pd.DataFrame,
    score_column: str,
    question_ids: set[str] | None = None,
) -> pd.DataFrame:
    if question_ids is None:
        subset = df
    else:
        subset = df[
            df["question_id"].isin(
                question_ids
            )
        ]

    indices = (
        subset.groupby("question_id")[
            score_column
        ]
        .idxmax()
    )

    return (
        subset.loc[indices]
        .copy()
        .reset_index(drop=True)
    )


def mean_tiou(
    df: pd.DataFrame,
    score_column: str,
    question_ids: set[str] | None = None,
) -> float:
    selected = select_by_score(
        df,
        score_column,
        question_ids,
    )

    return float(
        selected["target_tiou"].mean()
    )


def select_near_tie_earliest(
    df: pd.DataFrame,
    delta: float,
    question_ids: set[str] | None = None,
) -> pd.DataFrame:
    if question_ids is None:
        subset = df
    else:
        subset = df[
            df["question_id"].isin(
                question_ids
            )
        ]

    selected_rows = []

    for _, group in subset.groupby(
        "question_id",
        sort=False,
    ):
        top_score = float(
            group["oof_score"].max()
        )

        eligible = group[
            group["oof_score"]
            >= top_score - delta
        ]

        # Among semantically near-equivalent
        # candidates, prefer the earlier mention.
        chosen_index = (
            eligible["candidate_start"]
            .idxmin()
        )

        selected_rows.append(
            chosen_index
        )

    return (
        subset.loc[selected_rows]
        .copy()
        .reset_index(drop=True)
    )


def summarize(
    name: str,
    selected: pd.DataFrame,
):
    scores = (
        selected["target_tiou"]
        .astype(float)
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

    df = df[
        df["question_type"]
        == "positive"
    ].copy()

    if (
        df["question_id"].nunique()
        != 195
    ):
        raise RuntimeError(
            "Expected 195 positive questions."
        )

    # Limit to current top-20.
    df = (
        df.sort_values(
            [
                "question_id",
                "oof_score",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .groupby(
            "question_id",
            sort=False,
        )
        .head(20)
        .copy()
        .reset_index(drop=True)
    )

    # Ranker normalization within question.
    df["ranker_norm"] = (
        minmax_per_question(
            df,
            "oof_score",
        )
    )

    # Relative transcript position.
    transcript_end = (
        df.groupby(
            "transcript_id"
        )["candidate_end"]
        .transform("max")
    )

    df["relative_start"] = (
        df["candidate_start"]
        / transcript_end.clip(
            lower=1e-6
        )
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
        .reset_index(drop=True)
    )

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

    df["m062_penalty_score"] = np.nan

    chosen_near_tie = {}

    lambda_folds = []
    delta_folds = []

    print("=" * 78)
    print(
        "Nordic AI Cup 2026 — "
        "M062 Temporal Prior / Repeated Mention Audit"
    )
    print("=" * 78)

    baseline = (
        select_by_score(
            df,
            "oof_score",
        )
    )

    summarize(
        "M047 TOP-1",
        baseline,
    )

    print()
    print("=" * 78)
    print("A — TEMPORAL SCORE PENALTY")
    print("=" * 78)

    for fold, (
        train_q_idx,
        test_q_idx,
    ) in enumerate(
        splitter.split(
            question_table,
            groups=groups,
        ),
        start=1,
    ):
        train_ids = set(
            question_table.iloc[
                train_q_idx
            ]["question_id"]
        )

        test_ids = set(
            question_table.iloc[
                test_q_idx
            ]["question_id"]
        )

        best_lambda = 0.0
        best_train = -1.0

        for penalty in LAMBDA_GRID:
            temp = (
                df["ranker_norm"]
                - penalty
                * df["relative_start"]
            )

            df["_temp_score"] = temp

            result = mean_tiou(
                df,
                "_temp_score",
                train_ids,
            )

            if result > best_train:
                best_train = result
                best_lambda = float(
                    penalty
                )

        test_mask = (
            df["question_id"]
            .isin(test_ids)
        )

        df.loc[
            test_mask,
            "m062_penalty_score",
        ] = (
            df.loc[
                test_mask,
                "ranker_norm",
            ]
            - best_lambda
            * df.loc[
                test_mask,
                "relative_start",
            ]
        )

        baseline_fold = mean_tiou(
            df,
            "oof_score",
            test_ids,
        )

        test_fold = mean_tiou(
            df,
            "m062_penalty_score",
            test_ids,
        )

        lambda_folds.append(
            best_lambda
        )

        print(
            f"Fold {fold}: "
            f"lambda={best_lambda:.3f}, "
            f"train={best_train:.4f}, "
            f"M047={baseline_fold:.4f}, "
            f"M062={test_fold:.4f}, "
            f"delta="
            f"{test_fold-baseline_fold:+.4f}"
        )

    temporal_selected = (
        select_by_score(
            df,
            "m062_penalty_score",
        )
    )

    temporal_mean = summarize(
        "M062 TEMPORAL PENALTY OOF",
        temporal_selected,
    )

    print()
    print("=" * 78)
    print("B — NEAR-TIE EARLIEST")
    print("=" * 78)

    # Reset splitter because it is an iterator.
    splitter = GroupKFold(
        n_splits=N_SPLITS
    )

    near_tie_rows = []

    for fold, (
        train_q_idx,
        test_q_idx,
    ) in enumerate(
        splitter.split(
            question_table,
            groups=groups,
        ),
        start=1,
    ):
        train_ids = set(
            question_table.iloc[
                train_q_idx
            ]["question_id"]
        )

        test_ids = set(
            question_table.iloc[
                test_q_idx
            ]["question_id"]
        )

        best_delta = 0.0
        best_train = -1.0

        for delta in DELTA_GRID:
            selected = (
                select_near_tie_earliest(
                    df,
                    float(delta),
                    train_ids,
                )
            )

            result = float(
                selected[
                    "target_tiou"
                ].mean()
            )

            if result > best_train:
                best_train = result
                best_delta = float(
                    delta
                )

        selected_test = (
            select_near_tie_earliest(
                df,
                best_delta,
                test_ids,
            )
        )

        baseline_fold = mean_tiou(
            df,
            "oof_score",
            test_ids,
        )

        test_fold = float(
            selected_test[
                "target_tiou"
            ].mean()
        )

        selected_test[
            "m062_fold"
        ] = fold

        selected_test[
            "m062_delta"
        ] = best_delta

        near_tie_rows.append(
            selected_test
        )

        delta_folds.append(
            best_delta
        )

        print(
            f"Fold {fold}: "
            f"delta={best_delta:.3f}, "
            f"train={best_train:.4f}, "
            f"M047={baseline_fold:.4f}, "
            f"M062={test_fold:.4f}, "
            f"delta_score="
            f"{test_fold-baseline_fold:+.4f}"
        )

    near_tie_selected = (
        pd.concat(
            near_tie_rows,
            ignore_index=True,
        )
    )

    near_tie_mean = summarize(
        "M062 NEAR-TIE EARLIEST OOF",
        near_tie_selected,
    )

    # ---------------------------------------------------------
    # Failure analysis: how many zero-overlap mistakes fixed?
    # ---------------------------------------------------------

    base_lookup = (
        baseline
        .set_index("question_id")
        .sort_index()
    )

    near_lookup = (
        near_tie_selected
        .set_index("question_id")
        .reindex(base_lookup.index)
    )

    changed = (
        base_lookup[
            "candidate_index"
        ]
        != near_lookup[
            "candidate_index"
        ]
    )

    fixed_zero = (
        (
            base_lookup[
                "target_tiou"
            ]
            == 0
        )
        & (
            near_lookup[
                "target_tiou"
            ]
            > 0
        )
    )

    broke_overlap = (
        (
            base_lookup[
                "target_tiou"
            ]
            > 0
        )
        & (
            near_lookup[
                "target_tiou"
            ]
            == 0
        )
    )

    print()
    print("=" * 78)
    print("M062 SUMMARY")
    print("=" * 78)

    print(
        f"M047:                 "
        f"{baseline.target_tiou.mean():.4f}"
    )

    print(
        f"Temporal penalty:     "
        f"{temporal_mean:.4f}"
    )

    print(
        f"Near-tie earliest:    "
        f"{near_tie_mean:.4f}"
    )

    print(
        "Lambda folds:          "
        + ", ".join(
            f"{x:.3f}"
            for x
            in lambda_folds
        )
    )

    print(
        "Near-tie delta folds:  "
        + ", ".join(
            f"{x:.3f}"
            for x
            in delta_folds
        )
    )

    print(
        f"Near-tie selections "
        f"changed: {int(changed.sum())}"
    )

    print(
        f"Zero-overlap fixed:    "
        f"{int(fixed_zero.sum())}"
    )

    print(
        f"Overlap broken:        "
        f"{int(broke_overlap.sum())}"
    )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    near_tie_selected.to_csv(
        OUTPUT,
        index=False,
    )

    print()
    print(
        f"Saved: {OUTPUT}"
    )


if __name__ == "__main__":
    main()