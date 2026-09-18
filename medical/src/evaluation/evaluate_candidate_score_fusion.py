from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


HGB_PATH = Path(
    r"medical\artifacts\retrieval\m025_all_candidate_oof_scores.csv"
)

NEURAL_PATH = Path(
    r"medical\artifacts\retrieval\m030_all_candidate_oof_scores.csv"
)

CLASSIFICATION_PATH = Path(
    r"medical\artifacts\classification\nli_segmentwise_cv_results.csv"
)

OUTPUT = Path(
    r"medical\artifacts\evaluation\m033_candidate_score_fusion.csv"
)

METHOD = "max_segment_margin"
N_SPLITS = 5

ALPHAS = np.linspace(
    0.0,
    1.0,
    21,
)


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


def add_rank_scores(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    dataframe = dataframe.copy()

    dataframe["hgb_rank"] = (
        dataframe.groupby(
            "question_id"
        )[
            "hgb_score"
        ]
        .rank(
            method="average",
            pct=True,
        )
    )

    dataframe["neural_rank"] = (
        dataframe.groupby(
            "question_id"
        )[
            "neural_score"
        ]
        .rank(
            method="average",
            pct=True,
        )
    )

    return dataframe


def evaluate_alpha(
    dataframe: pd.DataFrame,
    alpha: float,
) -> float:
    work = dataframe.copy()

    work["fusion_score"] = (
        alpha
        * work["neural_rank"]
        + (
            1.0 - alpha
        )
        * work["hgb_rank"]
    )

    selected_index = (
        work.groupby(
            "question_id"
        )[
            "fusion_score"
        ]
        .idxmax()
    )

    selected = work.loc[
        selected_index
    ]

    return float(
        selected[
            "target_tiou"
        ].mean()
    )


def select_with_alpha(
    dataframe: pd.DataFrame,
    alpha: float,
) -> pd.DataFrame:
    work = dataframe.copy()

    work["fusion_score"] = (
        alpha
        * work["neural_rank"]
        + (
            1.0 - alpha
        )
        * work["hgb_rank"]
    )

    selected_index = (
        work.groupby(
            "question_id"
        )[
            "fusion_score"
        ]
        .idxmax()
    )

    return (
        work.loc[
            selected_index
        ]
        .copy()
    )


def main() -> None:
    hgb = pd.read_csv(
        HGB_PATH
    )

    neural = pd.read_csv(
        NEURAL_PATH
    )

    hgb_required = {
        "question_id",
        "transcript_id",
        "candidate_index",
        "candidate_start",
        "candidate_end",
        "target_tiou",
        "oof_predicted_tiou",
    }

    neural_required = {
        "question_id",
        "transcript_id",
        "candidate_index",
        "candidate_start",
        "candidate_end",
        "target_tiou",
        "oof_score",
    }

    if not hgb_required.issubset(
        hgb.columns
    ):
        raise RuntimeError(
            "Missing HGB columns: "
            f"{sorted(hgb_required - set(hgb.columns))}"
        )

    if not neural_required.issubset(
        neural.columns
    ):
        raise RuntimeError(
            "Missing neural columns: "
            f"{sorted(neural_required - set(neural.columns))}"
        )

    hgb = hgb[
        [
            "question_id",
            "transcript_id",
            "candidate_index",
            "candidate_start",
            "candidate_end",
            "target_tiou",
            "oof_predicted_tiou",
        ]
    ].rename(
        columns={
            "transcript_id": "hgb_transcript_id",
            "candidate_start": "hgb_start",
            "candidate_end": "hgb_end",
            "target_tiou": "hgb_target_tiou",
            "oof_predicted_tiou": "hgb_score",
        }
    )

    neural = neural[
        [
            "question_id",
            "transcript_id",
            "candidate_index",
            "candidate_start",
            "candidate_end",
            "target_tiou",
            "oof_score",
        ]
    ].rename(
        columns={
            "transcript_id": "neural_transcript_id",
            "candidate_start": "neural_start",
            "candidate_end": "neural_end",
            "target_tiou": "neural_target_tiou",
            "oof_score": "neural_score",
        }
    )

    df = hgb.merge(
        neural,
        on=[
            "question_id",
            "candidate_index",
        ],
        how="inner",
        validate="one_to_one",
    )

    if len(df) != len(hgb):
        raise RuntimeError(
            "Candidate merge lost rows: "
            f"HGB={len(hgb)}, merged={len(df)}"
        )

    if not np.allclose(
        df["hgb_start"],
        df["neural_start"],
        atol=1e-6,
    ):
        raise RuntimeError(
            "Candidate starts differ."
        )

    if not np.allclose(
        df["hgb_end"],
        df["neural_end"],
        atol=1e-6,
    ):
        raise RuntimeError(
            "Candidate ends differ."
        )

    if not np.allclose(
        df["hgb_target_tiou"],
        df["neural_target_tiou"],
        atol=1e-9,
    ):
        raise RuntimeError(
            "Candidate targets differ."
        )

    df["transcript_id"] = (
        df["hgb_transcript_id"]
    )

    df["candidate_start"] = (
        df["hgb_start"]
    )

    df["candidate_end"] = (
        df["hgb_end"]
    )

    df["target_tiou"] = (
        df["hgb_target_tiou"]
    )

    df = add_rank_scores(
        df
    )

    question_table = (
        df[
            [
                "question_id",
                "transcript_id",
            ]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    if len(question_table) != 195:
        raise RuntimeError(
            f"Expected 195 questions, "
            f"got {len(question_table)}"
        )

    splitter = GroupKFold(
        n_splits=N_SPLITS
    )

    groups = (
        question_table[
            "transcript_id"
        ].astype(str)
    )

    selected_parts = []
    fold_alphas = []

    for fold, (
        train_index,
        test_index,
    ) in enumerate(
        splitter.split(
            question_table,
            groups=groups,
        ),
        start=1,
    ):
        train_ids = set(
            question_table.iloc[
                train_index
            ]["question_id"]
        )

        test_ids = set(
            question_table.iloc[
                test_index
            ]["question_id"]
        )

        train = df[
            df["question_id"].isin(
                train_ids
            )
        ]

        test = df[
            df["question_id"].isin(
                test_ids
            )
        ]

        best_alpha = None
        best_score = -1.0

        for alpha in ALPHAS:
            score = evaluate_alpha(
                train,
                float(alpha),
            )

            if score > best_score:
                best_score = score
                best_alpha = float(
                    alpha
                )

        assert best_alpha is not None

        selected = select_with_alpha(
            test,
            best_alpha,
        )

        selected["fold"] = fold
        selected["fold_alpha"] = (
            best_alpha
        )

        selected_parts.append(
            selected
        )

        fold_alphas.append(
            best_alpha
        )

        print(
            f"Fold {fold}: "
            f"alpha={best_alpha:.2f}, "
            f"train_tIoU={best_score:.4f}, "
            f"test_questions={len(test_ids)}"
        )

    result = pd.concat(
        selected_parts,
        ignore_index=True,
    )

    if len(result) != 195:
        raise RuntimeError(
            f"Expected 195 selected rows; "
            f"got {len(result)}"
        )

    result.to_csv(
        OUTPUT,
        index=False,
    )

    evidence_tiou = float(
        result[
            "target_tiou"
        ].mean()
    )

    # ---------------------------------------------------------
    # Composite with frozen M018 classifier.
    # ---------------------------------------------------------

    classification = pd.read_csv(
        CLASSIFICATION_PATH
    )

    classification = (
        classification[
            classification["method"]
            == METHOD
        ]
        .sort_values(
            "row_index"
        )
        .reset_index(drop=True)
    )

    classification["gold_bool"] = [
        parse_bool(value)
        for value
        in classification[
            "gold_yes"
        ]
    ]

    classification["pred_bool"] = [
        parse_bool(value)
        for value
        in classification[
            "predicted_yes"
        ]
    ]

    accuracy = float(
        (
            classification["gold_bool"]
            == classification["pred_bool"]
        ).mean()
    )

    positive = classification[
        classification["gold_bool"]
    ]

    evidence_by_id = (
        result.set_index(
            "question_id"
        )
    )

    scored = []

    for _, row in positive.iterrows():
        if not row["pred_bool"]:
            scored.append(0.0)
            continue

        scored.append(
            float(
                evidence_by_id.loc[
                    row["question_id"],
                    "target_tiou",
                ]
            )
        )

    scored_tiou = float(
        np.mean(scored)
    )

    composite = (
        0.4 * accuracy
        + 0.6 * scored_tiou
    )

    scores = result[
        "target_tiou"
    ]

    print()
    print("=" * 78)
    print(
        "Nordic AI Cup 2026 — "
        "M033 Candidate-Level Score Fusion"
    )
    print("=" * 78)

    print()
    print("OOF EVIDENCE")
    print("-" * 78)

    print(
        f"Mean tIoU:                 "
        f"{evidence_tiou:.4f}"
    )

    print(
        f"Median tIoU:               "
        f"{scores.median():.4f}"
    )

    print(
        f"Any overlap:               "
        f"{(scores > 0).mean():.4f}"
    )

    print(
        f"tIoU >= 0.50:              "
        f"{(scores >= 0.50).mean():.4f}"
    )

    print(
        f"tIoU >= 0.75:              "
        f"{(scores >= 0.75).mean():.4f}"
    )

    print()
    print("FUSION")
    print("-" * 78)

    print(
        "Fold neural weights:        "
        + ", ".join(
            f"{value:.2f}"
            for value
            in fold_alphas
        )
    )

    print(
        f"Mean neural weight:         "
        f"{np.mean(fold_alphas):.3f}"
    )

    print()
    print("COMPOSITE")
    print("-" * 78)

    print(
        f"Classification accuracy:   "
        f"{accuracy:.4f}"
    )

    print(
        f"Mean scored tIoU:          "
        f"{scored_tiou:.4f}"
    )

    print(
        f"Composite score:           "
        f"{composite:.4f}"
    )

    print()
    print("REFERENCE")
    print("-" * 78)

    print(
        "M025 evidence:             0.4771"
    )

    print(
        "M030 evidence:             0.5089"
    )

    print(
        "M031 composite:            0.6185"
    )

    print(
        "Candidate oracle:          0.7997"
    )

    print()
    print(
        f"Detailed results: {OUTPUT}"
    )


if __name__ == "__main__":
    main()