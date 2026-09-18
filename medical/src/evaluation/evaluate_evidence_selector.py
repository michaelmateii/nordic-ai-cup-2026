from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


HGB_PATH = Path(
    r"medical\artifacts\retrieval\supervised_sentence_ranker_results.csv"
)

NEURAL_PATH = Path(
    r"medical\artifacts\retrieval\finetuned_sentence_ranker_oof.csv"
)

CLASSIFICATION_PATH = Path(
    r"medical\artifacts\classification\nli_segmentwise_cv_results.csv"
)

OUTPUT = Path(
    r"medical\artifacts\evaluation\m032b_evidence_selector.csv"
)

METHOD = "max_segment_margin"
N_SPLITS = 5


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
        f"Cannot parse boolean: {value!r}"
    )


def best_threshold(
    delta: np.ndarray,
    hgb_tiou: np.ndarray,
    neural_tiou: np.ndarray,
) -> tuple[float, float]:
    """
    Choose threshold on training conversations.

    If delta >= threshold -> neural.
    Otherwise -> HGB.

    Objective: maximize mean evidence tIoU.
    """

    unique = np.unique(delta)

    if len(unique) == 1:
        candidates = np.asarray(
            [
                unique[0] - 1e-6,
                unique[0] + 1e-6,
            ],
            dtype=float,
        )
    else:
        midpoints = (
            unique[:-1]
            + unique[1:]
        ) / 2.0

        candidates = np.concatenate(
            [
                np.asarray(
                    [
                        unique[0] - 1e-6,
                    ]
                ),
                midpoints,
                np.asarray(
                    [
                        unique[-1] + 1e-6,
                    ]
                ),
            ]
        )

    best_value = -1.0
    best_cutoff = 0.0

    for threshold in candidates:
        use_neural = (
            delta >= threshold
        )

        chosen = np.where(
            use_neural,
            neural_tiou,
            hgb_tiou,
        )

        value = float(
            chosen.mean()
        )

        if value > best_value:
            best_value = value
            best_cutoff = float(
                threshold
            )

    return (
        best_cutoff,
        best_value,
    )


def main() -> None:
    hgb = pd.read_csv(
        HGB_PATH
    )

    neural = pd.read_csv(
        NEURAL_PATH
    )

    required_hgb = {
        "question_id",
        "transcript_id",
        "candidate_start",
        "candidate_end",
        "target_tiou",
        "oof_predicted_tiou",
    }

    required_neural = {
        "question_id",
        "transcript_id",
        "candidate_start",
        "candidate_end",
        "target_tiou",
        "oof_score",
    }

    missing_hgb = (
        required_hgb
        - set(hgb.columns)
    )

    missing_neural = (
        required_neural
        - set(neural.columns)
    )

    if missing_hgb:
        raise RuntimeError(
            "M025 CSV missing: "
            f"{sorted(missing_hgb)}"
        )

    if missing_neural:
        raise RuntimeError(
            "M030 CSV missing: "
            f"{sorted(missing_neural)}"
        )

    hgb = hgb[
        [
            "question_id",
            "transcript_id",
            "candidate_start",
            "candidate_end",
            "target_tiou",
            "oof_predicted_tiou",
        ]
    ].rename(
        columns={
            "transcript_id": (
                "hgb_transcript_id"
            ),
            "candidate_start": (
                "hgb_start"
            ),
            "candidate_end": (
                "hgb_end"
            ),
            "target_tiou": (
                "hgb_tiou"
            ),
            "oof_predicted_tiou": (
                "hgb_confidence"
            ),
        }
    )

    neural = neural[
        [
            "question_id",
            "transcript_id",
            "candidate_start",
            "candidate_end",
            "target_tiou",
            "oof_score",
        ]
    ].rename(
        columns={
            "transcript_id": (
                "neural_transcript_id"
            ),
            "candidate_start": (
                "neural_start"
            ),
            "candidate_end": (
                "neural_end"
            ),
            "target_tiou": (
                "neural_tiou"
            ),
            "oof_score": (
                "neural_confidence"
            ),
        }
    )

    df = hgb.merge(
        neural,
        on="question_id",
        how="inner",
        validate="one_to_one",
    )

    if len(df) != 195:
        raise RuntimeError(
            f"Expected 195 rows, "
            f"got {len(df)}"
        )

    if not (
        df["hgb_transcript_id"]
        .astype(str)
        .equals(
            df[
                "neural_transcript_id"
            ].astype(str)
        )
    ):
        raise RuntimeError(
            "Transcript IDs do not match."
        )

    df["transcript_id"] = (
        df["hgb_transcript_id"]
    )

    df["confidence_delta"] = (
        df["neural_confidence"]
        - df["hgb_confidence"]
    )

    # ---------------------------------------------------------
    # Simple raw confidence selector.
    # This is diagnostic only:
    # neural confidence >= HGB confidence.
    # ---------------------------------------------------------

    df["raw_use_neural"] = (
        df["confidence_delta"] >= 0.0
    )

    df["raw_selector_tiou"] = (
        np.where(
            df["raw_use_neural"],
            df["neural_tiou"],
            df["hgb_tiou"],
        )
    )

    # ---------------------------------------------------------
    # Conversation-disjoint threshold calibration.
    # ---------------------------------------------------------

    splitter = GroupKFold(
        n_splits=N_SPLITS
    )

    groups = (
        df["transcript_id"]
        .astype(str)
        .to_numpy()
    )

    oof_use_neural = np.zeros(
        len(df),
        dtype=bool,
    )

    fold_thresholds = []

    for fold, (
        train_index,
        test_index,
    ) in enumerate(
        splitter.split(
            df,
            groups=groups,
        ),
        start=1,
    ):
        train = df.iloc[
            train_index
        ]

        threshold, train_score = (
            best_threshold(
                train[
                    "confidence_delta"
                ].to_numpy(
                    dtype=float
                ),
                train[
                    "hgb_tiou"
                ].to_numpy(
                    dtype=float
                ),
                train[
                    "neural_tiou"
                ].to_numpy(
                    dtype=float
                ),
            )
        )

        test_delta = (
            df.iloc[
                test_index
            ][
                "confidence_delta"
            ].to_numpy(
                dtype=float
            )
        )

        oof_use_neural[
            test_index
        ] = (
            test_delta >= threshold
        )

        fold_thresholds.append(
            threshold
        )

        print(
            f"Fold {fold}: "
            f"threshold={threshold:+.6f}, "
            f"train_tIoU={train_score:.4f}, "
            f"test_questions="
            f"{len(test_index)}"
        )

    df["oof_use_neural"] = (
        oof_use_neural
    )

    df["selector_tiou"] = np.where(
        df["oof_use_neural"],
        df["neural_tiou"],
        df["hgb_tiou"],
    )

    df["selector_start"] = (
        np.where(
            df["oof_use_neural"],
            df["neural_start"],
            df["hgb_start"],
        )
    )

    df["selector_end"] = (
        np.where(
            df["oof_use_neural"],
            df["neural_end"],
            df["hgb_end"],
        )
    )

    df["oracle_two_tiou"] = (
        df[
            [
                "hgb_tiou",
                "neural_tiou",
            ]
        ].max(axis=1)
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
        .reset_index(
            drop=True
        )
    )

    if len(classification) != 390:
        raise RuntimeError(
            "Expected 390 classification rows."
        )

    classification["gold_bool"] = [
        parse_bool(value)
        for value
        in classification["gold_yes"]
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

    positive_classification = (
        classification[
            classification[
                "gold_bool"
            ]
        ].copy()
    )

    selector_by_id = df.set_index(
        "question_id"
    )

    scored_tious = []

    for _, row in (
        positive_classification
        .iterrows()
    ):
        if not row["pred_bool"]:
            scored_tious.append(
                0.0
            )
            continue

        question_id = (
            row["question_id"]
        )

        scored_tious.append(
            float(
                selector_by_id.loc[
                    question_id,
                    "selector_tiou",
                ]
            )
        )

    mean_scored_tiou = float(
        np.mean(
            scored_tious
        )
    )

    composite = (
        0.4 * accuracy
        + 0.6 * mean_scored_tiou
    )

    # ---------------------------------------------------------
    # Save + report.
    # ---------------------------------------------------------

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT,
        index=False,
    )

    hgb_mean = float(
        df["hgb_tiou"].mean()
    )

    neural_mean = float(
        df["neural_tiou"].mean()
    )

    raw_mean = float(
        df[
            "raw_selector_tiou"
        ].mean()
    )

    selector_mean = float(
        df["selector_tiou"].mean()
    )

    oracle_mean = float(
        df[
            "oracle_two_tiou"
        ].mean()
    )

    print()
    print("=" * 78)
    print(
        "Nordic AI Cup 2026 — "
        "M032B Confidence-Gated Evidence Selector"
    )
    print("=" * 78)

    print()
    print("EVIDENCE OOF")
    print("-" * 78)

    print(
        f"M025 HGB:                  "
        f"{hgb_mean:.4f}"
    )

    print(
        f"M030 neural:               "
        f"{neural_mean:.4f}"
    )

    print(
        f"Raw confidence selector:   "
        f"{raw_mean:.4f}"
    )

    print(
        f"CV threshold selector:     "
        f"{selector_mean:.4f}"
    )

    print(
        f"Oracle of two:             "
        f"{oracle_mean:.4f}"
    )

    print()
    print(
        "Selector gain vs M030:     "
        f"{selector_mean - neural_mean:+.4f}"
    )

    print(
        "Remaining oracle gap:      "
        f"{oracle_mean - selector_mean:+.4f}"
    )

    print()
    print("SELECTION")
    print("-" * 78)

    print(
        "OOF selected neural:       "
        f"{int(df['oof_use_neural'].sum())}"
        f"/{len(df)}"
    )

    print(
        "OOF selected HGB:          "
        f"{int((~df['oof_use_neural']).sum())}"
        f"/{len(df)}"
    )

    print(
        "Fold thresholds:           "
        + ", ".join(
            f"{value:+.6f}"
            for value
            in fold_thresholds
        )
    )

    print(
        "Mean threshold:            "
        f"{np.mean(fold_thresholds):+.6f}"
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
        f"{mean_scored_tiou:.4f}"
    )

    print(
        f"Composite score:           "
        f"{composite:.4f}"
    )

    print()
    print("REFERENCE")
    print("-" * 78)

    print(
        "M031 composite:            0.6185"
    )

    print(
        "M030 evidence mean:        0.5089"
    )

    print(
        "M032A two-choice oracle:   0.5862"
    )

    print()
    print(
        f"Detailed results: {OUTPUT}"
    )


if __name__ == "__main__":
    main()