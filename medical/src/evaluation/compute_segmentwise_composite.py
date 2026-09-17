from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_CLASSIFICATION = Path(
    r"medical\artifacts\classification\nli_segmentwise_cv_results.csv"
)

DEFAULT_EVIDENCE = Path(
    r"medical\artifacts\retrieval\cross_encoder_segment_results.csv"
)

DEFAULT_OUTPUT = Path(
    r"medical\artifacts\evaluation\segmentwise_composite_dev_score.csv"
)

METHOD = "max_segment_margin"


def parse_bool(value: object) -> bool:
    text = str(value).strip().lower()

    if text in {"true", "1", "yes"}:
        return True

    if text in {"false", "0", "no"}:
        return False

    raise ValueError(value)


def main() -> None:
    classification = pd.read_csv(
        DEFAULT_CLASSIFICATION
    )

    evidence = pd.read_csv(
        DEFAULT_EVIDENCE
    )

    classification = classification[
        classification["method"] == METHOD
    ].copy()

    classification = classification.sort_values(
        "row_index"
    ).reset_index(drop=True)

    classification["gold_yes_bool"] = [
        parse_bool(value)
        for value in classification["gold_yes"]
    ]

    classification["pred_yes_bool"] = [
        parse_bool(value)
        for value in classification["predicted_yes"]
    ]

    if len(classification) != 390:
        raise RuntimeError(
            f"Expected 390 questions, got {len(classification)}"
        )

    if int(
        classification["gold_yes_bool"].sum()
    ) != 195:
        raise RuntimeError(
            "Expected exactly 195 gold positives."
        )

    accuracy = float(
        (
            classification["gold_yes_bool"]
            == classification["pred_yes_bool"]
        ).mean()
    )

    evidence_by_id = evidence.set_index(
        "question_id"
    )

    positive = classification[
        classification["gold_yes_bool"]
    ].copy()

    scored_tious = []

    for _, row in positive.iterrows():
        if not row["pred_yes_bool"]:
            scored_tious.append(0.0)
            continue

        question_id = row["question_id"]

        if question_id not in evidence_by_id.index:
            raise RuntimeError(
                f"Missing evidence for {question_id}"
            )

        scored_tious.append(
            float(
                evidence_by_id.loc[
                    question_id,
                    "tiou",
                ]
            )
        )

    positive["scored_tiou"] = scored_tious

    mean_tiou = float(
        np.mean(scored_tious)
    )

    composite = (
        0.4 * accuracy
        + 0.6 * mean_tiou
    )

    returned = positive[
        positive["pred_yes_bool"]
    ]

    DEFAULT_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    positive.to_csv(
        DEFAULT_OUTPUT,
        index=False,
    )

    print("=" * 78)
    print(
        "Nordic AI Cup 2026 — "
        "M018 Composite Development Score"
    )
    print("=" * 78)

    print(
        f"Classification method: {METHOD}"
    )

    print(
        "Evidence method:       "
        "M010 cross-encoder segment"
    )

    print()

    print(
        f"Accuracy:              "
        f"{accuracy:.4f}"
    )

    print(
        f"Mean scored tIoU:      "
        f"{mean_tiou:.4f}"
    )

    print(
        f"Composite score:       "
        f"{composite:.4f}"
    )

    print()

    print(
        "Positive predicted YES: "
        f"{int(positive['pred_yes_bool'].sum())}"
        f"/{len(positive)}"
    )

    print(
        "tIoU among returned positives: "
        f"{returned['scored_tiou'].mean():.4f}"
    )

    print()
    print("REFERENCE")
    print("-" * 78)
    print(
        "M015 composite score: 0.5636"
    )


if __name__ == "__main__":
    main()