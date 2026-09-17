from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


CLASSIFICATION = Path(
    r"medical\artifacts\classification\nli_segmentwise_cv_results.csv"
)

EVIDENCE = Path(
    r"medical\artifacts\retrieval\supervised_sentence_ranker_results.csv"
)

OUTPUT = Path(
    r"medical\artifacts\evaluation\m025_composite_dev_score.csv"
)

METHOD = "max_segment_margin"


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value

    text = str(value).strip().lower()

    if text in {"true", "1", "yes"}:
        return True

    if text in {"false", "0", "no"}:
        return False

    raise ValueError(
        f"Cannot parse boolean: {value!r}"
    )


def main() -> None:
    classification = pd.read_csv(
        CLASSIFICATION
    )

    classification = classification[
        classification["method"] == METHOD
    ].copy()

    classification = classification.sort_values(
        "row_index"
    ).reset_index(drop=True)

    if len(classification) != 390:
        raise RuntimeError(
            f"Expected 390 classification rows; "
            f"got {len(classification)}"
        )

    classification["gold_yes_bool"] = [
        parse_bool(value)
        for value in classification["gold_yes"]
    ]

    classification["pred_yes_bool"] = [
        parse_bool(value)
        for value in classification["predicted_yes"]
    ]

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

    evidence = pd.read_csv(
        EVIDENCE
    )

    if len(evidence) != 195:
        raise RuntimeError(
            f"Expected 195 evidence rows; "
            f"got {len(evidence)}"
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
                    "target_tiou",
                ]
            )
        )

    positive["scored_tiou"] = (
        scored_tious
    )

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

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    positive.to_csv(
        OUTPUT,
        index=False,
    )

    print("=" * 78)
    print(
        "Nordic AI Cup 2026 — "
        "M025 Composite Development Score"
    )
    print("=" * 78)

    print(
        f"Classification:       {METHOD}"
    )

    print(
        "Evidence:             "
        "M025 supervised sentence ranker"
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
        "M019 composite:       0.5818"
    )

    print(
        "M010 evidence mean:   0.4347"
    )

    print(
        "M025 evidence mean:   0.4771"
    )


if __name__ == "__main__":
    main()