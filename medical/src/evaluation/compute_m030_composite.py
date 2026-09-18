from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


CLASSIFICATION = Path(
    r"medical\artifacts\classification\nli_segmentwise_cv_results.csv"
)

EVIDENCE = Path(
    r"medical\artifacts\retrieval\finetuned_sentence_ranker_oof.csv"
)

OUTPUT = Path(
    r"medical\artifacts\evaluation\m030_composite_dev_score.csv"
)

METHOD = "max_segment_margin"


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


def main() -> None:
    classification = pd.read_csv(
        CLASSIFICATION
    )

    classification = classification[
        classification["method"]
        == METHOD
    ].copy()

    classification = (
        classification
        .sort_values("row_index")
        .reset_index(drop=True)
    )

    if len(classification) != 390:
        raise RuntimeError(
            "Expected 390 classification rows; "
            f"got {len(classification)}"
        )

    classification["gold_bool"] = [
        parse_bool(value)
        for value
        in classification["gold_yes"]
    ]

    classification["pred_bool"] = [
        parse_bool(value)
        for value
        in classification["predicted_yes"]
    ]

    accuracy = float(
        (
            classification["gold_bool"]
            == classification["pred_bool"]
        ).mean()
    )

    positives = classification[
        classification["gold_bool"]
    ].copy()

    if len(positives) != 195:
        raise RuntimeError(
            "Expected 195 gold positives; "
            f"got {len(positives)}"
        )

    evidence = pd.read_csv(
        EVIDENCE
    )

    if len(evidence) != 195:
        raise RuntimeError(
            "Expected 195 evidence rows; "
            f"got {len(evidence)}"
        )

    evidence = evidence.set_index(
        "question_id"
    )

    scored_tious = []

    for _, row in positives.iterrows():
        if not row["pred_bool"]:
            scored_tious.append(0.0)
            continue

        question_id = row["question_id"]

        if question_id not in evidence.index:
            raise RuntimeError(
                "Missing evidence row for "
                f"{question_id}"
            )

        scored_tious.append(
            float(
                evidence.loc[
                    question_id,
                    "target_tiou",
                ]
            )
        )

    positives["scored_tiou"] = (
        scored_tious
    )

    mean_tiou = float(
        np.mean(scored_tious)
    )

    composite = (
        0.4 * accuracy
        + 0.6 * mean_tiou
    )

    returned = positives[
        positives["pred_bool"]
    ]

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    positives.to_csv(
        OUTPUT,
        index=False,
    )

    print("=" * 78)
    print(
        "Nordic AI Cup 2026 — "
        "M030 Composite Development Score"
    )
    print("=" * 78)

    print(
        f"Classification:       {METHOD}"
    )

    print(
        "Evidence:             "
        "M030 fine-tuned sentence cross-encoder"
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
        f"{int(positives['pred_bool'].sum())}"
        f"/{len(positives)}"
    )

    print(
        "tIoU among returned positives: "
        f"{returned['scored_tiou'].mean():.4f}"
    )

    print()
    print("REFERENCE")
    print("-" * 78)

    print(
        "M026 composite:          0.6038"
    )

    print(
        "M025 evidence OOF mean:  0.4771"
    )

    print(
        "M030 evidence OOF mean:  0.5089"
    )


if __name__ == "__main__":
    main()