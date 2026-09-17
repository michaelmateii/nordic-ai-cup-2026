from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_CLAIMS = Path(
    r"medical\artifacts\classification\nli_claim_results.csv"
)

DEFAULT_THRESHOLDS = Path(
    r"medical\artifacts\classification\nli_claim_threshold_cv_results.csv"
)

DEFAULT_EVIDENCE = Path(
    r"medical\artifacts\retrieval\cross_encoder_segment_results.csv"
)

DEFAULT_OUTPUT = Path(
    r"medical\artifacts\evaluation\composite_dev_score.csv"
)

METHOD = "entailment_ratio"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--claims",
        type=Path,
        default=DEFAULT_CLAIMS,
    )

    parser.add_argument(
        "--thresholds",
        type=Path,
        default=DEFAULT_THRESHOLDS,
    )

    parser.add_argument(
        "--evidence",
        type=Path,
        default=DEFAULT_EVIDENCE,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    return parser.parse_args()


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
    args = parse_args()

    claims = pd.read_csv(
        args.claims
    )

    thresholds = pd.read_csv(
        args.thresholds
    )

    evidence = pd.read_csv(
        args.evidence
    )

    calibrated = thresholds[
        thresholds["method"] == METHOD
    ].copy()

    if len(calibrated) != len(claims):
        raise RuntimeError(
            f"Expected {len(claims)} calibrated rows, "
            f"got {len(calibrated)}"
        )

    calibrated["row_index"] = (
        calibrated["row_index"].astype(int)
    )

    calibrated = calibrated.sort_values(
        "row_index"
    )

    if not np.array_equal(
        calibrated["row_index"].to_numpy(),
        np.arange(len(claims)),
    ):
        raise RuntimeError(
            "Calibration row indices do not map "
            "one-to-one to claim results."
        )

    claims = claims.reset_index(
        drop=True
    )

    claims["oof_predicted_yes"] = [
        parse_bool(value)
        for value in calibrated[
            "predicted_yes"
        ]
    ]

    claims["gold_yes_parsed"] = [
        parse_bool(value)
        for value in claims[
            "gold_yes"
        ]
    ]

    if int(
        claims[
            "gold_yes_parsed"
        ].sum()
    ) != 195:
        raise RuntimeError(
            "Expected exactly 195 gold-positive questions."
        )

    accuracy = float(
        (
            claims["oof_predicted_yes"]
            == claims["gold_yes_parsed"]
        ).mean()
    )

    # Evidence table contains only gold-positive questions.
    evidence_by_question = evidence.set_index(
        "question_id"
    )

    positive_rows = claims[
        claims["gold_yes_parsed"]
    ].copy()

    scored_tious = []

    missing = []

    for _, row in positive_rows.iterrows():
        question_id = row[
            "question_id"
        ]

        if not row[
            "oof_predicted_yes"
        ]:
            scored_tious.append(
                0.0
            )
            continue

        if (
            question_id
            not in evidence_by_question.index
        ):
            missing.append(
                question_id
            )
            scored_tious.append(
                0.0
            )
            continue

        evidence_row = (
            evidence_by_question.loc[
                question_id
            ]
        )

        scored_tious.append(
            float(
                evidence_row["tiou"]
            )
        )

    if missing:
        raise RuntimeError(
            f"Missing evidence results for "
            f"{len(missing)} questions: "
            f"{missing[:10]}"
        )

    mean_tiou = float(
        np.mean(scored_tious)
    )

    final_score = (
        0.4 * accuracy
        + 0.6 * mean_tiou
    )

    positive_rows[
        "scored_tiou"
    ] = scored_tious

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    positive_rows.to_csv(
        args.output,
        index=False,
    )

    print("=" * 78)
    print(
        "Nordic AI Cup 2026 — "
        "Composite Development Score"
    )
    print("=" * 78)

    print(
        f"Classification method: "
        f"{METHOD}"
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
        f"{final_score:.4f}"
    )

    print()

    print(
        "Positive questions predicted YES: "
        f"{positive_rows['oof_predicted_yes'].sum()}"
        f"/{len(positive_rows)}"
    )

    returned = positive_rows[
        positive_rows[
            "oof_predicted_yes"
        ]
    ]

    if len(returned):
        print(
            "tIoU among returned positive spans: "
            f"{returned['scored_tiou'].mean():.4f}"
        )

    print()

    print(
        f"Detailed results: "
        f"{args.output}"
    )


if __name__ == "__main__":
    main()