from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(
    r"medical\artifacts\classification"
    r"\m055_base_nli_runtime_signal_oof.csv"
)

PAIRS = Path(
    r"medical\artifacts\analysis"
    r"\m057_question_set_structure.csv"
)

OUTPUT = Path(
    r"medical\artifacts\classification"
    r"\m058_pair_consistency.csv"
)

THRESHOLDS = [
    0.80,
    0.70,
    0.60,
    0.50,
    0.45,
    0.40,
    0.35,
    0.30,
]


def parse_bool(value):
    if isinstance(value, bool):
        return value

    return (
        str(value)
        .strip()
        .lower()
        in {
            "true",
            "1",
            "yes",
        }
    )


def score(
    gold,
    pred,
    evidence,
):
    accuracy = float(
        np.mean(
            gold == pred
        )
    )

    positive = gold

    tiou = float(
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
        + 0.6 * tiou
    )

    return (
        accuracy,
        tiou,
        composite,
    )


def main():
    df = pd.read_csv(
        BASE
    )

    pairs = pd.read_csv(
        PAIRS
    )

    df[
        "gold_bool"
    ] = [
        parse_bool(x)
        for x
        in df[
            "gold_bool"
        ]
    ]

    df[
        "pred"
    ] = [
        parse_bool(x)
        for x
        in df[
            "m055_predicted_yes"
        ]
    ]

    # Confidence = absolute distance from
    # the fold-specific decision threshold.
    df[
        "confidence"
    ] = (
        df[
            "m055_probability"
        ]
        - df[
            "m055_threshold"
        ]
    ).abs()

    lookup = {
        qid: idx
        for idx, qid
        in enumerate(
            df[
                "question_id"
            ]
        )
    }

    gold = (
        df[
            "gold_bool"
        ]
        .to_numpy(
            dtype=bool
        )
    )

    evidence = (
        df[
            "evidence_tiou"
        ]
        .fillna(0.0)
        .to_numpy(
            dtype=float
        )
    )

    baseline = (
        df[
            "pred"
        ]
        .to_numpy(
            dtype=bool
        )
    )

    (
        base_acc,
        base_tiou,
        base_comp,
    ) = score(
        gold,
        baseline,
        evidence,
    )

    print("=" * 78)
    print(
        "Nordic AI Cup 2026 — "
        "M058 Pair-Consistency Correction"
    )
    print("=" * 78)

    print()
    print(
        f"Baseline M055: "
        f"acc={base_acc:.4f}, "
        f"tIoU={base_tiou:.4f}, "
        f"comp={base_comp:.4f}"
    )

    results = []

    for threshold in (
        THRESHOLDS
    ):
        pred = (
            baseline.copy()
        )

        corrections = 0

        considered = 0

        # Highest-similarity relationships
        # first so each correction uses the
        # strongest available pair.
        subset = (
            pairs[
                pairs[
                    "char_similarity"
                ]
                >= threshold
            ]
            .sort_values(
                "char_similarity",
                ascending=False,
            )
        )

        for _, row in (
            subset.iterrows()
        ):
            qa = (
                row[
                    "question_id_a"
                ]
            )

            qb = (
                row[
                    "question_id_b"
                ]
            )

            if (
                qa not in lookup
                or qb not in lookup
            ):
                continue

            ia = lookup[qa]
            ib = lookup[qb]

            considered += 1

            # The structural rule matters
            # only when the classifier gave
            # the pair identical labels.
            if pred[ia] != pred[ib]:
                continue

            ca = float(
                df.iloc[
                    ia
                ][
                    "confidence"
                ]
            )

            cb = float(
                df.iloc[
                    ib
                ][
                    "confidence"
                ]
            )

            # Keep the more confident label;
            # flip the less confident member
            # to enforce opposition.
            if ca >= cb:
                pred[ib] = (
                    not pred[ia]
                )
            else:
                pred[ia] = (
                    not pred[ib]
                )

            corrections += 1

        (
            accuracy,
            tiou,
            composite,
        ) = score(
            gold,
            pred,
            evidence,
        )

        print()
        print(
            f"Similarity >= "
            f"{threshold:.2f}"
        )

        print(
            f"  considered:  "
            f"{considered}"
        )

        print(
            f"  corrections: "
            f"{corrections}"
        )

        print(
            f"  accuracy:    "
            f"{accuracy:.4f}"
        )

        print(
            f"  tIoU:        "
            f"{tiou:.4f}"
        )

        print(
            f"  composite:   "
            f"{composite:.4f}"
        )

        print(
            f"  delta comp:  "
            f"{composite - base_comp:+.4f}"
        )

        results.append(
            {
                "threshold":
                    threshold,

                "considered":
                    considered,

                "corrections":
                    corrections,

                "accuracy":
                    accuracy,

                "scored_tiou":
                    tiou,

                "composite":
                    composite,

                "delta_composite":
                    composite
                    - base_comp,
            }
        )

    output = pd.DataFrame(
        results
    )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        OUTPUT,
        index=False,
    )

    best = (
        output.sort_values(
            "composite",
            ascending=False,
        )
        .iloc[0]
    )

    print()
    print("=" * 78)
    print("BEST M058 RESULT")
    print("=" * 78)

    print(
        f"Threshold:      "
        f"{best.threshold:.2f}"
    )

    print(
        f"Corrections:    "
        f"{int(best.corrections)}"
    )

    print(
        f"Accuracy:       "
        f"{best.accuracy:.4f}"
    )

    print(
        f"Scored tIoU:    "
        f"{best.scored_tiou:.4f}"
    )

    print(
        f"Composite:      "
        f"{best.composite:.4f}"
    )

    print(
        f"Gain vs M055:   "
        f"{best.delta_composite:+.4f}"
    )

    print()
    print(
        f"Saved: {OUTPUT}"
    )


if __name__ == "__main__":
    main()