from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


HGB = Path(
    r"medical\artifacts\retrieval\supervised_sentence_ranker_results.csv"
)

NEURAL = Path(
    r"medical\artifacts\retrieval\finetuned_sentence_ranker_oof.csv"
)

OUTPUT = Path(
    r"medical\artifacts\analysis\evidence_ranker_complementarity.csv"
)


def main() -> None:
    hgb = pd.read_csv(HGB)

    neural = pd.read_csv(NEURAL)

    required_hgb = {
        "question_id",
        "candidate_start",
        "candidate_end",
        "target_tiou",
    }

    required_neural = {
        "question_id",
        "candidate_start",
        "candidate_end",
        "target_tiou",
    }

    if not required_hgb.issubset(hgb.columns):
        raise RuntimeError(
            "M025 CSV missing columns: "
            f"{required_hgb - set(hgb.columns)}"
        )

    if not required_neural.issubset(
        neural.columns
    ):
        raise RuntimeError(
            "M030 CSV missing columns: "
            f"{required_neural - set(neural.columns)}"
        )

    hgb = hgb[
        [
            "question_id",
            "candidate_start",
            "candidate_end",
            "target_tiou",
        ]
    ].rename(
        columns={
            "candidate_start": "hgb_start",
            "candidate_end": "hgb_end",
            "target_tiou": "hgb_tiou",
        }
    )

    neural = neural[
        [
            "question_id",
            "candidate_start",
            "candidate_end",
            "target_tiou",
        ]
    ].rename(
        columns={
            "candidate_start": "neural_start",
            "candidate_end": "neural_end",
            "target_tiou": "neural_tiou",
        }
    )

    merged = hgb.merge(
        neural,
        on="question_id",
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != 195:
        raise RuntimeError(
            f"Expected 195 questions, got {len(merged)}"
        )

    merged["same_span"] = (
        np.isclose(
            merged["hgb_start"],
            merged["neural_start"],
            atol=1e-6,
        )
        & np.isclose(
            merged["hgb_end"],
            merged["neural_end"],
            atol=1e-6,
        )
    )

    merged["neural_minus_hgb"] = (
        merged["neural_tiou"]
        - merged["hgb_tiou"]
    )

    merged["winner"] = np.where(
        merged["neural_tiou"]
        > merged["hgb_tiou"] + 1e-9,
        "neural",
        np.where(
            merged["hgb_tiou"]
            > merged["neural_tiou"] + 1e-9,
            "hgb",
            "tie",
        ),
    )

    merged["oracle_two_tiou"] = (
        merged[
            [
                "hgb_tiou",
                "neural_tiou",
            ]
        ].max(axis=1)
    )

    merged["average_two_tiou"] = (
        (
            merged["hgb_tiou"]
            + merged["neural_tiou"]
        )
        / 2.0
    )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    merged.to_csv(
        OUTPUT,
        index=False,
    )

    winner_counts = (
        merged["winner"]
        .value_counts()
    )

    print("=" * 78)
    print(
        "Nordic AI Cup 2026 — "
        "M032A Evidence Ranker Complementarity"
    )
    print("=" * 78)

    print()
    print("INDIVIDUAL")
    print("-" * 78)

    print(
        f"M025 HGB mean tIoU:       "
        f"{merged['hgb_tiou'].mean():.4f}"
    )

    print(
        f"M030 neural mean tIoU:    "
        f"{merged['neural_tiou'].mean():.4f}"
    )

    print()
    print("COMPLEMENTARITY")
    print("-" * 78)

    print(
        f"Same selected span:        "
        f"{merged['same_span'].mean():.4f} "
        f"({int(merged['same_span'].sum())}/195)"
    )

    print(
        f"Neural wins:               "
        f"{winner_counts.get('neural', 0)}"
    )

    print(
        f"HGB wins:                  "
        f"{winner_counts.get('hgb', 0)}"
    )

    print(
        f"Ties:                      "
        f"{winner_counts.get('tie', 0)}"
    )

    print()
    print(
        f"Mean neural-HGB delta:     "
        f"{merged['neural_minus_hgb'].mean():+.4f}"
    )

    print()
    print("ORACLE OF THE TWO CHOICES")
    print("-" * 78)

    oracle = merged[
        "oracle_two_tiou"
    ]

    print(
        f"Mean tIoU:                 "
        f"{oracle.mean():.4f}"
    )

    print(
        f"Median tIoU:               "
        f"{oracle.median():.4f}"
    )

    print(
        f"tIoU >= 0.50:              "
        f"{(oracle >= 0.50).mean():.4f}"
    )

    print(
        f"tIoU >= 0.75:              "
        f"{(oracle >= 0.75).mean():.4f}"
    )

    print()
    print(
        f"Oracle gain over M030:     "
        f"{oracle.mean() - merged['neural_tiou'].mean():+.4f}"
    )

    print()
    print("REFERENCE")
    print("-" * 78)

    print(
        "M023 all-candidate oracle:  0.7997"
    )

    print()
    print(
        f"Detailed results: {OUTPUT}"
    )


if __name__ == "__main__":
    main()