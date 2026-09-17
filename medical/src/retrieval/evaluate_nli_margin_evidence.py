from __future__ import annotations

from pathlib import Path

import pandas as pd


QUESTIONS = Path(
    r"C:\Users\Calle\Projects\Nordic-AI-Cup-2026-official"
    r"\medical-appointment\data\question_train.csv"
)

NLI_RESULTS = Path(
    r"medical\artifacts\classification\nli_segmentwise_results.csv"
)

OUTPUT = Path(
    r"medical\artifacts\retrieval\nli_margin_segment_evidence.csv"
)


def temporal_iou(
    pred_start: float,
    pred_end: float,
    gold_start: float,
    gold_end: float,
) -> float:
    intersection = max(
        0.0,
        min(pred_end, gold_end)
        - max(pred_start, gold_start),
    )

    union = (
        max(pred_end, gold_end)
        - min(pred_start, gold_start)
    )

    if union <= 0:
        return 0.0

    return intersection / union


def main() -> None:
    questions = pd.read_csv(
        QUESTIONS
    )

    nli = pd.read_csv(
        NLI_RESULTS
    )

    positives = questions[
        questions["question_type"]
        == "positive"
    ].copy()

    merged = positives.merge(
        nli[
            [
                "question_id",
                "margin_selected_start",
                "margin_selected_end",
                "margin_selected_text",
                "max_segment_margin",
            ]
        ],
        on="question_id",
        how="left",
        validate="one_to_one",
    )

    if len(merged) != 195:
        raise RuntimeError(
            f"Expected 195 positives, got {len(merged)}"
        )

    if merged[
        "margin_selected_start"
    ].isna().any():
        raise RuntimeError(
            "Missing NLI evidence spans."
        )

    merged["tiou"] = [
        temporal_iou(
            float(row.margin_selected_start),
            float(row.margin_selected_end),
            float(row.evidence_start),
            float(row.evidence_end),
        )
        for row in merged.itertuples()
    ]

    scores = merged["tiou"]

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    merged.to_csv(
        OUTPUT,
        index=False,
    )

    print("=" * 78)
    print(
        "Nordic AI Cup 2026 — "
        "NLI Margin Segment Evidence"
    )
    print("=" * 78)

    print(
        f"Positive questions: {len(merged)}"
    )

    print()
    print(
        f"Mean tIoU:       {scores.mean():.4f}"
    )

    print(
        f"Median tIoU:     {scores.median():.4f}"
    )

    print(
        f"Any overlap:     {(scores > 0).mean():.4f}"
    )

    print(
        f"tIoU >= 0.25:    {(scores >= 0.25).mean():.4f}"
    )

    print(
        f"tIoU >= 0.50:    {(scores >= 0.50).mean():.4f}"
    )

    print(
        f"tIoU >= 0.75:    {(scores >= 0.75).mean():.4f}"
    )

    print()
    print("REFERENCE")
    print("-" * 78)
    print(
        "M010 cross-encoder segment mean: 0.4347"
    )
    print(
        "M007 word oracle mean:            0.9294"
    )

    print()
    print(
        f"Detailed results: {OUTPUT}"
    )


if __name__ == "__main__":
    main()