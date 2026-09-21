from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_QUESTION_CSV = Path(
    Path(__file__).resolve().parents[3].parent
    / "Nordic-AI-Cup-2026-official"
    / "medical-appointment"
    / "data"
    / "question_train.csv"
)

DEFAULT_ASR_DIR = Path(
    r"medical\artifacts\asr\faster_distil_large_v3_int8f32_b1"
)

DEFAULT_OUTPUT = Path(
    r"medical\artifacts\retrieval\word_timestamp_ceiling.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--questions",
        type=Path,
        default=DEFAULT_QUESTION_CSV,
    )

    parser.add_argument(
        "--asr-dir",
        type=Path,
        default=DEFAULT_ASR_DIR,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    return parser.parse_args()


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


def normalize_transcript_id(value: object) -> str:
    text = Path(str(value).strip()).stem

    if text.startswith("conversation_sample_"):
        return text

    if text.startswith("sample_"):
        return f"conversation_{text}"

    if text.isdigit():
        return f"conversation_sample_{int(text)}"

    return text


def load_words(
    asr_dir: Path,
    transcript_id: object,
) -> list[dict]:
    stem = normalize_transcript_id(transcript_id)

    path = asr_dir / f"{stem}.json"

    if not path.exists():
        raise FileNotFoundError(path)

    data = json.loads(
        path.read_text(encoding="utf-8")
    )

    words = []

    for word in data["words"]:
        start = word.get("start")
        end = word.get("end")

        if start is None or end is None:
            continue

        words.append(
            {
                "start": float(start),
                "end": float(end),
                "word": str(word["word"]),
            }
        )

    return words


def best_word_window(
    words: list[dict],
    gold_start: float,
    gold_end: float,
) -> tuple[float, int, int]:
    """
    Exhaustively search contiguous word windows.

    This uses gold evidence and is ONLY an oracle diagnostic.
    """

    best_iou = 0.0
    best_start_index = -1
    best_end_index = -1

    n = len(words)

    # Only windows reasonably near the gold evidence need
    # to be considered. This keeps the diagnostic fast.
    candidates = [
        i
        for i, word in enumerate(words)
        if (
            word["end"] >= gold_start - 5.0
            and word["start"] <= gold_end + 5.0
        )
    ]

    if not candidates:
        candidates = list(range(n))

    min_index = max(0, min(candidates) - 3)
    max_index = min(n - 1, max(candidates) + 3)

    for start_index in range(
        min_index,
        max_index + 1,
    ):
        pred_start = words[start_index]["start"]

        for end_index in range(
            start_index,
            max_index + 1,
        ):
            pred_end = words[end_index]["end"]

            iou = temporal_iou(
                pred_start,
                pred_end,
                gold_start,
                gold_end,
            )

            if iou > best_iou:
                best_iou = iou
                best_start_index = start_index
                best_end_index = end_index

    return (
        best_iou,
        best_start_index,
        best_end_index,
    )


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.questions)

    positives = df[
        df["question_type"] == "positive"
    ].copy()

    cache: dict[str, list[dict]] = {}
    results = []

    for _, row in positives.iterrows():
        transcript_id = row["transcript_id"]
        stem = normalize_transcript_id(
            transcript_id
        )

        if stem not in cache:
            cache[stem] = load_words(
                args.asr_dir,
                transcript_id,
            )

        words = cache[stem]

        gold_start = float(
            row["evidence_start"]
        )
        gold_end = float(
            row["evidence_end"]
        )

        (
            best_iou,
            start_index,
            end_index,
        ) = best_word_window(
            words,
            gold_start,
            gold_end,
        )

        if start_index >= 0:
            pred_start = words[
                start_index
            ]["start"]

            pred_end = words[
                end_index
            ]["end"]

            pred_text = "".join(
                word["word"]
                for word in words[
                    start_index : end_index + 1
                ]
            ).strip()
        else:
            pred_start = np.nan
            pred_end = np.nan
            pred_text = ""

        results.append(
            {
                "question_id": row[
                    "question_id"
                ],
                "transcript_id": transcript_id,
                "question": row["question"],
                "gold_start": gold_start,
                "gold_end": gold_end,
                "pred_start": pred_start,
                "pred_end": pred_end,
                "pred_text": pred_text,
                "oracle_word_tiou": best_iou,
            }
        )

    result_df = pd.DataFrame(results)

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_df.to_csv(
        args.output,
        index=False,
    )

    scores = result_df[
        "oracle_word_tiou"
    ]

    print("=" * 78)
    print(
        "Nordic AI Cup 2026 — "
        "Word Timestamp Localization Ceiling"
    )
    print("=" * 78)

    print(
        f"Gold-positive questions: "
        f"{len(result_df)}"
    )

    print()
    print("WORD-TIMESTAMP ORACLE")
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
        f"Minimum tIoU:    "
        f"{scores.min():.4f}"
    )

    print(
        f"tIoU >= 0.50:    "
        f"{(scores >= 0.50).mean():.4f}"
    )

    print(
        f"tIoU >= 0.75:    "
        f"{(scores >= 0.75).mean():.4f}"
    )

    print(
        f"tIoU >= 0.90:    "
        f"{(scores >= 0.90).mean():.4f}"
    )

    print()

    print(
        "Previous oracle single-segment "
        "mean tIoU: 0.6117"
    )

    print(
        f"Word-window improvement: "
        f"{scores.mean() - 0.6117:+.4f}"
    )

    print()
    print(
        f"Detailed results: {args.output}"
    )


if __name__ == "__main__":
    main()