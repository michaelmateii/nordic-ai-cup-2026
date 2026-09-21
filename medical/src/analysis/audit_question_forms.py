from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd


CSV_PATH = Path(
    Path(__file__).resolve().parents[3].parent
    / "Nordic-AI-Cup-2026-official"
    / "medical-appointment"
    / "data"
    / "question_train.csv"
)


def main() -> None:
    df = pd.read_csv(CSV_PATH)

    questions = df["question"].astype(str)

    first_word = Counter()
    first_two = Counter()
    first_three = Counter()

    for question in questions:
        tokens = (
            question.strip()
            .rstrip("?")
            .split()
        )

        if not tokens:
            continue

        first_word[
            tokens[0].lower()
        ] += 1

        first_two[
            " ".join(
                token.lower()
                for token in tokens[:2]
            )
        ] += 1

        first_three[
            " ".join(
                token.lower()
                for token in tokens[:3]
            )
        ] += 1

    print("=" * 78)
    print("QUESTION FORM AUDIT")
    print("=" * 78)

    print()
    print("FIRST WORD")
    print("-" * 78)

    for value, count in (
        first_word.most_common()
    ):
        print(
            f"{value:20s} {count:3d}"
        )

    print()
    print("TOP FIRST TWO WORDS")
    print("-" * 78)

    for value, count in (
        first_two.most_common(40)
    ):
        print(
            f"{value:35s} {count:3d}"
        )

    print()
    print("TOP FIRST THREE WORDS")
    print("-" * 78)

    for value, count in (
        first_three.most_common(50)
    ):
        print(
            f"{value:50s} {count:3d}"
        )

    print()
    print("EXAMPLES BY FIRST WORD")
    print("-" * 78)

    for starter, count in (
        first_word.most_common()
    ):
        subset = df[
            questions.str.lower().str.startswith(
                starter + " "
            )
        ]

        print()
        print(
            f"[{starter}] n={count}"
        )

        for _, row in subset.head(
            8
        ).iterrows():
            print(
                f"  {row['question_type']:13s} "
                f"{row['question']}"
            )


if __name__ == "__main__":
    main()