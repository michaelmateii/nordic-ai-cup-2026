from __future__ import annotations

import re
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


QUESTIONS = Path(
    Path(__file__).resolve().parents[3].parent
    / "Nordic-AI-Cup-2026-official"
    / "medical-appointment"
    / "data"
    / "question_train.csv"
)

OUTPUT = Path(
    r"medical\artifacts\analysis"
    r"\m057_question_set_structure.csv"
)


NEGATION_WORDS = {
    "no",
    "not",
    "never",
    "none",
    "without",
    "deny",
    "denies",
    "denied",
    "free",
    "negative",
    "unchanged",
    "normal",
    "absent",
}


def normalize(text: str) -> str:
    text = str(text).lower()

    text = re.sub(
        r"\d+(?:\.\d+)?",
        "<NUM>",
        text,
    )

    text = re.sub(
        r"[^a-z0-9<>]+",
        " ",
        text,
    )

    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


def tokens(text: str) -> set[str]:
    return set(
        re.findall(
            r"\b[a-z0-9<>]+\b",
            normalize(text),
        )
    )


def negation_count(text: str) -> int:
    ts = tokens(text)

    return sum(
        word in ts
        for word in NEGATION_WORDS
    )


def main():
    df = pd.read_csv(
        QUESTIONS
    )

    if len(df) != 390:
        raise RuntimeError(
            f"Expected 390 rows, got {len(df)}"
        )

    df["gold_bool"] = (
        df["answer"]
        .astype(str)
        .str.lower()
        .eq("yes")
    )

    rows = []

    print("=" * 78)
    print(
        "Nordic AI Cup 2026 — "
        "M057 Intra-Conversation Question-Set Audit"
    )
    print("=" * 78)

    for transcript_id, group in df.groupby(
        "transcript_id",
        sort=False,
    ):
        group = group.reset_index(
            drop=True
        )

        texts = (
            group["question"]
            .astype(str)
            .tolist()
        )

        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            sublinear_tf=True,
        )

        matrix = vectorizer.fit_transform(
            texts
        )

        similarities = cosine_similarity(
            matrix
        )

        for i, j in combinations(
            range(len(group)),
            2,
        ):
            qi = str(
                group.loc[
                    i,
                    "question"
                ]
            )

            qj = str(
                group.loc[
                    j,
                    "question"
                ]
            )

            ti = tokens(qi)
            tj = tokens(qj)

            union = (
                ti | tj
            )

            jaccard = (
                len(ti & tj)
                / len(union)
                if union
                else 0.0
            )

            ni = negation_count(
                qi
            )

            nj = negation_count(
                qj
            )

            gold_i = bool(
                group.loc[
                    i,
                    "gold_bool"
                ]
            )

            gold_j = bool(
                group.loc[
                    j,
                    "gold_bool"
                ]
            )

            rows.append(
                {
                    "transcript_id":
                        transcript_id,

                    "question_id_a":
                        group.loc[
                            i,
                            "question_id"
                        ],

                    "question_id_b":
                        group.loc[
                            j,
                            "question_id"
                        ],

                    "question_a": qi,
                    "question_b": qj,

                    "gold_a": gold_i,
                    "gold_b": gold_j,

                    "same_label": (
                        gold_i
                        == gold_j
                    ),

                    "opposite_label": (
                        gold_i
                        != gold_j
                    ),

                    "char_similarity":
                        float(
                            similarities[
                                i,
                                j,
                            ]
                        ),

                    "token_jaccard":
                        float(
                            jaccard
                        ),

                    "negation_a": ni,
                    "negation_b": nj,

                    "negation_difference":
                        abs(
                            ni - nj
                        ),
                }
            )

    pairs = pd.DataFrame(
        rows
    )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pairs.to_csv(
        OUTPUT,
        index=False,
    )

    print(
        f"Conversations: "
        f"{df.transcript_id.nunique()}"
    )

    print(
        f"Question pairs: "
        f"{len(pairs)}"
    )

    print()

    for threshold in [
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
    ]:
        subset = pairs[
            pairs[
                "char_similarity"
            ]
            >= threshold
        ]

        if len(subset) == 0:
            continue

        print(
            f"Similarity >= "
            f"{threshold:.2f}"
        )

        print(
            f"  pairs: "
            f"{len(subset)}"
        )

        print(
            f"  opposite labels: "
            f"{subset.opposite_label.mean():.4f}"
        )

        print(
            f"  same labels:     "
            f"{subset.same_label.mean():.4f}"
        )

        print(
            f"  negation differs:"
            f" "
            f"{(subset.negation_difference > 0).mean():.4f}"
        )

    print()
    print("=" * 78)
    print(
        "HIGH-SIMILARITY OPPOSITE-LABEL PAIRS"
    )
    print("=" * 78)

    interesting = (
        pairs[
            pairs[
                "opposite_label"
            ]
        ]
        .sort_values(
            [
                "char_similarity",
                "token_jaccard",
            ],
            ascending=False,
        )
        .head(60)
    )

    print(
        interesting[
            [
                "transcript_id",
                "question_id_a",
                "question_id_b",
                "gold_a",
                "gold_b",
                "char_similarity",
                "token_jaccard",
                "negation_a",
                "negation_b",
                "question_a",
                "question_b",
            ]
        ]
        .to_string(
            index=False
        )
    )

    print()
    print("=" * 78)
    print(
        "HIGH-SIMILARITY SAME-LABEL PAIRS"
    )
    print("=" * 78)

    same = (
        pairs[
            pairs[
                "same_label"
            ]
        ]
        .sort_values(
            [
                "char_similarity",
                "token_jaccard",
            ],
            ascending=False,
        )
        .head(30)
    )

    print(
        same[
            [
                "transcript_id",
                "question_id_a",
                "question_id_b",
                "gold_a",
                "char_similarity",
                "token_jaccard",
                "question_a",
                "question_b",
            ]
        ]
        .to_string(
            index=False
        )
    )

    print()
    print(
        f"Saved: {OUTPUT}"
    )


if __name__ == "__main__":
    main()