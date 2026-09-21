from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


PAIRS = Path(
    r"medical\artifacts\analysis"
    r"\m057_question_set_structure.csv"
)

OUTPUT = Path(
    r"medical\artifacts\analysis"
    r"\m059_transformation_pairs.csv"
)


OPPOSITES = [
    ("stable", "unstable"),
    ("normal", "abnormal"),
    ("present", "absent"),
    ("continue", "stop"),
    ("continued", "stopped"),
    ("renewed", "stopped"),
    ("viral", "bacterial"),
    ("high", "low"),
    ("positive", "negative"),
    ("harmless", "suspicious"),
    ("constant", "come and go"),
    ("regularly", "rarely"),
    ("free of", "has"),
    ("free of", "develop"),
    ("deny", "report"),
    ("denies", "reports"),
    ("ruled out", "found"),
    ("no treatment", "treatment"),
    ("no risk", "risk"),
]


def normalize(text: str) -> str:
    text = str(text).lower()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def numbers(text: str):
    return re.findall(
        r"\d+(?:\.\d+)?",
        str(text),
    )


def detect_numeric_substitution(
    a: str,
    b: str,
):
    na = numbers(a)
    nb = numbers(b)

    if not na or not nb:
        return False

    return (
        na != nb
        and len(na) == len(nb)
    )


def detect_opposite_terms(
    a: str,
    b: str,
):
    a = normalize(a)
    b = normalize(b)

    found = []

    for left, right in OPPOSITES:
        if (
            left in a
            and right in b
        ):
            found.append(
                f"{left}<->{right}"
            )

        elif (
            right in a
            and left in b
        ):
            found.append(
                f"{left}<->{right}"
            )

    return found


def main():
    df = pd.read_csv(
        PAIRS
    )

    rows = []

    for _, row in df.iterrows():
        qa = str(
            row[
                "question_a"
            ]
        )

        qb = str(
            row[
                "question_b"
            ]
        )

        numeric = (
            detect_numeric_substitution(
                qa,
                qb,
            )
        )

        opposites = (
            detect_opposite_terms(
                qa,
                qb,
            )
        )

        transformation = (
            numeric
            or bool(opposites)
        )

        rows.append(
            {
                **row.to_dict(),

                "numeric_substitution":
                    numeric,

                "opposite_terms":
                    "|".join(
                        opposites
                    ),

                "transformation_detected":
                    transformation,
            }
        )

    out = pd.DataFrame(
        rows
    )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out.to_csv(
        OUTPUT,
        index=False,
    )

    transformed = out[
        out[
            "transformation_detected"
        ]
    ]

    print("=" * 78)
    print(
        "Nordic AI Cup 2026 — "
        "M059 Transformation Pair Audit"
    )
    print("=" * 78)

    print(
        f"All pairs: "
        f"{len(out)}"
    )

    print(
        f"Detected transformations: "
        f"{len(transformed)}"
    )

    if len(transformed):
        print(
            f"Opposite-label rate: "
            f"{transformed.opposite_label.mean():.4f}"
        )

        print(
            f"Same-label rate:     "
            f"{transformed.same_label.mean():.4f}"
        )

        numeric = transformed[
            transformed[
                "numeric_substitution"
            ]
        ]

        if len(numeric):
            print()
            print(
                "NUMERIC SUBSTITUTIONS"
            )

            print(
                f"pairs: {len(numeric)}"
            )

            print(
                f"opposite-label rate: "
                f"{numeric.opposite_label.mean():.4f}"
            )

        semantic = transformed[
            transformed[
                "opposite_terms"
            ].astype(str)
            != ""
        ]

        if len(semantic):
            print()
            print(
                "SEMANTIC OPPOSITES"
            )

            print(
                f"pairs: {len(semantic)}"
            )

            print(
                f"opposite-label rate: "
                f"{semantic.opposite_label.mean():.4f}"
            )

    print()
    print(
        "TOP DETECTED PAIRS"
    )
    print("-" * 78)

    print(
        transformed[
            [
                "transcript_id",
                "question_id_a",
                "question_id_b",
                "gold_a",
                "gold_b",
                "char_similarity",
                "numeric_substitution",
                "opposite_terms",
                "question_a",
                "question_b",
            ]
        ]
        .sort_values(
            "char_similarity",
            ascending=False,
        )
        .head(80)
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