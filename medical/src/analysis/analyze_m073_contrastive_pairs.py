from pathlib import Path

import pandas as pd


QUESTIONS = Path(
    Path(__file__).resolve().parents[3].parent
    / "Nordic-AI-Cup-2026-official"
    / "medical-appointment"
    / "data"
    / "question_train.csv"
)

CANDIDATES = Path(
    r"medical\artifacts\retrieval"
    r"\m047_all_questions_candidates_oof.csv"
)


def main():
    q = pd.read_csv(QUESTIONS)
    c = pd.read_csv(CANDIDATES)

    # Best available candidate corresponding to each positive's
    # annotated evidence.
    positives = q[
        q["question_type"] == "positive"
    ].copy()

    rows = []

    for _, question in positives.iterrows():
        subset = c[
            c["question_id"]
            == question["question_id"]
        ]

        if subset.empty:
            continue

        best = subset.loc[
            subset["target_tiou"].idxmax()
        ]

        rows.append(
            {
                "question_id":
                    question["question_id"],

                "transcript_id":
                    question["transcript_id"],

                "question":
                    question["question"],

                "evidence_text":
                    best["candidate_text"],

                "oracle_tiou":
                    float(
                        best["target_tiou"]
                    ),
            }
        )

    evidence = pd.DataFrame(rows)

    pair_rows = []

    # Pair EVERY question with every positive evidence span
    # from the same transcript.
    for transcript_id, group in q.groupby(
        "transcript_id"
    ):
        ev = evidence[
            evidence["transcript_id"]
            == transcript_id
        ]

        for _, question in group.iterrows():
            for _, span in ev.iterrows():

                own = (
                    question["question_id"]
                    == span["question_id"]
                )

                target = (
                    1
                    if (
                        question["question_type"]
                        == "positive"
                        and own
                    )
                    else 0
                )

                pair_rows.append(
                    {
                        "transcript_id":
                            transcript_id,

                        "question_id":
                            question[
                                "question_id"
                            ],

                        "question_type":
                            question[
                                "question_type"
                            ],

                        "question":
                            question[
                                "question"
                            ],

                        "evidence_question_id":
                            span[
                                "question_id"
                            ],

                        "evidence_text":
                            span[
                                "evidence_text"
                            ],

                        "target":
                            target,

                        "own_evidence":
                            own,
                    }
                )

    pairs = pd.DataFrame(pair_rows)

    print("=" * 78)
    print("M073 CONTRASTIVE PAIR AUDIT")
    print("=" * 78)

    print(
        "Positive evidence spans:",
        len(evidence),
    )

    print(
        "Total same-conversation pairs:",
        len(pairs),
    )

    print()
    print("TARGETS")
    print(pairs["target"].value_counts())

    print()
    print("NEGATIVES BY QUESTION TYPE")
    print(
        pairs[
            pairs["target"] == 0
        ]["question_type"]
        .value_counts()
    )

    print()
    print("PAIRS / TRANSCRIPT")
    print(
        pairs.groupby(
            "transcript_id"
        ).size().describe()
    )

    print()
    print("EXAMPLE HARD NEGATIVES")
    print("-" * 78)

    sample = (
        pairs[
            (pairs["target"] == 0)
            & (
                pairs["question_type"]
                == "hard_negative"
            )
        ]
        .head(20)
    )

    for _, row in sample.iterrows():
        print()
        print(
            "Q:",
            row["question"],
        )
        print(
            "E:",
            row["evidence_text"],
        )

    output = Path(
        r"medical\artifacts\analysis"
        r"\m073_contrastive_pair_audit.csv"
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pairs.to_csv(
        output,
        index=False,
    )

    print()
    print("Saved:", output)


if __name__ == "__main__":
    main()
    