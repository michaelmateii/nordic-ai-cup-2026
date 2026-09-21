from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


BASE = Path(
    r"medical\artifacts\classification"
    r"\m055_base_nli_runtime_signal_oof.csv"
)

QUESTIONS = Path(
    Path(__file__).resolve().parents[3].parent
    / "Nordic-AI-Cup-2026-official"
    / "medical-appointment"
    / "data"
    / "question_train.csv"
)

OUTPUT = Path(
    r"medical\artifacts\classification"
    r"\m060_precision_constraints.csv"
)


SEMANTIC_OPPOSITES = [
    ("stable", "unstable"),
    ("normal", "abnormal"),
    ("continue", "stop"),
    ("continued", "stopped"),
    ("renewed", "stopped"),
    ("viral", "bacterial"),
    ("positive", "negative"),
    ("harmless", "suspicious"),
    ("constant", "come and go"),
    ("regularly", "rarely"),
    ("deny", "report"),
    ("denies", "reports"),
    ("ruled out", "found"),
    ("absent", "seen"),
    ("absent", "present"),
]


CHAR_THRESHOLDS = [
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
]

NUMERIC_NORMALIZED_THRESHOLD = 0.72


def parse_bool(value):
    if isinstance(value, bool):
        return value

    return (
        str(value)
        .strip()
        .lower()
        in {"true", "1", "yes"}
    )


def normalize(text: str) -> str:
    text = str(text).lower()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def normalize_numbers(text: str) -> str:
    text = normalize(text)

    text = re.sub(
        r"\d+(?:\.\d+)?",
        "<NUM>",
        text,
    )

    return text


def extract_numbers(text: str):
    return re.findall(
        r"\d+(?:\.\d+)?",
        str(text),
    )


def char_similarity(
    a: str,
    b: str,
) -> float:
    if not a or not b:
        return 0.0

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        sublinear_tf=True,
    )

    matrix = vectorizer.fit_transform(
        [a, b]
    )

    return float(
        cosine_similarity(
            matrix[0],
            matrix[1],
        )[0, 0]
    )


def detect_numeric_pair(
    a: str,
    b: str,
):
    na = extract_numbers(a)
    nb = extract_numbers(b)

    if (
        not na
        or not nb
        or len(na) != len(nb)
        or na == nb
    ):
        return False

    # Questions must be essentially the same
    # proposition once numeric values are removed.
    similarity = char_similarity(
        normalize_numbers(a),
        normalize_numbers(b),
    )

    return (
        similarity
        >= NUMERIC_NORMALIZED_THRESHOLD
    )


def detect_semantic_opposite(
    a: str,
    b: str,
):
    a = normalize(a)
    b = normalize(b)

    for left, right in (
        SEMANTIC_OPPOSITES
    ):
        if (
            left in a
            and right in b
        ):
            return True

        if (
            right in a
            and left in b
        ):
            return True

    return False


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
    base = pd.read_csv(
        BASE
    )

    questions = pd.read_csv(
        QUESTIONS
    )

    base["gold_bool"] = [
        parse_bool(x)
        for x
        in base["gold_bool"]
    ]

    base["pred"] = [
        parse_bool(x)
        for x
        in base[
            "m055_predicted_yes"
        ]
    ]

    base["confidence"] = (
        base["m055_probability"]
        - base["m055_threshold"]
    ).abs()

    qmap = (
        questions.set_index(
            "question_id"
        )["question"]
        .astype(str)
        .to_dict()
    )

    lookup = {
        qid: i
        for i, qid
        in enumerate(
            base["question_id"]
        )
    }

    gold = (
        base["gold_bool"]
        .to_numpy(dtype=bool)
    )

    evidence = (
        base["evidence_tiou"]
        .fillna(0.0)
        .to_numpy(dtype=float)
    )

    baseline = (
        base["pred"]
        .to_numpy(dtype=bool)
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

    # Build only same-conversation pairs.
    rows = []

    for transcript_id, group in (
        questions.groupby(
            "transcript_id",
            sort=False,
        )
    ):
        ids = (
            group["question_id"]
            .tolist()
        )

        for i in range(
            len(ids)
        ):
            for j in range(
                i + 1,
                len(ids),
            ):
                qa = ids[i]
                qb = ids[j]

                text_a = qmap[qa]
                text_b = qmap[qb]

                raw_sim = (
                    char_similarity(
                        text_a,
                        text_b,
                    )
                )

                numeric = (
                    detect_numeric_pair(
                        text_a,
                        text_b,
                    )
                )

                semantic = (
                    detect_semantic_opposite(
                        text_a,
                        text_b,
                    )
                )

                if (
                    numeric
                    or semantic
                ):
                    rows.append(
                        {
                            "question_id_a":
                                qa,

                            "question_id_b":
                                qb,

                            "char_similarity":
                                raw_sim,

                            "numeric_pair":
                                numeric,

                            "semantic_pair":
                                semantic,
                        }
                    )

    pairs = pd.DataFrame(
        rows
    )

    print("=" * 78)
    print(
        "Nordic AI Cup 2026 — "
        "M060 Precision Transformation Constraints"
    )
    print("=" * 78)

    print(
        f"Detected candidate pairs: "
        f"{len(pairs)}"
    )

    print(
        f"Baseline: "
        f"acc={base_acc:.4f}, "
        f"tIoU={base_tiou:.4f}, "
        f"comp={base_comp:.4f}"
    )

    results = []

    for threshold in (
        CHAR_THRESHOLDS
    ):
        pred = (
            baseline.copy()
        )

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

        corrections = 0

        for _, row in (
            subset.iterrows()
        ):
            qa = row[
                "question_id_a"
            ]

            qb = row[
                "question_id_b"
            ]

            ia = lookup[qa]
            ib = lookup[qb]

            # Already opposite => constraint
            # satisfied.
            if pred[ia] != pred[ib]:
                continue

            ca = float(
                base.iloc[
                    ia
                ]["confidence"]
            )

            cb = float(
                base.iloc[
                    ib
                ]["confidence"]
            )

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

        results.append(
            {
                "threshold":
                    threshold,

                "pairs":
                    len(subset),

                "corrections":
                    corrections,

                "accuracy":
                    accuracy,

                "tiou":
                    tiou,

                "composite":
                    composite,

                "delta":
                    composite
                    - base_comp,
            }
        )

        print()
        print(
            f"Threshold >= "
            f"{threshold:.2f}"
        )

        print(
            f"  pairs:       "
            f"{len(subset)}"
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
            f"  delta:       "
            f"{composite - base_comp:+.4f}"
        )

    results = pd.DataFrame(
        results
    )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        OUTPUT,
        index=False,
    )

    best = (
        results.sort_values(
            "composite",
            ascending=False,
        )
        .iloc[0]
    )

    print()
    print("=" * 78)
    print("M060 BEST")
    print("=" * 78)

    print(
        f"Threshold:   "
        f"{best.threshold:.2f}"
    )

    print(
        f"Pairs:       "
        f"{int(best.pairs)}"
    )

    print(
        f"Corrections: "
        f"{int(best.corrections)}"
    )

    print(
        f"Accuracy:    "
        f"{best.accuracy:.4f}"
    )

    print(
        f"tIoU:        "
        f"{best.tiou:.4f}"
    )

    print(
        f"Composite:   "
        f"{best.composite:.4f}"
    )

    print(
        f"Gain:        "
        f"{best.delta:+.4f}"
    )


if __name__ == "__main__":
    main()