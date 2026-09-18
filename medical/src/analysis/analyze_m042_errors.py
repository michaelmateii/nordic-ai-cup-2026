from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


NLI_RESULTS = Path(
    r"medical\artifacts\classification\nli_segmentwise_results.csv"
)

M018_CV = Path(
    r"medical\artifacts\classification\nli_segmentwise_cv_results.csv"
)

M042_RESULTS = Path(
    r"medical\artifacts\classification\m042_meta_classifier_oof.csv"
)

OUTPUT_CSV = Path(
    r"medical\artifacts\analysis\m045_m042_error_analysis.csv"
)

OUTPUT_TXT = Path(
    r"medical\artifacts\analysis\m045_m042_error_report.txt"
)

M018_METHOD = "max_segment_margin"


NEGATION_WORDS = {
    "no",
    "not",
    "never",
    "none",
    "without",
    "neither",
    "nor",
    "unchanged",
    "discontinued",
    "stop",
    "stopped",
    "denied",
    "deny",
}

MEDICATION_WORDS = {
    "medication",
    "medicine",
    "drug",
    "prescription",
    "prescribed",
    "renewed",
    "renew",
    "dose",
    "tablet",
    "tablets",
    "treatment",
}

NORMALITY_WORDS = {
    "normal",
    "abnormal",
    "stable",
    "unstable",
    "unchanged",
    "positive",
    "negative",
}

EXAM_WORDS = {
    "examination",
    "examined",
    "auscultation",
    "heard",
    "sounds",
    "lungs",
    "heart",
    "blood",
    "sample",
    "test",
}

NUMBER_PATTERN = re.compile(
    r"\b\d+(?:\.\d+)?\b"
)

UNIT_PATTERN = re.compile(
    r"\b(?:"
    r"mg|mcg|g|kg|ml|l|"
    r"mmol|mmol/l|mg/dl|"
    r"cm|mm|%|bpm"
    r")\b",
    flags=re.IGNORECASE,
)


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value

    text = str(value).strip().lower()

    if text in {
        "true",
        "1",
        "yes",
    }:
        return True

    if text in {
        "false",
        "0",
        "no",
    }:
        return False

    raise ValueError(
        f"Cannot parse bool: {value!r}"
    )


def contains_any(
    text: str,
    words: set[str],
) -> bool:
    lower = text.lower()

    return any(
        re.search(
            rf"\b{re.escape(word)}\b",
            lower,
        )
        for word in words
    )


def first_word(
    text: str,
) -> str:
    parts = (
        text.strip()
        .lower()
        .split()
    )

    if not parts:
        return ""

    return parts[0].strip(
        "?!.,:;"
    )


def main() -> None:
    base = pd.read_csv(
        NLI_RESULTS
    )

    m018 = pd.read_csv(
        M018_CV
    )

    m018 = m018[
        m018["method"]
        == M018_METHOD
    ].copy()

    m042 = pd.read_csv(
        M042_RESULTS
    )

    m042 = m042[
        m042["model"]
        == "hgb"
    ].copy()

    if len(base) != 390:
        raise RuntimeError(
            f"Expected 390 NLI rows, got {len(base)}"
        )

    if len(m018) != 390:
        raise RuntimeError(
            f"Expected 390 M018 rows, got {len(m018)}"
        )

    if len(m042) != 390:
        raise RuntimeError(
            f"Expected 390 M042 rows, got {len(m042)}"
        )

    base = base[
        [
            "question_id",
            "transcript_id",
            "question",
            "hypothesis",
            "question_type",
            "gold_yes",
            "max_segment_ratio",
            "max_segment_entailment",
            "max_segment_margin",
            "max_segment_vs_other",
            "selected_entailment",
            "selected_contradiction",
            "selected_neutral",
            "selected_start",
            "selected_end",
            "selected_text",
            "margin_selected_start",
            "margin_selected_end",
            "margin_selected_text",
        ]
    ].copy()

    m018 = m018[
        [
            "question_id",
            "score",
            "predicted_yes",
            "correct",
        ]
    ].rename(
        columns={
            "score": "m018_score",
            "predicted_yes": "m018_predicted_yes",
            "correct": "m018_correct",
        }
    )

    m042 = m042[
        [
            "question_id",
            "oof_probability",
            "oof_predicted_yes",
            "fold_threshold",
            "correct",
        ]
    ].rename(
        columns={
            "oof_probability": "m042_probability",
            "oof_predicted_yes": "m042_predicted_yes",
            "fold_threshold": "m042_threshold",
            "correct": "m042_correct",
        }
    )

    df = (
        base
        .merge(
            m018,
            on="question_id",
            how="inner",
            validate="one_to_one",
        )
        .merge(
            m042,
            on="question_id",
            how="inner",
            validate="one_to_one",
        )
    )

    if len(df) != 390:
        raise RuntimeError(
            f"Merge produced {len(df)} rows"
        )

    df["gold_bool"] = [
        parse_bool(value)
        for value in df["gold_yes"]
    ]

    df["m018_pred"] = [
        parse_bool(value)
        for value in df["m018_predicted_yes"]
    ]

    df["m042_pred"] = [
        parse_bool(value)
        for value in df["m042_predicted_yes"]
    ]

    df["m018_is_correct"] = (
        df["m018_pred"]
        == df["gold_bool"]
    )

    df["m042_is_correct"] = (
        df["m042_pred"]
        == df["gold_bool"]
    )

    df["m042_fixed_m018"] = (
        (~df["m018_is_correct"])
        & df["m042_is_correct"]
    )

    df["m042_broke_m018"] = (
        df["m018_is_correct"]
        & (~df["m042_is_correct"])
    )

    df["both_wrong"] = (
        (~df["m018_is_correct"])
        & (~df["m042_is_correct"])
    )

    df["error_type_m042"] = np.where(
        df["m042_is_correct"],
        "correct",
        np.where(
            df["gold_bool"],
            "false_negative",
            "false_positive",
        ),
    )

    questions = (
        df["question"]
        .fillna("")
        .astype(str)
    )

    df["first_word"] = questions.map(
        first_word
    )

    df["question_word_count"] = (
        questions.str.split().str.len()
    )

    df["has_number"] = (
        questions.str.contains(
            NUMBER_PATTERN,
            regex=True,
        )
        .astype(int)
    )

    df["has_unit"] = (
        questions.str.contains(
            UNIT_PATTERN,
            regex=True,
        )
        .astype(int)
    )

    df["has_negation"] = questions.map(
        lambda text: int(
            contains_any(
                text,
                NEGATION_WORDS,
            )
        )
    )

    df["medication_related"] = (
        questions.map(
            lambda text: int(
                contains_any(
                    text,
                    MEDICATION_WORDS,
                )
            )
        )
    )

    df["normality_related"] = (
        questions.map(
            lambda text: int(
                contains_any(
                    text,
                    NORMALITY_WORDS,
                )
            )
        )
    )

    df["exam_related"] = (
        questions.map(
            lambda text: int(
                contains_any(
                    text,
                    EXAM_WORDS,
                )
            )
        )
    )

    df["m042_margin_to_threshold"] = (
        df["m042_probability"]
        - df["m042_threshold"]
    )

    df["m018_margin_distance"] = (
        df["max_segment_margin"]
        - 0.000556
    )

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    lines: list[str] = []

    def emit(
        text: str = "",
    ) -> None:
        lines.append(text)

    emit("=" * 78)
    emit(
        "Nordic AI Cup 2026 — "
        "M045 M042 Differential Error Analysis"
    )
    emit("=" * 78)

    emit()
    emit("OVERALL")
    emit("-" * 78)

    m018_errors = int(
        (~df["m018_is_correct"]).sum()
    )

    m042_errors = int(
        (~df["m042_is_correct"]).sum()
    )

    fixed = int(
        df["m042_fixed_m018"].sum()
    )

    broken = int(
        df["m042_broke_m018"].sum()
    )

    both_wrong = int(
        df["both_wrong"].sum()
    )

    emit(
        f"M018 errors:              {m018_errors}"
    )

    emit(
        f"M042 errors:              {m042_errors}"
    )

    emit(
        f"M042 fixed from M018:     {fixed}"
    )

    emit(
        f"M042 broke from M018:     {broken}"
    )

    emit(
        f"Wrong in both:            {both_wrong}"
    )

    emit()
    emit("M042 ERROR BREAKDOWN")
    emit("-" * 78)

    errors = df[
        ~df["m042_is_correct"]
    ].copy()

    table = pd.crosstab(
        errors["question_type"],
        errors["error_type_m042"],
    )

    emit(
        table.to_string()
    )

    emit()
    emit("SURFACE FEATURES AMONG M042 ERRORS")
    emit("-" * 78)

    for column in [
        "has_number",
        "has_unit",
        "has_negation",
        "medication_related",
        "normality_related",
        "exam_related",
    ]:
        emit(
            f"{column:<22} "
            f"{int(errors[column].sum())}"
        )

    emit()
    emit("FIRST WORDS AMONG M042 ERRORS")
    emit("-" * 78)

    counts = (
        errors["first_word"]
        .value_counts()
        .head(20)
    )

    emit(
        counts.to_string()
    )

    def print_questions(
        title: str,
        subset: pd.DataFrame,
        *,
        sort_column: str | None = None,
        ascending: bool = True,
    ) -> None:
        emit()
        emit(title)
        emit("-" * 78)

        if sort_column is not None:
            subset = (
                subset.sort_values(
                    sort_column,
                    ascending=ascending,
                )
            )

        if subset.empty:
            emit("(none)")
            return

        for _, row in subset.iterrows():
            emit()
            emit(
                f"{row['question_id']} "
                f"[{row['question_type']}]"
            )

            emit(
                f"Q: {row['question']}"
            )

            emit(
                f"H: {row['hypothesis']}"
            )

            emit(
                f"Gold: {row['gold_bool']} | "
                f"M018: {row['m018_pred']} | "
                f"M042: {row['m042_pred']}"
            )

            emit(
                "M042 "
                f"p={row['m042_probability']:.4f} "
                f"thr={row['m042_threshold']:.4f} "
                f"delta="
                f"{row['m042_margin_to_threshold']:+.4f}"
            )

            emit(
                "NLI "
                f"ratio={row['max_segment_ratio']:.6f} "
                f"E={row['max_segment_entailment']:.6f} "
                f"margin={row['max_segment_margin']:.6f} "
                f"vs_other={row['max_segment_vs_other']:.6f}"
            )

            emit(
                "Selected probs "
                f"E={row['selected_entailment']:.6f} "
                f"C={row['selected_contradiction']:.6f} "
                f"N={row['selected_neutral']:.6f}"
            )

            emit(
                "Evidence candidate: "
                f"{row['margin_selected_start']}–"
                f"{row['margin_selected_end']}"
            )

            emit(
                f"P: {row['margin_selected_text']}"
            )

            feature_bits = []

            for column in [
                "has_number",
                "has_unit",
                "has_negation",
                "medication_related",
                "normality_related",
                "exam_related",
            ]:
                if bool(
                    row[column]
                ):
                    feature_bits.append(
                        column
                    )

            emit(
                "Features: "
                + (
                    ", ".join(
                        feature_bits
                    )
                    if feature_bits
                    else "(none)"
                )
            )

    print_questions(
        "M042 FIXED RELATIVE TO M018",
        df[
            df["m042_fixed_m018"]
        ],
        sort_column="m042_margin_to_threshold",
        ascending=False,
    )

    print_questions(
        "M042 INTRODUCED NEW ERRORS",
        df[
            df["m042_broke_m018"]
        ],
        sort_column="m042_margin_to_threshold",
        ascending=False,
    )

    print_questions(
        "M042 FALSE NEGATIVES",
        errors[
            errors["error_type_m042"]
            == "false_negative"
        ],
        sort_column="m042_margin_to_threshold",
        ascending=False,
    )

    print_questions(
        "M042 FALSE POSITIVES",
        errors[
            errors["error_type_m042"]
            == "false_positive"
        ],
        sort_column="m042_margin_to_threshold",
        ascending=False,
    )

    emit()
    emit("NEAR-THRESHOLD M042 ERRORS")
    emit("-" * 78)

    near = (
        errors.assign(
            absolute_distance=(
                errors[
                    "m042_margin_to_threshold"
                ].abs()
            )
        )
        .sort_values(
            "absolute_distance"
        )
        .head(20)
    )

    for _, row in near.iterrows():
        emit(
            f"{row['question_id']:<28} "
            f"{row['error_type_m042']:<15} "
            f"delta="
            f"{row['m042_margin_to_threshold']:+.4f} "
            f"| {row['question']}"
        )

    emit()
    emit("POTENTIAL RULE-AUDIT GROUPS")
    emit("-" * 78)

    for label, mask in [
        (
            "negation",
            errors["has_negation"]
            == 1,
        ),
        (
            "normality",
            errors[
                "normality_related"
            ]
            == 1,
        ),
        (
            "medication",
            errors[
                "medication_related"
            ]
            == 1,
        ),
        (
            "exam",
            errors[
                "exam_related"
            ]
            == 1,
        ),
        (
            "numbers",
            errors["has_number"]
            == 1,
        ),
    ]:
        subset = errors[
            mask
        ]

        fp = int(
            (
                subset[
                    "error_type_m042"
                ]
                == "false_positive"
            ).sum()
        )

        fn = int(
            (
                subset[
                    "error_type_m042"
                ]
                == "false_negative"
            ).sum()
        )

        emit(
            f"{label:<12} "
            f"n={len(subset):>2} "
            f"FP={fp:>2} "
            f"FN={fn:>2}"
        )

    OUTPUT_TXT.write_text(
        "\n".join(
            lines
        ),
        encoding="utf-8",
    )

    print(
        "\n".join(
            lines[:80]
        )
    )

    print()
    print(
        f"Full report: {OUTPUT_TXT}"
    )

    print(
        f"Detailed CSV: {OUTPUT_CSV}"
    )


if __name__ == "__main__":
    main()