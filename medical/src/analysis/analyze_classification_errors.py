from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


DEFAULT_CLAIMS = Path(
    r"medical\artifacts\classification\nli_claim_results.csv"
)

DEFAULT_CV = Path(
    r"medical\artifacts\classification\nli_claim_threshold_cv_results.csv"
)

DEFAULT_OUTPUT = Path(
    r"medical\artifacts\analysis\classification_errors.csv"
)

METHOD = "entailment_ratio"


NUMBER_RE = re.compile(
    r"\b\d+(?:[.,]\d+)?\b"
)

UNIT_RE = re.compile(
    r"\b("
    r"mg|mcg|g|kg|ml|l|mmol/l|mmhg|%"
    r"|days?|weeks?|months?|years?"
    r"|times?|tablets?|capsules?|puffs?"
    r")\b",
    re.IGNORECASE,
)

NEGATION_TERMS = {
    "no",
    "not",
    "never",
    "none",
    "without",
    "denied",
    "deny",
    "absent",
    "negative",
    "normal",
    "unchanged",
    "stable",
    "stop",
    "stopped",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--claims",
        type=Path,
        default=DEFAULT_CLAIMS,
    )

    parser.add_argument(
        "--cv",
        type=Path,
        default=DEFAULT_CV,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    return parser.parse_args()


def parse_bool(value: object) -> bool:
    text = str(value).strip().lower()

    if text in {"true", "1", "yes"}:
        return True

    if text in {"false", "0", "no"}:
        return False

    raise ValueError(
        f"Cannot parse boolean value: {value!r}"
    )


def extract_numbers(text: str) -> str:
    return " | ".join(
        NUMBER_RE.findall(text)
    )


def extract_units(text: str) -> str:
    return " | ".join(
        match.group(0)
        for match in UNIT_RE.finditer(text)
    )


def extract_negations(text: str) -> str:
    tokens = re.findall(
        r"\b[\w'-]+\b",
        text.lower(),
    )

    found = [
        token
        for token in tokens
        if token in NEGATION_TERMS
    ]

    return " | ".join(found)


def main() -> None:
    args = parse_args()

    claims = pd.read_csv(
        args.claims
    ).reset_index(drop=True)

    cv = pd.read_csv(
        args.cv
    )

    cv = cv[
        cv["method"] == METHOD
    ].copy()

    cv["row_index"] = (
        cv["row_index"].astype(int)
    )

    cv = cv.sort_values(
        "row_index"
    ).reset_index(drop=True)

    if len(cv) != len(claims):
        raise RuntimeError(
            "Claim and OOF result lengths differ: "
            f"{len(claims)} vs {len(cv)}"
        )

    if not (
        cv["row_index"].to_numpy()
        == range(len(claims))
    ).all():
        raise RuntimeError(
            "OOF row mapping is invalid."
        )

    df = claims.copy()

    df["oof_predicted_yes"] = [
        parse_bool(value)
        for value in cv["predicted_yes"]
    ]

    df["gold_yes_parsed"] = [
        parse_bool(value)
        for value in df["gold_yes"]
    ]

    df["error_type"] = ""

    false_positive = (
        (~df["gold_yes_parsed"])
        & df["oof_predicted_yes"]
    )

    false_negative = (
        df["gold_yes_parsed"]
        & (~df["oof_predicted_yes"])
    )

    df.loc[
        false_positive,
        "error_type",
    ] = "false_positive"

    df.loc[
        false_negative,
        "error_type",
    ] = "false_negative"

    errors = df[
        df["error_type"] != ""
    ].copy()

    errors["question_numbers"] = (
        errors["question"]
        .astype(str)
        .map(extract_numbers)
    )

    errors["passage_numbers"] = (
        errors["selected_text"]
        .astype(str)
        .map(extract_numbers)
    )

    errors["question_units"] = (
        errors["question"]
        .astype(str)
        .map(extract_units)
    )

    errors["passage_units"] = (
        errors["selected_text"]
        .astype(str)
        .map(extract_units)
    )

    errors["question_negations"] = (
        errors["question"]
        .astype(str)
        .map(extract_negations)
    )

    errors["passage_negations"] = (
        errors["selected_text"]
        .astype(str)
        .map(extract_negations)
    )

    errors["numeric_question"] = (
        errors["question_numbers"] != ""
    )

    errors["unit_question"] = (
        errors["question_units"] != ""
    )

    errors["negation_question"] = (
        errors["question_negations"] != ""
    )

    columns = [
        "question_id",
        "transcript_id",
        "question_type",
        "error_type",
        "question",
        "hypothesis",
        "gold_yes_parsed",
        "oof_predicted_yes",
        "max_entailment",
        "max_contradiction",
        "max_neutral",
        "question_numbers",
        "passage_numbers",
        "question_units",
        "passage_units",
        "question_negations",
        "passage_negations",
        "selected_start",
        "selected_end",
        "selected_text",
    ]

    errors = errors[
        columns
    ]

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    errors.to_csv(
        args.output,
        index=False,
    )

    print("=" * 78)
    print(
        "Nordic AI Cup 2026 — "
        "M014 Classification Error Analysis"
    )
    print("=" * 78)

    print(
        f"Total OOF errors: "
        f"{len(errors)}"
    )

    print()

    print("ERROR TYPE")
    print("-" * 78)
    print(
        errors["error_type"]
        .value_counts()
        .to_string()
    )

    print()
    print("QUESTION TYPE")
    print("-" * 78)

    table = pd.crosstab(
        errors["question_type"],
        errors["error_type"],
    )

    print(table.to_string())

    print()
    print("SURFACE FEATURES")
    print("-" * 78)

    print(
        "Errors with explicit number in question: "
        f"{errors['question_numbers'].ne('').sum()}"
    )

    print(
        "Errors with explicit unit in question:   "
        f"{errors['question_units'].ne('').sum()}"
    )

    print(
        "Errors with negation term in question:   "
        f"{errors['question_negations'].ne('').sum()}"
    )

    print()
    print("FALSE POSITIVES")
    print("-" * 78)

    fp = errors[
        errors["error_type"]
        == "false_positive"
    ]

    for _, row in fp.iterrows():
        print()
        print(
            f"{row['question_id']} "
            f"[{row['question_type']}]"
        )
        print(
            f"Q: {row['question']}"
        )
        print(
            f"H: {row['hypothesis']}"
        )
        print(
            f"P: {row['selected_text']}"
        )
        print(
            "NLI: "
            f"E={row['max_entailment']:.4f} "
            f"C={row['max_contradiction']:.4f} "
            f"N={row['max_neutral']:.4f}"
        )

    print()
    print("=" * 78)
    print("FALSE NEGATIVES")
    print("=" * 78)

    fn = errors[
        errors["error_type"]
        == "false_negative"
    ]

    for _, row in fn.iterrows():
        print()
        print(
            f"{row['question_id']} "
            f"[{row['question_type']}]"
        )
        print(
            f"Q: {row['question']}"
        )
        print(
            f"H: {row['hypothesis']}"
        )
        print(
            f"P: {row['selected_text']}"
        )
        print(
            "NLI: "
            f"E={row['max_entailment']:.4f} "
            f"C={row['max_contradiction']:.4f} "
            f"N={row['max_neutral']:.4f}"
        )

    print()
    print(
        f"CSV report: {args.output}"
    )


if __name__ == "__main__":
    main()