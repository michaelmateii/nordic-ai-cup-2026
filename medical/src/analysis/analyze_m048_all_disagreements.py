from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


BASE_PATH = Path(
    r"medical\artifacts\classification\nli_segmentwise_results.csv"
)

M018_PATH = Path(
    r"medical\artifacts\classification\nli_segmentwise_cv_results.csv"
)

M042_PATH = Path(
    r"medical\artifacts\classification\m042_meta_classifier_oof.csv"
)

M047_PATH = Path(
    r"medical\artifacts\retrieval\m047_all_questions_hybrid_oof.csv"
)

OUTPUT = Path(
    r"medical\artifacts\analysis\m048_all_disagreement_evidence.csv"
)

MODEL_ID = "cross-encoder/nli-deberta-v3-small"
M018_METHOD = "max_segment_margin"

BATCH_SIZE = 32
MAX_LENGTH = 256


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


def resolve_label_ids(model):
    mapping = {
        str(key).lower(): int(value)
        for key, value
        in model.config.label2id.items()
    }

    print(
        "NLI labels:",
        mapping,
    )

    def find(name: str) -> int:
        if name in mapping:
            return mapping[name]

        for key, value in mapping.items():
            if name in key:
                return value

        raise RuntimeError(
            f"Could not resolve {name!r} "
            f"from {mapping}"
        )

    return {
        "contradiction": find(
            "contradiction"
        ),
        "entailment": find(
            "entailment"
        ),
        "neutral": find(
            "neutral"
        ),
    }


def score_pairs(
    premises: list[str],
    hypotheses: list[str],
):
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 78)
    print(
        "Loading M048 evidence NLI"
    )
    print("=" * 78)
    print(
        f"Model:  {MODEL_ID}"
    )
    print(
        f"Device: {device}"
    )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            MODEL_ID
        )
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            MODEL_ID
        )
        .to(device)
        .eval()
    )

    ids = resolve_label_ids(
        model
    )

    probability_batches = []

    for start in range(
        0,
        len(premises),
        BATCH_SIZE,
    ):
        end = min(
            start + BATCH_SIZE,
            len(premises),
        )

        encoded = tokenizer(
            premises[start:end],
            hypotheses[start:end],
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        encoded = {
            key: value.to(device)
            for key, value
            in encoded.items()
        }

        with torch.inference_mode():
            logits = model(
                **encoded
            ).logits

            probabilities = torch.softmax(
                logits,
                dim=-1,
            )

        probability_batches.append(
            probabilities
            .cpu()
            .numpy()
        )

    probabilities = np.concatenate(
        probability_batches,
        axis=0,
    )

    contradiction = probabilities[
        :,
        ids["contradiction"],
    ]

    entailment = probabilities[
        :,
        ids["entailment"],
    ]

    neutral = probabilities[
        :,
        ids["neutral"],
    ]

    return (
        entailment,
        contradiction,
        neutral,
    )


def main() -> None:
    base = pd.read_csv(
        BASE_PATH
    )

    m018 = pd.read_csv(
        M018_PATH
    )

    m018 = m018[
        m018["method"]
        == M018_METHOD
    ].copy()

    m042 = pd.read_csv(
        M042_PATH
    )

    m042 = m042[
        m042["model"]
        == "hgb"
    ].copy()

    evidence = pd.read_csv(
        M047_PATH
    )

    if len(base) != 390:
        raise RuntimeError(
            f"Base has {len(base)} rows"
        )

    if len(m018) != 390:
        raise RuntimeError(
            f"M018 has {len(m018)} rows"
        )

    if len(m042) != 390:
        raise RuntimeError(
            f"M042 has {len(m042)} rows"
        )

    if len(evidence) != 390:
        raise RuntimeError(
            f"M047 has {len(evidence)} rows"
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
        ]
    ].copy()

    m018 = m018[
        [
            "question_id",
            "score",
            "predicted_yes",
        ]
    ].rename(
        columns={
            "score": "m018_score",
            "predicted_yes":
                "m018_predicted_yes",
        }
    )

    m042 = m042[
        [
            "question_id",
            "oof_probability",
            "oof_predicted_yes",
            "fold_threshold",
        ]
    ].rename(
        columns={
            "oof_probability":
                "m042_probability",
            "oof_predicted_yes":
                "m042_predicted_yes",
            "fold_threshold":
                "m042_threshold",
        }
    )

    required_evidence = {
        "question_id",
        "candidate_text",
        "candidate_kind",
        "candidate_start",
        "candidate_end",
        "oof_score",
    }

    missing = (
        required_evidence
        - set(evidence.columns)
    )

    if missing:
        raise RuntimeError(
            "M047 missing columns: "
            f"{sorted(missing)}"
        )

    evidence = evidence[
        [
            "question_id",
            "candidate_text",
            "candidate_kind",
            "candidate_start",
            "candidate_end",
            "oof_score",
            "target_tiou",
        ]
    ].rename(
        columns={
            "candidate_text":
                "evidence_text",
            "candidate_kind":
                "evidence_kind",
            "candidate_start":
                "evidence_start",
            "candidate_end":
                "evidence_end",
            "oof_score":
                "evidence_ranker_score",
            "target_tiou":
                "evidence_gold_tiou",
        }
    )

    df = (
        base
        .merge(
            m018,
            on="question_id",
            validate="one_to_one",
        )
        .merge(
            m042,
            on="question_id",
            validate="one_to_one",
        )
        .merge(
            evidence,
            on="question_id",
            validate="one_to_one",
        )
    )

    if len(df) != 390:
        raise RuntimeError(
            f"Merged rows={len(df)}"
        )

    df["gold_bool"] = [
        parse_bool(value)
        for value
        in df["gold_yes"]
    ]

    df["m018_pred"] = [
        parse_bool(value)
        for value
        in df[
            "m018_predicted_yes"
        ]
    ]

    df["m042_pred"] = [
        parse_bool(value)
        for value
        in df[
            "m042_predicted_yes"
        ]
    ]

    df["m018_correct"] = (
        df["m018_pred"]
        == df["gold_bool"]
    )

    df["m042_correct"] = (
        df["m042_pred"]
        == df["gold_bool"]
    )

    df["disagree"] = (
        df["m018_pred"]
        != df["m042_pred"]
    )

    df["m042_fixed"] = (
        df["disagree"]
        & df["m042_correct"]
    )

    df["m042_broke"] = (
        df["disagree"]
        & df["m018_correct"]
    )

    df["disagreement_direction"] = np.where(
        (~df["m018_pred"])
        & df["m042_pred"],
        "NO_to_YES",
        np.where(
            df["m018_pred"]
            & (~df["m042_pred"]),
            "YES_to_NO",
            "agree",
        ),
    )

    (
        entailment,
        contradiction,
        neutral,
    ) = score_pairs(
        df[
            "evidence_text"
        ].astype(str).tolist(),
        df[
            "hypothesis"
        ].astype(str).tolist(),
    )

    df[
        "evidence_nli_entailment"
    ] = entailment

    df[
        "evidence_nli_contradiction"
    ] = contradiction

    df[
        "evidence_nli_neutral"
    ] = neutral

    df[
        "evidence_nli_margin"
    ] = (
        df[
            "evidence_nli_entailment"
        ]
        - df[
            "evidence_nli_contradiction"
        ]
    )

    df[
        "evidence_nli_ratio"
    ] = (
        df[
            "evidence_nli_entailment"
        ]
        / (
            df[
                "evidence_nli_entailment"
            ]
            + df[
                "evidence_nli_contradiction"
            ]
            + 1e-12
        )
    )

    df[
        "m042_delta"
    ] = (
        df["m042_probability"]
        - df["m042_threshold"]
    )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT,
        index=False,
    )

    disagreements = df[
        df["disagree"]
    ].copy()

    print()
    print("=" * 78)
    print(
        "Nordic AI Cup 2026 — "
        "M048 All-Disagreement Evidence Audit"
    )
    print("=" * 78)

    print(
        f"All questions:          "
        f"{len(df)}"
    )

    print(
        f"Disagreements:          "
        f"{len(disagreements)}"
    )

    print(
        f"M042 useful changes:    "
        f"{int(disagreements['m042_fixed'].sum())}"
    )

    print(
        f"M042 harmful changes:   "
        f"{int(disagreements['m042_broke'].sum())}"
    )

    print()
    print("DIRECTION")
    print("-" * 78)

    print(
        disagreements[
            "disagreement_direction"
        ]
        .value_counts()
        .to_string()
    )

    print()
    print("GROUP MEANS")
    print("-" * 78)

    metric_columns = [
        "m042_probability",
        "m042_delta",
        "max_segment_ratio",
        "max_segment_entailment",
        "max_segment_margin",
        "max_segment_vs_other",
        "evidence_ranker_score",
        "evidence_nli_entailment",
        "evidence_nli_contradiction",
        "evidence_nli_neutral",
        "evidence_nli_margin",
        "evidence_nli_ratio",
    ]

    for name, subset in [
        (
            "M042 FIXED M018",
            disagreements[
                disagreements[
                    "m042_fixed"
                ]
            ],
        ),
        (
            "M042 BROKE M018",
            disagreements[
                disagreements[
                    "m042_broke"
                ]
            ],
        ),
    ]:
        print()
        print(name)

        for column in metric_columns:
            print(
                f"{column:<28} "
                f"{subset[column].mean():.6f}"
            )

    print()
    print("=" * 78)
    print("ALL 19 DISAGREEMENTS")
    print("=" * 78)

    columns = [
        "question_id",
        "question_type",
        "question",
        "gold_bool",
        "m018_pred",
        "m042_pred",
        "m042_fixed",
        "m042_broke",
        "disagreement_direction",
        "m042_probability",
        "m042_threshold",
        "m042_delta",
        "max_segment_ratio",
        "max_segment_entailment",
        "max_segment_margin",
        "evidence_ranker_score",
        "evidence_nli_entailment",
        "evidence_nli_contradiction",
        "evidence_nli_neutral",
        "evidence_nli_margin",
        "evidence_nli_ratio",
        "evidence_kind",
        "evidence_text",
    ]

    print(
        disagreements[
            columns
        ]
        .sort_values(
            [
                "m042_broke",
                "evidence_nli_ratio",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .to_string(
            index=False
        )
    )

    print()
    print("SIMPLE SEPARATION DIAGNOSTICS")
    print("-" * 78)

    fixed = disagreements[
        disagreements[
            "m042_fixed"
        ]
    ]

    broke = disagreements[
        disagreements[
            "m042_broke"
        ]
    ]

    for column in [
        "evidence_ranker_score",
        "evidence_nli_entailment",
        "evidence_nli_margin",
        "evidence_nli_ratio",
        "m042_delta",
        "max_segment_ratio",
        "max_segment_margin",
    ]:
        fixed_values = (
            fixed[column]
            .astype(float)
        )

        broke_values = (
            broke[column]
            .astype(float)
        )

        print()
        print(column)

        print(
            f"  fixed range: "
            f"{fixed_values.min():.6f} "
            f"to "
            f"{fixed_values.max():.6f}"
        )

        print(
            f"  broke range: "
            f"{broke_values.min():.6f} "
            f"to "
            f"{broke_values.max():.6f}"
        )

        print(
            f"  fixed mean:  "
            f"{fixed_values.mean():.6f}"
        )

        print(
            f"  broke mean:  "
            f"{broke_values.mean():.6f}"
        )

    print()
    print("REFERENCE")
    print("-" * 78)

    print(
        "M018 errors: 46"
    )

    print(
        "M042 errors: 49"
    )

    print(
        "Oracle choose M018/M042: "
        "38 errors"
    )

    print()
    print(
        f"Detailed CSV: {OUTPUT}"
    )


if __name__ == "__main__":
    main()