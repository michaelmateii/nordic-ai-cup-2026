from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


BASE_PATH = Path(
    r"medical\artifacts\classification\nli_segmentwise_results.csv"
)

M018_PATH = Path(
    r"medical\artifacts\classification\nli_segmentwise_cv_results.csv"
)

M042_PATH = Path(
    r"medical\artifacts\classification\m042_meta_classifier_oof.csv"
)

EVIDENCE_PATH = Path(
    r"medical\artifacts\retrieval\finetuned_hybrid_ranker_oof.csv"
)

OUTPUT = Path(
    r"medical\artifacts\analysis\m046_evidence_nli_disagreements.csv"
)

MODEL_ID = "cross-encoder/nli-deberta-v3-small"
M018_METHOD = "max_segment_margin"

BATCH_SIZE = 32
MAX_LENGTH = 256


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value

    text = str(value).strip().lower()

    if text in {"true", "1", "yes"}:
        return True

    if text in {"false", "0", "no"}:
        return False

    raise ValueError(
        f"Cannot parse bool: {value!r}"
    )


def resolve_label_ids(model):
    mapping = {
        str(k).lower(): int(v)
        for k, v
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
            f"Could not resolve NLI label {name!r} "
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
    print("Loading evidence-verification NLI")
    print("=" * 78)
    print(f"Model:  {MODEL_ID}")
    print(f"Device: {device}")

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

    all_probs = []

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

            probs = torch.softmax(
                logits,
                dim=-1,
            )

        all_probs.append(
            probs.cpu().numpy()
        )

    probs = np.concatenate(
        all_probs,
        axis=0,
    )

    contradiction = probs[
        :,
        ids["contradiction"],
    ]

    entailment = probs[
        :,
        ids["entailment"],
    ]

    neutral = probs[
        :,
        ids["neutral"],
    ]

    return (
        entailment,
        contradiction,
        neutral,
    )


def main():
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
        EVIDENCE_PATH
    )

    if len(base) != 390:
        raise RuntimeError(
            f"Expected 390 base rows, got {len(base)}"
        )

    if len(m018) != 390:
        raise RuntimeError(
            f"Expected 390 M018 rows, got {len(m018)}"
        )

    if len(m042) != 390:
        raise RuntimeError(
            f"Expected 390 M042 rows, got {len(m042)}"
        )

    print(
        "Evidence columns:",
        list(evidence.columns),
    )

    required_evidence = {
        "question_id",
        "candidate_text",
        "target_tiou",
        "oof_score",
    }

    missing = (
        required_evidence
        - set(evidence.columns)
    )

    if missing:
        raise RuntimeError(
            "Evidence file missing columns: "
            f"{sorted(missing)}"
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
        ]
    ].copy()

    m018 = m018[
        [
            "question_id",
            "predicted_yes",
            "score",
        ]
    ].rename(
        columns={
            "predicted_yes":
                "m018_predicted_yes",
            "score":
                "m018_score",
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

    evidence = evidence[
        [
            "question_id",
            "candidate_text",
            "candidate_start",
            "candidate_end",
            "target_tiou",
            "oof_score",
            "candidate_kind",
        ]
    ].rename(
        columns={
            "candidate_text":
                "evidence_text",
            "candidate_start":
                "evidence_start",
            "candidate_end":
                "evidence_end",
            "target_tiou":
                "evidence_gold_tiou",
            "oof_score":
                "evidence_ranker_score",
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
    )

    # Evidence exists only for gold-positive
    # training questions in M036.
    df = df.merge(
        evidence,
        on="question_id",
        how="left",
        validate="one_to_one",
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

    df["disagree"] = (
        df["m018_pred"]
        != df["m042_pred"]
    )

    df["m018_correct"] = (
        df["m018_pred"]
        == df["gold_bool"]
    )

    df["m042_correct"] = (
        df["m042_pred"]
        == df["gold_bool"]
    )

    df["m042_fixed"] = (
        df["disagree"]
        & df["m042_correct"]
    )

    df["m042_broke"] = (
        df["disagree"]
        & df["m018_correct"]
    )

    # -----------------------------------------------------
    # Important:
    # M036 only contains gold-positive questions.
    #
    # To audit ALL disagreements, fall back to the
    # classifier-selected text for negatives would mix
    # evidence sources, so for now score only disagreements
    # for which M036 has an OOF selected span.
    # -----------------------------------------------------

    scored = df[
        df["evidence_text"].notna()
    ].copy()

    entailment, contradiction, neutral = (
        score_pairs(
            scored[
                "evidence_text"
            ].astype(str).tolist(),
            scored[
                "hypothesis"
            ].astype(str).tolist(),
        )
    )

    scored[
        "evidence_nli_entailment"
    ] = entailment

    scored[
        "evidence_nli_contradiction"
    ] = contradiction

    scored[
        "evidence_nli_neutral"
    ] = neutral

    scored[
        "evidence_nli_margin"
    ] = (
        scored[
            "evidence_nli_entailment"
        ]
        - scored[
            "evidence_nli_contradiction"
        ]
    )

    scored[
        "evidence_nli_ratio"
    ] = (
        scored[
            "evidence_nli_entailment"
        ]
        / (
            scored[
                "evidence_nli_entailment"
            ]
            + scored[
                "evidence_nli_contradiction"
            ]
            + 1e-12
        )
    )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    scored.to_csv(
        OUTPUT,
        index=False,
    )

    disagreements = scored[
        scored["disagree"]
    ].copy()

    print()
    print("=" * 78)
    print(
        "Nordic AI Cup 2026 — "
        "M046 Evidence-Aware Disagreement Audit"
    )
    print("=" * 78)

    print(
        f"Scored positive questions: "
        f"{len(scored)}"
    )

    print(
        f"Scored disagreements:      "
        f"{len(disagreements)}"
    )

    print()

    columns = [
        "question_id",
        "question",
        "gold_bool",
        "m018_pred",
        "m042_pred",
        "m042_fixed",
        "m042_broke",
        "m042_probability",
        "m042_threshold",
        "evidence_ranker_score",
        "evidence_gold_tiou",
        "evidence_nli_entailment",
        "evidence_nli_contradiction",
        "evidence_nli_neutral",
        "evidence_nli_margin",
        "evidence_nli_ratio",
        "candidate_kind",
        "evidence_text",
    ]

    print(
        disagreements[
            columns
        ]
        .sort_values(
            "evidence_nli_ratio",
            ascending=False,
        )
        .to_string(
            index=False
        )
    )

    print()
    print("GROUP MEANS")
    print("-" * 78)

    for name, subset in [
        (
            "M042 fixed M018",
            disagreements[
                disagreements[
                    "m042_fixed"
                ]
            ],
        ),
        (
            "M042 broke M018",
            disagreements[
                disagreements[
                    "m042_broke"
                ]
            ],
        ),
    ]:
        print()
        print(name)

        if len(subset) == 0:
            print("(none)")
            continue

        for column in [
            "evidence_ranker_score",
            "evidence_nli_entailment",
            "evidence_nli_contradiction",
            "evidence_nli_margin",
            "evidence_nli_ratio",
            "evidence_gold_tiou",
        ]:
            print(
                f"{column:<28} "
                f"{subset[column].mean():.6f}"
            )

    print()
    print(
        f"Detailed results: {OUTPUT}"
    )


if __name__ == "__main__":
    main()