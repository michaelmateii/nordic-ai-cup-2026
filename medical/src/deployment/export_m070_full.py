from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch

from medical.src.classification.evaluate_m070_task_specific_nli import (
    MODEL_ID,
    prepare_candidates,
    build_training_pairs,
    train_fold,
)


ROOT = Path("medical")

CANDIDATES = (
    ROOT
    / "artifacts"
    / "retrieval"
    / "m047_all_questions_candidates_oof.csv"
)

OUTPUT = (
    ROOT
    / "artifacts"
    / "models"
    / "m070_task_specific_nli"
)

METADATA = (
    ROOT
    / "artifacts"
    / "models"
    / "m070_task_specific_nli_metadata.json"
)


def main():
    print("=" * 78)
    print(
        "Nordic AI Cup 2026 — "
        "M070 Full-Data Export"
    )
    print("=" * 78)

    raw = pd.read_csv(
        CANDIDATES
    )

    candidates = prepare_candidates(
        raw
    )

    question_ids = set(
        candidates[
            "question_id"
        ].unique()
    )

    if len(question_ids) != 390:
        raise RuntimeError(
            f"Expected 390 questions, "
            f"got {len(question_ids)}"
        )

    pairs = build_training_pairs(
        candidates,
        question_ids,
    )

    print(
        f"Questions:      "
        f"{len(question_ids)}"
    )

    print(
        f"Candidates:     "
        f"{len(candidates)}"
    )

    print(
        f"Training pairs: "
        f"{len(pairs)}"
    )

    print()
    print(
        pairs[
            "source"
        ]
        .value_counts()
        .to_string()
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print(
        f"Device: {device}"
    )

    (
        tokenizer,
        model,
        entailment_index,
        contradiction_index,
        neutral_index,
    ) = train_fold(
        pairs,
        device,
        fold=0,
    )

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    tokenizer.save_pretrained(
        OUTPUT
    )

    model.save_pretrained(
        OUTPUT
    )

    metadata = {
        "model_id":
            MODEL_ID,

        "training_questions":
            len(question_ids),

        "training_pairs":
            len(pairs),

        "entailment_index":
            int(entailment_index),

        "contradiction_index":
            int(contradiction_index),

        "neutral_index":
            int(neutral_index),

        # Robust values from the conversation-disjoint
        # fusion folds.
        #
        # Four of five folds selected 0.60-0.70 M055.
        # Median fold weight is 0.70.
        "m055_weight":
            0.70,

        # Fold thresholds:
        # .3936, .3053, .3053, .2958, .1934
        # Median = .3053.
        "fusion_threshold":
            0.3053,

        "oof_m055_composite":
            0.6534,

        "oof_m070_fusion_composite":
            0.6660,
    }

    METADATA.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print("EXPORT COMPLETE")
    print("=" * 78)

    print(
        f"Model:    {OUTPUT}"
    )

    print(
        f"Metadata: {METADATA}"
    )

    print(
        "M055 weight:      0.70"
    )

    print(
        "M070 weight:      0.30"
    )

    print(
        "Fusion threshold: 0.3053"
    )


if __name__ == "__main__":
    main()