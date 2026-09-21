from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

from medical.src.classification.evaluate_m076_multipassage_pilot import (
    QUESTIONS,
    CANDIDATES,
    MODEL_ID,
    TOP_PASSAGES,
    EPOCHS,
    BATCH_SIZE,
    MAX_LENGTH,
    LR,
    DS,
    build_examples,
)


OUTPUT = Path(
    r"medical\artifacts\models"
    r"\m076_multipassage_classifier"
)

METADATA = Path(
    r"medical\artifacts\models"
    r"\m076_multipassage_metadata.json"
)

SEED = 42

# Fixed deployment fusion.
M076_WEIGHT = 0.35

# Current hidden-validated M074 champion.
M074_THRESHOLD = 0.3226589362393797


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    questions = pd.read_csv(
        QUESTIONS
    )

    candidates = pd.read_csv(
        CANDIDATES
    )

    examples = build_examples(
        questions,
        candidates,
    )

    print("=" * 78)
    print(
        "Nordic AI Cup 2026 — "
        "M076 Full-Data Export"
    )
    print("=" * 78)

    print(
        f"Training questions: {len(examples)}"
    )

    print(
        f"Top passages:       {TOP_PASSAGES}"
    )

    print(
        f"Epochs:             {EPOCHS}"
    )

    device = "cuda"

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_ID
        )
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            MODEL_ID,
            num_labels=2,
            ignore_mismatched_sizes=True,
        )
        .to(device)
    )

    loader = DataLoader(
        DS(
            examples,
            tokenizer,
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=0.01,
    )

    model.train()

    for epoch in range(
        1,
        EPOCHS + 1,
    ):
        losses = []

        for batch in loader:
            optimizer.zero_grad(
                set_to_none=True
            )

            output = model(
                input_ids=(
                    batch["input_ids"]
                    .to(device)
                ),
                attention_mask=(
                    batch["attention_mask"]
                    .to(device)
                ),
                labels=(
                    batch["labels"]
                    .to(device)
                ),
            )

            loss = output.loss

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            optimizer.step()

            losses.append(
                float(
                    loss.detach().cpu()
                )
            )

        print(
            f"epoch {epoch}/{EPOCHS}: "
            f"loss={np.mean(losses):.6f}"
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

    # Same calibration interpolation used by pilot,
    # now with fixed 0.35 deployment weight.
    fusion_threshold = (
        (1.0 - M076_WEIGHT)
        * M074_THRESHOLD
        + M076_WEIGHT
        * 0.5
    )

    metadata = {
        "training_questions":
            len(examples),

        "base_model":
            MODEL_ID,

        "top_passages":
            TOP_PASSAGES,

        "max_length":
            MAX_LENGTH,

        "epochs":
            EPOCHS,

        "m076_weight":
            M076_WEIGHT,

        "m074_weight":
            1.0 - M076_WEIGHT,

        "m074_threshold":
            M074_THRESHOLD,

        "fusion_threshold":
            fusion_threshold,

        "pilot_fold_1_delta":
            0.0100,

        "pilot_fold_2_delta":
            0.0100,
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
    print("M076 EXPORT COMPLETE")
    print("=" * 78)

    print(
        f"Model:     {OUTPUT}"
    )

    print(
        f"Metadata:  {METADATA}"
    )

    print(
        f"M074 weight: "
        f"{1-M076_WEIGHT:.2f}"
    )

    print(
        f"M076 weight: "
        f"{M076_WEIGHT:.2f}"
    )

    print(
        f"Threshold: "
        f"{fusion_threshold:.10f}"
    )


if __name__ == "__main__":
    main()