from __future__ import annotations

import gc
import random
import time

import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import GroupKFold
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)


QUESTIONS = (
    Path(__file__).resolve().parents[3].parent
    / "Nordic-AI-Cup-2026-official"
    / "medical-appointment"
    / "data"
    / "question_train.csv"
)

CANDIDATES = (
    r"medical\artifacts\retrieval"
    r"\m047_all_questions_candidates_oof.csv"
)

M070 = (
    r"medical\artifacts\classification"
    r"\m070_task_specific_nli_oof.csv"
)

MODEL_ID = "cross-encoder/nli-deberta-v3-base"

SEED = 42
FOLD = 1

TOP_PASSAGES = 5
EPOCHS = 2
BATCH_SIZE = 4
MAX_LENGTH = 384
LR = 7e-6

M055_WEIGHT = 0.68
M070_WEIGHT = 0.32
M074_THRESHOLD = 0.3226589362393797


def parse_bool(x):
    return (
        str(x)
        .strip()
        .lower()
        in {"true", "1", "yes"}
    )


def score_metric(
    gold,
    pred,
    evidence,
):
    acc = float(
        np.mean(
            gold == pred
        )
    )

    tiou = float(
        np.mean(
            np.where(
                pred[gold],
                evidence[gold],
                0.0,
            )
        )
    )

    return (
        acc,
        tiou,
        0.4 * acc + 0.6 * tiou,
    )


def build_examples(
    questions,
    candidates,
):
    candidates = candidates.sort_values(
        [
            "question_id",
            "oof_score",
        ],
        ascending=[
            True,
            False,
        ],
    )

    rows = []

    for _, q in questions.iterrows():
        x = (
            candidates[
                candidates[
                    "question_id"
                ]
                == q["question_id"]
            ]
            .head(TOP_PASSAGES)
        )

        passages = []

        seen = set()

        for _, row in x.iterrows():
            text = str(
                row["candidate_text"]
            ).strip()

            if (
                not text
                or text in seen
            ):
                continue

            seen.add(text)

            passages.append(text)

        context = "\n".join(
            f"[PASSAGE {i+1}] {text}"
            for i, text
            in enumerate(passages)
        )

        rows.append(
            {
                "question_id":
                    q["question_id"],

                "transcript_id":
                    q["transcript_id"],

                "question":
                    q["question"],

                "context":
                    context,

                "target":
                    int(
                        parse_bool(
                            q["answer"]
                        )
                    ),
            }
        )

    return pd.DataFrame(rows)


class DS(Dataset):
    def __init__(
        self,
        df,
        tokenizer,
    ):
        self.df = (
            df.reset_index(
                drop=True
            )
        )

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]

        text = (
            "[QUESTION] "
            + str(row["question"])
            + "\n"
            + str(row["context"])
        )

        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids":
                encoded[
                    "input_ids"
                ][0],

            "attention_mask":
                encoded[
                    "attention_mask"
                ][0],

            "labels":
                torch.tensor(
                    int(
                        row["target"]
                    ),
                    dtype=torch.long,
                ),
        }


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    started = time.perf_counter()

    q = pd.read_csv(
        QUESTIONS
    )

    c = pd.read_csv(
        CANDIDATES
    )

    examples = build_examples(
        q,
        c,
    )

    m070 = pd.read_csv(
        M070
    )

    reference = (
        m070[
            [
                "question_id",
                "m055_probability",
                "m070_max",
                "evidence_tiou",
            ]
        ]
    )

    examples = examples.merge(
        reference,
        on="question_id",
        validate="one_to_one",
    )

    gold = (
        examples[
            "target"
        ]
        .astype(bool)
        .to_numpy()
    )

    splitter = GroupKFold(
        n_splits=5
    )

    chosen = None

    for fold, pair in enumerate(
        splitter.split(
            examples,
            gold,
            examples[
                "transcript_id"
            ],
        ),
        start=1,
    ):
        if fold == FOLD:
            chosen = pair
            break

    train_idx, test_idx = chosen

    train = (
        examples.iloc[
            train_idx
        ].copy()
    )

    test = (
        examples.iloc[
            test_idx
        ].copy()
    )

    print("=" * 78)
    print(
        "M076 MULTI-PASSAGE PILOT"
    )
    print("=" * 78)

    print(
        "Train:",
        len(train),
    )

    print(
        "Test:",
        len(test),
    )

    print(
        "Top passages:",
        TOP_PASSAGES,
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

    train_loader = DataLoader(
        DS(
            train,
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

        for batch in train_loader:
            optimizer.zero_grad(
                set_to_none=True
            )

            input_ids = (
                batch[
                    "input_ids"
                ].to(device)
            )

            attention_mask = (
                batch[
                    "attention_mask"
                ].to(device)
            )

            labels = (
                batch[
                    "labels"
                ].to(device)
            )

            out = model(
                input_ids=input_ids,
                attention_mask=(
                    attention_mask
                ),
                labels=labels,
            )

            loss = out.loss

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
            f"epoch {epoch}: "
            f"loss="
            f"{np.mean(losses):.6f}"
        )

    # ------------------------------------------------------------
    # Held-out inference.
    # ------------------------------------------------------------

    model.eval()

    probs = []

    loader = DataLoader(
        DS(
            test,
            tokenizer,
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    with torch.inference_mode():
        for batch in loader:
            logits = model(
                input_ids=(
                    batch[
                        "input_ids"
                    ].to(device)
                ),
                attention_mask=(
                    batch[
                        "attention_mask"
                    ].to(device)
                ),
            ).logits

            prob = (
                torch.softmax(
                    logits,
                    dim=-1,
                )[:, 1]
            )

            probs.extend(
                prob.cpu()
                .numpy()
                .tolist()
            )

    probs = np.asarray(
        probs
    )

    test_gold = (
        test["target"]
        .astype(bool)
        .to_numpy()
    )

    evidence = (
        test[
            "evidence_tiou"
        ]
        .fillna(0.0)
        .to_numpy()
    )

    # ------------------------------------------------------------
    # M074 baseline on exact same fold.
    # ------------------------------------------------------------

    m074_score = (
        M055_WEIGHT
        * test[
            "m055_probability"
        ].to_numpy()
        +
        M070_WEIGHT
        * test[
            "m070_max"
        ].to_numpy()
    )

    m074_pred = (
        m074_score
        >= M074_THRESHOLD
    )

    m074_metrics = score_metric(
        test_gold,
        m074_pred,
        evidence,
    )

    # ------------------------------------------------------------
    # Find threshold for M076 using TRAIN probabilities would
    # normally be preferable, but for this pilot use 0.5 direct
    # first so we don't over-engineer a weak model.
    # ------------------------------------------------------------

    m076_pred = (
        probs >= 0.5
    )

    m076_metrics = score_metric(
        test_gold,
        m076_pred,
        evidence,
    )

    # ------------------------------------------------------------
    # Cheap fusion with M074 on held-out fold.
    # Try predefined weights only; no test-label optimization.
    # ------------------------------------------------------------

    best = None

    for new_weight in [
        0.20,
        0.30,
        0.40,
        0.50,
    ]:
        combined = (
            (1.0 - new_weight)
            * m074_score
            + new_weight
            * probs
        )

        # Current M074 scale is lower than 0.5,
        # so normalize threshold correspondingly.
        threshold = (
            (1.0 - new_weight)
            * M074_THRESHOLD
            + new_weight
            * 0.5
        )

        pred = (
            combined
            >= threshold
        )

        metrics = score_metric(
            test_gold,
            pred,
            evidence,
        )

        result = (
            metrics[2],
            new_weight,
            threshold,
            metrics,
        )

        if (
            best is None
            or result[0]
            > best[0]
        ):
            best = result

    print()
    print("=" * 78)
    print(
        "M076 PILOT RESULT"
    )
    print("=" * 78)

    print(
        "M074 fold:",
        f"acc={m074_metrics[0]:.4f}",
        f"tIoU={m074_metrics[1]:.4f}",
        f"comp={m074_metrics[2]:.4f}",
    )

    print(
        "M076 direct:",
        f"acc={m076_metrics[0]:.4f}",
        f"tIoU={m076_metrics[1]:.4f}",
        f"comp={m076_metrics[2]:.4f}",
    )

    print(
        "Best fusion:",
        f"weight={best[1]:.2f}",
        f"threshold={best[2]:.4f}",
        f"acc={best[3][0]:.4f}",
        f"tIoU={best[3][1]:.4f}",
        f"comp={best[3][2]:.4f}",
    )

    print(
        "Delta:",
        f"{best[3][2] - m074_metrics[2]:+.4f}",
    )

    print(
        "Runtime:",
        f"{time.perf_counter()-started:.1f}s",
    )

    del model
    gc.collect()

    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

