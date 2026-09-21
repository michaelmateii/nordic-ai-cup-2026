from __future__ import annotations

import gc
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


INPUT = Path(
    r"medical\artifacts\retrieval"
    r"\m063_all_candidate_oof_scores.csv"
)

OUTPUT = Path(
    r"medical\artifacts\retrieval"
    r"\m065_pairwise_span_comparator_oof.csv"
)

MODEL_ID = (
    "cross-encoder/ms-marco-MiniLM-L6-v2"
)

TOP_K = 5
N_SPLITS = 5

SEED = 42
EPOCHS = 4

LR = 1e-5
WEIGHT_DECAY = 0.01

BATCH_SIZE = 16
MAX_LENGTH = 256

MIN_TIOU_GAP = 0.10


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ComparatorDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        return (
            str(row["question"]),
            str(row["text_a"]),
            str(row["text_b"]),
            int(row["label"]),
            float(row["weight"]),
        )


def prepare_top5(raw):
    df = raw.sort_values(
        [
            "question_id",
            "oof_score",
        ],
        ascending=[
            True,
            False,
        ],
    ).copy()

    df["m047_rank"] = (
        df.groupby("question_id")
        .cumcount()
        + 1
    )

    return (
        df[
            df["m047_rank"]
            <= TOP_K
        ]
        .copy()
        .reset_index(drop=True)
    )


def build_pairs(df):
    rows = []

    for question_id, group in (
        df.groupby(
            "question_id",
            sort=False,
        )
    ):
        group = group.reset_index(
            drop=True
        )

        for i in range(len(group)):
            for j in range(
                i + 1,
                len(group),
            ):
                tiou_a = float(
                    group.loc[
                        i,
                        "target_tiou",
                    ]
                )

                tiou_b = float(
                    group.loc[
                        j,
                        "target_tiou",
                    ]
                )

                gap = abs(
                    tiou_a
                    - tiou_b
                )

                if gap < MIN_TIOU_GAP:
                    continue

                label = int(
                    tiou_a
                    > tiou_b
                )

                weight = (
                    1.0
                    + gap
                )

                # Especially important:
                # one candidate has zero
                # overlap and the other does not.
                if (
                    (tiou_a == 0)
                    != (tiou_b == 0)
                ):
                    weight *= 1.5

                base = {
                    "question_id":
                        question_id,

                    "transcript_id":
                        group.loc[
                            i,
                            "transcript_id",
                        ],

                    "question":
                        group.loc[
                            i,
                            "question",
                        ],

                    "text_a":
                        group.loc[
                            i,
                            "candidate_text",
                        ],

                    "text_b":
                        group.loc[
                            j,
                            "candidate_text",
                        ],

                    "label":
                        label,

                    "weight":
                        weight,
                }

                rows.append(base)

                # Symmetric reversed pair.
                reverse = dict(base)

                reverse["text_a"] = (
                    base["text_b"]
                )

                reverse["text_b"] = (
                    base["text_a"]
                )

                reverse["label"] = (
                    1 - label
                )

                rows.append(reverse)

    return pd.DataFrame(rows)


def make_input(
    tokenizer,
    questions,
    texts_a,
    texts_b,
    device,
):
    first = [
        f"Question: {q}"
        for q in questions
    ]

    second = [
        (
            f"Candidate A: {a} "
            f"[SEP] Candidate B: {b}"
        )
        for a, b
        in zip(
            texts_a,
            texts_b,
        )
    ]

    encoded = tokenizer(
        first,
        second,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    return {
        key: value.to(device)
        for key, value
        in encoded.items()
    }


def load_model(device):
    tokenizer = (
        AutoTokenizer
        .from_pretrained(
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

    return tokenizer, model


def train_model(
    train_pairs,
    device,
):
    tokenizer, model = (
        load_model(device)
    )

    loader = DataLoader(
        ComparatorDataset(
            train_pairs
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    model.train()

    for epoch in range(
        1,
        EPOCHS + 1,
    ):
        total_loss = 0.0
        batches = 0

        for (
            questions,
            text_a,
            text_b,
            labels,
            weights,
        ) in loader:

            optimizer.zero_grad(
                set_to_none=True
            )

            encoded = make_input(
                tokenizer,
                questions,
                text_a,
                text_b,
                device,
            )

            labels = labels.to(
                device=device,
                dtype=torch.long,
            )

            weights = weights.to(
                device=device,
                dtype=torch.float32,
            )

            logits = model(
                **encoded
            ).logits

            loss_each = (
                torch.nn.functional
                .cross_entropy(
                    logits,
                    labels,
                    reduction="none",
                )
            )

            loss = (
                loss_each
                * weights
            ).mean()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            optimizer.step()

            total_loss += float(
                loss.detach().cpu()
            )

            batches += 1

        print(
            f"  epoch "
            f"{epoch}/{EPOCHS} "
            f"loss="
            f"{total_loss / batches:.6f}"
        )

    return tokenizer, model


def compare(
    tokenizer,
    model,
    device,
    question,
    text_a,
    text_b,
):
    encoded = make_input(
        tokenizer,
        [question],
        [text_a],
        [text_b],
        device,
    )

    with torch.inference_mode():
        probability = (
            torch.softmax(
                model(
                    **encoded
                ).logits,
                dim=-1,
            )[0, 1]
            .detach()
            .cpu()
            .item()
        )

    return float(probability)


def select_between_47_63(
    frame,
    tokenizer,
    model,
    device,
):
    # Rank by M047.
    rank47 = frame.sort_values(
        "oof_score",
        ascending=False,
    )

    row47 = rank47.iloc[0]

    # Rank by M063.
    rank63 = frame.sort_values(
        "m063_oof_score",
        ascending=False,
    )

    row63 = rank63.iloc[0]

    if (
        int(row47["candidate_index"])
        == int(row63["candidate_index"])
    ):
        return (
            row47,
            0.5,
            False,
        )

    p63 = compare(
        tokenizer,
        model,
        device,
        str(row47["question"]),
        str(row63["candidate_text"]),
        str(row47["candidate_text"]),
    )

    if p63 >= 0.5:
        return (
            row63,
            p63,
            True,
        )

    return (
        row47,
        p63,
        False,
    )


def main():
    seed_everything(SEED)

    start_time = (
        time.perf_counter()
    )

    raw = pd.read_csv(INPUT)

    if (
        raw["question_id"]
        .nunique()
        != 195
    ):
        raise RuntimeError(
            "Expected 195 questions."
        )

    top5 = prepare_top5(raw)

    question_table = (
        raw[
            [
                "question_id",
                "transcript_id",
            ]
        ]
        .drop_duplicates(
            "question_id"
        )
        .reset_index(drop=True)
    )

    groups = (
        question_table[
            "transcript_id"
        ]
        .astype(str)
        .to_numpy()
    )

    splitter = GroupKFold(
        n_splits=N_SPLITS
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 78)
    print(
        "Nordic AI Cup 2026 — "
        "M065 Pairwise Span Comparator"
    )
    print("=" * 78)

    print(
        f"Model:  {MODEL_ID}"
    )

    print(
        f"Device: {device}"
    )

    output_rows = []

    for fold, (
        train_idx,
        test_idx,
    ) in enumerate(
        splitter.split(
            question_table,
            groups=groups,
        ),
        start=1,
    ):
        print()
        print("=" * 78)
        print(
            f"FOLD {fold}/{N_SPLITS}"
        )
        print("=" * 78)

        train_ids = set(
            question_table.iloc[
                train_idx
            ]["question_id"]
        )

        test_ids = set(
            question_table.iloc[
                test_idx
            ]["question_id"]
        )

        train_candidates = top5[
            top5["question_id"]
            .isin(train_ids)
        ]

        train_pairs = build_pairs(
            train_candidates
        )

        print(
            f"Train questions: "
            f"{len(train_ids)}"
        )

        print(
            f"Training pairs: "
            f"{len(train_pairs)}"
        )

        tokenizer, model = (
            train_model(
                train_pairs,
                device,
            )
        )

        model.eval()

        fold_rows = []

        for question_id in test_ids:
            group = raw[
                raw["question_id"]
                == question_id
            ]

            chosen, probability, switched = (
                select_between_47_63(
                    group,
                    tokenizer,
                    model,
                    device,
                )
            )

            row47 = (
                group.sort_values(
                    "oof_score",
                    ascending=False,
                )
                .iloc[0]
            )

            row63 = (
                group.sort_values(
                    "m063_oof_score",
                    ascending=False,
                )
                .iloc[0]
            )

            fold_rows.append(
                {
                    "question_id":
                        question_id,

                    "transcript_id":
                        chosen[
                            "transcript_id"
                        ],

                    "candidate_index":
                        int(
                            chosen[
                                "candidate_index"
                            ]
                        ),

                    "candidate_text":
                        chosen[
                            "candidate_text"
                        ],

                    "target_tiou":
                        float(
                            chosen[
                                "target_tiou"
                            ]
                        ),

                    "m047_tiou":
                        float(
                            row47[
                                "target_tiou"
                            ]
                        ),

                    "m063_tiou":
                        float(
                            row63[
                                "target_tiou"
                            ]
                        ),

                    "m065_probability":
                        probability,

                    "m065_switched_to_m063":
                        switched,

                    "fold":
                        fold,
                }
            )

        fold_df = pd.DataFrame(
            fold_rows
        )

        baseline = float(
            fold_df[
                "m047_tiou"
            ].mean()
        )

        m063 = float(
            fold_df[
                "m063_tiou"
            ].mean()
        )

        m065 = float(
            fold_df[
                "target_tiou"
            ].mean()
        )

        print()
        print(
            f"Fold {fold}:"
        )

        print(
            f"  M047: "
            f"{baseline:.4f}"
        )

        print(
            f"  M063: "
            f"{m063:.4f}"
        )

        print(
            f"  M065: "
            f"{m065:.4f}"
        )

        print(
            f"  delta: "
            f"{m065-baseline:+.4f}"
        )

        print(
            f"  switched: "
            f"{int(fold_df.m065_switched_to_m063.sum())}"
        )

        output_rows.append(
            fold_df
        )

        del model
        del tokenizer

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    result = pd.concat(
        output_rows,
        ignore_index=True,
    )

    result = result.sort_values(
        "question_id"
    )

    mean_tiou = float(
        result[
            "target_tiou"
        ].mean()
    )

    baseline = float(
        result[
            "m047_tiou"
        ].mean()
    )

    m063 = float(
        result[
            "m063_tiou"
        ].mean()
    )

    oracle = float(
        result[
            [
                "m047_tiou",
                "m063_tiou",
            ]
        ]
        .max(axis=1)
        .mean()
    )

    switched = (
        result[
            "m065_switched_to_m063"
        ]
        .astype(bool)
    )

    useful = int(
        (
            switched
            & (
                result[
                    "m063_tiou"
                ]
                > result[
                    "m047_tiou"
                ]
            )
        ).sum()
    )

    harmful = int(
        (
            switched
            & (
                result[
                    "m063_tiou"
                ]
                < result[
                    "m047_tiou"
                ]
            )
        ).sum()
    )

    neutral = int(
        (
            switched
            & np.isclose(
                result[
                    "m063_tiou"
                ],
                result[
                    "m047_tiou"
                ],
            )
        ).sum()
    )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT,
        index=False,
    )

    print()
    print("=" * 78)
    print(
        "M065 FINAL RESULT"
    )
    print("=" * 78)

    print(
        f"M047:             "
        f"{baseline:.4f}"
    )

    print(
        f"M063:             "
        f"{m063:.4f}"
    )

    print(
        f"M064 HGB:         "
        f"0.5271"
    )

    print(
        f"M065:             "
        f"{mean_tiou:.4f}"
    )

    print(
        f"Gain vs M047:     "
        f"{mean_tiou-baseline:+.4f}"
    )

    print(
        f"Switched to M063: "
        f"{int(switched.sum())}"
    )

    print(
        f"Useful switches:  "
        f"{useful}"
    )

    print(
        f"Harmful switches: "
        f"{harmful}"
    )

    print(
        f"Neutral switches: "
        f"{neutral}"
    )

    print(
        f"Oracle:            "
        f"{oracle:.4f}"
    )

    print(
        f"Oracle captured:   "
        f"{(
            (mean_tiou-baseline)
            /
            max(
                oracle-baseline,
                1e-12,
            )
        ):.1%}"
    )

    print(
        f"Wall time:         "
        f"{time.perf_counter()-start_time:.1f}s"
    )

    print()
    print(
        f"Saved: {OUTPUT}"
    )


if __name__ == "__main__":
    main()