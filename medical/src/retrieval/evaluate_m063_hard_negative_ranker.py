from __future__ import annotations

import gc
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import GroupKFold
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


# =============================================================================
# Configuration
# =============================================================================

INPUT = Path(
    r"medical\artifacts\retrieval"
    r"\m047_all_questions_candidates_oof.csv"
)

OOF_OUTPUT = Path(
    r"medical\artifacts\retrieval"
    r"\m063_hard_negative_ranker_oof.csv"
)

ALL_SCORE_OUTPUT = Path(
    r"medical\artifacts\retrieval"
    r"\m063_all_candidate_oof_scores.csv"
)

BASE_MODEL = (
    "cross-encoder/ms-marco-MiniLM-L6-v2"
)

TOP_K = 20
N_SPLITS = 5

SEED = 42

EPOCHS = 4
LEARNING_RATE = 1.0e-5
WEIGHT_DECAY = 0.01

PAIR_BATCH_SIZE = 16
INFERENCE_BATCH_SIZE = 64

MAX_LENGTH = 192

# Pair-generation rules.
MIN_GOOD_TIOU = 0.50
MIN_TIOU_GAP = 0.30

MAX_GOOD_PER_QUESTION = 5
MAX_BAD_PER_GOOD = 12

# Extra emphasis on catastrophic current-model mistakes.
CATASTROPHIC_TIOU = 0.0
CATASTROPHIC_TOP_RANK = 10

# Pairwise logistic loss temperature.
TEMPERATURE = 1.0


# =============================================================================
# Reproducibility
# =============================================================================

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )


# =============================================================================
# Pair Dataset
# =============================================================================

class PairDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
    ) -> None:
        self.df = dataframe.reset_index(
            drop=True
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(
        self,
        index: int,
    ):
        row = self.df.iloc[
            index
        ]

        return (
            str(row["question"]),
            str(row["good_text"]),
            str(row["bad_text"]),
            float(row["pair_weight"]),
        )


# =============================================================================
# Candidate preparation
# =============================================================================

def get_topk_candidates(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    positive = dataframe[
        dataframe["question_type"]
        == "positive"
    ].copy()

    if (
        positive["question_id"]
        .nunique()
        != 195
    ):
        raise RuntimeError(
            "Expected 195 positive "
            "questions."
        )

    top = (
        positive.sort_values(
            [
                "question_id",
                "oof_score",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .groupby(
            "question_id",
            sort=False,
        )
        .head(TOP_K)
        .copy()
        .reset_index(drop=True)
    )

    top[
        "m047_rank"
    ] = (
        top.groupby(
            "question_id"
        )
        .cumcount()
        + 1
    )

    return top


# =============================================================================
# Hard-pair construction
# =============================================================================

def build_pairs(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for question_id, group in (
        dataframe.groupby(
            "question_id",
            sort=False,
        )
    ):
        group = (
            group.sort_values(
                "target_tiou",
                ascending=False,
            )
            .copy()
        )

        good = group[
            group["target_tiou"]
            >= MIN_GOOD_TIOU
        ].head(
            MAX_GOOD_PER_QUESTION
        )

        # If top-20 contains no >= 0.50
        # candidate, still learn from the
        # best available candidate.
        if len(good) == 0:
            good = group.head(1)

        for _, good_row in (
            good.iterrows()
        ):
            good_tiou = float(
                good_row[
                    "target_tiou"
                ]
            )

            # Candidate must be meaningfully
            # worse than the good span.
            bad = group[
                group["target_tiou"]
                <= (
                    good_tiou
                    - MIN_TIOU_GAP
                )
            ].copy()

            if len(bad) == 0:
                continue

            # Most important negatives are
            # those the existing M047 model
            # believes are strongest.
            bad = bad.sort_values(
                [
                    "oof_score",
                    "m047_rank",
                ],
                ascending=[
                    False,
                    True,
                ],
            )

            selected_bad = (
                bad.head(
                    MAX_BAD_PER_GOOD
                )
            )

            for _, bad_row in (
                selected_bad.iterrows()
            ):
                bad_tiou = float(
                    bad_row[
                        "target_tiou"
                    ]
                )

                gap = (
                    good_tiou
                    - bad_tiou
                )

                # Base weight grows with how
                # different the gold quality is.
                pair_weight = (
                    1.0
                    + gap
                )

                # Give additional weight to
                # catastrophic high-ranked
                # current-model mistakes.
                catastrophic = (
                    bad_tiou
                    <= CATASTROPHIC_TIOU
                    and int(
                        bad_row[
                            "m047_rank"
                        ]
                    )
                    <= CATASTROPHIC_TOP_RANK
                )

                if catastrophic:
                    pair_weight *= 2.0

                rows.append(
                    {
                        "question_id":
                            question_id,

                        "transcript_id":
                            good_row[
                                "transcript_id"
                            ],

                        "question":
                            good_row[
                                "question"
                            ],

                        "good_index":
                            int(
                                good_row[
                                    "candidate_index"
                                ]
                            ),

                        "bad_index":
                            int(
                                bad_row[
                                    "candidate_index"
                                ]
                            ),

                        "good_text":
                            good_row[
                                "candidate_text"
                            ],

                        "bad_text":
                            bad_row[
                                "candidate_text"
                            ],

                        "good_tiou":
                            good_tiou,

                        "bad_tiou":
                            bad_tiou,

                        "tiou_gap":
                            gap,

                        "bad_m047_score":
                            float(
                                bad_row[
                                    "oof_score"
                                ]
                            ),

                        "bad_m047_rank":
                            int(
                                bad_row[
                                    "m047_rank"
                                ]
                            ),

                        "catastrophic":
                            catastrophic,

                        "pair_weight":
                            pair_weight,
                    }
                )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# Model scoring
# =============================================================================

def encode_pairs(
    tokenizer,
    questions,
    candidates,
    device,
):
    encoded = tokenizer(
        list(questions),
        list(candidates),
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


def model_scores(
    model,
    encoded,
):
    logits = model(
        **encoded
    ).logits

    if logits.ndim == 2:
        if logits.shape[1] == 1:
            logits = logits[:, 0]
        else:
            # Defensive fallback.
            logits = logits[:, -1]

    return logits.reshape(-1)


# =============================================================================
# Training
# =============================================================================

def train_fold(
    train_pairs: pd.DataFrame,
    device: str,
    fold: int,
):
    print(
        f"Loading {BASE_MODEL} "
        f"for fold {fold}..."
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            BASE_MODEL
        )
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            BASE_MODEL
        )
        .to(device)
    )

    model.train()

    loader = DataLoader(
        PairDataset(
            train_pairs
        ),
        batch_size=PAIR_BATCH_SIZE,
        shuffle=True,
        drop_last=False,
    )

    optimizer = (
        torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )
    )

    total_steps = (
        len(loader)
        * EPOCHS
    )

    print(
        f"Training pairs: "
        f"{len(train_pairs)}"
    )

    print(
        f"Catastrophic pairs: "
        f"{int(train_pairs.catastrophic.sum())}"
    )

    print(
        f"Steps: {total_steps}"
    )

    for epoch in range(
        1,
        EPOCHS + 1,
    ):
        epoch_loss = 0.0
        n_batches = 0

        for (
            questions,
            good_texts,
            bad_texts,
            weights,
        ) in loader:
            optimizer.zero_grad(
                set_to_none=True
            )

            good_encoded = (
                encode_pairs(
                    tokenizer,
                    questions,
                    good_texts,
                    device,
                )
            )

            bad_encoded = (
                encode_pairs(
                    tokenizer,
                    questions,
                    bad_texts,
                    device,
                )
            )

            good_scores = (
                model_scores(
                    model,
                    good_encoded,
                )
            )

            bad_scores = (
                model_scores(
                    model,
                    bad_encoded,
                )
            )

            weights = (
                weights
                .to(
                    device=device,
                    dtype=torch.float32,
                )
            )

            difference = (
                good_scores
                - bad_scores
            ) / TEMPERATURE

            # RankNet / pairwise logistic loss:
            #
            #   -log sigmoid(good - bad)
            #
            per_example = (
                F.softplus(
                    -difference
                )
            )

            loss = (
                per_example
                * weights
            ).mean()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            epoch_loss += float(
                loss.detach().cpu()
            )

            n_batches += 1

        mean_loss = (
            epoch_loss
            / max(
                n_batches,
                1,
            )
        )

        print(
            f"  epoch "
            f"{epoch}/{EPOCHS} "
            f"loss="
            f"{mean_loss:.6f}"
        )

    return (
        tokenizer,
        model,
    )


# =============================================================================
# Inference
# =============================================================================

def score_candidates(
    dataframe: pd.DataFrame,
    tokenizer,
    model,
    device: str,
) -> np.ndarray:
    model.eval()

    scores = []

    for start in range(
        0,
        len(dataframe),
        INFERENCE_BATCH_SIZE,
    ):
        end = min(
            start
            + INFERENCE_BATCH_SIZE,
            len(dataframe),
        )

        batch = dataframe.iloc[
            start:end
        ]

        encoded = encode_pairs(
            tokenizer,
            batch[
                "question"
            ].astype(str).tolist(),
            batch[
                "candidate_text"
            ].astype(str).tolist(),
            device,
        )

        with torch.inference_mode():
            batch_scores = (
                model_scores(
                    model,
                    encoded,
                )
            )

        scores.extend(
            batch_scores
            .detach()
            .cpu()
            .numpy()
            .tolist()
        )

    return np.asarray(
        scores,
        dtype=float,
    )


# =============================================================================
# Evaluation
# =============================================================================

def select_best(
    dataframe: pd.DataFrame,
    score_column: str,
):
    indices = (
        dataframe.groupby(
            "question_id"
        )[
            score_column
        ]
        .idxmax()
    )

    return (
        dataframe.loc[
            indices
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )


def summarize(
    name: str,
    dataframe: pd.DataFrame,
    score_column: str,
):
    selected = (
        select_best(
            dataframe,
            score_column,
        )
    )

    values = (
        selected[
            "target_tiou"
        ]
        .astype(float)
    )

    metrics = {
        "mean_tiou":
            float(
                values.mean()
            ),

        "median_tiou":
            float(
                values.median()
            ),

        "any_overlap":
            float(
                (values > 0).mean()
            ),

        "tiou_ge_025":
            float(
                (
                    values >= 0.25
                ).mean()
            ),

        "tiou_ge_050":
            float(
                (
                    values >= 0.50
                ).mean()
            ),

        "tiou_ge_075":
            float(
                (
                    values >= 0.75
                ).mean()
            ),
    }

    print()
    print(name)
    print("-" * 78)

    print(
        f"Mean tIoU:       "
        f"{metrics['mean_tiou']:.4f}"
    )

    print(
        f"Median tIoU:     "
        f"{metrics['median_tiou']:.4f}"
    )

    print(
        f"Any overlap:     "
        f"{metrics['any_overlap']:.4f}"
    )

    print(
        f"tIoU >= 0.25:    "
        f"{metrics['tiou_ge_025']:.4f}"
    )

    print(
        f"tIoU >= 0.50:    "
        f"{metrics['tiou_ge_050']:.4f}"
    )

    print(
        f"tIoU >= 0.75:    "
        f"{metrics['tiou_ge_075']:.4f}"
    )

    return (
        metrics,
        selected,
    )


# =============================================================================
# Main
# =============================================================================

def main():
    seed_everything(
        SEED
    )

    wall_start = (
        time.perf_counter()
    )

    if not INPUT.exists():
        raise FileNotFoundError(
            INPUT
        )

    raw = pd.read_csv(
        INPUT
    )

    top = get_topk_candidates(
        raw
    )

    print("=" * 78)
    print(
        "Nordic AI Cup 2026 — "
        "M063 Hard-Negative Evidence Ranker"
    )
    print("=" * 78)

    print(
        f"Base model:     "
        f"{BASE_MODEL}"
    )

    print(
        f"Questions:      "
        f"{top.question_id.nunique()}"
    )

    print(
        f"Candidate pool: "
        f"{len(top)}"
    )

    print(
        f"Top K:          "
        f"{TOP_K}"
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device:         "
        f"{device}"
    )

    # ---------------------------------------------------------
    # Pair statistics before CV.
    # ---------------------------------------------------------

    all_pairs = build_pairs(
        top
    )

    print()
    print("PAIR DATA")
    print("-" * 78)

    print(
        f"Total pairs:        "
        f"{len(all_pairs)}"
    )

    print(
        f"Catastrophic pairs: "
        f"{int(all_pairs.catastrophic.sum())}"
    )

    print(
        f"Mean tIoU gap:      "
        f"{all_pairs.tiou_gap.mean():.4f}"
    )

    print(
        f"Median tIoU gap:    "
        f"{all_pairs.tiou_gap.median():.4f}"
    )

    # ---------------------------------------------------------
    # References.
    # ---------------------------------------------------------

    summarize(
        "M047 TOP-1 REFERENCE",
        top,
        "oof_score",
    )

    top[
        "_oracle"
    ] = (
        top[
            "target_tiou"
        ]
    )

    summarize(
        "TOP-20 ORACLE",
        top,
        "_oracle",
    )

    # ---------------------------------------------------------
    # Conversation-disjoint CV.
    # ---------------------------------------------------------

    question_table = (
        top[
            [
                "question_id",
                "transcript_id",
            ]
        ]
        .drop_duplicates(
            "question_id"
        )
        .reset_index(
            drop=True
        )
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

    top[
        "m063_oof_score"
    ] = np.nan

    print()
    print("=" * 78)
    print(
        "M063 CONVERSATION-DISJOINT CV"
    )
    print("=" * 78)

    for fold, (
        train_q_idx,
        test_q_idx,
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
                train_q_idx
            ][
                "question_id"
            ]
        )

        test_ids = set(
            question_table.iloc[
                test_q_idx
            ][
                "question_id"
            ]
        )

        train_candidates = top[
            top[
                "question_id"
            ].isin(
                train_ids
            )
        ].copy()

        test_candidates = top[
            top[
                "question_id"
            ].isin(
                test_ids
            )
        ].copy()

        train_pairs = build_pairs(
            train_candidates
        )

        print(
            f"Train questions: "
            f"{len(train_ids)}"
        )

        print(
            f"Test questions:  "
            f"{len(test_ids)}"
        )

        tokenizer, model = (
            train_fold(
                train_pairs,
                device,
                fold,
            )
        )

        fold_scores = (
            score_candidates(
                test_candidates,
                tokenizer,
                model,
                device,
            )
        )

        top.loc[
            test_candidates.index,
            "m063_oof_score",
        ] = fold_scores

        temp = (
            test_candidates.copy()
        )

        temp[
            "m063_score"
        ] = fold_scores

        baseline_selected = (
            select_best(
                temp,
                "oof_score",
            )
        )

        m063_selected = (
            select_best(
                temp,
                "m063_score",
            )
        )

        baseline_mean = float(
            baseline_selected[
                "target_tiou"
            ].mean()
        )

        m063_mean = float(
            m063_selected[
                "target_tiou"
            ].mean()
        )

        base_zero = int(
            (
                baseline_selected[
                    "target_tiou"
                ]
                == 0
            ).sum()
        )

        new_zero = int(
            (
                m063_selected[
                    "target_tiou"
                ]
                == 0
            ).sum()
        )

        print()
        print(
            f"Fold {fold} result:"
        )

        print(
            f"  M047:       "
            f"{baseline_mean:.4f}"
        )

        print(
            f"  M063:       "
            f"{m063_mean:.4f}"
        )

        print(
            f"  delta:      "
            f"{m063_mean-baseline_mean:+.4f}"
        )

        print(
            f"  zero spans: "
            f"{base_zero} -> {new_zero}"
        )

        # Free VRAM before next fold.
        del model
        del tokenizer

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if (
        top[
            "m063_oof_score"
        ]
        .isna()
        .any()
    ):
        raise RuntimeError(
            "Missing M063 OOF scores."
        )

    # ---------------------------------------------------------
    # Final OOF metrics.
    # ---------------------------------------------------------

    metrics, selected = summarize(
        "M063 HARD-NEGATIVE OOF",
        top,
        "m063_oof_score",
    )

    baseline_selected = (
        select_best(
            top,
            "oof_score",
        )
        .set_index(
            "question_id"
        )
        .sort_index()
    )

    selected_lookup = (
        selected
        .set_index(
            "question_id"
        )
        .reindex(
            baseline_selected.index
        )
    )

    base_zero = (
        baseline_selected[
            "target_tiou"
        ]
        == 0
    )

    new_zero = (
        selected_lookup[
            "target_tiou"
        ]
        == 0
    )

    zero_fixed = int(
        (
            base_zero
            & (~new_zero)
        ).sum()
    )

    zero_created = int(
        (
            (~base_zero)
            & new_zero
        ).sum()
    )

    changed = int(
        (
            baseline_selected[
                "candidate_index"
            ]
            != selected_lookup[
                "candidate_index"
            ]
        ).sum()
    )

    # ---------------------------------------------------------
    # Save.
    # ---------------------------------------------------------

    ALL_SCORE_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    top.to_csv(
        ALL_SCORE_OUTPUT,
        index=False,
    )

    selected.to_csv(
        OOF_OUTPUT,
        index=False,
    )

    elapsed = (
        time.perf_counter()
        - wall_start
    )

    print()
    print("=" * 78)
    print("M063 FINAL RESULT")
    print("=" * 78)

    print(
        f"M047 mean tIoU:       "
        f"0.5158"
    )

    print(
        f"M063 mean tIoU:       "
        f"{metrics['mean_tiou']:.4f}"
    )

    print(
        f"Gain vs M047:         "
        f"{metrics['mean_tiou'] - 0.5158:+.4f}"
    )

    print(
        f"Selections changed:   "
        f"{changed}"
    )

    print(
        f"Zero-overlap fixed:    "
        f"{zero_fixed}"
    )

    print(
        f"Zero-overlap created:  "
        f"{zero_created}"
    )

    print(
        f"Top-20 oracle:        "
        f"0.7966"
    )

    print(
        f"Remaining oracle gap:  "
        f"{0.7966 - metrics['mean_tiou']:+.4f}"
    )

    print(
        f"Wall time:             "
        f"{elapsed:.1f} s"
    )

    print()
    print(
        f"Selected OOF: "
        f"{OOF_OUTPUT}"
    )

    print(
        f"All OOF scores: "
        f"{ALL_SCORE_OUTPUT}"
    )


if __name__ == "__main__":
    main()