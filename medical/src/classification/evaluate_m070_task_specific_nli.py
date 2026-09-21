from __future__ import annotations

import gc
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


# ============================================================================
# Paths
# ============================================================================

QUESTIONS = Path(
    Path(__file__).resolve().parents[3].parent
    / "Nordic-AI-Cup-2026-official"
    / "medical-appointment"
    / "data"
    / "question_train.csv"
)

CANDIDATES = Path(
    r"medical\artifacts\retrieval"
    r"\m047_all_questions_candidates_oof.csv"
)

M055 = Path(
    r"medical\artifacts\classification"
    r"\m055_base_nli_runtime_signal_oof.csv"
)

OUTPUT = Path(
    r"medical\artifacts\classification"
    r"\m070_task_specific_nli_oof.csv"
)


# ============================================================================
# Configuration
# ============================================================================

MODEL_ID = "cross-encoder/nli-deberta-v3-base"

SEED = 42
N_SPLITS = 5

TOP_K = 12

EPOCHS = 3
LEARNING_RATE = 5e-6
WEIGHT_DECAY = 0.01

TRAIN_BATCH_SIZE = 8
INFERENCE_BATCH_SIZE = 32

MAX_LENGTH = 192

# Positive evidence examples.
POSITIVE_TIOU = 0.50
MAX_POSITIVE_PER_QUESTION = 4

# Hard incorrect evidence for positive questions.
NEGATIVE_TIOU = 0.10
MAX_NEGATIVE_PER_POSITIVE_QUESTION = 8

# For hard-negative/off-topic questions.
MAX_NEGATIVE_PER_NEGATIVE_QUESTION = 8


# ============================================================================
# Utilities
# ============================================================================

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_bool(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    return (
        str(value)
        .strip()
        .lower()
        in {"true", "1", "yes"}
    )


def competition_score(
    gold: np.ndarray,
    pred: np.ndarray,
    evidence: np.ndarray,
):
    accuracy = float(
        np.mean(
            gold == pred
        )
    )

    positive = gold

    scored_tiou = float(
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
        + 0.6 * scored_tiou
    )

    return (
        accuracy,
        scored_tiou,
        composite,
    )


# ============================================================================
# Dataset
# ============================================================================

class EvidencePairDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
    ):
        self.df = dataframe.reset_index(
            drop=True
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        return (
            str(row["question"]),
            str(row["candidate_text"]),
            float(row["target"]),
            float(row["weight"]),
        )


# ============================================================================
# Candidate preparation
# ============================================================================

def prepare_candidates(
    raw: pd.DataFrame,
) -> pd.DataFrame:
    df = raw.copy()

    df = df.sort_values(
        [
            "question_id",
            "oof_score",
        ],
        ascending=[
            True,
            False,
        ],
    )

    df["rank"] = (
        df.groupby(
            "question_id"
        )
        .cumcount()
        + 1
    )

    df = df[
        df["rank"]
        <= TOP_K
    ].copy()

    return df.reset_index(
        drop=True
    )


# ============================================================================
# Training-example construction
# ============================================================================

def build_training_pairs(
    candidates: pd.DataFrame,
    allowed_questions: set[str],
) -> pd.DataFrame:

    df = candidates[
        candidates["question_id"]
        .isin(
            allowed_questions
        )
    ].copy()

    rows = []

    for question_id, group in (
        df.groupby(
            "question_id",
            sort=False,
        )
    ):
        question_type = str(
            group.iloc[0][
                "question_type"
            ]
        )

        question = str(
            group.iloc[0][
                "question"
            ]
        )

        transcript_id = str(
            group.iloc[0][
                "transcript_id"
            ]
        )

        # ============================================================
        # Positive question
        # ============================================================

        if question_type == "positive":

            positive = (
                group[
                    group["target_tiou"]
                    >= POSITIVE_TIOU
                ]
                .sort_values(
                    "target_tiou",
                    ascending=False,
                )
                .head(
                    MAX_POSITIVE_PER_QUESTION
                )
            )

            # Some questions may have no >= .50 span in top-K.
            # Still allow a decent best candidate so we don't discard
            # the whole question.
            if len(positive) == 0:
                best = (
                    group.sort_values(
                        "target_tiou",
                        ascending=False,
                    )
                    .head(1)
                )

                if (
                    len(best)
                    and float(
                        best.iloc[0][
                            "target_tiou"
                        ]
                    )
                    >= 0.25
                ):
                    positive = best

            for _, row in positive.iterrows():

                tiou = float(
                    row[
                        "target_tiou"
                    ]
                )

                rows.append(
                    {
                        "question_id":
                            question_id,

                        "transcript_id":
                            transcript_id,

                        "question":
                            question,

                        "candidate_text":
                            str(
                                row[
                                    "candidate_text"
                                ]
                            ),

                        "target":
                            1.0,

                        "weight":
                            1.0
                            + tiou,

                        "source":
                            "positive_evidence",
                    }
                )

            # Hard incorrect candidates that M047 ranked highly.
            negative = (
                group[
                    group["target_tiou"]
                    <= NEGATIVE_TIOU
                ]
                .sort_values(
                    "oof_score",
                    ascending=False,
                )
                .head(
                    MAX_NEGATIVE_PER_POSITIVE_QUESTION
                )
            )

            for _, row in negative.iterrows():

                rank = int(
                    row["rank"]
                )

                # Rank-1 catastrophic failures are exactly what we
                # especially want the model to learn from.
                weight = (
                    1.75
                    if rank <= 2
                    else 1.25
                )

                rows.append(
                    {
                        "question_id":
                            question_id,

                        "transcript_id":
                            transcript_id,

                        "question":
                            question,

                        "candidate_text":
                            str(
                                row[
                                    "candidate_text"
                                ]
                            ),

                        "target":
                            0.0,

                        "weight":
                            weight,

                        "source":
                            "positive_hard_negative",
                    }
                )

        # ============================================================
        # Hard-negative / off-topic question
        # ============================================================

        else:
            negative = (
                group.sort_values(
                    "oof_score",
                    ascending=False,
                )
                .head(
                    MAX_NEGATIVE_PER_NEGATIVE_QUESTION
                )
            )

            for _, row in negative.iterrows():

                rank = int(
                    row["rank"]
                )

                weight = (
                    1.75
                    if rank <= 2
                    else 1.25
                )

                rows.append(
                    {
                        "question_id":
                            question_id,

                        "transcript_id":
                            transcript_id,

                        "question":
                            question,

                        "candidate_text":
                            str(
                                row[
                                    "candidate_text"
                                ]
                            ),

                        "target":
                            0.0,

                        "weight":
                            weight,

                        "source":
                            question_type,
                    }
                )

    result = pd.DataFrame(
        rows
    )

    if len(result) == 0:
        raise RuntimeError(
            "No M070 training examples built."
        )

    return result


# ============================================================================
# Model helpers
# ============================================================================

def get_label_indices(model):
    mapping = {
        int(k): str(v).lower()
        for k, v
        in model.config.id2label.items()
    }

    entailment = None
    contradiction = None
    neutral = None

    for index, label in mapping.items():

        if "entail" in label:
            entailment = index

        elif "contrad" in label:
            contradiction = index

        elif "neutral" in label:
            neutral = index

    if (
        entailment is None
        or contradiction is None
        or neutral is None
    ):
        raise RuntimeError(
            f"Could not identify NLI labels: "
            f"{mapping}"
        )

    return (
        entailment,
        contradiction,
        neutral,
    )


def binary_entailment_logit(
    logits: torch.Tensor,
    entailment_index: int,
    contradiction_index: int,
    neutral_index: int,
):
    entailment = logits[
        :,
        entailment_index,
    ]

    other = torch.stack(
        [
            logits[
                :,
                contradiction_index,
            ],
            logits[
                :,
                neutral_index,
            ],
        ],
        dim=1,
    )

    non_entailment = (
        torch.logsumexp(
            other,
            dim=1,
        )
    )

    return (
        entailment
        - non_entailment
    )


def encode(
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
        key: value.to(
            device
        )
        for key, value
        in encoded.items()
    }


# ============================================================================
# Fold training
# ============================================================================

def train_fold(
    training_pairs: pd.DataFrame,
    device: str,
    fold: int,
):
    print()
    print(
        f"Loading {MODEL_ID} "
        f"for fold {fold}..."
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_ID
        )
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            MODEL_ID
        )
        .to(device)
    )

    (
        entailment_index,
        contradiction_index,
        neutral_index,
    ) = get_label_indices(
        model
    )

    print(
        "Labels:",
        model.config.id2label,
    )

    dataset = EvidencePairDataset(
        training_pairs
    )

    loader = DataLoader(
        dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        drop_last=False,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    model.train()

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        epoch_loss = 0.0
        batches = 0

        for (
            questions,
            candidate_texts,
            targets,
            weights,
        ) in loader:

            optimizer.zero_grad(
                set_to_none=True
            )

            encoded = encode(
                tokenizer,
                questions,
                candidate_texts,
                device,
            )

            logits = model(
                **encoded
            ).logits

            binary_logits = (
                binary_entailment_logit(
                    logits,
                    entailment_index,
                    contradiction_index,
                    neutral_index,
                )
            )

            targets = targets.to(
                device=device,
                dtype=torch.float32,
            )

            weights = weights.to(
                device=device,
                dtype=torch.float32,
            )

            loss_each = (
                F.binary_cross_entropy_with_logits(
                    binary_logits,
                    targets,
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
                max_norm=1.0,
            )

            optimizer.step()

            epoch_loss += float(
                loss.detach().cpu()
            )

            batches += 1

        print(
            f"  epoch {epoch}/{EPOCHS} "
            f"loss="
            f"{epoch_loss/max(batches,1):.6f}"
        )

    return (
        tokenizer,
        model,
        entailment_index,
        contradiction_index,
        neutral_index,
    )


# ============================================================================
# Candidate inference
# ============================================================================

def score_candidates(
    dataframe: pd.DataFrame,
    tokenizer,
    model,
    entailment_index,
    contradiction_index,
    neutral_index,
    device,
):
    result = np.zeros(
        len(dataframe),
        dtype=float,
    )

    model.eval()

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

        encoded = encode(
            tokenizer,
            batch["question"]
            .astype(str)
            .tolist(),
            batch["candidate_text"]
            .astype(str)
            .tolist(),
            device,
        )

        with torch.inference_mode():

            logits = model(
                **encoded
            ).logits

            binary_logits = (
                binary_entailment_logit(
                    logits,
                    entailment_index,
                    contradiction_index,
                    neutral_index,
                )
            )

            probability = (
                torch.sigmoid(
                    binary_logits
                )
            )

        result[
            start:end
        ] = (
            probability
            .detach()
            .cpu()
            .float()
            .numpy()
        )

    return result


# ============================================================================
# Aggregate candidate scores into question signal
# ============================================================================

def aggregate_questions(
    scored_candidates: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for question_id, group in (
        scored_candidates.groupby(
            "question_id",
            sort=False,
        )
    ):

        scores = np.sort(
            group[
                "m070_candidate_score"
            ]
            .to_numpy(
                dtype=float
            )
        )[::-1]

        max_score = float(
            scores[0]
        )

        second = (
            float(scores[1])
            if len(scores) >= 2
            else max_score
        )

        top3_mean = float(
            np.mean(
                scores[
                    :min(
                        3,
                        len(scores),
                    )
                ]
            )
        )

        rows.append(
            {
                "question_id":
                    question_id,

                "transcript_id":
                    str(
                        group.iloc[0][
                            "transcript_id"
                        ]
                    ),

                "m070_max":
                    max_score,

                "m070_second":
                    second,

                "m070_gap":
                    max_score
                    - second,

                "m070_top3_mean":
                    top3_mean,

                "m070_mean":
                    float(
                        np.mean(
                            scores
                        )
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================================
# Threshold / fusion helpers
# ============================================================================

def threshold_candidates(
    probabilities: np.ndarray,
):
    values = np.unique(
        probabilities
    )

    if len(values) == 1:
        return np.asarray(
            [
                values[0] - 1e-9,
                values[0] + 1e-9,
            ]
        )

    mids = (
        values[:-1]
        + values[1:]
    ) / 2.0

    return np.concatenate(
        [
            [
                values[0] - 1e-9
            ],
            mids,
            [
                values[-1] + 1e-9
            ],
        ]
    )


def best_threshold(
    scores,
    gold,
    evidence,
):
    best_t = 0.5
    best_comp = -1.0

    for threshold in (
        threshold_candidates(
            scores
        )
    ):

        pred = (
            scores
            >= threshold
        )

        _, _, composite = (
            competition_score(
                gold,
                pred,
                evidence,
            )
        )

        if composite > best_comp:
            best_comp = composite
            best_t = float(
                threshold
            )

    return (
        best_t,
        best_comp,
    )


# ============================================================================
# Main
# ============================================================================

def main():
    seed_everything(
        SEED
    )

    wall_start = (
        time.perf_counter()
    )

    questions = pd.read_csv(
        QUESTIONS
    )

    questions[
        "gold_bool"
    ] = [
        parse_bool(x)
        for x
        in questions[
            "answer"
        ]
    ]

    raw_candidates = (
        pd.read_csv(
            CANDIDATES
        )
    )

    candidates = (
        prepare_candidates(
            raw_candidates
        )
    )

    m055 = pd.read_csv(
        M055
    )[
        [
            "question_id",
            "m055_probability",
            "m055_predicted_yes",
            "m055_threshold",
            "evidence_tiou",
        ]
    ].copy()

    base = (
        questions[
            [
                "question_id",
                "transcript_id",
                "question",
                "question_type",
                "gold_bool",
            ]
        ]
        .merge(
            m055,
            on="question_id",
            validate="one_to_one",
        )
    )

    if len(base) != 390:
        raise RuntimeError(
            f"Expected 390 questions, "
            f"got {len(base)}"
        )

    groups = (
        base[
            "transcript_id"
        ]
        .astype(str)
        .to_numpy()
    )

    splitter = GroupKFold(
        n_splits=N_SPLITS
    )

    candidates[
        "m070_candidate_score"
    ] = np.nan

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 78)
    print(
        "Nordic AI Cup 2026 — "
        "M070 Challenge-Specific NLI Fine-Tune"
    )
    print("=" * 78)

    print(
        f"Model:       {MODEL_ID}"
    )

    print(
        f"Device:      {device}"
    )

    print(
        f"Questions:   {len(base)}"
    )

    print(
        f"Candidates:  {len(candidates)}"
    )

    print(
        f"Top K:       {TOP_K}"
    )

    # =================================================================
    # Conversation-disjoint fine-tuning.
    # =================================================================

    for fold, (
        train_idx,
        test_idx,
    ) in enumerate(
        splitter.split(
            base,
            base[
                "gold_bool"
            ],
            groups,
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
            base.iloc[
                train_idx
            ]["question_id"]
        )

        test_ids = set(
            base.iloc[
                test_idx
            ]["question_id"]
        )

        pairs = build_training_pairs(
            candidates,
            train_ids,
        )

        print(
            f"Train questions: "
            f"{len(train_ids)}"
        )

        print(
            f"Test questions:  "
            f"{len(test_ids)}"
        )

        print(
            f"Training pairs:  "
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

        (
            tokenizer,
            model,
            entailment_index,
            contradiction_index,
            neutral_index,
        ) = train_fold(
            pairs,
            device,
            fold,
        )

        test_mask = (
            candidates[
                "question_id"
            ].isin(
                test_ids
            )
        )

        test_candidates = (
            candidates.loc[
                test_mask
            ].copy()
        )

        scores = score_candidates(
            test_candidates,
            tokenizer,
            model,
            entailment_index,
            contradiction_index,
            neutral_index,
            device,
        )

        candidates.loc[
            test_mask,
            "m070_candidate_score",
        ] = scores

        del model
        del tokenizer

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if (
        candidates[
            "m070_candidate_score"
        ]
        .isna()
        .any()
    ):
        raise RuntimeError(
            "Missing M070 OOF candidate scores."
        )

    # =================================================================
    # Question-level OOF signal.
    # =================================================================

    q_scores = aggregate_questions(
        candidates
    )

    df = (
        base.merge(
            q_scores,
            on=[
                "question_id",
                "transcript_id",
            ],
            validate="one_to_one",
        )
    )

    gold = (
        df[
            "gold_bool"
        ]
        .astype(bool)
        .to_numpy()
    )

    evidence = (
        df[
            "evidence_tiou"
        ]
        .fillna(0.0)
        .to_numpy(
            dtype=float
        )
    )

    # =================================================================
    # M055 reference.
    # =================================================================

    p55 = (
        df[
            "m055_predicted_yes"
        ]
        .astype(bool)
        .to_numpy()
    )

    m055_metrics = (
        competition_score(
            gold,
            p55,
            evidence,
        )
    )

    # =================================================================
    # Direct M070 threshold, nested at meta level.
    # =================================================================

    splitter = GroupKFold(
        n_splits=N_SPLITS
    )

    direct_pred = np.zeros(
        len(df),
        dtype=bool,
    )

    direct_threshold = np.zeros(
        len(df),
        dtype=float,
    )

    print()
    print("=" * 78)
    print(
        "M070 DIRECT CLASSIFIER"
    )
    print("=" * 78)

    for fold, (
        train_idx,
        test_idx,
    ) in enumerate(
        splitter.split(
            df,
            gold,
            df[
                "transcript_id"
            ],
        ),
        start=1,
    ):

        threshold, train_comp = (
            best_threshold(
                df.iloc[
                    train_idx
                ][
                    "m070_max"
                ]
                .to_numpy(),
                gold[
                    train_idx
                ],
                evidence[
                    train_idx
                ],
            )
        )

        pred = (
            df.iloc[
                test_idx
            ][
                "m070_max"
            ]
            .to_numpy()
            >= threshold
        )

        direct_pred[
            test_idx
        ] = pred

        direct_threshold[
            test_idx
        ] = threshold

        metrics = (
            competition_score(
                gold[
                    test_idx
                ],
                pred,
                evidence[
                    test_idx
                ],
            )
        )

        print(
            f"Fold {fold}: "
            f"threshold={threshold:.4f}, "
            f"train={train_comp:.4f}, "
            f"acc={metrics[0]:.4f}, "
            f"tIoU={metrics[1]:.4f}, "
            f"comp={metrics[2]:.4f}"
        )

    direct_metrics = (
        competition_score(
            gold,
            direct_pred,
            evidence,
        )
    )

    # =================================================================
    # M055 + M070 fusion.
    #
    # Both base signals are already OOF, so we can cheaply learn the
    # interpolation weight and decision threshold conversation-disjoint.
    # =================================================================

    splitter = GroupKFold(
        n_splits=N_SPLITS
    )

    fusion_pred = np.zeros(
        len(df),
        dtype=bool,
    )

    fusion_score = np.zeros(
        len(df),
        dtype=float,
    )

    fusion_thresholds = []
    fusion_weights = []

    weight_grid = np.asarray(
        [
            0.0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0,
        ]
    )

    print()
    print("=" * 78)
    print(
        "M070 + M055 FUSION"
    )
    print("=" * 78)

    for fold, (
        train_idx,
        test_idx,
    ) in enumerate(
        splitter.split(
            df,
            gold,
            df[
                "transcript_id"
            ],
        ),
        start=1,
    ):

        best_weight = None
        best_t = None
        best_comp = -1.0

        for weight in weight_grid:

            train_score = (
                weight
                * df.iloc[
                    train_idx
                ][
                    "m055_probability"
                ]
                .to_numpy()
                +
                (
                    1.0
                    - weight
                )
                * df.iloc[
                    train_idx
                ][
                    "m070_max"
                ]
                .to_numpy()
            )

            threshold, comp = (
                best_threshold(
                    train_score,
                    gold[
                        train_idx
                    ],
                    evidence[
                        train_idx
                    ],
                )
            )

            if comp > best_comp:
                best_comp = comp
                best_weight = float(
                    weight
                )
                best_t = float(
                    threshold
                )

        test_score = (
            best_weight
            * df.iloc[
                test_idx
            ][
                "m055_probability"
            ]
            .to_numpy()
            +
            (
                1.0
                - best_weight
            )
            * df.iloc[
                test_idx
            ][
                "m070_max"
            ]
            .to_numpy()
        )

        pred = (
            test_score
            >= best_t
        )

        fusion_score[
            test_idx
        ] = test_score

        fusion_pred[
            test_idx
        ] = pred

        fusion_weights.append(
            best_weight
        )

        fusion_thresholds.append(
            best_t
        )

        metrics = (
            competition_score(
                gold[
                    test_idx
                ],
                pred,
                evidence[
                    test_idx
                ],
            )
        )

        print(
            f"Fold {fold}: "
            f"weight_m055="
            f"{best_weight:.2f}, "
            f"threshold="
            f"{best_t:.4f}, "
            f"train="
            f"{best_comp:.4f}, "
            f"acc="
            f"{metrics[0]:.4f}, "
            f"tIoU="
            f"{metrics[1]:.4f}, "
            f"comp="
            f"{metrics[2]:.4f}"
        )

    fusion_metrics = (
        competition_score(
            gold,
            fusion_pred,
            evidence,
        )
    )

    # =================================================================
    # Save.
    # =================================================================

    df[
        "m070_direct_prediction"
    ] = direct_pred

    df[
        "m070_direct_threshold"
    ] = direct_threshold

    df[
        "m070_fusion_score"
    ] = fusion_score

    df[
        "m070_fusion_prediction"
    ] = fusion_pred

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT,
        index=False,
    )

    # =================================================================
    # Final result.
    # =================================================================

    print()
    print("=" * 78)
    print(
        "M070 FINAL RESULT"
    )
    print("=" * 78)

    print()
    print(
        f"M055 reference:"
        f" acc={m055_metrics[0]:.4f},"
        f" tIoU={m055_metrics[1]:.4f},"
        f" comp={m055_metrics[2]:.4f}"
    )

    print(
        f"M070 direct:"
        f"    acc={direct_metrics[0]:.4f},"
        f" tIoU={direct_metrics[1]:.4f},"
        f" comp={direct_metrics[2]:.4f}"
    )

    print(
        f"M070 fusion:"
        f"    acc={fusion_metrics[0]:.4f},"
        f" tIoU={fusion_metrics[1]:.4f},"
        f" comp={fusion_metrics[2]:.4f}"
    )

    print()
    print(
        "Fusion weights: "
        + ", ".join(
            f"{x:.2f}"
            for x
            in fusion_weights
        )
    )

    print(
        "Fusion thresholds: "
        + ", ".join(
            f"{x:.4f}"
            for x
            in fusion_thresholds
        )
    )

    print()
    print(
        f"Gain vs M055: "
        f"{fusion_metrics[2] - m055_metrics[2]:+.4f}"
    )

    print(
        f"Wall time: "
        f"{time.perf_counter()-wall_start:.1f} s"
    )

    print()
    print(
        f"Saved: {OUTPUT}"
    )


if __name__ == "__main__":
    main()