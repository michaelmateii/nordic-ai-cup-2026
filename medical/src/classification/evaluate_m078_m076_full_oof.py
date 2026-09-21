from __future__ import annotations

import gc
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

from medical.src.classification.evaluate_m076_multipassage_pilot import (
    QUESTIONS,
    CANDIDATES,
    M070,
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
    r"medical\artifacts\classification"
    r"\m078_m076_full_oof.csv"
)

SEED = 42
N_SPLITS = 5

# Hidden-validated M074 champion.
M055_WEIGHT = 0.68
M070_WEIGHT = 0.32
M074_THRESHOLD = 0.3226589362393797

# Current M077 deployment guess.
CURRENT_M076_WEIGHT = 0.35
CURRENT_THRESHOLD = 0.3847283085555968


def competition_score(
    gold: np.ndarray,
    pred: np.ndarray,
    evidence: np.ndarray,
):
    accuracy = float(
        np.mean(gold == pred)
    )

    positive = gold

    tiou = float(
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
        + 0.6 * tiou
    )

    return accuracy, tiou, composite


def train_one_fold(
    train: pd.DataFrame,
    test: pd.DataFrame,
    fold: int,
    device: str,
):
    print()
    print("=" * 78)
    print(f"TRAINING FOLD {fold}/{N_SPLITS}")
    print("=" * 78)

    print("Train:", len(train))
    print("Test: ", len(test))

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID
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

            output = model(
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
                labels=(
                    batch[
                        "labels"
                    ].to(device)
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

    # ------------------------------------------------------------
    # Held-out inference
    # ------------------------------------------------------------

    model.eval()

    test_loader = DataLoader(
        DS(
            test,
            tokenizer,
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    probabilities = []

    with torch.inference_mode():
        for batch in test_loader:
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

            probability = (
                torch.softmax(
                    logits,
                    dim=-1,
                )[:, 1]
            )

            probabilities.extend(
                probability
                .detach()
                .cpu()
                .float()
                .numpy()
                .tolist()
            )

    del model
    del tokenizer

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return np.asarray(
        probabilities,
        dtype=float,
    )


def threshold_candidates(
    scores: np.ndarray,
):
    values = np.unique(scores)

    if len(values) == 1:
        return np.asarray(
            [
                values[0] - 1e-9,
                values[0] + 1e-9,
            ]
        )

    return np.concatenate(
        [
            [values[0] - 1e-9],
            (
                values[:-1]
                + values[1:]
            )
            / 2.0,
            [values[-1] + 1e-9],
        ]
    )


def best_fusion_on_training(
    m074: np.ndarray,
    m076: np.ndarray,
    gold: np.ndarray,
    evidence: np.ndarray,
):
    best = None

    # Explore broadly, but M076 probably belongs around 0.2–0.5.
    for weight in np.arange(
        0.0,
        0.701,
        0.01,
    ):
        combined = (
            (1.0 - weight)
            * m074
            + weight
            * m076
        )

        for threshold in (
            threshold_candidates(
                combined
            )
        ):
            pred = (
                combined
                >= threshold
            )

            metrics = competition_score(
                gold,
                pred,
                evidence,
            )

            candidate = (
                metrics[2],
                float(weight),
                float(threshold),
                metrics[0],
                metrics[1],
            )

            if (
                best is None
                or candidate[0]
                > best[0]
            ):
                best = candidate

    return best


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            SEED
        )

    started = time.perf_counter()

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

    reference = pd.read_csv(
        M070
    )[
        [
            "question_id",
            "m055_probability",
            "m070_max",
            "evidence_tiou",
        ]
    ]

    examples = examples.merge(
        reference,
        on="question_id",
        validate="one_to_one",
    )

    examples[
        "m074_probability"
    ] = (
        M055_WEIGHT
        * examples[
            "m055_probability"
        ]
        + M070_WEIGHT
        * examples[
            "m070_max"
        ]
    )

    examples[
        "m076_probability"
    ] = np.nan

    examples["fold"] = 0

    gold = (
        examples["target"]
        .astype(bool)
        .to_numpy()
    )

    groups = (
        examples[
            "transcript_id"
        ]
        .astype(str)
        .to_numpy()
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 78)
    print(
        "Nordic AI Cup 2026 — "
        "M078 Full 5-Fold M076 OOF"
    )
    print("=" * 78)

    print(
        f"Questions:     {len(examples)}"
    )

    print(
        f"Model:         {MODEL_ID}"
    )

    print(
        f"Device:        {device}"
    )

    print(
        f"Epochs/fold:   {EPOCHS}"
    )

    print(
        f"Top passages:  {TOP_PASSAGES}"
    )

    splitter = GroupKFold(
        n_splits=N_SPLITS
    )

    for fold, (
        train_idx,
        test_idx,
    ) in enumerate(
        splitter.split(
            examples,
            gold,
            groups,
        ),
        start=1,
    ):
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

        probabilities = train_one_fold(
            train,
            test,
            fold,
            device,
        )

        examples.loc[
            examples.index[
                test_idx
            ],
            "m076_probability",
        ] = probabilities

        examples.loc[
            examples.index[
                test_idx
            ],
            "fold",
        ] = fold

    if (
        examples[
            "m076_probability"
        ]
        .isna()
        .any()
    ):
        raise RuntimeError(
            "Missing OOF M076 predictions."
        )

    # ------------------------------------------------------------
    # Evaluate current hidden-winning M077 rule.
    # ------------------------------------------------------------

    gold = (
        examples["target"]
        .astype(bool)
        .to_numpy()
    )

    evidence = (
        examples[
            "evidence_tiou"
        ]
        .fillna(0.0)
        .to_numpy(dtype=float)
    )

    m074 = (
        examples[
            "m074_probability"
        ]
        .to_numpy(dtype=float)
    )

    m076 = (
        examples[
            "m076_probability"
        ]
        .to_numpy(dtype=float)
    )

    current_score = (
        (1.0 - CURRENT_M076_WEIGHT)
        * m074
        + CURRENT_M076_WEIGHT
        * m076
    )

    current_pred = (
        current_score
        >= CURRENT_THRESHOLD
    )

    current_metrics = competition_score(
        gold,
        current_pred,
        evidence,
    )

    print()
    print("=" * 78)
    print("CURRENT M077 OOF")
    print("=" * 78)

    print(
        f"Accuracy:    "
        f"{current_metrics[0]:.4f}"
    )

    print(
        f"Scored tIoU: "
        f"{current_metrics[1]:.4f}"
    )

    print(
        f"Composite:   "
        f"{current_metrics[2]:.4f}"
    )

    # ------------------------------------------------------------
    # Global optimum: diagnostic only.
    # ------------------------------------------------------------

    global_best = (
        best_fusion_on_training(
            m074,
            m076,
            gold,
            evidence,
        )
    )

    print()
    print("=" * 78)
    print("GLOBAL OOF OPTIMUM — DIAGNOSTIC")
    print("=" * 78)

    print(
        f"Composite:   "
        f"{global_best[0]:.4f}"
    )

    print(
        f"M076 weight: "
        f"{global_best[1]:.2f}"
    )

    print(
        f"Threshold:   "
        f"{global_best[2]:.6f}"
    )

    print(
        f"Accuracy:    "
        f"{global_best[3]:.4f}"
    )

    print(
        f"tIoU:        "
        f"{global_best[4]:.4f}"
    )

    # ------------------------------------------------------------
    # Nested/leave-one-fold-out calibration.
    # This is the result we should actually trust.
    # ------------------------------------------------------------

    nested_pred = np.zeros(
        len(examples),
        dtype=bool,
    )

    nested_score = np.zeros(
        len(examples),
        dtype=float,
    )

    fold_weights = []
    fold_thresholds = []

    print()
    print("=" * 78)
    print("LEAVE-ONE-FOLD-OUT FUSION CALIBRATION")
    print("=" * 78)

    for fold in range(
        1,
        N_SPLITS + 1,
    ):
        train_mask = (
            examples["fold"]
            .to_numpy()
            != fold
        )

        test_mask = (
            examples["fold"]
            .to_numpy()
            == fold
        )

        best = (
            best_fusion_on_training(
                m074[train_mask],
                m076[train_mask],
                gold[train_mask],
                evidence[train_mask],
            )
        )

        weight = best[1]
        threshold = best[2]

        test_score = (
            (1.0 - weight)
            * m074[test_mask]
            + weight
            * m076[test_mask]
        )

        test_pred = (
            test_score
            >= threshold
        )

        nested_score[
            test_mask
        ] = test_score

        nested_pred[
            test_mask
        ] = test_pred

        metrics = competition_score(
            gold[test_mask],
            test_pred,
            evidence[test_mask],
        )

        fold_weights.append(
            weight
        )

        fold_thresholds.append(
            threshold
        )

        print(
            f"Fold {fold}: "
            f"weight={weight:.2f}, "
            f"threshold={threshold:.6f}, "
            f"acc={metrics[0]:.4f}, "
            f"tIoU={metrics[1]:.4f}, "
            f"comp={metrics[2]:.4f}"
        )

    nested_metrics = competition_score(
        gold,
        nested_pred,
        evidence,
    )

    print()
    print("=" * 78)
    print("M078 FINAL")
    print("=" * 78)

    print(
        f"Current M077: "
        f"{current_metrics[2]:.4f}"
    )

    print(
        f"Nested tuned: "
        f"{nested_metrics[2]:.4f}"
    )

    print(
        f"Nested gain:  "
        f"{nested_metrics[2]-current_metrics[2]:+.4f}"
    )

    print()

    print(
        "Fold weights: "
        + ", ".join(
            f"{x:.2f}"
            for x
            in fold_weights
        )
    )

    print(
        "Fold thresholds: "
        + ", ".join(
            f"{x:.6f}"
            for x
            in fold_thresholds
        )
    )

    median_weight = float(
        np.median(
            fold_weights
        )
    )

    median_threshold = float(
        np.median(
            fold_thresholds
        )
    )

    print()
    print(
        f"Median deployment weight: "
        f"{median_weight:.3f}"
    )

    print(
        f"Median deployment threshold: "
        f"{median_threshold:.9f}"
    )

    # Evaluate median fixed rule over all OOF rows.
    median_score = (
        (1.0 - median_weight)
        * m074
        + median_weight
        * m076
    )

    median_pred = (
        median_score
        >= median_threshold
    )

    median_metrics = competition_score(
        gold,
        median_pred,
        evidence,
    )

    print(
        f"Median fixed OOF composite: "
        f"{median_metrics[2]:.4f}"
    )

    print(
        f"Median fixed gain vs M077: "
        f"{median_metrics[2]-current_metrics[2]:+.4f}"
    )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    examples[
        "m078_nested_score"
    ] = nested_score

    examples[
        "m078_nested_prediction"
    ] = nested_pred

    examples.to_csv(
        OUTPUT,
        index=False,
    )

    print()
    print(
        f"Saved: {OUTPUT}"
    )

    print(
        f"Wall time: "
        f"{time.perf_counter()-started:.1f}s"
    )


if __name__ == "__main__":
    main()