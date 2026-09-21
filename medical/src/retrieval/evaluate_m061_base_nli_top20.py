from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import GroupKFold
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


INPUT = Path(
    r"medical\artifacts\retrieval"
    r"\m047_all_questions_candidates_oof.csv"
)

OUTPUT = Path(
    r"medical\artifacts\retrieval"
    r"\m061_base_nli_top20_oof.csv"
)

ALL_SCORES_OUTPUT = Path(
    r"medical\artifacts\retrieval"
    r"\m061_base_nli_top20_scores.csv"
)

MODEL_ID = (
    "cross-encoder/nli-deberta-v3-base"
)

TOP_K = 20
N_SPLITS = 5
BATCH_SIZE = 32
MAX_LENGTH = 256


# ---------------------------------------------------------------------------
# Reuse exactly the same question -> declarative conversion as the
# segmentwise NLI experiments.
# ---------------------------------------------------------------------------

SRC_ROOT = Path(
    r"medical\src\classification"
).resolve()

sys.path.insert(
    0,
    str(SRC_ROOT),
)

from evaluate_nli_segmentwise_base import question_to_claim  # noqa: E402


def get_label_indices(
    model,
):
    mapping = {
        int(k): str(v).lower()
        for k, v
        in model.config.id2label.items()
    }

    result = {}

    for index, label in (
        mapping.items()
    ):
        for target in (
            "contradiction",
            "entailment",
            "neutral",
        ):
            if target in label:
                result[target] = index

    if len(result) != 3:
        return {
            "contradiction": 0,
            "entailment": 1,
            "neutral": 2,
        }

    return result


def add_within_question_normalized(
    df: pd.DataFrame,
    column: str,
    output_column: str,
):
    df = df.copy()

    def normalize(group):
        values = (
            group[column]
            .astype(float)
            .to_numpy()
        )

        minimum = float(
            np.min(values)
        )

        maximum = float(
            np.max(values)
        )

        if (
            maximum
            - minimum
            < 1e-12
        ):
            return pd.Series(
                np.zeros(
                    len(group),
                    dtype=float,
                ),
                index=group.index,
            )

        return pd.Series(
            (
                values
                - minimum
            )
            / (
                maximum
                - minimum
            ),
            index=group.index,
        )

    df[output_column] = (
        df.groupby(
            "question_id",
            group_keys=False,
        )
        .apply(
            normalize,
            include_groups=False,
        )
    )

    return df


def selected_mean_tiou(
    df: pd.DataFrame,
    score_column: str,
    question_ids=None,
):
    if question_ids is not None:
        subset = df[
            df[
                "question_id"
            ].isin(
                question_ids
            )
        ]
    else:
        subset = df

    indices = (
        subset.groupby(
            "question_id"
        )[
            score_column
        ]
        .idxmax()
    )

    return float(
        subset.loc[
            indices,
            "target_tiou",
        ].mean()
    )


def summarize(
    name: str,
    df: pd.DataFrame,
    score_column: str,
):
    indices = (
        df.groupby(
            "question_id"
        )[
            score_column
        ]
        .idxmax()
    )

    selected = (
        df.loc[
            indices
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    scores = (
        selected[
            "target_tiou"
        ]
        .astype(float)
    )

    print()
    print(name)
    print("-" * 78)

    print(
        f"Mean tIoU:       "
        f"{scores.mean():.4f}"
    )

    print(
        f"Median tIoU:     "
        f"{scores.median():.4f}"
    )

    print(
        f"Any overlap:     "
        f"{(scores > 0).mean():.4f}"
    )

    print(
        f"tIoU >= 0.25:    "
        f"{(scores >= 0.25).mean():.4f}"
    )

    print(
        f"tIoU >= 0.50:    "
        f"{(scores >= 0.50).mean():.4f}"
    )

    print(
        f"tIoU >= 0.75:    "
        f"{(scores >= 0.75).mean():.4f}"
    )

    return (
        float(
            scores.mean()
        ),
        selected,
    )


def main():
    df = pd.read_csv(
        INPUT
    )

    positive = (
        df[
            df[
                "question_type"
            ]
            == "positive"
        ]
        .copy()
    )

    if (
        positive[
            "question_id"
        ].nunique()
        != 195
    ):
        raise RuntimeError(
            "Expected 195 "
            "positive questions."
        )

    # Keep top-20 according to the already
    # conversation-disjoint first-stage score.
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
        .head(
            TOP_K
        )
        .copy()
        .reset_index(
            drop=True
        )
    )

    print("=" * 78)
    print(
        "Nordic AI Cup 2026 — "
        "M061 DeBERTa-v3-base Top-20 Evidence Reranker"
    )
    print("=" * 78)

    print(
        f"Questions:  "
        f"{top.question_id.nunique()}"
    )

    print(
        f"Candidates: "
        f"{len(top)}"
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device:     {device}"
    )

    print()
    print(
        "Loading DeBERTa-v3-base NLI..."
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
        .eval()
    )

    labels = (
        get_label_indices(
            model
        )
    )

    premises = (
        top[
            "candidate_text"
        ]
        .astype(str)
        .tolist()
    )

    hypotheses = [
        question_to_claim(
            question
        )
        for question
        in top[
            "question"
        ]
        .astype(str)
        .tolist()
    ]

    entailment = []
    contradiction = []
    neutral = []

    for start in range(
        0,
        len(top),
        BATCH_SIZE,
    ):
        end = min(
            start
            + BATCH_SIZE,
            len(top),
        )

        encoded = tokenizer(
            premises[
                start:end
            ],
            hypotheses[
                start:end
            ],
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        encoded = {
            key: value.to(
                device
            )
            for key, value
            in encoded.items()
        }

        with torch.inference_mode():
            logits = (
                model(
                    **encoded
                ).logits
            )

            probs = (
                torch.softmax(
                    logits,
                    dim=-1,
                )
                .detach()
                .cpu()
                .numpy()
            )

        entailment.extend(
            probs[
                :,
                labels[
                    "entailment"
                ],
            ].tolist()
        )

        contradiction.extend(
            probs[
                :,
                labels[
                    "contradiction"
                ],
            ].tolist()
        )

        neutral.extend(
            probs[
                :,
                labels[
                    "neutral"
                ],
            ].tolist()
        )

        if (
            end % 500 < BATCH_SIZE
            or end == len(top)
        ):
            print(
                f"[{end:04d}/"
                f"{len(top)}] "
                "NLI scored"
            )

    top[
        "base_entailment"
    ] = np.asarray(
        entailment,
        dtype=float,
    )

    top[
        "base_contradiction"
    ] = np.asarray(
        contradiction,
        dtype=float,
    )

    top[
        "base_neutral"
    ] = np.asarray(
        neutral,
        dtype=float,
    )

    top[
        "base_margin"
    ] = (
        top[
            "base_entailment"
        ]
        - top[
            "base_contradiction"
        ]
    )

    top[
        "base_ratio"
    ] = (
        top[
            "base_entailment"
        ]
        / (
            top[
                "base_entailment"
            ]
            + top[
                "base_contradiction"
            ]
            + 1e-12
        )
    )

    # Normalize scores within each question.
    top = add_within_question_normalized(
        top,
        "oof_score",
        "ranker_norm",
    )

    top = add_within_question_normalized(
        top,
        "base_entailment",
        "entailment_norm",
    )

    top = add_within_question_normalized(
        top,
        "base_margin",
        "margin_norm",
    )

    top = add_within_question_normalized(
        top,
        "base_ratio",
        "ratio_norm",
    )

    ALL_SCORES_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    top.to_csv(
        ALL_SCORES_OUTPUT,
        index=False,
    )

    # ------------------------------------------------------------
    # Standalone methods.
    # ------------------------------------------------------------

    summarize(
        "M047 TOP-1 WITHIN TOP-20",
        top,
        "oof_score",
    )

    summarize(
        "BASE NLI — ENTAILMENT",
        top,
        "base_entailment",
    )

    summarize(
        "BASE NLI — MARGIN",
        top,
        "base_margin",
    )

    summarize(
        "BASE NLI — RATIO",
        top,
        "base_ratio",
    )

    top[
        "oracle_score"
    ] = top[
        "target_tiou"
    ]

    summarize(
        "TOP-20 ORACLE",
        top,
        "oracle_score",
    )

    # ------------------------------------------------------------
    # Conversation-disjoint fusion.
    #
    # Tune only the interpolation weight on the training
    # conversations of each fold.
    # ------------------------------------------------------------

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

    weights = np.linspace(
        0.0,
        1.0,
        21,
    )

    methods = {
        "entailment":
            "entailment_norm",

        "margin":
            "margin_norm",

        "ratio":
            "ratio_norm",
    }

    best_global_method = None
    best_global_mean = -1.0
    all_results = []

    for method_name, (
        nli_column
    ) in methods.items():

        output_column = (
            f"m061_{method_name}"
        )

        top[
            output_column
        ] = np.nan

        fold_weights = []

        print()
        print("=" * 78)
        print(
            f"CV FUSION — "
            f"{method_name.upper()}"
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

            best_weight = None
            best_train = -1.0

            for weight in weights:
                temp_column = (
                    "_fusion_temp"
                )

                top[
                    temp_column
                ] = (
                    weight
                    * top[
                        "ranker_norm"
                    ]
                    + (
                        1.0
                        - weight
                    )
                    * top[
                        nli_column
                    ]
                )

                train_mean = (
                    selected_mean_tiou(
                        top,
                        temp_column,
                        train_ids,
                    )
                )

                if (
                    train_mean
                    > best_train
                ):
                    best_train = (
                        train_mean
                    )

                    best_weight = float(
                        weight
                    )

            test_mask = (
                top[
                    "question_id"
                ]
                .isin(
                    test_ids
                )
            )

            top.loc[
                test_mask,
                output_column,
            ] = (
                best_weight
                * top.loc[
                    test_mask,
                    "ranker_norm",
                ]
                + (
                    1.0
                    - best_weight
                )
                * top.loc[
                    test_mask,
                    nli_column,
                ]
            )

            fold_weights.append(
                best_weight
            )

            test_mean = (
                selected_mean_tiou(
                    top,
                    output_column,
                    test_ids,
                )
            )

            baseline = (
                selected_mean_tiou(
                    top,
                    "oof_score",
                    test_ids,
                )
            )

            print(
                f"Fold {fold}: "
                f"weight_ranker="
                f"{best_weight:.2f}, "
                f"train="
                f"{best_train:.4f}, "
                f"M047="
                f"{baseline:.4f}, "
                f"M061="
                f"{test_mean:.4f}, "
                f"delta="
                f"{test_mean-baseline:+.4f}"
            )

        if (
            top[
                output_column
            ]
            .isna()
            .any()
        ):
            raise RuntimeError(
                f"Missing OOF scores "
                f"for {method_name}"
            )

        mean_tiou, selected = (
            summarize(
                f"M061 OOF FUSION — "
                f"{method_name.upper()}",
                top,
                output_column,
            )
        )

        all_results.append(
            (
                method_name,
                mean_tiou,
                fold_weights,
                output_column,
            )
        )

        if (
            mean_tiou
            > best_global_mean
        ):
            best_global_mean = (
                mean_tiou
            )

            best_global_method = (
                method_name
            )

    best = [
        row
        for row
        in all_results
        if row[0]
        == best_global_method
    ][0]

    (
        method_name,
        mean_tiou,
        fold_weights,
        output_column,
    ) = best

    indices = (
        top.groupby(
            "question_id"
        )[
            output_column
        ]
        .idxmax()
    )

    selected = (
        top.loc[
            indices
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    selected.to_csv(
        OUTPUT,
        index=False,
    )

    print()
    print("=" * 78)
    print("M061 BEST RESULT")
    print("=" * 78)

    print(
        f"Method:           "
        f"{method_name}"
    )

    print(
        f"Mean tIoU:        "
        f"{mean_tiou:.4f}"
    )

    print(
        "Fold weights:     "
        + ", ".join(
            f"{x:.2f}"
            for x
            in fold_weights
        )
    )

    print(
        "M047 reference:   "
        "0.5158"
    )

    print(
        "Top-20 oracle:    "
        "0.7966"
    )

    print(
        f"Gain vs M047:     "
        f"{mean_tiou - 0.5158:+.4f}"
    )

    print()
    print(
        f"Selected rows: "
        f"{OUTPUT}"
    )

    print(
        f"All scores:    "
        f"{ALL_SCORES_OUTPUT}"
    )


if __name__ == "__main__":
    main()