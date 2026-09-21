from __future__ import annotations

import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import GroupKFold

import medical.src.classification.evaluate_m070_task_specific_nli as m070


RAW_CANDIDATES = Path(
    r"medical\artifacts\retrieval"
    r"\m047_all_questions_candidates_oof.csv"
)

M055 = Path(
    r"medical\artifacts\classification"
    r"\m055_base_nli_runtime_signal_oof.csv"
)

OUTPUT = Path(
    r"medical\artifacts\classification"
    r"\m072_self_hard_negative_mining_oof.csv"
)

N_SPLITS = 5
SEED = 42

# Runtime still consumes M047 top-12.
RUNTIME_TOP_K = 12

# Broader pool used only for hard-negative mining.
MINING_TOP_K = 40

# How many additional self-mined negatives to inject.
MINED_NEG_PER_POSITIVE = 6
MINED_NEG_PER_NEGATIVE = 8

# Wrong/near-wrong spans for positive questions.
MAX_NEG_TIOU = 0.10

# Rank highly scored mistakes much more strongly.
MINED_WEIGHT_TOP = 2.25
MINED_WEIGHT_OTHER = 1.75

# Stage 1 only needs to become a competent miner.
STAGE1_EPOCHS = 2

# Stage 2 is the actual final fold model.
STAGE2_EPOCHS = 3


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


def prepare_pool(
    raw: pd.DataFrame,
    top_k: int,
):
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

    df["rank"] = (
        df.groupby("question_id")
        .cumcount()
        + 1
    )

    return (
        df[
            df["rank"] <= top_k
        ]
        .copy()
        .reset_index(drop=True)
    )


def build_self_mined_pairs(
    scored_pool: pd.DataFrame,
    allowed_ids: set[str],
):
    df = scored_pool[
        scored_pool[
            "question_id"
        ].isin(allowed_ids)
    ].copy()

    rows = []

    for question_id, group in df.groupby(
        "question_id",
        sort=False,
    ):
        first = group.iloc[0]

        qtype = str(
            first["question_type"]
        )

        question = str(
            first["question"]
        )

        transcript_id = str(
            first["transcript_id"]
        )

        # ------------------------------------------------------------
        # Positive question:
        # explicitly mine spans that strongly fool M070 but have
        # essentially zero localization overlap.
        # ------------------------------------------------------------

        if qtype == "positive":
            bad = (
                group[
                    group["target_tiou"]
                    <= MAX_NEG_TIOU
                ]
                .sort_values(
                    "stage1_score",
                    ascending=False,
                )
                .head(
                    MINED_NEG_PER_POSITIVE
                )
            )

        # ------------------------------------------------------------
        # Negative/off-topic question:
        # every candidate is non-supporting, so take the spans the
        # stage-1 model most confidently thinks entail the question.
        # ------------------------------------------------------------

        else:
            bad = (
                group.sort_values(
                    "stage1_score",
                    ascending=False,
                )
                .head(
                    MINED_NEG_PER_NEGATIVE
                )
            )

        for local_rank, (
            _,
            row,
        ) in enumerate(
            bad.iterrows(),
            start=1,
        ):
            weight = (
                MINED_WEIGHT_TOP
                if local_rank <= 2
                else MINED_WEIGHT_OTHER
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
                        (
                            "self_mined_"
                            + qtype
                        ),

                    "stage1_score":
                        float(
                            row[
                                "stage1_score"
                            ]
                        ),

                    "candidate_rank":
                        int(
                            row["rank"]
                        ),

                    "target_tiou":
                        float(
                            row[
                                "target_tiou"
                            ]
                        ),
                }
            )

    return pd.DataFrame(rows)


def combine_pairs(
    original: pd.DataFrame,
    mined: pd.DataFrame,
):
    base = original.copy()

    # Make schemas compatible.
    for column in [
        "stage1_score",
        "candidate_rank",
        "target_tiou",
    ]:
        if column not in base:
            base[column] = np.nan

    combined = pd.concat(
        [
            base,
            mined,
        ],
        ignore_index=True,
        sort=False,
    )

    # If the same Q/span is already present, keep the stronger weight.
    combined = (
        combined.sort_values(
            "weight",
            ascending=False,
        )
        .drop_duplicates(
            subset=[
                "question_id",
                "candidate_text",
                "target",
            ],
            keep="first",
        )
        .reset_index(drop=True)
    )

    return combined


def threshold_candidates(
    values: np.ndarray,
):
    unique = np.unique(values)

    if len(unique) == 1:
        return np.asarray(
            [
                unique[0] - 1e-9,
                unique[0] + 1e-9,
            ]
        )

    mids = (
        unique[:-1]
        + unique[1:]
    ) / 2.0

    return np.concatenate(
        [
            [unique[0] - 1e-9],
            mids,
            [unique[-1] + 1e-9],
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
        threshold_candidates(scores)
    ):
        prediction = (
            scores >= threshold
        )

        _, _, comp = competition_score(
            gold,
            prediction,
            evidence,
        )

        if comp > best_comp:
            best_comp = comp
            best_t = float(
                threshold
            )

    return best_t, best_comp


def main():
    m070.seed_everything(SEED)

    wall_start = time.perf_counter()

    raw = pd.read_csv(
        RAW_CANDIDATES
    )

    runtime_candidates = (
        prepare_pool(
            raw,
            RUNTIME_TOP_K,
        )
    )

    mining_candidates = (
        prepare_pool(
            raw,
            MINING_TOP_K,
        )
    )

    # One row/question source.
    base = (
        raw[
            [
                "question_id",
                "transcript_id",
                "question",
                "question_type",
            ]
        ]
        .drop_duplicates(
            "question_id"
        )
        .reset_index(drop=True)
    )

    m055 = pd.read_csv(
        M055
    )[
        [
            "question_id",
            "gold_bool",
            "m055_probability",
            "m055_predicted_yes",
            "m055_threshold",
            "evidence_tiou",
        ]
    ].copy()

    base = base.merge(
        m055,
        on="question_id",
        validate="one_to_one",
    )

    base["gold_bool"] = [
        parse_bool(x)
        for x
        in base["gold_bool"]
    ]

    if len(base) != 390:
        raise RuntimeError(
            f"Expected 390 questions, got {len(base)}"
        )

    groups = (
        base[
            "transcript_id"
        ]
        .astype(str)
        .to_numpy()
    )

    gold = (
        base["gold_bool"]
        .astype(bool)
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
        "M072 Self Hard-Negative Mining"
    )
    print("=" * 78)

    print(
        f"Model:             {m070.MODEL_ID}"
    )

    print(
        f"Device:            {device}"
    )

    print(
        f"Runtime top-K:     {RUNTIME_TOP_K}"
    )

    print(
        f"Mining top-K:      {MINING_TOP_K}"
    )

    print(
        f"Runtime candidates:{len(runtime_candidates)}"
    )

    print(
        f"Mining candidates: {len(mining_candidates)}"
    )

    splitter = GroupKFold(
        n_splits=N_SPLITS
    )

    # OOF score for candidates seen by runtime.
    runtime_candidates[
        "m072_candidate_score"
    ] = np.nan

    mining_stats = []

    # ================================================================
    # OUTER CV
    # ================================================================

    for fold, (
        train_idx,
        test_idx,
    ) in enumerate(
        splitter.split(
            base,
            gold,
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

        # ------------------------------------------------------------
        # STAGE 1
        # Train original M070-style model ONLY on outer training data.
        # ------------------------------------------------------------

        stage1_candidates = (
            runtime_candidates[
                runtime_candidates[
                    "question_id"
                ].isin(train_ids)
            ].copy()
        )

        stage1_pairs = (
            m070.build_training_pairs(
                stage1_candidates,
                train_ids,
            )
        )

        print(
            f"Train questions:      {len(train_ids)}"
        )

        print(
            f"Test questions:       {len(test_ids)}"
        )

        print(
            f"Stage-1 pairs:        {len(stage1_pairs)}"
        )

        original_epochs = m070.EPOCHS

        m070.EPOCHS = (
            STAGE1_EPOCHS
        )

        (
            miner_tokenizer,
            miner_model,
            entail_idx,
            contradiction_idx,
            neutral_idx,
        ) = m070.train_fold(
            stage1_pairs,
            device,
            fold=(
                fold * 10 + 1
            ),
        )

        # ------------------------------------------------------------
        # Score broad TRAINING pool with stage-1 model.
        # No held-out information participates in mining.
        # ------------------------------------------------------------

        train_pool_mask = (
            mining_candidates[
                "question_id"
            ].isin(train_ids)
        )

        train_pool = (
            mining_candidates.loc[
                train_pool_mask
            ].copy()
        )

        train_pool[
            "stage1_score"
        ] = m070.score_candidates(
            train_pool,
            miner_tokenizer,
            miner_model,
            entail_idx,
            contradiction_idx,
            neutral_idx,
            device,
        )

        mined = build_self_mined_pairs(
            train_pool,
            train_ids,
        )

        stage2_pairs = combine_pairs(
            stage1_pairs,
            mined,
        )

        print(
            f"Self-mined pairs:     {len(mined)}"
        )

        print(
            f"Stage-2 total pairs:  {len(stage2_pairs)}"
        )

        print()
        print(
            mined[
                "source"
            ]
            .value_counts()
            .to_string()
        )

        print()
        print(
            "Mined score mean: "
            f"{mined['stage1_score'].mean():.4f}"
        )

        print(
            "Mined score p90:  "
            f"{mined['stage1_score'].quantile(.90):.4f}"
        )

        mining_stats.append(
            {
                "fold":
                    fold,

                "original_pairs":
                    len(stage1_pairs),

                "mined_pairs":
                    len(mined),

                "total_pairs":
                    len(stage2_pairs),

                "mined_mean_score":
                    float(
                        mined[
                            "stage1_score"
                        ].mean()
                    ),
            }
        )

        del miner_model
        del miner_tokenizer

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ------------------------------------------------------------
        # STAGE 2
        # Fresh base checkpoint; augmented training set.
        # ------------------------------------------------------------

        m070.EPOCHS = (
            STAGE2_EPOCHS
        )

        (
            tokenizer,
            model,
            entail_idx,
            contradiction_idx,
            neutral_idx,
        ) = m070.train_fold(
            stage2_pairs,
            device,
            fold=(
                fold * 10 + 2
            ),
        )

        # Restore global for safety.
        m070.EPOCHS = (
            original_epochs
        )

        # ------------------------------------------------------------
        # Evaluate ONLY on held-out outer-fold questions.
        # Runtime distribution remains top-12.
        # ------------------------------------------------------------

        test_mask = (
            runtime_candidates[
                "question_id"
            ].isin(test_ids)
        )

        test_candidates = (
            runtime_candidates.loc[
                test_mask
            ].copy()
        )

        scores = m070.score_candidates(
            test_candidates,
            tokenizer,
            model,
            entail_idx,
            contradiction_idx,
            neutral_idx,
            device,
        )

        runtime_candidates.loc[
            test_mask,
            "m072_candidate_score",
        ] = scores

        del model
        del tokenizer

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if (
        runtime_candidates[
            "m072_candidate_score"
        ]
        .isna()
        .any()
    ):
        raise RuntimeError(
            "Missing M072 OOF candidate scores."
        )

    # ================================================================
    # Question-level M072 signal.
    # ================================================================

    scored = (
        runtime_candidates.rename(
            columns={
                "m072_candidate_score":
                    "m070_candidate_score"
            }
        )
    )

    q_scores = (
        m070.aggregate_questions(
            scored
        )
        .rename(
            columns={
                "m070_max":
                    "m072_max",

                "m070_second":
                    "m072_second",

                "m070_gap":
                    "m072_gap",

                "m070_top3_mean":
                    "m072_top3_mean",

                "m070_mean":
                    "m072_mean",
            }
        )
    )

    df = base.merge(
        q_scores,
        on=[
            "question_id",
            "transcript_id",
        ],
        validate="one_to_one",
    )

    gold = (
        df["gold_bool"]
        .astype(bool)
        .to_numpy()
    )

    evidence = (
        df["evidence_tiou"]
        .fillna(0.0)
        .to_numpy(dtype=float)
    )

    p55 = (
        df[
            "m055_predicted_yes"
        ]
        .astype(bool)
        .to_numpy()
    )

    m055_metrics = competition_score(
        gold,
        p55,
        evidence,
    )

    # ================================================================
    # DIRECT M072
    # ================================================================

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
    print("M072 DIRECT")
    print("=" * 78)

    for fold, (
        train_idx,
        test_idx,
    ) in enumerate(
        splitter.split(
            df,
            gold,
            df["transcript_id"],
        ),
        start=1,
    ):
        threshold, train_comp = (
            best_threshold(
                df.iloc[
                    train_idx
                ][
                    "m072_max"
                ].to_numpy(),
                gold[train_idx],
                evidence[train_idx],
            )
        )

        prediction = (
            df.iloc[
                test_idx
            ][
                "m072_max"
            ].to_numpy()
            >= threshold
        )

        direct_pred[
            test_idx
        ] = prediction

        direct_threshold[
            test_idx
        ] = threshold

        metrics = competition_score(
            gold[test_idx],
            prediction,
            evidence[test_idx],
        )

        print(
            f"Fold {fold}: "
            f"threshold={threshold:.4f}, "
            f"train={train_comp:.4f}, "
            f"acc={metrics[0]:.4f}, "
            f"tIoU={metrics[1]:.4f}, "
            f"comp={metrics[2]:.4f}"
        )

    direct_metrics = competition_score(
        gold,
        direct_pred,
        evidence,
    )

    # ================================================================
    # M055 + M072 FUSION
    # ================================================================

    fusion_pred = np.zeros(
        len(df),
        dtype=bool,
    )

    fusion_score = np.zeros(
        len(df),
        dtype=float,
    )

    fusion_weights = []
    fusion_thresholds = []

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

    splitter = GroupKFold(
        n_splits=N_SPLITS
    )

    print()
    print("=" * 78)
    print("M072 + M055 FUSION")
    print("=" * 78)

    for fold, (
        train_idx,
        test_idx,
    ) in enumerate(
        splitter.split(
            df,
            gold,
            df["transcript_id"],
        ),
        start=1,
    ):
        best_weight = 1.0
        best_t = 0.5
        best_comp = -1.0

        for weight in weight_grid:
            train_score = (
                weight
                * df.iloc[
                    train_idx
                ][
                    "m055_probability"
                ].to_numpy()
                +
                (
                    1.0
                    - weight
                )
                * df.iloc[
                    train_idx
                ][
                    "m072_max"
                ].to_numpy()
            )

            threshold, comp = (
                best_threshold(
                    train_score,
                    gold[train_idx],
                    evidence[train_idx],
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
            ].to_numpy()
            +
            (
                1.0
                - best_weight
            )
            * df.iloc[
                test_idx
            ][
                "m072_max"
            ].to_numpy()
        )

        prediction = (
            test_score
            >= best_t
        )

        fusion_score[
            test_idx
        ] = test_score

        fusion_pred[
            test_idx
        ] = prediction

        fusion_weights.append(
            best_weight
        )

        fusion_thresholds.append(
            best_t
        )

        metrics = competition_score(
            gold[test_idx],
            prediction,
            evidence[test_idx],
        )

        print(
            f"Fold {fold}: "
            f"weight_m055={best_weight:.2f}, "
            f"threshold={best_t:.4f}, "
            f"train={best_comp:.4f}, "
            f"acc={metrics[0]:.4f}, "
            f"tIoU={metrics[1]:.4f}, "
            f"comp={metrics[2]:.4f}"
        )

    fusion_metrics = competition_score(
        gold,
        fusion_pred,
        evidence,
    )

    # ================================================================
    # SAVE
    # ================================================================

    df[
        "m072_direct_prediction"
    ] = direct_pred

    df[
        "m072_direct_threshold"
    ] = direct_threshold

    df[
        "m072_fusion_score"
    ] = fusion_score

    df[
        "m072_fusion_prediction"
    ] = fusion_pred

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT,
        index=False,
    )

    print()
    print("=" * 78)
    print("M072 FINAL RESULT")
    print("=" * 78)

    print(
        f"M055 reference: "
        f"acc={m055_metrics[0]:.4f}, "
        f"tIoU={m055_metrics[1]:.4f}, "
        f"comp={m055_metrics[2]:.4f}"
    )

    print(
        f"M072 direct:    "
        f"acc={direct_metrics[0]:.4f}, "
        f"tIoU={direct_metrics[1]:.4f}, "
        f"comp={direct_metrics[2]:.4f}"
    )

    print(
        f"M072 fusion:    "
        f"acc={fusion_metrics[0]:.4f}, "
        f"tIoU={fusion_metrics[1]:.4f}, "
        f"comp={fusion_metrics[2]:.4f}"
    )

    print()
    print(
        "Fusion weights: "
        + ", ".join(
            f"{x:.2f}"
            for x in fusion_weights
        )
    )

    print(
        "Fusion thresholds: "
        + ", ".join(
            f"{x:.4f}"
            for x in fusion_thresholds
        )
    )

    print()
    print(
        f"Gain vs M055: "
        f"{fusion_metrics[2] - m055_metrics[2]:+.4f}"
    )

    print(
        f"Gain vs M070 fusion (0.6660): "
        f"{fusion_metrics[2] - 0.6660:+.4f}"
    )

    print(
        f"Wall time: "
        f"{time.perf_counter()-wall_start:.1f}s"
    )

    print()
    print("MINING SUMMARY")
    print("-" * 78)

    print(
        pd.DataFrame(
            mining_stats
        ).to_string(
            index=False
        )
    )

    print()
    print(
        f"Saved: {OUTPUT}"
    )


if __name__ == "__main__":
    main()