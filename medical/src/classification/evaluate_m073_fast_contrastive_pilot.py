from __future__ import annotations

import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GroupKFold

import medical.src.classification.evaluate_m070_task_specific_nli as m070


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

M070_OOF = Path(
    r"medical\artifacts\classification"
    r"\m070_task_specific_nli_oof.csv"
)

AUDIT = Path(
    r"medical\artifacts\analysis"
    r"\m073_contrastive_pair_audit.csv"
)


SEED = 42
FOLD_TO_RUN = 1
EPOCHS = 2

# Actual deployed M071 rule.
M055_WEIGHT = 0.70
M070_WEIGHT = 0.30
FUSION_THRESHOLD = 0.3053

# Added contrastive negatives per question.
POSITIVE_NEGATIVES = 2
HARD_NEGATIVES = 3
OFF_TOPIC_NEGATIVES = 1


def parse_bool(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    return (
        str(value).strip().lower()
        in {"true", "1", "yes"}
    )


def competition_score(
    gold,
    pred,
    evidence,
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


def lexical_similarity(
    question: str,
    evidence_texts: list[str],
):
    if not evidence_texts:
        return np.asarray([])

    documents = [
        question,
        *evidence_texts,
    ]

    # Word similarity.
    word = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b\w+\b",
        sublinear_tf=True,
    )

    word_matrix = (
        word.fit_transform(
            documents
        )
    )

    word_scores = cosine_similarity(
        word_matrix[0],
        word_matrix[1:],
    )[0]

    # Character similarity helps with medications,
    # measurements and morphology.
    char = TfidfVectorizer(
        lowercase=True,
        analyzer="char_wb",
        ngram_range=(3, 5),
        sublinear_tf=True,
    )

    char_matrix = (
        char.fit_transform(
            documents
        )
    )

    char_scores = cosine_similarity(
        char_matrix[0],
        char_matrix[1:],
    )[0]

    return (
        0.55 * word_scores
        + 0.45 * char_scores
    )


def build_contrastive_pairs(
    audit: pd.DataFrame,
    allowed_ids: set[str],
):
    df = audit[
        audit["question_id"]
        .isin(allowed_ids)
    ].copy()

    # Only negatives; positive own-evidence pairs already
    # exist in standard M070 training.
    df = df[
        df["target"] == 0
    ].copy()

    output = []

    for question_id, group in (
        df.groupby(
            "question_id",
            sort=False,
        )
    ):
        first = group.iloc[0]

        question = str(
            first["question"]
        )

        qtype = str(
            first["question_type"]
        )

        texts = (
            group[
                "evidence_text"
            ]
            .astype(str)
            .tolist()
        )

        similarity = lexical_similarity(
            question,
            texts,
        )

        group = group.copy()

        group[
            "contrastive_similarity"
        ] = similarity

        group = group.sort_values(
            "contrastive_similarity",
            ascending=False,
        )

        if qtype == "positive":
            keep = POSITIVE_NEGATIVES

        elif qtype == "hard_negative":
            keep = HARD_NEGATIVES

        else:
            keep = OFF_TOPIC_NEGATIVES

        group = group.head(
            keep
        )

        for local_rank, (
            _,
            row,
        ) in enumerate(
            group.iterrows(),
            start=1,
        ):
            similarity_value = float(
                row[
                    "contrastive_similarity"
                ]
            )

            # Strongly related negatives are the valuable ones.
            weight = (
                2.25
                if local_rank == 1
                else 1.75
            )

            # Additional boost when lexical similarity is genuinely high.
            if similarity_value >= 0.35:
                weight += 0.25

            output.append(
                {
                    "question_id":
                        question_id,

                    "transcript_id":
                        str(
                            row[
                                "transcript_id"
                            ]
                        ),

                    "question":
                        question,

                    "candidate_text":
                        str(
                            row[
                                "evidence_text"
                            ]
                        ),

                    "target":
                        0.0,

                    "weight":
                        weight,

                    "source":
                        (
                            "contrastive_"
                            + qtype
                        ),

                    "contrastive_similarity":
                        similarity_value,
                }
            )

    return pd.DataFrame(
        output
    )


def combine_pairs(
    original,
    contrastive,
):
    original = original.copy()

    if (
        "contrastive_similarity"
        not in original.columns
    ):
        original[
            "contrastive_similarity"
        ] = np.nan

    combined = pd.concat(
        [
            original,
            contrastive,
        ],
        ignore_index=True,
        sort=False,
    )

    # Avoid simply duplicating an existing training pair.
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


def main():
    m070.seed_everything(
        SEED
    )

    started = time.perf_counter()

    questions = pd.read_csv(
        QUESTIONS
    )

    questions[
        "gold_bool"
    ] = [
        parse_bool(x)
        for x
        in questions["answer"]
    ]

    raw_candidates = pd.read_csv(
        CANDIDATES
    )

    # Match M070 runtime distribution.
    old_top_k = m070.TOP_K
    m070.TOP_K = 12

    candidates = (
        m070.prepare_candidates(
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
            "evidence_tiou",
        ]
    ]

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

    audit = pd.read_csv(
        AUDIT
    )

    # Existing M070 OOF score lets us compare against
    # the exact current method on this same held-out fold.
    m070_oof = pd.read_csv(
        M070_OOF
    )[
        [
            "question_id",
            "m070_max",
        ]
    ]

    base = base.merge(
        m070_oof,
        on="question_id",
        validate="one_to_one",
    )

    gold = (
        base["gold_bool"]
        .astype(bool)
        .to_numpy()
    )

    groups = (
        base["transcript_id"]
        .astype(str)
        .to_numpy()
    )

    splitter = GroupKFold(
        n_splits=5
    )

    chosen = None

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
        if fold == FOLD_TO_RUN:
            chosen = (
                train_idx,
                test_idx,
            )
            break

    if chosen is None:
        raise RuntimeError(
            "Requested fold not found."
        )

    train_idx, test_idx = chosen

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
    # Standard successful M070 training set.
    # ------------------------------------------------------------

    original_pairs = (
        m070.build_training_pairs(
            candidates,
            train_ids,
        )
    )

    # ------------------------------------------------------------
    # Same-conversation confounders.
    # ------------------------------------------------------------

    contrastive = (
        build_contrastive_pairs(
            audit,
            train_ids,
        )
    )

    training_pairs = combine_pairs(
        original_pairs,
        contrastive,
    )

    print("=" * 78)
    print(
        "M073 FAST CONTRASTIVE PILOT"
    )
    print("=" * 78)

    print(
        f"Fold:                  "
        f"{FOLD_TO_RUN}"
    )

    print(
        f"Train questions:       "
        f"{len(train_ids)}"
    )

    print(
        f"Test questions:        "
        f"{len(test_ids)}"
    )

    print(
        f"Original pairs:        "
        f"{len(original_pairs)}"
    )

    print(
        f"Contrastive added:     "
        f"{len(contrastive)}"
    )

    print(
        f"Combined pairs:        "
        f"{len(training_pairs)}"
    )

    print()
    print(
        contrastive[
            "source"
        ]
        .value_counts()
        .to_string()
    )

    print()
    print(
        "Similarity mean:   "
        f"{contrastive['contrastive_similarity'].mean():.4f}"
    )

    print(
        "Similarity median: "
        f"{contrastive['contrastive_similarity'].median():.4f}"
    )

    print(
        "Similarity p90:    "
        f"{contrastive['contrastive_similarity'].quantile(.90):.4f}"
    )

    print()
    print("TOP CONTRASTIVE EXAMPLES")
    print("-" * 78)

    examples = (
        contrastive.sort_values(
            "contrastive_similarity",
            ascending=False,
        )
        .head(12)
    )

    for _, row in examples.iterrows():
        print()
        print(
            f"[{row['source']}] "
            f"sim="
            f"{row['contrastive_similarity']:.3f}"
        )
        print(
            "Q:",
            row["question"],
        )
        print(
            "E:",
            row["candidate_text"],
        )

    # ------------------------------------------------------------
    # Train only one model.
    # ------------------------------------------------------------

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    old_epochs = m070.EPOCHS
    m070.EPOCHS = EPOCHS

    (
        tokenizer,
        model,
        entail_idx,
        contradiction_idx,
        neutral_idx,
    ) = m070.train_fold(
        training_pairs,
        device,
        fold=73,
    )

    m070.EPOCHS = old_epochs

    # ------------------------------------------------------------
    # Held-out candidate scoring.
    # ------------------------------------------------------------

    test_candidates = (
        candidates[
            candidates[
                "question_id"
            ].isin(test_ids)
        ]
        .copy()
        .reset_index(drop=True)
    )

    test_candidates[
        "m070_candidate_score"
    ] = m070.score_candidates(
        test_candidates,
        tokenizer,
        model,
        entail_idx,
        contradiction_idx,
        neutral_idx,
        device,
    )

    q_scores = (
        m070.aggregate_questions(
            test_candidates
        )
    )

    test = (
        base.iloc[
            test_idx
        ]
        .copy()
        .merge(
            q_scores[
                [
                    "question_id",
                    "m070_max",
                ]
            ].rename(
                columns={
                    "m070_max":
                        "m073_max"
                }
            ),
            on="question_id",
            validate="one_to_one",
        )
    )

    test_gold = (
        test["gold_bool"]
        .astype(bool)
        .to_numpy()
    )

    evidence = (
        test["evidence_tiou"]
        .fillna(0.0)
        .to_numpy(dtype=float)
    )

    # ------------------------------------------------------------
    # Actual M071 deployed rule on this fold.
    # ------------------------------------------------------------

    baseline_score = (
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

    baseline_pred = (
        baseline_score
        >= FUSION_THRESHOLD
    )

    baseline = competition_score(
        test_gold,
        baseline_pred,
        evidence,
    )

    # ------------------------------------------------------------
    # Same rule, only substitute new M073 signal.
    # ------------------------------------------------------------

    pilot_score = (
        M055_WEIGHT
        * test[
            "m055_probability"
        ].to_numpy()
        +
        M070_WEIGHT
        * test[
            "m073_max"
        ].to_numpy()
    )

    pilot_pred = (
        pilot_score
        >= FUSION_THRESHOLD
    )

    pilot = competition_score(
        test_gold,
        pilot_pred,
        evidence,
    )

    changed = int(
        np.sum(
            baseline_pred
            != pilot_pred
        )
    )

    pilot_fixes = int(
        np.sum(
            (baseline_pred != test_gold)
            & (pilot_pred == test_gold)
        )
    )

    pilot_breaks = int(
        np.sum(
            (baseline_pred == test_gold)
            & (pilot_pred != test_gold)
        )
    )

    print()
    print("=" * 78)
    print(
        "M073 PILOT RESULT"
    )
    print("=" * 78)

    print(
        f"M071 baseline fold {FOLD_TO_RUN}: "
        f"acc={baseline[0]:.4f}, "
        f"tIoU={baseline[1]:.4f}, "
        f"comp={baseline[2]:.4f}"
    )

    print(
        f"M073 pilot fold {FOLD_TO_RUN}:    "
        f"acc={pilot[0]:.4f}, "
        f"tIoU={pilot[1]:.4f}, "
        f"comp={pilot[2]:.4f}"
    )

    print()
    print(
        f"Delta composite: "
        f"{pilot[2]-baseline[2]:+.4f}"
    )

    print(
        f"Predictions changed: {changed}"
    )

    print(
        f"Fixes:               {pilot_fixes}"
    )

    print(
        f"Breaks:              {pilot_breaks}"
    )

    print(
        f"Wall time: "
        f"{time.perf_counter()-started:.1f}s"
    )

    del model
    del tokenizer

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    m070.TOP_K = old_top_k


if __name__ == "__main__":
    main()