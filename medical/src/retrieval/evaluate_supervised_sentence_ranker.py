from __future__ import annotations

import json
import re
import time
import joblib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import CrossEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GroupKFold


QUESTIONS = Path(
    r"C:\Users\Calle\Projects\Nordic-AI-Cup-2026-official"
    r"\medical-appointment\data\question_train.csv"
)

ASR_DIR = Path(
    r"medical\artifacts\asr\faster_distil_large_v3_int8f32_b1"
)

M010 = Path(
    r"medical\artifacts\retrieval\cross_encoder_segment_results.csv"
)

OUTPUT = Path(
    r"medical\artifacts\retrieval\supervised_sentence_ranker_results.csv"
)

ALL_CANDIDATES_OUTPUT = Path(
    r"medical\artifacts\retrieval\m025_all_candidate_oof_scores.csv"
)

FINAL_MODEL_OUTPUT = Path(
    r"medical\artifacts\models\sentence_ranker.joblib"
)

MODEL_ID = "cross-encoder/ms-marco-MiniLM-L6-v2"

MAX_SENTENCES = 3
N_SPLITS = 5


def normalize_transcript_id(value: object) -> str:
    text = Path(str(value).strip()).stem

    if text.startswith("conversation_sample_"):
        return text

    if text.startswith("sample_"):
        return f"conversation_{text}"

    if text.isdigit():
        return f"conversation_sample_{int(text)}"

    return text


def temporal_iou(
    start_a: float,
    end_a: float,
    start_b: float,
    end_b: float,
) -> float:
    intersection = max(
        0.0,
        min(end_a, end_b)
        - max(start_a, start_b),
    )

    union = (
        max(end_a, end_b)
        - min(start_a, start_b)
    )

    if union <= 0:
        return 0.0

    return intersection / union


def load_words(
    transcript_id: object,
) -> list[dict]:
    stem = normalize_transcript_id(
        transcript_id
    )

    path = ASR_DIR / f"{stem}.json"

    data = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    words = []

    for item in data["words"]:
        if (
            item.get("start") is None
            or item.get("end") is None
        ):
            continue

        words.append(
            {
                "start": float(item["start"]),
                "end": float(item["end"]),
                "word": str(item["word"]),
            }
        )

    return words


def is_sentence_end(text: str) -> bool:
    return bool(
        re.search(
            r'[.!?]["\']?$',
            text.strip(),
        )
    )


def make_sentences(
    words: list[dict],
) -> list[dict]:
    sentences = []
    start_index = 0

    for index, word in enumerate(words):
        if not is_sentence_end(
            word["word"]
        ):
            continue

        text = "".join(
            item["word"]
            for item in words[
                start_index:
                index + 1
            ]
        ).strip()

        if text:
            sentences.append(
                {
                    "start": words[
                        start_index
                    ]["start"],
                    "end": words[
                        index
                    ]["end"],
                    "text": text,
                }
            )

        start_index = index + 1

    if start_index < len(words):
        text = "".join(
            item["word"]
            for item in words[
                start_index:
            ]
        ).strip()

        if text:
            sentences.append(
                {
                    "start": words[
                        start_index
                    ]["start"],
                    "end": words[-1]["end"],
                    "text": text,
                }
            )

    return sentences


def make_candidates(
    sentences: list[dict],
) -> list[dict]:
    candidates = []

    for size in range(
        1,
        MAX_SENTENCES + 1,
    ):
        for start in range(
            len(sentences) - size + 1
        ):
            group = sentences[
                start:
                start + size
            ]

            text = " ".join(
                item["text"]
                for item in group
            )

            candidates.append(
                {
                    "sentence_count": size,
                    "start": group[0]["start"],
                    "end": group[-1]["end"],
                    "text": text,
                    "word_count": len(
                        text.split()
                    ),
                }
            )

    return candidates


def lexical_scores(
    question: str,
    texts: list[str],
) -> np.ndarray:
    documents = texts + [question]

    word_vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        sublinear_tf=True,
        token_pattern=r"(?u)\b\w+\b",
    )

    word_matrix = (
        word_vectorizer.fit_transform(
            documents
        )
    )

    word_scores = cosine_similarity(
        word_matrix[-1],
        word_matrix[:-1],
    )[0]

    char_vectorizer = TfidfVectorizer(
        lowercase=True,
        analyzer="char_wb",
        ngram_range=(3, 5),
        sublinear_tf=True,
    )

    char_matrix = (
        char_vectorizer.fit_transform(
            documents
        )
    )

    char_scores = cosine_similarity(
        char_matrix[-1],
        char_matrix[:-1],
    )[0]

    return (
        0.55 * word_scores
        + 0.45 * char_scores
    )


def main() -> None:
    questions = pd.read_csv(
        QUESTIONS
    )

    positives = questions[
        questions["question_type"]
        == "positive"
    ].copy()

    m010 = pd.read_csv(
        M010
    ).set_index(
        "question_id"
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 78)
    print(
        "Nordic AI Cup 2026 — "
        "Supervised Sentence Span Ranker"
    )
    print("=" * 78)
    print(f"Semantic model: {MODEL_ID}")
    print(f"Device:         {device}")
    print()

    semantic_model = CrossEncoder(
        MODEL_ID,
        device=device,
        max_length=256,
    )

    candidate_cache = {}

    all_candidate_rows = []

    feature_start = time.perf_counter()

    for row_number, (_, row) in enumerate(
        positives.iterrows(),
        start=1,
    ):
        question_id = row[
            "question_id"
        ]

        transcript_id = row[
            "transcript_id"
        ]

        stem = normalize_transcript_id(
            transcript_id
        )

        if stem not in candidate_cache:
            words = load_words(
                transcript_id
            )

            sentences = make_sentences(
                words
            )

            candidate_cache[stem] = (
                make_candidates(
                    sentences
                )
            )

        candidates = candidate_cache[
            stem
        ]

        question = str(
            row["question"]
        )

        texts = [
            candidate["text"]
            for candidate in candidates
        ]

        lexical = lexical_scores(
            question,
            texts,
        )

        pairs = [
            (
                question,
                text,
            )
            for text in texts
        ]

        semantic = np.asarray(
            semantic_model.predict(
                pairs,
                batch_size=64,
                show_progress_bar=False,
            )
        ).reshape(-1)

        anchor = m010.loc[
            question_id
        ]

        anchor_start = float(
            anchor["pred_start"]
        )

        anchor_end = float(
            anchor["pred_end"]
        )

        anchor_mid = (
            anchor_start
            + anchor_end
        ) / 2.0

        gold_start = float(
            row["evidence_start"]
        )

        gold_end = float(
            row["evidence_end"]
        )

        for index, candidate in enumerate(
            candidates
        ):
            candidate_start = float(
                candidate["start"]
            )

            candidate_end = float(
                candidate["end"]
            )

            candidate_mid = (
                candidate_start
                + candidate_end
            ) / 2.0

            duration = (
                candidate_end
                - candidate_start
            )

            anchor_overlap = temporal_iou(
                candidate_start,
                candidate_end,
                anchor_start,
                anchor_end,
            )

            target = temporal_iou(
                candidate_start,
                candidate_end,
                gold_start,
                gold_end,
            )

            all_candidate_rows.append(
                {
                    "question_id": (
                        question_id
                    ),
                    "transcript_id": (
                        transcript_id
                    ),
                    "candidate_index": index,
                    "candidate_start": (
                        candidate_start
                    ),
                    "candidate_end": (
                        candidate_end
                    ),
                    "candidate_text": (
                        candidate["text"]
                    ),
                    "sentence_count": (
                        candidate[
                            "sentence_count"
                        ]
                    ),
                    "word_count": (
                        candidate[
                            "word_count"
                        ]
                    ),
                    "duration": duration,
                    "lexical_score": float(
                        lexical[index]
                    ),
                    "semantic_score": float(
                        semantic[index]
                    ),
                    "anchor_iou": (
                        anchor_overlap
                    ),
                    "anchor_mid_distance": (
                        abs(
                            candidate_mid
                            - anchor_mid
                        )
                    ),
                    "anchor_start_distance": (
                        abs(
                            candidate_start
                            - anchor_start
                        )
                    ),
                    "anchor_end_distance": (
                        abs(
                            candidate_end
                            - anchor_end
                        )
                    ),
                    "contains_anchor_mid": (
                        float(
                            candidate_start
                            <= anchor_mid
                            <= candidate_end
                        )
                    ),
                    "target_tiou": (
                        target
                    ),
                }
            )

        if (
            row_number % 25 == 0
            or row_number
            == len(positives)
        ):
            print(
                f"[{row_number:03d}/"
                f"{len(positives):03d}] "
                f"features generated"
            )

    candidate_df = pd.DataFrame(
        all_candidate_rows
    )

    feature_columns = [
        "sentence_count",
        "word_count",
        "duration",
        "lexical_score",
        "semantic_score",
        "anchor_iou",
        "anchor_mid_distance",
        "anchor_start_distance",
        "anchor_end_distance",
        "contains_anchor_mid",
    ]

    question_df = positives[
        [
            "question_id",
            "transcript_id",
        ]
    ].reset_index(drop=True)

    groups = question_df[
        "transcript_id"
    ].astype(str)

    splitter = GroupKFold(
        n_splits=N_SPLITS
    )

    predictions = {}

    fold_number = 0

    for train_q, test_q in splitter.split(
        question_df,
        groups=groups,
    ):
        fold_number += 1

        train_ids = set(
            question_df.iloc[
                train_q
            ]["question_id"]
        )

        test_ids = set(
            question_df.iloc[
                test_q
            ]["question_id"]
        )

        train_mask = candidate_df[
            "question_id"
        ].isin(train_ids)

        test_mask = candidate_df[
            "question_id"
        ].isin(test_ids)

        train = candidate_df[
            train_mask
        ]

        test = candidate_df[
            test_mask
        ]

        # Each question contributes equal total weight.
        candidate_counts = (
            train.groupby(
                "question_id"
            )["question_id"]
            .transform("count")
            .to_numpy()
        )

        sample_weight = (
            1.0
            / candidate_counts
        )

        model = (
            HistGradientBoostingRegressor(
                learning_rate=0.06,
                max_iter=250,
                max_leaf_nodes=15,
                min_samples_leaf=30,
                l2_regularization=1.0,
                random_state=42,
            )
        )

        model.fit(
            train[
                feature_columns
            ].to_numpy(),
            train[
                "target_tiou"
            ].to_numpy(),
            sample_weight=sample_weight,
        )

        fold_pred = model.predict(
            test[
                feature_columns
            ].to_numpy()
        )

        for dataframe_index, score in zip(
            test.index,
            fold_pred,
        ):
            predictions[
                dataframe_index
            ] = float(score)

        print(
            f"Fold {fold_number}: "
            f"train_questions={len(train_ids)}, "
            f"test_questions={len(test_ids)}"
        )

    if len(predictions) != len(
        candidate_df
    ):
        raise RuntimeError(
            "Not all candidate rows received "
            "an OOF prediction."
        )

    candidate_df[
        "oof_predicted_tiou"
    ] = [
        predictions[index]
        for index in candidate_df.index
    ]
    
    ALL_CANDIDATES_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    candidate_df.to_csv(
        ALL_CANDIDATES_OUTPUT,
        index=False,
    )

    print(
        f"All-candidate OOF scores: "
        f"{ALL_CANDIDATES_OUTPUT}"
    )

    selected_rows = []

    for question_id, group in (
        candidate_df.groupby(
            "question_id",
            sort=False,
        )
    ):
        best_index = group[
            "oof_predicted_tiou"
        ].idxmax()

        selected = candidate_df.loc[
            best_index
        ].copy()

        oracle_tiou = float(
            group[
                "target_tiou"
            ].max()
        )

        selected[
            "candidate_oracle_tiou"
        ] = oracle_tiou

        selected_rows.append(
            selected
        )

    result_df = pd.DataFrame(
        selected_rows
    )

    result_df = result_df.merge(
        positives[
            [
                "question_id",
                "question",
                "evidence_start",
                "evidence_end",
            ]
        ],
        on="question_id",
        how="left",
        validate="one_to_one",
    )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_df.to_csv(
        OUTPUT,
        index=False,
    )

    scores = result_df[
        "target_tiou"
    ]

    oracle = result_df[
        "candidate_oracle_tiou"
    ]

    elapsed = (
        time.perf_counter()
        - feature_start
    )
    
        # ---------------------------------------------------------
    # Train deployment model on ALL supplied positive questions.
    #
    # OOF metrics above remain the development estimate.
    # This full-data model is ONLY for hidden validation/evaluation.
    # ---------------------------------------------------------

    full_candidate_counts = (
        candidate_df.groupby(
            "question_id"
        )["question_id"]
        .transform("count")
        .to_numpy()
    )

    full_sample_weight = (
        1.0
        / full_candidate_counts
    )

    final_model = HistGradientBoostingRegressor(
        learning_rate=0.06,
        max_iter=250,
        max_leaf_nodes=15,
        min_samples_leaf=30,
        l2_regularization=1.0,
        random_state=42,
    )

    final_model.fit(
        candidate_df[
            feature_columns
        ].to_numpy(),
        candidate_df[
            "target_tiou"
        ].to_numpy(),
        sample_weight=full_sample_weight,
    )

    FINAL_MODEL_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        {
            "model": final_model,
            "feature_columns": feature_columns,
            "max_sentences": MAX_SENTENCES,
            "training_questions": int(
                positives["question_id"].nunique()
            ),
        },
        FINAL_MODEL_OUTPUT,
    )

    print()
    print(
        f"Deployment ranker saved: "
        f"{FINAL_MODEL_OUTPUT}"
    )

    print()
    print("=" * 78)
    print("RESULT")
    print("=" * 78)

    print(
        f"Questions:       "
        f"{len(result_df)}"
    )

    print()
    print("OOF LOCALIZATION")
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

    print()
    print("CANDIDATE CEILING")
    print("-" * 78)

    print(
        f"Oracle mean:     "
        f"{oracle.mean():.4f}"
    )

    print(
        f"Oracle median:   "
        f"{oracle.median():.4f}"
    )

    print()
    print("REFERENCE")
    print("-" * 78)

    print(
        "M010 segment mean:           0.4347"
    )

    print(
        "M024 zero-shot sentence:     0.3576"
    )

    print(
        "M023 sentence oracle:        0.7997"
    )

    print()
    print(
        f"Experiment wall time: "
        f"{elapsed:.2f} s"
    )

    print(
        f"Detailed results: {OUTPUT}"
    )


if __name__ == "__main__":
    main()