from __future__ import annotations

import json
import random
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


# ============================================================================
# Paths
# ============================================================================

QUESTIONS = Path(
    r"C:\Users\Calle\Projects\Nordic-AI-Cup-2026-official"
    r"\medical-appointment\data\question_train.csv"
)

ASR_DIR = Path(
    r"medical\artifacts\asr\faster_distil_large_v3_int8f32_b1"
)

NLI_RESULTS = Path(
    r"medical\artifacts\classification\nli_segmentwise_results.csv"
)

MODEL_DIR = Path(
    r"medical\artifacts\models"
)

META_CLASSIFIER_PATH = (
    MODEL_DIR
    / "meta_classifier_m042.joblib"
)

EVIDENCE_MODEL_DIR = (
    MODEL_DIR
    / "hybrid_evidence_ranker_m049"
)

DEPLOYMENT_METADATA_PATH = (
    MODEL_DIR
    / "m049_deployment_metadata.json"
)


# ============================================================================
# Deployment constants
# ============================================================================

SEED = 42

# M049 deployment gate chosen from CV threshold behavior.
EVIDENCE_GATE_THRESHOLD = 0.369304

# Evidence model = M036/M047 configuration.
EVIDENCE_MODEL_ID = (
    "cross-encoder/"
    "ms-marco-MiniLM-L6-v2"
)

MAX_SENTENCES = 3

WORD_WINDOW_SIZES = (
    6,
    10,
)

WORD_WINDOW_STRIDE = 3

MAX_LENGTH = 192
BATCH_SIZE = 32
EVAL_BATCH_SIZE = 128

EPOCHS = 4
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01

POSITIVE_TIOU_THRESHOLD = 0.10

MAX_POSITIVE_CANDIDATES = 18
MAX_HARD_NEGATIVES = 18
MAX_RANDOM_NEGATIVES = 6


# ============================================================================
# M042 feature definition
# ============================================================================

NLI_FEATURES = [
    "max_segment_ratio",
    "max_segment_entailment",
    "max_segment_margin",
    "max_segment_vs_other",
    "selected_entailment",
    "selected_contradiction",
    "selected_neutral",
]

SEMANTIC_SURFACE_FEATURES = [
    "question_word_count",
    "question_char_count",
    "has_number",
    "has_unit",
    "has_negation",
]

AUXILIARY_FEATURES = [
    "starts_did",
    "starts_does",
    "starts_is",
    "starts_are",
    "starts_was",
    "starts_were",
    "starts_has",
    "starts_have",
    "starts_will",
    "starts_should",
]

META_FEATURES = (
    NLI_FEATURES
    + SEMANTIC_SURFACE_FEATURES
    + AUXILIARY_FEATURES
)


NEGATION_TERMS = {
    "no",
    "not",
    "never",
    "none",
    "without",
    "unchanged",
    "discontinued",
    "stop",
    "stopped",
}


UNIT_PATTERN = re.compile(
    r"\b(?:"
    r"mg|mcg|g|kg|ml|l|"
    r"mmol|mmol/l|mg/dl|"
    r"cm|mm|%|bpm"
    r")\b",
    flags=re.IGNORECASE,
)

NUMBER_PATTERN = re.compile(
    r"\b\d+(?:\.\d+)?\b"
)


# ============================================================================
# General helpers
# ============================================================================

def seed_everything(
    seed: int,
) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )


def parse_bool(
    value: object,
) -> bool:
    if isinstance(value, bool):
        return value

    text = str(
        value
    ).strip().lower()

    if text in {
        "true",
        "1",
        "yes",
    }:
        return True

    if text in {
        "false",
        "0",
        "no",
    }:
        return False

    raise ValueError(
        f"Cannot parse bool: "
        f"{value!r}"
    )


def normalize_transcript_id(
    value: object,
) -> str:
    text = Path(
        str(value).strip()
    ).stem

    if text.startswith(
        "conversation_sample_"
    ):
        return text

    if text.startswith(
        "sample_"
    ):
        return (
            f"conversation_{text}"
        )

    if text.isdigit():
        return (
            f"conversation_sample_"
            f"{int(text)}"
        )

    return text


# ============================================================================
# M042 question-surface features
# ============================================================================

def add_question_features(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    df = dataframe.copy()

    questions = (
        df["question"]
        .fillna("")
        .astype(str)
    )

    lower = (
        questions.str.lower()
    )

    df[
        "question_word_count"
    ] = (
        questions
        .str.split()
        .str.len()
    )

    df[
        "question_char_count"
    ] = (
        questions.str.len()
    )

    df[
        "has_number"
    ] = (
        questions.str.contains(
            NUMBER_PATTERN,
            regex=True,
        )
        .astype(int)
    )

    df[
        "has_unit"
    ] = (
        questions.str.contains(
            UNIT_PATTERN,
            regex=True,
        )
        .astype(int)
    )

    df[
        "has_negation"
    ] = lower.map(
        lambda text: int(
            any(
                re.search(
                    rf"\b"
                    rf"{re.escape(term)}"
                    rf"\b",
                    text,
                )
                for term
                in NEGATION_TERMS
            )
        )
    )

    for auxiliary in [
        "did",
        "does",
        "is",
        "are",
        "was",
        "were",
        "has",
        "have",
        "will",
        "should",
    ]:
        df[
            f"starts_{auxiliary}"
        ] = (
            lower
            .str.startswith(
                auxiliary + " "
            )
            .astype(int)
        )

    return df


# ============================================================================
# M042 export
# ============================================================================

def make_meta_classifier():
    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "model",
                HistGradientBoostingClassifier(
                    learning_rate=0.05,
                    max_iter=150,
                    max_leaf_nodes=7,
                    min_samples_leaf=20,
                    l2_regularization=1.0,
                    random_state=42,
                ),
            ),
        ]
    )


def train_meta_classifier():
    print()
    print("=" * 78)
    print(
        "TRAINING DEPLOYMENT "
        "M042 META-CLASSIFIER"
    )
    print("=" * 78)

    df = pd.read_csv(
        NLI_RESULTS
    )

    if len(df) != 390:
        raise RuntimeError(
            f"Expected 390 NLI rows; "
            f"got {len(df)}"
        )

    df = add_question_features(
        df
    )

    df[
        "gold_bool"
    ] = [
        parse_bool(value)
        for value
        in df["gold_yes"]
    ]

    forbidden = {
        "question_type",
        "gold_yes",
        "gold_bool",
        "correct",
    }

    leakage = (
        forbidden
        & set(
            META_FEATURES
        )
    )

    if leakage:
        raise RuntimeError(
            "Leakage in deployment "
            "feature set: "
            f"{sorted(leakage)}"
        )

    X = df[
        META_FEATURES
    ]

    y = (
        df["gold_bool"]
        .to_numpy(
            dtype=bool
        )
    )

    model = (
        make_meta_classifier()
    )

    model.fit(
        X,
        y,
    )

    training_probabilities = (
        model.predict_proba(
            X
        )[:, 1]
    )

    print(
        f"Training rows: "
        f"{len(df)}"
    )

    print(
        f"Features:      "
        f"{len(META_FEATURES)}"
    )

    print(
        f"Mean p(YES):   "
        f"{training_probabilities.mean():.4f}"
    )

    payload = {
        "model": model,
        "feature_columns": (
            META_FEATURES
        ),
        "training_rows": int(
            len(df)
        ),
        "model_name": (
            "M042 full HGB"
        ),
    }

    META_CLASSIFIER_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        payload,
        META_CLASSIFIER_PATH,
    )

    print(
        "Saved: "
        f"{META_CLASSIFIER_PATH}"
    )

    return payload


# ============================================================================
# Evidence candidate generation
# ============================================================================

def temporal_iou(
    pred_start: float,
    pred_end: float,
    gold_start: float,
    gold_end: float,
) -> float:
    intersection = max(
        0.0,
        min(
            pred_end,
            gold_end,
        )
        - max(
            pred_start,
            gold_start,
        ),
    )

    union = (
        max(
            pred_end,
            gold_end,
        )
        - min(
            pred_start,
            gold_start,
        )
    )

    if union <= 0:
        return 0.0

    return (
        intersection
        / union
    )


def load_words(
    transcript_id: object,
) -> list[dict]:
    stem = (
        normalize_transcript_id(
            transcript_id
        )
    )

    path = (
        ASR_DIR
        / f"{stem}.json"
    )

    if not path.exists():
        raise FileNotFoundError(
            path
        )

    data = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    words = []

    for item in data["words"]:
        start = item.get(
            "start"
        )

        end = item.get(
            "end"
        )

        if (
            start is None
            or end is None
        ):
            continue

        words.append(
            {
                "start": float(
                    start
                ),
                "end": float(
                    end
                ),
                "word": str(
                    item["word"]
                ),
            }
        )

    return words


def is_sentence_end(
    text: str,
) -> bool:
    return bool(
        re.search(
            r'[.!?]["\']?$',
            text.strip(),
        )
    )


def make_sentences(
    words: list[dict],
) -> list[dict]:
    if not words:
        return []

    sentences = []

    start_index = 0

    for index, word in enumerate(
        words
    ):
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
                    "start": float(
                        words[
                            start_index
                        ]["start"]
                    ),
                    "end": float(
                        words[
                            index
                        ]["end"]
                    ),
                    "text": text,
                }
            )

        start_index = (
            index + 1
        )

    if start_index < len(
        words
    ):
        text = "".join(
            item["word"]
            for item in words[
                start_index:
            ]
        ).strip()

        if text:
            sentences.append(
                {
                    "start": float(
                        words[
                            start_index
                        ]["start"]
                    ),
                    "end": float(
                        words[-1]["end"]
                    ),
                    "text": text,
                }
            )

    return sentences


def make_sentence_candidates(
    sentences: list[dict],
) -> list[dict]:
    candidates = []

    for size in range(
        1,
        MAX_SENTENCES + 1,
    ):
        for start in range(
            len(sentences)
            - size
            + 1
        ):
            group = sentences[
                start:
                start + size
            ]

            candidates.append(
                {
                    "candidate_kind": (
                        f"sentence_{size}"
                    ),
                    "start": float(
                        group[0]["start"]
                    ),
                    "end": float(
                        group[-1]["end"]
                    ),
                    "text": " ".join(
                        item["text"]
                        for item
                        in group
                    ),
                }
            )

    return candidates


def make_word_candidates(
    words: list[dict],
) -> list[dict]:
    candidates = []

    for size in (
        WORD_WINDOW_SIZES
    ):
        if len(words) < size:
            continue

        start_indices = list(
            range(
                0,
                len(words)
                - size
                + 1,
                WORD_WINDOW_STRIDE,
            )
        )

        final_start = (
            len(words)
            - size
        )

        if (
            final_start
            not in start_indices
        ):
            start_indices.append(
                final_start
            )

        for start_index in (
            start_indices
        ):
            end_index = (
                start_index
                + size
                - 1
            )

            text = "".join(
                item["word"]
                for item in words[
                    start_index:
                    end_index + 1
                ]
            ).strip()

            candidates.append(
                {
                    "candidate_kind": (
                        f"word_{size}"
                    ),
                    "start": float(
                        words[
                            start_index
                        ]["start"]
                    ),
                    "end": float(
                        words[
                            end_index
                        ]["end"]
                    ),
                    "text": text,
                }
            )

    return candidates


def deduplicate_candidates(
    candidates: list[dict],
) -> list[dict]:
    seen = set()
    output = []

    for candidate in candidates:
        key = (
            round(
                candidate["start"],
                3,
            ),
            round(
                candidate["end"],
                3,
            ),
        )

        if key in seen:
            continue

        seen.add(key)

        output.append(
            candidate
        )

    return output


def build_positive_training_dataframe(
    questions: pd.DataFrame,
) -> pd.DataFrame:
    positives = questions[
        questions["question_type"]
        == "positive"
    ].copy()

    if len(positives) != 195:
        raise RuntimeError(
            f"Expected 195 positives; "
            f"got {len(positives)}"
        )

    transcript_cache = {}

    rows = []

    for number, (_, row) in enumerate(
        positives.iterrows(),
        start=1,
    ):
        transcript_id = (
            row[
                "transcript_id"
            ]
        )

        stem = (
            normalize_transcript_id(
                transcript_id
            )
        )

        if stem not in (
            transcript_cache
        ):
            words = load_words(
                transcript_id
            )

            sentences = (
                make_sentences(
                    words
                )
            )

            candidates = (
                make_sentence_candidates(
                    sentences
                )
                + make_word_candidates(
                    words
                )
            )

            transcript_cache[
                stem
            ] = (
                deduplicate_candidates(
                    candidates
                )
            )

        gold_start = float(
            row[
                "evidence_start"
            ]
        )

        gold_end = float(
            row[
                "evidence_end"
            ]
        )

        for (
            candidate_index,
            candidate,
        ) in enumerate(
            transcript_cache[
                stem
            ]
        ):
            tiou = temporal_iou(
                candidate[
                    "start"
                ],
                candidate[
                    "end"
                ],
                gold_start,
                gold_end,
            )

            rows.append(
                {
                    "question_id": (
                        row[
                            "question_id"
                        ]
                    ),
                    "question": str(
                        row[
                            "question"
                        ]
                    ),
                    "candidate_index": (
                        candidate_index
                    ),
                    "candidate_kind": (
                        candidate[
                            "candidate_kind"
                        ]
                    ),
                    "candidate_start": (
                        candidate[
                            "start"
                        ]
                    ),
                    "candidate_end": (
                        candidate[
                            "end"
                        ]
                    ),
                    "candidate_text": (
                        candidate[
                            "text"
                        ]
                    ),
                    "target_tiou": (
                        tiou
                    ),
                }
            )

        if (
            number % 25 == 0
            or number
            == len(positives)
        ):
            print(
                f"[{number:03d}/"
                f"{len(positives):03d}] "
                "training candidates built"
            )

    return pd.DataFrame(
        rows
    )


# ============================================================================
# Evidence model training
# ============================================================================

class PairDataset(
    Dataset
):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer,
    ):
        self.df = (
            dataframe
            .reset_index(
                drop=True
            )
        )

        self.tokenizer = (
            tokenizer
        )

    def __len__(self):
        return len(
            self.df
        )

    def __getitem__(
        self,
        index: int,
    ):
        row = (
            self.df.iloc[
                index
            ]
        )

        encoded = (
            self.tokenizer(
                str(
                    row["question"]
                ),
                str(
                    row[
                        "candidate_text"
                    ]
                ),
                truncation=True,
                max_length=(
                    MAX_LENGTH
                ),
                padding=False,
            )
        )

        encoded[
            "labels"
        ] = float(
            row[
                "target_tiou"
            ]
        )

        return encoded


def make_collator(
    tokenizer,
):
    def collate(batch):
        labels = torch.tensor(
            [
                item.pop(
                    "labels"
                )
                for item
                in batch
            ],
            dtype=torch.float32,
        )

        encoded = tokenizer.pad(
            batch,
            padding=True,
            return_tensors="pt",
        )

        encoded[
            "labels"
        ] = labels

        return encoded

    return collate


def score_raw(
    model,
    tokenizer,
    dataframe: pd.DataFrame,
    device: str,
) -> np.ndarray:
    values = []

    model.eval()

    for start in range(
        0,
        len(dataframe),
        EVAL_BATCH_SIZE,
    ):
        batch = dataframe.iloc[
            start:
            start
            + EVAL_BATCH_SIZE
        ]

        encoded = tokenizer(
            batch[
                "question"
            ]
            .astype(str)
            .tolist(),
            batch[
                "candidate_text"
            ]
            .astype(str)
            .tolist(),
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

        with (
            torch.inference_mode()
        ):
            logits = model(
                **encoded
            ).logits.reshape(
                -1
            )

        values.extend(
            logits
            .detach()
            .cpu()
            .float()
            .numpy()
            .tolist()
        )

    return np.asarray(
        values,
        dtype=np.float32,
    )


def sample_training_candidates(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    selected = []

    rng = (
        np.random.default_rng(
            SEED
        )
    )

    for _, group in (
        dataframe.groupby(
            "question_id",
            sort=False,
        )
    ):
        positive = group[
            group[
                "target_tiou"
            ]
            >= (
                POSITIVE_TIOU_THRESHOLD
            )
        ].sort_values(
            "target_tiou",
            ascending=False,
        ).head(
            MAX_POSITIVE_CANDIDATES
        )

        negative = group[
            group[
                "target_tiou"
            ]
            < (
                POSITIVE_TIOU_THRESHOLD
            )
        ]

        hard = (
            negative
            .sort_values(
                "base_score",
                ascending=False,
            )
            .head(
                MAX_HARD_NEGATIVES
            )
        )

        remaining = (
            negative.drop(
                index=hard.index,
                errors="ignore",
            )
        )

        if len(
            remaining
        ) > 0:
            count = min(
                MAX_RANDOM_NEGATIVES,
                len(
                    remaining
                ),
            )

            indices = (
                rng.choice(
                    remaining.index
                    .to_numpy(),
                    size=count,
                    replace=False,
                )
            )

            random_negative = (
                remaining.loc[
                    indices
                ]
            )
        else:
            random_negative = (
                remaining
            )

        chosen = pd.concat(
            [
                positive,
                hard,
                random_negative,
            ]
        ).drop_duplicates(
            subset=[
                "question_id",
                "candidate_index",
            ]
        )

        selected.append(
            chosen
        )

    return pd.concat(
        selected,
        ignore_index=True,
    )


def train_evidence_ranker():
    print()
    print("=" * 78)
    print(
        "TRAINING DEPLOYMENT "
        "HYBRID EVIDENCE RANKER"
    )
    print("=" * 78)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for "
            "deployment evidence training."
        )

    device = "cuda"

    questions = pd.read_csv(
        QUESTIONS
    )

    training_df = (
        build_positive_training_dataframe(
            questions
        )
    )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            EVIDENCE_MODEL_ID
        )
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            EVIDENCE_MODEL_ID,
            num_labels=1,
            ignore_mismatched_sizes=True,
        )
        .to(device)
    )

    print()
    print(
        "Computing pretrained "
        "hard-negative scores..."
    )

    training_df[
        "base_score"
    ] = score_raw(
        model,
        tokenizer,
        training_df,
        device,
    )

    sampled = (
        sample_training_candidates(
            training_df
        )
    )

    print(
        f"All candidates: "
        f"{len(training_df)}"
    )

    print(
        f"Training pairs: "
        f"{len(sampled)}"
    )

    dataset = PairDataset(
        sampled,
        tokenizer,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=(
            make_collator(
                tokenizer
            )
        ),
    )

    optimizer = (
        torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=(
                WEIGHT_DECAY
            ),
        )
    )

    total_steps = (
        len(loader)
        * EPOCHS
    )

    warmup_steps = max(
        1,
        int(
            0.1
            * total_steps
        ),
    )

    def lr_factor(
        step: int,
    ):
        if (
            step
            < warmup_steps
        ):
            return (
                step + 1
            ) / warmup_steps

        remaining = max(
            1,
            total_steps
            - warmup_steps,
        )

        return max(
            0.0,
            (
                total_steps
                - step
            )
            / remaining,
        )

    scheduler = (
        torch.optim.lr_scheduler
        .LambdaLR(
            optimizer,
            lr_factor,
        )
    )

    model.train()

    for epoch in range(
        1,
        EPOCHS + 1,
    ):
        total_loss = 0.0
        batches = 0

        for batch in loader:
            labels = batch.pop(
                "labels"
            ).to(device)

            batch = {
                key: value.to(
                    device
                )
                for key, value
                in batch.items()
            }

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(
                **batch
            ).logits.reshape(
                -1
            )

            prediction = (
                torch.sigmoid(
                    logits
                )
            )

            loss = (
                torch.nn.functional
                .mse_loss(
                    prediction,
                    labels,
                )
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            optimizer.step()
            scheduler.step()

            total_loss += float(
                loss.item()
            )

            batches += 1

        print(
            f"epoch "
            f"{epoch}/{EPOCHS} "
            f"loss="
            f"{total_loss / max(1, batches):.6f}"
        )

    EVIDENCE_MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save_pretrained(
        EVIDENCE_MODEL_DIR
    )

    tokenizer.save_pretrained(
        EVIDENCE_MODEL_DIR
    )

    print(
        "Saved evidence model: "
        f"{EVIDENCE_MODEL_DIR}"
    )

    return {
        "training_candidates": int(
            len(training_df)
        ),
        "training_pairs": int(
            len(sampled)
        ),
    }


# ============================================================================
# Metadata
# ============================================================================

def save_metadata(
    *,
    meta_payload: dict,
    evidence_stats: dict,
) -> None:
    metadata = {
        "system": "M049",
        "seed": SEED,

        "classification": {
            "base_classifier": (
                "M018 max_segment_margin"
            ),
            "meta_classifier": (
                "M042 full HGB"
            ),
            "meta_classifier_path": str(
                META_CLASSIFIER_PATH
            ),
            "meta_features": (
                META_FEATURES
            ),
            "training_rows": (
                meta_payload[
                    "training_rows"
                ]
            ),
        },

        "evidence": {
            "model_id": (
                EVIDENCE_MODEL_ID
            ),
            "model_path": str(
                EVIDENCE_MODEL_DIR
            ),
            "candidate_types": [
                "sentence_1",
                "sentence_2",
                "sentence_3",
                "word_6",
                "word_10",
            ],
            "word_stride": (
                WORD_WINDOW_STRIDE
            ),
            "training_candidates": (
                evidence_stats[
                    "training_candidates"
                ]
            ),
            "training_pairs": (
                evidence_stats[
                    "training_pairs"
                ]
            ),
        },

        "arbitration": {
            "evidence_gate_threshold": (
                EVIDENCE_GATE_THRESHOLD
            ),
            "rule": (
                "If M018 and M042 agree, "
                "use that prediction. "
                "If they disagree, use M042 "
                "only when evidence_ranker_score "
                ">= threshold; otherwise use M018."
            ),
        },

        "development_reference": {
            "m049_oof_composite": 0.6354,
            "m050_fixed_threshold_diagnostic": (
                0.6429189764124594
            ),
            "fixed_threshold_diagnostic_is_unbiased": (
                False
            ),
        },
    }

    DEPLOYMENT_METADATA_PATH.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "Saved metadata: "
        f"{DEPLOYMENT_METADATA_PATH}"
    )


# ============================================================================
# Main
# ============================================================================

def main():
    seed_everything(
        SEED
    )

    print("=" * 78)
    print(
        "Nordic AI Cup 2026 — "
        "M049 Deployment Model Export"
    )
    print("=" * 78)

    print(
        f"Seed: "
        f"{SEED}"
    )

    print(
        f"Evidence gate: "
        f"{EVIDENCE_GATE_THRESHOLD:.6f}"
    )

    meta_payload = (
        train_meta_classifier()
    )

    evidence_stats = (
        train_evidence_ranker()
    )

    save_metadata(
        meta_payload=meta_payload,
        evidence_stats=(
            evidence_stats
        ),
    )

    print()
    print("=" * 78)
    print(
        "M049 DEPLOYMENT EXPORT COMPLETE"
    )
    print("=" * 78)

    print(
        "Meta-classifier:"
    )

    print(
        f"  {META_CLASSIFIER_PATH}"
    )

    print(
        "Evidence model:"
    )

    print(
        f"  {EVIDENCE_MODEL_DIR}"
    )

    print(
        "Metadata:"
    )

    print(
        f"  {DEPLOYMENT_METADATA_PATH}"
    )

    print()

    print(
        "Next step: patch MedicalPredictor "
        "and run the official local evaluator "
        "before touching the public endpoint."
    )


if __name__ == "__main__":
    main()