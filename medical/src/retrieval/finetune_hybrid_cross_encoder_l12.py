from __future__ import annotations

import json
import math
import random
import re
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


SEED = 42

QUESTIONS = Path(
    Path(__file__).resolve().parents[3].parent
    / "Nordic-AI-Cup-2026-official"
    / "medical-appointment"
    / "data"
    / "question_train.csv"
)

ASR_DIR = Path(
    r"medical\artifacts\asr\faster_distil_large_v3_int8f32_b1"
)

OUTPUT = Path(
    r"medical\artifacts\retrieval\finetuned_hybrid_l12_oof.csv"
)

ALL_CANDIDATES_OUTPUT = Path(
    r"medical\artifacts\retrieval\m038_all_candidate_oof_scores.csv"
)

MODEL_ID = "cross-encoder/ms-marco-MiniLM-L12-v2"

N_SPLITS = 5
MAX_SENTENCES = 3

WORD_WINDOW_SIZES = (
    6,
    10,
)

WORD_WINDOW_STRIDE = 3

MAX_LENGTH = 192
BATCH_SIZE = 12
EVAL_BATCH_SIZE = 48

EPOCHS = 4
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01

# Training-set reduction:
# keep useful positive/near-positive candidates plus hard negatives.
POSITIVE_TIOU_THRESHOLD = 0.10
MAX_POSITIVE_CANDIDATES = 18
MAX_HARD_NEGATIVES = 18
MAX_RANDOM_NEGATIVES = 6


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
    pred_start: float,
    pred_end: float,
    gold_start: float,
    gold_end: float,
) -> float:
    intersection = max(
        0.0,
        min(pred_end, gold_end)
        - max(pred_start, gold_start),
    )

    union = (
        max(pred_end, gold_end)
        - min(pred_start, gold_start)
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

    if not path.exists():
        raise FileNotFoundError(path)

    data = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    words = []

    for item in data["words"]:
        start = item.get("start")
        end = item.get("end")

        if start is None or end is None:
            continue

        words.append(
            {
                "start": float(start),
                "end": float(end),
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

    if not words:
        return sentences

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

            candidates.append(
                {
                    "sentence_count": size,
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
                        for item in group
                    ),
                }
            )

    return candidates

def make_word_candidates(
    words: list[dict],
) -> list[dict]:
    candidates = []

    for size in WORD_WINDOW_SIZES:
        if len(words) < size:
            continue

        start_indices = list(
            range(
                0,
                len(words) - size + 1,
                WORD_WINDOW_STRIDE,
            )
        )

        final_start = (
            len(words)
            - size
        )

        if final_start not in start_indices:
            start_indices.append(
                final_start
            )

        for start_index in start_indices:
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
                    "sentence_count": 0,
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
                float(
                    candidate["start"]
                ),
                3,
            ),
            round(
                float(
                    candidate["end"]
                ),
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


class PairDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer,
    ) -> None:
        self.df = dataframe.reset_index(
            drop=True
        )

        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(
        self,
        index: int,
    ):
        row = self.df.iloc[index]

        encoded = self.tokenizer(
            str(row["question"]),
            str(row["candidate_text"]),
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
        )

        encoded["labels"] = float(
            row["target_tiou"]
        )

        return encoded


def make_collator(tokenizer):
    def collate(batch):
        labels = torch.tensor(
            [
                item.pop("labels")
                for item in batch
            ],
            dtype=torch.float32,
        )

        encoded = tokenizer.pad(
            batch,
            padding=True,
            return_tensors="pt",
        )

        encoded["labels"] = labels

        return encoded

    return collate


def build_candidate_dataframe(
    positives: pd.DataFrame,
) -> pd.DataFrame:
    cache = {}
    rows = []

    for row_number, (_, row) in enumerate(
        positives.iterrows(),
        start=1,
    ):
        transcript_id = row[
            "transcript_id"
        ]

        stem = normalize_transcript_id(
            transcript_id
        )

        if stem not in cache:
            words = load_words(
                transcript_id
            )

            sentences = make_sentences(
                words
            )

            sentence_spans = make_candidates(
                sentences
            )

            word_spans = (
                make_word_candidates(
                    words
                )
            )

            cache[stem] = (
                deduplicate_candidates(
                    sentence_spans
                    + word_spans
                )
            )

        gold_start = float(
            row["evidence_start"]
        )

        gold_end = float(
            row["evidence_end"]
        )

        for candidate_index, candidate in enumerate(
            cache[stem]
        ):
            tiou = temporal_iou(
                candidate["start"],
                candidate["end"],
                gold_start,
                gold_end,
            )

            rows.append(
                {
                    "question_id": (
                        row["question_id"]
                    ),
                    "transcript_id": (
                        transcript_id
                    ),
                    "question": str(
                        row["question"]
                    ),
                    "candidate_index": (
                        candidate_index
                    ),
                    "candidate_start": (
                        candidate["start"]
                    ),
                    "candidate_kind": (
                        candidate[
                            "candidate_kind"
                        ]
                    ),
                    "candidate_end": (
                        candidate["end"]
                    ),
                    "candidate_text": (
                        candidate["text"]
                    ),
                    "sentence_count": (
                        candidate[
                            "sentence_count"
                        ]
                    ),
                    "target_tiou": tiou,
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
                "candidates built"
            )

    return pd.DataFrame(rows)


def initial_scores(
    model,
    tokenizer,
    dataframe: pd.DataFrame,
    device: str,
) -> np.ndarray:
    scores = []

    model.eval()

    for start in range(
        0,
        len(dataframe),
        EVAL_BATCH_SIZE,
    ):
        batch = dataframe.iloc[
            start:
            start + EVAL_BATCH_SIZE
        ]

        encoded = tokenizer(
            batch["question"].astype(
                str
            ).tolist(),
            batch[
                "candidate_text"
            ].astype(str).tolist(),
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        encoded = {
            key: value.to(device)
            for key, value
            in encoded.items()
        }

        with torch.inference_mode():
            logits = model(
                **encoded
            ).logits.reshape(-1)

        scores.extend(
            logits.detach()
            .cpu()
            .float()
            .numpy()
            .tolist()
        )

    return np.asarray(
        scores,
        dtype=np.float32,
    )


def sample_training_candidates(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    selected = []

    rng = np.random.default_rng(
        SEED
    )

    for _, group in dataframe.groupby(
        "question_id",
        sort=False,
    ):
        positive = group[
            group["target_tiou"]
            >= POSITIVE_TIOU_THRESHOLD
        ].sort_values(
            "target_tiou",
            ascending=False,
        ).head(
            MAX_POSITIVE_CANDIDATES
        )

        negative = group[
            group["target_tiou"]
            < POSITIVE_TIOU_THRESHOLD
        ]

        hard = negative.sort_values(
            "base_score",
            ascending=False,
        ).head(
            MAX_HARD_NEGATIVES
        )

        remaining = negative.drop(
            index=hard.index,
            errors="ignore",
        )

        if len(remaining) > 0:
            random_count = min(
                MAX_RANDOM_NEGATIVES,
                len(remaining),
            )

            random_indices = rng.choice(
                remaining.index.to_numpy(),
                size=random_count,
                replace=False,
            )

            random_negative = (
                remaining.loc[
                    random_indices
                ]
            )
        else:
            random_negative = (
                remaining
            )

        question_rows = pd.concat(
            [
                positive,
                hard,
                random_negative,
            ],
            axis=0,
        ).drop_duplicates(
            subset=[
                "question_id",
                "candidate_index",
            ]
        )

        selected.append(
            question_rows
        )

    return pd.concat(
        selected,
        ignore_index=True,
    )


def train_one_fold(
    train_df: pd.DataFrame,
    device: str,
    fold: int,
):
    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_ID
        )
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            MODEL_ID,
            num_labels=1,
            ignore_mismatched_sizes=True,
        )
        .to(device)
    )

    # Preserve the useful pretrained ranking signal before
    # selecting hard negatives.
    train_df = train_df.copy()

    train_df[
        "base_score"
    ] = initial_scores(
        model,
        tokenizer,
        train_df,
        device,
    )

    sampled = (
        sample_training_candidates(
            train_df
        )
    )

    print(
        f"Fold {fold}: "
        f"{len(train_df)} candidates -> "
        f"{len(sampled)} training pairs"
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
        collate_fn=make_collator(
            tokenizer
        ),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
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

    def lr_factor(step: int) -> float:
        if step < warmup_steps:
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

    global_step = 0

    for epoch in range(
        1,
        EPOCHS + 1,
    ):
        epoch_loss = 0.0
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
            ).logits.reshape(-1)

            loss = torch.nn.functional.mse_loss(
                torch.sigmoid(
                    logits
                ),
                labels,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            optimizer.step()
            scheduler.step()

            epoch_loss += float(
                loss.item()
            )

            batches += 1
            global_step += 1

        print(
            f"  epoch {epoch}/{EPOCHS} "
            f"loss="
            f"{epoch_loss / max(1, batches):.6f}"
        )

    return (
        model.eval(),
        tokenizer,
    )


def score_candidates(
    model,
    tokenizer,
    dataframe: pd.DataFrame,
    device: str,
) -> np.ndarray:
    raw = initial_scores(
        model,
        tokenizer,
        dataframe,
        device,
    )

    return (
        1.0
        / (
            1.0
            + np.exp(
                -raw
            )
        )
    )


def main() -> None:
    seed_everything(
        SEED
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    if device != "cuda":
        raise RuntimeError(
            "M030 is intended for "
            "the Windows GTX 1060."
        )

    questions = pd.read_csv(
        QUESTIONS
    )

    positives = questions[
        questions["question_type"]
        == "positive"
    ].copy()

    if len(positives) != 195:
        raise RuntimeError(
            f"Expected 195 positives; "
            f"got {len(positives)}"
        )

    print("=" * 78)
    print(
        "Nordic AI Cup 2026 — "
        "M038 Fine-Tuned Hybrid MiniLM-L12 Cross-Encoder"
    )
    print("=" * 78)
    print(f"Base model: {MODEL_ID}")
    print(f"Device:     {device}")
    print()

    candidate_df = (
        build_candidate_dataframe(
            positives
        )
    )

    question_df = (
        positives[
            [
                "question_id",
                "transcript_id",
            ]
        ]
        .reset_index(
            drop=True
        )
    )

    groups = question_df[
        "transcript_id"
    ].astype(str)

    splitter = GroupKFold(
        n_splits=N_SPLITS
    )

    oof_scores: dict[int, float] = {}

    wall_start = time.perf_counter()

    for fold, (
        train_q,
        test_q,
    ) in enumerate(
        splitter.split(
            question_df,
            groups=groups,
        ),
        start=1,
    ):
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

        train_df = candidate_df[
            candidate_df[
                "question_id"
            ].isin(train_ids)
        ].copy()

        test_df = candidate_df[
            candidate_df[
                "question_id"
            ].isin(test_ids)
        ].copy()

        print()
        print("=" * 78)
        print(
            f"FOLD {fold}/{N_SPLITS}"
        )
        print("=" * 78)
        print(
            f"Train questions: "
            f"{len(train_ids)}"
        )
        print(
            f"Test questions:  "
            f"{len(test_ids)}"
        )

        model, tokenizer = (
            train_one_fold(
                train_df,
                device,
                fold,
            )
        )

        fold_scores = (
            score_candidates(
                model,
                tokenizer,
                test_df,
                device,
            )
        )

        for dataframe_index, score in zip(
            test_df.index,
            fold_scores,
        ):
            oof_scores[
                int(dataframe_index)
            ] = float(score)

        del model
        del tokenizer

        torch.cuda.empty_cache()

    if len(oof_scores) != len(
        candidate_df
    ):
        raise RuntimeError(
            "Not every held-out candidate "
            "received an OOF score."
        )

    candidate_df[
        "oof_score"
    ] = [
        oof_scores[
            int(index)
        ]
        for index
        in candidate_df.index
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
            "oof_score"
        ].idxmax()

        selected = candidate_df.loc[
            best_index
        ].copy()

        selected[
            "candidate_oracle_tiou"
        ] = float(
            group[
                "target_tiou"
            ].max()
        )

        selected_rows.append(
            selected
        )

    result_df = pd.DataFrame(
        selected_rows
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
        - wall_start
    )

    print()
    print("=" * 78)
    print("M030 OOF RESULT")
    print("=" * 78)

    print(
        f"Questions:       "
        f"{len(result_df)}"
    )

    print()
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
    print(
        f"Oracle mean:     "
        f"{oracle.mean():.4f}"
    )

    print(
        f"Wall time:       "
        f"{elapsed:.1f} s"
    )

    print()
    print("REFERENCE")
    print("-" * 78)
    print(
    "M030 sentence L6:        0.5089"
    )

    print(
        "M036 hybrid L6:          0.5144"
    )

    print(
        "M035 compact oracle:     0.8447"
    )

    print()
    print(
        f"Detailed results: {OUTPUT}"
    )


if __name__ == "__main__":
    main()