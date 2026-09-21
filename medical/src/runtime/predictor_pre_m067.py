from __future__ import annotations

import base64
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sentence_transformers import CrossEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForSequenceClassification, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[3]

SENTENCE_RANKER_PATH = (
    REPO_ROOT
    / "medical"
    / "artifacts"
    / "models"
    / "sentence_ranker.joblib"
)

ASR_MODEL_ID = "distil-large-v3"
RETRIEVER_MODEL_ID = "cross-encoder/ms-marco-MiniLM-L6-v2"
NLI_MODEL_ID = "cross-encoder/nli-deberta-v3-small"

ASR_COMPUTE_TYPE = "int8_float32"
ASR_BEAM_SIZE = 1

LEXICAL_TOP_K = 8
NLI_TOP_K = 5

CLASSIFICATION_THRESHOLD = 0.000585

META_CLASSIFIER_PATH = (
    REPO_ROOT
    / "medical"
    / "artifacts"
    / "models"
    / "meta_classifier_m042.joblib"
)

EVIDENCE_RANKER_PATH = (
    REPO_ROOT
    / "medical"
    / "artifacts"
    / "models"
    / "hybrid_evidence_ranker_m049"
)

# Median of the M042 fold-specific thresholds.
META_CLASSIFICATION_THRESHOLD = 0.384878

# M049 evidence-confidence arbitration threshold.
EVIDENCE_GATE_THRESHOLD = 0.369304

HYBRID_MAX_SENTENCES = 3

HYBRID_WORD_WINDOW_SIZES = (
    6,
    10,
)

HYBRID_WORD_WINDOW_STRIDE = 3

EVIDENCE_MAX_LENGTH = 192
EVIDENCE_BATCH_SIZE = 128


def configure_windows_cuda_dlls() -> None:
    """
    Add the local faster-whisper CUDA runtime directory before importing
    CTranslate2/faster-whisper.

    This keeps the known-working Pascal-compatible PyTorch installation
    untouched.
    """

    if os.name != "nt":
        return

    root = (
        REPO_ROOT
        / "tools"
        / "faster-whisper-cuda"
    )

    if not root.exists():
        return

    dll_matches = list(
        root.rglob("cublas64_12.dll")
    )

    if not dll_matches:
        return

    dll_dir = dll_matches[0].parent

    os.environ["PATH"] = (
        str(dll_dir)
        + os.pathsep
        + os.environ.get("PATH", "")
    )

    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(
            str(dll_dir)
        )


configure_windows_cuda_dlls()

from faster_whisper import WhisperModel  # noqa: E402

def question_to_claim(question: str) -> str:
    text = question.strip()

    if text.endswith("?"):
        text = text[:-1].strip()

    prefix = ""

    if "," in text:
        first_part, remainder = text.split(
            ",",
            1,
        )

        remainder = remainder.strip()

        auxiliary_starters = (
            "does ",
            "do ",
            "did ",
            "is ",
            "are ",
            "was ",
            "were ",
            "has ",
            "have ",
            "will ",
            "should ",
            "can ",
        )

        if remainder.lower().startswith(
            auxiliary_starters
        ):
            prefix = (
                first_part.strip()
                + ", "
            )

            text = remainder

    words = text.split()

    if len(words) < 2:
        return (
            question.rstrip("?")
            + "."
        )

    auxiliary = words[0].lower()

    if (
        auxiliary
        in {
            "is",
            "are",
            "was",
            "were",
        }
        and words[1].lower()
        == "there"
    ):
        remainder = " ".join(
            words[2:]
        )

        claim = (
            f"There {words[0].lower()} "
            f"{remainder}"
        )

        return prefix + claim + "."

    if auxiliary in {
        "does",
        "do",
        "did",
        "is",
        "are",
        "was",
        "were",
        "has",
        "have",
        "will",
        "should",
        "can",
    }:
        remainder = words[1:]

        if (
            len(remainder) >= 2
            and remainder[0].lower()
            == "the"
        ):
            subject_length = 2

            if len(remainder) >= 3:
                three_token_subjects = {
                    "the blood pressure",
                    "the heart rate",
                    "the treatment plan",
                    "the kidney function",
                    "the liver function",
                    "the diabetes medication",
                    "the current medication",
                    "the prescribed medication",
                }

                candidate = " ".join(
                    token.lower()
                    for token
                    in remainder[:3]
                )

                if (
                    candidate
                    in three_token_subjects
                ):
                    subject_length = 3

            subject = " ".join(
                remainder[
                    :subject_length
                ]
            )

            predicate = " ".join(
                remainder[
                    subject_length:
                ]
            )

        elif (
            len(remainder) >= 2
            and remainder[0].lower()
            in {
                "both",
                "any",
                "all",
                "either",
            }
        ):
            subject = " ".join(
                remainder[:2]
            )

            predicate = " ".join(
                remainder[2:]
            )

        else:
            subject = remainder[0]

            predicate = " ".join(
                remainder[1:]
            )

        claim = (
            f"{subject} "
            f"{words[0].lower()} "
            f"{predicate}"
        ).strip()

        claim = (
            claim[0].upper()
            + claim[1:]
        )

        return prefix + claim + "."

    claim = text

    if claim:
        claim = (
            claim[0].upper()
            + claim[1:]
        )

    return prefix + claim + "."


def lexical_scores(
    query: str,
    texts: list[str],
) -> np.ndarray:
    if not texts:
        return np.zeros(
            0,
            dtype=float,
        )

    documents = texts + [query]

    word_vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        sublinear_tf=True,
        token_pattern=r"(?u)\b\w+\b",
    )

    word_matrix = (
        word_vectorizer
        .fit_transform(
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
        char_vectorizer
        .fit_transform(
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
    words: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not words:
        return []

    sentences = []
    start_index = 0

    for index, word in enumerate(
        words
    ):
        if not is_sentence_end(
            str(word["word"])
        ):
            continue

        text = "".join(
            str(item["word"])
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
                        words[index][
                            "end"
                        ]
                    ),
                    "text": text,
                }
            )

        start_index = index + 1

    if start_index < len(words):
        text = "".join(
            str(item["word"])
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
    sentences: list[dict[str, Any]],
    max_sentences: int,
) -> list[dict[str, Any]]:
    candidates = []

    for size in range(
        1,
        max_sentences + 1,
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

            text = " ".join(
                str(item["text"])
                for item in group
            )

            candidates.append(
                {
                    "sentence_count": size,
                    "start": float(
                        group[0]["start"]
                    ),
                    "end": float(
                        group[-1]["end"]
                    ),
                    "text": text,
                    "word_count": len(
                        text.split()
                    ),
                }
            )

    return candidates



META_NEGATION_TERMS = {
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

META_NUMBER_PATTERN = re.compile(
    r"\b\d+(?:\.\d+)?\b"
)

META_UNIT_PATTERN = re.compile(
    r"\b(?:"
    r"mg|mcg|g|kg|ml|l|"
    r"mmol|mmol/l|mg/dl|"
    r"cm|mm|%|bpm"
    r")\b",
    flags=re.IGNORECASE,
)


def make_word_candidates(
    words: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    candidates = []

    for size in HYBRID_WORD_WINDOW_SIZES:
        if len(words) < size:
            continue

        start_indices = list(
            range(
                0,
                len(words) - size + 1,
                HYBRID_WORD_WINDOW_STRIDE,
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
                str(item["word"])
                for item in words[
                    start_index:
                    end_index + 1
                ]
            ).strip()

            if not text:
                continue

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
                    "word_count": size,
                }
            )

    return candidates


def deduplicate_hybrid_candidates(
    candidates: list[
        dict[str, Any]
    ],
) -> list[dict[str, Any]]:
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


def make_meta_question_features(
    question: str,
) -> dict[str, float]:
    text = str(
        question
    )

    lower = (
        text.lower()
    )

    words = text.split()

    features = {
        "question_word_count": float(
            len(words)
        ),
        "question_char_count": float(
            len(text)
        ),
        "has_number": float(
            bool(
                META_NUMBER_PATTERN.search(
                    text
                )
            )
        ),
        "has_unit": float(
            bool(
                META_UNIT_PATTERN.search(
                    text
                )
            )
        ),
        "has_negation": float(
            any(
                re.search(
                    rf"\b"
                    rf"{re.escape(term)}"
                    rf"\b",
                    lower,
                )
                is not None
                for term
                in META_NEGATION_TERMS
            )
        ),
    }

    for auxiliary in (
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
    ):
        features[
            f"starts_{auxiliary}"
        ] = float(
            lower.startswith(
                auxiliary + " "
            )
        )

    return features

def get_label_indices(
    model,
) -> dict[str, int]:
    mapping = {
        int(key): str(value).lower()
        for key, value
        in model.config.id2label.items()
    }

    result: dict[str, int] = {}

    for index, label in mapping.items():
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


class MedicalPredictor:
    def __init__(self) -> None:
        if not SENTENCE_RANKER_PATH.exists():
            raise FileNotFoundError(
                "Deployment sentence ranker "
                f"not found: "
                f"{SENTENCE_RANKER_PATH}"
            )

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. "
                "Medical runtime expects "
                "the GTX 1060 CUDA environment."
            )

        self.device = "cuda"

        print(
            "[medical] loading "
            "faster-whisper..."
        )

        self.asr = WhisperModel(
            ASR_MODEL_ID,
            device="cuda",
            compute_type=(
                ASR_COMPUTE_TYPE
            ),
        )

        print(
            "[medical] loading "
            "MS MARCO reranker..."
        )

        self.retriever = CrossEncoder(
            RETRIEVER_MODEL_ID,
            device=self.device,
            max_length=256,
        )

        print(
            "[medical] loading NLI..."
        )

        self.nli_tokenizer = (
            AutoTokenizer.from_pretrained(
                NLI_MODEL_ID
            )
        )

        self.nli_model = (
            AutoModelForSequenceClassification
            .from_pretrained(
                NLI_MODEL_ID
            )
            .to(self.device)
            .eval()
        )

        self.nli_labels = (
            get_label_indices(
                self.nli_model
            )
        )

        if not META_CLASSIFIER_PATH.exists():
            raise FileNotFoundError(
                "M042 meta-classifier "
                f"not found: "
                f"{META_CLASSIFIER_PATH}"
            )

        if not EVIDENCE_RANKER_PATH.exists():
            raise FileNotFoundError(
                "M049 evidence model "
                f"not found: "
                f"{EVIDENCE_RANKER_PATH}"
            )

        print(
            "[medical] loading "
            "M042 meta-classifier..."
        )

        meta_bundle = joblib.load(
            META_CLASSIFIER_PATH
        )

        self.meta_classifier = (
            meta_bundle["model"]
        )

        self.meta_feature_columns = list(
            meta_bundle[
                "feature_columns"
            ]
        )

        print(
            "[medical] loading "
            "M049 hybrid evidence ranker..."
        )

        self.evidence_tokenizer = (
            AutoTokenizer.from_pretrained(
                EVIDENCE_RANKER_PATH
            )
        )

        self.evidence_model = (
            AutoModelForSequenceClassification
            .from_pretrained(
                EVIDENCE_RANKER_PATH
            )
            .to(self.device)
            .eval()
        )

        print(
            "[medical] runtime ready"
        )

    def _transcribe(
        self,
        audio_bytes: bytes,
        suffix: str,
    ) -> tuple[
        list[dict[str, Any]],
        list[dict[str, Any]],
    ]:
        temp_path: str | None = None

        try:
            with tempfile.NamedTemporaryFile(
                suffix=suffix,
                delete=False,
            ) as handle:
                handle.write(
                    audio_bytes
                )

                temp_path = (
                    handle.name
                )

            segment_generator, _ = (
                self.asr.transcribe(
                    temp_path,
                    language="en",
                    beam_size=(
                        ASR_BEAM_SIZE
                    ),
                    word_timestamps=True,
                )
            )

            raw_segments = list(
                segment_generator
            )

            segments = []
            words = []

            for segment in raw_segments:
                text = str(
                    segment.text
                ).strip()

                if text:
                    segments.append(
                        {
                            "start": float(
                                segment.start
                            ),
                            "end": float(
                                segment.end
                            ),
                            "text": text,
                        }
                    )

                if not segment.words:
                    continue

                for word in segment.words:
                    if (
                        word.start is None
                        or word.end is None
                    ):
                        continue

                    words.append(
                        {
                            "start": float(
                                word.start
                            ),
                            "end": float(
                                word.end
                            ),
                            "word": str(
                                word.word
                            ),
                        }
                    )

            if not segments:
                raise RuntimeError(
                    "ASR returned no segments."
                )

            if not words:
                raise RuntimeError(
                    "ASR returned no "
                    "timestamped words."
                )

            return (
                segments,
                words,
            )

        finally:
            if (
                temp_path is not None
                and os.path.exists(
                    temp_path
                )
            ):
                try:
                    os.remove(
                        temp_path
                    )
                except OSError:
                    pass

    def _semantic_segment_ranking(
        self,
        question: str,
        segments: list[
            dict[str, Any]
        ],
    ) -> list[int]:
        texts = [
            str(segment["text"])
            for segment in segments
        ]

        lexical = lexical_scores(
            question,
            texts,
        )

        lexical_indices = (
            np.argsort(
                lexical
            )[::-1][
                :min(
                    LEXICAL_TOP_K,
                    len(segments),
                )
            ]
        )

        pairs = [
            (
                question,
                texts[int(index)],
            )
            for index
            in lexical_indices
        ]

        semantic = np.asarray(
            self.retriever.predict(
                pairs,
                batch_size=32,
                show_progress_bar=False,
            )
        ).reshape(-1)

        order = np.argsort(
            semantic
        )[::-1]

        return [
            int(
                lexical_indices[
                    position
                ]
            )
            for position in order
        ]

    def _classify(
        self,
        question: str,
        segments: list[
            dict[str, Any]
        ],
        ranked_segments: list[int],
    ) -> dict[str, Any]:
        claim = question_to_claim(
            question
        )

        selected_indices = (
            ranked_segments[
                :min(
                    NLI_TOP_K,
                    len(
                        ranked_segments
                    ),
                )
            ]
        )

        if not selected_indices:
            raise RuntimeError(
                "No segments available "
                "for NLI classification."
            )

        premises = [
            str(
                segments[index][
                    "text"
                ]
            )
            for index
            in selected_indices
        ]

        hypotheses = [
            claim
            for _ in premises
        ]

        encoded = (
            self.nli_tokenizer(
                premises,
                hypotheses,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
        )

        encoded = {
            key: value.to(
                self.device
            )
            for key, value
            in encoded.items()
        }

        with torch.inference_mode():
            logits = (
                self.nli_model(
                    **encoded
                ).logits
            )

        probabilities = (
            torch.softmax(
                logits,
                dim=-1,
            )
            .detach()
            .cpu()
            .numpy()
        )

        entailment = probabilities[
            :,
            self.nli_labels[
                "entailment"
            ],
        ]

        contradiction = probabilities[
            :,
            self.nli_labels[
                "contradiction"
            ],
        ]

        neutral = probabilities[
            :,
            self.nli_labels[
                "neutral"
            ],
        ]

        ratios = (
            entailment
            / (
                entailment
                + contradiction
                + 1e-12
            )
        )

        margins = (
            entailment
            - contradiction
        )

        vs_other = (
            entailment
            - np.maximum(
                contradiction,
                neutral,
            )
        )

        ratio_index = int(
            np.argmax(
                ratios
            )
        )

        max_segment_ratio = float(
            np.max(
                ratios
            )
        )

        max_segment_entailment = float(
            np.max(
                entailment
            )
        )

        max_segment_margin = float(
            np.max(
                margins
            )
        )

        max_segment_vs_other = float(
            np.max(
                vs_other
            )
        )

        feature_values = {
            "max_segment_ratio": (
                max_segment_ratio
            ),
            "max_segment_entailment": (
                max_segment_entailment
            ),
            "max_segment_margin": (
                max_segment_margin
            ),
            "max_segment_vs_other": (
                max_segment_vs_other
            ),
            "selected_entailment": float(
                entailment[
                    ratio_index
                ]
            ),
            "selected_contradiction": float(
                contradiction[
                    ratio_index
                ]
            ),
            "selected_neutral": float(
                neutral[
                    ratio_index
                ]
            ),
        }

        feature_values.update(
            make_meta_question_features(
                question
            )
        )

        meta_row = pd.DataFrame(
            [
                {
                    name: (
                        feature_values[
                            name
                        ]
                    )
                    for name
                    in self.meta_feature_columns
                }
            ],
            columns=(
                self.meta_feature_columns
            ),
        )

        meta_probability = float(
            self.meta_classifier
            .predict_proba(
                meta_row
            )[0, 1]
        )

        m018_predicted_yes = (
            max_segment_margin
            >= CLASSIFICATION_THRESHOLD
        )

        m042_predicted_yes = (
            meta_probability
            >= META_CLASSIFICATION_THRESHOLD
        )

        return {
            "m018_predicted_yes": bool(
                m018_predicted_yes
            ),
            "m042_predicted_yes": bool(
                m042_predicted_yes
            ),
            "m018_score": (
                max_segment_margin
            ),
            "m042_probability": (
                meta_probability
            ),
            **feature_values,
        }

    def _select_evidence(
        self,
        question: str,
        segments: list[
            dict[str, Any]
        ],
        words: list[
            dict[str, Any]
        ],
        ranked_segments: list[int],
    ) -> tuple[
        float,
        float,
        float,
    ]:
        sentences = make_sentences(
            words
        )

        sentence_candidates = (
            make_sentence_candidates(
                sentences,
                HYBRID_MAX_SENTENCES,
            )
        )

        for candidate in (
            sentence_candidates
        ):
            candidate[
                "candidate_kind"
            ] = (
                "sentence_"
                + str(
                    candidate[
                        "sentence_count"
                    ]
                )
            )

        word_candidates = (
            make_word_candidates(
                words
            )
        )

        candidates = (
            deduplicate_hybrid_candidates(
                sentence_candidates
                + word_candidates
            )
        )

        if not candidates:
            if not ranked_segments:
                raise RuntimeError(
                    "No evidence candidates."
                )

            anchor = segments[
                ranked_segments[0]
            ]

            return (
                float(
                    anchor["start"]
                ),
                float(
                    anchor["end"]
                ),
                0.0,
            )

        texts = [
            str(
                candidate["text"]
            )
            for candidate
            in candidates
        ]

        scores = []

        for start in range(
            0,
            len(candidates),
            EVIDENCE_BATCH_SIZE,
        ):
            batch_texts = texts[
                start:
                start
                + EVIDENCE_BATCH_SIZE
            ]

            encoded = (
                self.evidence_tokenizer(
                    [
                        question
                        for _ in batch_texts
                    ],
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=(
                        EVIDENCE_MAX_LENGTH
                    ),
                    return_tensors="pt",
                )
            )

            encoded = {
                key: value.to(
                    self.device
                )
                for key, value
                in encoded.items()
            }

            with torch.inference_mode():
                logits = (
                    self.evidence_model(
                        **encoded
                    )
                    .logits
                    .reshape(-1)
                )

                batch_scores = (
                    torch.sigmoid(
                        logits
                    )
                )

            scores.extend(
                batch_scores
                .detach()
                .cpu()
                .float()
                .numpy()
                .tolist()
            )

        score_array = np.asarray(
            scores,
            dtype=float,
        )

        best_index = int(
            np.argmax(
                score_array
            )
        )

        best = candidates[
            best_index
        ]

        return (
            float(
                best["start"]
            ),
            float(
                best["end"]
            ),
            float(
                score_array[
                    best_index
                ]
            ),
        )

    def predict_request(
        self,
        *,
        audio_base64: str,
        audio_filename: str,
        questions: list[str],
    ) -> dict[str, list[Any]]:
        count = len(questions)

        fallback = {
            "answers": [
                False
                for _ in range(count)
            ],
            "evidence_start": [
                None
                for _ in range(count)
            ],
            "evidence_end": [
                None
                for _ in range(count)
            ],
        }

        if count == 0:
            return fallback

        try:
            audio_bytes = (
                base64.b64decode(
                    audio_base64,
                    validate=True,
                )
            )

            suffix = (
                Path(
                    audio_filename
                    or "audio.mp3"
                ).suffix
                or ".mp3"
            )

            segments, words = (
                self._transcribe(
                    audio_bytes,
                    suffix,
                )
            )

        except Exception as exc:
            print(
                "[medical] ASR failure:",
                repr(exc),
                file=sys.stderr,
            )

            return fallback

        answers = []
        evidence_start = []
        evidence_end = []

        for question in questions:
            try:
                ranked_segments = (
                    self
                    ._semantic_segment_ranking(
                        question,
                        segments,
                    )
                )

                classification = (
                    self._classify(
                        question,
                        segments,
                        ranked_segments,
                    )
                )

                m018_yes = bool(
                    classification[
                        "m018_predicted_yes"
                    ]
                )

                m042_yes = bool(
                    classification[
                        "m042_predicted_yes"
                    ]
                )

                # Agreement on NO requires no
                # evidence-ranker inference.
                if (
                    not m018_yes
                    and not m042_yes
                ):
                    answers.append(
                        False
                    )

                    evidence_start.append(
                        None
                    )

                    evidence_end.append(
                        None
                    )

                    continue

                # Evidence is required when:
                # 1. both classifiers say YES, or
                # 2. classifiers disagree and M049
                #    needs evidence confidence.
                (
                    start,
                    end,
                    evidence_score,
                ) = self._select_evidence(
                    question,
                    segments,
                    words,
                    ranked_segments,
                )

                if m018_yes == m042_yes:
                    predicted_yes = (
                        m018_yes
                    )

                else:
                    if (
                        evidence_score
                        >= EVIDENCE_GATE_THRESHOLD
                    ):
                        predicted_yes = (
                            m042_yes
                        )
                    else:
                        predicted_yes = (
                            m018_yes
                        )

                if not predicted_yes:
                    answers.append(
                        False
                    )

                    evidence_start.append(
                        None
                    )

                    evidence_end.append(
                        None
                    )

                    continue

                if (
                    not np.isfinite(
                        start
                    )
                    or not np.isfinite(
                        end
                    )
                    or end <= start
                ):
                    raise RuntimeError(
                        "Invalid evidence span "
                        f"{start}-{end}"
                    )

                answers.append(
                    True
                )

                evidence_start.append(
                    round(
                        float(
                            start
                        ),
                        3,
                    )
                )

                evidence_end.append(
                    round(
                        float(
                            end
                        ),
                        3,
                    )
                )

            except Exception as exc:
                print(
                    "[medical] "
                    "question failure:",
                    repr(
                        exc
                    ),
                    file=sys.stderr,
                )

                answers.append(
                    False
                )

                evidence_start.append(
                    None
                )

                evidence_end.append(
                    None
                )

        return {
            "answers": answers,
            "evidence_start": (
                evidence_start
            ),
            "evidence_end": (
                evidence_end
            ),
        }


def decode_and_predict(
    predictor: MedicalPredictor,
    payload: dict[str, Any],
) -> dict[str, list[Any]]:
    questions = payload.get(
        "questions",
        []
    )

    if not isinstance(
        questions,
        list,
    ):
        questions = []

    questions = [
        str(question)
        for question in questions
    ]

    return predictor.predict_request(
        audio_base64=str(
            payload.get(
                "audio_base64",
                "",
            )
        ),
        audio_filename=str(
            payload.get(
                "audio_filename",
                "audio.mp3",
            )
        ),
        questions=questions,
    )