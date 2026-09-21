from __future__ import annotations

import base64
import json
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
NLI_MODEL_ID = "cross-encoder/nli-deberta-v3-base"

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



# ============================================================================
# M067_CLASSIFICATION_PATCH
# M055 DeBERTa-v3-base runtime-signal classifier.
# Evidence selection remains the original M049/M047 deployment selector.
# ============================================================================

M055_CLASSIFIER_PATH = (
    Path(__file__).resolve().parents[2]
    / "artifacts"
    / "models"
    / "meta_classifier_m055.joblib"
)


# ============================================================================
# M071_M070_FUSION_PATCH
# M067 M055 classifier + task-specific M070 NLI fusion.
# Evidence selection remains unchanged.
# ============================================================================

M070_MODEL_PATH = (
    Path(__file__).resolve().parents[2]
    / "artifacts"
    / "models"
    / "m070_task_specific_nli"
)

M070_METADATA_PATH = (
    Path(__file__).resolve().parents[2]
    / "artifacts"
    / "models"
    / "m070_task_specific_nli_metadata.json"
)

M070_TOP_K = 12

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

        if not M055_CLASSIFIER_PATH.exists():
            raise FileNotFoundError(
                "M055 classifier not found: "
                f"{M055_CLASSIFIER_PATH}"
            )

        print(
            "[medical] loading "
            "M055 runtime-signal classifier..."
        )

        m055_bundle = joblib.load(
            M055_CLASSIFIER_PATH
        )

        self.m055_classifier = (
            m055_bundle["model"]
        )

        self.m055_feature_columns = list(
            m055_bundle[
                "feature_columns"
            ]
        )

        self.m055_threshold = float(
            m055_bundle[
                "threshold"
            ]
        )

        if not M070_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"M070 model not found: "
                f"{M070_MODEL_PATH}"
            )

        if not M070_METADATA_PATH.exists():
            raise FileNotFoundError(
                f"M070 metadata not found: "
                f"{M070_METADATA_PATH}"
            )

        print(
            "[medical] loading "
            "M070 task-specific NLI..."
        )

        self.m070_tokenizer = (
            AutoTokenizer.from_pretrained(
                M070_MODEL_PATH
            )
        )

        self.m070_model = (
            AutoModelForSequenceClassification
            .from_pretrained(
                M070_MODEL_PATH
            )
            .to(self.device)
            .eval()
        )

        with open(
            M070_METADATA_PATH,
            "r",
            encoding="utf-8",
        ) as handle:
            m070_metadata = json.load(
                handle
            )

        self.m070_entailment_index = int(
            m070_metadata[
                "entailment_index"
            ]
        )

        self.m070_contradiction_index = int(
            m070_metadata[
                "contradiction_index"
            ]
        )

        self.m070_neutral_index = int(
            m070_metadata[
                "neutral_index"
            ]
        )

        self.m070_m055_weight = float(
            m070_metadata[
                "m055_weight"
            ]
        )

        self.m070_threshold = float(
            m070_metadata[
                "fusion_threshold"
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

    def _m055_evidence_features(
        self,
        question: str,
        words: list[
            dict[str, Any]
        ],
    ) -> dict[str, float]:
        """
        Reproduce the four inference-safe M055 evidence features:

        evidence_max_score
        evidence_mean_score
        evidence_std_score
        evidence_top_gap

        Scores come from the existing deployed M049/M047 hybrid
        evidence ranker. This intentionally does not change the
        evidence span returned by the endpoint.
        """

        sentences = make_sentences(
            words
        )

        sentence_candidates = (
            make_sentence_candidates(
                sentences,
                HYBRID_MAX_SENTENCES,
            )
        )

        for candidate in sentence_candidates:
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
            return {
                "evidence_max_score": 0.0,
                "evidence_mean_score": 0.0,
                "evidence_std_score": 0.0,
                "evidence_top_gap": 0.0,
            }

        texts = [
            str(
                candidate["text"]
            )
            for candidate
            in candidates
        ]

        scores: list[float] = []

        for start in range(
            0,
            len(texts),
            EVIDENCE_BATCH_SIZE,
        ):
            batch_texts = texts[
                start:
                start + EVIDENCE_BATCH_SIZE
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

        ranked = np.sort(
            score_array
        )[::-1]

        if len(ranked) >= 2:
            top_gap = float(
                ranked[0]
                - ranked[1]
            )
        else:
            top_gap = 0.0

        if len(score_array) >= 2:
            # pandas GroupBy.std uses ddof=1,
            # so reproduce it exactly.
            std_score = float(
                np.std(
                    score_array,
                    ddof=1,
                )
            )
        else:
            std_score = 0.0

        return {
            "evidence_max_score":
                float(
                    np.max(
                        score_array
                    )
                ),

            "evidence_mean_score":
                float(
                    np.mean(
                        score_array
                    )
                ),

            "evidence_std_score":
                std_score,

            "evidence_top_gap":
                top_gap,
        }

    def _classify_m055(
        self,
        *,
        question: str,
        question_position: int,
        segments: list[
            dict[str, Any]
        ],
        ranked_segments: list[int],
        evidence_features: dict[
            str,
            float
        ],
    ) -> dict[str, Any]:
        """
        Exact runtime analogue of M055:
        DeBERTa-v3-base segment NLI + evidence score distribution
        + question position.
        """

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
                "for M055 classification."
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

        feature_values = {
            "question_position":
                float(
                    question_position
                ),

            "max_segment_ratio":
                float(
                    np.max(
                        ratios
                    )
                ),

            "max_segment_entailment":
                float(
                    np.max(
                        entailment
                    )
                ),

            "max_segment_margin":
                float(
                    np.max(
                        margins
                    )
                ),

            "max_segment_vs_other":
                float(
                    np.max(
                        vs_other
                    )
                ),

            "selected_entailment":
                float(
                    entailment[
                        ratio_index
                    ]
                ),

            "selected_contradiction":
                float(
                    contradiction[
                        ratio_index
                    ]
                ),

            "selected_neutral":
                float(
                    neutral[
                        ratio_index
                    ]
                ),

            **evidence_features,
        }

        missing = (
            set(
                self.m055_feature_columns
            )
            - set(
                feature_values
            )
        )

        if missing:
            raise RuntimeError(
                "Missing M055 features: "
                f"{sorted(missing)}"
            )

        row = pd.DataFrame(
            [
                {
                    name:
                        feature_values[
                            name
                        ]
                    for name
                    in self.m055_feature_columns
                }
            ],
            columns=(
                self.m055_feature_columns
            ),
        )

        probability = float(
            self.m055_classifier
            .predict_proba(
                row
            )[0, 1]
        )

        predicted_yes = (
            probability
            >= self.m055_threshold
        )

        return {
            "m055_predicted_yes":
                bool(
                    predicted_yes
                ),

            "m055_probability":
                probability,

            **feature_values,
        }

    def _m070_max_score(
        self,
        question: str,
        words: list[
            dict[str, Any]
        ],
    ) -> float:
        """
        Runtime analogue of M070 OOF inference:

        1. Build the same hybrid evidence candidates.
        2. Rank them with the existing M047/M049 evidence ranker.
        3. Keep top 12.
        4. Score those with task-specific M070 NLI.
        5. Return max entailment-vs-non-entailment probability.
        """

        sentences = make_sentences(
            words
        )

        sentence_candidates = (
            make_sentence_candidates(
                sentences,
                HYBRID_MAX_SENTENCES,
            )
        )

        for candidate in sentence_candidates:
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
            return 0.0

        texts = [
            str(
                candidate["text"]
            )
            for candidate
            in candidates
        ]

        # ------------------------------------------------------------
        # First-stage M047/M049 score.
        # ------------------------------------------------------------

        evidence_scores: list[float] = []

        for start in range(
            0,
            len(texts),
            EVIDENCE_BATCH_SIZE,
        ):
            batch_texts = texts[
                start:
                start + EVIDENCE_BATCH_SIZE
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

            evidence_scores.extend(
                batch_scores
                .detach()
                .cpu()
                .float()
                .numpy()
                .tolist()
            )

        evidence_scores = np.asarray(
            evidence_scores,
            dtype=float,
        )

        order = (
            np.argsort(
                evidence_scores
            )[::-1]
        )

        top_indices = order[
            :min(
                M070_TOP_K,
                len(order),
            )
        ]

        top_texts = [
            texts[
                int(index)
            ]
            for index
            in top_indices
        ]

        # ------------------------------------------------------------
        # Task-specific M070 NLI.
        # ------------------------------------------------------------

        scores: list[float] = []

        for start in range(
            0,
            len(top_texts),
            NLI_TOP_K,
        ):
            batch_texts = top_texts[
                start:
                start + NLI_TOP_K
            ]

            encoded = (
                self.m070_tokenizer(
                    [
                        question
                        for _ in batch_texts
                    ],
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=192,
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
                    self.m070_model(
                        **encoded
                    ).logits
                )

                entailment = logits[
                    :,
                    self.m070_entailment_index,
                ]

                others = torch.stack(
                    [
                        logits[
                            :,
                            self.m070_contradiction_index,
                        ],
                        logits[
                            :,
                            self.m070_neutral_index,
                        ],
                    ],
                    dim=1,
                )

                non_entailment = (
                    torch.logsumexp(
                        others,
                        dim=1,
                    )
                )

                binary_logit = (
                    entailment
                    - non_entailment
                )

                probabilities = (
                    torch.sigmoid(
                        binary_logit
                    )
                )

            scores.extend(
                probabilities
                .detach()
                .cpu()
                .float()
                .numpy()
                .tolist()
            )

        if not scores:
            return 0.0

        return float(
            np.max(
                np.asarray(
                    scores,
                    dtype=float,
                )
            )
        )

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

        for question_position, question in enumerate(questions, start=1):
            try:
                ranked_segments = (
                    self
                    ._semantic_segment_ranking(
                        question,
                        segments,
                    )
                )

                evidence_features = (
                    self._m055_evidence_features(
                        question,
                        words,
                    )
                )

                classification = (
                    self._classify_m055(
                        question=question,
                        question_position=(
                            question_position
                        ),
                        segments=segments,
                        ranked_segments=(
                            ranked_segments
                        ),
                        evidence_features=(
                            evidence_features
                        ),
                    )
                )

                m055_probability = float(
                    classification[
                        "m055_probability"
                    ]
                )

                m070_probability = (
                    self._m070_max_score(
                        question,
                        words,
                    )
                )

                fusion_probability = (
                    self.m070_m055_weight
                    * m055_probability
                    + (
                        1.0
                        - self.m070_m055_weight
                    )
                    * m070_probability
                )

                predicted_yes = bool(
                    fusion_probability
                    >= self.m070_threshold
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

                (
                    start,
                    end,
                    _evidence_score,
                ) = self._select_evidence(
                    question,
                    segments,
                    words,
                    ranked_segments,
                )

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
