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

        print(
            "[medical] loading "
            "sentence ranker..."
        )

        bundle = joblib.load(
            SENTENCE_RANKER_PATH
        )

        self.sentence_ranker = (
            bundle["model"]
        )

        self.feature_columns = list(
            bundle[
                "feature_columns"
            ]
        )

        self.max_sentences = int(
            bundle["max_sentences"]
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
    ) -> tuple[
        bool,
        float,
    ]:
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

        margins = (
            entailment
            - contradiction
        )

        best_margin = float(
            margins.max()
        )

        predicted_yes = (
            best_margin
            >= CLASSIFICATION_THRESHOLD
        )

        return (
            predicted_yes,
            best_margin,
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
    ]:
        sentences = make_sentences(
            words
        )

        candidates = (
            make_sentence_candidates(
                sentences,
                self.max_sentences,
            )
        )

        if not candidates:
            anchor = segments[
                ranked_segments[0]
            ]

            return (
                float(anchor["start"]),
                float(anchor["end"]),
            )

        # M010 coarse anchor:
        # top MS-MARCO segment.
        anchor = segments[
            ranked_segments[0]
        ]

        anchor_start = float(
            anchor["start"]
        )

        anchor_end = float(
            anchor["end"]
        )

        anchor_mid = (
            anchor_start
            + anchor_end
        ) / 2.0

        candidate_texts = [
            str(candidate["text"])
            for candidate
            in candidates
        ]

        lexical = lexical_scores(
            question,
            candidate_texts,
        )

        pairs = [
            (
                question,
                text,
            )
            for text
            in candidate_texts
        ]

        semantic = np.asarray(
            self.retriever.predict(
                pairs,
                batch_size=64,
                show_progress_bar=False,
            )
        ).reshape(-1)

        feature_rows = []

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

            feature_values = {
                "sentence_count": float(
                    candidate[
                        "sentence_count"
                    ]
                ),
                "word_count": float(
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
                    temporal_iou(
                        candidate_start,
                        candidate_end,
                        anchor_start,
                        anchor_end,
                    )
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
                "contains_anchor_mid": float(
                    candidate_start
                    <= anchor_mid
                    <= candidate_end
                ),
            }

            feature_rows.append(
                [
                    feature_values[
                        name
                    ]
                    for name
                    in self.feature_columns
                ]
            )

        predicted_tiou = (
            self.sentence_ranker
            .predict(
                np.asarray(
                    feature_rows,
                    dtype=float,
                )
            )
        )

        best_index = int(
            np.argmax(
                predicted_tiou
            )
        )

        best = candidates[
            best_index
        ]

        return (
            float(best["start"]),
            float(best["end"]),
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

                predicted_yes, _ = (
                    self._classify(
                        question,
                        segments,
                        ranked_segments,
                    )
                )

                if not predicted_yes:
                    answers.append(False)
                    evidence_start.append(
                        None
                    )
                    evidence_end.append(
                        None
                    )
                    continue

                start, end = (
                    self._select_evidence(
                        question,
                        segments,
                        words,
                        ranked_segments,
                    )
                )

                if (
                    not np.isfinite(start)
                    or not np.isfinite(end)
                    or end <= start
                ):
                    raise RuntimeError(
                        "Invalid evidence span "
                        f"{start}-{end}"
                    )

                answers.append(True)

                evidence_start.append(
                    round(
                        float(start),
                        3,
                    )
                )

                evidence_end.append(
                    round(
                        float(end),
                        3,
                    )
                )

            except Exception as exc:
                print(
                    "[medical] question failure:",
                    repr(exc),
                    file=sys.stderr,
                )

                answers.append(False)
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