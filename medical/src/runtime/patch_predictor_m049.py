from __future__ import annotations

import ast
import re
import shutil
from pathlib import Path


PATH = Path(
    r"medical\src\runtime\predictor.py"
)

BACKUP = Path(
    r"medical\src\runtime\predictor_pre_m049.py"
)


def replace_method(
    text: str,
    method_name: str,
    next_method_name: str,
    replacement: str,
) -> str:
    start_marker = (
        f"    def {method_name}("
    )

    end_marker = (
        f"    def {next_method_name}("
    )

    start = text.find(
        start_marker
    )

    end = text.find(
        end_marker,
        start,
    )

    if start < 0:
        raise RuntimeError(
            f"Could not find "
            f"{method_name}"
        )

    if end < 0:
        raise RuntimeError(
            f"Could not find "
            f"{next_method_name}"
        )

    return (
        text[:start]
        + replacement.rstrip()
        + "\n\n"
        + text[end:]
    )


if not PATH.exists():
    raise FileNotFoundError(
        PATH
    )

if not BACKUP.exists():
    shutil.copy2(
        PATH,
        BACKUP,
    )

text = PATH.read_text(
    encoding="utf-8"
)


# ============================================================================
# Imports
# ============================================================================

old = """import joblib
import numpy as np
import torch
"""

new = """import joblib
import numpy as np
import pandas as pd
import torch
"""

if old not in text:
    raise RuntimeError(
        "Could not patch pandas import."
    )

text = text.replace(
    old,
    new,
    1,
)


# ============================================================================
# Runtime paths / thresholds
# ============================================================================

old = """CLASSIFICATION_THRESHOLD = 0.000585
"""

new = """CLASSIFICATION_THRESHOLD = 0.000585

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
"""

if old not in text:
    raise RuntimeError(
        "Could not patch constants."
    )

text = text.replace(
    old,
    new,
    1,
)


# ============================================================================
# Add hybrid-candidate + M042 surface helpers
# ============================================================================

marker = "\ndef get_label_indices(\n"

if marker not in text:
    raise RuntimeError(
        "Could not find "
        "get_label_indices marker."
    )

helpers = r'''

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
'''

text = text.replace(
    marker,
    helpers + marker,
    1,
)


# ============================================================================
# Replace old sentence-ranker initialization with M042 + M049 models
# ============================================================================

start_marker = '''        print(
            "[medical] loading "
            "sentence ranker..."
        )
'''

end_marker = '''        print(
            "[medical] runtime ready"
        )
'''

start = text.find(
    start_marker
)

end = text.find(
    end_marker,
    start,
)

if start < 0 or end < 0:
    raise RuntimeError(
        "Could not find old "
        "sentence-ranker init block."
    )

end += len(
    end_marker
)

replacement = '''        if not META_CLASSIFIER_PATH.exists():
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
'''

text = (
    text[:start]
    + replacement
    + text[end:]
)


# ============================================================================
# Replace _classify
# ============================================================================

classify_method = r'''    def _classify(
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
'''

text = replace_method(
    text,
    "_classify",
    "_select_evidence",
    classify_method,
)


# ============================================================================
# Replace _select_evidence
# ============================================================================

evidence_method = r'''    def _select_evidence(
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
'''

text = replace_method(
    text,
    "_select_evidence",
    "predict_request",
    evidence_method,
)


# ============================================================================
# Patch predict_request question loop
# ============================================================================

loop_start = text.find(
    "        for question in questions:\n"
)

loop_end = text.find(
    "        return {\n"
    "            \"answers\": answers,",
    loop_start,
)

if loop_start < 0 or loop_end < 0:
    raise RuntimeError(
        "Could not find "
        "predict_request loop."
    )

new_loop = r'''        for question in questions:
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

'''

text = (
    text[:loop_start]
    + new_loop
    + text[loop_end:]
)


# ============================================================================
# Validate generated Python before writing
# ============================================================================

ast.parse(
    text
)

PATH.write_text(
    text,
    encoding="utf-8",
)

print(
    "M049 predictor patch complete."
)

print(
    f"Patched: {PATH}"
)

print(
    f"Backup:  {BACKUP}"
)

print(
    "M018 threshold:",
    0.000585,
)

print(
    "M042 threshold:",
    0.384878,
)

print(
    "M049 evidence gate:",
    0.369304,
)