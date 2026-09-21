from __future__ import annotations

import shutil
from pathlib import Path


PREDICTOR = Path(
    r"medical\src\runtime\predictor.py"
)

BACKUP = Path(
    r"medical\src\runtime\predictor_pre_m067.py"
)


def main():
    text = PREDICTOR.read_text(
        encoding="utf-8"
    )

    if "M067_CLASSIFICATION_PATCH" in text:
        print("M067 already patched.")
        return

    shutil.copy2(
        PREDICTOR,
        BACKUP,
    )

    # ================================================================
    # 1. Upgrade NLI from DeBERTa-v3-small to the exact M055 base model.
    # ================================================================

    old_model = (
        'NLI_MODEL_ID = '
        '"cross-encoder/nli-deberta-v3-small"'
    )

    new_model = (
        'NLI_MODEL_ID = '
        '"cross-encoder/nli-deberta-v3-base"'
    )

    if old_model not in text:
        raise RuntimeError(
            "Could not find small NLI model constant."
        )

    text = text.replace(
        old_model,
        new_model,
        1,
    )

    # ================================================================
    # 2. Add M055 deployment model path.
    # ================================================================

    class_marker = "class MedicalPredictor:"

    if class_marker not in text:
        raise RuntimeError(
            "MedicalPredictor class not found."
        )

    constants = r'''
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

'''

    text = text.replace(
        class_marker,
        constants + class_marker,
        1,
    )

    # ================================================================
    # 3. Load M055 classifier.
    # Insert immediately before M049 evidence-ranker loading.
    # ================================================================

    marker = '''        print(
            "[medical] loading "
            "M049 hybrid evidence ranker..."
        )
'''

    if marker not in text:
        raise RuntimeError(
            "Could not find M049 loading marker."
        )

    loader = r'''        if not M055_CLASSIFIER_PATH.exists():
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

'''

    text = text.replace(
        marker,
        loader + marker,
        1,
    )

    # ================================================================
    # 4. Add M055 evidence-statistics + classification methods.
    # ================================================================

    evidence_marker = (
        "    def _select_evidence("
    )

    pos = text.find(
        evidence_marker
    )

    if pos == -1:
        raise RuntimeError(
            "_select_evidence not found."
        )

    methods = r'''    def _m055_evidence_features(
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

'''

    text = (
        text[:pos]
        + methods
        + text[pos:]
    )

    # ================================================================
    # 5. Make question position available at runtime.
    # ================================================================

    old_loop = (
        "        for question in questions:\n"
    )

    new_loop = (
        "        for question_position, "
        "question in enumerate("
        "questions, start=1):\n"
    )

    if old_loop not in text:
        raise RuntimeError(
            "Question loop not found."
        )

    text = text.replace(
        old_loop,
        new_loop,
        1,
    )

    # ================================================================
    # 6. Replace M049 classification/gating block with M055.
    #
    # Leave original _select_evidence completely untouched.
    # ================================================================

    block_start = text.find(
        "                classification = (\n",
        text.find(
            "for question_position"
        ),
    )

    block_end = text.find(
        "                if (\n"
        "                    not np.isfinite(\n",
        block_start,
    )

    if (
        block_start == -1
        or block_end == -1
    ):
        raise RuntimeError(
            "Could not locate old "
            "classification/evidence block."
        )

    new_block = r'''                evidence_features = (
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

                predicted_yes = bool(
                    classification[
                        "m055_predicted_yes"
                    ]
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

'''

    text = (
        text[:block_start]
        + new_block
        + text[block_end:]
    )

    PREDICTOR.write_text(
        text,
        encoding="utf-8",
    )

    print(
        "M067 classification patch complete."
    )

    print(
        f"Patched: {PREDICTOR}"
    )

    print(
        f"Backup:  {BACKUP}"
    )

    print(
        "Classification: M055"
    )

    print(
        "Evidence: original pre-M066 selector"
    )

    print(
        "NLI: cross-encoder/"
        "nli-deberta-v3-base"
    )


if __name__ == "__main__":
    main()