from __future__ import annotations

import re
import shutil
from pathlib import Path


PREDICTOR = Path(
    r"medical\src\runtime\predictor.py"
)

BACKUP = Path(
    r"medical\src\runtime\predictor_pre_m066.py"
)


def main():
    if not PREDICTOR.exists():
        raise FileNotFoundError(
            PREDICTOR
        )

    text = PREDICTOR.read_text(
        encoding="utf-8"
    )

    # ------------------------------------------------------------------
    # Refuse to patch twice.
    # ------------------------------------------------------------------

    if "M066_EVIDENCE_PATCH" in text:
        print(
            "M066 patch already present."
        )
        return

    shutil.copy2(
        PREDICTOR,
        BACKUP,
    )

    # ------------------------------------------------------------------
    # Add deployment paths directly before the class.
    #
    # predictor.py:
    #   medical/src/runtime/predictor.py
    #
    # parents[2]:
    #   medical/
    # ------------------------------------------------------------------

    marker = (
        "class MedicalPredictor:"
    )

    if marker not in text:
        raise RuntimeError(
            "Could not find "
            "MedicalPredictor class."
        )

    constants = r'''
# ============================================================================
# M066_EVIDENCE_PATCH
# M063 hard-negative evidence ranker + M064 evidence arbiter.
#
# Classification intentionally remains on the proven M049 runtime path.
# ============================================================================

M063_EVIDENCE_RANKER_PATH = (
    Path(__file__).resolve().parents[2]
    / "artifacts"
    / "models"
    / "hard_negative_evidence_ranker_m063"
)

M064_EVIDENCE_ARBITER_PATH = (
    Path(__file__).resolve().parents[2]
    / "artifacts"
    / "models"
    / "evidence_arbiter_m064.joblib"
)

M066_TOP_K = 20

'''

    text = text.replace(
        marker,
        constants + marker,
        1,
    )

    # ------------------------------------------------------------------
    # Insert M063/M064 loading immediately before "[medical] runtime ready".
    # ------------------------------------------------------------------

    runtime_ready_pattern = re.compile(
        r'''
        (?P<indent>[ ]{8})print\(
        \s*
        "\[medical\]\ runtime\ ready"
        \s*
        \)
        ''',
        re.VERBOSE,
    )

    match = runtime_ready_pattern.search(
        text
    )

    if not match:
        raise RuntimeError(
            "Could not locate "
            "runtime-ready print block."
        )

    load_block = r'''        if not M063_EVIDENCE_RANKER_PATH.exists():
            raise FileNotFoundError(
                "M063 evidence ranker "
                f"not found: "
                f"{M063_EVIDENCE_RANKER_PATH}"
            )

        if not M064_EVIDENCE_ARBITER_PATH.exists():
            raise FileNotFoundError(
                "M064 evidence arbiter "
                f"not found: "
                f"{M064_EVIDENCE_ARBITER_PATH}"
            )

        print(
            "[medical] loading "
            "M063 hard-negative evidence ranker..."
        )

        self.m063_tokenizer = (
            AutoTokenizer.from_pretrained(
                M063_EVIDENCE_RANKER_PATH
            )
        )

        self.m063_model = (
            AutoModelForSequenceClassification
            .from_pretrained(
                M063_EVIDENCE_RANKER_PATH
            )
            .to(self.device)
            .eval()
        )

        print(
            "[medical] loading "
            "M064 evidence arbiter..."
        )

        m064_bundle = joblib.load(
            M064_EVIDENCE_ARBITER_PATH
        )

        self.m064_arbiter = (
            m064_bundle["model"]
        )

        self.m064_feature_columns = list(
            m064_bundle[
                "feature_columns"
            ]
        )

        self.m064_threshold = float(
            m064_bundle[
                "threshold"
            ]
        )

'''

    text = (
        text[:match.start()]
        + load_block
        + text[
            match.start():
        ]
    )

    # ------------------------------------------------------------------
    # Replace _select_evidence with dual-ranker + arbiter implementation.
    # ------------------------------------------------------------------

    start_marker = (
        "    def _select_evidence("
    )

    end_marker = (
        "    def predict_request("
    )

    start = text.find(
        start_marker
    )

    end = text.find(
        end_marker,
        start,
    )

    if start == -1 or end == -1:
        raise RuntimeError(
            "Could not locate "
            "_select_evidence block."
        )

    replacement = r'''    def _score_evidence_candidates(
        self,
        *,
        question: str,
        texts: list[str],
        tokenizer: Any,
        model: Any,
        apply_sigmoid: bool,
    ) -> np.ndarray:
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

            encoded = tokenizer(
                [
                    question
                    for _ in batch_texts
                ],
                batch_texts,
                padding=True,
                truncation=True,
                max_length=EVIDENCE_MAX_LENGTH,
                return_tensors="pt",
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
                    model(
                        **encoded
                    )
                    .logits
                    .reshape(-1)
                )

                if apply_sigmoid:
                    values = torch.sigmoid(
                        logits
                    )
                else:
                    values = logits

            scores.extend(
                values
                .detach()
                .cpu()
                .float()
                .numpy()
                .tolist()
            )

        return np.asarray(
            scores,
            dtype=float,
        )

    @staticmethod
    def _m066_token_jaccard(
        first: str,
        second: str,
    ) -> float:
        left = set(
            str(first)
            .lower()
            .split()
        )

        right = set(
            str(second)
            .lower()
            .split()
        )

        union = (
            left
            | right
        )

        if not union:
            return 0.0

        return float(
            len(
                left
                & right
            )
            / len(union)
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
        """
        M066 evidence selection.

        M047/M049 remains the first-stage evidence ranker.

        M063 reranks only the M047 top-20 candidate set.

        M064 predicts whether switching from M047's top candidate
        to M063's top candidate is beneficial.

        IMPORTANT:
        The third return value remains M047's original top evidence
        score. The M049 classification disagreement gate was calibrated
        on that score, so M066 evidence arbitration must not change it.
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

        # --------------------------------------------------------------
        # M047/M049 first-stage scores.
        #
        # Preserve sigmoid here because all M047/M049 gate calibration
        # was performed using these [0, 1] scores.
        # --------------------------------------------------------------

        m047_scores = (
            self._score_evidence_candidates(
                question=question,
                texts=texts,
                tokenizer=(
                    self.evidence_tokenizer
                ),
                model=(
                    self.evidence_model
                ),
                apply_sigmoid=True,
            )
        )

        order47 = np.argsort(
            m047_scores
        )[::-1]

        base_index = int(
            order47[0]
        )

        base_candidate = (
            candidates[
                base_index
            ]
        )

        # Score returned to the existing M049
        # classification gate.
        classification_evidence_score = (
            float(
                m047_scores[
                    base_index
                ]
            )
        )

        # Nothing meaningful to arbitrate.
        if len(order47) < 2:
            return (
                float(
                    base_candidate[
                        "start"
                    ]
                ),
                float(
                    base_candidate[
                        "end"
                    ]
                ),
                classification_evidence_score,
            )

        # --------------------------------------------------------------
        # M063 only sees M047's top-20.
        # This matches the OOF experiment distribution.
        # --------------------------------------------------------------

        top_indices = order47[
            :min(
                M066_TOP_K,
                len(order47),
            )
        ]

        top_texts = [
            texts[
                int(index)
            ]
            for index
            in top_indices
        ]

        # M063 OOF scores were raw scalar logits,
        # so DO NOT sigmoid these.
        m063_top_scores = (
            self._score_evidence_candidates(
                question=question,
                texts=top_texts,
                tokenizer=(
                    self.m063_tokenizer
                ),
                model=(
                    self.m063_model
                ),
                apply_sigmoid=False,
            )
        )

        order63_local = (
            np.argsort(
                m063_top_scores
            )[::-1]
        )

        m063_local_index = int(
            order63_local[0]
        )

        m063_global_index = int(
            top_indices[
                m063_local_index
            ]
        )

        m063_candidate = (
            candidates[
                m063_global_index
            ]
        )

        # Same choice -> no gate needed.
        if (
            m063_global_index
            == base_index
        ):
            return (
                float(
                    base_candidate[
                        "start"
                    ]
                ),
                float(
                    base_candidate[
                        "end"
                    ]
                ),
                classification_evidence_score,
            )

        # --------------------------------------------------------------
        # Build exactly the inference-safe M064 features.
        # --------------------------------------------------------------

        rank63_in_47 = int(
            np.where(
                top_indices
                == m063_global_index
            )[0][0]
        ) + 1

        score47 = float(
            m047_scores[
                base_index
            ]
        )

        score63_in_47 = float(
            m047_scores[
                m063_global_index
            ]
        )

        if len(order47) >= 2:
            m047_second = float(
                m047_scores[
                    int(
                        order47[1]
                    )
                ]
            )
        else:
            m047_second = score47

        m047_top_gap = (
            score47
            - m047_second
        )

        score63 = float(
            m063_top_scores[
                m063_local_index
            ]
        )

        if len(order63_local) >= 2:
            m063_second = float(
                m063_top_scores[
                    int(
                        order63_local[1]
                    )
                ]
            )
        else:
            m063_second = score63

        m063_margin = (
            score63
            - m063_second
        )

        m047_gap_to_m063 = (
            score47
            - score63_in_47
        )

        m047_start = float(
            base_candidate["start"]
        )

        m047_end = float(
            base_candidate["end"]
        )

        m063_start = float(
            m063_candidate["start"]
        )

        m063_end = float(
            m063_candidate["end"]
        )

        m047_duration = (
            m047_end
            - m047_start
        )

        m063_duration = (
            m063_end
            - m063_start
        )

        duration_difference = (
            m063_duration
            - m047_duration
        )

        duration_ratio = (
            m063_duration
            / (
                m047_duration
                + 1e-6
            )
        )

        m047_text = str(
            base_candidate["text"]
        )

        m063_text = str(
            m063_candidate["text"]
        )

        m047_word_count = float(
            len(
                m047_text.split()
            )
        )

        m063_word_count = float(
            len(
                m063_text.split()
            )
        )

        word_count_difference = (
            m063_word_count
            - m047_word_count
        )

        start_difference = (
            m063_start
            - m047_start
        )

        end_difference = (
            m063_end
            - m047_end
        )

        center47 = (
            m047_start
            + m047_end
        ) / 2.0

        center63 = (
            m063_start
            + m063_end
        ) / 2.0

        center_difference = (
            center63
            - center47
        )

        m063_is_earlier = float(
            center63
            < center47
        )

        text_token_jaccard = (
            self._m066_token_jaccard(
                m047_text,
                m063_text,
            )
        )

        feature_values = {
            "rank63_in_47":
                float(
                    rank63_in_47
                ),

            "m063_margin":
                m063_margin,

            "m047_top_gap":
                m047_top_gap,

            "m047_gap_to_m063":
                m047_gap_to_m063,

            "score47":
                score47,

            "score63_in_47":
                score63_in_47,

            "m047_duration":
                m047_duration,

            "m063_duration":
                m063_duration,

            "duration_difference":
                duration_difference,

            "duration_ratio":
                duration_ratio,

            "m047_word_count":
                m047_word_count,

            "m063_word_count":
                m063_word_count,

            "word_count_difference":
                word_count_difference,

            "start_difference":
                start_difference,

            "end_difference":
                end_difference,

            "center_difference":
                center_difference,

            "text_token_jaccard":
                text_token_jaccard,

            "m063_is_earlier":
                m063_is_earlier,
        }

        missing = (
            set(
                self.m064_feature_columns
            )
            - set(
                feature_values
            )
        )

        if missing:
            raise RuntimeError(
                "Missing M064 features: "
                f"{sorted(missing)}"
            )

        arbiter_row = pd.DataFrame(
            [
                {
                    name:
                        feature_values[
                            name
                        ]
                    for name
                    in self.m064_feature_columns
                }
            ],
            columns=(
                self.m064_feature_columns
            ),
        )

        predicted_utility = float(
            self.m064_arbiter.predict(
                arbiter_row
            )[0]
        )

        choose_m063 = (
            predicted_utility
            >= self.m064_threshold
        )

        if choose_m063:
            chosen = (
                m063_candidate
            )
        else:
            chosen = (
                base_candidate
            )

        return (
            float(
                chosen["start"]
            ),
            float(
                chosen["end"]
            ),
            classification_evidence_score,
        )

'''

    text = (
        text[:start]
        + replacement
        + text[end:]
    )

    PREDICTOR.write_text(
        text,
        encoding="utf-8",
    )

    print(
        "M066 evidence patch complete."
    )

    print(
        f"Patched: {PREDICTOR}"
    )

    print(
        f"Backup:  {BACKUP}"
    )

    print(
        "Classification: existing M049 "
        "(unchanged)"
    )

    print(
        "Evidence: M047/M049 + M063 + M064"
    )


if __name__ == "__main__":
    main()