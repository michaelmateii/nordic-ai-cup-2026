from __future__ import annotations

import shutil
from pathlib import Path


PREDICTOR = Path(
    r"medical\src\runtime\predictor.py"
)

BACKUP = Path(
    r"medical\src\runtime\predictor_pre_m077.py"
)


def main():
    text = PREDICTOR.read_text(
        encoding="utf-8"
    )

    if "M077_M076_FUSION_PATCH" in text:
        print("M077 already patched.")
        return

    if "M071_M070_FUSION_PATCH" not in text:
        raise RuntimeError(
            "Expected M071/M074 runtime as base."
        )

    shutil.copy2(
        PREDICTOR,
        BACKUP,
    )

    # ================================================================
    # 1. Constants
    # ================================================================

    marker = "class MedicalPredictor:"

    constants = r'''
# ============================================================================
# M077_M076_FUSION_PATCH
#
# Hidden-validated M074:
#   0.68 * M055 + 0.32 * M070
#
# New M077:
#   0.65 * M074 + 0.35 * M076
#
# Evidence selection is unchanged.
# ============================================================================

M076_MODEL_PATH = (
    Path(__file__).resolve().parents[2]
    / "artifacts"
    / "models"
    / "m076_multipassage_classifier"
)

M076_METADATA_PATH = (
    Path(__file__).resolve().parents[2]
    / "artifacts"
    / "models"
    / "m076_multipassage_metadata.json"
)

'''

    if marker not in text:
        raise RuntimeError(
            "MedicalPredictor not found."
        )

    text = text.replace(
        marker,
        constants + marker,
        1,
    )

    # ================================================================
    # 2. Load M076 after M070
    # ================================================================

    load_marker = '''        self.m070_threshold = float(
            m070_metadata[
                "fusion_threshold"
            ]
        )
'''

    if load_marker not in text:
        raise RuntimeError(
            "M070 metadata loading block not found."
        )

    loader = r'''

        if not M076_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"M076 model not found: "
                f"{M076_MODEL_PATH}"
            )

        if not M076_METADATA_PATH.exists():
            raise FileNotFoundError(
                f"M076 metadata not found: "
                f"{M076_METADATA_PATH}"
            )

        print(
            "[medical] loading "
            "M076 multi-passage classifier..."
        )

        self.m076_tokenizer = (
            AutoTokenizer.from_pretrained(
                M076_MODEL_PATH
            )
        )

        self.m076_model = (
            AutoModelForSequenceClassification
            .from_pretrained(
                M076_MODEL_PATH
            )
            .to(self.device)
            .eval()
        )

        with open(
            M076_METADATA_PATH,
            "r",
            encoding="utf-8",
        ) as handle:
            m076_metadata = json.load(
                handle
            )

        self.m076_weight = float(
            m076_metadata[
                "m076_weight"
            ]
        )

        self.m076_m074_weight = float(
            m076_metadata[
                "m074_weight"
            ]
        )

        self.m076_threshold = float(
            m076_metadata[
                "fusion_threshold"
            ]
        )

        self.m076_top_passages = int(
            m076_metadata[
                "top_passages"
            ]
        )

        self.m076_max_length = int(
            m076_metadata[
                "max_length"
            ]
        )
'''

    text = text.replace(
        load_marker,
        load_marker + loader,
        1,
    )

    # ================================================================
    # 3. Add runtime M076 scorer before _select_evidence
    # ================================================================

    method_marker = (
        "    def _select_evidence("
    )

    pos = text.find(
        method_marker
    )

    if pos == -1:
        raise RuntimeError(
            "_select_evidence not found."
        )

    method = r'''    def _m076_probability(
        self,
        question: str,
        words: list[
            dict[str, Any]
        ],
    ) -> float:
        """
        Build the same M047 hybrid candidate pool used during
        M076 training, keep the top passages by evidence-ranker
        score, concatenate them, and return the direct YES
        probability from the trained multi-passage classifier.
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
            ).strip()
            for candidate
            in candidates
        ]

        # ------------------------------------------------------------
        # Rank all hybrid candidates using the same M047/M049
        # evidence cross-encoder.
        # ------------------------------------------------------------

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

        order = (
            np.argsort(
                np.asarray(
                    scores,
                    dtype=float,
                )
            )[::-1]
        )

        # Match training-time de-duplication:
        # top passages by M047 score, unique exact text.
        passages: list[str] = []
        seen: set[str] = set()

        for index in order:
            passage = texts[
                int(index)
            ]

            if (
                not passage
                or passage in seen
            ):
                continue

            seen.add(
                passage
            )

            passages.append(
                passage
            )

            if (
                len(passages)
                >= self.m076_top_passages
            ):
                break

        context = "\n".join(
            f"[PASSAGE {i + 1}] {passage}"
            for i, passage
            in enumerate(passages)
        )

        model_text = (
            "[QUESTION] "
            + question
            + "\n"
            + context
        )

        encoded = (
            self.m076_tokenizer(
                model_text,
                truncation=True,
                max_length=(
                    self.m076_max_length
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
                self.m076_model(
                    **encoded
                ).logits
            )

            probability = (
                torch.softmax(
                    logits,
                    dim=-1,
                )[0, 1]
            )

        return float(
            probability
            .detach()
            .cpu()
        )

'''

    text = (
        text[:pos]
        + method
        + text[pos:]
    )

    # ================================================================
    # 4. Replace M071/M074 classification decision with M077.
    # ================================================================

    old_block = '''                fusion_probability = (
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
'''

    new_block = '''                m074_probability = (
                    self.m070_m055_weight
                    * m055_probability
                    + (
                        1.0
                        - self.m070_m055_weight
                    )
                    * m070_probability
                )

                m076_probability = (
                    self._m076_probability(
                        question,
                        words,
                    )
                )

                fusion_probability = (
                    self.m076_m074_weight
                    * m074_probability
                    + self.m076_weight
                    * m076_probability
                )

                predicted_yes = bool(
                    fusion_probability
                    >= self.m076_threshold
                )
'''

    if old_block not in text:
        raise RuntimeError(
            "Could not find M074 prediction block."
        )

    text = text.replace(
        old_block,
        new_block,
        1,
    )

    PREDICTOR.write_text(
        text,
        encoding="utf-8",
    )

    print(
        "M077 M074/M076 fusion patch complete."
    )

    print(
        f"Patched: {PREDICTOR}"
    )

    print(
        f"Backup:  {BACKUP}"
    )

    print(
        "M074 weight: 0.65"
    )

    print(
        "M076 weight: 0.35"
    )

    print(
        "Threshold:   0.3847283086"
    )

    print(
        "Evidence selector: unchanged"
    )


if __name__ == "__main__":
    main()