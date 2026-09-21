from __future__ import annotations

import json
import shutil
from pathlib import Path


PREDICTOR = Path(
    r"medical\src\runtime\predictor.py"
)

BACKUP = Path(
    r"medical\src\runtime\predictor_pre_m071.py"
)


def main():
    text = PREDICTOR.read_text(
        encoding="utf-8"
    )

    if "import json\n" not in text:
        text = text.replace(
            "import base64\n",
            "import base64\nimport json\n",
            1,
        )
        
    if "M071_M070_FUSION_PATCH" in text:
        print("M071 already patched.")
        return

    # Require the known-good M067 runtime.
    if "M067_CLASSIFICATION_PATCH" not in text:
        raise RuntimeError(
            "Current predictor is not M067. "
            "Restore the known-good M067 predictor first."
        )

    shutil.copy2(
        PREDICTOR,
        BACKUP,
    )

    # ================================================================
    # 1. Add M070 paths/constants.
    # ================================================================

    class_marker = "class MedicalPredictor:"

    constants = r'''
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

'''

    text = text.replace(
        class_marker,
        constants + class_marker,
        1,
    )

    # ================================================================
    # 2. Load M070 immediately after M055.
    # ================================================================

    marker = '''        self.m055_threshold = float(
            m055_bundle[
                "threshold"
            ]
        )
'''

    if marker not in text:
        raise RuntimeError(
            "Could not find M055 loading block."
        )

    loader = r'''
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
'''

    text = text.replace(
        marker,
        marker + loader,
        1,
    )

    # ================================================================
    # 3. Add M070 candidate scoring method before _select_evidence.
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

    methods = r'''    def _m070_max_score(
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

'''

    text = (
        text[:pos]
        + methods
        + text[pos:]
    )

    # ================================================================
    # 4. Change M067 prediction from M055-only to M055/M070 fusion.
    # ================================================================

    old_block = '''                predicted_yes = bool(
                    classification[
                        "m055_predicted_yes"
                    ]
                )
'''

    new_block = '''                m055_probability = float(
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
'''

    if old_block not in text:
        raise RuntimeError(
            "Could not find M067 prediction block."
        )

    text = text.replace(
        old_block,
        new_block,
        1,
    )

    # ================================================================
    # Save.
    # ================================================================

    PREDICTOR.write_text(
        text,
        encoding="utf-8",
    )

    print(
        "M071 M055/M070 fusion patch complete."
    )

    print(
        f"Patched: {PREDICTOR}"
    )

    print(
        f"Backup:  {BACKUP}"
    )

    print(
        "M055 weight:      0.70"
    )

    print(
        "M070 weight:      0.30"
    )

    print(
        "Fusion threshold: 0.3053"
    )

    print(
        "Evidence selector: unchanged"
    )


if __name__ == "__main__":
    main()
    