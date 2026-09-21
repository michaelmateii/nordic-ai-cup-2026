from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier

# Reuse the exact M063/M064 implementations we already validated.
from medical.src.retrieval.evaluate_m063_hard_negative_ranker import (
    build_pairs,
    get_topk_candidates,
    train_fold,
)
from medical.src.retrieval.evaluate_m064_ranker_arbiter import (
    NUMERIC_FEATURES as M064_FEATURES,
    build_question_table,
    make_hgb,
    tune_threshold,
)


ROOT = Path("medical")

MODELS = (
    ROOT
    / "artifacts"
    / "models"
)

MODELS.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# M055 CLASSIFIER
# =============================================================================

M055_DATA = (
    ROOT
    / "artifacts"
    / "classification"
    / "m055_base_nli_runtime_signal_oof.csv"
)

M055_OUTPUT = (
    MODELS
    / "meta_classifier_m055.joblib"
)

M055_FEATURES = [
    "question_position",

    "max_segment_ratio",
    "max_segment_entailment",
    "max_segment_margin",
    "max_segment_vs_other",

    "selected_entailment",
    "selected_contradiction",
    "selected_neutral",

    "evidence_max_score",
    "evidence_mean_score",
    "evidence_std_score",
    "evidence_top_gap",
]

# Use the OOF-calibrated mean threshold rather than
# optimizing another in-sample threshold here.
M055_THRESHOLD = 0.4989


def parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value

    return (
        str(value)
        .strip()
        .lower()
        in {
            "true",
            "1",
            "yes",
        }
    )


def train_m055():
    print()
    print("=" * 78)
    print("TRAINING FULL-DATA M055 CLASSIFIER")
    print("=" * 78)

    df = pd.read_csv(
        M055_DATA
    )

    missing = (
        set(M055_FEATURES)
        - set(df.columns)
    )

    if missing:
        raise RuntimeError(
            f"M055 missing features: "
            f"{sorted(missing)}"
        )

    y = np.asarray(
        [
            parse_bool(x)
            for x in df[
                "gold_bool"
            ]
        ],
        dtype=bool,
    )

    X = df[
        M055_FEATURES
    ]

    model = (
        HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=200,
            max_leaf_nodes=7,
            min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=42,
        )
    )

    model.fit(
        X,
        y,
    )

    probs = (
        model.predict_proba(
            X
        )[:, 1]
    )

    bundle = {
        "model":
            model,

        "feature_columns":
            M055_FEATURES,

        "threshold":
            M055_THRESHOLD,

        "training_rows":
            len(df),

        "model_name":
            "M055 full-data HGB",

        "mean_training_probability":
            float(
                probs.mean()
            ),
    }

    joblib.dump(
        bundle,
        M055_OUTPUT,
    )

    print(
        f"Rows:       {len(df)}"
    )

    print(
        f"Features:   {len(M055_FEATURES)}"
    )

    print(
        f"Threshold:  {M055_THRESHOLD:.4f}"
    )

    print(
        f"Saved:      {M055_OUTPUT}"
    )


# =============================================================================
# M063 EVIDENCE RANKER
# =============================================================================

M047_CANDIDATES = (
    ROOT
    / "artifacts"
    / "retrieval"
    / "m047_all_questions_candidates_oof.csv"
)

M063_OUTPUT = (
    MODELS
    / "hard_negative_evidence_ranker_m063"
)


def train_m063():
    print()
    print("=" * 78)
    print(
        "TRAINING FULL-DATA M063 HARD-NEGATIVE RANKER"
    )
    print("=" * 78)

    raw = pd.read_csv(
        M047_CANDIDATES
    )

    candidates = (
        get_topk_candidates(
            raw
        )
    )

    pairs = build_pairs(
        candidates
    )

    print(
        f"Questions:          "
        f"{candidates.question_id.nunique()}"
    )

    print(
        f"Top-20 candidates:  "
        f"{len(candidates)}"
    )

    print(
        f"Training pairs:     "
        f"{len(pairs)}"
    )

    print(
        f"Catastrophic pairs: "
        f"{int(pairs.catastrophic.sum())}"
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device:             {device}"
    )

    tokenizer, model = (
        train_fold(
            pairs,
            device,
            fold=0,
        )
    )

    M063_OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    tokenizer.save_pretrained(
        M063_OUTPUT
    )

    model.save_pretrained(
        M063_OUTPUT
    )

    print(
        f"Saved: {M063_OUTPUT}"
    )

    del model
    del tokenizer

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =============================================================================
# M064 ARBITER
# =============================================================================

M063_OOF = (
    ROOT
    / "artifacts"
    / "retrieval"
    / "m063_all_candidate_oof_scores.csv"
)

M064_OUTPUT = (
    MODELS
    / "evidence_arbiter_m064.joblib"
)


def train_m064():
    print()
    print("=" * 78)
    print(
        "TRAINING FULL-DATA M064 EVIDENCE ARBITER"
    )
    print("=" * 78)

    raw = pd.read_csv(
        M063_OOF
    )

    q = build_question_table(
        raw
    )

    changed = q[
        q["changed"]
    ].copy()

    informative = changed[
        ~np.isclose(
            changed[
                "utility"
            ],
            0.0,
        )
    ].copy()

    model = make_hgb()

    model.fit(
        informative[
            M064_FEATURES
        ],
        informative[
            "utility"
        ],
    )

    # Calibrate switch threshold on the complete
    # OOF disagreement table.
    scores = model.predict(
        changed[
            M064_FEATURES
        ]
    )

    threshold, selected_mean = (
        tune_threshold(
            changed,
            scores,
        )
    )

    bundle = {
        "model":
            model,

        "feature_columns":
            M064_FEATURES,

        "threshold":
            float(
                threshold
            ),

        "training_disagreements":
            len(changed),

        "informative_disagreements":
            len(informative),

        "oof_calibration_mean_tiou":
            float(
                selected_mean
            ),

        "model_name":
            "M064 full-data HGB utility arbiter",
    }

    joblib.dump(
        bundle,
        M064_OUTPUT,
    )

    print(
        f"Disagreements:       "
        f"{len(changed)}"
    )

    print(
        f"Informative:         "
        f"{len(informative)}"
    )

    print(
        f"Features:            "
        f"{len(M064_FEATURES)}"
    )

    print(
        f"Switch threshold:    "
        f"{threshold:.6f}"
    )

    print(
        f"OOF calibration mean:"
        f" {selected_mean:.4f}"
    )

    print(
        f"Saved: {M064_OUTPUT}"
    )


# =============================================================================
# METADATA
# =============================================================================

METADATA_OUTPUT = (
    MODELS
    / "m066_fast_deployment_metadata.json"
)


def write_metadata():
    metadata = {
        "deployment":
            "M066-fast",

        "classification": {
            "model":
                str(
                    M055_OUTPUT
                ),

            "threshold":
                M055_THRESHOLD,

            "oof_accuracy":
                0.9128,

            "oof_composite":
                0.6534,
        },

        "evidence": {
            "base_ranker":
                "existing M047/M049 runtime ranker",

            "hard_negative_ranker":
                str(
                    M063_OUTPUT
                ),

            "arbiter":
                str(
                    M064_OUTPUT
                ),

            "m047_oof_tiou":
                0.5158,

            "m063_oof_tiou":
                0.5205,

            "m064_oof_tiou":
                0.5271,
        },

        "notes": [
            "Deploy only after local endpoint smoke test.",
            "Local evaluator score is in-sample and not a benchmark.",
            "Public validation is the meaningful deployment test.",
        ],
    }

    METADATA_OUTPUT.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"Saved metadata: "
        f"{METADATA_OUTPUT}"
    )


# =============================================================================


def main():
    print("=" * 78)
    print(
        "Nordic AI Cup 2026 — "
        "M066 FAST DEPLOYMENT EXPORT"
    )
    print("=" * 78)

    train_m055()
    train_m063()
    train_m064()
    write_metadata()

    print()
    print("=" * 78)
    print(
        "M066 EXPORT COMPLETE"
    )
    print("=" * 78)

    print()
    print(
        f"M055 classifier:\n  "
        f"{M055_OUTPUT}"
    )

    print(
        f"M063 ranker:\n  "
        f"{M063_OUTPUT}"
    )

    print(
        f"M064 arbiter:\n  "
        f"{M064_OUTPUT}"
    )

    print(
        f"Metadata:\n  "
        f"{METADATA_OUTPUT}"
    )

    print()
    print(
        "NEXT: patch predictor.py once, "
        "smoke-test, then PUBLIC VALIDATION."
    )


if __name__ == "__main__":
    main()