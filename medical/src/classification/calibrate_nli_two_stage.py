from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold


DEFAULT_RESULTS = Path(
    r"medical\artifacts\classification\nli_claim_results.csv"
)

DEFAULT_OUTPUT = Path(
    r"medical\artifacts\classification\nli_two_stage_cv_results.csv"
)

N_SPLITS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--results",
        type=Path,
        default=DEFAULT_RESULTS,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    return parser.parse_args()


def parse_bool_series(
    series: pd.Series,
) -> np.ndarray:
    if series.dtype == bool:
        return series.to_numpy(dtype=bool)

    values = (
        series.astype(str)
        .str.strip()
        .str.lower()
    )

    mapping = {
        "true": True,
        "false": False,
        "yes": True,
        "no": False,
        "1": True,
        "0": False,
    }

    parsed = values.map(mapping)

    if parsed.isna().any():
        raise ValueError(
            "Unable to parse some gold labels."
        )

    return parsed.to_numpy(dtype=bool)


def midpoint_thresholds(
    values: np.ndarray,
) -> np.ndarray:
    unique = np.unique(values)

    if len(unique) == 1:
        return unique.copy()

    midpoint = (
        unique[:-1]
        + unique[1:]
    ) / 2.0

    return np.concatenate(
        (
            [unique[0] - 1e-9],
            midpoint,
            [unique[-1] + 1e-9],
        )
    )


def accuracy(
    gold: np.ndarray,
    predicted: np.ndarray,
) -> float:
    return float(
        np.mean(gold == predicted)
    )


def find_best_thresholds(
    ratio: np.ndarray,
    entailment: np.ndarray,
    gold: np.ndarray,
) -> tuple[float, float, float]:
    ratio_candidates = midpoint_thresholds(
        ratio
    )

    entailment_candidates = midpoint_thresholds(
        entailment
    )

    best_accuracy = -1.0
    best_ratio = 0.0
    best_entailment = 0.0

    for ratio_threshold in ratio_candidates:
        ratio_pass = (
            ratio >= ratio_threshold
        )

        for entailment_threshold in entailment_candidates:
            predicted = (
                ratio_pass
                & (
                    entailment
                    >= entailment_threshold
                )
            )

            score = accuracy(
                gold,
                predicted,
            )

            if score > best_accuracy:
                best_accuracy = score
                best_ratio = float(
                    ratio_threshold
                )
                best_entailment = float(
                    entailment_threshold
                )

    return (
        best_ratio,
        best_entailment,
        best_accuracy,
    )


def main() -> None:
    args = parse_args()

    df = pd.read_csv(
        args.results
    )

    gold = parse_bool_series(
        df["gold_yes"]
    )

    if int(gold.sum()) != 195:
        raise RuntimeError(
            f"Expected 195 positives; got {int(gold.sum())}"
        )

    entailment = df[
        "max_entailment"
    ].to_numpy(dtype=float)

    contradiction = df[
        "max_contradiction"
    ].to_numpy(dtype=float)

    ratio = (
        entailment
        / (
            entailment
            + contradiction
            + 1e-9
        )
    )

    groups = (
        df["transcript_id"]
        .astype(str)
        .to_numpy()
    )

    splitter = GroupKFold(
        n_splits=N_SPLITS
    )

    oof_prediction = np.zeros(
        len(df),
        dtype=bool,
    )

    fold_ratio_thresholds = []
    fold_entailment_thresholds = []
    fold_train_accuracy = []

    for fold, (
        train_index,
        test_index,
    ) in enumerate(
        splitter.split(
            df,
            gold,
            groups,
        ),
        start=1,
    ):
        (
            ratio_threshold,
            entailment_threshold,
            train_accuracy,
        ) = find_best_thresholds(
            ratio[train_index],
            entailment[train_index],
            gold[train_index],
        )

        test_prediction = (
            (
                ratio[test_index]
                >= ratio_threshold
            )
            & (
                entailment[test_index]
                >= entailment_threshold
            )
        )

        oof_prediction[
            test_index
        ] = test_prediction

        fold_ratio_thresholds.append(
            ratio_threshold
        )

        fold_entailment_thresholds.append(
            entailment_threshold
        )

        fold_train_accuracy.append(
            train_accuracy
        )

        print(
            f"Fold {fold}: "
            f"ratio>={ratio_threshold:.6f}, "
            f"entailment>={entailment_threshold:.6f}, "
            f"train_acc={train_accuracy:.4f}"
        )

    overall = accuracy(
        gold,
        oof_prediction,
    )

    matrix = confusion_matrix(
        gold,
        oof_prediction,
        labels=[False, True],
    )

    print()
    print("=" * 78)
    print(
        "Nordic AI Cup 2026 — "
        "Two-Stage NLI Calibration"
    )
    print("=" * 78)

    print(
        f"OOF accuracy:       "
        f"{overall:.4f}"
    )

    print(
        f"Predicted YES rate: "
        f"{oof_prediction.mean():.4f}"
    )

    print(
        f"Mean ratio threshold:      "
        f"{np.mean(fold_ratio_thresholds):.6f}"
    )

    print(
        f"Mean entailment threshold: "
        f"{np.mean(fold_entailment_thresholds):.6f}"
    )

    print(
        f"Confusion matrix: "
        f"{matrix.tolist()}"
    )

    print()

    for question_type in (
        "positive",
        "hard_negative",
        "off_topic",
    ):
        mask = (
            df["question_type"]
            == question_type
        ).to_numpy()

        score = accuracy(
            gold[mask],
            oof_prediction[mask],
        )

        correct = int(
            (
                gold[mask]
                == oof_prediction[mask]
            ).sum()
        )

        total = int(
            mask.sum()
        )

        print(
            f"{question_type:15s} "
            f"{score:.4f} "
            f"({correct}/{total})"
        )

    output = df[
        [
            "question_id",
            "transcript_id",
            "question_type",
            "question",
            "hypothesis",
            "max_entailment",
            "max_contradiction",
            "max_neutral",
        ]
    ].copy()

    output[
        "entailment_ratio"
    ] = ratio

    output[
        "gold_yes"
    ] = gold

    output[
        "oof_predicted_yes"
    ] = oof_prediction

    output[
        "correct"
    ] = (
        gold
        == oof_prediction
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        args.output,
        index=False,
    )

    print()
    print(
        f"Detailed results: "
        f"{args.output}"
    )


if __name__ == "__main__":
    main()