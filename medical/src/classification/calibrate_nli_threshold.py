from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold


DEFAULT_RESULTS = Path(
    r"medical\artifacts\classification\nli_baseline_results.csv"
)

DEFAULT_OUTPUT = Path(
    r"medical\artifacts\classification\nli_threshold_cv_results.csv"
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


def as_bool(series: pd.Series) -> np.ndarray:
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
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
    }

    parsed = values.map(mapping)

    if parsed.isna().any():
        bad = values[
            parsed.isna()
        ].unique()

        raise ValueError(
            f"Could not parse booleans: {bad}"
        )

    return parsed.to_numpy(dtype=bool)


def build_scores(
    df: pd.DataFrame,
) -> dict[str, np.ndarray]:
    entailment = df[
        "max_entailment"
    ].to_numpy(dtype=float)

    contradiction = df[
        "max_contradiction"
    ].to_numpy(dtype=float)

    neutral = df[
        "max_neutral"
    ].to_numpy(dtype=float)

    return {
        "entailment": entailment,

        "entailment_minus_contradiction": (
            entailment - contradiction
        ),

        "entailment_minus_max_other": (
            entailment
            - np.maximum(
                contradiction,
                neutral,
            )
        ),

        "entailment_ratio": (
            entailment
            / (
                entailment
                + contradiction
                + 1e-9
            )
        ),
    }


def candidate_thresholds(
    values: np.ndarray,
) -> np.ndarray:
    unique = np.unique(values)

    if len(unique) == 1:
        return unique

    midpoints = (
        unique[:-1]
        + unique[1:]
    ) / 2.0

    return np.concatenate(
        [
            [unique[0] - 1e-9],
            midpoints,
            [unique[-1] + 1e-9],
        ]
    )


def accuracy(
    gold: np.ndarray,
    pred: np.ndarray,
) -> float:
    return float(
        np.mean(gold == pred)
    )


def find_best_threshold(
    values: np.ndarray,
    gold: np.ndarray,
) -> tuple[float, float]:
    best_threshold = 0.0
    best_accuracy = -1.0

    for threshold in candidate_thresholds(
        values
    ):
        pred = values >= threshold

        score = accuracy(
            gold,
            pred,
        )

        if score > best_accuracy:
            best_accuracy = score
            best_threshold = float(
                threshold
            )

    return (
        best_threshold,
        best_accuracy,
    )


def main() -> None:
    args = parse_args()

    df = pd.read_csv(
        args.results
    )

    required = {
        "transcript_id",
        "question_type",
        "gold_yes",
        "max_entailment",
        "max_contradiction",
        "max_neutral",
    }

    missing = required - set(
        df.columns
    )

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    gold = as_bool(
        df["gold_yes"]
    )

    if int(gold.sum()) != 195:
        raise RuntimeError(
            f"Expected 195 positives, "
            f"got {int(gold.sum())}"
        )

    groups = (
        df["transcript_id"]
        .astype(str)
        .to_numpy()
    )

    score_sets = build_scores(
        df
    )

    splitter = GroupKFold(
        n_splits=N_SPLITS
    )

    all_method_results = []

    print("=" * 78)
    print(
        "Nordic AI Cup 2026 — "
        "Conversation-Level NLI Threshold Calibration"
    )
    print("=" * 78)

    for method_name, values in (
        score_sets.items()
    ):
        oof_predictions = np.zeros(
            len(df),
            dtype=bool,
        )

        fold_thresholds = []
        fold_train_accuracies = []

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
            threshold, train_accuracy = (
                find_best_threshold(
                    values[train_index],
                    gold[train_index],
                )
            )

            oof_predictions[
                test_index
            ] = (
                values[test_index]
                >= threshold
            )

            fold_thresholds.append(
                threshold
            )

            fold_train_accuracies.append(
                train_accuracy
            )

        overall_accuracy = accuracy(
            gold,
            oof_predictions,
        )

        predicted_yes_rate = float(
            oof_predictions.mean()
        )

        matrix = confusion_matrix(
            gold,
            oof_predictions,
            labels=[False, True],
        )

        print()
        print(method_name)
        print("-" * 78)

        print(
            f"OOF accuracy:       "
            f"{overall_accuracy:.4f}"
        )

        print(
            f"Predicted YES rate: "
            f"{predicted_yes_rate:.4f}"
        )

        print(
            "Fold thresholds:    "
            + ", ".join(
                f"{value:.4f}"
                for value in fold_thresholds
            )
        )

        print(
            f"Mean threshold:     "
            f"{np.mean(fold_thresholds):.4f}"
        )

        print(
            "Confusion matrix:   "
            f"{matrix.tolist()}"
        )

        for question_type in (
            "positive",
            "hard_negative",
            "off_topic",
        ):
            mask = (
                df["question_type"]
                == question_type
            ).to_numpy()

            type_accuracy = accuracy(
                gold[mask],
                oof_predictions[mask],
            )

            print(
                f"{question_type:15s} "
                f"{type_accuracy:.4f}"
            )

        for index in range(len(df)):
            all_method_results.append(
                {
                    "method": method_name,
                    "row_index": index,
                    "transcript_id": (
                        df.iloc[index][
                            "transcript_id"
                        ]
                    ),
                    "question_type": (
                        df.iloc[index][
                            "question_type"
                        ]
                    ),
                    "gold_yes": bool(
                        gold[index]
                    ),
                    "score": float(
                        values[index]
                    ),
                    "predicted_yes": bool(
                        oof_predictions[
                            index
                        ]
                    ),
                    "correct": bool(
                        oof_predictions[index]
                        == gold[index]
                    ),
                }
            )

    output_df = pd.DataFrame(
        all_method_results
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_df.to_csv(
        args.output,
        index=False,
    )

    print()
    print(
        f"Detailed OOF results: "
        f"{args.output}"
    )


if __name__ == "__main__":
    main()