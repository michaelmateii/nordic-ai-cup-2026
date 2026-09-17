from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold


DEFAULT_RESULTS = Path(
    r"medical\artifacts\classification\nli_segmentwise_results.csv"
)

DEFAULT_OUTPUT = Path(
    r"medical\artifacts\classification\nli_segmentwise_cv_results.csv"
)

N_SPLITS = 5

METHODS = (
    "max_segment_ratio",
    "max_segment_entailment",
    "max_segment_margin",
    "max_segment_vs_other",
)


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


def parse_bool(
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
        bad = values[
            parsed.isna()
        ].unique()

        raise ValueError(
            f"Bad boolean values: {bad}"
        )

    return parsed.to_numpy(dtype=bool)


def thresholds(
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
        (
            [unique[0] - 1e-9],
            midpoints,
            [unique[-1] + 1e-9],
        )
    )


def best_threshold(
    values: np.ndarray,
    gold: np.ndarray,
) -> tuple[float, float]:
    best_value = 0.0
    best_accuracy = -1.0

    for threshold in thresholds(values):
        prediction = values >= threshold

        score = float(
            np.mean(
                prediction == gold
            )
        )

        if score > best_accuracy:
            best_accuracy = score
            best_value = float(
                threshold
            )

    return (
        best_value,
        best_accuracy,
    )


def main() -> None:
    args = parse_args()

    df = pd.read_csv(
        args.results
    )

    required = {
        "question_id",
        "transcript_id",
        "question_type",
        "gold_yes",
        *METHODS,
    }

    missing = required - set(
        df.columns
    )

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    gold = parse_bool(
        df["gold_yes"]
    )

    if int(gold.sum()) != 195:
        raise RuntimeError(
            f"Expected 195 positives; "
            f"got {int(gold.sum())}"
        )

    groups = (
        df["transcript_id"]
        .astype(str)
        .to_numpy()
    )

    splitter = GroupKFold(
        n_splits=N_SPLITS
    )

    output_rows = []

    print("=" * 78)
    print(
        "Nordic AI Cup 2026 — "
        "Segmentwise NLI Calibration"
    )
    print("=" * 78)

    for method in METHODS:
        values = df[
            method
        ].to_numpy(dtype=float)

        predictions = np.zeros(
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
            (
                threshold,
                train_accuracy,
            ) = best_threshold(
                values[train_index],
                gold[train_index],
            )

            predictions[
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

        overall_accuracy = float(
            np.mean(
                predictions == gold
            )
        )

        matrix = confusion_matrix(
            gold,
            predictions,
            labels=[False, True],
        )

        print()
        print(method)
        print("-" * 78)

        print(
            f"OOF accuracy:       "
            f"{overall_accuracy:.4f}"
        )

        print(
            f"Predicted YES rate: "
            f"{predictions.mean():.4f}"
        )

        print(
            "Fold thresholds:    "
            + ", ".join(
                f"{value:.6f}"
                for value
                in fold_thresholds
            )
        )

        print(
            f"Mean threshold:     "
            f"{np.mean(fold_thresholds):.6f}"
        )

        print(
            f"Mean train acc:     "
            f"{np.mean(fold_train_accuracies):.4f}"
        )

        print(
            f"Confusion matrix:   "
            f"{matrix.tolist()}"
        )

        for question_type in (
            "positive",
            "hard_negative",
            "off_topic",
        ):
            mask = (
                df[
                    "question_type"
                ]
                == question_type
            ).to_numpy()

            score = float(
                np.mean(
                    predictions[mask]
                    == gold[mask]
                )
            )

            correct = int(
                (
                    predictions[mask]
                    == gold[mask]
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

        for index in range(
            len(df)
        ):
            output_rows.append(
                {
                    "method": method,
                    "row_index": index,
                    "question_id": (
                        df.iloc[index][
                            "question_id"
                        ]
                    ),
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
                    "score": float(
                        values[index]
                    ),
                    "gold_yes": bool(
                        gold[index]
                    ),
                    "predicted_yes": bool(
                        predictions[
                            index
                        ]
                    ),
                    "correct": bool(
                        predictions[
                            index
                        ]
                        == gold[index]
                    ),
                }
            )

    output_df = pd.DataFrame(
        output_rows
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