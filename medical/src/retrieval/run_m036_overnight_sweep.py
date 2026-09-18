from __future__ import annotations

import contextlib
import importlib.util
import sys
import time
import traceback
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]

BASE_SCRIPT = (
    REPO_ROOT
    / "medical"
    / "src"
    / "retrieval"
    / "finetune_hybrid_cross_encoder.py"
)

OUTPUT_DIR = (
    REPO_ROOT
    / "medical"
    / "artifacts"
    / "retrieval"
    / "overnight_m039"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "summary.csv"
)

MODEL_ID = (
    "cross-encoder/"
    "ms-marco-MiniLM-L6-v2"
)


CONFIGS = [
    {
        "name": "baseline_repro",
        "epochs": 4,
        "lr": 2e-5,
        "max_length": 192,
        "pos_threshold": 0.10,
        "max_positive": 18,
        "hard_negatives": 18,
        "random_negatives": 6,
    },
    {
        "name": "epochs_8_lr_2e5",
        "epochs": 8,
        "lr": 2e-5,
        "max_length": 192,
        "pos_threshold": 0.10,
        "max_positive": 18,
        "hard_negatives": 18,
        "random_negatives": 6,
    },
    {
        "name": "epochs_8_lr_1e5",
        "epochs": 8,
        "lr": 1e-5,
        "max_length": 192,
        "pos_threshold": 0.10,
        "max_positive": 18,
        "hard_negatives": 18,
        "random_negatives": 6,
    },
    {
        "name": "epochs_12_lr_1e5",
        "epochs": 12,
        "lr": 1e-5,
        "max_length": 192,
        "pos_threshold": 0.10,
        "max_positive": 18,
        "hard_negatives": 18,
        "random_negatives": 6,
    },
    {
        "name": "epochs_12_lr_7e6",
        "epochs": 12,
        "lr": 7e-6,
        "max_length": 192,
        "pos_threshold": 0.10,
        "max_positive": 18,
        "hard_negatives": 18,
        "random_negatives": 6,
    },
    {
        "name": "hardneg_30",
        "epochs": 8,
        "lr": 1e-5,
        "max_length": 192,
        "pos_threshold": 0.10,
        "max_positive": 18,
        "hard_negatives": 30,
        "random_negatives": 6,
    },
    {
        "name": "hardneg_30_random_3",
        "epochs": 8,
        "lr": 1e-5,
        "max_length": 192,
        "pos_threshold": 0.10,
        "max_positive": 18,
        "hard_negatives": 30,
        "random_negatives": 3,
    },
    {
        "name": "hardneg_24_random_12",
        "epochs": 8,
        "lr": 1e-5,
        "max_length": 192,
        "pos_threshold": 0.10,
        "max_positive": 18,
        "hard_negatives": 24,
        "random_negatives": 12,
    },
    {
        "name": "positive_threshold_005",
        "epochs": 8,
        "lr": 1e-5,
        "max_length": 192,
        "pos_threshold": 0.05,
        "max_positive": 24,
        "hard_negatives": 24,
        "random_negatives": 6,
    },
    {
        "name": "positive_threshold_015",
        "epochs": 8,
        "lr": 1e-5,
        "max_length": 192,
        "pos_threshold": 0.15,
        "max_positive": 18,
        "hard_negatives": 24,
        "random_negatives": 6,
    },
    {
        "name": "maxlen_256",
        "epochs": 8,
        "lr": 1e-5,
        "max_length": 256,
        "pos_threshold": 0.10,
        "max_positive": 18,
        "hard_negatives": 24,
        "random_negatives": 6,
    },
    {
        "name": "maxlen_256_hardneg30",
        "epochs": 8,
        "lr": 1e-5,
        "max_length": 256,
        "pos_threshold": 0.10,
        "max_positive": 18,
        "hard_negatives": 30,
        "random_negatives": 6,
    },
]


def load_training_module():
    spec = importlib.util.spec_from_file_location(
        "m036_training",
        BASE_SCRIPT,
    )

    if (
        spec is None
        or spec.loader is None
    ):
        raise RuntimeError(
            f"Cannot import {BASE_SCRIPT}"
        )

    module = (
        importlib.util.module_from_spec(
            spec
        )
    )

    spec.loader.exec_module(
        module
    )

    return module


def evaluate_result(
    csv_path: Path,
) -> dict[str, float]:
    df = pd.read_csv(
        csv_path
    )

    scores = df[
        "target_tiou"
    ].astype(float)

    return {
        "mean_tiou": float(
            scores.mean()
        ),
        "median_tiou": float(
            scores.median()
        ),
        "any_overlap": float(
            (scores > 0).mean()
        ),
        "tiou_ge_025": float(
            (scores >= 0.25).mean()
        ),
        "tiou_ge_050": float(
            (scores >= 0.50).mean()
        ),
        "tiou_ge_075": float(
            (scores >= 0.75).mean()
        ),
    }


def save_summary(
    rows: list[dict],
) -> None:
    frame = pd.DataFrame(
        rows
    )

    if "mean_tiou" in frame:
        frame = frame.sort_values(
            "mean_tiou",
            ascending=False,
            na_position="last",
        )

    frame.to_csv(
        SUMMARY_PATH,
        index=False,
    )


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    print("=" * 78)
    print(
        "Nordic AI Cup 2026 — "
        "M039 Overnight M036 Sweep"
    )
    print("=" * 78)

    print(
        f"Experiments: {len(CONFIGS)}"
    )

    print(
        f"Base model: {MODEL_ID}"
    )

    print(
        f"Summary:    {SUMMARY_PATH}"
    )

    for experiment_number, config in enumerate(
        CONFIGS,
        start=1,
    ):
        name = config["name"]

        print()
        print("=" * 78)
        print(
            f"[{experiment_number}/"
            f"{len(CONFIGS)}] "
            f"{name}"
        )
        print("=" * 78)

        output_path = (
            OUTPUT_DIR
            / f"{name}_oof.csv"
        )

        all_candidates_path = (
            OUTPUT_DIR
            / f"{name}_all_candidates.csv"
        )

        log_path = (
            OUTPUT_DIR
            / f"{name}.log"
        )

        started = time.perf_counter()

        row = {
            "name": name,
            **config,
            "status": "running",
        }

        try:
            module = (
                load_training_module()
            )

            module.MODEL_ID = MODEL_ID

            module.OUTPUT = output_path

            module.ALL_CANDIDATES_OUTPUT = (
                all_candidates_path
            )

            module.EPOCHS = int(
                config["epochs"]
            )

            module.LEARNING_RATE = float(
                config["lr"]
            )

            module.MAX_LENGTH = int(
                config["max_length"]
            )

            module.POSITIVE_TIOU_THRESHOLD = (
                float(
                    config[
                        "pos_threshold"
                    ]
                )
            )

            module.MAX_POSITIVE_CANDIDATES = (
                int(
                    config[
                        "max_positive"
                    ]
                )
            )

            module.MAX_HARD_NEGATIVES = int(
                config[
                    "hard_negatives"
                ]
            )

            module.MAX_RANDOM_NEGATIVES = int(
                config[
                    "random_negatives"
                ]
            )

            # Known-good GTX 1060 settings.
            module.BATCH_SIZE = 32
            module.EVAL_BATCH_SIZE = 128

            with log_path.open(
                "w",
                encoding="utf-8",
            ) as log_file:
                with contextlib.redirect_stdout(
                    log_file
                ), contextlib.redirect_stderr(
                    log_file
                ):
                    print(
                        f"EXPERIMENT: {name}",
                        flush=True,
                    )

                    print(
                        config,
                        flush=True,
                    )

                    module.main()

            metrics = evaluate_result(
                output_path
            )

            row.update(
                metrics
            )

            row["status"] = "ok"

        except Exception as exc:
            row["status"] = "failed"
            row["error"] = repr(
                exc
            )

            with log_path.open(
                "a",
                encoding="utf-8",
            ) as log_file:
                traceback.print_exc(
                    file=log_file
                )

        finally:
            elapsed = (
                time.perf_counter()
                - started
            )

            row["wall_seconds"] = (
                elapsed
            )

            results.append(
                row
            )

            save_summary(
                results
            )

            if (
                row["status"]
                == "ok"
            ):
                print(
                    f"{name}: "
                    f"mean_tIoU="
                    f"{row['mean_tiou']:.4f}, "
                    f"median="
                    f"{row['median_tiou']:.4f}, "
                    f"wall="
                    f"{elapsed / 60:.1f} min"
                )
            else:
                print(
                    f"{name}: FAILED "
                    f"after "
                    f"{elapsed / 60:.1f} min"
                )

                print(
                    f"See: {log_path}"
                )

        # Remove the imported module so every
        # experiment starts from clean globals.
        sys.modules.pop(
            "m036_training",
            None,
        )

    print()
    print("=" * 78)
    print("OVERNIGHT SWEEP COMPLETE")
    print("=" * 78)

    summary = pd.read_csv(
        SUMMARY_PATH
    )

    columns = [
        "name",
        "mean_tiou",
        "median_tiou",
        "tiou_ge_050",
        "tiou_ge_075",
        "wall_seconds",
        "status",
    ]

    available = [
        column
        for column in columns
        if column in summary.columns
    ]

    print(
        summary[
            available
        ].to_string(
            index=False
        )
    )

    print()
    print(
        f"Summary: {SUMMARY_PATH}"
    )


if __name__ == "__main__":
    main()