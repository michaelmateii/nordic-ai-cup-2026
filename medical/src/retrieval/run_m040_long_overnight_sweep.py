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
    / "overnight_m040_long"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "summary.csv"
)

MODEL_ID = (
    "cross-encoder/"
    "ms-marco-MiniLM-L6-v2"
)

SEEDS = (
    42,
    1337,
    2026,
    31415,
)


BASE_CONFIGS = [
    {
        "name": "e4_lr2e5_base",
        "epochs": 4,
        "lr": 2e-5,
        "max_length": 192,
        "pos_threshold": 0.10,
        "max_positive": 18,
        "hard_negatives": 18,
        "random_negatives": 6,
    },
    {
        "name": "e8_lr2e5",
        "epochs": 8,
        "lr": 2e-5,
        "max_length": 192,
        "pos_threshold": 0.10,
        "max_positive": 18,
        "hard_negatives": 18,
        "random_negatives": 6,
    },
    {
        "name": "e8_lr1e5",
        "epochs": 8,
        "lr": 1e-5,
        "max_length": 192,
        "pos_threshold": 0.10,
        "max_positive": 18,
        "hard_negatives": 18,
        "random_negatives": 6,
    },
    {
        "name": "e12_lr1e5",
        "epochs": 12,
        "lr": 1e-5,
        "max_length": 192,
        "pos_threshold": 0.10,
        "max_positive": 18,
        "hard_negatives": 18,
        "random_negatives": 6,
    },
    {
        "name": "e8_lr1e5_hard30",
        "epochs": 8,
        "lr": 1e-5,
        "max_length": 192,
        "pos_threshold": 0.10,
        "max_positive": 18,
        "hard_negatives": 30,
        "random_negatives": 6,
    },
    {
        "name": "e8_lr1e5_pos005",
        "epochs": 8,
        "lr": 1e-5,
        "max_length": 192,
        "pos_threshold": 0.05,
        "max_positive": 24,
        "hard_negatives": 24,
        "random_negatives": 6,
    },
    {
        "name": "e8_lr1e5_pos015",
        "epochs": 8,
        "lr": 1e-5,
        "max_length": 192,
        "pos_threshold": 0.15,
        "max_positive": 18,
        "hard_negatives": 24,
        "random_negatives": 6,
    },
    {
        "name": "e8_lr1e5_len256_hard30",
        "epochs": 8,
        "lr": 1e-5,
        "max_length": 256,
        "pos_threshold": 0.10,
        "max_positive": 18,
        "hard_negatives": 30,
        "random_negatives": 6,
    },
]


CONFIGS = []

for seed in SEEDS:
    for base in BASE_CONFIGS:
        config = dict(base)
        config["seed"] = seed
        config["name"] = (
            f"{base['name']}"
            f"_seed{seed}"
        )
        CONFIGS.append(config)


def load_training_module():
    module_name = "m036_training_dynamic"

    spec = importlib.util.spec_from_file_location(
        module_name,
        BASE_SCRIPT,
    )

    if (
        spec is None
        or spec.loader is None
    ):
        raise RuntimeError(
            f"Could not import {BASE_SCRIPT}"
        )

    module = (
        importlib.util.module_from_spec(
            spec
        )
    )

    sys.modules[
        module_name
    ] = module

    spec.loader.exec_module(
        module
    )

    return module_name, module


def evaluate_result(
    csv_path: Path,
) -> dict[str, float]:
    df = pd.read_csv(
        csv_path
    )

    scores = (
        df["target_tiou"]
        .astype(float)
    )

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

    if (
        len(frame)
        and "mean_tiou"
        in frame.columns
    ):
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
        "M040 Long Overnight L6 Sweep"
    )
    print("=" * 78)

    print(
        f"Configurations: {len(CONFIGS)}"
    )

    print(
        f"Seeds:          {SEEDS}"
    )

    print(
        f"Base model:     {MODEL_ID}"
    )

    print(
        f"Output:         {OUTPUT_DIR}"
    )

    print(
        f"Summary:        {SUMMARY_PATH}"
    )

    print(
        flush=True
    )

    sweep_start = (
        time.perf_counter()
    )

    for experiment_number, config in enumerate(
        CONFIGS,
        start=1,
    ):
        name = config["name"]

        print()
        print("=" * 78)
        print(
            f"[{experiment_number:02d}/"
            f"{len(CONFIGS):02d}] "
            f"{name}"
        )
        print("=" * 78)
        print(
            config
        )
        print(
            flush=True
        )

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

        started = (
            time.perf_counter()
        )

        row = {
            **config,
            "status": "running",
        }

        module_name = None

        try:
            (
                module_name,
                module,
            ) = load_training_module()

            module.MODEL_ID = MODEL_ID

            module.SEED = int(
                config["seed"]
            )

            module.OUTPUT = (
                output_path
            )

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
                config[
                    "max_length"
                ]
            )

            module.POSITIVE_TIOU_THRESHOLD = float(
                config[
                    "pos_threshold"
                ]
            )

            module.MAX_POSITIVE_CANDIDATES = int(
                config[
                    "max_positive"
                ]
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

            # Known-good GTX 1060 settings
            module.BATCH_SIZE = 32
            module.EVAL_BATCH_SIZE = 128

            with log_path.open(
                "w",
                encoding="utf-8",
            ) as log_file:
                with (
                    contextlib.redirect_stdout(
                        log_file
                    ),
                    contextlib.redirect_stderr(
                        log_file
                    ),
                ):
                    print(
                        "=" * 78,
                        flush=True,
                    )

                    print(
                        f"M040 SUB-EXPERIMENT: {name}",
                        flush=True,
                    )

                    print(
                        config,
                        flush=True,
                    )

                    print(
                        "=" * 78,
                        flush=True,
                    )

                    module.main()

                    log_file.flush()

            metrics = evaluate_result(
                output_path
            )

            row.update(
                metrics
            )

            row["status"] = "ok"

        except Exception as exc:
            row["status"] = (
                "failed"
            )

            row["error"] = repr(
                exc
            )

            with log_path.open(
                "a",
                encoding="utf-8",
            ) as log_file:
                print(
                    "\n"
                    + "=" * 78,
                    file=log_file,
                )

                print(
                    "EXPERIMENT FAILED",
                    file=log_file,
                )

                print(
                    "=" * 78,
                    file=log_file,
                )

                traceback.print_exc(
                    file=log_file
                )

        finally:
            elapsed = (
                time.perf_counter()
                - started
            )

            row[
                "wall_seconds"
            ] = elapsed

            row[
                "wall_minutes"
            ] = elapsed / 60.0

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
                    f"DONE {name}"
                )

                print(
                    f"  mean_tIoU: "
                    f"{row['mean_tiou']:.4f}"
                )

                print(
                    f"  median:    "
                    f"{row['median_tiou']:.4f}"
                )

                print(
                    f"  >=0.50:    "
                    f"{row['tiou_ge_050']:.4f}"
                )

                print(
                    f"  >=0.75:    "
                    f"{row['tiou_ge_075']:.4f}"
                )

                print(
                    f"  wall:      "
                    f"{elapsed / 60:.1f} min"
                )

            else:
                print(
                    f"FAILED {name}"
                )

                print(
                    f"  wall: "
                    f"{elapsed / 60:.1f} min"
                )

                print(
                    f"  log:  "
                    f"{log_path}"
                )

            total_elapsed = (
                time.perf_counter()
                - sweep_start
            )

            print(
                f"Total elapsed: "
                f"{total_elapsed / 3600:.2f} h"
            )

            print(
                f"Completed: "
                f"{experiment_number}/"
                f"{len(CONFIGS)}"
            )

            print(
                flush=True
            )

            if module_name is not None:
                sys.modules.pop(
                    module_name,
                    None,
                )

            try:
                import torch

                if (
                    torch.cuda
                    .is_available()
                ):
                    torch.cuda.empty_cache()

            except Exception:
                pass

    total_elapsed = (
        time.perf_counter()
        - sweep_start
    )

    print()
    print("=" * 78)
    print(
        "M040 LONG OVERNIGHT "
        "SWEEP COMPLETE"
    )
    print("=" * 78)

    print(
        f"Total wall time: "
        f"{total_elapsed / 3600:.2f} h"
    )

    summary = pd.read_csv(
        SUMMARY_PATH
    )

    wanted = [
        "name",
        "seed",
        "epochs",
        "lr",
        "mean_tiou",
        "median_tiou",
        "tiou_ge_050",
        "tiou_ge_075",
        "wall_minutes",
        "status",
    ]

    columns = [
        column
        for column in wanted
        if column
        in summary.columns
    ]

    print()

    print(
        summary[
            columns
        ]
        .head(20)
        .to_string(
            index=False
        )
    )

    print()

    print(
        f"Summary: "
        f"{SUMMARY_PATH}"
    )


if __name__ == "__main__":
    main()