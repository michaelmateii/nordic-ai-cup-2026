import csv
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PYTHON = sys.executable
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

SEEDS = [101, 202, 303, 404, 606]

CONFIGS = [
    {
        "name": "baseline",
        "stuck_steps": 12,
        "escape_steps": 15,
        "min_progress": 2.0,
    },

    # Stuck detection sensitivity.
    {
        "name": "stuck_8",
        "stuck_steps": 8,
        "escape_steps": 15,
        "min_progress": 2.0,
    },
    {
        "name": "stuck_16",
        "stuck_steps": 16,
        "escape_steps": 15,
        "min_progress": 2.0,
    },
    {
        "name": "stuck_20",
        "stuck_steps": 20,
        "escape_steps": 15,
        "min_progress": 2.0,
    },

    # Escape duration.
    {
        "name": "escape_8",
        "stuck_steps": 12,
        "escape_steps": 8,
        "min_progress": 2.0,
    },
    {
        "name": "escape_25",
        "stuck_steps": 12,
        "escape_steps": 25,
        "min_progress": 2.0,
    },

    # Progress threshold.
    {
        "name": "progress_1",
        "stuck_steps": 12,
        "escape_steps": 15,
        "min_progress": 1.0,
    },
    {
        "name": "progress_3",
        "stuck_steps": 12,
        "escape_steps": 15,
        "min_progress": 3.0,
    },
]


def extract_metric(pattern, text):
    match = re.search(pattern, text)

    if not match:
        raise RuntimeError(
            f"Could not parse metric: {pattern}"
        )

    return float(match.group(1))


def run_config(config):
    env = os.environ.copy()

    env["SURV_STUCK_STEPS"] = str(
        config["stuck_steps"]
    )

    env["SURV_ESCAPE_STEPS"] = str(
        config["escape_steps"]
    )

    env["SURV_MIN_PROGRESS"] = str(
        config["min_progress"]
    )

    command = [
        PYTHON,
        "benchmark.py",
        "--controller",
        "s015_tunable_policy",
        "--experiment",
        f"TUNE-STUCK-{config['name']}",
        "--seeds",
        *[str(seed) for seed in SEEDS],
        "--workers",
        "4",
    ]

    print("\n" + "=" * 72)
    print(config["name"])
    print(
        f"stuck_steps={config['stuck_steps']} "
        f"escape_steps={config['escape_steps']} "
        f"min_progress={config['min_progress']}"
    )
    print("=" * 72)

    completed = subprocess.run(
        command,
        env=env,
        text=True,
        capture_output=True,
    )

    print(completed.stdout)

    if completed.returncode != 0:
        print(completed.stderr)

        return {
            **config,
            "status": "FAILED",
        }

    output = completed.stdout

    return {
        **config,
        "status": "OK",
        "mean_score": extract_metric(
            r"Mean score:\s+([0-9.]+)",
            output,
        ),
        "median_score": extract_metric(
            r"Median score:\s+([0-9.]+)",
            output,
        ),
        "min_score": extract_metric(
            r"Min score:\s+([0-9.]+)",
            output,
        ),
        "mean_survival": extract_metric(
            r"Mean survival:\s+([0-9.]+)",
            output,
        ),
        "wall_seconds": extract_metric(
            r"Wall-clock runtime:\s+([0-9.]+)",
            output,
        ),
    }


def main():
    results = [
        run_config(config)
        for config in CONFIGS
    ]

    successful = [
        result
        for result in results
        if result["status"] == "OK"
    ]

    successful.sort(
        key=lambda result: (
            result["mean_score"],
            result["min_score"],
        ),
        reverse=True,
    )

    print("\n" + "=" * 90)
    print("RANKING")
    print("=" * 90)

    for rank, result in enumerate(
        successful,
        start=1,
    ):
        print(
            f"{rank:02d}. "
            f"{result['name']:<14} "
            f"mean={result['mean_score']:8.2f} "
            f"median={result['median_score']:8.2f} "
            f"min={result['min_score']:8.2f} "
            f"survival={result['mean_survival']:8.2f}s"
        )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    output_path = (
        RESULTS_DIR
        / f"S015_stuck_search_{timestamp}.csv"
    )

    fieldnames = [
        "name",
        "stuck_steps",
        "escape_steps",
        "min_progress",
        "status",
        "mean_score",
        "median_score",
        "min_score",
        "mean_survival",
        "wall_seconds",
    ]

    with output_path.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for result in successful:
            writer.writerow(result)

    print()
    print(f"Saved search summary: {output_path}")


if __name__ == "__main__":
    main()
