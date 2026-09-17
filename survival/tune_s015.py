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

# Fast development subset.
# These represent different S007 behavior:
# 202 = solid
# 404 = weak
# 606 = strong
SEARCH_SEEDS = [202, 404, 606]

# Deliberately small search around S007.
# Default S007 values:
# immediate=95
# manipulate=180
# energy_ratio=0.55
# exploration=0.15
CONFIGS = [
    {
        "name": "baseline",
        "immediate": 95,
        "manipulate": 180,
        "energy": 0.55,
        "explore": 0.15,
    },
    {
        "name": "pred_close_80",
        "immediate": 80,
        "manipulate": 180,
        "energy": 0.55,
        "explore": 0.15,
    },
    {
        "name": "pred_close_110",
        "immediate": 110,
        "manipulate": 180,
        "energy": 0.55,
        "explore": 0.15,
    },
    {
        "name": "manip_150",
        "immediate": 95,
        "manipulate": 150,
        "energy": 0.55,
        "explore": 0.15,
    },
    {
        "name": "manip_210",
        "immediate": 95,
        "manipulate": 210,
        "energy": 0.55,
        "explore": 0.15,
    },
    {
        "name": "energy_045",
        "immediate": 95,
        "manipulate": 180,
        "energy": 0.45,
        "explore": 0.15,
    },
    {
        "name": "energy_065",
        "immediate": 95,
        "manipulate": 180,
        "energy": 0.65,
        "explore": 0.15,
    },
    {
        "name": "explore_010",
        "immediate": 95,
        "manipulate": 180,
        "energy": 0.55,
        "explore": 0.10,
    },
    {
        "name": "explore_020",
        "immediate": 95,
        "manipulate": 180,
        "energy": 0.55,
        "explore": 0.20,
    },
    {
        "name": "combo_defensive",
        "immediate": 110,
        "manipulate": 160,
        "energy": 0.65,
        "explore": 0.15,
    },
    {
        "name": "combo_aggressive",
        "immediate": 80,
        "manipulate": 210,
        "energy": 0.45,
        "explore": 0.15,
    },
    {
        "name": "combo_balanced",
        "immediate": 90,
        "manipulate": 200,
        "energy": 0.60,
        "explore": 0.12,
    },
]


def extract_metric(pattern, text):
    match = re.search(pattern, text)

    if not match:
        raise RuntimeError(
            f"Could not parse metric using: {pattern}"
        )

    return float(match.group(1))


def run_config(config):
    env = os.environ.copy()

    env["SURV_IMMEDIATE_PREDATOR_DISTANCE"] = str(
        config["immediate"]
    )
    env["SURV_MANIPULATE_PREDATOR_DISTANCE"] = str(
        config["manipulate"]
    )
    env["SURV_MANIPULATE_ENERGY_RATIO"] = str(
        config["energy"]
    )
    env["SURV_EXPLORATION_PROB"] = str(
        config["explore"]
    )

    experiment = (
        f"TUNE-S015-{config['name']}"
    )

    cmd = [
        PYTHON,
        "benchmark.py",
        "--controller",
        "s015_tunable_policy",
        "--experiment",
        experiment,
        "--seeds",
        *[str(seed) for seed in SEARCH_SEEDS],
        "--workers",
        "3",
    ]

    print("\n" + "=" * 72)
    print(f"CONFIG: {config['name']}")
    print(
        f"immediate={config['immediate']} "
        f"manipulate={config['manipulate']} "
        f"energy={config['energy']} "
        f"explore={config['explore']}"
    )
    print("=" * 72)

    completed = subprocess.run(
        cmd,
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
    results = []

    for index, config in enumerate(
        CONFIGS,
        start=1,
    ):
        print(
            f"\nRUN {index}/{len(CONFIGS)}"
        )

        result = run_config(config)
        results.append(result)

    successful = [
        result
        for result in results
        if result["status"] == "OK"
    ]

    # Primary ranking by mean score.
    # Minimum score acts as a secondary robustness signal.
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
            f"{result['name']:<20} "
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
        / f"S015_parameter_search_{timestamp}.csv"
    )

    fieldnames = [
        "name",
        "immediate",
        "manipulate",
        "energy",
        "explore",
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
