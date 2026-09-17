import argparse
import csv
import importlib
import json
import random
import statistics
import subprocess
from datetime import datetime
from pathlib import Path

from src.core import SimulationCore


SEEDS = [
    101,
    202,
    303,
    404,
    505,
    606,
    707,
    808,
    909,
    1010,
]

RESULTS_DIR = Path(__file__).parent / "results"


def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def load_controller(controller_name: str):
    module_path = f"src.utils.controllers.{controller_name}"
    module = importlib.import_module(module_path)

    if not hasattr(module, "action_decision"):
        raise AttributeError(
            f"{module_path} does not contain action_decision()"
        )

    return module.action_decision


def run_simulation(seed: int, action_decision):
    sim = SimulationCore(seed=seed)
    action_rng = random.Random(seed)

    actions = []

    while True:
        state = sim.step(actions)

        actions = []

        for agent, agent_state in zip(
            sim.env.agents,
            state["observations"],
        ):
            action = action_decision(agent_state, action_rng)
            actions.append((agent.agent_id, action))

        if state["num_agents"] == 0 or sim.env.time > 3000:
            return {
                "seed": seed,
                "score": float(state["score"]),
                "survival_time": float(sim.env.time),
            }


def calculate_summary(results):
    scores = [result["score"] for result in results]
    times = [result["survival_time"] for result in results]

    return {
        "runs": len(results),
        "mean_score": statistics.mean(scores),
        "median_score": statistics.median(scores),
        "min_score": min(scores),
        "max_score": max(scores),
        "score_stddev": (
            statistics.stdev(scores)
            if len(scores) > 1
            else 0.0
        ),
        "mean_survival": statistics.mean(times),
        "median_survival": statistics.median(times),
        "min_survival": min(times),
        "max_survival": max(times),
    }


def save_results(
    controller_name,
    experiment_name,
    results,
    summary,
):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{experiment_name}_{controller_name}_{timestamp}"

    csv_path = RESULTS_DIR / f"{base_name}.csv"
    json_path = RESULTS_DIR / f"{base_name}.json"

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "seed",
                "score",
                "survival_time",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    metadata = {
        "experiment": experiment_name,
        "controller": controller_name,
        "timestamp": timestamp,
        "git_commit": get_git_commit(),
        "seeds": SEEDS,
        "summary": summary,
        "results": results,
    }

    with json_path.open("w") as f:
        json.dump(metadata, f, indent=2)

    return csv_path, json_path


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--controller",
        required=True,
        help=(
            "Controller module name, for example "
            "s001_energy_policy"
        ),
    )

    parser.add_argument(
        "--experiment",
        required=True,
        help="Experiment label, for example EXP-S001",
    )

    args = parser.parse_args()

    action_decision = load_controller(args.controller)

    print("=" * 60)
    print(f"Experiment: {args.experiment}")
    print(f"Controller: {args.controller}")
    print(f"Git commit: {get_git_commit()}")
    print("=" * 60)

    results = []

    for index, seed in enumerate(SEEDS, start=1):
        result = run_simulation(
            seed=seed,
            action_decision=action_decision,
        )

        results.append(result)

        print(
            f"[{index:02d}/{len(SEEDS)}] "
            f"seed={result['seed']} "
            f"score={result['score']:.4f} "
            f"time={result['survival_time']:.1f}s"
        )

    summary = calculate_summary(results)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(f"Runs:            {summary['runs']}")
    print(f"Mean score:      {summary['mean_score']:.4f}")
    print(f"Median score:    {summary['median_score']:.4f}")
    print(f"Min score:       {summary['min_score']:.4f}")
    print(f"Max score:       {summary['max_score']:.4f}")
    print(f"Score stddev:    {summary['score_stddev']:.4f}")

    print()
    print(
        f"Mean survival:   "
        f"{summary['mean_survival']:.2f}s"
    )
    print(
        f"Median survival: "
        f"{summary['median_survival']:.2f}s"
    )
    print(
        f"Min survival:    "
        f"{summary['min_survival']:.2f}s"
    )
    print(
        f"Max survival:    "
        f"{summary['max_survival']:.2f}s"
    )

    csv_path, json_path = save_results(
        controller_name=args.controller,
        experiment_name=args.experiment,
        results=results,
        summary=summary,
    )

    print()
    print(f"Saved CSV:  {csv_path}")
    print(f"Saved JSON: {json_path}")


if __name__ == "__main__":
    main()