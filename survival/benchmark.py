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


DEFAULT_SEEDS = [
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
LOGS_DIR = Path(__file__).parent / "logs"


def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def git_is_dirty():
    try:
        output = subprocess.check_output(
            ["git", "status", "--porcelain"],
            text=True,
        )
        return bool(output.strip())
    except Exception:
        return None


def load_controller(controller_name: str):
    module_path = f"src.utils.controllers.{controller_name}"
    module = importlib.import_module(module_path)

    if not hasattr(module, "action_decision"):
        raise AttributeError(
            f"{module_path} does not contain action_decision()"
        )

    return module


def run_simulation(
    seed: int,
    action_decision,
    trace: bool = False,
):
    sim = SimulationCore(seed=seed)
    action_rng = random.Random(seed)

    actions = []

    previous_agents = len(sim.env.agents)
    previous_score = sim.env.score

    trace_rows = []

    while True:
        state = sim.step(actions)

        current_agents = state["num_agents"]
        current_score = state["score"]

        agents = list(sim.env.agents)

        energies = [
            agent.energy
            for agent in agents
        ]

        avg_energy = (
            statistics.mean(energies)
            if energies
            else 0.0
        )

        min_energy = (
            min(energies)
            if energies
            else 0.0
        )

        max_energy = (
            max(energies)
            if energies
            else 0.0
        )

        score_delta = current_score - previous_score
        population_delta = current_agents - previous_agents

        # Population losses accompanied by a sufficiently negative
        # score movement indicate predator kills, because predator
        # consumption applies a score penalty.
        possible_predator_death = (
            population_delta < 0
            and score_delta < 0
        )

        if trace:
            trace_rows.append(
                {
                    "time": float(sim.env.time),
                    "score": float(current_score),
                    "score_delta": float(score_delta),
                    "agents": int(current_agents),
                    "population_delta": int(population_delta),
                    "avg_energy": float(avg_energy),
                    "min_energy": float(min_energy),
                    "max_energy": float(max_energy),
                    "fruits": len(sim.env.fruits),
                    "trees": len(sim.env.trees),
                    "predators": len(sim.env.predators),
                    "possible_predator_death": possible_predator_death,
                }
            )

            # Print only interesting events.
            if (
                population_delta != 0
                or possible_predator_death
            ):
                print(
                    f"TRACE "
                    f"t={sim.env.time:.1f} "
                    f"agents={current_agents} "
                    f"delta={population_delta:+d} "
                    f"score_delta={score_delta:+.3f} "
                    f"energy(avg/min)="
                    f"{avg_energy:.1f}/{min_energy:.1f} "
                    f"fruits={len(sim.env.fruits)} "
                    f"predators={len(sim.env.predators)} "
                    f"pred_kill={possible_predator_death}"
                )

        previous_agents = current_agents
        previous_score = current_score

        actions = []

        for agent, agent_state in zip(
            sim.env.agents,
            state["observations"],
        ):
            action = action_decision(
                agent_state,
                action_rng,
            )

            actions.append(
                (agent.agent_id, action)
            )

        if (
            state["num_agents"] == 0
            or sim.env.time > 3000
        ):
            return {
                "seed": seed,
                "score": float(state["score"]),
                "survival_time": float(sim.env.time),
                "trace": trace_rows,
            }


def calculate_summary(results):
    scores = [
        result["score"]
        for result in results
    ]

    times = [
        result["survival_time"]
        for result in results
    ]

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
    seeds,
    results,
    summary,
    trace,
):
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    LOGS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    base_name = (
        f"{experiment_name}_"
        f"{controller_name}_"
        f"{timestamp}"
    )

    csv_path = RESULTS_DIR / f"{base_name}.csv"
    json_path = RESULTS_DIR / f"{base_name}.json"

    simple_results = [
        {
            "seed": result["seed"],
            "score": result["score"],
            "survival_time": result[
                "survival_time"
            ],
        }
        for result in results
    ]

    with csv_path.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "seed",
                "score",
                "survival_time",
            ],
        )

        writer.writeheader()
        writer.writerows(simple_results)

    metadata = {
        "experiment": experiment_name,
        "controller": controller_name,
        "timestamp": timestamp,
        "git_commit": get_git_commit(),
        "git_dirty": git_is_dirty(),
        "seeds": seeds,
        "summary": summary,
        "results": simple_results,
    }

    with json_path.open("w") as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    trace_paths = []

    if trace:
        for result in results:
            trace_path = (
                LOGS_DIR
                / (
                    f"{base_name}_"
                    f"seed_{result['seed']}_trace.csv"
                )
            )

            rows = result["trace"]

            if rows:
                with trace_path.open(
                    "w",
                    newline="",
                ) as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=rows[0].keys(),
                    )

                    writer.writeheader()
                    writer.writerows(rows)

                trace_paths.append(trace_path)

    return (
        csv_path,
        json_path,
        trace_paths,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--controller",
        required=True,
    )

    parser.add_argument(
        "--experiment",
        required=True,
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        help=(
            "Seeds to run. "
            "Example: --seeds 303 404"
        ),
    )

    parser.add_argument(
        "--trace",
        action="store_true",
        help=(
            "Record detailed simulation diagnostics."
        ),
    )

    args = parser.parse_args()

    controller = load_controller(
        args.controller
    )

    commit = get_git_commit()
    dirty = git_is_dirty()

    print("=" * 60)
    print(
        f"Experiment: {args.experiment}"
    )
    print(
        f"Controller: {args.controller}"
    )
    print(
        f"Git commit: {commit}"
    )
    print(
        f"Working tree dirty: {dirty}"
    )
    print(
        f"Seeds: {args.seeds}"
    )
    print("=" * 60)

    results = []

    for index, seed in enumerate(
        args.seeds,
        start=1,
    ):
        if hasattr(controller, "reset_policy_state"):
            controller.reset_policy_state()
        result = run_simulation(
            seed=seed,
            action_decision=controller.action_decision,
            trace=args.trace,
        )

        results.append(result)

        print(
            f"[{index:02d}/{len(args.seeds)}] "
            f"seed={result['seed']} "
            f"score={result['score']:.4f} "
            f"time={result['survival_time']:.1f}s"
        )

    summary = calculate_summary(results)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(
        f"Runs:            "
        f"{summary['runs']}"
    )

    print(
        f"Mean score:      "
        f"{summary['mean_score']:.4f}"
    )

    print(
        f"Median score:    "
        f"{summary['median_score']:.4f}"
    )

    print(
        f"Min score:       "
        f"{summary['min_score']:.4f}"
    )

    print(
        f"Max score:       "
        f"{summary['max_score']:.4f}"
    )

    print(
        f"Score stddev:    "
        f"{summary['score_stddev']:.4f}"
    )

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

    (
        csv_path,
        json_path,
        trace_paths,
    ) = save_results(
        controller_name=args.controller,
        experiment_name=args.experiment,
        seeds=args.seeds,
        results=results,
        summary=summary,
        trace=args.trace,
    )

    print()
    print(f"Saved CSV:  {csv_path}")
    print(f"Saved JSON: {json_path}")

    for path in trace_paths:
        print(
            f"Saved trace: {path}"
        )


if __name__ == "__main__":
    main()