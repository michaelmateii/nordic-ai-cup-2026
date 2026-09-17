import random
import statistics

from src.core import SimulationCore
from src.utils.controllers.s001_energy_policy import action_decision

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


def run_simulation(seed: int):
    sim = SimulationCore(seed=seed)
    action_rng = random.Random(seed)

    actions = []

    while True:
        state = sim.step(actions)

        actions = []
        for agent, agent_state in zip(sim.env.agents, state["observations"]):
            action = action_decision(agent_state, action_rng)
            actions.append((agent.agent_id, action))

        if state["num_agents"] == 0 or sim.env.time > 3000:
            return {
                "seed": seed,
                "score": state["score"],
                "time": sim.env.time,
            }


def main():
    results = []

    for i, seed in enumerate(SEEDS, start=1):
        result = run_simulation(seed)
        results.append(result)

        print(
            f"[{i:02d}/{len(SEEDS)}] "
            f"seed={result['seed']} "
            f"score={result['score']:.4f} "
            f"time={result['time']:.1f}s"
        )

    scores = [r["score"] for r in results]
    times = [r["time"] for r in results]

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(f"Runs:          {len(results)}")
    print(f"Mean score:    {statistics.mean(scores):.4f}")
    print(f"Median score:  {statistics.median(scores):.4f}")
    print(f"Min score:     {min(scores):.4f}")
    print(f"Max score:     {max(scores):.4f}")
    print(f"Score stddev:  {statistics.stdev(scores):.4f}")

    print()
    print(f"Mean survival:   {statistics.mean(times):.2f}s")
    print(f"Median survival: {statistics.median(times):.2f}s")
    print(f"Min survival:    {min(times):.2f}s")
    print(f"Max survival:    {max(times):.2f}s")


if __name__ == "__main__":
    main()
