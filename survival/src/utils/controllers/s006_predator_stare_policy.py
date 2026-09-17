import math
import random

from src.utils.DTOs import ActionRequest


# Per-agent memory.
AGENT_MEMORY = {}


def reset_policy_state():
    AGENT_MEMORY.clear()


STUCK_STEPS = 12
MIN_PROGRESS = 2.0


def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def normalize_angle(angle):
    while angle > math.pi:
        angle -= 2 * math.pi

    while angle < -math.pi:
        angle += 2 * math.pi

    return angle


def get_memory(agent_id):
    if agent_id not in AGENT_MEMORY:
        AGENT_MEMORY[agent_id] = {
            "last_fruit_distance": None,
            "stuck_counter": 0,
            "escape_steps": 0,
            "escape_direction": 0.0,
        }

    return AGENT_MEMORY[agent_id]


def action_decision(
    observation_response: dict,
    rng: random.Random,
):
    """
    EXP-S006

    S005 + predator-facing escape strategy.

    Move away from predators while facing them at exploitable
    distances to influence predator chase behaviour.
    """

    agent_id = observation_response["agent_id"]
    observations = observation_response["observations"]

    energy = observation_response["energy"]
    speed = observation_response["speed"]
    sprint_speed = observation_response["sprint_speed"]
    max_energy = observation_response["max_energy"]

    memory = get_memory(agent_id)

    fruits = [
        obs
        for obs in observations
        if obs["type"] == "Fruit"
    ]

    predators = [
        obs
        for obs in observations
        if obs["type"] == "Predator"
    ]

    move_distance = speed
    move_direction = 0.0
    turn_angle = 0.0

    # -------------------------------------------------
    # 1. Predator avoidance always has highest priority.
    # -------------------------------------------------
    if predators:
        nearest_predator = min(
            predators,
            key=lambda obs: obs["distance"],
        )

        predator_angle = nearest_predator["angle"]
        predator_distance = nearest_predator["distance"]

        # Move directly away from the predator.
        flee_angle = normalize_angle(
            predator_angle + math.pi
        )

        move_direction = flee_angle

        # Exploit predator behaviour:
        # while sufficiently far away, keep facing the predator
        # while moving away from it.
        if predator_distance > 95:
            move_distance = speed

            # Face toward predator, not away from it.
            turn_angle = clamp(
                predator_angle,
                -math.pi / 3,
                math.pi / 3,
            )

        else:
            # Inside the close-chase zone, create distance quickly.
            move_distance = sprint_speed

            # Still attempt to keep predator in front.
            turn_angle = clamp(
                predator_angle,
                -math.pi / 2,
                math.pi / 2,
            )

        memory["stuck_counter"] = 0
        memory["last_fruit_distance"] = None

    # -------------------------------------------------
    # 2. Escape mode.
    # -------------------------------------------------
    elif memory["escape_steps"] > 0:
        move_direction = memory["escape_direction"]

        turn_angle = clamp(
            memory["escape_direction"],
            -math.pi / 4,
            math.pi / 4,
        )

        memory["escape_steps"] -= 1

    # -------------------------------------------------
    # 3. Fruit seeking.
    # -------------------------------------------------
    elif fruits:
        target = min(
            fruits,
            key=lambda obs: obs["distance"],
        )

        target_distance = target["distance"]
        target_angle = target["angle"]

        previous_distance = memory["last_fruit_distance"]

        if previous_distance is not None:
            progress = previous_distance - target_distance

            if progress < MIN_PROGRESS:
                memory["stuck_counter"] += 1
            else:
                memory["stuck_counter"] = 0

        memory["last_fruit_distance"] = target_distance

        # If we're repeatedly failing to get closer,
        # abandon the current trajectory temporarily.
        if memory["stuck_counter"] >= STUCK_STEPS:
            escape_direction = rng.choice([-1.0, 1.0]) * (
                math.pi / 2
            )

            memory["escape_direction"] = escape_direction
            memory["escape_steps"] = 15
            memory["stuck_counter"] = 0
            memory["last_fruit_distance"] = None

            move_direction = escape_direction
            turn_angle = clamp(
                escape_direction,
                -math.pi / 4,
                math.pi / 4,
            )

        else:
            move_direction = target_angle

            turn_angle = clamp(
                target_angle,
                -math.pi / 6,
                math.pi / 6,
            )

    # -------------------------------------------------
    # 4. Exploration when no food is sensed.
    # -------------------------------------------------
    else:
        memory["last_fruit_distance"] = None
        memory["stuck_counter"] = 0

        # Stronger exploration than S002/S003.
        if rng.random() < 0.15:
            turn_angle = rng.uniform(
                -math.pi / 3,
                math.pi / 3,
            )

    # -------------------------------------------------
    # Survival-first reproduction.
    #
    # Score depends mainly on keeping the lineage alive,
    # not on maintaining a large population.
    #
    # Reproduce mainly when:
    #   1. the agent is getting old, or
    #   2. it has a very large energy surplus.
    # -------------------------------------------------

    age = observation_response["age"]

    old_age_threshold = 55.0

    high_energy_threshold = max(
        400.0,
        0.85 * max_energy,
    )

    spawn_agent = (
        (
            age >= old_age_threshold
            and energy >= 180.0
        )
        or
        (
            energy >= high_energy_threshold
        )
    )

    return ActionRequest(
        agent_id=agent_id,
        move_distance=move_distance,
        move_direction=move_direction,
        turn_angle=turn_angle,
        spawn_agent=spawn_agent,
    )
