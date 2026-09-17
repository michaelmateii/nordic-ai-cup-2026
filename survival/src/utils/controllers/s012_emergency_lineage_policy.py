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
    EXP-S012

    S007 + emergency population-aware lineage preservation.

    Keep S007 reproduction completely unchanged during normal conditions.
    Only intervene when the global population falls to one or two agents.
    """

    agent_id = observation_response["agent_id"]
    observations = observation_response["observations"]

    energy = observation_response["energy"]
    speed = observation_response["speed"]
    sprint_speed = observation_response["sprint_speed"]
    max_energy = observation_response["max_energy"]
    population_size = observation_response.get(
    "_population_size",
    999,
)

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

    nearest_predator = (
        min(
            predators,
            key=lambda obs: obs["distance"],
        )
        if predators
        else None
    )

    nearest_fruit = (
        min(
            fruits,
            key=lambda obs: obs["distance"],
        )
        if fruits
        else None
    )

    # -------------------------------------------------
    # 1. Immediate danger
    # -------------------------------------------------

    if (
        nearest_predator is not None
        and nearest_predator["distance"] <= 95
    ):
        predator_angle = nearest_predator["angle"]

        flee_angle = normalize_angle(
            predator_angle + math.pi
        )

        move_direction = flee_angle
        move_distance = sprint_speed

        # Keep predator roughly in front while escaping.
        turn_angle = clamp(
            predator_angle,
            -math.pi / 2,
            math.pi / 2,
        )

        memory["stuck_counter"] = 0
        memory["last_fruit_distance"] = None

    # -------------------------------------------------
    # 2. Conditional predator manipulation
    # -------------------------------------------------

    elif (
        nearest_predator is not None
        and nearest_predator["distance"] <= 180
        and energy >= 0.55 * max_energy
    ):
        predator_angle = nearest_predator["angle"]

        flee_angle = normalize_angle(
            predator_angle + math.pi
        )

        # Walk away...
        move_direction = flee_angle
        move_distance = speed

        # ...while facing predator.
        turn_angle = clamp(
            predator_angle,
            -math.pi / 3,
            math.pi / 3,
        )

        memory["stuck_counter"] = 0
        memory["last_fruit_distance"] = None

    # -------------------------------------------------
    # 3. Escape mode from stuck detection
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
    # 4. Food seeking
    #
    # Distant predators are deliberately ignored here.
    # -------------------------------------------------

    elif nearest_fruit is not None:
        target_distance = nearest_fruit["distance"]
        target_angle = nearest_fruit["angle"]

        previous_distance = memory["last_fruit_distance"]

        if previous_distance is not None:
            progress = (
                previous_distance
                - target_distance
            )

            if progress < MIN_PROGRESS:
                memory["stuck_counter"] += 1
            else:
                memory["stuck_counter"] = 0

        memory["last_fruit_distance"] = (
            target_distance
        )

        if memory["stuck_counter"] >= STUCK_STEPS:
            escape_direction = (
                rng.choice([-1.0, 1.0])
                * math.pi / 2
            )

            memory["escape_direction"] = (
                escape_direction
            )
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
    # 5. Exploration
    # -------------------------------------------------

    else:
        memory["last_fruit_distance"] = None
        memory["stuck_counter"] = 0

        if rng.random() < 0.15:
            turn_angle = rng.uniform(
                -math.pi / 3,
                math.pi / 3,
            )

    # -------------------------------------------------
    # Base S007 reproduction.
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

    # -------------------------------------------------
    # Emergency lineage preservation.
    #
    # Do not change normal reproduction behavior.
    # Only lower the threshold when extinction risk is high.
    # -------------------------------------------------

    if population_size == 1:
        spawn_agent = (
            spawn_agent
            or (
                energy >= 200.0
                and age >= 25.0
            )
        )

    elif population_size == 2:
        spawn_agent = (
            spawn_agent
            or (
                energy >= 230.0
                and age >= 40.0
            )
        )

    return ActionRequest(
        agent_id=agent_id,
        move_distance=move_distance,
        move_direction=move_direction,
        turn_angle=turn_angle,
        spawn_agent=spawn_agent,
    )
