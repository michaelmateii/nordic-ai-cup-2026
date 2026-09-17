import math
import random

from src.utils.DTOs import ActionRequest


def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def action_decision(
    observation_response: dict,
    rng: random.Random,
):
    """
    EXP-S002

    Strategy:
    - Preserve S001 energy discipline.
    - Seek the nearest detected fruit.
    - Walk at normal speed only.
    - Use small occasional turns when no fruit is detected.
    - Ignore predators for now so we isolate the value of food seeking.
    """

    agent_id = observation_response["agent_id"]
    observations = observation_response["observations"]

    energy = observation_response["energy"]
    speed = observation_response["speed"]
    max_energy = observation_response["max_energy"]

    fruits = [
        observation
        for observation in observations
        if observation["type"] == "Fruit"
    ]

    # Default low-cost behaviour.
    move_distance = speed
    move_direction = 0.0
    turn_angle = 0.0

    if fruits:
        # Select nearest detected fruit.
        target = min(
            fruits,
            key=lambda observation: observation["distance"],
        )

        target_angle = target["angle"]

        # Movement direction is relative to current heading,
        # so the observed relative fruit angle can be used directly.
        move_direction = target_angle

        # Gradually rotate the agent's view toward the fruit.
        turn_angle = clamp(
            target_angle,
            -math.pi / 6,
            math.pi / 6,
        )

    else:
        # Cheap exploration.
        # Do not jitter continuously; occasionally alter heading.
        if rng.random() < 0.08:
            turn_angle = rng.uniform(
                -math.pi / 6,
                math.pi / 6,
            )

    # Same conservative reproduction rule as S001.
    spawn_threshold = max(
        250.0,
        0.65 * max_energy,
    )

    spawn_agent = energy >= spawn_threshold

    return ActionRequest(
        agent_id=agent_id,
        move_distance=move_distance,
        move_direction=move_direction,
        turn_angle=turn_angle,
        spawn_agent=spawn_agent,
    )
