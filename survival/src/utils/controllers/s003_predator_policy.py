import math
import random

from src.utils.DTOs import ActionRequest


def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def normalize_angle(angle):
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


def action_decision(
    observation_response: dict,
    rng: random.Random,
):
    """
    EXP-S003

    Strategy:
    - Preserve S002 fruit seeking.
    - Prioritize predator avoidance over food.
    - Flee directly away from nearest predator.
    - Sprint only when predator is close.
    - Otherwise use normal walking speed.
    """

    agent_id = observation_response["agent_id"]
    observations = observation_response["observations"]

    energy = observation_response["energy"]
    speed = observation_response["speed"]
    sprint_speed = observation_response["sprint_speed"]
    max_energy = observation_response["max_energy"]

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

    if predators:
        nearest_predator = min(
            predators,
            key=lambda obs: obs["distance"],
        )

        predator_angle = nearest_predator["angle"]
        predator_distance = nearest_predator["distance"]

        # Move in the direction opposite the predator.
        flee_angle = normalize_angle(
            predator_angle + math.pi
        )

        move_direction = flee_angle

        # Look away / orient travel in flee direction.
        turn_angle = clamp(
            flee_angle,
            -math.pi / 6,
            math.pi / 6,
        )

        # Sprint only for immediate danger.
        if predator_distance < 100:
            move_distance = sprint_speed
        else:
            move_distance = speed

    elif fruits:
        target = min(
            fruits,
            key=lambda obs: obs["distance"],
        )

        target_angle = target["angle"]

        move_direction = target_angle

        turn_angle = clamp(
            target_angle,
            -math.pi / 6,
            math.pi / 6,
        )

    else:
        # Controlled exploration.
        if rng.random() < 0.08:
            turn_angle = rng.uniform(
                -math.pi / 6,
                math.pi / 6,
            )

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
