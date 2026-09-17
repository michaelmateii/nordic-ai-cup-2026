import random

from src.utils.DTOs import ActionRequest


def action_decision(observation_response: dict, rng: random.Random):
    """
    S001:
    - Walk at normal speed
    - Move straight ahead
    - Do not waste energy turning
    - Reproduce only with a large energy reserve
    """

    agent_id = observation_response["agent_id"]
    energy = observation_response["energy"]
    speed = observation_response["speed"]
    max_energy = observation_response["max_energy"]

    # Cheap normal movement only; never sprint.
    move_distance = speed

    # Relative to current facing direction.
    move_direction = 0.0

    # No arbitrary turning.
    turn_angle = 0.0

    # Spawn only when we have enough energy to pay 100
    # and still retain a substantial reserve.
    spawn_threshold = max(250.0, 0.65 * max_energy)
    spawn_agent = energy >= spawn_threshold

    return ActionRequest(
        agent_id=agent_id,
        move_distance=move_distance,
        move_direction=move_direction,
        turn_angle=turn_angle,
        spawn_agent=spawn_agent,
    )
