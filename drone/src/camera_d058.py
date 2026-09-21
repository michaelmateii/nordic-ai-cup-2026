import math
from dataclasses import dataclass
from typing import Optional

from dtos import RequestedViewDto


L1_MIN_X = 960
L1_MAX_X = 2880
L1_MIN_Y = 540
L1_MAX_Y = 1620

L0_X = 1920
L0_Y = 1080

# From evaluator rules / Thomas implementation.
MAX_DELTA = {
    0: 2203.0,
    1: 1102.0,
    2: 551.0,
}

ALLOWED_FROM = {
    0: {0, 1},
    1: {0, 1, 2},
    2: {1, 2},
}

L1_TILES = [
    (960, 540),
    (2880, 540),
    (2880, 1620),
    (960, 1620),
]


@dataclass
class CameraPlan:
    phase: str = "init"
    tile_index: int = 0
    sweep_dir: int = 1

    # Our estimate of the actual camera state.
    cam_level: int = 0
    cam_cx: int = 1920
    cam_cy: int = 1080

    pending: Optional[tuple] = None
    pending_frame: int = -1


_PLANS = {}


def get_plan(sequence_id, frame_index):
    sid = str(sequence_id)

    if frame_index == 0:
        _PLANS.pop(sid, None)

    if sid not in _PLANS:
        _PLANS[sid] = CameraPlan()

    return _PLANS[sid]


def _sync(request, plan):
    v = request.view
    fb = getattr(
        request,
        "camera_command_feedback",
        None,
    )

    # Sequence start: trust actual request view.
    if request.frame_index == 0:
        plan.cam_level = int(v.resolution_level)
        plan.cam_cx = int(v.center_x)
        plan.cam_cy = int(v.center_y)
        plan.pending = None
        return

    # Thomas's important stale-view fix:
    # if our previous command was not explicitly rejected,
    # treat it as the true camera state even if this request
    # contains an older rendered view.
    if plan.pending is not None:
        refused = (
            fb is not None
            and int(fb.frame)
            == int(plan.pending_frame)
        )

        if not refused:
            (
                plan.cam_level,
                plan.cam_cx,
                plan.cam_cy,
            ) = plan.pending

        plan.pending = None


def _step_towards(cx, cy, tx, ty, max_step):
    dx = tx - cx
    dy = ty - cy

    d = math.hypot(dx, dy)

    if d <= max_step:
        return tx, ty

    s = max_step / d

    return (
        cx + dx * s,
        cy + dy * s,
    )


def _clamp_l1(x, y):
    x = int(
        min(
            max(round(x), L1_MIN_X),
            L1_MAX_X,
        )
    )

    y = int(
        min(
            max(round(y), L1_MIN_Y),
            L1_MAX_Y,
        )
    )

    return x, y


def choose_next_view(request):
    plan = get_plan(
        request.sequence_id,
        request.frame_index,
    )

    _sync(
        request,
        plan,
    )

    level = int(plan.cam_level)
    cx = int(plan.cam_cx)
    cy = int(plan.cam_cy)

    allowed = ALLOWED_FROM.get(
        level,
        set(),
    )

    limit = (
        MAX_DELTA[level]
        * 0.98
    )

    # -----------------------------------------
    # Initial transition L0 -> first L1 tile.
    # -----------------------------------------

    if plan.phase == "init":
        if 1 not in allowed:
            return None

        plan.phase = "tiles"
        plan.tile_index = 0

        tx, ty = L1_TILES[0]

        x, y = _step_towards(
            cx,
            cy,
            tx,
            ty,
            limit,
        )

        x, y = _clamp_l1(
            x,
            y,
        )

        req = RequestedViewDto(
            resolution_level=1,
            center_x=int(x),
            center_y=int(y),
        )

        plan.pending = (
            1,
            int(x),
            int(y),
        )

        plan.pending_frame = int(
            request.frame
        )

        return req

    # -----------------------------------------
    # Visit the 4 L1 tiles once.
    # -----------------------------------------

    if plan.phase == "tiles":

        tx, ty = L1_TILES[
            plan.tile_index
        ]

        if (
            level == 1
            and abs(cx - tx) < 5
            and abs(cy - ty) < 5
        ):
            plan.tile_index += 1

            if plan.tile_index >= len(
                L1_TILES
            ):
                plan.phase = "sweep"
            else:
                tx, ty = L1_TILES[
                    plan.tile_index
                ]

        if plan.phase == "tiles":

            x, y = _step_towards(
                cx,
                cy,
                tx,
                ty,
                limit,
            )

            x, y = _clamp_l1(
                x,
                y,
            )

            req = RequestedViewDto(
                resolution_level=1,
                center_x=int(x),
                center_y=int(y),
            )

            plan.pending = (
                1,
                int(x),
                int(y),
            )

            plan.pending_frame = int(
                request.frame
            )

            return req

    # -----------------------------------------
    # Continuous top-band L1 sweep.
    #
    # Objects enter the scene near the top,
    # so keep y at the legal top edge.
    # -----------------------------------------

    if level != 1:
        if 1 not in allowed:
            return None

        x, y = _step_towards(
            cx,
            cy,
            cx,
            L1_MIN_Y,
            limit,
        )

        x, y = _clamp_l1(
            x,
            y,
        )

    else:

        if (
            plan.sweep_dir > 0
            and cx >= L1_MAX_X - 2
        ):
            plan.sweep_dir = -1

        elif (
            plan.sweep_dir < 0
            and cx <= L1_MIN_X + 2
        ):
            plan.sweep_dir = 1

        target_x = (
            cx
            + plan.sweep_dir
            * int(MAX_DELTA[1] * 0.98)
        )

        target_x = min(
            max(
                target_x,
                L1_MIN_X,
            ),
            L1_MAX_X,
        )

        x, y = _step_towards(
            cx,
            cy,
            target_x,
            L1_MIN_Y,
            MAX_DELTA[1] * 0.98,
        )

        x, y = _clamp_l1(
            x,
            y,
        )

    # If we're effectively already there,
    # holding is safer than emitting duplicates.
    if (
        level == 1
        and abs(cx - x) < 2
        and abs(cy - y) < 2
    ):
        return None

    req = RequestedViewDto(
        resolution_level=1,
        center_x=int(x),
        center_y=int(y),
    )

    plan.pending = (
        1,
        int(x),
        int(y),
    )

    plan.pending_frame = int(
        request.frame
    )

    return req
