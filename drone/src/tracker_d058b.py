import math
from dataclasses import dataclass, field

import numpy as np

from dtos import DroneFlybyPredictionDto


# Thomas-style fitted source-frame motion.
MOTION_H = np.array(
    [
        [1.006756, -0.001473, -12.614238],
        [0.000337,  1.011899,  51.706644],
        [0.0,      -0.000001,  1.0],
    ],
    dtype=np.float64,
)

SOURCE_W = 3840.0
SOURCE_H = 2160.0

MATCH_IOU = 0.25
MATCH_DIST_FRAC = 0.90

MIN_HITS_FOR_MEMORY = 3
STRONG_START_CONF = 0.60
MEMORY_MIN_CONF = 0.03

OFFVIEW_DECAY = 0.995
INVIEW_DECAY = 0.85

DUPLICATE_IOU = 0.55
MERGE_IOU = 0.55


def iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    inter = (
        max(0.0, x2 - x1)
        * max(0.0, y2 - y1)
    )

    if inter <= 0:
        return 0.0

    aa = (
        max(0.0, a[2] - a[0])
        * max(0.0, a[3] - a[1])
    )

    bb = (
        max(0.0, b[2] - b[0])
        * max(0.0, b[3] - b[1])
    )

    return inter / max(
        1e-9,
        aa + bb - inter,
    )


def warp_point(x, y):
    v = MOTION_H @ np.array(
        [x, y, 1.0],
        dtype=np.float64,
    )

    return (
        float(v[0] / v[2]),
        float(v[1] / v[2]),
    )


def advance_box(box, steps):
    x1, y1, x2, y2 = box

    for _ in range(max(0, steps)):
        x1, y1 = warp_point(x1, y1)
        x2, y2 = warp_point(x2, y2)

    return (
        x1,
        y1,
        x2,
        y2,
    )


@dataclass
class Track:
    box: tuple
    frame: int
    confidence: float

    votes: dict = field(
        default_factory=dict
    )

    hits: int = 1
    strong_seen: bool = False
    misses_in_view: int = 0

    def cls(self):
        return max(
            self.votes.items(),
            key=lambda kv: kv[1],
        )[0]

    def predict_to(self, frame):
        steps = int(frame) - int(self.frame)

        if steps > 0:
            self.box = advance_box(
                self.box,
                steps,
            )

            self.frame = int(frame)


class SequenceTracker:
    def __init__(self):
        self.tracks = []

        self.last_region = (
            0.0,
            0.0,
            SOURCE_W,
            SOURCE_H,
        )

    def update(
        self,
        frame,
        detections,
        source_region_xyxy=None,
        original_width=3840,
        original_height=2160,
    ):
        frame = int(frame)

        ow = float(original_width)
        oh = float(original_height)

        if source_region_xyxy is not None:
            self.last_region = tuple(
                float(v)
                for v in source_region_xyxy
            )

        # First dead-reckon every existing track
        # to the current source frame.
        for t in self.tracks:
            t.predict_to(frame)

        # Remove tracks that have left the frame
        # or have decayed into irrelevance.
        alive = []

        for t in self.tracks:
            x1, y1, x2, y2 = t.box

            if (
                y1 > oh + 30
                or x2 < -30
                or x1 > ow + 30
                or t.confidence < 0.015
            ):
                continue

            alive.append(t)

        self.tracks = alive

        dets = []

        for ann in detections:
            x1, y1, x2, y2 = [
                float(v)
                for v in ann.bbox
            ]

            dets.append(
                (
                    str(ann.object_id),
                    float(ann.confidence),
                    (
                        x1 * ow,
                        y1 * oh,
                        x2 * ow,
                        y2 * oh,
                    ),
                )
            )

        # Candidate track/detection associations.
        pairs = []

        for ti, t in enumerate(
            self.tracks
        ):
            tx1, ty1, tx2, ty2 = t.box

            tcx = (tx1 + tx2) / 2
            tcy = (ty1 + ty2) / 2

            tw = max(1.0, tx2 - tx1)
            th = max(1.0, ty2 - ty1)

            for di, (_, conf, db) in enumerate(
                dets
            ):
                dx1, dy1, dx2, dy2 = db

                dcx = (dx1 + dx2) / 2
                dcy = (dy1 + dy2) / 2

                ov = iou(
                    t.box,
                    db,
                )

                dist = math.hypot(
                    dcx - tcx,
                    dcy - tcy,
                )

                if (
                    ov >= MATCH_IOU
                    or dist
                    < MATCH_DIST_FRAC
                    * max(
                        tw,
                        th,
                        20.0,
                    )
                ):
                    pairs.append(
                        (
                            ov
                            + 0.001
                            * conf,
                            ti,
                            di,
                        )
                    )

        pairs.sort(
            reverse=True
        )

        matched_tracks = set()
        used_dets = set()

        for _, ti, di in pairs:
            if (
                ti in matched_tracks
                or di in used_dets
            ):
                continue

            matched_tracks.add(ti)
            used_dets.add(di)

            t = self.tracks[ti]

            cls_name, conf, db = dets[di]

            # L1-ish correction strength:
            # retain motion prediction but pull toward
            # the new observation.
            alpha = 0.60

            t.box = tuple(
                (1.0 - alpha) * old
                + alpha * new
                for old, new in zip(
                    t.box,
                    db,
                )
            )

            t.votes[cls_name] = (
                t.votes.get(
                    cls_name,
                    0.0,
                )
                + conf * 2.0
            )

            t.confidence = max(
                conf,
                t.confidence * 0.92,
            )

            t.hits += 1

            if conf >= STRONG_START_CONF:
                t.strong_seen = True

            t.misses_in_view = 0

        # Decay unmatched tracks differently depending
        # on whether this camera crop should have seen them.
        rx1, ry1, rx2, ry2 = self.last_region

        for ti, t in enumerate(
            self.tracks
        ):
            if ti in matched_tracks:
                continue

            x1, y1, x2, y2 = t.box
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            inside = (
                rx1 <= cx <= rx2
                and ry1 <= cy <= ry2
            )

            if inside:
                t.confidence *= INVIEW_DECAY
                t.misses_in_view += 1
            else:
                # This is the key active-camera behaviour:
                # don't kill a track just because we're
                # currently looking somewhere else.
                t.confidence *= OFFVIEW_DECAY

        # Unmatched detections start tracks.
        for di, (
            cls_name,
            conf,
            box,
        ) in enumerate(dets):
            if di in used_dets:
                continue

            self.tracks.append(
                Track(
                    box=box,
                    frame=frame,
                    confidence=conf,
                    votes={
                        cls_name: conf * 2.0
                    },
                    hits=1,
                    strong_seen=(
                        conf
                        >= STRONG_START_CONF
                    ),
                )
            )

        self._merge_duplicates()

    def _merge_duplicates(self):
        self.tracks.sort(
            key=lambda t:
                t.confidence,
            reverse=True,
        )

        kept = []

        for t in self.tracks:
            duplicate = None

            for k in kept:
                if (
                    iou(
                        t.box,
                        k.box,
                    )
                    >= DUPLICATE_IOU
                ):
                    duplicate = k
                    break

            if duplicate is None:
                kept.append(t)
                continue

            for cls_name, vote in (
                t.votes.items()
            ):
                duplicate.votes[
                    cls_name
                ] = (
                    duplicate.votes.get(
                        cls_name,
                        0.0,
                    )
                    + vote
                )

            duplicate.hits = max(
                duplicate.hits,
                t.hits,
            )

            duplicate.strong_seen = (
                duplicate.strong_seen
                or t.strong_seen
            )

            duplicate.confidence = max(
                duplicate.confidence,
                t.confidence,
            )

        self.tracks = kept

    def memory_predictions(self):
        out = []

        for t in self.tracks:
            if (
                t.hits
                < MIN_HITS_FOR_MEMORY
            ):
                continue

            if not t.strong_seen:
                continue

            if (
                t.confidence
                < MEMORY_MIN_CONF
            ):
                continue

            x1, y1, x2, y2 = t.box

            x1 = max(
                0.0,
                min(
                    SOURCE_W,
                    x1,
                ),
            )

            y1 = max(
                0.0,
                min(
                    SOURCE_H,
                    y1,
                ),
            )

            x2 = max(
                0.0,
                min(
                    SOURCE_W,
                    x2,
                ),
            )

            y2 = max(
                0.0,
                min(
                    SOURCE_H,
                    y2,
                ),
            )

            if (
                x2 <= x1
                or y2 <= y1
            ):
                continue

            out.append(
                DroneFlybyPredictionDto(
                    object_id=t.cls(),
                    bbox=[
                        x1 / SOURCE_W,
                        y1 / SOURCE_H,
                        x2 / SOURCE_W,
                        y2 / SOURCE_H,
                    ],
                    confidence=float(
                        min(
                            1.0,
                            max(
                                0.0,
                                t.confidence,
                            ),
                        )
                    ),
                )
            )

        return out


def merge_current_with_memory(
    current_annotations,
    memory_annotations,
):
    # Preserve every current detector output.
    result = list(
        current_annotations
    )

    # Only add memory when it does not duplicate
    # something currently reported.
    for mem in sorted(
        memory_annotations,
        key=lambda a:
            float(a.confidence),
        reverse=True,
    ):
        mb = [
            float(v)
            for v in mem.bbox
        ]

        duplicate = False

        for cur in result:
            cb = [
                float(v)
                for v in cur.bbox
            ]

            if iou(
                mb,
                cb,
            ) >= MERGE_IOU:
                duplicate = True
                break

        if not duplicate:
            result.append(mem)

    # Hard evaluator cap is 500.
    result.sort(
        key=lambda a:
            float(a.confidence),
        reverse=True,
    )

    return result[:500]
