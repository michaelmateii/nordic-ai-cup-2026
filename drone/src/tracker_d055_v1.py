from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from dtos import DroneFlybyPredictionDto


# ------------------------------------------------------------
# EXP-D055A
# Conservative temporal memory.
#
# Current detections are never removed just because of tracking.
# Memory predictions are added only for tracks with strong evidence.
# ------------------------------------------------------------

MATCH_IOU = 0.20
MATCH_DISTANCE_FACTOR = 1.0

# A single strong detection can start a useful track, but weaker
# detections need repeated confirmation before they are propagated.
STRONG_START_CONF = 0.60

# Memory is deliberately short for the first experiment.
MAX_MISSED_FRAMES = 4

# Confidence decay for a propagated detection.
MEMORY_DECAY = 0.82

# A remembered prediction must remain above this.
MEMORY_MIN_CONF = 0.10

# Never propagate a weak one-frame hallucination.
MIN_HITS_FOR_MEMORY = 2

# Final scorer applies no NMS, so do one class-agnostically here.
REPORT_NMS_IOU = 0.55

# EMA factor for measured box motion.
VELOCITY_ALPHA = 0.65


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)

    inter = iw * ih

    if inter <= 0:
        return 0.0

    aa = max(0.0, ax2 - ax1) * max(
        0.0,
        ay2 - ay1,
    )

    bb = max(0.0, bx2 - bx1) * max(
        0.0,
        by2 - by1,
    )

    union = aa + bb - inter

    if union <= 0:
        return 0.0

    return inter / union


def center(box):
    return (
        (box[0] + box[2]) / 2.0,
        (box[1] + box[3]) / 2.0,
    )


def size(box):
    return (
        max(1e-9, box[2] - box[0]),
        max(1e-9, box[3] - box[1]),
    )


def clip_box(box):
    x1, y1, x2, y2 = box

    x1 = max(0.0, min(1.0, x1))
    y1 = max(0.0, min(1.0, y1))
    x2 = max(0.0, min(1.0, x2))
    y2 = max(0.0, min(1.0, y2))

    if x2 <= x1 or y2 <= y1:
        return None

    return [x1, y1, x2, y2]


@dataclass
class Track:
    bbox: List[float]
    frame: int

    best_conf: float
    last_conf: float

    hits: int = 1
    misses: int = 0

    vx: float = 0.0
    vy: float = 0.0
    vw: float = 0.0
    vh: float = 0.0

    votes: Dict[str, float] = field(
        default_factory=dict
    )

    @property
    def class_name(self):
        return max(
            self.votes.items(),
            key=lambda kv: kv[1],
        )[0]

    def predict_box(self, frame: int):
        steps = max(
            0,
            int(frame) - int(self.frame),
        )

        if steps == 0:
            return list(self.bbox)

        cx, cy = center(self.bbox)
        w, h = size(self.bbox)

        cx += self.vx * steps
        cy += self.vy * steps
        w = max(
            1e-6,
            w + self.vw * steps,
        )
        h = max(
            1e-6,
            h + self.vh * steps,
        )

        return [
            cx - w / 2.0,
            cy - h / 2.0,
            cx + w / 2.0,
            cy + h / 2.0,
        ]


class SequenceTracker:
    def __init__(self):
        self.tracks: List[Track] = []

    def reset(self):
        self.tracks.clear()

    def _match_score(
        self,
        track_box,
        det_box,
    ):
        overlap = iou(
            track_box,
            det_box,
        )

        tcx, tcy = center(track_box)
        dcx, dcy = center(det_box)

        tw, th = size(track_box)
        dw, dh = size(det_box)

        dx = dcx - tcx
        dy = dcy - tcy

        dist = (
            dx * dx
            + dy * dy
        ) ** 0.5

        scale = max(
            tw,
            th,
            dw,
            dh,
            0.005,
        )

        if (
            overlap < MATCH_IOU
            and dist
            > MATCH_DISTANCE_FACTOR * scale
        ):
            return None

        # Prefer overlap; distance provides fallback for
        # fast-moving small objects.
        return (
            overlap
            - 0.10 * (dist / scale)
        )

    def update(
        self,
        frame: int,
        detections: List[
            DroneFlybyPredictionDto
        ],
    ):
        frame = int(frame)

        det_rows = []

        for ann in detections:
            det_rows.append(
                {
                    "class_name":
                        str(ann.object_id),

                    "bbox":
                        [
                            float(x)
                            for x in ann.bbox
                        ],

                    "confidence":
                        float(
                            ann.confidence
                        ),
                }
            )

        # --------------------------------------------------
        # Predict each existing track to current frame.
        # --------------------------------------------------

        predicted = [
            t.predict_box(frame)
            for t in self.tracks
        ]

        # --------------------------------------------------
        # Build candidate associations.
        #
        # Matching is class-agnostic because our detector can
        # disagree between e.g. plane subclasses on the same
        # physical object. Class identity is stabilized through
        # accumulated confidence votes.
        # --------------------------------------------------

        pairs = []

        for ti, track in enumerate(
            self.tracks
        ):
            for di, det in enumerate(
                det_rows
            ):
                score = self._match_score(
                    predicted[ti],
                    det["bbox"],
                )

                if score is None:
                    continue

                # Small preference for same-class matches.
                if (
                    track.class_name
                    == det["class_name"]
                ):
                    score += 0.05

                # Prefer strong detections when geometry ties.
                score += (
                    0.01
                    * det["confidence"]
                )

                pairs.append(
                    (
                        score,
                        ti,
                        di,
                    )
                )

        pairs.sort(
            reverse=True
        )

        matched_tracks = set()
        matched_dets = set()

        # --------------------------------------------------
        # Apply best one-to-one associations.
        # --------------------------------------------------

        for _, ti, di in pairs:

            if ti in matched_tracks:
                continue

            if di in matched_dets:
                continue

            track = self.tracks[ti]
            det = det_rows[di]

            old_cx, old_cy = center(
                track.bbox
            )
            old_w, old_h = size(
                track.bbox
            )

            new_cx, new_cy = center(
                det["bbox"]
            )
            new_w, new_h = size(
                det["bbox"]
            )

            dt = max(
                1,
                frame - track.frame,
            )

            measured_vx = (
                new_cx - old_cx
            ) / dt

            measured_vy = (
                new_cy - old_cy
            ) / dt

            measured_vw = (
                new_w - old_w
            ) / dt

            measured_vh = (
                new_h - old_h
            ) / dt

            a = VELOCITY_ALPHA

            track.vx = (
                (1.0 - a) * track.vx
                + a * measured_vx
            )

            track.vy = (
                (1.0 - a) * track.vy
                + a * measured_vy
            )

            track.vw = (
                (1.0 - a) * track.vw
                + a * measured_vw
            )

            track.vh = (
                (1.0 - a) * track.vh
                + a * measured_vh
            )

            track.bbox = list(
                det["bbox"]
            )

            track.frame = frame

            conf = det["confidence"]

            track.last_conf = conf

            track.best_conf = max(
                track.best_conf,
                conf,
            )

            track.votes[
                det["class_name"]
            ] = (
                track.votes.get(
                    det["class_name"],
                    0.0,
                )
                + conf
            )

            track.hits += 1
            track.misses = 0

            matched_tracks.add(ti)
            matched_dets.add(di)

        # --------------------------------------------------
        # Existing tracks missed this frame.
        # --------------------------------------------------

        for ti, track in enumerate(
            self.tracks
        ):
            if ti in matched_tracks:
                continue

            track.bbox = (
                track.predict_box(frame)
            )

            track.frame = frame
            track.misses += 1

        # --------------------------------------------------
        # New detections create tracks.
        # --------------------------------------------------

        for di, det in enumerate(
            det_rows
        ):
            if di in matched_dets:
                continue

            conf = det["confidence"]

            self.tracks.append(
                Track(
                    bbox=list(
                        det["bbox"]
                    ),
                    frame=frame,
                    best_conf=conf,
                    last_conf=conf,
                    votes={
                        det["class_name"]:
                            conf
                    },
                )
            )

        # --------------------------------------------------
        # Prune tracks.
        # --------------------------------------------------

        surviving = []

        for track in self.tracks:

            box = clip_box(
                track.bbox
            )

            if box is None:
                continue

            track.bbox = box

            if (
                track.misses
                > MAX_MISSED_FRAMES
            ):
                continue

            # One weak detection is never worth retaining.
            if (
                track.hits == 1
                and track.best_conf
                < STRONG_START_CONF
                and track.misses > 0
            ):
                continue

            surviving.append(
                track
            )

        self.tracks = surviving

    def memory_predictions(
        self,
    ):
        """
        Return only predictions that are being supplied by memory.

        Current-frame detector results are handled separately by the
        predictor and always take priority.
        """

        out = []

        for track in self.tracks:

            # Only missed tracks need memory output.
            if track.misses <= 0:
                continue

            if (
                track.hits
                < MIN_HITS_FOR_MEMORY
                and track.best_conf
                < STRONG_START_CONF
            ):
                continue

            conf = (
                track.best_conf
                * (
                    MEMORY_DECAY
                    ** track.misses
                )
            )

            if conf < MEMORY_MIN_CONF:
                continue

            box = clip_box(
                track.bbox
            )

            if box is None:
                continue

            out.append(
                DroneFlybyPredictionDto(
                    object_id=
                        track.class_name,

                    bbox=box,

                    confidence=float(
                        min(
                            1.0,
                            max(
                                0.0,
                                conf,
                            ),
                        )
                    ),
                )
            )

        return out


def final_nms(
    annotations:
        List[DroneFlybyPredictionDto],
):
    """
    Class-agnostic output NMS.

    Important because the evaluator does not perform NMS itself.
    """

    rows = sorted(
        annotations,
        key=lambda a:
            float(a.confidence),
        reverse=True,
    )

    kept = []

    for ann in rows:

        box = [
            float(v)
            for v in ann.bbox
        ]

        if all(
            iou(
                box,
                [
                    float(v)
                    for v in other.bbox
                ],
            )
            < REPORT_NMS_IOU

            for other in kept
        ):
            kept.append(
                ann
            )

    return kept[:500]
