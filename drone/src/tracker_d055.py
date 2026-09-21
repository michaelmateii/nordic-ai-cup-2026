from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from dtos import DroneFlybyPredictionDto


# ============================================================
# EXP-D055A-v2
# Conservative temporal memory with ghost-track suppression.
# ============================================================

MATCH_IOU = 0.20
MATCH_DISTANCE_FACTOR = 1.00

# Track must have produced at least one strong observation.
STRONG_START_CONF = 0.60

# Require repeated observations before we ever propagate it.
MIN_HITS_FOR_MEMORY = 3

# Keep missing tracks only briefly.
MAX_MISSED_FRAMES = 3

# Propagated confidence drops quickly.
MEMORY_DECAY = 0.75
MEMORY_MIN_CONF = 0.10

# Current + memory output duplicate suppression.
REPORT_NMS_IOU = 0.55

# Motion estimate smoothing.
VELOCITY_ALPHA = 0.65

# If an unmatched old track sits where a current detection is,
# consider that old track explained and kill it.
STALE_DUP_IOU = 0.25
STALE_DUP_DISTANCE_FACTOR = 0.45

# Extra internal track deduplication.
TRACK_MERGE_IOU = 0.70


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

    if inter <= 0.0:
        return 0.0

    area_a = (
        max(0.0, ax2 - ax1)
        * max(0.0, ay2 - ay1)
    )

    area_b = (
        max(0.0, bx2 - bx1)
        * max(0.0, by2 - by1)
    )

    union = area_a + area_b - inter

    if union <= 0.0:
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


def normalized_distance(a, b):
    acx, acy = center(a)
    bcx, bcy = center(b)

    aw, ah = size(a)
    bw, bh = size(b)

    dist = (
        (acx - bcx) ** 2
        + (acy - bcy) ** 2
    ) ** 0.5

    scale = max(
        aw,
        ah,
        bw,
        bh,
        0.005,
    )

    return dist / scale


def clip_box(box):
    x1, y1, x2, y2 = [
        float(v)
        for v in box
    ]

    x1 = max(0.0, min(1.0, x1))
    y1 = max(0.0, min(1.0, y1))
    x2 = max(0.0, min(1.0, x2))
    y2 = max(0.0, min(1.0, y2))

    if x2 <= x1 or y2 <= y1:
        return None

    return [
        x1,
        y1,
        x2,
        y2,
    ]


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

    def predict_box(
        self,
        frame: int,
    ):
        steps = max(
            0,
            int(frame) - int(self.frame),
        )

        if steps == 0:
            return list(self.bbox)

        cx, cy = center(
            self.bbox
        )

        w, h = size(
            self.bbox
        )

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

        dist_norm = normalized_distance(
            track_box,
            det_box,
        )

        if (
            overlap < MATCH_IOU
            and dist_norm > MATCH_DISTANCE_FACTOR
        ):
            return None

        return (
            overlap
            - 0.10 * dist_norm
        )

    def update(
        self,
        frame: int,
        detections:
            List[DroneFlybyPredictionDto],
    ):
        frame = int(frame)

        # ----------------------------------------------------
        # Convert DTO detections into lightweight rows.
        # ----------------------------------------------------

        det_rows = []

        for ann in detections:
            det_rows.append(
                {
                    "class_name":
                        str(ann.object_id),

                    "bbox":
                        [
                            float(v)
                            for v in ann.bbox
                        ],

                    "confidence":
                        float(
                            ann.confidence
                        ),
                }
            )

        # ----------------------------------------------------
        # Predict all old tracks into this frame.
        # ----------------------------------------------------

        predicted = [
            track.predict_box(frame)
            for track in self.tracks
        ]

        # ----------------------------------------------------
        # Build class-agnostic association candidates.
        #
        # Class-agnostic is intentional:
        # the same physical object may flicker between
        # plane/tower/launcher subclasses.
        # ----------------------------------------------------

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

                if (
                    track.class_name
                    == det["class_name"]
                ):
                    score += 0.05

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

        # ----------------------------------------------------
        # Greedy one-to-one association.
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Missed old tracks.
        #
        # IMPORTANT v2 change:
        #
        # If ANY current detection already occupies the same
        # location, the unmatched old track is a likely ghost.
        # Kill it instead of propagating another copy.
        # ----------------------------------------------------

        tracks_to_drop = set()

        current_boxes = [
            det["bbox"]
            for det in det_rows
        ]

        for ti, track in enumerate(
            self.tracks
        ):
            if ti in matched_tracks:
                continue

            predicted_box = (
                track.predict_box(frame)
            )

            duplicate_of_current = False

            for det_box in current_boxes:

                overlap = iou(
                    predicted_box,
                    det_box,
                )

                dist_norm = (
                    normalized_distance(
                        predicted_box,
                        det_box,
                    )
                )

                if (
                    overlap >= STALE_DUP_IOU
                    or
                    dist_norm
                    <= STALE_DUP_DISTANCE_FACTOR
                ):
                    duplicate_of_current = True
                    break

            if duplicate_of_current:
                tracks_to_drop.add(ti)
                continue

            track.bbox = predicted_box
            track.frame = frame
            track.misses += 1

        # ----------------------------------------------------
        # Remove stale duplicates BEFORE adding new tracks.
        # ----------------------------------------------------

        if tracks_to_drop:
            self.tracks = [
                track
                for i, track in enumerate(
                    self.tracks
                )
                if i not in tracks_to_drop
            ]

        # ----------------------------------------------------
        # New unmatched detections create tracks.
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Prune dead / invalid tracks.
        # ----------------------------------------------------

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

            # One weak observation should disappear immediately
            # once it is no longer detected.
            if (
                track.hits == 1
                and track.misses > 0
                and track.best_conf
                < STRONG_START_CONF
            ):
                continue

            surviving.append(
                track
            )

        # ----------------------------------------------------
        # Internal track deduplication.
        #
        # Prevent hidden duplicate tracks from surviving until
        # future frames even if report-level NMS hides them now.
        # ----------------------------------------------------

        surviving.sort(
            key=lambda t: (
                -t.hits,
                t.misses,
                -t.best_conf,
            )
        )

        deduped = []

        for track in surviving:

            duplicate = False

            for other in deduped:
                if (
                    iou(
                        track.bbox,
                        other.bbox,
                    )
                    >= TRACK_MERGE_IOU
                ):
                    duplicate = True
                    break

            if not duplicate:
                deduped.append(
                    track
                )

        self.tracks = deduped

    def memory_predictions(self):
        """
        Return ONLY currently-missed tracks.

        Current-frame detector predictions are returned separately.
        """

        out = []

        for track in self.tracks:

            if track.misses <= 0:
                continue

            # Both conditions are now mandatory.
            if (
                track.hits
                < MIN_HITS_FOR_MEMORY
            ):
                continue

            if (
                track.best_conf
                < STRONG_START_CONF
            ):
                continue

            # Use the most recent observed confidence, not the
            # historical maximum, to avoid overconfident ghosts.
            conf = (
                track.last_conf
                * (
                    MEMORY_DECAY
                    ** track.misses
                )
            )

            if (
                conf
                < MEMORY_MIN_CONF
            ):
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
    Final class-agnostic report NMS.

    The evaluator does not suppress duplicates for us.
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

        duplicate = False

        for other in kept:

            other_box = [
                float(v)
                for v in other.bbox
            ]

            if (
                iou(
                    box,
                    other_box,
                )
                >= REPORT_NMS_IOU
            ):
                duplicate = True
                break

        if not duplicate:
            kept.append(
                ann
            )

    return kept[:500]
def merge_current_with_memory(
    current_annotations,
    memory_annotations,
):
    """
    EXP-D055A-v3.

    Preserve every current detector prediction.

    Memory is supplemental only:
    - never delete current detections
    - reject a remembered box if a current box already
      occupies approximately the same physical location
    - reject duplicate memory boxes

    This avoids the previous class-agnostic report NMS
    deleting legitimate crowded objects.
    """

    result = list(
        current_annotations
    )

    accepted_memory = []

    for memory in sorted(
        memory_annotations,
        key=lambda a:
            float(a.confidence),
        reverse=True,
    ):
        memory_box = [
            float(v)
            for v in memory.bbox
        ]

        explained = False

        # Any current observation at approximately the
        # same location supersedes memory.
        for current in current_annotations:

            current_box = [
                float(v)
                for v in current.bbox
            ]

            overlap = iou(
                memory_box,
                current_box,
            )

            dist_norm = (
                normalized_distance(
                    memory_box,
                    current_box,
                )
            )

            if (
                overlap >= 0.25
                or dist_norm <= 0.45
            ):
                explained = True
                break

        if explained:
            continue

        # Also prevent two remembered tracks from reporting
        # essentially the same object.
        for other in accepted_memory:

            other_box = [
                float(v)
                for v in other.bbox
            ]

            if (
                iou(
                    memory_box,
                    other_box,
                )
                >= 0.50
            ):
                explained = True
                break

        if explained:
            continue

        accepted_memory.append(
            memory
        )

    result.extend(
        accepted_memory
    )

    return result[:500]
