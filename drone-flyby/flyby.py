"""Detector, object memory and camera policy for the drone flyby.

Three parts, one request at a time:

1. **Detect** with YOLO on the transmitted view and lift the boxes into
   source pixels.
2. **Remember.** Every object ever seen is kept as a track. The ground moves
   predictably between frames (see ``MOTION``), so each track is moved forward
   to the current frame, matched against the new detections, and reported for
   the whole frame even when the camera is looking elsewhere.
3. **Steer.** Sweep a 3x2 grid of Level-1 views, one step per answered
   frame; every move is checked with the evaluator's own rules first.

Configuration, all optional, through environment variables:

    DRONE_MODEL   path to the YOLO weights  (default: ~/models/drone-yolo11n-v1.pt)
    DRONE_DEVICE  torch device               (default: cpu)
    DRONE_IMGSZ   inference size             (default: 960)
    DRONE_DET_CONF    lowest detection reported at all       (default: 0.01)
    DRONE_TRACK_CONF  lowest detection remembered as a track (default: 0.25)
"""

import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from dtos import (
    FULL_FRAME_CENTER,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    OBJECT_CLASSES,
    DroneFlybyPredictionDto,
    DroneFlybyPredictRequestDto,
    DroneFlybyPredictResponseDto,
    RequestedViewDto,
)
from utils import clip_bbox_to_frame, decode_view, describe_camera_rejection

logger = logging.getLogger(__name__)

MODEL_PATH = Path(os.environ.get('DRONE_MODEL', Path.home() / 'models' / 'drone-yolo11n-v1.pt'))
DEVICE = os.environ.get('DRONE_DEVICE', 'cpu')
IMGSZ = int(os.environ.get('DRONE_IMGSZ', '960'))

# Two thresholds. mAP rewards ranked low-confidence guesses, so anything above
# DETECTION_CONFIDENCE is reported for the frame it was seen in. Only detections
# above NEW_TRACK_CONFIDENCE start a track that is remembered and reported on
# later frames; otherwise false alarms pile up (v1 reached 137 per frame).
DETECTION_CONFIDENCE = float(os.environ.get('DRONE_DET_CONF', '0.01'))
NEW_TRACK_CONFIDENCE = float(os.environ.get('DRONE_TRACK_CONF', '0.25'))
# One-frame guesses rank below remembered objects of the same confidence.
TRANSIENT_WEIGHT = 0.5
MAX_TRACKS = 120
NMS_IOU = 0.5

# Per-frame ground motion in source pixels, fitted on the Helsinki frames
# (residual under 1 px). The drone flies straight, so every point drifts down
# and slightly away from the centre as the ground gets closer:
#   dx = a + b*x + c*y,  dy = d + e*x + f*y
MOTION = (-13.62, 0.00708, 0.00007, 51.25, 0.00029, 0.01334)

# How much a detection at each level is trusted, for class votes and boxes.
LEVEL_WEIGHT = {0: 0.4, 1: 0.8, 2: 1.0}
MATCH_IOU = 0.2
# A track the camera looked at without finding it this many times is dropped.
MAX_MISSES = 3
# Confidence of a remembered track fades with every frame it goes unseen.
UNSEEN_DECAY = 0.97
# Also report a runner-up class when its vote is at least this share of the best.
RUNNER_UP_SHARE = 0.35
# Views whose edge is this close to a box do not count as missing it.
EDGE_MARGIN = 20

RESTART_GAP = 5
MAX_SEQUENCES = 8
# A detection this close to the view edge is probably cut off.
CUT_OFF_PIXELS = 3

# Level-1 sweep: every move is at most 1080 px, inside the 1102 px L1 limit.
SWEEP = [(960, 540), (1920, 540), (2880, 540), (2880, 1620), (1920, 1620), (960, 1620)]


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

_model = None
_model_lock = threading.Lock()


def load_model():
    """Load and warm up the detector once; None if the weights are missing."""
    global _model
    if _model is not None:
        return _model
    if not MODEL_PATH.exists():
        logger.error('No model at %s: answering with empty detections', MODEL_PATH)
        return None
    from ultralytics import YOLO

    model = YOLO(str(MODEL_PATH))
    # The first inference is the slow one; pay for it before the clock starts.
    model.predict(np.zeros((540, 960, 3), np.uint8), imgsz=IMGSZ, device=DEVICE, verbose=False)
    logger.info('Loaded %s on %s', MODEL_PATH, DEVICE)
    _model = model
    return _model


def detect(image: np.ndarray, source_region) -> List[Tuple[str, float, np.ndarray]]:
    """Run YOLO on one view; boxes come back in source pixels."""
    model = load_model()
    if model is None:
        return []
    with _model_lock:
        result = model.predict(
            image, imgsz=IMGSZ, device=DEVICE, conf=DETECTION_CONFIDENCE,
            iou=NMS_IOU, verbose=False,
        )[0]
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []

    rx1, ry1, rx2, ry2 = source_region
    height, width = image.shape[:2]
    scale = np.array([(rx2 - rx1) / width, (ry2 - ry1) / height] * 2)
    offset = np.array([rx1, ry1, rx1, ry1])
    xyxy = boxes.xyxy.cpu().numpy() * scale + offset
    names = [result.names[int(c)] for c in boxes.cls.cpu().numpy()]
    confidences = boxes.conf.cpu().numpy()
    return [
        (name, float(conf), box)
        for name, conf, box in zip(names, confidences, xyxy)
        if name in OBJECT_CLASSES
    ]


# --------------------------------------------------------------------------- #
# Memory
# --------------------------------------------------------------------------- #

def advance(box: np.ndarray, steps: int) -> np.ndarray:
    """Move a source-pixel box forward by ``steps`` frames of ground motion."""
    a, b, c, d, e, f = MOTION
    x1, y1, x2, y2 = box
    for _ in range(max(0, steps)):
        x1, y1 = x1 + a + b * x1 + c * y1, y1 + d + e * x1 + f * y1
        x2, y2 = x2 + a + b * x2 + c * y2, y2 + d + e * x2 + f * y2
    return np.array([x1, y1, x2, y2])


def iou(a: np.ndarray, b: np.ndarray) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Track:
    box: np.ndarray                 # source pixels at ``frame``
    frame: int
    votes: Dict[str, float] = field(default_factory=dict)
    best_confidence: float = 0.0
    last_seen: int = 0
    best_level: int = 0
    hits: int = 0
    misses: int = 0

    def label(self) -> Tuple[str, float]:
        name = max(self.votes, key=self.votes.get)
        return name, self.votes[name]


@dataclass
class Sequence:
    tracks: List[Track] = field(default_factory=list)
    sweep_index: int = 0
    last_frame: int = -1


_sequences: Dict[str, Sequence] = {}
_sequences_lock = threading.Lock()


def update_tracks(state: Sequence, frame: int, level: int, region, detections) -> list:
    """Fold this view's detections into memory; return the unremembered ones."""
    # Bring every track to this frame; forget the ones that left the ground.
    alive = []
    for track in state.tracks:
        track.box = advance(track.box, frame - track.frame)
        track.frame = frame
        x1, y1, x2, y2 = track.box
        if x2 > 0 and y2 > 0 and x1 < IMAGE_WIDTH and y1 < IMAGE_HEIGHT:
            alive.append(track)
    state.tracks = alive

    weight = LEVEL_WEIGHT[level]
    rx1, ry1, rx2, ry2 = region
    cut_off = CUT_OFF_PIXELS * (rx2 - rx1) / 960
    matched = set()
    transient = []
    for name, confidence, box in sorted(detections, key=lambda d: -d[1]):
        best, best_iou = None, MATCH_IOU
        for track in state.tracks:
            overlap = iou(track.box, box)
            if overlap > best_iou:
                best, best_iou = track, overlap
        if best is None:
            if confidence < NEW_TRACK_CONFIDENCE:
                transient.append((name, confidence * TRANSIENT_WEIGHT, box))
                continue
            best = Track(box=box, frame=frame, best_level=level)
            state.tracks.append(best)
        elif id(best) in matched:
            # A second detection of an object already handled this frame is a
            # class vote only; the first (most confident) one set the box.
            best.votes[name] = best.votes.get(name, 0.0) + weight * confidence
            continue
        else:
            # Finer views give better boxes; do not let a coarse look undo
            # them, nor a box the view edge has cut short.
            touches_edge = (box[0] <= rx1 + cut_off or box[1] <= ry1 + cut_off
                            or box[2] >= rx2 - cut_off or box[3] >= ry2 - cut_off)
            if touches_edge and not (box[0] <= cut_off or box[1] <= cut_off
                                     or box[2] >= IMAGE_WIDTH - cut_off or box[3] >= IMAGE_HEIGHT - cut_off):
                pass
            elif level >= best.best_level:
                best.box = box
                best.best_level = level
            else:
                best.box = 0.7 * best.box + 0.3 * box
        matched.add(id(best))
        best.votes[name] = best.votes.get(name, 0.0) + weight * confidence
        best.best_confidence = max(best.best_confidence, confidence * (0.6 + 0.4 * weight))
        best.last_seen = frame
        best.hits += 1
        best.misses = 0

    # Tracks well inside this view that nothing matched were probably wrong,
    # unless the view is coarser than the one that found them.
    for track in state.tracks:
        if id(track) in matched or level < track.best_level:
            continue
        x1, y1, x2, y2 = track.box
        if (x1 > rx1 + EDGE_MARGIN and y1 > ry1 + EDGE_MARGIN
                and x2 < rx2 - EDGE_MARGIN and y2 < ry2 - EDGE_MARGIN):
            track.misses += 1
    state.tracks = [t for t in state.tracks if t.misses < MAX_MISSES]
    if len(state.tracks) > MAX_TRACKS:
        state.tracks.sort(key=lambda t: -t.best_confidence)
        del state.tracks[MAX_TRACKS:]
    return transient


def annotations_for(state: Sequence, frame: int, transient=()) -> List[DroneFlybyPredictionDto]:
    annotations = []
    for name, confidence, box in transient:
        bbox = clip_bbox_to_frame((
            box[0] / IMAGE_WIDTH, box[1] / IMAGE_HEIGHT, box[2] / IMAGE_WIDTH, box[3] / IMAGE_HEIGHT,
        ))
        if bbox is not None:
            annotations.append(DroneFlybyPredictionDto(
                object_id=name, bbox=[round(c, 6) for c in bbox],
                confidence=round(float(np.clip(confidence, 0.001, 1.0)), 4),
            ))
    for track in state.tracks:
        bbox = clip_bbox_to_frame((
            track.box[0] / IMAGE_WIDTH, track.box[1] / IMAGE_HEIGHT,
            track.box[2] / IMAGE_WIDTH, track.box[3] / IMAGE_HEIGHT,
        ))
        if bbox is None:
            continue
        total = sum(track.votes.values())
        ranked = sorted(track.votes.items(), key=lambda item: -item[1])
        base = track.best_confidence
        base *= min(1.0, 0.7 + 0.1 * track.hits)
        base *= UNSEEN_DECAY ** (frame - track.last_seen)
        for rank, (name, vote) in enumerate(ranked[:2]):
            share = vote / total
            if rank == 1 and vote < RUNNER_UP_SHARE * ranked[0][1]:
                break
            confidence = float(np.clip(base * (0.5 + 0.5 * share), 0.001, 1.0))
            annotations.append(DroneFlybyPredictionDto(
                object_id=name,
                bbox=[round(c, 6) for c in bbox],
                confidence=round(confidence, 4),
            ))
    annotations.sort(key=lambda a: -a.confidence)
    return annotations[:500]


# --------------------------------------------------------------------------- #
# Camera
# --------------------------------------------------------------------------- #

def legal(request: DroneFlybyPredictRequestDto, level: int, x: int, y: int) -> bool:
    view = request.view
    return describe_camera_rejection(
        view.resolution_level, (view.center_x, view.center_y), level, (x, y)
    ) is None


def choose_next_view(request: DroneFlybyPredictRequestDto, state: Sequence) -> Optional[RequestedViewDto]:
    view = request.view
    if view.resolution_level == 2:
        # Back up to Level 1 as close as the 551 px limit allows.
        bounds = request.camera_constraints.bounds_for_level(1)
        x = int(min(max(view.center_x, bounds.minimum_center_x), bounds.maximum_center_x))
        y = int(min(max(view.center_y, bounds.minimum_center_y), bounds.maximum_center_y))
        target = (1, x, y)
    else:
        # Continue the sweep from the grid point nearest to where we are.
        if view.resolution_level == 1:
            here = min(
                range(len(SWEEP)),
                key=lambda i: (SWEEP[i][0] - view.center_x) ** 2 + (SWEEP[i][1] - view.center_y) ** 2,
            )
            state.sweep_index = here + 1
        x, y = SWEEP[state.sweep_index % len(SWEEP)]
        target = (1, x, y)

    level, x, y = target
    if legal(request, level, x, y):
        return RequestedViewDto(resolution_level=level, center_x=x, center_y=y)
    # Something unexpected: the full view is always one step from L1, and
    # from L0 we simply stay.
    if legal(request, 0, *FULL_FRAME_CENTER):
        return RequestedViewDto(resolution_level=0, center_x=FULL_FRAME_CENTER[0], center_y=FULL_FRAME_CENTER[1])
    return None


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def predict(request: DroneFlybyPredictRequestDto) -> DroneFlybyPredictResponseDto:
    if request.camera_command_feedback is not None:
        logger.warning('Camera command ignored: %s', request.camera_command_feedback.reason)

    with _sequences_lock:
        state = _sequences.get(request.sequence_id)
        # A new attempt, or the same id replayed from the start: forget.
        if state is None or request.frame_index == 0 or request.frame < state.last_frame - RESTART_GAP:
            state = _sequences[request.sequence_id] = Sequence()
            while len(_sequences) > MAX_SEQUENCES:
                _sequences.pop(next(iter(_sequences)))

    view = request.view
    try:
        detections = detect(decode_view(view), view.source_region_xyxy)
    except Exception:
        logger.exception('Detector failed on frame %s', request.frame)
        detections = []

    try:
        # Requests can overlap when one runs long; the newest frame wins.
        with _sequences_lock:
            transient = []
            if request.frame >= state.last_frame:
                transient = update_tracks(state, request.frame, view.resolution_level, view.source_region_xyxy, detections)
                state.last_frame = request.frame
            annotations = annotations_for(state, request.frame, transient)
            requested_view = choose_next_view(request, state)
    except Exception:
        logger.exception('Tracking failed on frame %s', request.frame)
        annotations, requested_view = [], None

    return DroneFlybyPredictResponseDto(
        request_id=request.request_id,
        frame=request.frame,
        annotations=annotations,
        requested_view=requested_view,
    )
