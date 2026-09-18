"""Detector, object memory and camera policy for the drone flyby.

Three parts, one request at a time:

1. **Detect** with YOLO on the transmitted view and lift the boxes into
   source pixels.
2. **Remember.** Every object ever seen is kept as a track. The ground moves
   predictably between frames (see ``MOTION``), so each track is moved forward
   to the current frame, matched against the new detections, and reported for
   the whole frame even when the camera is looking elsewhere.
3. **Steer.** Walk a fixed pattern of Level-1 views, one step per answered
   frame; every move is checked with the evaluator's own rules first.
   The evaluator renders a frame when it is emitted, usually before our
   previous answer has arrived, so a camera command lands one answer late.
   Moves are therefore planned from the last *requested* view, not from the
   view in the current image.

Configuration, all optional, through environment variables:

    DRONE_MODEL   path to the YOLO weights  (default: ~/models/drone-yolo11n-v4.pt)
    DRONE_MODEL_ALT  second weights taking alternate frames; the two models'
                     detections meet in the object memory, so a run gets the
                     union of what both can find (see detect())
                     DRONE_SET=BOTH_MODELS=1 runs both on every frame instead
    DRONE_DEVICE  torch device               (default: cpu)
    DRONE_IMGSZ   inference size             (default: 960)
                  one size per model, comma separated, to run the pair at two
                  scales: DRONE_IMGSZ=960,1280. The same weights may be named
                  twice, which makes a pair out of one model at two sizes.
    DRONE_THREADS CPU threads for inference  (default: 6)
    DRONE_DET_CONF    lowest detection reported at all       (default: 0.01)
    DRONE_TRACK_CONF  lowest detection remembered as a track (default: 0.25)
    DRONE_CAMERA      sweep pattern: full, top, mixed, hybrid or survey
                      quad0, full0, dwell or survey (data)    (default: full)
    DRONE_SET         NAME=value,... overrides any setting below, for experiments
                      DRONE_SET=MOTION_MIN_SAMPLES=999999 pins the ground motion
                      to the Helsinki fit again: the A/B for the online fit
    DRONE_INSPECT     zoom to Level 2 on small unsure objects (default: 0)
"""

import logging
import math
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from dtos import (
    ALLOWED_RESOLUTION_LEVELS,
    FULL_FRAME_CENTER,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    OBJECT_CLASSES,
    DroneFlybyPredictionDto,
    DroneFlybyPredictRequestDto,
    DroneFlybyPredictResponseDto,
    RequestedViewDto,
)
from utils import (
    center_bounds_for_level,
    clip_bbox_to_frame,
    decode_view,
    describe_camera_rejection,
)

logger = logging.getLogger(__name__)

MODEL_PATH = Path(os.environ.get('DRONE_MODEL', Path.home() / 'models' / 'drone-yolo11n-v4.pt'))
# A second set of weights, taking alternate frames. See detect().
ALT_MODEL_PATH = Path(os.environ['DRONE_MODEL_ALT']) if os.environ.get('DRONE_MODEL_ALT') else None
DEVICE = os.environ.get('DRONE_DEVICE', 'cpu')
# One size, or one per model ("960,1280"). Input size trades big objects for
# small ones: the same v4 weights at 1280 instead of 960 moved tank 0.043 ->
# 0.306 and large_tower 0.001 -> 0.313 offline while hangar fell 0.743 -> 0.435,
# so the two sizes miss different classes the same way two models do, and the
# pair can take one of each. IMGSZ stays the scalar the tools read.
IMGSZ_LIST = [int(v) for v in os.environ.get('DRONE_IMGSZ', '960').split(',') if v.strip()]
IMGSZ = IMGSZ_LIST[0]
# Measured on the i5-8350U: 6 threads 91 ms, 4 threads 110 ms, 8 threads 112 ms.
THREADS = int(os.environ.get('DRONE_THREADS', '6'))
# With two models loaded: 0 alternates them by frame, 1 runs both on every frame
# and concatenates. Alternating saves compute we do not need (the pair answers in
# ~25 ms of a 333 ms budget on the M4) and pays for it in variance: the model a
# frame gets is decided by request.frame, so every skipped frame flips the
# assignment, and offline the same pair scored 0.346 or 0.303 on that choice
# alone. Running both also stops update_tracks charging a miss to an object only
# the other model can see. DRONE_SET=BOTH_MODELS=1.
BOTH_MODELS = 0

# Two thresholds. mAP rewards ranked low-confidence guesses, so anything above
# DETECTION_CONFIDENCE is reported for the frame it was seen in. Only detections
# above NEW_TRACK_CONFIDENCE start a track that is remembered and reported on
# later frames; otherwise false alarms pile up (v1 reached 137 per frame).
DETECTION_CONFIDENCE = float(os.environ.get('DRONE_DET_CONF', '0.01'))
# Validation with v3: 0.10 -> 0.126, 0.25 -> 0.132, 0.40 -> 0.122.
NEW_TRACK_CONFIDENCE = float(os.environ.get('DRONE_TRACK_CONF', '0.25'))
# One-frame guesses rank below remembered objects of the same confidence.
TRANSIENT_WEIGHT = 0.5
MAX_TRACKS = 120
NMS_IOU = 0.5
MAX_DETECTIONS = 100

# Per-frame ground motion in source pixels, fitted on the Helsinki frames.
# The drone flies straight, so every point drifts down and slightly away from
# the centre as the ground gets closer:
#   dx = a + b*x + c*y,  dy = d + e*x + f*y
# This is only the starting guess. It was fitted on one flight, and a flight at
# a different altitude or speed moves the ground by a different number of
# pixels: measured against the recorded validation flight this is ~3 px/frame
# short vertically (66.2 vs 69.4 at the frame centre). That is under 5 %, but it
# compounds every frame, and a carried box that is 15 px out no longer overlaps
# a 37 px object at IoU 0.5 - so a track that is not re-detected for ~10 frames
# stops scoring. Since most answers come from memory rather than the current
# view, the motion is re-fitted from our own re-detections during the run.
MOTION = (-13.62, 0.00708, 0.00007, 51.25, 0.00029, 0.01334)

# Online motion fitting.
MOTION_MIN_SAMPLES = 8          # before that, the Helsinki prior stands alone
MOTION_SAMPLE_MEMORY = 240      # most recent samples kept
MOTION_MAX_GAP = 12             # frames between two sightings used as a sample
MOTION_TRIM = 0.75              # share of samples kept: vehicles move on their own
MOTION_PRIOR_STRENGTH = 12.0    # samples needed to outweigh the prior
MOTION_MAX_CORRECTION = 15.0    # px/frame at the frame centre; beyond this, distrust

# How much a detection at each level is trusted. It scales class votes only
# (add_votes), never a track's confidence, so scaling all three together
# cancels out -- what matters is the ratio between levels.
#
# Level 0 was at 0.4 on the intuition that a coarse whole-frame view is less
# reliable. Measured on five recorded runs (19 Sep), raising it to 1.0 helps in
# every one, and helps most where there is most evidence to judge on:
#
#   run        L0 views   0.4      0.8      1.0
#   5ace5364          3   0.288   +0.004   +0.010
#   8a1d65ee        130   0.333   +0.034   +0.040
#   a1c00d7c         90   0.366   +0.030   +0.039
#   81bf6bd3          3   0.282   +0.002   +0.009
#   04bef8d0          2   0.287   +0.001   +0.005
#
# The served `full` camera takes only 3 Level-0 views, so expect the top row
# (+0.010 offline, inside the real run-to-run noise of +/-0.01), not the +0.04.
# 2.0 was tried and rejected: it scores higher on the three-view runs than on
# the well-sampled ones, which is noise, not effect.
# DRONE_SET=LEVEL_WEIGHT={0:0.4,1:0.8,2:1.0} restores the old value for an A/B.
LEVEL_WEIGHT = {0: 1.0, 1: 0.8, 2: 1.0}
MATCH_IOU = 0.2
# A track the camera looked at without finding it this many times is dropped.
MAX_MISSES = 6
# Confidence of a remembered track fades with every frame it goes unseen.
UNSEEN_DECAY = 0.97
# Also report up to this many runner-up classes, when their vote is at least
# this share of the best one, at a confidence scaled by that share.
# Validation with v3 + full camera: 0 runner-ups 0.125, 2 (share 0.15) 0.130,
# 4 (share 0.03) 0.132.
RUNNER_UPS = 4
RUNNER_UP_SHARE = 0.03
# Class scores below this are not counted as votes.
MIN_VOTE_SCORE = 0.02
# Every reported box is scaled about its centre by this. Objects in the
# validation flight measure 0.55-0.85x their Helsinki box diagonal, so our
# boxes may be systematically too big for the 0.50 IoU the scorer needs.
BOX_SCALE = 1.0
# A response may carry 500 annotations and we send ~10, so naming every class
# on every object looked free. It is not: validation with v4 scored 0.134 with
# it at 0.01, against 0.143 without, because those floor boxes outrank genuine
# low-confidence detections of the same class in other frames. Off by default.
# Note for anyone tempted to retry this: average precision pools every frame
# before ranking, so "below the real answers" has to hold across the whole
# flight, not within one response. At 0.01 these boxes land in the same band as
# our own faint detections elsewhere, which is what cost the 0.009. The variant
# that has *not* been measured is a floor strictly under the 0.001 clip - low
# enough that it can never outrank a real answer in any frame. That may be
# worth one run; naming classes at 0.01 is not.
FLOOR_ALL_CLASSES = float(os.environ.get('DRONE_FLOOR_ALL', '0'))
# A box partly outside the view is a guess at the object's size: report it
# lower, and let any whole sighting replace it.
TRUNCATED_WEIGHT = 0.5
# Two boxes are the same object if they overlap this much, or if one covers
# this share of the smaller one (a cut-off half inside the whole object).
MATCH_MIN_COVER = 0.6
# Views whose edge is this close to a box do not count as missing it.
EDGE_MARGIN = 20

RESTART_GAP = 5
MAX_SEQUENCES = 8
# A detection this close to the view edge is probably cut off.
CUT_OFF_PIXELS = 3

# Level-1 sweep patterns; every step is at most 1080 px, inside the 1102 px
# L1 limit, including the step from the last point back to the first.
# New objects enter at the top edge and memory carries them down, so the top
# row matters most once the whole frame has been seen.
TL, TM, TR = (960, 540), (1920, 540), (2880, 540)
BL, BM, BR = (960, 1620), (1920, 1620), (2880, 1620)
FULL_SWEEP = [TL, TM, TR, BR, BM, BL]
TOP_SWEEP = [TL, TM, TR, TM]
SWEEPS = {
    'full': FULL_SWEEP,
    'top': FULL_SWEEP + TOP_SWEEP * 1000,          # one full look, then the top
    'mixed': FULL_SWEEP + TOP_SWEEP * 3,           # repeats: full now and then
    # Each point twice in a row: what the code did before moves were planned
    # from the pending view (the camera moved every second answer).
    'dwell': [point for point in FULL_SWEEP for _ in range(2)],
    # The whole frame every other answer, a Level-1 quarter in between. Level 0
    # is exempt from the distance limit, so the quarters need no middle stops.
    'quad0': [p for corner in (TL, TR, BR, BL) for p in ((0, 1920, 1080), (1, *corner))],
    # The full sweep with a whole-frame look after every second step.
    'full0': [TL, TM, (0, 1920, 1080), TR, BR, (0, 1920, 1080), BM, BL, (0, 1920, 1080)],
}
# Chosen on validation runs with v3 (same flight, same model):
# full 0.130, quad0 0.126, full0 0.119, dwell 0.117, top 0.108.
CAMERA = os.environ.get('DRONE_CAMERA', 'full')
# Every point as (level, x, y); plain (x, y) points are Level 1.
SWEEP = [p if len(p) == 3 else (1, *p) for p in SWEEPS.get(CAMERA, FULL_SWEEP)]

# 'survey': a data-collection pattern, not a scoring one. Level-2 views
# (native resolution) snake along two rows covering the top half, where every
# object enters; steps are at most 550 px, inside the 551 px L2 limit.
SURVEY_ROW_TOP = [(x, 270) for x in (480, 1030, 1580, 2130, 2680, 3230, 3360)]
SURVEY_ROW_LOW = [(x, 810) for x in (3360, 2810, 2260, 1710, 1160, 610, 480)]
SURVEY = SURVEY_ROW_TOP + SURVEY_ROW_LOW

# 'hybrid': acquire small objects at native resolution, keep Level-1 coverage.
#
# Why it exists. Every scored object has a fixed size in source pixels, and at
# Level 1 the transmitted view halves it: small_launcher 6.4 px, spacecraft
# 12.2, mine_roller 13.4, large_tower 14.8, tank 17.2. YOLO's finest stride is
# 8 px, so those five sit at or under the detection floor -- and they are 382 of
# 911 scored object-frames, all of them at ~0.00 AP in every configuration ever
# measured here. Level 2 is 1:1 and doubles every one of them.
#
# Measured, on the recorded survey run replayed through v4: mine_roller
# 0.000 -> 0.180 and small_launcher 0.000 -> 0.052, the first non-zero either
# class has ever scored, and tank 0.043 -> 0.084. But pure survey totals 0.080
# against 0.277, because it works the top half only and abandons jet_plane and
# large_tower entirely. Hence a hybrid rather than a switch.
#
# The shape follows the flight: the ground scrolls down ~69 px/frame, so every
# object enters at the top edge and crosses the whole frame in ~31 frames. An
# object needs to be caught once, near the top, while it is over the L2 rows;
# memory carries it down. The Level-1 phase is what stops those tracks drifting
# out of IoU and picks up the large classes L2's narrow view walks past.
#
# Both phase lengths are DRONE_SET-tunable, because the split between acquiring
# and covering cannot be measured offline -- a camera that would have looked
# somewhere else has no recorded view to replay -- so it has to be tuned on runs.
HYBRID_ACQUIRE = 9     # frames per Level-2 acquisition pass
HYBRID_COVER = 3       # frames per Level-1 coverage pass

# Level-2 inspection: a small object whose class is still unsure gets one
# close look, then the sweep resumes.
INSPECT = os.environ.get('DRONE_INSPECT', '0') == '1'
INSPECT_MAX_SIDE = 60          # source pixels: bigger ones read fine at L1
INSPECT_SURE_SHARE = 0.75      # vote share above which a class counts as settled
INSPECT_MAX_Y = 1500           # only while there is time left to use the answer
INSPECT_EVERY = 4              # answered frames between inspections, at least
INSPECT_LEAD = 3               # frames between deciding and the view arriving


for _item in filter(None, os.environ.get('DRONE_SET', '').split(',')):
    _name, _value = _item.split('=', 1)
    if _name not in globals() or not isinstance(globals()[_name], (int, float)):
        raise SystemExit(f'DRONE_SET: unknown numeric setting {_name}')
    globals()[_name] = type(globals()[_name])(float(_value))
    logger.warning('Setting %s = %s', _name, globals()[_name])


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

_model = None
_models = []
_model_lock = threading.Lock()


def size_for(which: int) -> int:
    """Inference size for model ``which``; the last size repeats if fewer given."""
    return IMGSZ_LIST[min(which, len(IMGSZ_LIST) - 1)]


def _load_one(path: Path, imgsz: int = None):
    from ultralytics import YOLO

    yolo = YOLO(str(path))
    # One ordinary prediction builds Ultralytics' inference wrapper, which runs
    # the network ~30 % faster on this CPU than calling the module directly.
    yolo.predict(np.zeros((540, 960, 3), np.uint8), imgsz=imgsz or IMGSZ, device=DEVICE, verbose=False)
    order = [OBJECT_CLASSES.index(yolo.names[i]) for i in range(len(yolo.names))]
    return yolo.predictor.model, order


def load_model():
    """Load and warm up the detector(s) once; None if the weights are missing."""
    global _model, _models
    if _model is not None:
        return _model
    if not MODEL_PATH.exists():
        logger.error('No model at %s: answering with empty detections', MODEL_PATH)
        return None
    import torch

    torch.set_num_threads(THREADS)
    _models = [_load_one(MODEL_PATH, size_for(0))]
    _model = _models[0]
    # The first inference is the slow one; pay for it before the clock starts.
    for _ in range(2):
        raw_detections(np.zeros((540, 960, 3), np.uint8))
    logger.info('Loaded %s on %s', MODEL_PATH, DEVICE)

    if ALT_MODEL_PATH is not None:
        if not ALT_MODEL_PATH.exists():
            logger.error('No alternate model at %s: running one model only', ALT_MODEL_PATH)
        else:
            _models.append(_load_one(ALT_MODEL_PATH, size_for(1)))
            for _ in range(2):
                raw_detections(np.zeros((540, 960, 3), np.uint8), 1)
            logger.info('Loaded alternate %s; models alternate per frame', ALT_MODEL_PATH)
    return _model


def raw_detections(image: np.ndarray, which: int = 0):
    """YOLO on one image: (boxes xyxy in image pixels, per-class scores).

    Ultralytics' own predictor keeps only the best class of each box. Doing the
    letterbox and NMS here keeps every class score, which lets the tracker
    report second guesses; mAP pays well for a right answer ranked lower.
    """
    import torch
    import torchvision

    index = which % len(_models) if _models else 0
    net, order = _models[index] if _models else _model
    height, width = image.shape[:2]
    ratio = size_for(index) / max(height, width)
    new_h, new_w = round(height * ratio), round(width * ratio)
    pad_h, pad_w = math.ceil(new_h / 32) * 32, math.ceil(new_w / 32) * 32
    top, left = (pad_h - new_h) // 2, (pad_w - new_w) // 2
    resized = image if ratio == 1 else cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((pad_h, pad_w, 3), 114, np.uint8)
    canvas[top:top + new_h, left:left + new_w] = resized
    tensor = torch.from_numpy(np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1)))
    tensor = tensor.to(DEVICE).float().div_(255).unsqueeze(0)

    with torch.inference_mode():
        out = net(tensor)
        out = out[0] if isinstance(out, (list, tuple)) else out
        pred = out[0].transpose(0, 1)                     # (anchors, 4 + classes)
        scores = pred[:, 4:]
        best = scores.max(dim=1).values
        keep = best >= DETECTION_CONFIDENCE
        pred, scores, best = pred[keep], scores[keep], best[keep]
        xy, wh = pred[:, :2], pred[:, 2:4]
        xyxy = torch.cat([xy - wh / 2, xy + wh / 2], dim=1)
        kept = torchvision.ops.nms(xyxy, best, NMS_IOU)[:MAX_DETECTIONS]
        xyxy, scores = xyxy[kept].cpu().numpy(), scores[kept].cpu().numpy()

    xyxy = (xyxy - [left, top, left, top]) / ratio
    probabilities = np.zeros((len(scores), len(OBJECT_CLASSES)), np.float32)
    probabilities[:, order] = scores
    return xyxy, probabilities


def detect(image: np.ndarray, source_region, frame: int = 0) -> list:
    """Detections on one view as (class, confidence, source box, class scores).

    With DRONE_MODEL_ALT set, the two models take alternate frames. They fail on
    different classes, and every detection goes into the same object memory, so
    a run sees the union of what both can find without paying for both on any
    one frame. Measured on the recorded flight (v4 + v5): the per-class mean hit
    rate is 42.6 % for v4 alone, 37.9 % for v5 alone and 46.2 % alternating.
    """
    if load_model() is None:
        return []
    with _model_lock:
        if BOTH_MODELS and len(_models) > 1:
            parts = [raw_detections(image, which) for which in range(len(_models))]
            xyxy = np.concatenate([part[0] for part in parts])
            probabilities = np.concatenate([part[1] for part in parts])
        else:
            xyxy, probabilities = raw_detections(image, frame)
    rx1, ry1, rx2, ry2 = source_region
    height, width = image.shape[:2]
    scale = np.array([(rx2 - rx1) / width, (ry2 - ry1) / height] * 2)
    boxes = xyxy * scale + [rx1, ry1, rx1, ry1]
    return [
        (OBJECT_CLASSES[int(p.argmax())], float(p.max()), box, p)
        for box, p in zip(boxes, probabilities)
    ]


# --------------------------------------------------------------------------- #
# Memory
# --------------------------------------------------------------------------- #

def advance(box: np.ndarray, steps: int, motion=MOTION) -> np.ndarray:
    """Move a source-pixel box forward by ``steps`` frames of ground motion."""
    a, b, c, d, e, f = motion
    x1, y1, x2, y2 = box
    for _ in range(max(0, steps)):
        x1, y1 = x1 + a + b * x1 + c * y1, y1 + d + e * x1 + f * y1
        x2, y2 = x2 + a + b * x2 + c * y2, y2 + d + e * x2 + f * y2
    return np.array([x1, y1, x2, y2])


def drift_at_centre(motion) -> float:
    """How far the frame centre moves in one frame, under ``motion``."""
    a, b, c, d, e, f = motion
    x, y = IMAGE_WIDTH / 2, IMAGE_HEIGHT / 2
    return float(np.hypot(a + b * x + c * y, d + e * x + f * y))


def fit_motion(samples, prior):
    """Ground motion fitted on our own re-detections, or ``prior`` if unsure.

    Each sample is (x, y, dx, dy): where an object was, and how many pixels a
    frame it has moved since we last saw it there. Most objects are ground, so
    the fit is the ground's motion - but vehicles and aircraft move on their
    own, so the worst quarter of the residuals is dropped before the final fit,
    and the result is shrunk toward the prior while samples are few.
    """
    if len(samples) < MOTION_MIN_SAMPLES:
        return prior
    data = np.asarray(samples[-MOTION_SAMPLE_MEMORY:], float)
    design = np.column_stack([np.ones(len(data)), data[:, 0], data[:, 1]])

    fitted = []
    for target in (data[:, 2], data[:, 3]):
        coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
        keep = np.abs(design @ coefficients - target) <= np.quantile(
            np.abs(design @ coefficients - target), MOTION_TRIM
        )
        if keep.sum() >= MOTION_MIN_SAMPLES:
            coefficients = np.linalg.lstsq(design[keep], target[keep], rcond=None)[0]
        fitted.append(coefficients)

    estimate = (*fitted[0], *fitted[1])
    weight = len(data) / (len(data) + MOTION_PRIOR_STRENGTH)
    blended = tuple(weight * new + (1 - weight) * old for new, old in zip(estimate, prior))
    # A fit that disagrees wildly with the prior is more likely to be a handful
    # of moving objects than a real flight: no fit beats a bad one.
    if abs(drift_at_centre(blended) - drift_at_centre(prior)) > MOTION_MAX_CORRECTION:
        return prior
    return blended


def cover(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over the smaller box."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return ix * iy / smaller if smaller > 0 else 0.0


def add_votes(track, name, confidence, probabilities, weight) -> None:
    if probabilities is None:
        track.votes[name] = track.votes.get(name, 0.0) + weight * confidence
        return
    for index in np.flatnonzero(probabilities >= MIN_VOTE_SCORE):
        cls = OBJECT_CLASSES[index]
        track.votes[cls] = track.votes.get(cls, 0.0) + weight * float(probabilities[index])


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
    inspected: bool = False
    truncated: bool = False
    # The last box as actually *observed*, not carried: one half of a motion sample.
    seen_box: Optional[np.ndarray] = None
    seen_frame: int = -1

    def label(self) -> Tuple[str, float]:
        name = max(self.votes, key=self.votes.get)
        return name, self.votes[name]


@dataclass
class Sequence:
    tracks: List[Track] = field(default_factory=list)
    sweep_index: int = 0
    last_frame: int = -1
    # The last view we asked for: (level, x, y).
    pending: Optional[Tuple[int, int, int]] = None
    answers_since_inspection: int = 0
    hybrid_step: int = 0
    # This flight's ground motion, re-fitted as re-detections come in.
    motion: Tuple[float, ...] = MOTION
    motion_samples: List[Tuple[float, float, float, float]] = field(default_factory=list)


_sequences: Dict[str, Sequence] = {}
_sequences_lock = threading.Lock()


def update_tracks(state: Sequence, frame: int, level: int, region, detections) -> list:
    """Fold this view's detections into memory; return the unremembered ones."""
    # Bring every track to this frame; forget the ones that left the ground.
    alive = []
    for track in state.tracks:
        track.box = advance(track.box, frame - track.frame, state.motion)
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
    for detection in sorted(detections, key=lambda d: -d[1]):
        name, confidence, box = detection[:3]
        probabilities = detection[3] if len(detection) > 3 else None
        # Cut short by the view edge (the frame edge cuts the true box too).
        truncated = (
            (box[0] <= rx1 + cut_off and rx1 > 0) or (box[1] <= ry1 + cut_off and ry1 > 0)
            or (box[2] >= rx2 - cut_off and rx2 < IMAGE_WIDTH)
            or (box[3] >= ry2 - cut_off and ry2 < IMAGE_HEIGHT)
        )

        best, best_score = None, 0.0
        for track in state.tracks:
            overlap = iou(track.box, box)
            covered = cover(track.box, box)
            if overlap > MATCH_IOU or covered > MATCH_MIN_COVER:
                score = max(overlap, covered * 0.9)
                if score > best_score:
                    best, best_score = track, score
        if best is None:
            if confidence < NEW_TRACK_CONFIDENCE:
                transient.append((name, confidence * TRANSIENT_WEIGHT * (TRUNCATED_WEIGHT if truncated else 1), box))
                continue
            best = Track(box=box, frame=frame, best_level=level, truncated=truncated)
            if not truncated:
                best.seen_box, best.seen_frame = box.copy(), frame
            state.tracks.append(best)
        elif id(best) in matched:
            # A second detection of an object already handled this frame is a
            # class vote only; the first (most confident) one set the box.
            add_votes(best, name, confidence, probabilities, weight)
            continue
        elif truncated:
            # Two partial views: the object spans at least both.
            if best.truncated:
                best.box = np.array([min(best.box[0], box[0]), min(best.box[1], box[1]),
                                     max(best.box[2], box[2]), max(best.box[3], box[3])])
        elif best.truncated or level >= best.best_level:
            # A whole sighting beats a partial one; finer views beat coarser.
            best.box = box
            best.best_level = level
            best.truncated = False
        else:
            best.box = 0.7 * best.box + 0.3 * box
        matched.add(id(best))
        # Seeing the same object twice measures how far the ground really moved
        # between those frames. Only whole sightings: a box the view edge cut
        # short has a centre that says more about the edge than the object.
        if not truncated:
            gap = frame - best.seen_frame
            if best.seen_box is not None and 0 < gap <= MOTION_MAX_GAP:
                was = best.seen_box
                state.motion_samples.append((
                    (was[0] + was[2]) / 2, (was[1] + was[3]) / 2,
                    ((box[0] + box[2]) - (was[0] + was[2])) / 2 / gap,
                    ((box[1] + box[3]) - (was[1] + was[3])) / 2 / gap,
                ))
            best.seen_box, best.seen_frame = box.copy(), frame
        add_votes(best, name, confidence, probabilities, weight * (TRUNCATED_WEIGHT if truncated else 1))
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

    if len(state.motion_samples) > MOTION_SAMPLE_MEMORY * 2:
        del state.motion_samples[:-MOTION_SAMPLE_MEMORY]
    state.motion = fit_motion(state.motion_samples, MOTION)
    return transient


def annotations_for(state: Sequence, frame: int, transient=()) -> List[DroneFlybyPredictionDto]:
    annotations = []

    def scaled(box):
        if BOX_SCALE == 1.0:
            return box
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = (x2 - x1) * BOX_SCALE / 2, (y2 - y1) * BOX_SCALE / 2
        return np.array([cx - w, cy - h, cx + w, cy + h])

    for name, confidence, box in transient:
        box = scaled(box)
        bbox = clip_bbox_to_frame((
            box[0] / IMAGE_WIDTH, box[1] / IMAGE_HEIGHT, box[2] / IMAGE_WIDTH, box[3] / IMAGE_HEIGHT,
        ))
        if bbox is not None:
            annotations.append(DroneFlybyPredictionDto(
                object_id=name, bbox=[round(c, 6) for c in bbox],
                confidence=round(float(np.clip(confidence, 0.001, 1.0)), 4),
            ))
    for track in state.tracks:
        box = scaled(track.box)
        bbox = clip_bbox_to_frame((
            box[0] / IMAGE_WIDTH, box[1] / IMAGE_HEIGHT,
            box[2] / IMAGE_WIDTH, box[3] / IMAGE_HEIGHT,
        ))
        if bbox is None:
            continue
        ranked = sorted(track.votes.items(), key=lambda item: -item[1])
        base = track.best_confidence
        base *= min(1.0, 0.7 + 0.1 * track.hits)
        # max(0, ...): a stale frame is behind tracks already moved to a later
        # one, and a negative exponent would *raise* the confidence above
        # best_confidence instead of decaying it.
        base *= UNSEEN_DECAY ** max(0, frame - track.last_seen)
        if track.truncated:
            base *= TRUNCATED_WEIGHT
        # A track can hold only zero-weight votes (LEVEL_WEIGHT of 0 for the
        # level it was seen at), and dividing by that raised ZeroDivisionError
        # inside the caller's try, which discards the WHOLE frame's annotations
        # and its camera command -- a silent, total loss that looks like a
        # quiet frame. Fall back to equal shares instead.
        top_vote = ranked[0][1] or 1.0
        named = set()
        for rank, (name, vote) in enumerate(ranked[:1 + RUNNER_UPS]):
            share = vote / top_vote
            if rank and share < RUNNER_UP_SHARE:
                break
            named.add(name)
            confidence = base if rank == 0 else base * 0.9 * share
            annotations.append(DroneFlybyPredictionDto(
                object_id=name,
                bbox=[round(c, 6) for c in bbox],
                confidence=round(float(np.clip(confidence, 0.001, 1.0)), 4),
            ))
        if FLOOR_ALL_CLASSES:
            for name in OBJECT_CLASSES:
                if name not in named:
                    annotations.append(DroneFlybyPredictionDto(
                        object_id=name,
                        bbox=[round(c, 6) for c in bbox],
                        confidence=round(float(np.clip(base * FLOOR_ALL_CLASSES, 0.001, 1.0)), 4),
                    ))
    annotations.sort(key=lambda a: -a.confidence)
    return annotations[:500]


# --------------------------------------------------------------------------- #
# Camera
# --------------------------------------------------------------------------- #

def legal(base: Tuple[int, int, int], level: int, x: int, y: int) -> bool:
    return describe_camera_rejection(base[0], (base[1], base[2]), level, (x, y)) is None


def inspection_target(state: Sequence, base, frame: int) -> Optional[Tuple[int, int, int]]:
    """The most doubtful small object reachable from ``base``, as an L2 view."""
    best, best_doubt = None, 0.0
    for track in state.tracks:
        if track.inspected or track.best_level >= 2:
            continue
        box = advance(track.box, INSPECT_LEAD, state.motion)
        width, height = box[2] - box[0], box[3] - box[1]
        if max(width, height) > INSPECT_MAX_SIDE or box[1] > INSPECT_MAX_Y or box[3] < 0:
            continue
        share = max(track.votes.values()) / sum(track.votes.values())
        doubt = (1 - share) + (1 - track.best_confidence)
        if share >= INSPECT_SURE_SHARE and track.best_confidence >= 0.6:
            continue
        x = int(min(max((box[0] + box[2]) / 2, 480), 3360))
        y = int(min(max((box[1] + box[3]) / 2, 270), 1890))
        if doubt > best_doubt and legal(base, 2, x, y):
            best, best_doubt = (track, (2, x, y)), doubt
    if best is None:
        return None
    best[0].inspected = True
    return best[1]


def step_towards(base, level: int, x: int, y: int) -> Optional[Tuple[int, int, int]]:
    """A legal move from ``base`` towards (level, x, y), shortened if too far.

    The evaluator checks three things in order -- the level transition, the
    centre bounds for the requested level, then the distance against the limit
    for the level you are *currently* at (L0 2203, L1 1102, L2 551). A move that
    fails any of them is refused and the camera idles, so this shortens the step
    until it passes rather than asking for something that cannot be granted.
    """
    if level not in ALLOWED_RESOLUTION_LEVELS.get(base[0], ()):
        # Levels change one step at a time; 1 is reachable from both 0 and 2.
        level = 1
    if level == 0:
        return (0, *FULL_FRAME_CENTER)
    low_x, high_x, low_y, high_y = center_bounds_for_level(level)
    x = int(min(max(x, low_x), high_x))
    y = int(min(max(y, low_y), high_y))
    if legal(base, level, x, y):
        return (level, x, y)
    # Too far: walk in from the target until the step fits. Clipping to bounds
    # can itself lengthen the move, so this checks rather than computes a ratio.
    dx, dy = x - base[1], y - base[2]
    for ratio in (0.9, 0.75, 0.6, 0.45, 0.3, 0.15):
        near_x = int(min(max(base[1] + dx * ratio, low_x), high_x))
        near_y = int(min(max(base[2] + dy * ratio, low_y), high_y))
        if legal(base, level, near_x, near_y):
            return (level, near_x, near_y)
    return None


def hybrid_next_view(base, state: Sequence) -> Optional[RequestedViewDto]:
    """Alternate Level-2 acquisition over the top rows with Level-1 coverage."""
    state.hybrid_step += 1
    cycle = max(1, HYBRID_ACQUIRE + HYBRID_COVER)
    acquiring = (state.hybrid_step % cycle) < HYBRID_ACQUIRE

    if acquiring:
        # Advance along the L2 snake only once the camera has actually arrived,
        # so a shortened step resumes towards the same point instead of skipping it.
        point = SURVEY[state.sweep_index % len(SURVEY)]
        if base[0] == 2 and (base[1], base[2]) == point:
            state.sweep_index = (state.sweep_index + 1) % len(SURVEY)
            point = SURVEY[state.sweep_index % len(SURVEY)]
        target = step_towards(base, 2, *point)
    else:
        # Coverage: the nearest Level-1 sweep point, so leaving L2 costs one move.
        reachable = [p for p in SWEEP if p[0] == 1 and legal(base, *p)]
        if reachable:
            target = min(reachable, key=lambda p: (p[1] - base[1]) ** 2 + (p[2] - base[2]) ** 2)
        else:
            target = step_towards(base, 1, base[1], base[2])

    if target is not None and legal(base, *target):
        state.pending = target
        return RequestedViewDto(resolution_level=target[0], center_x=target[1], center_y=target[2])
    state.pending = None
    return None


def survey_next_view(base, state: Sequence) -> Optional[RequestedViewDto]:
    if base[0] == 2 and (base[1], base[2]) == SURVEY[state.sweep_index % len(SURVEY)]:
        state.sweep_index = (state.sweep_index + 1) % len(SURVEY)
    elif base[0] == 2:
        # Off the pattern: rejoin at the nearest point that can be reached.
        state.sweep_index = min(
            range(len(SURVEY)),
            key=lambda i: (SURVEY[i][0] - base[1]) ** 2 + (SURVEY[i][1] - base[2]) ** 2,
        )
    x, y = SURVEY[state.sweep_index % len(SURVEY)]
    if base[0] == 0:
        # Zoom in one step, towards the first survey point.
        target = (1, int(min(max(x, 960), 2880)), int(min(max(y, 540), 1620)))
    else:
        target = (2, x, y)
        if not legal(base, *target):
            # Too far for one move: step towards it in a straight line.
            limit = 1100 if base[0] == 1 else 550
            dx, dy = x - base[1], y - base[2]
            ratio = min(1.0, limit / max(1.0, math.hypot(dx, dy)))
            target = (2, int(base[1] + dx * ratio), int(base[2] + dy * ratio))
            target = (2, int(min(max(target[1], 480), 3360)), int(min(max(target[2], 270), 1890)))
    if legal(base, *target):
        state.pending = target
        return RequestedViewDto(resolution_level=target[0], center_x=target[1], center_y=target[2])
    state.pending = None
    return None


def choose_next_view(request: DroneFlybyPredictRequestDto, state: Sequence) -> Optional[RequestedViewDto]:
    view = request.view
    here = (view.resolution_level, view.center_x, view.center_y)
    # Where the camera will be when this command is applied: our previous
    # request, unless the evaluator refused it.
    base = here
    if state.pending is not None and request.camera_command_feedback is None:
        base = state.pending

    if CAMERA == 'survey':
        return survey_next_view(base, state)
    if CAMERA == 'hybrid':
        return hybrid_next_view(base, state)

    target = None
    state.answers_since_inspection += 1
    if INSPECT and base[0] == 1 and state.answers_since_inspection >= INSPECT_EVERY:
        target = inspection_target(state, base, request.frame)
    if target is not None:
        state.answers_since_inspection = 0
    elif base[0] == 2:
        # Back up to Level 1 as close as the 551 px limit allows.
        x = int(min(max(base[1], 960), 2880))
        y = int(min(max(base[2], 540), 1620))
        target = (1, x, y)
    else:
        position = SWEEP[state.sweep_index % len(SWEEP)]
        if base == position:
            state.sweep_index = (state.sweep_index + 1) % len(SWEEP)
        elif base in SWEEP and base[0] == 1:
            # Off the pattern (a refusal, a restart): carry on from the nearest
            # matching point at or after the current index, so repeated points
            # in a pattern do not send the sweep back to its opening pass.
            offsets = range(len(SWEEP))
            step = next((k for k in offsets if SWEEP[(state.sweep_index + k) % len(SWEEP)] == base), 0)
            state.sweep_index = (state.sweep_index + step + 1) % len(SWEEP)
        target = SWEEP[state.sweep_index % len(SWEEP)]
        if base[0] == 1 and target[0] == 1 and not legal(base, *target):
            # Too far for one move: rejoin the pattern at the nearest Level-1 point.
            nearest = min(
                (i for i, p in enumerate(SWEEP) if p[0] == 1 and legal(base, *p)),
                key=lambda i: (SWEEP[i][1] - base[1]) ** 2 + (SWEEP[i][2] - base[2]) ** 2,
                default=None,
            )
            if nearest is not None:
                state.sweep_index = nearest
                target = SWEEP[nearest]

    if target is not None and legal(base, *target):
        state.pending = target
        return RequestedViewDto(resolution_level=target[0], center_x=target[1], center_y=target[2])
    # Something unexpected: the full view is always one step from L0 and L1.
    full = (0, *FULL_FRAME_CENTER)
    if legal(base, *full):
        state.pending = full
        return RequestedViewDto(resolution_level=0, center_x=FULL_FRAME_CENTER[0], center_y=FULL_FRAME_CENTER[1])
    state.pending = None
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
        detections = detect(decode_view(view), view.source_region_xyxy, request.frame)
    except Exception:
        logger.exception('Detector failed on frame %s', request.frame)
        detections = []

    annotations, requested_view = [], None
    try:
        # Requests can overlap when one runs long; the newest frame wins.
        with _sequences_lock:
            transient = []
            stale = request.frame < state.last_frame
            if not stale:
                transient = update_tracks(state, request.frame, view.resolution_level, view.source_region_xyxy, detections)
                state.last_frame = request.frame
            # Separate try blocks: these two answer different questions, and a
            # failure building annotations used to discard the camera command as
            # well, which steers the rest of the run, not just this frame.
            try:
                annotations = annotations_for(state, request.frame, transient)
            except Exception:
                logger.exception('Building annotations failed on frame %s', request.frame)
            # A late frame's answer is still scored, but its view is out of
            # date: leave the camera plan to the newest frame.
            if not stale:
                try:
                    requested_view = choose_next_view(request, state)
                except Exception:
                    logger.exception('Camera planning failed on frame %s', request.frame)
    except Exception:
        logger.exception('Tracking failed on frame %s', request.frame)

    return DroneFlybyPredictResponseDto(
        request_id=request.request_id,
        frame=request.frame,
        annotations=annotations,
        requested_view=requested_view,
    )
