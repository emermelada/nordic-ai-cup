"""World-frame tracker: objects sit on the ground, answers are projected 3D boxes.

The difference from ``flyby.py``: a track is not a 2D box carried by a fitted
2D motion, it is an OBJECT -- a ground position, an elevation, a yaw and a size
-- and every answer is its 3D box projected into the frame being answered.

That matters because the grader's boxes are exactly that projection
(probe/camera_model.py, IoU 0.935 against the official Helsinki truth), and a
2D carry cannot reproduce it: it grows every box's height ~1.3% a frame while a
tall object's box actually shrinks in height as it nears the nadir, and it gives
every object the same speed when an object 20 m higher than another moves
measurably faster.

Interface mirrors flyby.predict so api.py can serve either.
"""

import logging
import math
import os
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import geometry as G  # noqa: E402
import flyby  # noqa: E402  (detector, camera helpers and settings are reused)
from dtos import (  # noqa: E402
    OBJECT_CLASSES,
    DroneFlybyPredictionDto,
    DroneFlybyPredictRequestDto,
    DroneFlybyPredictResponseDto,
)
from utils import clip_bbox_to_frame, decode_view  # noqa: E402

logger = logging.getLogger(__name__)

# Reported box = projected 3D box scaled by this. The detector learns tight
# silhouettes and the 3D dimensions come from the official boxes, so a track
# whose size was fitted to detections needs growing back toward the convention.
BOX_GROW = float(os.environ.get('DRONE3D_GROW', '1.0'))
# Ground elevation (NED, metres) the flight is assumed to sit at before any
# track has been seen twice. Only ALT - z reaches the image, so this also
# absorbs a different altitude or speed.
Z0 = float(os.environ.get('DRONE3D_Z0', '0'))
Z_RANGE = 45.0                   # how far a single track's elevation may wander
Z_PRIOR = float(os.environ.get('DRONE3D_Z_PRIOR', '12'))   # frames of baseline the prior is worth
GROUND_MIN_SPAN = float(os.environ.get('DRONE3D_GROUND_SPAN', '8'))
GROUND_MIN_TRACKS = float(os.environ.get('DRONE3D_GROUND_TRACKS', '3'))
Z_WEIGHT = float(os.environ.get('DRONE3D_Z_WEIGHT', '1.0'))
NEW_TRACK_CONFIDENCE = float(os.environ.get('DRONE3D_TRACK_CONF', '0.10'))
DETECTION_CONFIDENCE = float(os.environ.get('DRONE3D_DET_CONF', '0.01'))
MATCH_IOU = 0.2
MATCH_COVER = 0.6
MAX_MISSES = 6
MAX_TRACKS = 150
UNSEEN_DECAY = float(os.environ.get('DRONE3D_UNSEEN_DECAY', '1.0'))
HITS_BASE, HITS_STEP = 0.7, 0.1
RUNNER_UPS = 4
RUNNER_UP_SHARE = 0.03
MIN_VOTE_SCORE = 0.02
TRANSIENT_WEIGHT = 0.5
TRUNCATED_WEIGHT = 0.5
LEVEL_WEIGHT = {0: 1.0, 1: 0.8, 2: 1.0}
EDGE_MARGIN = 20
CUT_OFF_PIXELS = 3
RESTART_GAP = 5

for _item in filter(None, os.environ.get('DRONE3D_SET', '').split(',')):
    _name, _value = _item.split('=', 1)
    if _name not in globals() or not isinstance(globals()[_name], (int, float)):
        raise SystemExit(f'DRONE3D_SET: unknown numeric setting {_name}')
    globals()[_name] = type(globals()[_name])(float(_value))
    logger.warning('Setting %s = %s', _name, globals()[_name])


# --------------------------------------------------------------------------- #
# One object
# --------------------------------------------------------------------------- #

@dataclass
class Track:
    cls: str                       # class used for the geometry (best vote)
    x: float                       # ground position, metres, flight frame
    y: float
    z: float                       # ground elevation (NED, + is lower)
    yaw: float
    scale: float                   # size relative to the class's Helsinki 3D box
    observations: List[Tuple[int, np.ndarray, bool]] = field(default_factory=list)
    votes: Dict[str, float] = field(default_factory=dict)
    best_confidence: float = 0.0
    last_seen: int = 0
    hits: int = 0
    misses: int = 0
    best_level: int = 0
    truncated: bool = False
    models: set = field(default_factory=set)
    dirty: bool = True
    ref: Optional[Tuple[int, np.ndarray]] = None

    def dims(self):
        L, Wd, Hh = G.CLASS_DIMS[self.cls]
        return L * self.scale, Wd * self.scale, Hh * self.scale

    def params(self):
        return (self.x, self.y, self.z, *self.dims(), self.yaw)

    def model_box(self, frame: int):
        return G.box_at(self.params(), frame, clip=False)

    def box_at(self, frame: int, clip=True):
        """Where this object's box is at `frame`, and how big.

        The 3D model supplies the centre (so each object gets its own speed,
        set by its elevation) and the RATIO by which each side changes (so a
        tall object's box shrinks in height as it nears the nadir). The shape
        itself stays the one the detector measured: fitting a yaw to one noisy
        Level-1 box gets elongated classes badly wrong -- measured, ta-ta
        recall 0.43 -> 0.08 -- while the ratio barely depends on yaw.
        """
        if self.ref is None:
            return G.box_at(self.params(), frame, clip=clip)
        ref_frame, ref_box = self.ref
        L, Wd, Hh = self.dims()
        centre = G.project(np.array([[self.x, self.y, self.z - 0.5 * Hh]]), frame)[0]
        now, before = self.model_box(frame), self.model_box(ref_frame)
        bw, bh = before[2] - before[0], before[3] - before[1]
        sw = (now[2] - now[0]) / bw if bw > 1e-6 else 1.0
        sh = (now[3] - now[1]) / bh if bh > 1e-6 else 1.0
        half_w = 0.5 * (ref_box[2] - ref_box[0]) * float(np.clip(sw, 0.2, 5.0))
        half_h = 0.5 * (ref_box[3] - ref_box[1]) * float(np.clip(sh, 0.2, 5.0))
        box = np.array([centre[0] - half_w, centre[1] - half_h,
                        centre[0] + half_w, centre[1] + half_h])
        if not clip:
            return box
        return np.array([max(0.0, box[0]), max(0.0, box[1]), min(G.W, box[2]), min(G.H, box[3])])

    def label(self):
        name = max(self.votes, key=self.votes.get)
        return name, self.votes[name]


def _centre_ray_position(box, frame, cls, scale, z):
    """Ground position whose 3D box centre projects to this box's centre."""
    L, Wd, Hh = G.CLASS_DIMS[cls]
    u, v = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    # The 2D centre of a projected 3D box is, to first order, the projection of
    # the box's 3D centre, which sits at half the object's height.
    point = G.ground_point(u, v, frame, z_ground=z - 0.5 * Hh * scale)
    return float(point[0]), float(point[1])


def _scale_for(box, frame, cls, x, y, z, yaw):
    """Size multiplier that makes the projected 3D box match this box's size."""
    unit = G.box_at((x, y, z, *G.CLASS_DIMS[cls], yaw), frame, clip=False)
    uw, uh = unit[2] - unit[0], unit[3] - unit[1]
    bw, bh = box[2] - box[0], box[3] - box[1]
    if uw <= 1e-6 or uh <= 1e-6:
        return 1.0
    return float(np.clip(0.5 * (bw / uw + bh / uh), 0.15, 4.0))


def new_track(cls, box, frame, level, truncated, z=Z0) -> Track:
    scale = 1.0
    x, y = _centre_ray_position(box, frame, cls, scale, z)
    yaw = 0.0
    for _ in range(2):                    # position and size settle in two passes
        scale = _scale_for(box, frame, cls, x, y, z, yaw)
        x, y = _centre_ray_position(box, frame, cls, scale, z)
    track = Track(cls=cls, x=x, y=y, z=z, yaw=yaw, scale=scale,
                  best_level=level, truncated=truncated)
    if not truncated:
        track.observations.append((frame, np.asarray(box, float), True))
        track.ref = (frame, np.asarray(box, float))
    return track


def refit(track: Track, ground_z: float = Z0) -> None:
    """Fit the object's ground position and elevation to its observed centres.

    Only (x, y, z), and only on box CENTRES. Size and yaw are left alone: the
    reported shape comes from the detector (see Track.box_at), and letting all
    five float together lets a wrong yaw pay for a wrong position -- measured,
    that cost 7 points of recall against the 2D carry it was meant to beat.

    Elevation is what gives an object its own image speed, and it is only
    observable across a baseline, so it is pulled back toward the flight's
    default in proportion to how short that baseline is.
    """
    obs = [o for o in track.observations if o[2]][-12:]
    if len(obs) < 2:
        return
    from scipy.optimize import least_squares

    Hh = track.dims()[2]
    frames = np.array([o[0] for o in obs], float)
    centres = np.array([[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for _, b, _ in obs])
    span = float(frames.max() - frames.min())
    # A short baseline cannot see elevation: 12 frames of separation is where
    # a ~2 px centring error stops swamping the ~0.5 %/10 m speed difference.
    pull = Z_PRIOR / (Z_PRIOR + max(0.0, span))

    def residual(p):
        x, y, z = p
        points = np.array([[x, y, z - 0.5 * Hh]])
        out = []
        for fr, c in zip(frames, centres):
            out.extend(G.project(points, fr)[0] - c)
        out.append(pull * Z_WEIGHT * (z - ground_z))
        return np.array(out)

    try:
        res = least_squares(residual, [track.x, track.y, track.z],
                            bounds=([-np.inf, -np.inf, ground_z - Z_RANGE], [np.inf, np.inf, ground_z + Z_RANGE]),
                            loss='soft_l1', f_scale=3.0, max_nfev=40, xtol=1e-3, ftol=1e-3)
    except Exception:
        return
    track.x, track.y, track.z = (float(v) for v in res.x)
    track.dirty = False


# --------------------------------------------------------------------------- #
# Sequence state
# --------------------------------------------------------------------------- #

@dataclass
class Sequence:
    tracks: List[Track] = field(default_factory=list)
    sweep_index: int = 0
    last_frame: int = -1
    pending: Optional[Tuple[int, int, int]] = None
    # The flight's own ground level, in the same units as a track's elevation.
    # Only ALT - z reaches the image, so this also absorbs a different altitude
    # or ground speed -- which is what makes the model transfer to a flight we
    # have never seen. Estimated from the tracks that have a long enough
    # baseline to see it, and used as the starting point for every new track.
    ground_z: float = Z0
    ground_samples: List[float] = field(default_factory=list)


def update_ground(state: Sequence) -> None:
    """Re-estimate the flight's ground level from well-observed tracks."""
    samples = []
    for track in state.tracks:
        whole = [o for o in track.observations if o[2]]
        if len(whole) >= 3 and whole[-1][0] - whole[0][0] >= GROUND_MIN_SPAN:
            samples.append(track.z)
    if len(samples) >= GROUND_MIN_TRACKS:
        state.ground_samples = samples
        state.ground_z = float(np.median(samples))


_sequences: Dict[str, Sequence] = {}
_lock = threading.Lock()


def update_tracks(state: Sequence, frame: int, level: int, region, detections):
    rx1, ry1, rx2, ry2 = region
    cut_off = CUT_OFF_PIXELS * (rx2 - rx1) / 960
    predicted = [(t, t.box_at(frame, clip=False)) for t in state.tracks]
    matched, transient = set(), []

    for detection in sorted(detections, key=lambda d: -d[1]):
        name, confidence, box = detection[:3]
        probabilities = detection[3] if len(detection) > 3 else None
        source = detection[4] if len(detection) > 4 else 0
        box = np.asarray(box, float)
        truncated = (
            (box[0] <= rx1 + cut_off and rx1 > 0) or (box[1] <= ry1 + cut_off and ry1 > 0)
            or (box[2] >= rx2 - cut_off and rx2 < G.W) or (box[3] >= ry2 - cut_off and ry2 < G.H))

        best, best_score = None, 0.0
        for track, pbox in predicted:
            overlap = G.iou(pbox, box)
            covered = _cover(pbox, box)
            if overlap > MATCH_IOU or covered > MATCH_COVER:
                score = max(overlap, covered * 0.9)
                if score > best_score:
                    best, best_score = track, score
        if best is None:
            if confidence < NEW_TRACK_CONFIDENCE or name not in G.CLASS_DIMS:
                transient.append((name, confidence * TRANSIENT_WEIGHT *
                                  (TRUNCATED_WEIGHT if truncated else 1.0), box))
                continue
            best = new_track(name, box, frame, level, truncated, state.ground_z)
            best.models.add(source)
            state.tracks.append(best)
            predicted.append((best, best.box_at(frame, clip=False)))
        elif id(best) in matched:
            best.models.add(source)
            _add_votes(best, name, confidence, probabilities, LEVEL_WEIGHT[level])
            continue
        else:
            if not truncated:
                best.observations.append((frame, box, True))
                best.ref = (frame, box.copy())
                best.dirty = True
            best.truncated = best.truncated and truncated
        matched.add(id(best))
        best.models.add(source)
        _add_votes(best, name, confidence, probabilities,
                   LEVEL_WEIGHT[level] * (TRUNCATED_WEIGHT if truncated else 1.0))
        best.best_confidence = max(best.best_confidence, confidence * (0.6 + 0.4 * LEVEL_WEIGHT[level]))
        best.last_seen = frame
        best.best_level = max(best.best_level, level)
        best.hits += 1
        best.misses = 0

    # Refit what changed, and re-key the geometry if the class vote moved.
    for track in state.tracks:
        if not track.dirty:
            continue
        top = track.label()[0]
        if top != track.cls and top in G.CLASS_DIMS:
            track.cls = top
            track.scale = 1.0
            last = track.observations[-1] if track.observations else None
            if last is not None:
                track.x, track.y = _centre_ray_position(last[1], last[0], top, 1.0, track.z)
                track.scale = _scale_for(last[1], last[0], top, track.x, track.y, track.z, track.yaw)
                track.ref = (last[0], np.asarray(last[1], float))
        refit(track, state.ground_z)

    update_ground(state)

    # A track the camera looked straight at and did not find is probably wrong.
    for track in state.tracks:
        if id(track) in matched or level < track.best_level:
            continue
        x1, y1, x2, y2 = track.box_at(frame, clip=False)
        if (x1 > rx1 + EDGE_MARGIN and y1 > ry1 + EDGE_MARGIN
                and x2 < rx2 - EDGE_MARGIN and y2 < ry2 - EDGE_MARGIN):
            track.misses += 1

    alive = []
    for track in state.tracks:
        if track.misses >= MAX_MISSES:
            continue
        box = track.box_at(frame, clip=False)
        if box[2] > 0 and box[3] > 0 and box[0] < G.W and box[1] < G.H:
            alive.append(track)
    if len(alive) > MAX_TRACKS:
        alive.sort(key=lambda t: -t.best_confidence)
        del alive[MAX_TRACKS:]
    state.tracks = alive
    return transient


def _cover(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return ix * iy / smaller if smaller > 0 else 0.0


def _add_votes(track, name, confidence, probabilities, weight) -> None:
    if probabilities is None:
        track.votes[name] = track.votes.get(name, 0.0) + weight * confidence
        return
    for index in np.flatnonzero(probabilities >= MIN_VOTE_SCORE):
        cls = OBJECT_CLASSES[index]
        track.votes[cls] = track.votes.get(cls, 0.0) + weight * float(probabilities[index])


def annotations_for(state: Sequence, frame: int, transient=()) -> List[DroneFlybyPredictionDto]:
    annotations = []
    for name, confidence, box in transient:
        bbox = clip_bbox_to_frame((box[0] / G.W, box[1] / G.H, box[2] / G.W, box[3] / G.H))
        if bbox is not None:
            annotations.append(DroneFlybyPredictionDto(
                object_id=name, bbox=[round(c, 6) for c in bbox],
                confidence=round(float(np.clip(confidence, 0.001, 1.0)), 4)))

    for track in state.tracks:
        box = G.shrink(track.box_at(frame, clip=False), BOX_GROW)
        bbox = clip_bbox_to_frame((box[0] / G.W, box[1] / G.H, box[2] / G.W, box[3] / G.H))
        if bbox is None:
            continue
        base = track.best_confidence * min(1.0, HITS_BASE + HITS_STEP * track.hits)
        base *= UNSEEN_DECAY ** max(0, frame - track.last_seen)
        if track.truncated:
            base *= TRUNCATED_WEIGHT
        ranked = sorted(track.votes.items(), key=lambda kv: -kv[1])
        top_vote = ranked[0][1] or 1.0
        for rank, (name, vote) in enumerate(ranked[:1 + RUNNER_UPS]):
            share = vote / top_vote
            if rank and share < RUNNER_UP_SHARE:
                break
            confidence = base if rank == 0 else base * 0.9 * share
            annotations.append(DroneFlybyPredictionDto(
                object_id=name, bbox=[round(c, 6) for c in bbox],
                confidence=round(float(np.clip(confidence, 0.001, 1.0)), 4)))
    annotations.sort(key=lambda a: -a.confidence)
    return annotations[:500]


def predict(request: DroneFlybyPredictRequestDto) -> DroneFlybyPredictResponseDto:
    with _lock:
        state = _sequences.get(request.sequence_id)
        if state is None or request.frame_index == 0 or request.frame < state.last_frame - RESTART_GAP:
            state = _sequences[request.sequence_id] = Sequence()

    view = request.view
    try:
        detections = flyby.detect(decode_view(view), view.source_region_xyxy, request.frame)
    except Exception:
        logger.exception('Detector failed on frame %s', request.frame)
        detections = []

    annotations, requested_view = [], None
    try:
        with _lock:
            transient = []
            stale = request.frame < state.last_frame
            if not stale:
                transient = update_tracks(state, request.frame, view.resolution_level,
                                          view.source_region_xyxy, detections)
                state.last_frame = request.frame
            try:
                annotations = annotations_for(state, request.frame, transient)
            except Exception:
                logger.exception('Building annotations failed on frame %s', request.frame)
            if not stale:
                try:
                    requested_view = flyby.choose_next_view(request, _as_flyby_state(state))
                except Exception:
                    logger.exception('Camera planning failed on frame %s', request.frame)
    except Exception:
        logger.exception('Tracking failed on frame %s', request.frame)

    return DroneFlybyPredictResponseDto(
        request_id=request.request_id, frame=request.frame,
        annotations=annotations, requested_view=requested_view)


class _CameraState:
    """Just enough of flyby.Sequence for flyby.choose_next_view to steer."""

    def __init__(self, state: Sequence):
        self._state = state
        self.tracks = []
        self.answers_since_inspection = 0
        self.hybrid_step = 0
        self.cover_index = 0

    @property
    def sweep_index(self):
        return self._state.sweep_index

    @sweep_index.setter
    def sweep_index(self, value):
        self._state.sweep_index = value

    @property
    def pending(self):
        return self._state.pending

    @pending.setter
    def pending(self, value):
        self._state.pending = value


def _as_flyby_state(state: Sequence) -> '_CameraState':
    return _CameraState(state)
