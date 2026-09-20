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
# One or more extra sets of weights, comma separated like DRONE_IMGSZ:
#   DRONE_MODEL_ALT=models/drone-yolo11s-v6.pt
#   DRONE_MODEL_ALT=models/drone-yolo11s-v6.pt,models/drone-yolo11m-v8.pt
# Every model's detections meet in the same object memory, so a run sees the
# union of what all of them find. Measured offline on the loose truth with box
# growth on: v4+v6 0.488, v4+v6+v8 0.497. The +0.010 is at the real-run noise
# floor; the better argument for a third model is that v6 and v8 fail on
# opposite classes (v8 helicopter 0.659 against v6's 0.527 and spacecraft 0.107
# against 0.003; v6 hangar 0.871 against v8's 0.505), and the final evaluation
# is a different flight whose class mix is unknown.
ALT_MODEL_PATHS = [Path(v) for v in os.environ.get('DRONE_MODEL_ALT', '').split(',') if v.strip()]
# Kept for the tools and /api, which report a single alternate.
ALT_MODEL_PATH = ALT_MODEL_PATHS[0] if ALT_MODEL_PATHS else None
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
# DRONE_LEVEL0_WEIGHT=0.4 restores the old value for an A/B. It is a separate
# variable because DRONE_SET takes numeric settings only -- it splits on commas
# and rejects anything that is not already an int or float, so passing a dict
# literal there makes the service exit at startup rather than run with the old
# value.
LEVEL_WEIGHT = {0: float(os.environ.get('DRONE_LEVEL0_WEIGHT', '1.0')), 1: 0.8, 2: 1.0}
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
# How much a track's reported confidence is scaled by how CLEANLY its class
# votes agree. Average precision reads ordering only, so two tracks with the
# same peak detection confidence rank identically today even when one has a
# clean 0.95 vote share for its class and the other is split 0.35/0.33/0.32.
# base *= share ** CLASS_SHARE_POWER, so 0.0 is exactly the served behaviour.
CLASS_SHARE_POWER = float(os.environ.get('DRONE_SHARE_POWER', '0'))
# Class scores below this are not counted as votes.
MIN_VOTE_SCORE = 0.02
# Every reported box is scaled about its centre by this. Objects in the
# validation flight measure 0.55-0.85x their Helsinki box diagonal, so our
# boxes may be systematically too big for the 0.50 IoU the scorer needs.
BOX_SCALE = 1.0
# Box growth toward the official box convention, applied to REPORTED boxes only.
#
# The evaluator's boxes look like the projected 3D box of each object -- rotor
# span, wingtips and height included -- while make_dataset.py labels every
# pasted cut-out with the tight box around its alpha mask. So our models learned
# tight boxes and are scored against loose ones.
#
# The strongest evidence is already in our own history: DRONE_BOX_SCALE=0.8
# collapsed a real run from 0.143 to 0.017. Shrinking a well-matched box by 20%
# gives IoU ~0.64, comfortably over the 0.50 threshold and worth a few points at
# most. An 88% collapse only happens if the boxes were already sitting just
# above the threshold -- which is what a systematic size mismatch looks like.
#
# Measured on this machine's Helsinki cut-outs (official patch box / tight mask
# box, median per class; training/measure_box_convention.py re-measures):
# scoring an offline replay against truth grown by these factors cuts the mean
# error against six real validation scores from 0.134 to 0.075.
#
#   DRONE_BOX_GROW=helsinki   the isotropic factors below, capped
#   DRONE_BOX_GROW_CAP=1.3    ... at this (default 1.3)
#   DRONE_BOX_GROW=1.2        one factor for every class
#   DRONE_BOX_GROW=tank=1.3,jet_plane=1.5     explicit per class
#   DRONE_BOX_GROW unset      off, the 0.3048 behaviour
#
# NOT confirmed on a real run yet. The offline case rests on a truth file grown
# by these same factors, which cannot be fully independent. One validation run
# decides it, and the effect should be far outside the +/-0.01 noise either way.
# ISOTROPIC, and deliberately so: these are the exact values that scored 0.4618
# on validation (branch BEST-WORKING-VERSION, commit 945e89b). The convention is
# really anisotropic -- small_launcher measures 1.75 wide against 2.14 tall,
# helicopter 1.38 against 1.70 -- and DRONE_BOX_GROW_WH=1 switches to the
# measured per-dimension factors below. That form is NOT validated and showed no
# offline gain (0.488 either way, because cap 1.3 clamps almost everything), so
# it stays off. Do not make it the default without a real run that beats 0.4618.
HELSINKI_BOX_FACTORS = {
    'condor': 1.496, 'hangar': 1.152, 'helicopter': 1.531, 'jammer': 1.078,
    'jet_plane': 1.489, 'large_launcher': 1.115, 'large_tower': 1.078,
    'medium_launcher': 2.302, 'medium_plane': 1.320, 'mine_roller': 1.077,
    'small_launcher': 1.936, 'small_plane': 1.075, 'small_tower': 1.184,
    'spacecraft': 1.104, 'ta-ta': 1.147, 'tank': 1.309,
}
HELSINKI_BOX_FACTORS_WH = {
    'condor': (1.57, 1.43), 'hangar': (1.08, 1.23), 'helicopter': (1.38, 1.70),
    'jammer': (1.07, 1.09), 'jet_plane': (1.41, 1.57), 'large_launcher': (1.23, 1.01),
    'large_tower': (1.07, 1.08), 'medium_launcher': (2.24, 2.37), 'medium_plane': (1.49, 1.17),
    'mine_roller': (1.08, 1.07), 'small_launcher': (1.75, 2.14), 'small_plane': (1.13, 1.02),
    'small_tower': (1.10, 1.28), 'spacecraft': (1.07, 1.14), 'ta-ta': (1.16, 1.13),
    'tank': (1.30, 1.32),
}
BOX_GROW_WH = os.environ.get('DRONE_BOX_GROW_WH', '') == '1'
BOX_GROW_CAP = float(os.environ.get('DRONE_BOX_GROW_CAP', '1.3'))


def _parse_box_grow(spec: str):
    """'helsinki' | '1.2' | 'tank=1.3,...' -> {class: (width factor, height factor)}.

    The cap applies per dimension, so a class stretched in one axis keeps that
    asymmetry up to the cap instead of being averaged away.
    """
    spec = (spec or '').strip()
    if not spec:
        return {}
    cap = lambda f: min(f, BOX_GROW_CAP)
    if spec == 'helsinki':
        if BOX_GROW_WH:
            return {n: (cap(w), cap(h)) for n, (w, h) in HELSINKI_BOX_FACTORS_WH.items()}
        return {n: (cap(f), cap(f)) for n, f in HELSINKI_BOX_FACTORS.items()}
    try:
        both = cap(float(spec))
        return {n: (both, both) for n in OBJECT_CLASSES}
    except ValueError:
        pass
    out = {}
    for item in spec.split(','):
        name, _, value = item.partition('=')
        name = name.strip()
        if name not in OBJECT_CLASSES:
            raise SystemExit(f'DRONE_BOX_GROW: unknown class {name!r}')
        out[name] = (cap(float(value)), cap(float(value)))
    return out


BOX_GROW = _parse_box_grow(os.environ.get('DRONE_BOX_GROW', ''))
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
# The floor variant the comment above calls unmeasured, built so that it cannot
# do what FLOOR_ALL_CLASSES did. Every unnamed class gets a box at confidence
# EXACTLY 0.0, which is strictly below every real answer we emit (measured on
# the best v4 run: 3272 answers, none below 0.0015, none at the 0.001 clip).
#
# Probed against faster_coco_eval itself on 19 Sep, because this rests on the
# scorer's exact semantics rather than on an argument about them:
#   * maxDets is 100 PER IMAGE PER CATEGORY, and we emit at most 8 of one class
#     in a frame, so the band has room;
#   * 150 junk boxes a frame at score 0.0 left a perfect class at AP 1.000 -
#     detections ranked strictly below every real one cannot lower AP;
#   * a class with no answers at all went 0.000 -> 0.121 when ten true boxes
#     rode in on such a band.
# Both conditions are load-bearing: tied at 0.0 and listed AFTER the junk, the
# true box was truncated away by maxDets and the class fell back to 0.000. So
# the band is appended after the real answers are sorted, never sorted with
# them, and the 0.0 must not pass through the np.clip(..., 0.001, ...) that
# every other confidence here does.
#
# What it is for: small_launcher was emitted 8 times in 247 frames. Its AP of
# 0.000 is a no-answer problem, and no re-ranking can fix a class we never
# answer for. The score is a mean over the classes present in the truth, so a
# class we never name is a free zero -- which matters more on the evaluation
# flight, whose class mix is unknown, than on validation.
# --------------------------------------------------------------------------- #
# The graded band
# --------------------------------------------------------------------------- #
# What the 20 Sep per-class calibration found. Replaying one recorded run
# (real score 0.5278) one class at a time through the real grader gives, as
# AP (score contribution x K, with K=13 present classes):
#
#   hangar .93  large_tower .88  jet_plane .85  helicopter .84  mine_roller .69
#   tank .63  large_launcher .63  small_plane .61  small_tower .58
#   small_launcher .15  medium_launcher .04  medium_plane .02  ta-ta .00003
#   spacecraft, condor, jammer  EXACTLY 0.0 -> absent from this flight's truth
#
# Two of the four dead classes are a bug, not a hard problem. DRONE_BOX_GROW is
# a flat 1.3 and DRONE_BOX_GROW_CAP clamps at 1.3, but the measured Helsinki
# convention wants 2.302 for medium_launcher and 1.936 for small_launcher. A
# box grown 1.3 when it needed 2.302 is nested at IoU (1.3/2.302)^2 = 0.319,
# and at 1.936 -> 0.451. Those are the ONLY two classes whose capped IoU falls
# under the scorer's 0.50 threshold, and they are exactly the two that
# collapse; every class at IoU >= 0.72 scores 0.58-0.93. The AP ordering even
# matches the IoU ordering. So they are not hard to see and not hard to fix.
#
# The band is the risk-free half of the fix. Average precision pools the whole
# flight and ranks by confidence, so a detection appended strictly BELOW every
# answer we believe in can never lower any class's AP -- it can only add recall
# in the tail. Every real answer here is clipped at 0.001, so the range
# [0.0001, 0.0009] is free space, and the DTO round-trips four decimals.
#
# Measured against faster_coco_eval itself (the evaluator's own library), on a
# class broken by exactly this kind of box mismatch:
#
#   single wrong scale                          AP 0.000
#   + 4 shape variants all tied at 0.0          AP 0.216
#   + 4 shape variants GRADED 9/7/5/3 e-4       AP 0.500
#   + 8 shape variants graded                   AP 0.500   (widening is free)
#
# That is the whole lesson, and it is why the existing FLOOR_ZERO was worth so
# much less than it should have been: boxes tied at one confidence dilute to
# about AP/k, because the scorer cannot order them. Graded ones do not dilute
# at all -- eight variants scored the same as four. So the band should be as
# wide as the 500-annotation budget allows, and its order is the prior.
BAND = os.environ.get('DRONE_BAND', '0') == '1'
# Absolute growth factors applied to the TIGHT track box, best guess first.
# None means 'this class's measured Helsinki factor', which is the variant the
# capped primary is missing and therefore the one most likely to land.
# The first entry is the measured convention and goes out at the best grade.
# The class hedge below is graded SECOND, ahead of the remaining shapes,
# because with the primary growth uncapped the extra shapes are only
# insurance while the hedge is the whole medium_plane fix -- and the 500-box
# budget is not big enough for both at 50 tracks a frame.
BAND_SCALES = (None, 1.65, 2.1, 0.75)
# Grade index the class hedge is emitted at, ahead of BAND_SCALES[1:].
BAND_HEDGE_RANK = 1
# Confidence per band rank. Strictly under the 0.001 clip on every real answer,
# strictly decreasing, and distinct at four decimals.
BAND_GRADES = (0.0009, 0.0008, 0.0007, 0.0006, 0.0005, 0.0004, 0.0003, 0.0002)
# Also name classes the track never voted for, at its own measured factor, in
# the last grades. This is the medium_plane case: 52 px, IoU 0.97, AP 0.02 --
# found, boxed, and called small_plane or jet_plane instead.
BAND_HEDGE = os.environ.get('DRONE_BAND_HEDGE', '1') == '1'
# A hedged class must be within this factor of the track box's size prior.
BAND_HEDGE_TOL = float(os.environ.get('DRONE_BAND_HEDGE_TOL', '2.5'))

FLOOR_ZERO = os.environ.get('DRONE_FLOOR_ZERO', '0') == '1'
# A track only gets floor boxes for classes whose size is plausible for it:
# emitting hangar on a 30 px track spends precision for nothing, and AP is
# precision at the recall achieved. The tolerance is deliberately loose because
# the validation flight renders objects at roughly 0.5-0.9x their Helsinki size
# -- too tight a filter drops the very class the band exists to answer.
# Lowering it raises the band's precision; 1.8 is the value to try second.
FLOOR_SIZE_TOL = float(os.environ.get('DRONE_FLOOR_SIZE_TOL', '2.5'))
# Median official box size per class, sqrt(w*h) in source pixels, measured on
# the 25 official Helsinki frames (src/helsinki/annotations) on 19 Sep. Used as
# a size prior only, never as a box.
CLASS_SIZE = {
    'condor': 169.5, 'hangar': 154.3, 'helicopter': 104.4, 'jammer': 37.7,
    'jet_plane': 79.5, 'large_launcher': 128.0, 'large_tower': 63.0,
    'medium_launcher': 46.0, 'medium_plane': 52.4, 'mine_roller': 56.4,
    'small_launcher': 25.7, 'small_plane': 46.4, 'small_tower': 58.5,
    'spacecraft': 46.4, 'ta-ta': 23.3, 'tank': 48.5,
}
# Average precision is ranking and nothing else, and two signals the tracker
# already holds never reach the reported confidence: best_confidence is a
# running maximum over sightings that never falls, and `misses` -- the number of
# times the camera looked straight at a track and did not find it -- only ever
# deletes the track at MAX_MISSES. So a track refuted five times is ranked
# exactly as high as one just seen. 1.0 is the served behaviour; 0.85 is the
# value to try.
MISS_PENALTY = float(os.environ.get('DRONE_MISS_PENALTY', '1.0'))
# How much a track is trusted when only SOME of the loaded models have ever
# found it. With BOTH_MODELS every model sees every frame, so a real object is
# normally found by several of them and a false alarm on a bush or a rooftop
# often by one -- and until now detect() concatenated all their detections and
# threw the model index away, so that signal was never used.
#
# Why it matters, measured 19 Sep on the best recorded run: mean recall over
# the twelve scored classes is 0.661 while mean AP is 0.494. We are not failing
# to FIND objects -- large_launcher is detected in 100 % of its frames,
# large_tower 93 %, mine_roller 84 % -- we bury them under our own false
# positives. 0.168 of score is pure ranking loss, concentrated in mine_roller
# (0.398), tank (0.338), small_plane (0.310) and large_launcher (0.275).
#
# base is multiplied by AGREEMENT_WEIGHT once per model that has NEVER matched
# the track, so a track all three models have seen is untouched and a
# single-model track is scaled by AGREEMENT_WEIGHT ** 2. 1.0 is the served
# behaviour; 0.7 is the value to try.
AGREEMENT_WEIGHT = 1.0
# The confidence of a track with few sightings. base *= min(1, HITS_BASE +
# HITS_STEP * hits), so the defaults give 0.8 at one hit and saturate at three.
# Phantom tracks are mostly one- and two-hit tracks, so lowering HITS_BASE
# demotes them; these are the served values.
HITS_BASE = 0.7
HITS_STEP = 0.1
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
    # One full sweep to find what is already on screen, then the top row only.
    #
    # New objects only ever enter at the top edge, and recall is worst exactly
    # there: measured on a recorded run, the served `full` sweep answers 0.546
    # of object-frames in the top third against 0.797 in the bottom, because an
    # object descends for several frames before the sweep looks its way. 16.8 %
    # of all object-frames are lost to that lag alone (probe/miss_anatomy.py).
    # Alternating the two top corners shows every new object within ~1 frame
    # and gives it ~8 looks while it crosses the top half instead of ~2, which
    # matters most for the small classes a single look finds only 30-45 % of
    # the time (probe/detect_curve.py).
    #
    # It pays for that by never looking back: everything below the top half is
    # carried. That is affordable only because the online motion fit carries
    # well -- 85.8 % of boxes still hit after a 20-frame gap, against 16 % with
    # the stale Helsinki prior (probe/carry_test.py). Expected recall over the
    # fitted objects: 0.884 against `full`'s 0.841 (probe/camera_plan.py).
    # NOTE the middle stops: TL -> TR is 1920 px and the L1 limit is 1102, so a
    # bare left-right alternation is refused on every step. choose_next_view
    # answers a refused step by rejoining at the nearest legal pattern point,
    # which is the one it is standing on -- the camera would stall on TL for the
    # whole flight while every log line looked healthy. 'top' (above) already
    # has the right shape; these are the same idea with a wider duty cycle.
    'toprow': FULL_SWEEP + [TL, TM, TR, TM] * 200,
    # Top row mostly, with an occasional look back at the bottom middle for
    # objects the first pass missed. TM -> BM is 1080 px, just inside the limit.
    'top_mostly': FULL_SWEEP + [TL, TM, TR, TM, TL, TM, TR, TM, BM, TM] * 80,
    # 'row0' = the full Level-1 sweep with a whole-frame Level-0 look after each
    # row (Javier, ffbb874). Additive: it keeps every Level-1 look `full` takes
    # and spends two extra frames per cycle on the whole frame. Measured in the
    # closed-loop simulator at recall 0.545 against full's 0.466, ahead on AP
    # against both truth files, and best of ten patterns.
    #
    # The Level-0 look only pays because of the imgsz 2560 pass: at Level 0 one
    # transmitted pixel is 4 source pixels, so a stride-8 cell at 2560 spans 12
    # source pixels -- exactly what a Level-1 view had at 1280. v4@960 finds 0
    # of 18 objects at Level 0 and v6@1280 finds 12 of 18, which is why full0
    # and quad0 lost when they were judged on v3.
    #
    # Note this is the opposite trade to 'top_mostly', which was measured on a
    # real run at 0.5150/0.5098 against full's 0.5270: giving UP the bottom-row
    # looks costs more than the earlier acquisition gains, because objects are
    # ~1.5x larger at the bottom and that is where the class votes are reliable.
    'row0': [TL, TM, TR, (0, 1920, 1080), BR, BM, BL, (0, 1920, 1080)],
    # Level-2 entry band: every new object at NATIVE resolution.
    #
    # The L2 move limit is 551 px and adjacent L2 centres 480 px apart tile the
    # full width, so a pure-L2 sweep of the entry band is legal at one step per
    # frame -- seven frames for the whole width, against the ~7.8 frames an
    # object spends inside y in [0, 540]. That matters because the object's
    # size in the transmitted view is what decides detection: measured on real
    # detections, an object arriving under 10 px is found 30 % of the time and
    # one arriving at 20-30 px is found 95 % (probe/detect_curve.py). A ta-ta
    # arrives at ~8 px at L1 and at ~16-23 px at L2.
    #
    # The wrap is the cost: 3360 -> 480 is 2880 px, so the ring returns through
    # two L1 bridges, which are useful looks in their own right. Everything
    # below the band is then answered from memory, so this rests on the carry
    # holding up (85.8 % of boxes still hit after a 20-frame gap once the
    # motion is fitted). Idea from NORDIC_DRONE_PLAN.md section 7.1.
    'entry_ring': FULL_SWEEP + [TL] + [
        (2, 480, 270), (2, 960, 270), (2, 1440, 270), (2, 1920, 270),
        (2, 2400, 270), (2, 2880, 270), (2, 3360, 270),
        (1, 2880, 540),          # 3360,270 -> 2880,540 is 550.7, inside 551
        (2, 1920, 270),          # 2880,540 -> 1920,270 is 996.9, inside 1102
        (1, 1440, 540),          # 1920,270 -> 1440,540 is 550.7, inside 551
    ] * 80,
    # The same ring, entered only after several full sweeps, so objects already
    # on screen at frame 1 are acquired before the camera commits to the band.
    'entry_ring_late': FULL_SWEEP * 5 + [TL] + [
        (2, 480, 270), (2, 960, 270), (2, 1440, 270), (2, 1920, 270),
        (2, 2400, 270), (2, 2880, 270), (2, 3360, 270),
        (1, 2880, 540), (2, 1920, 270), (1, 1440, 540),
    ] * 80,
}
# Chosen on validation runs with v3 (same flight, same model):
# full 0.130, quad0 0.126, full0 0.119, dwell 0.117, top 0.108.
CAMERA = os.environ.get('DRONE_CAMERA', 'full')
# Every point as (level, x, y); plain (x, y) points are Level 1.
SWEEP = [p if len(p) == 3 else (1, *p) for p in SWEEPS.get(CAMERA, FULL_SWEEP)]
# Whether this pattern steers to Level 2 at all; see choose_next_view.
SWEEP_HAS_L2 = any(p[0] == 2 for p in SWEEP)

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

    for index, path in enumerate(ALT_MODEL_PATHS, start=1):
        if not path.exists():
            # Do not quietly serve fewer models than asked for: a half-loaded
            # pair answers 200s with plausible boxes and has cost an attempt.
            logger.error('No alternate model at %s: running %d model(s) only', path, len(_models))
            continue
        _models.append(_load_one(path, size_for(index)))
        for _ in range(2):
            raw_detections(np.zeros((540, 960, 3), np.uint8), index)
        logger.info('Loaded alternate %s at imgsz %d', path, size_for(index))
    if len(_models) > 1:
        logger.info('%d models loaded; %s', len(_models),
                    'all run on every frame' if BOTH_MODELS else 'they take alternate frames')
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
            # Which model produced each box. Concatenating threw this away.
            source = np.concatenate([np.full(len(part[0]), i, np.int8)
                                     for i, part in enumerate(parts)])
        else:
            xyxy, probabilities = raw_detections(image, frame)
            source = np.full(len(xyxy), frame % max(1, len(_models)), np.int8)
    rx1, ry1, rx2, ry2 = source_region
    height, width = image.shape[:2]
    scale = np.array([(rx2 - rx1) / width, (ry2 - ry1) / height] * 2)
    boxes = xyxy * scale + [rx1, ry1, rx1, ry1]
    return [
        (OBJECT_CLASSES[int(p.argmax())], float(p.max()), box, p, int(m))
        for box, p, m in zip(boxes, probabilities, source)
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
    # Indices of the models that have ever matched this track. See AGREEMENT_WEIGHT.
    models: set = field(default_factory=set)
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
    # Where the hybrid camera's Level-1 coverage phase has got to. Separate from
    # sweep_index, which the hybrid uses for the Level-2 snake.
    cover_index: int = 0
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
        source = detection[4] if len(detection) > 4 else 0
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
            best.models.add(source)
            if not truncated:
                best.seen_box, best.seen_frame = box.copy(), frame
            state.tracks.append(best)
        elif id(best) in matched:
            # A second detection of an object already handled this frame is a
            # class vote only; the first (most confident) one set the box. It is
            # also where model agreement shows up -- this is another model
            # finding the same object -- so the source is recorded before the
            # early return.
            best.models.add(source)
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
        best.models.add(source)
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
    # Kept apart from `annotations` on purpose: the floor band is appended after
    # the real answers are sorted, never sorted together with them. See FLOOR_ZERO.
    floor = []

    def scaled(box):
        if BOX_SCALE == 1.0:
            return box
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = (x2 - x1) * BOX_SCALE / 2, (y2 - y1) * BOX_SCALE / 2
        return np.array([cx - w, cy - h, cx + w, cy + h])

    def reported(box, name):
        """Track box -> the bbox we answer with, grown for this class.

        Growth is per class and a track answers several classes (the winner plus
        runner-ups), so it cannot be folded into the single box computed per
        track. It is applied here and nowhere else: the stored track box must
        stay tight, or matching, motion fitting and truncation all shift with it.
        """
        grow = BOX_GROW.get(name)
        if grow and grow != (1.0, 1.0):
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            w, h = (x2 - x1) * grow[0] / 2, (y2 - y1) * grow[1] / 2
            box = np.array([cx - w, cy - h, cx + w, cy + h])
        return clip_bbox_to_frame((
            box[0] / IMAGE_WIDTH, box[1] / IMAGE_HEIGHT,
            box[2] / IMAGE_WIDTH, box[3] / IMAGE_HEIGHT,
        ))

    def at_factor(box, name, factor):
        """The same box grown by an explicit factor instead of BOX_GROW.

        `factor` None means this class's measured Helsinki convention, which is
        the one the 1.3 cap throws away for medium_launcher and small_launcher.
        """
        if factor is None:
            factor = HELSINKI_BOX_FACTORS.get(name, 1.0)
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = (x2 - x1) * factor / 2, (y2 - y1) * factor / 2
        return clip_bbox_to_frame((
            (cx - w) / IMAGE_WIDTH, (cy - h) / IMAGE_HEIGHT,
            (cx + w) / IMAGE_WIDTH, (cy + h) / IMAGE_HEIGHT,
        ))

    for name, confidence, box in transient:
        bbox = reported(scaled(box), name)
        if bbox is not None:
            annotations.append(DroneFlybyPredictionDto(
                object_id=name, bbox=[round(c, 6) for c in bbox],
                confidence=round(float(np.clip(confidence, 0.001, 1.0)), 4),
            ))
    for track in state.tracks:
        box = scaled(track.box)
        # Whether the track is on screen at all is judged on the ungrown box, so
        # turning growth on never changes which tracks are answered.
        if reported(box, '') is None:
            continue
        ranked = sorted(track.votes.items(), key=lambda item: -item[1])
        base = track.best_confidence
        base *= min(1.0, HITS_BASE + HITS_STEP * track.hits)
        # Models that have never found this track. At the served
        # AGREEMENT_WEIGHT of 1.0 this is a no-op, byte for byte.
        if AGREEMENT_WEIGHT != 1.0 and len(_models) > 1:
            base *= AGREEMENT_WEIGHT ** max(0, len(_models) - len(track.models))
        # max(0, ...): a stale frame is behind tracks already moved to a later
        # one, and a negative exponent would *raise* the confidence above
        # best_confidence instead of decaying it.
        base *= UNSEEN_DECAY ** max(0, frame - track.last_seen)
        # Times the camera looked at this track and did not find it. At the served
        # MISS_PENALTY of 1.0 this is a no-op, byte for byte.
        if MISS_PENALTY != 1.0 and track.misses:
            base *= MISS_PENALTY ** track.misses
        if track.truncated:
            base *= TRUNCATED_WEIGHT
        # A track can hold only zero-weight votes (LEVEL_WEIGHT of 0 for the
        # level it was seen at), and dividing by that raised ZeroDivisionError
        # inside the caller's try, which discards the WHOLE frame's annotations
        # and its camera command -- a silent, total loss that looks like a
        # quiet frame. Fall back to equal shares instead.
        top_vote = ranked[0][1] or 1.0
        total_vote = sum(v for _, v in ranked) or 1.0
        # How much of this track's evidence points at its winning class.
        clean = (ranked[0][1] / total_vote) ** CLASS_SHARE_POWER if CLASS_SHARE_POWER else 1.0
        named = set()
        for rank, (name, vote) in enumerate(ranked[:1 + RUNNER_UPS]):
            share = vote / top_vote
            if rank and share < RUNNER_UP_SHARE:
                break
            named.add(name)
            confidence = base * clean if rank == 0 else base * clean * 0.9 * share
            bbox = reported(box, name)
            if bbox is None:
                continue
            annotations.append(DroneFlybyPredictionDto(
                object_id=name,
                bbox=[round(c, 6) for c in bbox],
                confidence=round(float(np.clip(confidence, 0.001, 1.0)), 4),
            ))
        if FLOOR_ALL_CLASSES:
            for name in OBJECT_CLASSES:
                if name not in named:
                    bbox = reported(box, name)
                    if bbox is None:
                        continue
                    annotations.append(DroneFlybyPredictionDto(
                        object_id=name,
                        bbox=[round(c, 6) for c in bbox],
                        confidence=round(float(np.clip(base * FLOOR_ALL_CLASSES, 0.001, 1.0)), 4),
                    ))
        if BAND:
            # Shape variants of the classes this track DID vote for. The
            # primary already went out at BOX_GROW; these are the growths it
            # is not allowed to use, appended where they cannot cost anything.
            def grade_for(rank):
                rank = rank if rank < BAND_HEDGE_RANK else rank + 1
                return BAND_GRADES[min(rank, len(BAND_GRADES) - 1)]

            for rank, factor in enumerate(BAND_SCALES):
                grade = grade_for(rank)
                for name in named:
                    bbox = at_factor(box, name, factor)
                    if bbox is not None:
                        floor.append(DroneFlybyPredictionDto(
                            object_id=name,
                            bbox=[round(c, 6) for c in bbox],
                            confidence=grade,
                        ))
            if BAND_HEDGE:
                # Classes the track never voted for, at their own convention.
                # Graded under every shape variant above, so the hedge can
                # never outrank a class we actually believe this track is.
                side = math.sqrt(max(1e-6, float(
                    (box[2] - box[0]) * (box[3] - box[1]))))
                grade = BAND_GRADES[BAND_HEDGE_RANK]
                for name in OBJECT_CLASSES:
                    if name in named:
                        continue
                    prior = CLASS_SIZE.get(name)
                    grown = side * HELSINKI_BOX_FACTORS.get(name, 1.0)
                    if prior and not (1 / BAND_HEDGE_TOL <= grown / prior <= BAND_HEDGE_TOL):
                        continue
                    bbox = at_factor(box, name, None)
                    if bbox is not None:
                        floor.append(DroneFlybyPredictionDto(
                            object_id=name,
                            bbox=[round(c, 6) for c in bbox],
                            confidence=grade,
                        ))
        if FLOOR_ZERO:
            # sqrt(w*h) of the track box, against the class's Helsinki size prior.
            side = math.sqrt(max(1e-6, float((box[2] - box[0]) * (box[3] - box[1]))))
            for name in OBJECT_CLASSES:
                if name in named:
                    continue
                prior = CLASS_SIZE.get(name)
                if prior and not (1 / FLOOR_SIZE_TOL <= side / prior <= FLOOR_SIZE_TOL):
                    continue
                bbox = reported(box, name)
                if bbox is None:
                    continue
                # Exactly 0.0, and deliberately NOT through the np.clip above:
                # clipping it to 0.001 would put it back in the band our own
                # faint answers live in, which is what cost 0.009 last time.
                floor.append(DroneFlybyPredictionDto(
                    object_id=name,
                    bbox=[round(c, 6) for c in bbox],
                    confidence=0.0,
                ))
    annotations.sort(key=lambda a: -a.confidence)
    # After the sort: every real answer outranks the whole band, and the 500 cap
    # then drops floor boxes rather than anything we actually believe. The band
    # is sorted separately so that the cap takes the faintest grades first --
    # sorting it TOGETHER with the answers is what FLOOR_ALL_CLASSES did, and it
    # cost 0.009.
    floor.sort(key=lambda a: -a.confidence)
    annotations.extend(floor)
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
        # Coverage. Leaving L2 costs one move, so rejoin at the nearest point --
        # but then ADVANCE along the sweep on every following coverage frame.
        #
        # This used to pick the nearest reachable Level-1 point every time, and
        # once the camera had arrived the nearest point was the one it was
        # already standing on. Driving choose_next_view through the evaluator's
        # own Camera showed the whole coverage phase as L1(2880,540) three times
        # running: HYBRID_COVER=3 bought one sixth of the frame, looked at
        # three times. Worst 60 px cell over a flight: 1 look, against 42 for
        # `full`. The 0.1234 the hybrid scored measured that, not Level 2.
        l1 = [p for p in SWEEP if p[0] == 1] or [(1, *FULL_FRAME_CENTER)]
        target = None
        if base[0] != 1:
            reachable = [(i, p) for i, p in enumerate(l1) if legal(base, *p)]
            if reachable:
                state.cover_index, target = min(
                    reachable,
                    key=lambda item: (item[1][1] - base[1]) ** 2 + (item[1][2] - base[2]) ** 2)
        else:
            for step in range(1, len(l1) + 1):
                index = (state.cover_index + step) % len(l1)
                if legal(base, *l1[index]):
                    state.cover_index, target = index, l1[index]
                    break
        if target is None:
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
    elif base[0] == 2 and not SWEEP_HAS_L2:
        # Back up to Level 1 as close as the 551 px limit allows. Only when the
        # pattern itself has no Level-2 points: this branch predates any L2
        # pattern and, left unguarded, it overrides them on the very first L2
        # frame. Measured with tools/camsim.py on the entry ring: 1 distinct L2
        # centre visited and 25 % of the top band covered, because every L2
        # arrival was immediately answered with a move back to L1.
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

    if target is not None and not legal(base, *target):
        # Shorten the move (and fix the level) instead of abandoning the
        # pattern: from Level 2 the full-view fallback below is an illegal
        # level change, so an unreachable target used to mean no command at
        # all, and the camera sat still.
        stepped = step_towards(base, *target)
        if stepped is not None:
            target = stepped
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
