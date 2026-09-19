"""Compare camera patterns on the one thing that costs the most: time to first look.

Charged against the fitted objects (probe/fit3d.py), which give every object's
exact box in every frame, so no imagery is needed. The evaluator's own movement
rules are applied, including the one-frame command latency: a command sent with
the answer to frame t only takes effect for frame t+1.

    python probe/camera_plan.py --pattern full --pattern toprow --pattern row0

Reported per pattern:
  first-look lag   frames between an object appearing and the camera showing it
  shown share      share of object-frames where the object is inside the view
  answerable       share of object-frames at or after that object's first look
                   -- the ceiling on recall for that camera, with a perfect
                   detector and perfect carry
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'probe'))

import geometry as G  # noqa: E402
from utils import center_bounds_for_level, describe_camera_rejection, source_region_for_view  # noqa: E402

TL, TM, TR = (1, 960, 540), (1, 1920, 540), (1, 2880, 540)
BL, BM, BR = (1, 960, 1620), (1, 1920, 1620), (1, 2880, 1620)
L0 = (0, 1920, 1080)
ML, MR = (1, 960, 1080), (1, 2880, 1080)

def _flyby_patterns():
    """The real patterns, read from flyby so the two cannot drift apart.

    Every step here is checked against the evaluator's rules: a bare TL/TR
    alternation is 1920 px against an L1 limit of 1102 and stalls the camera
    on the spot, which no simulation that ignores the rules would show.
    """
    import importlib
    import os
    out = {}
    for name in ('full', 'top_mostly', 'full0', 'quad0', 'entry_ring', 'entry_ring_late'):
        os.environ['DRONE_CAMERA'] = name
        import flyby
        importlib.reload(flyby)
        out[name] = list(flyby.SWEEP)
        seq = out[name] + [out[name][0]]
        for a, b in zip(seq, seq[1:]):
            if describe_camera_rejection(a[0], (a[1], a[2]), b[0], (b[1], b[2])):
                raise SystemExit(f'pattern {name} has an illegal step {a} -> {b}')
    return out


PATTERNS = _flyby_patterns()


def simulate(pattern, frames):
    """Return {frame: (level, cx, cy)} actually rendered, under the real rules."""
    level, cx, cy = 0, 1920, 1080
    index = 0
    pending = None
    out = {}
    for frame in frames:
        # A command from the previous answer lands before this frame is rendered.
        if pending is not None:
            if describe_camera_rejection(level, (cx, cy), pending[0], (pending[1], pending[2])) is None:
                level, cx, cy = pending
            pending = None
        out[frame] = (level, cx, cy)
        # Plan the next view from where the camera will be.
        target = pattern[index % len(pattern)]
        if (level, cx, cy) == target:
            index += 1
            target = pattern[index % len(pattern)]
        if describe_camera_rejection(level, (cx, cy), target[0], (target[1], target[2])) is not None:
            # Shorten the step until it is legal, like flyby.step_towards.
            lo_x, hi_x, lo_y, hi_y = center_bounds_for_level(target[0])
            done = False
            for ratio in (0.9, 0.75, 0.6, 0.45, 0.3):
                nx = int(min(max(cx + (target[1] - cx) * ratio, lo_x), hi_x))
                ny = int(min(max(cy + (target[2] - cy) * ratio, lo_y), hi_y))
                if describe_camera_rejection(level, (cx, cy), target[0], (nx, ny)) is None:
                    pending = (target[0], nx, ny)
                    done = True
                    break
            if not done:
                pending = None
        else:
            pending = target
    return out


DIVISOR = {0: 4.0, 1: 2.0, 2: 1.0}

# Measured on real detections (probe/carry_test.py, affine_fitted column): the
# chance a carried box still overlaps the truth at IoU 0.5, by frames since the
# last sighting. A camera that never looks back is only as good as this curve.
CARRY_HIT_FITTED = {0: 0.97, 1: 0.956, 3: 0.961, 6: 0.926, 10: 0.916, 15: 0.871,
                    20: 0.858, 25: 0.768, 30: 0.481, 40: 0.30, 60: 0.15}
# probe/carry_test.py, model3d+oracle_z column: what an exact carry with the
# object's own elevation delivers. This is the precondition the L2 ring rests on.
CARRY_HIT_EXACT = {0: 0.98, 1: 0.961, 3: 0.967, 6: 0.956, 10: 0.963, 15: 0.930,
                   20: 0.915, 25: 0.884, 30: 0.556, 40: 0.40, 60: 0.20}
CARRY_HIT = CARRY_HIT_FITTED


def carry_hit(gap):
    edges = sorted(CARRY_HIT)
    lo = max([e for e in edges if e <= gap], default=edges[0])
    hi = min([e for e in edges if e >= gap], default=edges[-1])
    if lo == hi:
        return CARRY_HIT[lo]
    t = (gap - lo) / (hi - lo)
    return CARRY_HIT[lo] * (1 - t) + CARRY_HIT[hi] * t


def load_curve(path):
    """Detection probability by arrived size, from probe/detect_curve.py."""
    raw = json.loads(Path(path).read_text())
    buckets = sorted((int(k), v) for k, v in raw.items())

    def p(arrived):
        best = 0.0
        for edge, value in buckets:
            if arrived >= edge:
                best = value
        # Never credit more than measured at the top of the curve.
        return float(np.clip(best, 0.0, 0.98))
    return p


def evaluate(pattern_name, objects, frames, margin=0.6, curve=None):
    views = simulate(PATTERNS[pattern_name], frames)
    lags, shown, answerable, total, never = [], 0, 0, 0, 0
    expected = 0.0
    for obj in objects:
        params = (obj['x'], obj['y'], obj['z'], *G.CLASS_DIMS[obj['cls']], obj['yaw'])
        first_visible, first_look = None, None
        life = 0
        undetected = 1.0        # probability the object has NOT been found yet
        last_look = None
        for frame in frames:
            box = G.box_at(params, frame)
            if not G.visible(box):
                continue
            life += 1
            if first_visible is None:
                first_visible = frame
            level, cx, cy = views[frame]
            region = source_region_for_view(level, cx, cy)
            ix = max(0, min(box[2], region[2]) - max(box[0], region[0]))
            iy = max(0, min(box[3], region[3]) - max(box[1], region[1]))
            area = max(1e-6, (box[2] - box[0]) * (box[3] - box[1]))
            if ix * iy / area >= margin:
                shown += 1
                if first_look is None:
                    first_look = frame
                if curve is not None:
                    side = float(np.sqrt(area)) / DIVISOR[level]
                    if curve(side) > 0.25:      # a look too coarse to find it is not a sighting
                        last_look = frame
                    undetected *= (1.0 - curve(side))
                else:
                    last_look = frame
            if first_look is not None and frame >= first_look:
                answerable += 1
            # Answered only if it has been found AND the carry since the last
            # sighting still lands on it.
            expected += (1.0 - undetected) * carry_hit(frame - last_look if last_look is not None else 0)
        total += life
        if first_look is None:
            never += 1
        elif first_visible is not None:
            lags.append(first_look - first_visible)
    return {
        'pattern': pattern_name,
        'objects': len(objects),
        'never_shown': never,
        'lag_median': float(np.median(lags)) if lags else float('nan'),
        'lag_mean': float(np.mean(lags)) if lags else float('nan'),
        'shown_share': shown / max(1, total),
        'answerable': answerable / max(1, total),
        'expected': expected / max(1, total),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fit', default=str(ROOT / 'data/probe/fit3d_scene.json'))
    ap.add_argument('--pattern', action='append', default=[])
    ap.add_argument('--frames', type=int, default=249)
    ap.add_argument('--margin', type=float, default=0.6)
    ap.add_argument('--curve', default=str(ROOT / 'data/probe/detect_curve.json'))
    ap.add_argument('--carry', default='fitted', choices=['fitted', 'exact'])
    a = ap.parse_args()

    data = json.loads(Path(a.fit).read_text())['objects']
    objects = []
    for o in data:
        if o['fit_iou_median'] < 0.6:
            continue
        x, y, z, yaw, k = o['params']
        objects.append({'cls': o['class'], 'x': x, 'y': y, 'z': z, 'yaw': yaw})
    frames = list(range(1, a.frames + 1))
    names = a.pattern or list(PATTERNS)
    print(f'{len(objects)} fitted objects, {len(frames)} frames\n')
    print(f'{"pattern":12s} {"never":>6s} {"lag med":>8s} {"shown":>7s} {"ceiling":>8s} {"expected":>9s}')
    global CARRY_HIT
    CARRY_HIT = CARRY_HIT_EXACT if a.carry == 'exact' else CARRY_HIT_FITTED
    curve = load_curve(a.curve) if Path(a.curve).exists() else None
    rows = [evaluate(n, objects, frames, a.margin, curve) for n in names]
    for r in sorted(rows, key=lambda r: -r['expected']):
        print(f'{r["pattern"]:12s} {r["never_shown"]:6d} {r["lag_median"]:8.1f} '
              f'{r["shown_share"]:7.3f} {r["answerable"]:8.3f} {r["expected"]:9.3f}')


if __name__ == '__main__':
    main()
