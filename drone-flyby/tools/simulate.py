"""Simulate a whole evaluation attempt, offline and fast, to compare policies.

    python tools/simulate.py                         # defaults: 540 ms per answer
    python tools/simulate.py --latency-ms 400 --camera top --seeds 5

No HTTP, no images, no model. It replays the evaluator's clock the way the real
one behaves (seen in our validation recordings):

* one request at a time; the next one carries the newest emitted frame;
* a frame's view is rendered when the frame is emitted, so a camera command
  that arrives after that only shows up in the frame after;

over a synthetic 250-frame flight whose objects enter at the top and drift
with the fitted ground motion. The detector is a stand-in with rough, tunable
flaws: small objects are missed or misnamed more, boxes jitter, and a few false
alarms appear per view. Absolute scores mean little; differences between
policies and latencies are the point.

The scene is written to src/sim_<seed>/ (annotations only) so the official
scorer in local_evaluator.py can score it.
"""

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

FRAMES = 250
INTERVAL = 1 / 3
OBJECTS_IN_VIEW = 11          # about what a Helsinki frame holds

# Rough object sizes in source pixels (from the Helsinki labels).
SIZES = {
    'condor': (170, 170), 'hangar': (190, 130), 'helicopter': (115, 95),
    'jammer': (34, 44), 'jet_plane': (80, 60), 'large_launcher': (150, 110),
    'large_tower': (60, 45), 'medium_launcher': (48, 30), 'medium_plane': (56, 40),
    'mine_roller': (53, 60), 'small_launcher': (22, 30), 'small_plane': (44, 50),
    'small_tower': (58, 58), 'spacecraft': (45, 48), 'ta-ta': (30, 19), 'tank': (50, 30),
}


def make_scene(seed: int) -> str:
    import flyby
    from dtos import IMAGE_HEIGHT, IMAGE_WIDTH, OBJECT_CLASSES

    rng = random.Random(seed)
    name = f'sim_{seed}'
    folder = ROOT / 'src' / name
    (folder / 'annotations').mkdir(parents=True, exist_ok=True)
    (folder / 'images').mkdir(parents=True, exist_ok=True)

    def spawn(y_range):
        cls = rng.choice(OBJECT_CLASSES)
        w, h = SIZES[cls]
        x = rng.uniform(0, IMAGE_WIDTH - w)
        y = rng.uniform(*y_range)
        return [cls, np.array([x, y, x + w, y + h])]

    objects = [spawn((0, IMAGE_HEIGHT - 200)) for _ in range(OBJECTS_IN_VIEW)]
    lifetime = IMAGE_HEIGHT / 65
    for frame in range(FRAMES):
        if frame:
            for obj in objects:
                obj[1] = flyby.advance(obj[1], 1)
            if rng.random() < OBJECTS_IN_VIEW / lifetime:
                objects.append(spawn((-200, -20)))
        annotations = []
        for cls, box in objects:
            x1, y1 = max(box[0], 0), max(box[1], 0)
            x2, y2 = min(box[2], IMAGE_WIDTH), min(box[3], IMAGE_HEIGHT)
            if x2 - x1 >= 2 and y2 - y1 >= 2:
                annotations.append({'object_id': cls, 'bbox': [int(x1), int(y1), int(x2), int(y2)]})
        objects = [o for o in objects if o[1][1] < IMAGE_HEIGHT]
        (folder / 'annotations' / f'frame_{frame:06d}.json').write_text(
            json.dumps({'frame': frame, 'annotations': annotations}))
        marker = folder / 'images' / f'frame_{frame:06d}.png'
        if not marker.exists():
            marker.touch()     # frame_numbers() only globs the names
    return name


def fake_detector(rng, scene, frame, level, region, false_alarms):
    """Ground truth seen through a lossy detector."""
    from dtos import OBJECT_CLASSES
    from utils import load_annotations

    rx1, ry1, rx2, ry2 = region
    scale = 960 / (rx2 - rx1)
    found = []
    for a in load_annotations(frame, scene):
        x1, y1, x2, y2 = a['bbox']
        ix = max(0, min(x2, rx2) - max(x1, rx1))
        iy = max(0, min(y2, ry2) - max(y1, ry1))
        if ix * iy < 0.5 * (x2 - x1) * (y2 - y1):
            continue
        side = min(x2 - x1, y2 - y1) * scale       # pixels in the view
        p_detect = float(np.clip((side - 4) / 12, 0, 0.95))
        if rng.random() > p_detect:
            continue
        p_right = float(np.clip((side - 4) / 16, 0.3, 0.95))
        cls = a['object_id'] if rng.random() < p_right else rng.choice(OBJECT_CLASSES)
        jitter = 0.08 * np.array([x2 - x1, y2 - y1] * 2) * np.array([rng.gauss(0, 1) for _ in range(4)])
        box = np.array([max(x1, rx1), max(y1, ry1), min(x2, rx2), min(y2, ry2)], float) + jitter
        found.append((cls, float(np.clip(rng.gauss(0.35 + 0.5 * p_right, 0.1), 0.02, 0.99)), box))
    for _ in range(np.random.default_rng(rng.randrange(1 << 30)).poisson(false_alarms)):
        w = h = rng.uniform(15, 80) / scale
        x, y = rng.uniform(rx1, rx2 - w), rng.uniform(ry1, ry2 - h)
        found.append((rng.choice(OBJECT_CLASSES), rng.uniform(0.02, 0.45), np.array([x, y, x + w, y + h])))
    return found


def simulate(scene, latency_ms, jitter_ms, false_alarms, seed):
    import flyby
    import local_evaluator as le
    from dtos import DroneFlybyPredictRequestDto

    rng = random.Random(seed)
    current = {}
    flyby.decode_view = lambda view: None
    flyby.detect = lambda image, region, frame=0: fake_detector(
        rng, scene, current['frame'], current['level'], region, false_alarms)
    flyby._sequences.clear()

    camera = le.Camera()          # the real camera: moves are checked on it
    view = le.Camera()            # what a frame was rendered with
    history = [(0.0, (0, 1920, 1080))]     # camera state changes over time

    def camera_at(t):
        state = history[0][1]
        for when, value in history:
            if when <= t:
                state = value
        return state

    predictions, answered = {}, 0
    t, index, feedback = 0.0, 0, None
    while index < FRAMES:
        emitted = index * INTERVAL
        level, cx, cy = camera_at(emitted)
        view.resolution_level, view.center_x, view.center_y = level, cx, cy
        payload = le.build_request(index, index, view, '', feedback)
        payload['sequence_id'] = f'sim-{seed}'
        current.update(frame=index, level=level)
        response = flyby.predict(DroneFlybyPredictRequestDto(**payload))
        answered += 1
        t = max(t, emitted) + max(50.0, rng.gauss(latency_ms, jitter_ms)) / 1000
        predictions[index] = [
            {'object_id': a.object_id, 'bbox': le.global_bbox_to_source(a.bbox, 3840, 2160),
             'confidence': float(a.confidence)}
            for a in response.annotations
        ]
        if response.requested_view is not None:
            r = response.requested_view
            try:
                camera.apply(r.resolution_level, r.center_x, r.center_y)
                history.append((t, (r.resolution_level, r.center_x, r.center_y)))
                feedback = None
            except le.CameraRejection as exc:
                feedback = {'frame': index, 'requested_view': r.model_dump(), 'reason': str(exc)}
        # Only the newest emitted frame is sent next.
        index = max(index + 1, int(t / INTERVAL))
    mean_ap, _ = le.score(scene, predictions)
    return mean_ap, answered


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--latency-ms', type=float, nargs='*', default=[540.0])
    parser.add_argument('--jitter-ms', type=float, default=60.0)
    parser.add_argument('--false-alarms', type=float, default=0.3, help='Per view, on average.')
    parser.add_argument('--camera', nargs='*', default=['full', 'top', 'mixed'])
    parser.add_argument('--seeds', type=int, default=3)
    args = parser.parse_args()

    import contextlib
    import importlib
    import io
    import logging
    logging.disable(logging.WARNING)

    scenes = [make_scene(seed) for seed in range(args.seeds)]
    for latency in args.latency_ms:
        for camera in args.camera:
            os.environ['DRONE_CAMERA'] = camera
            import flyby
            importlib.reload(flyby)
            scores, answered = [], []
            for seed, scene in enumerate(scenes):
                with contextlib.redirect_stdout(io.StringIO()):
                    score, count = simulate(scene, latency, args.jitter_ms, args.false_alarms, seed)
                scores.append(score)
                answered.append(count)
            print(f'latency {latency:5.0f} ms  camera {camera:6s}  mAP {np.mean(scores):.3f} '
                  f'(± {np.std(scores):.3f})  answered {np.mean(answered):.0f}/{FRAMES}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
