"""Score tracker settings offline on the recorded validation runs.

    python tools/bench_recordings.py                                  # current settings
    python tools/bench_recordings.py --set TRUNCATED_WEIGHT=1 --set RUNNER_UPS=0

The model runs once per recorded view (cached in data/bench_cache/), then the
recorded requests are replayed through flyby.predict with the cached
detections. Answers are checked against the objects in
training/validation_objects.json, carried to every frame by the ground-motion
model. Those are only some of the objects in the flight, so this measures
recall on known objects (hit = right class and IoU >= 0.5), how the misses
split, and how many boxes are reported; it is not the real score.
"""

import argparse
import base64
import collections
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'training'))

RECORDINGS = ROOT / 'data' / 'recordings'
CACHE = ROOT / 'data' / 'bench_cache'
REACH = 30      # frames a known object's box is trusted from its nearest sighting


def cached_detections(model_path: Path):
    import cv2
    import flyby

    key = hashlib.sha1(model_path.read_bytes()).hexdigest()[:12] + f'_{flyby.IMGSZ}_{flyby.DETECTION_CONFIDENCE}'
    path = CACHE / f'{key}.pkl'
    out = pickle.loads(path.read_bytes()) if path.exists() else {}
    # New recordings since the cache was written are filled in.
    # as_posix: the lookup key is built with '/', which is not what str() gives
    # on Windows, and a mismatch silently reads as "no detections at all".
    missing = [png for png in sorted(RECORDINGS.glob('*/*.png'))
               if png.relative_to(RECORDINGS).as_posix() not in out]
    if missing:
        flyby.load_model()
        for png in missing:
            image = cv2.imread(str(png))
            with flyby._model_lock:
                out[png.relative_to(RECORDINGS).as_posix()] = flyby.raw_detections(image)
        CACHE.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(out))
    return out


# Objects that cannot move, so any drift they show is the ground motion model
# being wrong rather than the object going somewhere.
STATIC_CLASSES = ('large_tower', 'small_tower', 'hangar')


def truth_motion(objects):
    """Ground motion fitted on this flight's static objects.

    Carrying the ground truth with the same MOTION the tracker uses makes both
    drift together and cancels the error, which is exactly how a wrong motion
    model stayed invisible here: the bench reported carried boxes as fine while
    they no longer overlapped anything real. The truth gets its own fit.
    """
    samples = []
    for obj in objects:
        if obj['class'] not in STATIC_CLASSES:
            continue
        obs = sorted(obj['observations'], key=lambda o: o['frame'])
        for before, after in zip(obs, obs[1:]):
            gap = after['frame'] - before['frame']
            if not (0 < gap <= 20):
                continue
            p, q = np.array(before['box'], float), np.array(after['box'], float)
            samples.append(((p[0]+p[2])/2, (p[1]+p[3])/2,
                            ((q[0]+q[2])-(p[0]+p[2]))/2/gap,
                            ((q[1]+q[3])-(p[1]+p[3]))/2/gap))
    import flyby
    if len(samples) < 6:
        return flyby.MOTION
    data = np.array(samples)
    design = np.column_stack([np.ones(len(data)), data[:, 0], data[:, 1]])
    dx = np.linalg.lstsq(design, data[:, 2], rcond=None)[0]
    dy = np.linalg.lstsq(design, data[:, 3], rcond=None)[0]
    return (*dx, *dy)


def carry(box, steps, motion):
    """Move a box ``steps`` frames of ground motion, forwards or backwards."""
    a, b, c, d, e, f = motion
    forward = np.array([[1 + b, c], [e, 1 + f]])
    shift = np.array([a, d])
    step = (lambda p: p @ forward.T + shift) if steps > 0 else (
        lambda p, back=np.linalg.inv(forward): (p - shift) @ back.T)
    points = np.array([[box[0], box[1]], [box[2], box[3]]], float)
    for _ in range(abs(steps)):
        points = step(points)
    return points.reshape(-1)


def known_boxes(objects, frame, motion=None):
    import flyby

    motion = flyby.MOTION if motion is None else motion
    out = []
    for obj in objects:
        near = min(obj['observations'], key=lambda o: abs(o['frame'] - frame))
        if abs(near['frame'] - frame) > REACH:
            continue
        box = carry(np.array(near['box'], float), frame - near['frame'], motion)
        box = np.clip(box, 0, [3840, 2160, 3840, 2160])
        if box[2] - box[0] > 2 and box[3] - box[1] > 2:
            out.append((obj['id'], obj['class'], box))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', type=Path, default=Path.home() / 'models' / 'drone-yolo11n-v2.pt')
    parser.add_argument('--set', action='append', default=[], help='NAME=value flyby setting to override.')
    parser.add_argument('--sequences', nargs='*', default=None)
    parser.add_argument('--per-object', action='store_true')
    parser.add_argument('--per-class', action='store_true')
    parser.add_argument('--from-frame', type=int, default=0,
                        help='Only score frames from here on (e.g. 200: frames no model trained on).')
    args = parser.parse_args()

    os.environ['DRONE_MODEL'] = str(args.model)
    import logging
    logging.disable(logging.WARNING)
    import flyby
    from dtos import DroneFlybyPredictRequestDto

    for item in args.set:
        name, value = item.split('=', 1)
        setattr(flyby, name, type(getattr(flyby, name))(eval(value)))

    detections = cached_detections(args.model)
    objects = json.loads((ROOT / 'training' / 'validation_objects.json').read_text())['objects']
    gt_motion = truth_motion(objects)
    print(f'ground truth carried at {flyby.drift_at_centre(gt_motion):.2f} px/frame '
          f'(flyby prior: {flyby.drift_at_centre(flyby.MOTION):.2f})')
    sequences = args.sequences or sorted(p.name for p in RECORDINGS.iterdir() if len(list(p.glob('*.json'))) > 50)

    current = {}
    flyby.decode_view = lambda view: None

    def fake_detect(image, region, frame=0):
        xyxy, probabilities = detections[current['key']]
        rx1, ry1, rx2, ry2 = region
        scale = np.array([(rx2 - rx1) / 960, (ry2 - ry1) / 540] * 2)
        boxes = xyxy * scale + [rx1, ry1, rx1, ry1]
        return [(flyby.OBJECT_CLASSES[int(p.argmax())], float(p.max()), b, p)
                for b, p in zip(boxes, probabilities) if p.max() >= flyby.DETECTION_CONFIDENCE]

    flyby.detect = fake_detect
    totals = collections.Counter()
    per_object = collections.defaultdict(collections.Counter)
    per_class = collections.defaultdict(collections.Counter)
    reported = []
    for sequence in sequences:
        flyby._sequences.clear()
        for meta_path in sorted((RECORDINGS / sequence).glob('*.json')):
            meta = json.loads(meta_path.read_text())
            meta.pop('response', None)
            meta['view']['image'] = ''
            current['key'] = f'{sequence}/{meta_path.stem}.png'
            response = flyby.predict(DroneFlybyPredictRequestDto(**meta))
            answers = [(a.object_id, a.confidence,
                        np.array(a.bbox) * [3840, 2160, 3840, 2160]) for a in response.annotations]
            reported.append(len(answers))
            if meta['frame'] < args.from_frame:
                continue
            for oid, cls, box in known_boxes(objects, meta['frame'], gt_motion):
                totals['present'] += 1
                right = [c for n, c, b in answers if n == cls and flyby.iou(b, box) >= 0.5]
                best = max((flyby.iou(b, box) for _, _, b in answers), default=0.0)
                outcome = 'hit' if right else 'wrong_class' if best >= 0.5 else 'bad_box' if best >= 0.2 else 'missing'
                totals[outcome] += 1
                per_object[oid][outcome] += 1
                per_class[cls][outcome] += 1
                if right:
                    # Rank of the right answer among everything reported this frame.
                    totals['rank_sum'] += sum(c > max(right) for _, c, _ in answers)

    n = max(1, totals['present'])
    print(f"known-object frames {n}: " + '  '.join(
        f'{k} {totals[k] / n:.1%}' for k in ('hit', 'wrong_class', 'bad_box', 'missing')))
    print(f'reported per frame: mean {np.mean(reported):.1f}  max {max(reported)}'
          f'   mean rank of hits {totals["rank_sum"] / max(1, totals["hit"]):.1f}')
    if args.per_class:
        for cls, counts in sorted(per_class.items()):
            total = sum(counts.values())
            print(f"  {cls:16s} n={total:4d}  hit {counts['hit'] / total:5.1%}  "
                  f"wrong class {counts['wrong_class'] / total:5.1%}  missing {counts['missing'] / total:5.1%}")
    if args.per_object:
        for obj in objects:
            print(f"  #{obj['id']:2d} {obj['class']:12s} {dict(per_object[obj['id']])}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
