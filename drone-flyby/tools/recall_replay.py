"""Per-frame recall on the confirmed objects, replayed offline.

    python tools/recall_replay.py --run <id> --build     # one-time detection cache
    python tools/recall_replay.py --run <id>             # score the served config
    python tools/recall_replay.py --run <id> --set BOX_GROW_CAP=1.15

Why this exists, and why it is not score_offline.py: the truth file is missing
many real objects, so offline *precision* -- and therefore offline AP -- is
measurement error (HANDOVER, 2026-09-19 17:45). That argument does not touch
**recall on the 32 objects we have confirmed**: an unlabelled object cannot
turn a real one into a miss. So this tool reports recall only, and breaks the
misses down into the two things that cause them:

    hit                     IoU >= 0.5
    near miss               a same-class box is there and centred, but wrong size
    nothing                 no same-class box within two box sides

Measured on the served 0.527 configuration (run b5544ad3), this reads 59 % hit,
28 % near miss, 13 % nothing -- the near misses are centred to 0.07 box sides
and 1.49x too large. **Never quote a precision or an AP from this tool.**

The size ratio is against training/validation_objects.json, whose boxes are our
own tight detections, NOT the evaluator's convention. So the 1.49 is a ratio
between passes and against a tight truth -- it does not prove the served boxes
are too big for the grader, and real runs say flat 1.3 is near-optimal. Use it
to compare passes and configurations with each other.
"""

import argparse
import collections
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'training'))

RECORDINGS = ROOT / 'data' / 'recordings'
CACHE = ROOT / 'data' / 'bench_cache'
OBJECTS = ROOT / 'training' / 'validation_objects.json'
W, H = 3840, 2160

# The served stack, as HANDOVER's START HERE block runs it.
SERVED = [('models/drone-yolo11n-v4.pt', 960), ('models/drone-yolo11s-v6.pt', 1280),
          ('models/drone-yolo11m-v8.pt', 1280), ('models/drone-yolo11m-v8.pt', 2560)]


def cache_path(run: str, passes) -> Path:
    key = hashlib.sha1(('|'.join(f'{p}@{s}' for p, s in passes)).encode()).hexdigest()[:12]
    return CACHE / f'recall_{run[:8]}_{key}.pkl'


def build_cache(run: str, passes, device: str):
    """raw_detections for every recorded view of ``run``, one entry per pass.

    Per-run on purpose: bench_recordings.cached_detections fills in every
    recording it can find, which is ~8000 images per pass and hours on a CPU box.
    """
    import os
    os.environ['DRONE_MODEL'] = passes[0][0]
    os.environ['DRONE_MODEL_ALT'] = ','.join(p for p, _ in passes[1:])
    os.environ['DRONE_IMGSZ'] = ','.join(str(s) for _, s in passes)
    os.environ['DRONE_DEVICE'] = device
    import cv2
    import flyby
    flyby.load_model()
    if len(flyby._models) != len(passes):
        raise SystemExit(f'loaded {len(flyby._models)} models, asked for {len(passes)}')
    pngs = sorted((RECORDINGS / run).glob('*.png'))
    out = {}
    for n, png in enumerate(pngs, 1):
        image = cv2.imread(str(png))
        for i in range(len(passes)):
            out[(png.stem, i)] = flyby.raw_detections(image, i)
        if n % 20 == 0:
            print(f'  {n}/{len(pngs)} views', flush=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    path = cache_path(run, passes)
    path.write_bytes(pickle.dumps(out))
    print(f'wrote {path} ({len(pngs)} views x {len(passes)} passes)')


def replay(run: str, passes, overrides):
    """Recorded requests back through flyby.predict, detections from the cache."""
    import os
    os.environ['DRONE_MODEL'] = passes[0][0]
    os.environ['DRONE_MODEL_ALT'] = ','.join(p for p, _ in passes[1:])
    os.environ['DRONE_IMGSZ'] = ','.join(str(s) for _, s in passes)
    os.environ.setdefault('DRONE_BOX_GROW', '1.3')
    os.environ.setdefault('DRONE_BOX_GROW_CAP', '1.3')
    import flyby
    from dtos import DroneFlybyPredictRequestDto

    flyby.BOTH_MODELS = 1
    flyby.NEW_TRACK_CONFIDENCE = 0.10
    for item in overrides:
        name, value = item.split('=', 1)
        setattr(flyby, name, type(getattr(flyby, name))(eval(value)))

    cache = pickle.loads(cache_path(run, passes).read_bytes())
    current = {}

    def fake_detect(image, region, frame: int = 0):
        parts = [cache[(current['stem'], i)] for i in range(len(passes))]
        xyxy = np.concatenate([p[0] for p in parts])
        probabilities = np.concatenate([p[1] for p in parts])
        rx1, ry1, rx2, ry2 = region
        scale = np.array([(rx2 - rx1) / 960, (ry2 - ry1) / 540] * 2)
        boxes = xyxy * scale + [rx1, ry1, rx1, ry1]
        return [(flyby.OBJECT_CLASSES[int(p.argmax())], float(p.max()), b, p)
                for b, p in zip(boxes, probabilities) if p.max() >= flyby.DETECTION_CONFIDENCE]

    flyby.decode_view = lambda view: None
    flyby.detect = fake_detect
    flyby._sequences.clear()
    answers = collections.defaultdict(list)
    for meta_path in sorted((RECORDINGS / run).glob('*.json')):
        meta = json.loads(meta_path.read_text())
        meta.pop('response', None)
        meta['view']['image'] = ''
        current['stem'] = meta_path.stem
        response = flyby.predict(DroneFlybyPredictRequestDto(**meta))
        for a in response.annotations:
            box = np.array(a.bbox, float) * [W, H, W, H]
            answers[meta['frame']].append((a.object_id, box, a.confidence))
    return answers


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def trajectory(obs):
    """Per-object quadratic through its own observations, at every frame in span.

    The observations only exist on frames where the object was IN VIEW (they are
    v2 detections), so scoring them alone measures freshly detected boxes and is
    blind to the ~2/3 of an object's life where the box is carried by the motion
    model. A quadratic in frame index fits the real trajectories to 2.6 px RMS
    over 21-31 frame spans (measured 2026-09-19), so interpolating inside the
    observed span is trustworthy truth for those carried frames. Interpolation
    only -- never extrapolated past the span, where the residual grows.
    """
    frames = sorted(obs)
    if len(frames) < 4:
        return {f: np.array(obs[f], float) for f in frames}
    f = np.array(frames, float)
    deg = 2 if len(frames) >= 6 else 1
    corners = [np.polyfit(f, np.array([obs[k][i] for k in frames], float), deg) for i in range(4)]
    return {int(k): np.array([np.polyval(c, k) for c in corners])
            for k in range(frames[0], frames[-1] + 1)}


def score(answers, min_obs: int, carry: bool = False):
    """Recall and the miss breakdown, on frames where an object was observed."""
    objects = json.loads(OBJECTS.read_text())['objects']
    buckets = collections.Counter()
    offsets, ratios, per_class = [], [], collections.defaultdict(lambda: [0, 0])
    per_bucket = collections.defaultdict(collections.Counter)
    ratio_class = []
    hit_conf = collections.defaultdict(list)
    for o in objects:
        obs = {}
        for ob in o['observations']:
            obs.setdefault(ob['frame'], ob['box'])
        if len(obs) < min_obs:
            continue
        if carry:
            obs = trajectory(obs)
        for frame, truth in obs.items():
            truth = np.array(truth, float)
            centre = np.array([(truth[0] + truth[2]) / 2, (truth[1] + truth[3]) / 2])
            side = ((truth[2] - truth[0]) * (truth[3] - truth[1])) ** 0.5
            near = [(iou(b, truth), b, c) for name, b, c in answers.get(frame, ())
                    if name == o['class']
                    and np.hypot((b[0] + b[2]) / 2 - centre[0], (b[1] + b[3]) / 2 - centre[1]) < 2 * side]
            per_class[o['class']][1] += 1
            if not near:
                buckets['nothing'] += 1
                per_bucket[o['class']]['nothing'] += 1
                continue
            best, box, bconf = max(near, key=lambda t: t[0])
            if best >= 0.5:
                hit_conf[o['class']].append(bconf)
            if best >= 0.5:
                buckets['hit'] += 1
                per_class[o['class']][0] += 1
                per_bucket[o['class']]['hit'] += 1
            else:
                buckets['near' if best >= 0.2 else 'misplaced'] += 1
                per_bucket[o['class']]['near'] += 1
                bc = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
                offsets.append(float(np.hypot(*(bc - centre)) / side))
                ratios.append((((box[2] - box[0]) * (box[3] - box[1])) ** 0.5) / side)
                ratio_class.append(o['class'])
    total = sum(buckets.values())
    print(f'\nobject-frames with confirmed truth: {total}')
    for key, label in (('hit', 'hit (IoU>=0.5)'), ('near', 'near miss (0.2<=IoU<0.5)'),
                       ('misplaced', 'present, badly placed (IoU<0.2)'),
                       ('nothing', 'nothing of that class nearby')):
        print(f'  {buckets[key]:>4}  {buckets[key] / total * 100:5.1f}%  {label}')
    print(f'\nRECALL@0.5 = {buckets["hit"] / total:.3f}')
    if offsets:
        print(f'on the {len(offsets)} misses with a same-class box nearby:'
              f'  centre offset {np.median(offsets):.2f} box sides,'
              f'  size ratio {np.median(ratios):.2f}')
    sent = collections.defaultdict(list)
    for boxes in answers.values():
        for name, _b, c in boxes:
            sent[name].append(c)
    print('\nRANKING -- boxes of a class that outrank that class\'s median confirmed hit')
    print(f'  {"class":<15} {"sent":>7} {"hits":>5} {"median hit conf":>16} {"outranking":>11}')
    for name in sorted(hit_conf):
        med = float(np.median(hit_conf[name]))
        above = int((np.array(sent[name]) > med).sum())
        print(f'  {name:<15} {len(sent[name]):>7} {len(hit_conf[name]):>5} {med:>16.3f} {above:>11}')
    print(f'\nper class   {"hit":>5} {"near miss":>10} {"nothing":>8}   {"median size ratio":>17}')
    for name in sorted(per_class):
        b = per_bucket[name]
        rs = [r for r, n in zip(ratios, ratio_class) if n == name]
        rt = f'{np.median(rs):.2f}' if rs else '-'
        print(f'  {name:<15} {b["hit"]:>4} {b["near"]:>10} {b["nothing"]:>8}   {rt:>17}')
    return buckets['hit'] / total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', required=True)
    parser.add_argument('--build', action='store_true', help='Compute the detection cache and exit.')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--min-obs', type=int, default=5,
                        help='Only objects with at least this many observations (default 5).')
    parser.add_argument('--set', dest='overrides', action='append', default=[],
                        metavar='NAME=value', help='Override a flyby setting, e.g. BOX_GROW_CAP=1.15')
    parser.add_argument('--carry', action='store_true',
                        help='Also score the carried frames between observations (see trajectory()).')
    parser.add_argument('--stored', action='store_true',
                        help='Score the answers the run really sent, instead of replaying.')
    args = parser.parse_args()

    passes = [(str(ROOT / p), s) for p, s in SERVED]
    if args.build:
        build_cache(args.run, passes, args.device)
        return 0
    if args.stored:
        answers = collections.defaultdict(list)
        for meta_path in sorted((RECORDINGS / args.run).glob('*.json')):
            meta = json.loads(meta_path.read_text())
            for a in (meta.get('response') or {}).get('annotations', []):
                bb = a['bbox']
                answers[meta['frame']].append((a['object_id'], np.array([bb[0] * W, bb[1] * H, bb[2] * W, bb[3] * H]), a['confidence']))
    else:
        if not cache_path(args.run, passes).exists():
            raise SystemExit('no detection cache for this run; run with --build first')
        answers = replay(args.run, passes, args.overrides)
    score(answers, args.min_obs, args.carry)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
