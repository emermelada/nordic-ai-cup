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
    if path.exists():
        return pickle.loads(path.read_bytes())
    flyby.load_model()
    out = {}
    for png in sorted(RECORDINGS.glob('*/*.png')):
        image = cv2.imread(str(png))
        with flyby._model_lock:
            out[str(png.relative_to(RECORDINGS))] = flyby.raw_detections(image)
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(out))
    return out


def known_boxes(objects, frame):
    from harvest_validation_patches import move

    out = []
    for obj in objects:
        near = min(obj['observations'], key=lambda o: abs(o['frame'] - frame))
        if abs(near['frame'] - frame) > REACH:
            continue
        box = move(near['box'], frame - near['frame'])
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
    sequences = args.sequences or sorted(p.name for p in RECORDINGS.iterdir() if len(list(p.glob('*.json'))) > 50)

    current = {}
    flyby.decode_view = lambda view: None

    def fake_detect(image, region):
        xyxy, probabilities = detections[current['key']]
        rx1, ry1, rx2, ry2 = region
        scale = np.array([(rx2 - rx1) / 960, (ry2 - ry1) / 540] * 2)
        boxes = xyxy * scale + [rx1, ry1, rx1, ry1]
        return [(flyby.OBJECT_CLASSES[int(p.argmax())], float(p.max()), b, p)
                for b, p in zip(boxes, probabilities) if p.max() >= flyby.DETECTION_CONFIDENCE]

    flyby.detect = fake_detect
    totals = collections.Counter()
    per_object = collections.defaultdict(collections.Counter)
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
            for oid, cls, box in known_boxes(objects, meta['frame']):
                totals['present'] += 1
                right = [c for n, c, b in answers if n == cls and flyby.iou(b, box) >= 0.5]
                best = max((flyby.iou(b, box) for _, _, b in answers), default=0.0)
                outcome = 'hit' if right else 'wrong_class' if best >= 0.5 else 'bad_box' if best >= 0.2 else 'missing'
                totals[outcome] += 1
                per_object[oid][outcome] += 1
                if right:
                    # Rank of the right answer among everything reported this frame.
                    totals['rank_sum'] += sum(c > max(right) for _, c, _ in answers)

    n = totals['present']
    print(f"known-object frames {n}: " + '  '.join(
        f'{k} {totals[k] / n:.1%}' for k in ('hit', 'wrong_class', 'bad_box', 'missing')))
    print(f'reported per frame: mean {np.mean(reported):.1f}  max {max(reported)}'
          f'   mean rank of hits {totals["rank_sum"] / max(1, totals["hit"]):.1f}')
    if args.per_object:
        for obj in objects:
            print(f"  #{obj['id']:2d} {obj['class']:12s} {dict(per_object[obj['id']])}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
