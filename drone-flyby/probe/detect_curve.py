"""How likely is a look to find an object, given how big it arrives?

Measured on a recorded run with cached detections: for every truth object-frame
whose object was inside the requested view, did any pass put a box on it? The
answer, bucketed by the object's size IN THE TRANSMITTED IMAGE (source size
divided by 4, 2 or 1 for Level 0, 1, 2), is the curve a camera planner needs:
a look is only worth taking if the object arrives big enough to be found.

    python probe/detect_curve.py --run <id> --pass ... --pass ...
"""

import argparse
import collections
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'probe'))

import geometry as G  # noqa: E402
from replay_score import load_truth  # noqa: E402

DIVISOR = {0: 4.0, 1: 2.0, 2: 1.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--pass', dest='passes', action='append', required=True)
    ap.add_argument('--truth', default=str(ROOT / 'data/probe/fit3d_scene.json'))
    ap.add_argument('--truth-grow', type=float, default=1.3)
    ap.add_argument('--recordings', default=str(ROOT / 'data/recordings'))
    ap.add_argument('--cache', default=str(ROOT / 'data/det_cache'))
    ap.add_argument('--out', default=str(ROOT / 'data/probe/detect_curve.json'))
    a = ap.parse_args()

    truth = load_truth(a.truth, a.truth_grow)
    caches = []
    for spec in a.passes:
        path, _, size = spec.partition(':')
        caches.append(pickle.loads((Path(a.cache) / f'{a.run}__{Path(path).stem}_{size}.pkl').read_bytes()))

    buckets = collections.defaultdict(lambda: [0, 0])          # size bucket -> [looks, found]
    per_class = collections.defaultdict(lambda: [0, 0])
    by_level = collections.defaultdict(lambda: [0, 0])
    for meta_path in sorted((Path(a.recordings) / a.run).glob('*.json')):
        meta = json.loads(meta_path.read_text())
        frame = meta['frame']
        level = meta['view']['resolution_level']
        region = meta['view']['source_region_xyxy']
        rx1, ry1, rx2, ry2 = region
        scale = np.array([(rx2 - rx1) / 960, (ry2 - ry1) / 540] * 2)
        boxes = []
        for cache in caches:
            xyxy, probs = cache[meta_path.stem]
            keep = probs.max(axis=1) >= 0.01
            boxes.append(xyxy[keep] * scale + [rx1, ry1, rx1, ry1])
        dets = np.concatenate(boxes) if boxes else np.zeros((0, 4))
        for cls, box in truth.get(frame, []):
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            if not (rx1 <= cx <= rx2 and ry1 <= cy <= ry2):
                continue
            side = float(np.sqrt(max(1.0, (box[2] - box[0]) * (box[3] - box[1]))))
            arrived = side / DIVISOR[level]
            found = len(dets) and max((G.iou(box, d) for d in dets), default=0.0) >= 0.3
            bucket = int(np.clip(arrived // 5 * 5, 0, 80))
            buckets[bucket][0] += 1
            buckets[bucket][1] += int(found)
            per_class[cls][0] += 1
            per_class[cls][1] += int(found)
            by_level[level][0] += 1
            by_level[level][1] += int(found)

    print(f'{"arrived px":>10s} {"looks":>7s} {"found":>7s} {"p":>6s}')
    curve = {}
    for bucket in sorted(buckets):
        looks, found = buckets[bucket]
        curve[bucket] = found / max(1, looks)
        print(f'{bucket:6d}-{bucket+5:<3d} {looks:7d} {found:7d} {found/max(1,looks):6.3f}')
    print()
    for level in sorted(by_level):
        looks, found = by_level[level]
        print(f'  level {level}: {found}/{looks} = {found/max(1,looks):.3f}')
    print()
    for cls in sorted(per_class):
        looks, found = per_class[cls]
        print(f'  {cls:16s} {found:5d}/{looks:<5d} {found/max(1,looks):.3f}')
    Path(a.out).write_text(json.dumps({str(k): v for k, v in curve.items()}))
    print(f'\nwrote {a.out}')


if __name__ == '__main__':
    main()
