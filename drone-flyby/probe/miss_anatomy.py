"""Why each truth instance was missed: camera, detector, or tracker.

Every (frame, object) the pipeline failed to answer at IoU >= 0.5 is charged to
exactly one cause, in this order:

  in-view-undetected  the object was inside the requested view and no cached
                      detection of any pass overlapped it   -> DETECTOR
  in-view-dropped     it was in view, a detection did overlap it, but the
                      answer did not                        -> TRACKER
  never-seen-yet      the camera had not yet shown this object at all
                      (its first look is still ahead)       -> CAMERA (latency)
  out-of-view         it was seen earlier but is outside the view now, and the
                      carried answer is wrong or gone       -> CARRY/MEMORY

    python probe/miss_anatomy.py --run <id> --pass v4:960 --pass ...
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
from replay_score import load_truth, replay  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--tracker', default='flyby')
    ap.add_argument('--pass', dest='passes', action='append', required=True)
    ap.add_argument('--truth', default=str(ROOT / 'data/probe/fit3d_scene.json'))
    ap.add_argument('--truth-grow', type=float, default=1.3)
    ap.add_argument('--recordings', default=str(ROOT / 'data/recordings'))
    ap.add_argument('--cache', default=str(ROOT / 'data/det_cache'))
    ap.add_argument('--env', action='append', default=[])
    a = ap.parse_args()

    truth = load_truth(a.truth, a.truth_grow)
    predictions = replay(a.run, a.tracker, a.passes, a.recordings, a.cache, env=a.env)

    # Views and raw detections per frame, from the recording and the caches.
    views, dets = {}, {}
    caches = []
    for spec in a.passes:
        path, _, size = spec.partition(':')
        caches.append(pickle.loads((Path(a.cache) / f'{a.run}__{Path(path).stem}_{size}.pkl').read_bytes()))
    for meta_path in sorted((Path(a.recordings) / a.run).glob('*.json')):
        meta = json.loads(meta_path.read_text())
        frame, region = meta['frame'], meta['view']['source_region_xyxy']
        views[frame] = region
        boxes = []
        rx1, ry1, rx2, ry2 = region
        scale = np.array([(rx2 - rx1) / 960, (ry2 - ry1) / 540] * 2)
        for cache in caches:
            xyxy, probs = cache[meta_path.stem]
            keep = probs.max(axis=1) >= 0.01
            boxes.append(xyxy[keep] * scale + [rx1, ry1, rx1, ry1])
        dets[frame] = np.concatenate(boxes) if boxes else np.zeros((0, 4))

    frames = sorted(truth)
    seen_before = collections.defaultdict(bool)
    causes = collections.Counter()
    per_class = collections.defaultdict(collections.Counter)

    # An object is keyed by class plus its ground position, which the truth
    # file holds directly, so identity across frames is exact.
    for frame in frames:
        region = views.get(frame)
        for cls, box in truth[frame]:
            key = (cls, round(box[0] / 50), round(box[1] / 50 - frame * 1.3))
            answers = [b for name, _, b in predictions.get(frame, []) if name == cls]
            best = max((G.iou(box, b) for b in answers), default=0.0)
            in_view = False
            if region is not None:
                cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                in_view = region[0] <= cx <= region[2] and region[1] <= cy <= region[3]
            detected = in_view and len(dets.get(frame, ())) and max(
                (G.iou(box, d) for d in dets[frame]), default=0.0) >= 0.3
            if in_view and detected:
                seen_before[key] = True
            if best >= 0.5:
                causes['answered'] += 1
                per_class[cls]['answered'] += 1
                continue
            if in_view and not detected:
                cause = 'in-view-undetected'
            elif in_view:
                cause = 'in-view-dropped'
            elif not seen_before[key]:
                cause = 'never-shown-yet'
            else:
                cause = 'carried-wrong'
            causes[cause] += 1
            per_class[cls][cause] += 1

    order = ['answered', 'in-view-undetected', 'in-view-dropped', 'never-shown-yet', 'carried-wrong']
    total = sum(causes.values())
    print(f'\n{"class":16s} ' + ' '.join(f'{k[:13]:>15s}' for k in order))
    for cls in sorted(per_class):
        row = per_class[cls]
        n = sum(row.values())
        print(f'{cls:16s} ' + ' '.join(f'{row[k]:6d} {row[k]/max(1,n):6.1%}' for k in order))
    print(f'{"TOTAL":16s} ' + ' '.join(f'{causes[k]:6d} {causes[k]/max(1,total):6.1%}' for k in order))


if __name__ == '__main__':
    main()
