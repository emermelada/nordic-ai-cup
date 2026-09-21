"""Replay a recorded run through either tracker, score it, and say where it loses.

Detections come from probe/cache_dets.py, so both trackers see byte-identical
input and the only difference measured is the tracking and the answer policy.

    python probe/replay_score.py --run <id> --tracker flyby --pass v4:960 ...
    python probe/replay_score.py --run <id> --tracker flyby3d

The truth is a fitted-object file (probe/fit3d.py), whose boxes are official
convention for every frame each object is visible. It is mined by our own
models, so it is INCOMPLETE: read recall and the hit/near/miss split, and read
AP only when comparing two runs against the same truth.
"""

import argparse
import collections
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'probe'))

import geometry as G  # noqa: E402


def load_truth(path, grow=1.0):
    """{frame: [(class, box)]} from a fit3d file."""
    data = json.loads(Path(path).read_text())['objects']
    truth = collections.defaultdict(list)
    for obj in data:
        if obj['fit_iou_median'] < 0.6:
            continue
        x, y, z, yaw, k = obj['params']
        params = (x, y, z, *G.CLASS_DIMS[obj['class']], yaw)
        for frame in range(1, 250):
            box = G.shrink(G.box_at(params, frame, clip=False), k * grow)
            clipped = [max(0.0, box[0]), max(0.0, box[1]), min(G.W, box[2]), min(G.H, box[3])]
            if clipped[2] - clipped[0] > 1 and clipped[3] - clipped[1] > 1:
                truth[frame].append((obj['class'], np.array(clipped)))
    return truth


def replay(run, tracker_name, passes, recordings, cache_dir, both=True, env=(), only_top=False):
    for item in env:
        name, _, value = item.partition('=')
        os.environ[name] = value
    os.environ.setdefault('DRONE_MODEL', str(ROOT / 'models/drone-yolo11n-v4.pt'))
    import flyby
    from dtos import DroneFlybyPredictRequestDto

    caches = []
    for spec in passes:
        path, _, size = spec.partition(':')
        stem = Path(path).stem
        caches.append(pickle.loads((Path(cache_dir) / f'{run}__{stem}_{size}.pkl').read_bytes()))
    flyby._models = [None] * len(caches)          # so len(_models) is right for alternation
    flyby.decode_view = lambda view: None
    current = {}

    def fake_detect(image, region, frame: int = 0):
        # only_top: pretend the camera never looked below the top row, which is
        # the regime a top-row camera puts every object into once it descends.
        if only_top and (region[1] + region[3]) / 2 > 1000:
            return []
        if both:
            parts = [c[current['stem']] for c in caches]
            xyxy = np.concatenate([p[0] for p in parts])
            probs = np.concatenate([p[1] for p in parts])
            source = np.concatenate([np.full(len(p[0]), i, np.int8) for i, p in enumerate(parts)])
        else:
            xyxy, probs = caches[frame % len(caches)][current['stem']]
            source = np.full(len(xyxy), frame % len(caches), np.int8)
        rx1, ry1, rx2, ry2 = region
        scale = np.array([(rx2 - rx1) / 960, (ry2 - ry1) / 540] * 2)
        boxes = xyxy * scale + [rx1, ry1, rx1, ry1]
        out = []
        for box, p, m in zip(boxes, probs, source):
            if p.max() >= flyby.DETECTION_CONFIDENCE:
                out.append((flyby.OBJECT_CLASSES[int(p.argmax())], float(p.max()), box, p, int(m)))
        return out

    flyby.detect = fake_detect
    if tracker_name == 'flyby3d':
        import flyby3d as tracker
        tracker.flyby.detect = fake_detect
        # flyby3d imported decode_view by name, so patching flyby's is not enough.
        tracker.decode_view = lambda view: None
    else:
        tracker = flyby
    tracker._sequences.clear()

    predictions = collections.defaultdict(list)
    for meta_path in sorted((Path(recordings) / run).glob('*.json')):
        meta = json.loads(meta_path.read_text())
        meta.pop('response', None)
        meta.pop('answer_ms', None)
        meta.pop('received_at', None)
        meta['view']['image'] = ''
        current['stem'] = meta_path.stem
        response = tracker.predict(DroneFlybyPredictRequestDto(**meta))
        for a in response.annotations:
            box = np.array(a.bbox, float) * [G.W, G.H, G.W, G.H]
            predictions[meta['frame']].append((a.object_id, float(a.confidence), box))
    return predictions


def recall_report(predictions, truth):
    """Per class: share of truth instances answered at IoU >= 0.5, and how we miss."""
    stats = collections.defaultdict(lambda: [0, 0, 0])       # hit, near (0.2-0.5), nothing
    for frame, items in truth.items():
        answers = predictions.get(frame, [])
        for cls, box in items:
            same = [b for name, _, b in answers if name == cls]
            best = max((G.iou(box, b) for b in same), default=0.0)
            slot = 0 if best >= 0.5 else (1 if best >= 0.2 else 2)
            stats[cls][slot] += 1
    return stats


def coco_map(predictions, truth):
    from faster_coco_eval import COCO, COCOeval_faster
    frames = sorted(truth)
    classes = sorted({c for items in truth.values() for c, _ in items})
    cat = {c: i + 1 for i, c in enumerate(classes)}
    images = [{'id': f, 'file_name': f'{f}.png', 'width': G.W, 'height': G.H} for f in frames]
    anns, n = [], 1
    for f in frames:
        for c, b in truth[f]:
            anns.append({'id': n, 'image_id': f, 'category_id': cat[c],
                         'bbox': [b[0], b[1], b[2] - b[0], b[3] - b[1]],
                         'area': float((b[2] - b[0]) * (b[3] - b[1])), 'iscrowd': 0})
            n += 1
    gt = {'info': {}, 'licenses': [], 'images': images,
          'categories': [{'id': v, 'name': k} for k, v in cat.items()], 'annotations': anns}
    dets = []
    for f in frames:
        for name, score, b in predictions.get(f, []):
            if name in cat and b[2] > b[0] and b[3] > b[1]:
                dets.append({'image_id': f, 'category_id': cat[name],
                             'bbox': [b[0], b[1], b[2] - b[0], b[3] - b[1]], 'score': score})
    if not dets:
        return 0.0, {c: 0.0 for c in classes}
    coco_gt = COCO(gt)
    ev = COCOeval_faster(coco_gt, coco_gt.loadRes(dets), 'bbox')
    ev.params.imgIds = frames
    ev.params.catIds = list(cat.values())
    ev.params.iouThrs = np.array([0.5])
    ev.evaluate(); ev.accumulate()
    per = {}
    for i, c in enumerate(classes):
        p = ev.eval['precision'][0, :, i, 0, -1]
        valid = p[p > -1]
        per[c] = float(np.mean(valid)) if valid.size else 0.0
    return float(np.mean(list(per.values()))), per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--tracker', default='flyby', choices=['flyby', 'flyby3d'])
    ap.add_argument('--pass', dest='passes', action='append', required=True)
    ap.add_argument('--truth', default=str(ROOT / 'data/probe/fit3d_scene.json'))
    ap.add_argument('--truth-grow', type=float, default=1.0)
    ap.add_argument('--recordings', default=str(ROOT / 'data/recordings'))
    ap.add_argument('--cache', default=str(ROOT / 'data/det_cache'))
    ap.add_argument('--alternate', action='store_true', help='models take alternate frames')
    ap.add_argument('--env', action='append', default=[])
    ap.add_argument('--label', default='')
    ap.add_argument('--only-top', action='store_true')
    a = ap.parse_args()

    truth = load_truth(a.truth, a.truth_grow)
    predictions = replay(a.run, a.tracker, a.passes, a.recordings, a.cache,
                         both=not a.alternate, env=a.env, only_top=a.only_top)
    stats = recall_report(predictions, truth)
    total, per = coco_map(predictions, truth)
    n_boxes = sum(len(v) for v in predictions.values())
    print(f'\n=== {a.label or a.tracker}  ({n_boxes} answers, '
          f'{n_boxes / max(1, len(predictions)):.1f}/frame)')
    print(f'{"class":16s} {"n":>5s} {"hit":>6s} {"near":>6s} {"none":>6s}  {"recall":>6s} {"AP":>6s}')
    hits = nears = nones = 0
    for cls in sorted(stats):
        h, nr, no = stats[cls]
        hits += h; nears += nr; nones += no
        print(f'{cls:16s} {h+nr+no:5d} {h:6d} {nr:6d} {no:6d}  {h/max(1,h+nr+no):6.3f} {per.get(cls,0):6.3f}')
    n = max(1, hits + nears + nones)
    print(f'{"TOTAL":16s} {n:5d} {hits:6d} {nears:6d} {nones:6d}  {hits/n:6.3f} {total:6.3f}')
    print(f'recall {hits/n:.3f}   near-miss (box/carry error) {nears/n:.3f}   no answer {nones/n:.3f}')


if __name__ == '__main__':
    main()
