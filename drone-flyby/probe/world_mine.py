"""Mine objects by clustering detections in WORLD coordinates.

A rendered prop is static ground: every sighting of it, from any frame and any
view, maps to the same point on the ground. A false positive on a bush or a
roof does too -- but a detector artefact does not, and neither does anything
whose apparent world position drifts. Clustering in the world frame is
therefore a far sharper consistency filter than linking boxes in the image,
and it needs no mosaic: the camera model turns any view into world metres.

Its purpose is to answer a question no offline score can: how many objects does
our own detector see that our truth file does not contain?

    python probe/world_mine.py --run <id> --pass ... --pass ...
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


def world_of(box, frame, cls, z_ground):
    """Ground position of a detection, allowing for the class's height."""
    height = G.CLASS_DIMS.get(cls, (5, 5, 3))[2]
    u, v = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    point = G.ground_point(u, v, frame, z_ground=z_ground - 0.5 * height * 0.7)
    return float(point[0]), float(point[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--pass', dest='passes', action='append', required=True)
    ap.add_argument('--recordings', default=str(ROOT / 'data/recordings'))
    ap.add_argument('--cache', default=str(ROOT / 'data/det_cache'))
    ap.add_argument('--fit', default=str(ROOT / 'data/probe/fit3d_scene.json'))
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--radius', type=float, default=12.0, help='metres')
    ap.add_argument('--min-frames', type=int, default=4)
    ap.add_argument('--z', type=float, default=6.6, help='flight ground level')
    ap.add_argument('--out', default=str(ROOT / 'data/probe/world_objects.json'))
    a = ap.parse_args()

    caches = []
    for spec in a.passes:
        path, _, size = spec.partition(':')
        caches.append(pickle.loads((Path(a.cache) / f'{a.run}__{Path(path).stem}_{size}.pkl').read_bytes()))

    from dtos import OBJECT_CLASSES
    sightings = []
    for meta_path in sorted((Path(a.recordings) / a.run).glob('*.json')):
        meta = json.loads(meta_path.read_text())
        frame = meta['frame']
        rx1, ry1, rx2, ry2 = meta['view']['source_region_xyxy']
        scale = np.array([(rx2 - rx1) / 960, (ry2 - ry1) / 540] * 2)
        for cache in caches:
            xyxy, probs = cache[meta_path.stem]
            for box, p in zip(xyxy, probs):
                if p.max() < a.conf:
                    continue
                source_box = box * scale + [rx1, ry1, rx1, ry1]
                cls = OBJECT_CLASSES[int(p.argmax())]
                x, y = world_of(source_box, frame, cls, a.z)
                sightings.append({'frame': frame, 'cls': cls, 'conf': float(p.max()),
                                  'x': x, 'y': y, 'box': source_box.tolist()})

    # Greedy clustering in world metres, per class.
    clusters = []
    for s in sorted(sightings, key=lambda s: -s['conf']):
        best = None
        for c in clusters:
            if c['cls'] != s['cls']:
                continue
            d = np.hypot(c['x'] - s['x'], c['y'] - s['y'])
            if d < a.radius and (best is None or d < best[0]):
                best = (d, c)
        if best is None:
            clusters.append({'cls': s['cls'], 'x': s['x'], 'y': s['y'], 'members': [s]})
        else:
            c = best[1]
            c['members'].append(s)
            n = len(c['members'])
            c['x'] += (s['x'] - c['x']) / n
            c['y'] += (s['y'] - c['y']) / n

    kept = []
    for c in clusters:
        frames = {m['frame'] for m in c['members']}
        if len(frames) >= a.min_frames:
            c['n_frames'] = len(frames)
            c['best_conf'] = max(m['conf'] for m in c['members'])
            c['spread'] = float(np.median([np.hypot(m['x'] - c['x'], m['y'] - c['y']) for m in c['members']]))
            kept.append(c)

    # Which of these are already in the truth file?
    known = []
    for o in json.loads(Path(a.fit).read_text())['objects']:
        if o['fit_iou_median'] >= 0.6:
            known.append((o['class'], o['params'][0], o['params'][1]))
    new = []
    for c in kept:
        match = any(k[0] == c['cls'] and np.hypot(k[1] - c['x'], k[2] - c['y']) < 25 for k in known)
        if not match:
            new.append(c)

    by_class = collections.Counter(c['cls'] for c in kept)
    new_by_class = collections.Counter(c['cls'] for c in new)
    print(f'{len(sightings)} sightings -> {len(clusters)} clusters -> '
          f'{len(kept)} objects seen in >= {a.min_frames} frames')
    print(f'{len(known)} in the truth file, {len(new)} NOT in it\n')
    print(f'{"class":16s} {"mined":>6s} {"new":>5s}')
    for cls in sorted(by_class):
        print(f'{cls:16s} {by_class[cls]:6d} {new_by_class[cls]:5d}')
    print('\nstrongest objects missing from the truth file:')
    for c in sorted(new, key=lambda c: -(c['n_frames'] * c['best_conf']))[:20]:
        print(f'  {c["cls"]:16s} frames {c["n_frames"]:3d} best conf {c["best_conf"]:.2f} '
              f'spread {c["spread"]:5.1f} m  world ({c["x"]:7.1f}, {c["y"]:7.1f})')
    Path(a.out).write_text(json.dumps(
        {'objects': [{k: v for k, v in c.items() if k != 'members'} for c in kept]}, indent=1))


if __name__ == '__main__':
    main()
