"""Detect on captured NATIVE-resolution frames and cluster the hits in world metres.

A capture run parks the camera on one Level-2 tile, so every frame of it is
real native detail of one sixteenth of the frame -- the resolution our served
Level-1 pipeline never sees. Running the detector there and clustering by
ground position answers the question that decides where the remaining effort
goes: are there objects the flight contains that our pipeline cannot see?

    python probe/mine_capture.py --capture data/captures/<dir> --device cuda
"""

import argparse
import collections
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'probe'))

import geometry as G  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--capture', required=True)
    ap.add_argument('--model', default=str(ROOT / 'models/drone-yolo11m-v8.pt'))
    ap.add_argument('--model-alt', default=str(ROOT / 'models/drone-yolo11s-v6.pt'))
    ap.add_argument('--imgsz', type=int, default=1280)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--conf', type=float, default=0.30)
    ap.add_argument('--radius', type=float, default=12.0)
    ap.add_argument('--min-frames', type=int, default=4)
    ap.add_argument('--z', type=float, default=6.6)
    ap.add_argument('--fit', default=str(ROOT / 'data/probe/fit3d_scene.json'))
    ap.add_argument('--out', default=str(ROOT / 'data/probe/capture_objects.json'))
    a = ap.parse_args()

    os.environ['DRONE_MODEL'] = a.model
    os.environ['DRONE_MODEL_ALT'] = a.model_alt
    os.environ['DRONE_IMGSZ'] = f'{a.imgsz},{a.imgsz}'
    os.environ['DRONE_DEVICE'] = a.device
    os.environ['DRONE_DET_CONF'] = '0.05'
    import cv2
    import flyby
    from dtos import OBJECT_CLASSES
    flyby.load_model()
    print(f'{len(flyby._models)} models at imgsz {a.imgsz} on {a.device}')

    sightings = []
    metas = sorted(Path(a.capture).glob('*.json'))
    for n, meta_path in enumerate(metas):
        meta = json.loads(meta_path.read_text())
        png = meta_path.with_suffix('.png')
        if not png.exists() or meta['view']['resolution_level'] != 2:
            continue
        image = cv2.imread(str(png))
        rx1, ry1, rx2, ry2 = meta['view']['source_region_xyxy']
        scale = np.array([(rx2 - rx1) / image.shape[1], (ry2 - ry1) / image.shape[0]] * 2)
        for which in range(len(flyby._models)):
            xyxy, probs = flyby.raw_detections(image, which)
            for box, p in zip(xyxy, probs):
                if p.max() < a.conf:
                    continue
                source = box * scale + [rx1, ry1, rx1, ry1]
                cls = OBJECT_CLASSES[int(p.argmax())]
                height = G.CLASS_DIMS.get(cls, (5, 5, 3))[2]
                u, v = (source[0] + source[2]) / 2, (source[1] + source[3]) / 2
                point = G.ground_point(u, v, meta['frame'], z_ground=a.z - 0.35 * height)
                sightings.append({'frame': meta['frame'], 'cls': cls, 'conf': float(p.max()),
                                  'x': float(point[0]), 'y': float(point[1]),
                                  'side': float(np.sqrt(max(1.0, (source[2] - source[0]) * (source[3] - source[1]))))})
        if n % 50 == 0:
            print(f'  {n}/{len(metas)} frames, {len(sightings)} sightings', flush=True)

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
            c['side'] = float(np.median([m['side'] for m in c['members']]))
            kept.append(c)

    known = [(o['class'], o['params'][0], o['params'][1])
             for o in json.loads(Path(a.fit).read_text())['objects'] if o['fit_iou_median'] >= 0.6]
    new = [c for c in kept
           if not any(k[0] == c['cls'] and np.hypot(k[1] - c['x'], k[2] - c['y']) < 25 for k in known)]

    print(f'\n{len(sightings)} sightings -> {len(kept)} objects in >= {a.min_frames} frames; '
          f'{len(new)} not in the truth file')
    print(f'{"class":16s} {"found":>6s} {"new":>5s}')
    found_by = collections.Counter(c['cls'] for c in kept)
    new_by = collections.Counter(c['cls'] for c in new)
    for cls in sorted(found_by):
        print(f'{cls:16s} {found_by[cls]:6d} {new_by[cls]:5d}')
    print('\nstrongest NEW objects (native resolution):')
    for c in sorted(new, key=lambda c: -(c['n_frames'] * c['best_conf']))[:25]:
        print(f'  {c["cls"]:16s} frames {c["n_frames"]:3d} conf {c["best_conf"]:.2f} '
              f'side {c["side"]:5.1f} px  world ({c["x"]:7.1f}, {c["y"]:7.1f})')
    Path(a.out).write_text(json.dumps(
        {'objects': [{k: v for k, v in c.items() if k != 'members'} for c in kept]}, indent=1))


if __name__ == '__main__':
    main()
