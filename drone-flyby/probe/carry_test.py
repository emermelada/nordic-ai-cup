"""How well does a box survive being carried k frames without a new sighting?

This is the question the camera policy turns on. A top-row camera detects an
object early and then never looks at it again, so it is only worth having if a
carried box still overlaps the truth 20-30 frames later.

Compared, on real detections matched to fitted objects:

  affine    flyby's 2D ground motion (the served behaviour)
  model3d   the object's 3D box re-projected, elevation from the flight default
  model3d+z the same with the elevation fitted from the sightings so far,
            which is what gives each object its own image speed

    python probe/carry_test.py --run <id> --pass ... --pass ...
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--pass', dest='passes', action='append', required=True)
    ap.add_argument('--fit', default=str(ROOT / 'data/probe/fit3d_scene.json'))
    ap.add_argument('--recordings', default=str(ROOT / 'data/recordings'))
    ap.add_argument('--cache', default=str(ROOT / 'data/det_cache'))
    ap.add_argument('--grow', type=float, default=1.3)
    a = ap.parse_args()

    import flyby
    from scipy.optimize import least_squares

    objects = [o for o in json.loads(Path(a.fit).read_text())['objects'] if o['fit_iou_median'] >= 0.6]

    caches = []
    for spec in a.passes:
        path, _, size = spec.partition(':')
        caches.append(pickle.loads((Path(a.cache) / f'{a.run}__{Path(path).stem}_{size}.pkl').read_bytes()))

    # Detections per frame, lifted to source pixels.
    dets = {}
    for meta_path in sorted((Path(a.recordings) / a.run).glob('*.json')):
        meta = json.loads(meta_path.read_text())
        rx1, ry1, rx2, ry2 = meta['view']['source_region_xyxy']
        scale = np.array([(rx2 - rx1) / 960, (ry2 - ry1) / 540] * 2)
        boxes = []
        for cache in caches:
            xyxy, probs = cache[meta_path.stem]
            keep = probs.max(axis=1) >= 0.05
            boxes.append(xyxy[keep] * scale + [rx1, ry1, rx1, ry1])
        dets[meta['frame']] = np.concatenate(boxes) if boxes else np.zeros((0, 4))

    # For each object: the frames where a detection really landed on it.
    sightings = collections.defaultdict(dict)
    truth_box = {}
    for i, o in enumerate(objects):
        x, y, z, yaw, k = o['params']
        params = (x, y, z, *G.CLASS_DIMS[o['class']], yaw)
        for frame in range(1, 250):
            box = G.shrink(G.box_at(params, frame, clip=False), k * a.grow)
            clipped = np.array([max(0.0, box[0]), max(0.0, box[1]), min(G.W, box[2]), min(G.H, box[3])])
            if clipped[2] - clipped[0] < 2 or clipped[3] - clipped[1] < 2:
                continue
            truth_box[(i, frame)] = clipped
            best, best_iou = None, 0.0
            for d in dets.get(frame, ()):
                v = G.iou(clipped, G.shrink(d, a.grow))
                if v > best_iou:
                    best, best_iou = d, v
            if best_iou >= 0.5:
                sightings[i][frame] = best

    gaps = [1, 3, 6, 10, 15, 20, 25, 30]
    fitted = tuple(json.loads(Path(ROOT / 'data/probe/fitted_motion.json').read_text()))
    results = {name: collections.defaultdict(list) for name in ('affine_prior', 'affine_fitted', 'model3d', 'model3d+z', 'model3d+oracle_z')}
    for i, o in enumerate(objects):
        frames = sorted(sightings[i])
        if not frames:
            continue
        dims = G.CLASS_DIMS[o['class']]
        for start in frames:
            box0 = sightings[i][start]
            # Elevation fitted from sightings up to and including `start`.
            prior = [f for f in frames if f <= start]
            z_fit = 0.0
            if len(prior) >= 2:
                cs = np.array([[(sightings[i][f][0] + sightings[i][f][2]) / 2,
                                (sightings[i][f][1] + sightings[i][f][3]) / 2] for f in prior])
                scale0 = _scale_for(box0, start, dims, 0.0)

                def res(p):
                    x, y, z = p
                    pt = np.array([[x, y, z - 0.5 * dims[2] * scale0]])
                    return np.concatenate([G.project(pt, f)[0] - c for f, c in zip(prior, cs)])
                g = G.ground_point((box0[0] + box0[2]) / 2, (box0[1] + box0[3]) / 2, start,
                                   -0.5 * dims[2] * scale0)
                try:
                    z_fit = float(least_squares(res, [g[0], g[1], 0.0], max_nfev=40).x[2])
                except Exception:
                    z_fit = 0.0
            for gap in gaps:
                target = start + gap
                if (i, target) not in truth_box or target in sightings[i]:
                    continue
                truth = truth_box[(i, target)]
                results['affine_prior'][gap].append(
                    G.iou(truth, G.shrink(flyby.advance(box0, gap, flyby.MOTION), a.grow)))
                results['affine_fitted'][gap].append(
                    G.iou(truth, G.shrink(flyby.advance(box0, gap, fitted), a.grow)))
                for name, z in (('model3d', 0.0), ('model3d+z', z_fit), ('model3d+oracle_z', o['params'][2])):
                    results[name][gap].append(G.iou(truth, G.shrink(_carry3d(box0, start, target, dims, z), a.grow)))

    print(f'{"gap":>5s} {"n":>6s} ' + ' '.join(f'{k:>22s}' for k in results))
    for gap in gaps:
        n = len(results['affine_prior'][gap])
        if not n:
            continue
        cells = []
        for name in results:
            v = np.array(results[name][gap])
            cells.append(f'IoU {np.median(v):5.3f}  hit {np.mean(v >= 0.5):5.1%}')
        print(f'{gap:5d} {n:6d} ' + ' '.join(f'{c:>22s}' for c in cells))


def _scale_for(box, frame, dims, z):
    unit = G.box_at((0, 0, z, *dims, 0.0), frame, clip=False)
    g = G.ground_point((box[0] + box[2]) / 2, (box[1] + box[3]) / 2, frame, z)
    unit = G.box_at((g[0], g[1], z, *dims, 0.0), frame, clip=False)
    uw, uh = unit[2] - unit[0], unit[3] - unit[1]
    if uw < 1e-6 or uh < 1e-6:
        return 1.0
    return float(np.clip(0.5 * ((box[2] - box[0]) / uw + (box[3] - box[1]) / uh), 0.15, 4.0))


def _carry3d(box, start, target, dims, z):
    """Carry a box with the 3D model: centre from the object, sides by ratio."""
    scale = _scale_for(box, start, dims, z)
    d = (dims[0] * scale, dims[1] * scale, dims[2] * scale)
    g = G.ground_point((box[0] + box[2]) / 2, (box[1] + box[3]) / 2, start, z - 0.5 * d[2])
    params = (g[0], g[1], z, *d, 0.0)
    centre = G.project(np.array([[g[0], g[1], z - 0.5 * d[2]]]), target)[0]
    now, before = G.box_at(params, target, clip=False), G.box_at(params, start, clip=False)
    sw = (now[2] - now[0]) / max(1e-6, before[2] - before[0])
    sh = (now[3] - now[1]) / max(1e-6, before[3] - before[1])
    half_w = 0.5 * (box[2] - box[0]) * sw
    half_h = 0.5 * (box[3] - box[1]) * sh
    return np.array([centre[0] - half_w, centre[1] - half_h, centre[0] + half_w, centre[1] + half_h])


if __name__ == '__main__':
    main()
