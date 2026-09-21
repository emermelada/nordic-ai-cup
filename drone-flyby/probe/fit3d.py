"""Fit a 3D box to every object in a truth-style file, then emit official boxes.

Each object's detector boxes (from any number of frames) are explained as the
projected 3D box of its class, shrunk about its centre by k -- detectors learn
tight silhouettes, the grader scores the loose projected box. Free per object:
ground position (x, y), ground elevation, yaw and k. The fitted object then
gives its official box in EVERY frame it is visible, clipped to the frame.

    python probe/fit3d.py training/scene_objects.json data/probe/fit3d_scene.json \
        --answers data/probe/mined3d_all.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parent))
import geometry as G  # noqa: E402

FRAMES = range(1, 250)


def fit_object(cls, obs, z0=0.0):
    dims = G.CLASS_DIMS[cls]
    frames = sorted(obs)
    mid = frames[len(frames) // 2]
    b = obs[mid]

    def model(p, fr):
        x, y, z, yaw, k = p
        box = G.box_at((x, y, z, *dims, yaw), fr, clip=False)
        box = G.shrink(box, k)
        return np.array([max(0, box[0]), max(0, box[1]), min(G.W, box[2]), min(G.H, box[3])])

    def res(p):
        return np.concatenate([model(p, fr) - np.asarray(obs[fr], float) for fr in frames])

    best = None
    for yaw0 in np.radians([0, 45, 90, 135]):
        g = G.ground_point((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, mid, z0)
        x0 = np.array([g[0], g[1], z0, yaw0, 0.8])
        r = least_squares(res, x0, loss='soft_l1', f_scale=2.0, max_nfev=2000,
                          bounds=([-np.inf, -np.inf, z0 - 60, -10, 0.45], [np.inf, np.inf, z0 + 60, 10, 1.15]))
        if best is None or r.cost < best.cost:
            best = r
    p = best.x
    fit_ious = [G.iou(model(p, fr), obs[fr]) for fr in frames]
    return p, fit_ious


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('truth')
    ap.add_argument('out')
    ap.add_argument('--answers', help='also write an answers file (official boxes, all visible frames)')
    ap.add_argument('--k-report', type=float, default=1.0, help='scale of the reported box vs the projected 3D box')
    a = ap.parse_args()

    objects = json.loads(Path(a.truth).read_text())['objects']
    fitted = []
    for o in objects:
        obs = {ob['frame']: ob['box'] for ob in o['observations']}
        p, fit_ious = fit_object(o['class'], obs)
        fitted.append({'id': o['id'], 'class': o['class'], 'params': [float(v) for v in p],
                       'n_obs': len(obs), 'fit_iou_median': float(np.median(fit_ious)),
                       'fit_iou_min': float(np.min(fit_ious))})
        print(f"{o['id']:6} {o['class']:16s} n={len(obs):2d} z={p[2]:6.1f} yaw={np.degrees(p[3]) % 180:5.1f} "
              f"k={p[4]:.2f} fitIoU med {np.median(fit_ious):.2f} min {np.min(fit_ious):.2f}")
    Path(a.out).write_text(json.dumps({'objects': fitted}, indent=1))
    ks = np.array([f['params'][4] for f in fitted]); zs = np.array([f['params'][2] for f in fitted])
    print(f'k: median {np.median(ks):.3f} IQR {np.quantile(ks, .25):.3f}-{np.quantile(ks, .75):.3f};  '
          f'z: median {np.median(zs):.1f} IQR {np.quantile(zs, .25):.1f}-{np.quantile(zs, .75):.1f}')

    if a.answers:
        ranked = sorted(fitted, key=lambda f: -f['n_obs'])
        answers, n = {}, len(ranked)
        for rank, f in enumerate(ranked):
            x, y, z, yaw, _ = f['params']
            obj = (x, y, z, *G.CLASS_DIMS[f['class']], yaw)
            conf = round(0.99 - 0.9 * rank / max(1, n), 4)
            for fr in FRAMES:
                box = G.box_at(obj, fr)
                if not G.visible(box):
                    continue
                box = G.shrink(box, a.k_report)
                box = [max(0.0, box[0]) / G.W, max(0.0, box[1]) / G.H, min(G.W, box[2]) / G.W, min(G.H, box[3]) / G.H]
                if box[2] - box[0] <= 1e-6 or box[3] - box[1] <= 1e-6:
                    continue
                answers.setdefault(str(fr), []).append(
                    {'object_id': f['class'], 'bbox': [round(v, 6) for v in box], 'confidence': conf})
        Path(a.answers).write_text(json.dumps(answers))
        print(f'{a.answers}: {len(answers)} frames, {sum(map(len, answers.values()))} boxes')


if __name__ == '__main__':
    main()
