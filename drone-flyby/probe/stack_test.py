"""Does stacking repeated Level-1 views beat detecting on a single view?

Every ground point is transmitted several times as it crosses the frame, each
time sampled at a different sub-pixel phase by the evaluator's 2x box
downsample. Warped into a common frame through the ground homography and
combined, those samples carry more detail than any one of them -- the classic
multi-frame super-resolution setting, and here the motion is known in closed
form rather than estimated.

Compared on the same frames, same detector, same truth:

  single   the current view, upsampled to native scale
  stacked  that view plus the previous overlapping views, median-combined

    python probe/stack_test.py --run <id> --model models/drone-yolo11m-v8.pt
"""

import argparse
import collections
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'probe'))

import geometry as G  # noqa: E402
from align_test import ground_homography  # noqa: E402
from replay_score import load_truth  # noqa: E402


def view_matrix(region):
    """view pixels -> source pixels for a 960x540 view of `region`."""
    x1, y1, x2, y2 = region
    return np.array([[(x2 - x1) / 960.0, 0, x1], [0, (y2 - y1) / 540.0, y1], [0, 0, 1]], np.float64)


def build_stack(target, history, z_ground=0.0, refine=True):
    """Warp every view in `history` into the target view's region at native scale.

    Returns the median-combined image and how many views contributed.
    """
    region = target['region']
    x1, y1, x2, y2 = region
    out_w, out_h = x2 - x1, y2 - y1            # native resolution of that region
    # source pixels of the target frame -> output pixels
    to_out = np.array([[1, 0, -x1], [0, 1, -y1], [0, 0, 1]], np.float64)

    layers = []
    for item in history:
        H = ground_homography(item['frame'], target['frame'], z_ground)
        M = to_out @ H @ view_matrix(item['region'])
        warped = cv2.warpPerspective(item['image'], M, (out_w, out_h),
                                     flags=cv2.INTER_CUBIC, borderValue=0)
        covered = cv2.warpPerspective(np.full((540, 960), 255, np.uint8), M, (out_w, out_h),
                                      flags=cv2.INTER_NEAREST, borderValue=0)
        if (covered > 0).mean() < 0.15:
            continue
        layers.append((warped, covered > 0))

    if not layers:
        return None, 0
    if refine and len(layers) > 1:
        base = cv2.cvtColor(layers[-1][0], cv2.COLOR_BGR2GRAY).astype(np.float32)
        aligned = [layers[-1]]
        for warped, mask in layers[:-1]:
            grey = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY).astype(np.float32)
            both = mask & layers[-1][1]
            if both.mean() < 0.2:
                continue
            ys, xs = np.where(both)
            y0, y1_, x0, x1_ = ys.min(), ys.max(), xs.min(), xs.max()
            ph, pw = y1_ - y0, x1_ - x0
            if ph < 64 or pw < 64:
                continue
            win = cv2.createHanningWindow((pw, ph), cv2.CV_32F)
            (dx, dy), _ = cv2.phaseCorrelate(grey[y0:y0 + ph, x0:x0 + pw],
                                             base[y0:y0 + ph, x0:x0 + pw], win)
            if abs(dx) > 8 or abs(dy) > 8:
                continue                      # a duplicate frame or a bad pair
            shift = np.array([[1, 0, dx], [0, 1, dy]], np.float32)
            aligned.append((cv2.warpAffine(warped, shift, (out_w, out_h), flags=cv2.INTER_CUBIC),
                            cv2.warpAffine(mask.astype(np.uint8) * 255, shift, (out_w, out_h)) > 0))
        layers = aligned

    stack = np.stack([layer.astype(np.float32) for layer, _ in layers])
    masks = np.stack([m for _, m in layers])
    stack[~masks] = np.nan
    with np.errstate(all='ignore'):
        combined = np.nanmedian(stack, axis=0)
    combined = np.nan_to_num(combined, nan=0.0)
    return combined.astype(np.uint8), len(layers)


def detect_on(image, region, model, imgsz, conf=0.05):
    """Detections (source-pixel boxes) from one image covering `region`."""
    import flyby
    h, w = image.shape[:2]
    tiles = []
    step = 1920
    for x in range(0, max(1, w - 1), step):
        for y in range(0, max(1, h - 1), 1080):
            tiles.append((x, y, min(x + step, w), min(y + 1080, h)))
    out = []
    for tx1, ty1, tx2, ty2 in tiles:
        crop = image[ty1:ty2, tx1:tx2]
        if crop.shape[0] < 32 or crop.shape[1] < 32:
            continue
        xyxy, probs = flyby.raw_detections(crop)
        sx = (tx2 - tx1) / crop.shape[1]
        sy = (ty2 - ty1) / crop.shape[0]
        for box, p in zip(xyxy, probs):
            if p.max() < conf:
                continue
            b = [box[0] * sx + tx1 + region[0], box[1] * sy + ty1 + region[1],
                 box[2] * sx + tx1 + region[0], box[3] * sy + ty1 + region[1]]
            out.append((flyby.OBJECT_CLASSES[int(p.argmax())], float(p.max()), np.array(b)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--model', default=str(ROOT / 'models/drone-yolo11m-v8.pt'))
    ap.add_argument('--imgsz', type=int, default=1920)
    ap.add_argument('--depth', type=int, default=6, help='views combined, including the current one')
    ap.add_argument('--frames', type=int, default=40)
    ap.add_argument('--truth', default=str(ROOT / 'data/probe/fit3d_scene.json'))
    ap.add_argument('--recordings', default=str(ROOT / 'data/recordings'))
    ap.add_argument('--device', default='cpu')
    a = ap.parse_args()

    os.environ['DRONE_MODEL'] = a.model
    os.environ['DRONE_IMGSZ'] = str(a.imgsz)
    os.environ['DRONE_DEVICE'] = a.device
    os.environ['DRONE_DET_CONF'] = '0.02'
    import flyby
    flyby.load_model()

    truth = load_truth(a.truth, 1.3)
    views = []
    for meta_path in sorted((Path(a.recordings) / a.run).glob('*.json')):
        meta = json.loads(meta_path.read_text())
        png = meta_path.with_suffix('.png')
        if not png.exists() or meta['view']['resolution_level'] != 1:
            continue
        views.append({'frame': meta['frame'], 'region': meta['view']['source_region_xyxy'], 'path': png})

    stats = {'single': [0, 0], 'stacked': [0, 0]}
    per_class = collections.defaultdict(lambda: {'single': [0, 0], 'stacked': [0, 0]})
    step = max(1, len(views) // a.frames)
    for index in range(a.depth, len(views), step):
        target = dict(views[index])
        target['image'] = cv2.imread(str(target['path']))
        history = []
        for item in views[max(0, index - a.depth + 1):index + 1]:
            entry = dict(item)
            entry['image'] = cv2.imread(str(entry['path']))
            history.append(entry)
        stacked, depth = build_stack(target, history)
        if stacked is None or depth < 2:
            continue
        x1, y1, x2, y2 = target['region']
        single = cv2.resize(target['image'], (x2 - x1, y2 - y1), interpolation=cv2.INTER_CUBIC)

        wanted = [(cls, box) for cls, box in truth.get(target['frame'], [])
                  if x1 <= (box[0] + box[2]) / 2 <= x2 and y1 <= (box[1] + box[3]) / 2 <= y2]
        if not wanted:
            continue
        for name, image in (('single', single), ('stacked', stacked)):
            dets = detect_on(image, target['region'], a.model, a.imgsz)
            for cls, box in wanted:
                found = any(G.iou(box, d[2]) >= 0.3 for d in dets)
                stats[name][0] += 1
                stats[name][1] += int(found)
                per_class[cls][name][0] += 1
                per_class[cls][name][1] += int(found)
        print(f'  frame {target["frame"]:3d} depth {depth} objects {len(wanted)} '
              f'single {stats["single"][1]}/{stats["single"][0]} '
              f'stacked {stats["stacked"][1]}/{stats["stacked"][0]}', flush=True)

    print(f'\n{"":16s} {"single":>14s} {"stacked":>14s}')
    for cls in sorted(per_class):
        s, t = per_class[cls]['single'], per_class[cls]['stacked']
        print(f'{cls:16s} {s[1]:5d}/{s[0]:<5d} {s[1]/max(1,s[0]):5.2f} {t[1]:5d}/{t[0]:<5d} {t[1]/max(1,t[0]):5.2f}')
    s, t = stats['single'], stats['stacked']
    print(f'{"TOTAL":16s} {s[1]:5d}/{s[0]:<5d} {s[1]/max(1,s[0]):5.2f} {t[1]:5d}/{t[0]:<5d} {t[1]/max(1,t[0]):5.2f}')


if __name__ == '__main__':
    main()
