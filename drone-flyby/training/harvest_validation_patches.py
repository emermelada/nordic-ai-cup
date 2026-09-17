"""Cut the known validation objects out of every recorded view that shows them.

    python training/harvest_validation_patches.py     # -> data/patches_val/<class>/*.png

training/validation_objects.json lists objects seen in the recorded validation
flight (each with a few confident sightings). The ground moves predictably, so
each object's box can be carried to every other recorded frame, including the
ones where the model missed it; those views are the valuable ones. Each view is
cut out with a GrabCut mask like extract_patches.py does, then scaled to
source-pixel size so make_dataset.py can paste it like any other patch.

Recorded views are Level 1 (half resolution), so these patches are softer than
the Helsinki ones; that is fine for Level 0/1 views, which is where they help.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from extract_patches import object_mask  # noqa: E402
from flyby import MOTION  # noqa: E402

MAX_FRAME_DISTANCE = 20     # carry a box at most this many frames
EDGE = 4                    # view pixels the object must stay inside the view by
MARGIN = 0.35               # context around the box, as a share of its size
FILLED = 0.72               # a mask this full is GrabCut's ellipse fallback
MIN_IOU = 0.6               # mask box vs matched box
SEARCH = 12                 # view pixels to search around the carried box
MIN_MATCH = 0.55            # template match score needed to trust a view

a, b, c, d, e, f = MOTION
FORWARD = np.array([[1 + b, c], [e, 1 + f]])
SHIFT = np.array([a, d])
BACKWARD = np.linalg.inv(FORWARD)


def box_iou(p, q) -> float:
    ix = max(0.0, min(p[2], q[2]) - max(p[0], q[0]))
    iy = max(0.0, min(p[3], q[3]) - max(p[1], q[1]))
    inter = ix * iy
    union = (p[2] - p[0]) * (p[3] - p[1]) + (q[2] - q[0]) * (q[3] - q[1]) - inter
    return inter / union if union > 0 else 0.0


def move(box, steps: int) -> np.ndarray:
    points = np.array([[box[0], box[1]], [box[2], box[3]]], float)
    for _ in range(abs(steps)):
        points = points @ FORWARD.T + SHIFT if steps > 0 else (points - SHIFT) @ BACKWARD.T
    return points.reshape(-1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--objects', type=Path, default=HERE / 'validation_objects.json')
    parser.add_argument('--recordings', type=Path, default=ROOT / 'data' / 'recordings')
    parser.add_argument('--out', type=Path, default=ROOT / 'data' / 'patches_val')
    args = parser.parse_args()

    objects = json.loads(args.objects.read_text())['objects']
    views = []
    for meta_path in sorted(args.recordings.glob('*/*.json')):
        meta = json.loads(meta_path.read_text())
        if meta['view']['resolution_level'] == 1:
            views.append((meta_path.with_suffix('.png'), meta['frame'], meta['view']['source_region_xyxy']))

    counts, tiles = {}, []
    images = {}

    def load(png):
        if png not in images:
            images[png] = cv2.imread(str(png))
        return images[png]

    def to_view(box, region):
        rx1, ry1, rx2, _ = region
        scale = (rx2 - rx1) / 960
        return (np.asarray(box, float) - [rx1, ry1, rx1, ry1]) / scale

    for obj in objects:
        # Templates: the object as seen in the views where it was detected.
        templates = []
        for observation in obj['observations']:
            for png, frame, region in views:
                if frame != observation['frame']:
                    continue
                x1, y1, x2, y2 = to_view(observation['box'], region)
                if x1 >= 0 and y1 >= 0 and x2 <= 960 and y2 <= 540:
                    patch = load(png)[int(y1):int(np.ceil(y2)), int(x1):int(np.ceil(x2))]
                    if patch.size:
                        templates.append((frame, patch))
                    break
        if not templates:
            continue

        for png, frame, region in views:
            near = min(obj['observations'], key=lambda o: abs(o['frame'] - frame))
            if abs(near['frame'] - frame) > MAX_FRAME_DISTANCE:
                continue
            x1, y1, x2, y2 = to_view(move(near['box'], frame - near['frame']), region)
            if x1 < EDGE or y1 < EDGE or x2 > 960 - EDGE or y2 > 540 - EDGE:
                continue
            image = load(png)

            # Pin the carried box down by matching the nearest template around it.
            template = min(templates, key=lambda t: abs(t[0] - frame))[1]
            th, tw = template.shape[:2]
            sx1, sy1 = int(max(0, x1 - SEARCH)), int(max(0, y1 - SEARCH))
            sx2, sy2 = int(min(960, x1 + tw + SEARCH)), int(min(540, y1 + th + SEARCH))
            window = image[sy1:sy2, sx1:sx2]
            if window.shape[0] < th or window.shape[1] < tw:
                continue
            scores = cv2.matchTemplate(window, template, cv2.TM_CCOEFF_NORMED)
            _, score, _, (mx, my) = cv2.minMaxLoc(scores)
            if score < MIN_MATCH:
                continue
            x1, y1 = sx1 + mx, sy1 + my
            x2, y2 = x1 + tw, y1 + th
            if x1 < EDGE or y1 < EDGE or x2 > 960 - EDGE or y2 > 540 - EDGE:
                continue

            pad_x, pad_y = int((x2 - x1) * MARGIN) + 3, int((y2 - y1) * MARGIN) + 3
            cx1, cy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
            cx2, cy2 = min(960, x2 + pad_x), min(540, y2 + pad_y)
            crop = image[cy1:cy2, cx1:cx2]
            box = (x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1)
            mask = object_mask(crop, box)
            ys, xs = np.nonzero(mask)
            good = len(xs) >= 20
            if good:
                bx1, by1, bx2, by2 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
                good = (len(xs) <= FILLED * (bx2 - bx1) * (by2 - by1)
                        and box_iou(np.array([bx1, by1, bx2, by2], float), np.array(box, float)) >= MIN_IOU)
            if good:
                alpha = mask
            else:
                # No usable mask: a soft-edged rectangle, so the pasted ground
                # fades out instead of leaving a hard seam.
                bx1, by1, bx2, by2 = box
                alpha = np.zeros(crop.shape[:2], np.uint8)
                alpha[by1:by2, bx1:bx2] = 255
                alpha = cv2.GaussianBlur(alpha, (0, 0), 1.2)
                alpha[by1 + 1:by2 - 1, bx1 + 1:bx2 - 1] = 255
            patch = np.dstack([crop[by1:by2, bx1:bx2], alpha[by1:by2, bx1:bx2]])
            # Back to source-pixel size, like every other patch.
            scale = (region[2] - region[0]) / 960
            patch = cv2.resize(patch, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            folder = args.out / obj['class']
            folder.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(folder / f'val{obj["id"]:02d}_{png.parent.name[:6]}_{png.stem}.png'), patch)
            counts[obj['class']] = counts.get(obj['class'], 0) + 1
            if len(tiles) < 66 and counts[obj['class']] % 3 == 1:
                shown = cv2.bitwise_and(crop, crop, mask=alpha)
                tiles.append(cv2.resize(np.hstack([crop, shown]), (192, 96), interpolation=cv2.INTER_NEAREST))

    for name, count in sorted(counts.items()):
        print(f'{name:16s} {count:4d} patches')
    if tiles:
        while len(tiles) % 6:
            tiles.append(np.zeros_like(tiles[0]))
        sheet = np.vstack([np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)])
        args.out.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.out / '_sheet.png'), sheet)
    return 0


if __name__ == '__main__':
    sys.exit(main())
