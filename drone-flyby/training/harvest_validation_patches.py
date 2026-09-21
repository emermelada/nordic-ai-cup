"""Cut the known validation objects out of every recorded view that shows them.

    python training/harvest_validation_patches.py     # -> data/patches_val/<class>/*.png

training/validation_objects.json lists objects seen in the recorded validation
flight (each with a few confident sightings). The ground moves predictably, so
each object's box can be carried to every other recorded frame, including the
ones where the model missed it; those views are the valuable ones. Each view is
cut out with a GrabCut mask like extract_patches.py does, then scaled to
source-pixel size so make_dataset.py can paste it like any other patch.

Level-2 views (survey runs) give native-resolution patches; Level-1 views give
half-resolution ones, softer than the Helsinki patches but fine for Level 0/1.
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
SEARCH = 24                 # source pixels to search around the carried box
MIN_MATCH = 0.55            # template match score needed to trust a view
# Many recorded runs show the same flight: keep at most this many views per
# object, one per frame and level, spread over its frames.
MAX_PER_OBJECT = 40

# Boxes are carried with a motion fitted on this flight's own objects, not
# flyby.MOTION. The Helsinki prior runs ~2.6 px/frame short here, which over the
# 20-frame carry below is ~52 px against a SEARCH window of 24 source px: past
# about nine frames the object is outside the window the template match looks
# in, so the view is dropped or the patch is cut off-centre. Same trap Franek
# found in the tracker and score_offline, one layer further back -- here it
# quietly costs training data instead of score.
FORWARD = SHIFT = BACKWARD = None


def set_motion(motion) -> None:
    global FORWARD, SHIFT, BACKWARD
    a, b, c, d, e, f = motion
    FORWARD = np.array([[1 + b, c], [e, 1 + f]])
    SHIFT = np.array([a, d])
    BACKWARD = np.linalg.inv(FORWARD)


set_motion(MOTION)


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


class Locator:
    """Finds known validation objects in recorded views.

    A box is carried from the object's nearest sighting with the ground-motion
    model, then pinned down by matching the object's appearance (a template
    from a view where it was detected) in a small window around it.
    """

    def __init__(self, objects, recordings: Path, levels=(1, 2)):
        self.objects = objects
        self.views = []
        for meta_path in sorted(recordings.glob('*/*.json')):
            meta = json.loads(meta_path.read_text())
            if meta['view']['resolution_level'] in levels:
                self.views.append((meta_path.with_suffix('.png'), meta['frame'], meta['view']['source_region_xyxy']))
        self._images = {}
        self._templates = {}

    def image(self, png):
        if png not in self._images:
            if len(self._images) > 300:
                self._images.clear()
            self._images[png] = cv2.imread(str(png))
        return self._images[png]

    @staticmethod
    def to_view(box, region):
        rx1, ry1, rx2, _ = region
        scale = (rx2 - rx1) / 960
        return (np.asarray(box, float) - [rx1, ry1, rx1, ry1]) / scale

    def templates(self, obj):
        if obj['id'] not in self._templates:
            found = []
            for observation in obj['observations']:
                for png, frame, region in self.views:
                    if frame != observation['frame']:
                        continue
                    x1, y1, x2, y2 = self.to_view(observation['box'], region)
                    if x1 >= 0 and y1 >= 0 and x2 <= 960 and y2 <= 540:
                        patch = self.image(png)[int(y1):int(np.ceil(y2)), int(x1):int(np.ceil(x2))]
                        if patch.size:
                            found.append((frame, patch.copy(), (region[2] - region[0]) / 960))
                        break
            self._templates[obj['id']] = found
        return self._templates[obj['id']]

    def carried(self, obj, frame, reach=MAX_FRAME_DISTANCE):
        """The motion-model box at ``frame`` (source pixels), or None if too far."""
        near = min(obj['observations'], key=lambda o: abs(o['frame'] - frame))
        if abs(near['frame'] - frame) > reach:
            return None
        return move(near['box'], frame - near['frame'])

    def locate(self, obj, png, frame, region, edge=EDGE):
        """The object's box in this view (view pixels), or None."""
        box = self.carried(obj, frame)
        templates = self.templates(obj)
        if box is None or not templates:
            return None
        x1, y1, x2, y2 = self.to_view(box, region)
        if x1 < edge or y1 < edge or x2 > 960 - edge or y2 > 540 - edge:
            return None
        _, template, template_scale = min(templates, key=lambda t: abs(t[0] - frame))
        view_scale = (region[2] - region[0]) / 960
        if template_scale != view_scale:
            # Templates from a Level-1 view are half the size of the same
            # object in a Level-2 view, and the other way round.
            factor = template_scale / view_scale
            template = cv2.resize(template, None, fx=factor, fy=factor,
                                  interpolation=cv2.INTER_CUBIC if factor > 1 else cv2.INTER_AREA)
        th, tw = template.shape[:2]
        if th < 6 or tw < 6:
            return None
        search = SEARCH / view_scale
        sx1, sy1 = int(max(0, x1 - search)), int(max(0, y1 - search))
        sx2, sy2 = int(min(960, x1 + tw + search)), int(min(540, y1 + th + search))
        window = self.image(png)[sy1:sy2, sx1:sx2]
        if window.shape[0] < th or window.shape[1] < tw:
            return None
        scores = cv2.matchTemplate(window, template, cv2.TM_CCOEFF_NORMED)
        _, score, _, (mx, my) = cv2.minMaxLoc(scores)
        if score < MIN_MATCH:
            return None
        x1, y1 = sx1 + mx, sy1 + my
        x2, y2 = x1 + tw, y1 + th
        if x1 < edge or y1 < edge or x2 > 960 - edge or y2 > 540 - edge:
            return None
        return x1, y1, x2, y2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--objects', type=Path, default=HERE / 'validation_objects.json')
    parser.add_argument('--recordings', type=Path, default=ROOT / 'data' / 'recordings')
    parser.add_argument('--out', type=Path, default=ROOT / 'data' / 'patches_val')
    parser.add_argument('--prior-motion', action='store_true',
                        help='Carry boxes with flyby.MOTION again, to A/B the fit.')
    args = parser.parse_args()

    objects = json.loads(args.objects.read_text())['objects']
    if not args.prior_motion:
        sys.path.insert(0, str(ROOT / 'tools'))
        from score_offline import fit_truth_motion
        fitted, samples = fit_truth_motion(objects)
        if samples:
            set_motion(fitted)
            import flyby
            print(f'carrying boxes at {flyby.drift_at_centre(fitted):.2f} px/frame '
                  f'({samples} samples; Helsinki prior {flyby.drift_at_centre(MOTION):.2f})')
    locator = Locator(objects, args.recordings)
    counts, tiles = {}, []

    for obj in objects:
        chosen, seen = [], set()
        for png, frame, region in locator.views:
            key = (frame, region[2] - region[0])
            if key in seen or locator.carried(obj, frame) is None:
                continue
            seen.add(key)
            chosen.append((png, frame, region))
        located = []
        for png, frame, region in chosen:
            found = locator.locate(obj, png, frame, region)
            if found is not None:
                located.append((png, frame, region, found))
        step = max(1, len(located) // MAX_PER_OBJECT)
        for png, frame, region, found in located[::step][:MAX_PER_OBJECT]:
            x1, y1, x2, y2 = found
            image = locator.image(png)
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
