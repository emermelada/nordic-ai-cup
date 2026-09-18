"""Clean backgrounds cut from the real validation terrain.

v1-v5 were trained on cut-outs pasted onto unrelated aerial photos (Helsinki,
INRIA, LandCover), and the measured result is a detector that is near perfect on
Helsinki (median IoU 0.93) and at 31% recall on validation. The backgrounds are
the domain gap, and we already hold 249 rebuilt frames of the real thing.

Those frames contain the real objects, so they cannot be used as backgrounds as
they are: an unlabelled object would teach the model to ignore that object. Here
everything that might be an object is covered first --- known objects from
validation_objects.json, plus anything an ensemble detects above a very low
floor --- by copying clean terrain from elsewhere in the same frame. Copying
rather than inpainting keeps the texture statistics real; a blurred smudge would
be its own kind of fake.

The detection floor is deliberately high (0.5, not 0.02). Our models fire
constantly on bushes, sheds and boats --- that is most of what is wrong with
them --- so covering everything they react to would delete exactly the hard
negatives this background set exists to teach. Only confident detections that
are not already known objects get covered, as those are the plausible
unlabelled objects.

    python training/make_real_backgrounds.py          # -> data/backgrounds_real/
    python training/make_real_backgrounds.py --every 1 --floor 0.02

Feed the result to make_dataset.py --backgrounds data/backgrounds_real.
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

SCENE = ROOT / 'data' / 'scene'
OUT = ROOT / 'data' / 'backgrounds_real'
OBJECTS = HERE / 'validation_objects.json'

WIDTH, HEIGHT = 3840, 2160
HOLDOUT_FROM = 200      # keep late frames out of training, as make_real_views.py does
REACH = 45              # frames a known object is still masked around
PAD = 0.6               # grow a masked box by this share of its size
FEATHER = 0.25       # blend ramp at the edge of a copied patch, as a share of its size
TRIES = 60           # random source positions tried per covered box
MAX_UNSEEN = 0.005   # frames with more never-recorded area than this are skipped:
                     # a large gap cannot be covered from the same frame and would
                     # leave a black rectangle for the model to learn as a feature


def known_boxes(frame: int):
    from harvest_validation_patches import move
    objects = json.loads(OBJECTS.read_text())['objects']
    out = []
    for obj in objects:
        near = min(obj['observations'], key=lambda o: abs(o['frame'] - frame))
        if abs(near['frame'] - frame) > REACH:
            continue
        out.append(move(near['box'], frame - near['frame']))
    return out


def detections(image, floor: float):
    """Confident detections over an L1 tiling, with whichever model is loaded."""
    import flyby
    out = []
    for cx in (960, 1920, 2880):
        for cy in (540, 1620):
            x1, y1 = cx - 960, cy - 540
            view = cv2.resize(image[y1:y1 + 1080, x1:x1 + 1920], (960, 540), interpolation=cv2.INTER_AREA)
            for _, confidence, box, _ in flyby.detect(view, (x1, y1, x1 + 1920, y1 + 1080)):
                if confidence >= floor:
                    out.append(box)
    return out


def detections_for(frames, models, floor: float):
    """frame -> confident boxes, loading each model once rather than per frame."""
    import flyby
    found = {frame: [] for frame in frames}
    for model in models:
        flyby.MODEL_PATH = Path(model)
        flyby._model = None
        flyby.load_model()
        print(f'  screening with {Path(model).name}', flush=True)
        for frame in frames:
            image = cv2.imread(str(SCENE / f'frame_{frame:06d}.png'))
            if image is not None:
                found[frame].extend(detections(image, floor))
    return found


def grown_boxes(boxes):
    out = []
    for box in boxes:
        width, height = box[2] - box[0], box[3] - box[1]
        x1 = int(max(0, box[0] - width * PAD))
        y1 = int(max(0, box[1] - height * PAD))
        x2 = int(min(WIDTH, box[2] + width * PAD))
        y2 = int(min(HEIGHT, box[3] + height * PAD))
        if x2 > x1 and y2 > y1:
            out.append((x1, y1, x2, y2))
    return out


def mask_for(boxes) -> np.ndarray:
    mask = np.zeros((HEIGHT, WIDTH), np.uint8)
    for x1, y1, x2, y2 in boxes:
        mask[y1:y2, x1:x2] = 255
    return mask


def cover(image, mask, boxes, rng) -> np.ndarray:
    """Replace each covered box with clean terrain copied from the same frame."""
    out = image.copy()
    integral = cv2.integral(mask // 255)
    for x1, y1, x2, y2 in boxes:
        width, height = x2 - x1, y2 - y1
        if width >= WIDTH - 2 or height >= HEIGHT - 2:
            continue
        around = image[max(0, y1 - height):y2 + height, max(0, x1 - width):x2 + width].astype(np.float32)
        target = around.mean(axis=(0, 1))
        target_spread = around.std(axis=(0, 1))
        best = None
        for _ in range(TRIES):
            sx = rng.integers(0, WIDTH - width)
            sy = rng.integers(0, HEIGHT - height)
            # any masked pixel inside the source rectangle disqualifies it
            covered = (integral[sy + height, sx + width] - integral[sy, sx + width]
                       - integral[sy + height, sx] + integral[sy, sx])
            if covered:
                continue
            patch = image[sy:sy + height, sx:sx + width]
            distance = float(np.abs(patch.astype(np.float32).mean(axis=(0, 1)) - target).sum())
            spread = float(np.abs(patch.astype(np.float32).std(axis=(0, 1)) - target_spread).sum())
            if best is None or distance + spread < best[0]:
                best = (distance + spread, patch)
                if distance + spread < 10:
                    break
        if best is None:
            continue
        feather = int(min(width, height) * FEATHER)
        alpha = np.ones((height, width), np.float32)
        if feather > 0:
            ramp = np.linspace(0, 1, feather, dtype=np.float32)
            alpha[:feather] *= ramp[:, None]
            alpha[-feather:] *= ramp[::-1][:, None]
            alpha[:, :feather] *= ramp[None, :]
            alpha[:, -feather:] *= ramp[::-1][None, :]
        alpha = alpha[:, :, None]
        out[y1:y2, x1:x2] = (best[1] * alpha + out[y1:y2, x1:x2] * (1 - alpha)).astype(np.uint8)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--every', type=int, default=2, help='Use every Nth frame (consecutive frames overlap).')
    parser.add_argument('--floor', type=float, default=0.50,
                        help='Confident detections above this, not already known, are covered too. '
                             'Keep it high: low-confidence firings are the hard negatives we want kept.')
    parser.add_argument('--models', nargs='*', default=[str(Path.home() / 'models' / 'drone-yolo11n-v4.pt'),
                                                        str(Path.home() / 'models' / 'drone-yolo11m-v5.pt')])
    parser.add_argument('--holdout-from', type=int, default=HOLDOUT_FROM)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--format', choices=['jpg', 'png'], default='jpg',
                        help='jpg keeps the set small enough to push to the training box.')
    parser.add_argument('--quality', type=int, default=95)
    args = parser.parse_args()

    models = [m for m in args.models if Path(m).exists()]
    if not models:
        raise SystemExit('no models found to screen the frames with')
    args.out.mkdir(parents=True, exist_ok=True)

    frames = sorted(int(p.stem.split('_')[1]) for p in SCENE.glob('frame_*.png'))
    frames = [f for f in frames if f < args.holdout_from and f % args.every == 0]
    print(f'{len(frames)} frames, screened by {len(models)} models at floor {args.floor}', flush=True)

    found = detections_for(frames, models, args.floor)
    rng = np.random.default_rng(0)
    written = 0
    for frame in frames:
        image = cv2.imread(str(SCENE / f'frame_{frame:06d}.png'))
        if image is None:
            continue
        levels = cv2.imread(str(SCENE / f'levels_{frame:06d}.png'), cv2.IMREAD_GRAYSCALE)
        boxes = grown_boxes(known_boxes(frame) + found[frame])
        mask = mask_for(boxes)
        if levels is not None:
            unseen = (levels == 0).astype(np.uint8) * 255
            unseen_share = float((unseen > 0).mean())
            if unseen_share > MAX_UNSEEN:
                print(f'  frame {frame}: {unseen_share:.1%} never recorded, skipped')
                continue
            mask = np.maximum(mask, unseen)
            boxes = boxes + [(int(x), int(y), int(x + w), int(y + h))
                             for x, y, w, h in [cv2.boundingRect(c) for c in
                                                cv2.findContours(unseen, cv2.RETR_EXTERNAL,
                                                                 cv2.CHAIN_APPROX_SIMPLE)[0]]]
        covered_share = float((mask > 0).mean())
        if covered_share > 0.30:
            print(f'  frame {frame}: {covered_share:.0%} covered, skipped')
            continue
        clean = cover(image, mask, boxes, rng)
        path = args.out / f'bg_{frame:06d}.{args.format}'
        params = [cv2.IMWRITE_JPEG_QUALITY, args.quality] if args.format == 'jpg' else []
        cv2.imwrite(str(path), clean, params)
        written += 1
        if written % 10 == 0:
            print(f'  {written} written (frame {frame}, {covered_share:.1%} covered)', flush=True)

    print(f'{written} backgrounds in {args.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
