"""Sheets for labelling the recorded validation flight by eye.

The models can only propose objects of classes they already handle, so the four
classes we have never confirmed (condor, ta-ta, medium_plane, medium_launcher)
can only be found by looking. An object crosses the frame in roughly 33 frames,
so sampling every 20th rebuilt frame shows every object at least once.

    python tools/review_scene.py --sheets              # -> data/review/*.png
    python tools/review_scene.py --zoom 141:2300,900   # full-res look at a blob
    python tools/review_scene.py --add 141:2280,880,2330,930:tank

Sheets carry a labelled pixel grid, already-known objects in red, and (with
--propose) low-confidence model guesses in blue, so the eye only has to hunt for
what is not already boxed. --add writes straight into
training/validation_objects.json, the same file mine_validation.py appends to.
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
sys.path.insert(0, str(ROOT / 'training'))

SCENE = ROOT / 'data' / 'scene'
REVIEW = ROOT / 'data' / 'review'
OBJECTS = ROOT / 'training' / 'validation_objects.json'

WIDTH, HEIGHT = 3840, 2160
GRID = 200
REACH = 30


def known_boxes(frame: int):
    from harvest_validation_patches import move
    objects = json.loads(OBJECTS.read_text())['objects']
    out = []
    for obj in objects:
        near = min(obj['observations'], key=lambda o: abs(o['frame'] - frame))
        if abs(near['frame'] - frame) > REACH:
            continue
        box = move(near['box'], frame - near['frame'])
        if box[2] > 0 and box[3] > 0 and box[0] < WIDTH and box[1] < HEIGHT:
            out.append((obj['id'], obj['class'], box))
    return out


def draw_grid(image, x_offset=0):
    for x in range(0, image.shape[1], GRID):
        cv2.line(image, (x, 0), (x, image.shape[0]), (255, 255, 0), 1)
        cv2.putText(image, str(x + x_offset), (x + 3, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    for y in range(0, image.shape[0], GRID):
        cv2.line(image, (0, y), (image.shape[1], y), (255, 255, 0), 1)
        cv2.putText(image, str(y), (3, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)


def proposals(image, models, confidence: float):
    """Low-confidence detections over an L1 tiling of the full frame."""
    import flyby
    out = []
    for path in models:
        flyby.MODEL_PATH = Path(path)
        flyby._model = None
        flyby.load_model()
        for cx in (960, 1920, 2880):
            for cy in (540, 1620):
                x1, y1 = cx - 960, cy - 540
                view = cv2.resize(image[y1:y1 + 1080, x1:x1 + 1920], (960, 540), interpolation=cv2.INTER_AREA)
                for name, conf, box, _ in flyby.detect(view, (x1, y1, x1 + 1920, y1 + 1080)):
                    if conf >= confidence:
                        out.append((name, float(conf), box))
    return out


def sheets(args):
    REVIEW.mkdir(parents=True, exist_ok=True)
    frames = sorted(int(p.stem.split('_')[1]) for p in SCENE.glob('frame_*.png'))
    chosen = [f for f in frames if (f - frames[0]) % args.every == 0]
    guesses = []
    for frame in chosen:
        image = cv2.imread(str(SCENE / f'frame_{frame:06d}.png'))
        if image is None:
            continue
        if args.propose:
            guesses = proposals(image, args.models, args.confidence)
            for name, conf, box in guesses:
                p1 = (int(box[0]) - 2, int(box[1]) - 2)
                p2 = (int(box[2]) + 2, int(box[3]) + 2)
                cv2.rectangle(image, p1, p2, (255, 120, 0), 2)
                cv2.putText(image, f'{name} {conf:.2f}', (p1[0], max(14, p1[1] - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 120, 0), 2)
        for oid, cls, box in known_boxes(frame):
            p1 = (int(box[0]) - 3, int(box[1]) - 3)
            p2 = (int(box[2]) + 3, int(box[3]) + 3)
            cv2.rectangle(image, p1, p2, (0, 0, 255), 3)
            cv2.putText(image, f'{oid}:{cls}', (p1[0], max(16, p1[1] - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        for half, x0 in (('left', 0), ('right', 1920)):
            crop = image[:, x0:x0 + 1920].copy()
            draw_grid(crop, x_offset=x0)
            cv2.putText(crop, f'frame {frame}  x {x0}-{x0 + 1920}', (10, HEIGHT - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.imwrite(str(REVIEW / f'f{frame:04d}_{half}.png'), crop)
    print(f'{len(chosen)} frames -> {REVIEW} (every {args.every}: {chosen})')


def zoom(args):
    REVIEW.mkdir(parents=True, exist_ok=True)
    spec = args.zoom
    frame, point = spec.split(':')
    x, y = (int(v) for v in point.split(',')[:2])
    size = args.size
    image = cv2.imread(str(SCENE / f'frame_{int(frame):06d}.png'))
    if image is None:
        raise SystemExit(f'frame {frame} not rebuilt')
    x1, y1 = max(0, x - size // 2), max(0, y - size // 2)
    crop = image[y1:y1 + size, x1:x1 + size].copy()
    scale = max(1, 900 // max(1, size))
    if scale > 1:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    step = 50 * scale
    for gx in range(0, crop.shape[1], step):
        cv2.line(crop, (gx, 0), (gx, crop.shape[0]), (255, 255, 0), 1)
        cv2.putText(crop, str(x1 + gx // scale), (gx + 2, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
    for gy in range(0, crop.shape[0], step):
        cv2.line(crop, (0, gy), (crop.shape[1], gy), (255, 255, 0), 1)
        cv2.putText(crop, str(y1 + gy // scale), (2, gy + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
    out = REVIEW / f'zoom_f{int(frame):04d}_{x}_{y}.png'
    cv2.imwrite(str(out), crop)
    print(f'{out}  ({size}px at {scale}x)')


def add(args):
    import flyby
    data = json.loads(OBJECTS.read_text())
    next_id = max(o['id'] for o in data['objects']) + 1
    for spec in args.add:
        frame, box, cls = spec.split(':')
        if cls not in flyby.OBJECT_CLASSES:
            raise SystemExit(f'unknown class {cls}')
        values = [float(v) for v in box.split(',')]
        if len(values) != 4:
            raise SystemExit(f'need x1,y1,x2,y2 in {spec}')
        data['objects'].append({
            'id': next_id, 'class': cls, 'found_by': 'review_scene',
            'observations': [{'frame': int(frame), 'box': values}],
        })
        print(f'  +{next_id} {cls} at frame {frame} {values}')
        next_id += 1
    OBJECTS.write_text(json.dumps(data, indent=1))
    print(f'{len(data["objects"])} objects in {OBJECTS.name}')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sheets', action='store_true')
    parser.add_argument('--every', type=int, default=20)
    parser.add_argument('--propose', action='store_true', help='Also draw low-confidence model guesses.')
    parser.add_argument('--models', nargs='*', default=[str(Path.home() / 'models' / 'drone-yolo11n-v4.pt')])
    parser.add_argument('--confidence', type=float, default=0.10)
    parser.add_argument('--zoom', help='FRAME:X,Y - full-resolution crop around a point')
    parser.add_argument('--size', type=int, default=300, help='Zoom crop size in source pixels.')
    parser.add_argument('--add', nargs='*', help='FRAME:X1,Y1,X2,Y2:CLASS objects to record')
    args = parser.parse_args()

    if args.sheets:
        sheets(args)
    if args.zoom:
        zoom(args)
    if args.add:
        add(args)
    if not (args.sheets or args.zoom or args.add):
        parser.print_help()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
