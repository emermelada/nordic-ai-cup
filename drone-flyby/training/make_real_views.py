"""Turn recorded validation views into labelled YOLO training images.

    python training/make_real_views.py --model ~/models/drone-yolo11n-v3.pt
        -> data/real_views/{images,labels}/*  (+ preview.jpg)

Each recorded Level-1/2 view gets a label for every known object
(validation_objects.json) that the Locator can pin down in it. Everything
else that might be an object is painted over, so it is neither taught as an
object nor as background:

* known objects that are near the view but could not be located precisely
  (cut by the edge, or too far from a sighting to trust the box);
* anything the model detects at DETECTION_FLOOR or more that is not a
  located known object.

Frames from HOLDOUT_FROM on are skipped, so tools/bench_recordings.py can
still be run on views no model has trained on.

Not used for v4: while validation_objects.json covers only part of the
flight, many real objects in these views are neither labelled nor detected
(so not painted over either), and training on them would teach the model to
ignore objects. The pasted cut-outs carry the same appearance without that.
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
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(HERE))

from dtos import OBJECT_CLASSES  # noqa: E402
from harvest_validation_patches import Locator  # noqa: E402

HOLDOUT_FROM = 200          # frames >= this stay out of training
DETECTION_FLOOR = 0.05
HIDE_REACH = 45             # frames a known object is still hidden around
HIDE_PAD = 0.5              # growth of a hidden box, as a share of its size
MIN_BOX = 4                 # view pixels


def hide(image, boxes):
    """Paint over boxes with the surrounding ground."""
    if not boxes:
        return image
    mask = np.zeros(image.shape[:2], np.uint8)
    for x1, y1, x2, y2 in boxes:
        px, py = (x2 - x1) * HIDE_PAD + 3, (y2 - y1) * HIDE_PAD + 3
        cv2.rectangle(mask, (int(x1 - px), int(y1 - py)), (int(x2 + px), int(y2 + py)), 255, -1)
    return cv2.inpaint(image, mask, 5, cv2.INPAINT_TELEA)


def overlaps(box, others, share=0.3):
    x1, y1, x2, y2 = box
    area = max(1.0, (x2 - x1) * (y2 - y1))
    for a1, b1, a2, b2 in others:
        ix = max(0.0, min(x2, a2) - max(x1, a1))
        iy = max(0.0, min(y2, b2) - max(y1, b1))
        if ix * iy > share * min(area, (a2 - a1) * (b2 - b1)):
            return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', type=Path, default=Path.home() / 'models' / 'drone-yolo11n-v2.pt')
    parser.add_argument('--out', type=Path, default=ROOT / 'data' / 'real_views')
    parser.add_argument('--min-objects', type=int, default=1, help='Skip views with fewer labelled objects.')
    args = parser.parse_args()

    import os
    import logging
    logging.disable(logging.WARNING)
    os.environ['DRONE_MODEL'] = str(args.model)
    from bench_recordings import cached_detections

    detections = cached_detections(args.model)
    objects = json.loads((HERE / 'validation_objects.json').read_text())['objects']
    locator = Locator(objects, ROOT / 'data' / 'recordings')
    for kind in ('images', 'labels'):
        (args.out / kind).mkdir(parents=True, exist_ok=True)

    written, labelled, hidden, preview = 0, 0, 0, []
    for png, frame, region in locator.views:
        if frame >= HOLDOUT_FROM:
            continue
        labels, masked = [], []
        for obj in objects:
            found = locator.locate(obj, png, frame, region, edge=2)
            if found is not None:
                labels.append((obj['class'], found))
                continue
            near = locator.carried(obj, frame, reach=HIDE_REACH)
            if near is not None:
                x1, y1, x2, y2 = locator.to_view(near, region)
                if x2 > 0 and y2 > 0 and x1 < 960 and y1 < 540:
                    masked.append((x1, y1, x2, y2))
        if len(labels) < args.min_objects:
            continue

        key = str(png.relative_to(ROOT / 'data' / 'recordings'))
        xyxy, probabilities = detections.get(key, (np.zeros((0, 4)), np.zeros((0, 16))))
        label_boxes = [box for _, box in labels]
        for box, p in zip(xyxy, probabilities):
            if p.max() >= DETECTION_FLOOR and not overlaps(box, label_boxes):
                masked.append(tuple(box))
        # Never paint over a labelled object, including the painting margin.
        def padded(box):
            x1, y1, x2, y2 = box
            px, py = (x2 - x1) * HIDE_PAD + 3, (y2 - y1) * HIDE_PAD + 3
            return x1 - px, y1 - py, x2 + px, y2 + py
        masked = [m for m in masked if not overlaps(padded(m), label_boxes, share=0.0)]

        image = hide(locator.image(png).copy(), masked)
        stem = f'{png.parent.name[:6]}_{png.stem}'
        cv2.imwrite(str(args.out / 'images' / f'{stem}.png'), image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        lines = []
        for cls, (x1, y1, x2, y2) in labels:
            if min(x2 - x1, y2 - y1) < MIN_BOX:
                continue
            lines.append(f'{OBJECT_CLASSES.index(cls)} {(x1 + x2) / 1920:.6f} {(y1 + y2) / 1080:.6f} '
                         f'{(x2 - x1) / 960:.6f} {(y2 - y1) / 540:.6f}')
        (args.out / 'labels' / f'{stem}.txt').write_text('\n'.join(lines) + '\n')
        written += 1
        labelled += len(lines)
        hidden += len(masked)
        if len(preview) < 6 and written % 25 == 1:
            shown = image.copy()
            for cls, (x1, y1, x2, y2) in labels:
                cv2.rectangle(shown, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)
                cv2.putText(shown, cls, (int(x1), int(y1) - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            preview.append(shown)

    print(f'{written} views, {labelled} labelled objects, {hidden} regions painted over -> {args.out}')
    if preview:
        while len(preview) % 2:
            preview.append(np.zeros_like(preview[0]))
        cv2.imwrite(str(args.out / 'preview.jpg'),
                    np.vstack([np.hstack(preview[i:i + 2]) for i in range(0, len(preview), 2)]))
    return 0


if __name__ == '__main__':
    sys.exit(main())
