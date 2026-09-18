"""Score a validation run offline, the way the grader would.

Every configuration change so far has cost a 15-minute validation attempt with a
run-to-run noise floor of about +/-0.01, which is the same size as most of the
effects we were trying to measure. The validation flight is deterministic and we
have it recorded, so the same comparison can be made offline in minutes.

    python tools/score_offline.py --run 5ace53648dd5429bbf95332494a69ac0
    python tools/score_offline.py --replay --model ~/models/drone-yolo11n-v4.pt
    python tools/score_offline.py --replay --set RUNNER_UPS=8 --from-frame 200

--run scores the answers a recorded run actually sent. --replay puts the
recorded requests back through flyby.predict with cached detections, so tracker
and answer-policy settings can be compared without touching the network.

The ground truth is training/validation_objects.json carried to every frame by
the ground-motion model. It is NOT the real ground truth: it holds only the
objects we have confirmed, and those were mostly found by our own models, so the
absolute number reads high (0.226 against a real 0.1445 for the same run when
this was calibrated). Use it to compare configurations, and read the per-class
column to see which class to attack; do not quote it as the score.
"""

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'training'))
sys.path.insert(0, str(HERE))

RECORDINGS = ROOT / 'data' / 'recordings'
OBJECTS = ROOT / 'training' / 'validation_objects.json'
REACH = 30      # frames a confirmed sighting is carried


def known_boxes(objects, frame: int):
    from harvest_validation_patches import move
    out = []
    for obj in objects:
        near = min(obj['observations'], key=lambda o: abs(o['frame'] - frame))
        if abs(near['frame'] - frame) > REACH:
            continue
        box = np.clip(move(near['box'], frame - near['frame']), 0, [3840, 2160, 3840, 2160])
        if box[2] - box[0] > 2 and box[3] - box[1] > 2:
            out.append((obj['class'], box))
    return out


def score(predictions, frames, objects):
    """COCO mAP@0.50, macro over the classes present, as local_evaluator does."""
    from faster_coco_eval import COCO, COCOeval_faster
    from utils import OBJECT_CLASSES

    categories = {name: index for index, name in enumerate(OBJECT_CLASSES, start=1)}
    annotations, present, annotation_id = [], set(), 1
    for frame in frames:
        for name, box in known_boxes(objects, frame):
            present.add(name)
            annotations.append({
                'id': annotation_id, 'image_id': frame, 'category_id': categories[name],
                'bbox': [box[0], box[1], box[2] - box[0], box[3] - box[1]],
                'area': float((box[2] - box[0]) * (box[3] - box[1])), 'iscrowd': 0})
            annotation_id += 1
    evaluated = tuple(name for name in OBJECT_CLASSES if name in present)
    if not evaluated:
        raise SystemExit('no confirmed objects in these frames')

    ground_truth = {
        'info': {'description': 'recorded validation flight'}, 'licenses': [],
        'images': [{'id': f, 'file_name': f'frame_{f:06d}.png', 'width': 3840, 'height': 2160}
                   for f in frames],
        'categories': [{'id': i, 'name': n, 'supercategory': 'object'} for n, i in categories.items()],
        'annotations': annotations}

    detections = []
    for frame in frames:
        for name, confidence, box in predictions.get(frame, []):
            if box[2] - box[0] <= 0 or box[3] - box[1] <= 0:
                continue
            detections.append({'image_id': frame, 'category_id': categories[name],
                               'bbox': [box[0], box[1], box[2] - box[0], box[3] - box[1]],
                               'score': float(confidence)})
    if not detections:
        return 0.0, {name: 0.0 for name in evaluated}, len(annotations)

    coco_gt = COCO(ground_truth)
    evaluator = COCOeval_faster(coco_gt, coco_gt.loadRes(detections), 'bbox')
    evaluator.params.imgIds = list(frames)
    evaluator.params.catIds = [categories[n] for n in evaluated]
    evaluator.params.iouThrs = np.array([0.50])
    evaluator.evaluate()
    evaluator.accumulate()

    precision = evaluator.eval['precision']
    by_class = {}
    for index, name in enumerate(evaluated):
        values = precision[0, :, index, 0, -1]
        values = values[values > -1]
        by_class[name] = float(np.mean(values)) if values.size else 0.0
    return sum(by_class.values()) / len(by_class), by_class, len(annotations)


def from_recording(run: str):
    """The answers this run actually sent."""
    predictions = collections.defaultdict(list)
    for meta_path in sorted((RECORDINGS / run).glob('*.json')):
        meta = json.loads(meta_path.read_text())
        for annotation in (meta.get('response') or {}).get('annotations') or []:
            box = np.array(annotation['bbox'], float) * [3840, 2160, 3840, 2160]
            predictions[meta['frame']].append((annotation['object_id'], annotation['confidence'], box))
    return predictions


def from_replay(run: str, model: Path, overrides):
    """Replay the recorded requests through flyby.predict with cached detections."""
    import logging
    logging.disable(logging.WARNING)
    import os
    os.environ['DRONE_MODEL'] = str(model)
    import flyby
    from bench_recordings import cached_detections
    from dtos import DroneFlybyPredictRequestDto

    for item in overrides:
        name, value = item.split('=', 1)
        setattr(flyby, name, type(getattr(flyby, name))(eval(value)))

    cache = cached_detections(model)
    current = {}
    flyby.decode_view = lambda view: None

    def fake_detect(image, region):
        xyxy, probabilities = cache[current['key']]
        rx1, ry1, rx2, ry2 = region
        scale = np.array([(rx2 - rx1) / 960, (ry2 - ry1) / 540] * 2)
        boxes = xyxy * scale + [rx1, ry1, rx1, ry1]
        return [(flyby.OBJECT_CLASSES[int(p.argmax())], float(p.max()), b, p)
                for b, p in zip(boxes, probabilities) if p.max() >= flyby.DETECTION_CONFIDENCE]

    flyby.detect = fake_detect
    flyby._sequences.clear()
    predictions = collections.defaultdict(list)
    for meta_path in sorted((RECORDINGS / run).glob('*.json')):
        meta = json.loads(meta_path.read_text())
        meta.pop('response', None)
        meta['view']['image'] = ''
        current['key'] = f'{run}/{meta_path.stem}.png'
        response = flyby.predict(DroneFlybyPredictRequestDto(**meta))
        for annotation in response.annotations:
            box = np.array(annotation.bbox, float) * [3840, 2160, 3840, 2160]
            predictions[meta['frame']].append((annotation.object_id, annotation.confidence, box))
    return predictions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', default='5ace53648dd5429bbf95332494a69ac0',
                        help='Recorded run to score (default: the best v4 run).')
    parser.add_argument('--replay', action='store_true', help='Re-run flyby.predict instead of scoring stored answers.')
    parser.add_argument('--model', type=Path, default=Path.home() / 'models' / 'drone-yolo11n-v4.pt')
    parser.add_argument('--set', action='append', default=[], help='NAME=value flyby setting override (with --replay).')
    parser.add_argument('--from-frame', type=int, default=0, help='Only score frames from here on.')
    parser.add_argument('--frames', type=int, default=249)
    args = parser.parse_args()

    if not (RECORDINGS / args.run).is_dir():
        raise SystemExit(f'no recording {args.run}')
    objects = json.loads(OBJECTS.read_text())['objects']
    frames = [f for f in range(1, args.frames + 1) if f >= args.from_frame]

    predictions = from_replay(args.run, args.model, args.set) if args.replay else from_recording(args.run)
    total = sum(len(v) for v in predictions.values())
    overall, by_class, instances = score(predictions, frames, objects)

    label = f'replay {args.model.name}' if args.replay else f'recorded answers of {args.run[:12]}'
    print(f'{label}')
    print(f'{len(frames)} frames, {instances} confirmed object-frames, {total} predictions '
          f'({total / max(1, len(frames)):.1f} per frame, COCO caps at 100)')
    if args.set:
        print('overrides: ' + ', '.join(args.set))
    print(f'\nmAP@0.50 (confirmed objects only) = {overall:.3f}\n')
    for name, value in sorted(by_class.items(), key=lambda item: -item[1]):
        print(f'   {name:18s} {value:.3f}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
