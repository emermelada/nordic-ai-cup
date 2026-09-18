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
a ground motion fitted on the flight's own objects --- NOT flyby.MOTION, which
was fitted on Helsinki and runs ~2 px/frame short here. Carrying the truth with
the same motion the tracker uses makes both drift together and cancels the
error, which is how a wrong motion stays invisible offline (found by Franek on
branch franek-drone-flyby-motion-fit; the same trap was in this tool).

It is still NOT the real ground truth: it holds only the
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
IGNORE = ROOT / 'training' / 'validation_ignore.json'
REACH = 30      # frames a confirmed sighting is carried


MOTION_MAX_GAP = 12     # frames between two sightings that still make a sample


def fit_truth_motion(objects):
    """Per-frame ground motion fitted on the confirmed objects themselves.

    Every object in the scene is a static prop, so the drift between two
    sightings of the same object is the ground motion. Fits dx and dy as
    a + b*x + c*y over both corners of each pair, the same form as flyby.MOTION.
    """
    import flyby
    points, deltas = [], []
    for obj in objects:
        seen = sorted(obj['observations'], key=lambda o: o['frame'])
        for first, second in zip(seen, seen[1:]):
            gap = second['frame'] - first['frame']
            if not 1 <= gap <= MOTION_MAX_GAP:
                continue
            for i in (0, 2):
                start = np.array([first['box'][i], first['box'][i + 1]], float)
                end = np.array([second['box'][i], second['box'][i + 1]], float)
                points.append(start)
                deltas.append((end - start) / gap)
    if len(points) < 12:
        return flyby.MOTION, 0
    points = np.array(points)
    deltas = np.array(deltas)
    design = np.column_stack([np.ones(len(points)), points[:, 0], points[:, 1]])
    (a, b, c), *_ = np.linalg.lstsq(design, deltas[:, 0], rcond=None)
    (d, e, f), *_ = np.linalg.lstsq(design, deltas[:, 1], rcond=None)
    return (float(a), float(b), float(c), float(d), float(e), float(f)), len(points)


def carry(box, steps: int, motion):
    """Move a box that many frames of ground motion, forwards or backwards."""
    a, b, c, d, e, f = motion
    forward = np.array([[1 + b, c], [e, 1 + f]])
    shift = np.array([a, d])
    points = np.array([[box[0], box[1]], [box[2], box[3]]], float)
    for _ in range(abs(steps)):
        points = (points @ forward.T + shift if steps > 0
                  else (points - shift) @ np.linalg.inv(forward).T)
    return points.reshape(-1)


def known_boxes(objects, frame: int, motion=None):
    import flyby
    motion = flyby.MOTION if motion is None else motion
    out = []
    for obj in objects:
        near = min(obj['observations'], key=lambda o: abs(o['frame'] - frame))
        if abs(near['frame'] - frame) > REACH:
            continue
        box = np.clip(carry(near['box'], frame - near['frame'], motion), 0, [3840, 2160, 3840, 2160])
        if box[2] - box[0] > 2 and box[3] - box[1] > 2:
            out.append((obj['class'], box))
    return out


def ignore_regions(verdicts=('confirmed',)):
    """Regions that are real objects we never labelled, as COCO ignore boxes.

    The confirmed object list holds 29 objects mined by v2, but the flight has
    more: cross-model mining found a second large_launcher, a second and third
    large_tower, a second mine_roller and another jet_plane, all seen by two
    independently trained models over 18-26 recorded runs, several at Level 2.
    Scoring counted every detection of those as a false positive, which is a
    precision penalty that falls hardest on whichever config emits the most
    boxes --- and with a 12-class macro average resting on single objects for
    four of those classes, one missing object moves the total enough to invert
    a model choice. Measured: with these regions ignored, the offline ranking of
    six configurations with real validation scores goes from Spearman +0.77 to
    +0.94, and v4@960+v6@1280 correctly overtakes v7@1280.

    An ignore region is not a label. It says only "do not score anything here",
    so a wrong one costs a little precision signal and cannot poison training.
    """
    if not IGNORE.exists():
        return []
    data = json.loads(IGNORE.read_text())
    return [r for r in data['regions'] if r['verdict'] in verdicts]


def ignore_boxes(regions, frame: int, motion):
    out = []
    for region in regions:
        near = min(region['observations'], key=lambda o: abs(o['frame'] - frame))
        if abs(near['frame'] - frame) > 20:
            continue
        box = np.clip(carry(near['box'], frame - near['frame'], motion), 0, [3840, 2160, 3840, 2160])
        if box[2] - box[0] > 2 and box[3] - box[1] > 2:
            out.append(box)
    return out


def score(predictions, frames, objects, motion=None, regions=()):
    """COCO mAP@0.50, macro over the classes present, as local_evaluator does."""
    from faster_coco_eval import COCO, COCOeval_faster
    from utils import OBJECT_CLASSES

    categories = {name: index for index, name in enumerate(OBJECT_CLASSES, start=1)}
    annotations, present, annotation_id = [], set(), 1
    for frame in frames:
        for name, box in known_boxes(objects, frame, motion):
            present.add(name)
            annotations.append({
                'id': annotation_id, 'image_id': frame, 'category_id': categories[name],
                'bbox': [box[0], box[1], box[2] - box[0], box[3] - box[1]],
                'area': float((box[2] - box[0]) * (box[3] - box[1])), 'iscrowd': 0})
            annotation_id += 1
        # iscrowd=1 in every class: we do not know what is here, so a detection
        # landing on it is neither a true nor a false positive. COCO matches
        # real annotations first, so a confirmed object nearby still scores.
        for box in ignore_boxes(regions, frame, motion):
            for name in categories:
                annotations.append({
                    'id': annotation_id, 'image_id': frame, 'category_id': categories[name],
                    'bbox': [box[0], box[1], box[2] - box[0], box[3] - box[1]],
                    'area': float((box[2] - box[0]) * (box[3] - box[1])), 'iscrowd': 1})
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


def from_replay(run: str, model: Path, overrides, model_alt: Path = None,
                imgsz=None, imgsz_alt=None):
    """Replay the recorded requests through flyby.predict with cached detections.

    With ``model_alt``, this mirrors what the service actually does with two sets
    of weights: flyby.raw_detections picks ``_models[which % len(_models)]``, so
    the models alternate on the request frame, and DRONE_SET=BOTH_MODELS=1 runs
    both on every frame and concatenates. The patch below replaces flyby.detect,
    which sits *above* that selection, so without this the served pair was
    invisible here and --model-alt would have changed nothing.
    """
    import logging
    logging.disable(logging.WARNING)
    import os
    os.environ['DRONE_MODEL'] = str(model)
    if model_alt is not None:
        os.environ['DRONE_MODEL_ALT'] = str(model_alt)
    import flyby
    from bench_recordings import cached_detections
    from dtos import DroneFlybyPredictRequestDto

    for item in overrides:
        name, value = item.split('=', 1)
        setattr(flyby, name, type(getattr(flyby, name))(eval(value)))

    def cache_at(path: Path, size):
        """Detections for ``path`` at ``size``, computing them only if needed.

        Both globals get set. flyby.IMGSZ is what cached_detections keys the
        cache file on; flyby.IMGSZ_LIST is what raw_detections actually resizes
        with. Setting only the first names the file 1280 while computing at 960,
        which is a poisoned cache that reads as a working one.
        """
        if size is not None:
            flyby.IMGSZ, flyby.IMGSZ_LIST = size, [size]
        return cached_detections(path)

    caches = [cache_at(model, imgsz)]
    if model_alt is not None:
        caches.append(cache_at(model_alt, imgsz_alt))
    current = {}
    flyby.decode_view = lambda view: None

    def fake_detect(image, region, frame: int = 0):
        if flyby.BOTH_MODELS and len(caches) > 1:
            parts = [cache[current['key']] for cache in caches]
            xyxy = np.concatenate([part[0] for part in parts])
            probabilities = np.concatenate([part[1] for part in parts])
        else:
            xyxy, probabilities = caches[frame % len(caches)][current['key']]
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
    parser.add_argument('--model', type=Path, default=ROOT / 'models' / 'drone-yolo11n-v4.pt')
    parser.add_argument('--model-alt', type=Path, default=None,
                        help='Second weights, alternating per frame like the served pair. '
                             'Either model may carry an inference size as PATH:SIZE, so '
                             'one model at two scales is a pair too. '
                             'Add --set BOTH_MODELS=1 to run both on every frame instead.')
    parser.add_argument('--set', action='append', default=[], help='NAME=value flyby setting override (with --replay).')
    parser.add_argument('--from-frame', type=int, default=0, help='Only score frames from here on.')
    parser.add_argument('--frames', type=int, default=249)
    parser.add_argument('--no-ignore', action='store_true',
                        help='Score unlabelled-object regions as false positives, the old '
                             'behaviour. Anti-correlated with real scores; see ignore_regions().')
    parser.add_argument('--ignore-verdicts', default='confirmed',
                        help="Which validation_ignore.json verdicts to honour "
                             "(comma-separated; default 'confirmed').")
    parser.add_argument('--truth-motion', choices=['fitted', 'prior'], default='fitted',
                        help="'fitted' carries the truth with a motion fitted on the "
                             "confirmed objects; 'prior' uses flyby.MOTION, which drifts "
                             'with the tracker and hides motion error.')
    args = parser.parse_args()

    def split_size(value):
        """'weights.pt:1280' -> (Path, 1280); a bare path -> (Path, None)."""
        if value is None:
            return None, None
        text = str(value)
        base, sep, size = text.rpartition(':')
        if sep and size.isdigit():
            return Path(base), int(size)
        return Path(text), None

    model, imgsz = split_size(args.model)
    model_alt, imgsz_alt = split_size(args.model_alt)

    if not (RECORDINGS / args.run).is_dir():
        raise SystemExit(f'no recording {args.run}')
    objects = json.loads(OBJECTS.read_text())['objects']
    frames = [f for f in range(1, args.frames + 1) if f >= args.from_frame]

    predictions = (from_replay(args.run, model, args.set, model_alt, imgsz, imgsz_alt)
                   if args.replay else from_recording(args.run))
    # Only the frames actually scored, or --frames makes the per-frame rate nonsense.
    total = sum(len(predictions.get(f, [])) for f in frames)
    import flyby
    motion = flyby.MOTION if args.truth_motion == 'prior' else fit_truth_motion(objects)[0]
    samples = fit_truth_motion(objects)[1] if args.truth_motion != 'prior' else 0
    regions = () if args.no_ignore else ignore_regions(tuple(args.ignore_verdicts.split(',')))
    overall, by_class, instances = score(predictions, frames, objects, motion, regions)

    if args.replay:
        label = f'replay {model.name}' + (f'@{imgsz}' if imgsz else '')
        if model_alt is not None:
            mode = 'both every frame' if flyby.BOTH_MODELS else 'alternating'
            label += f' + {model_alt.name}' + (f'@{imgsz_alt}' if imgsz_alt else '') + f' ({mode})'
    else:
        label = f'recorded answers of {args.run[:12]}'
    print(f'{label}')
    print(f'{len(frames)} frames, {instances} confirmed object-frames, {total} predictions '
          f'({total / max(1, len(frames)):.1f} per frame, COCO caps at 100)')
    centre = lambda m: m[3] + m[4] * 1920 + m[5] * 1080
    print(f'truth carried at {centre(motion):.2f} px/frame at the frame centre '
          f'({args.truth_motion}'
          f'{f", {samples} samples" if samples else ""}; '
          f'flyby prior {centre(flyby.MOTION):.2f})')
    if args.set:
        print('overrides: ' + ', '.join(args.set))
    print(f'{len(regions)} unlabelled-object regions ignored'
          if regions else 'no ignore regions (--no-ignore): unlabelled objects score as false positives')
    print(f'\nmAP@0.50 (confirmed objects only) = {overall:.3f}\n')
    for name, value in sorted(by_class.items(), key=lambda item: -item[1]):
        print(f'   {name:18s} {value:.3f}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
