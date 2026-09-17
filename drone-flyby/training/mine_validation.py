"""Find more objects in the recorded validation runs, for review by eye.

    python training/mine_validation.py --model ~/models/drone-yolo11n-v3.pt
    python training/mine_validation.py --accept 3:tank 7:helicopter 12:condor

The model runs over every recorded view (cached by tools/bench_recordings.py),
confident detections are converted to source pixels and grouped into objects
with the ground-motion model, and objects already listed in
validation_objects.json are skipped. The rest are written to
data/mining/candidates.json with review sheets (data/mining/sheet_*.jpg: each
row is one candidate, crops from the views that show it, close-ups first).

After looking at the sheets, --accept adds the chosen candidates, with the
class you give them, to validation_objects.json.
"""

import argparse
import collections
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

RECORDINGS = ROOT / 'data' / 'recordings'
MINING = ROOT / 'data' / 'mining'
OBJECTS = HERE / 'validation_objects.json'
REACH = 25          # frames a sighting is carried when collecting crops
ROWS_PER_SHEET = 16
CROPS_PER_ROW = 8


def views():
    out = []
    for meta_path in sorted(RECORDINGS.glob('*/*.json')):
        meta = json.loads(meta_path.read_text())
        view = meta['view']
        out.append({
            'key': str(meta_path.with_suffix('.png').relative_to(RECORDINGS)),
            'png': meta_path.with_suffix('.png'),
            'frame': meta['frame'],
            'level': view['resolution_level'],
            'region': view['source_region_xyxy'],
        })
    return out


def group(sightings, same):
    """Greedy grouping in frame order; ``same(object, sighting)`` decides."""
    objects = []
    for sighting in sorted(sightings, key=lambda s: s['frame']):
        for obj in objects:
            if same(obj, sighting):
                obj['sightings'].append(sighting)
                break
        else:
            objects.append({'sightings': [sighting]})
    return objects


def mine(args):
    import os
    os.environ['DRONE_MODEL'] = str(args.model)
    import flyby
    from bench_recordings import cached_detections
    from harvest_validation_patches import move

    detections = cached_detections(args.model)
    known = json.loads(OBJECTS.read_text())['objects']
    all_views = views()

    sightings = []
    for view in all_views:
        if view['level'] == 0 or view['key'] not in detections:
            continue
        xyxy, probabilities = detections[view['key']]
        rx1, ry1, rx2, ry2 = view['region']
        scale = (rx2 - rx1) / 960
        for box, p in zip(xyxy, probabilities):
            if p.max() < args.min_conf:
                continue
            # Skip boxes cut by the view edge: their size is wrong.
            if box[0] < 3 or box[1] < 3 or box[2] > 957 or box[3] > 537:
                continue
            sightings.append({
                'frame': view['frame'], 'level': view['level'], 'probs': p,
                'box': box * scale + [rx1, ry1, rx1, ry1],
            })

    def same(obj, sighting):
        last = obj['sightings'][-1]
        carried = move(last['box'], sighting['frame'] - last['frame'])
        return (sighting['frame'] - last['frame'] <= 40
                and (flyby.iou(carried, sighting['box']) > 0.3 or flyby.cover(carried, sighting['box']) > 0.7))

    def is_known(obj):
        s = obj['sightings'][0]
        for k in known:
            near = min(k['observations'], key=lambda o: abs(o['frame'] - s['frame']))
            if abs(near['frame'] - s['frame']) > 40:
                continue
            if flyby.cover(move(near['box'], s['frame'] - near['frame']), s['box']) > 0.3:
                return True
        return False

    candidates = []
    for obj in group(sightings, same):
        if len(obj['sightings']) < args.min_sightings or is_known(obj):
            continue
        votes = sum(s['probs'] for s in obj['sightings'])
        ranked = np.argsort(-votes)[:3]
        candidates.append({
            'id': len(candidates),
            'guess': [(flyby.OBJECT_CLASSES[i], round(float(votes[i]), 2)) for i in ranked],
            'observations': [{'frame': s['frame'], 'level': s['level'],
                              'box': [round(float(v), 1) for v in s['box']]} for s in obj['sightings']],
        })

    MINING.mkdir(parents=True, exist_ok=True)
    (MINING / 'candidates.json').write_text(json.dumps(candidates, indent=1))
    for old in MINING.glob('sheet_*.jpg'):
        old.unlink()

    rows = []
    for cand in candidates:
        crops = []
        for view in sorted(all_views, key=lambda v: -v['level']):
            if view['level'] == 0:
                continue
            near = min(cand['observations'], key=lambda o: abs(o['frame'] - view['frame']))
            if abs(near['frame'] - view['frame']) > REACH:
                continue
            rx1, ry1, rx2, ry2 = view['region']
            scale = (rx2 - rx1) / 960
            x1, y1, x2, y2 = (move(near['box'], view['frame'] - near['frame']) - [rx1, ry1, rx1, ry1]) / scale
            if x1 < 0 or y1 < 0 or x2 > 960 or y2 > 540:
                continue
            image = cv2.imread(str(view['png']))
            cx, cy, r = (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1) + 8
            crop = image[int(max(0, cy - r)):int(cy + r), int(max(0, cx - r)):int(cx + r)]
            if crop.size == 0:
                continue
            crop = cv2.resize(crop, (110, 110), interpolation=cv2.INTER_CUBIC)
            cv2.putText(crop, f'L{view["level"]}', (2, 106), 0, 0.35, (0, 255, 255), 1)
            crops.append(crop)
            if len(crops) == CROPS_PER_ROW:
                break
        crops += [np.zeros((110, 110, 3), np.uint8)] * (CROPS_PER_ROW - len(crops))
        label = np.zeros((110, 190, 3), np.uint8)
        cv2.putText(label, f'#{cand["id"]}  n={len(cand["observations"])}', (3, 18), 0, 0.45, (255, 255, 255), 1)
        for k, (name, vote) in enumerate(cand['guess']):
            cv2.putText(label, f'{name} {vote}', (3, 42 + 20 * k), 0, 0.4, (200, 200, 200), 1)
        rows.append(np.hstack([label] + crops))
    for start in range(0, len(rows), ROWS_PER_SHEET):
        cv2.imwrite(str(MINING / f'sheet_{start // ROWS_PER_SHEET:02d}.jpg'),
                    np.vstack(rows[start:start + ROWS_PER_SHEET]))
    print(f'{len(sightings)} sightings -> {len(candidates)} new candidates; '
          f'sheets in {MINING}')


def accept(pairs):
    import flyby
    candidates = {c['id']: c for c in json.loads((MINING / 'candidates.json').read_text())}
    data = json.loads(OBJECTS.read_text())
    next_id = max(o['id'] for o in data['objects']) + 1
    for pair in pairs:
        cid, cls = pair.split(':')
        if cls not in flyby.OBJECT_CLASSES:
            raise SystemExit(f'unknown class {cls}')
        cand = candidates[int(cid)]
        data['objects'].append({
            'id': next_id, 'class': cls, 'mined_as': int(cid),
            'observations': [{'frame': o['frame'], 'box': o['box']} for o in cand['observations']],
        })
        next_id += 1
    OBJECTS.write_text(json.dumps(data, indent=1))
    print(f'{len(pairs)} objects added; {len(data["objects"])} in {OBJECTS.name}')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', type=Path, default=Path.home() / 'models' / 'drone-yolo11n-v2.pt')
    parser.add_argument('--min-conf', type=float, default=0.4)
    parser.add_argument('--min-sightings', type=int, default=2)
    parser.add_argument('--accept', nargs='*', help='candidate:class pairs to add to the object list')
    args = parser.parse_args()
    import logging
    logging.disable(logging.WARNING)
    if args.accept:
        accept(args.accept)
    else:
        mine(args)
    return 0


if __name__ == '__main__':
    sys.exit(main())
