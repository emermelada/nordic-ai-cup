"""Promote a reviewed ignore-region candidate into the confirmed object list.

    python training/promote_candidates.py --list
    python training/promote_candidates.py --accept 2:large_launcher --dry-run
    python training/promote_candidates.py --accept 2:large_launcher 3:large_tower

`training/validation_ignore.json` holds regions that are almost certainly real
objects the ground truth never listed. As ignore regions they only stop the
scorer counting detections there as false positives, which is safe: a wrong one
costs a little precision signal and reaches nothing else. Promoting one to a
real object in `validation_objects.json` is a different and larger commitment --
it becomes ground truth for scoring AND a source of real cut-outs for
`harvest_validation_patches.py`, so a wrong one poisons the training set too.

That is why nothing is promoted automatically and the verdict in the file is not
enough: you name each candidate and its class yourself, because the class in the
candidate file is the detector's vote, not a human's judgement.

A promoted region has its verdict set to "promoted", which stops it being used
as an ignore region as well -- an object cannot be both the answer and a hole in
the answer sheet.
"""

import argparse
import collections
import json
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

OBJECTS = HERE / 'validation_objects.json'
IGNORE = HERE / 'validation_ignore.json'


def observations_for(candidate):
    """One observation per frame: the sharpest, then most confident, sighting.

    Candidates are mined across all 28 recorded runs, so the same frame usually
    appears several times -- different runs saw the same ground at different
    resolution levels. Leaving duplicates in would bias `fit_truth_motion`,
    which pairs consecutive sightings, and `known_boxes`, which picks the
    nearest one by frame.
    """
    best = {}
    for observation in candidate['observations' if 'observations' in candidate else 'obs']:
        frame = observation['frame']
        rank = (observation.get('level', 1), observation.get('conf', 0.0))
        if frame not in best or rank > best[frame][0]:
            best[frame] = (rank, {'frame': frame, 'box': [round(float(v), 1) for v in observation['box']]})
    return [entry for _, (_rank, entry) in sorted(best.items())]


def load():
    if not IGNORE.exists():
        raise SystemExit(f'no {IGNORE}')
    return json.loads(OBJECTS.read_text()), json.loads(IGNORE.read_text())


def show(data, ignore):
    promoted = {o.get('promoted_from') for o in data['objects'] if o.get('promoted_from') is not None}
    print(f"{'id':>3s}  {'verdict':<18s} {'guess':<16s} {'models':<10s} {'runs':>5s} {'frames':>7s} {'conf':>5s}  note")
    for region in ignore['regions']:
        mark = '  <- already promoted' if region['id'] in promoted else ''
        print(f"{region['id']:>3d}  {region['verdict']:<18s} {region['guess_class']:<16s} "
              f"{'+'.join(region['models']):<10s} {region['runs']:>5d} {region['n_frames']:>7d} "
              f"{region['max_conf']:>5.2f}  {region['why']}{mark}")
    counts = collections.Counter(o['class'] for o in data['objects'])
    print(f"\n{len(data['objects'])} confirmed objects: " +
          ', '.join(f'{name} {n}' for name, n in sorted(counts.items())))


def promote(pairs, dry_run):
    from utils import OBJECT_CLASSES

    data, ignore = load()
    regions = {region['id']: region for region in ignore['regions']}
    already = {o.get('promoted_from') for o in data['objects'] if o.get('promoted_from') is not None}
    next_id = max(o['id'] for o in data['objects']) + 1
    before = collections.Counter(o['class'] for o in data['objects'])

    added = []
    for pair in pairs:
        if ':' not in pair:
            raise SystemExit(f'expected id:class, got {pair!r}')
        text, name = pair.split(':', 1)
        if not text.isdigit() or int(text) not in regions:
            raise SystemExit(f'no candidate {text}; run --list')
        if name not in OBJECT_CLASSES:
            raise SystemExit(f'unknown class {name!r}')
        region = regions[int(text)]
        if region['id'] in already:
            raise SystemExit(f'candidate {region["id"]} was promoted already')
        if region['verdict'] == 'rejected':
            print(f'  ! candidate {region["id"]} is marked rejected in {IGNORE.name}; promoting anyway')
        if name != region['guess_class']:
            print(f'  ! candidate {region["id"]}: you say {name}, the detectors voted '
                  f'{region["guess_class"]}')
        observations = observations_for(region)
        added.append({
            'id': next_id, 'class': name, 'promoted_from': region['id'],
            'provenance': (f"mined by cross-model agreement ({'+'.join(region['models'])}) over "
                           f"{region['runs']} recorded runs, max confidence {region['max_conf']:.2f}, "
                           f"reviewed by eye"),
            'observations': observations})
        print(f'  + {name:16s} from candidate {region["id"]:>2d}: {len(observations)} observations, '
              f"frames {observations[0]['frame']}-{observations[-1]['frame']}")
        next_id += 1

    if not added:
        raise SystemExit('nothing to promote')

    data['objects'].extend(added)
    for entry in added:
        regions[entry['promoted_from']]['verdict'] = 'promoted'

    after = collections.Counter(o['class'] for o in data['objects'])
    print(f'\n{len(data["objects"]) - len(added)} -> {len(data["objects"])} confirmed objects')
    for name in sorted(set(before) | set(after)):
        if before[name] != after[name]:
            print(f'   {name:16s} {before[name]} -> {after[name]}')

    if dry_run:
        print('\n--dry-run: nothing written')
        return 0

    stamp = time.strftime('%Y%m%d-%H%M%S')
    for path in (OBJECTS, IGNORE):
        shutil.copy2(path, path.with_suffix(f'.json.bak-{stamp}'))
    OBJECTS.write_text(json.dumps(data, indent=1))
    IGNORE.write_text(json.dumps(ignore, indent=1))
    print(f'\nwritten; backups at *.json.bak-{stamp}')
    print('\nCheck the effect before trusting it -- a promoted object changes the scale of the\n'
          'offline score, so compare against a re-run baseline, not against an older number:\n'
          '  python tools/score_offline.py --replay --model models/drone-yolo11n-v4.pt:960 \\\n'
          '      --model-alt models/drone-yolo11s-v6.pt:1280 --set BOTH_MODELS=1')
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--list', action='store_true', help='Show the candidates and stop.')
    parser.add_argument('--accept', nargs='*', default=[], metavar='ID:CLASS',
                        help='Candidates to promote, each with the class YOU judge it to be.')
    parser.add_argument('--dry-run', action='store_true', help='Report what would change, write nothing.')
    args = parser.parse_args()
    if args.list or not args.accept:
        show(*load())
        return 0
    return promote(args.accept, args.dry_run)


if __name__ == '__main__':
    raise SystemExit(main())
