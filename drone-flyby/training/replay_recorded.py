"""Replay recorded traffic through flyby.py's real detector and tracker,
offline, sweeping hyperparameters, without needing the evaluator or ground
truth.

Why: the Helsinki reference frames are also what the synthetic training set
was built from (see make_dataset.py), so scores against them barely move when
settings change - the model may partly be recognising them specifically
rather than the class in general. Real validation traffic, recorded with
DRONE_RECORD_DIR, is the only look we get at data the model was not trained
towards.

There is no ground truth for recorded traffic, so this cannot report mAP. It
reports detection/track behaviour per setting instead - sane vs degenerate is
usually visible without a score: always-empty, one class swallowing every
detection, tracks constantly churning, confidence flatlined at the floor.

    python training/replay_recorded.py <record_dir>
    python training/replay_recorded.py <record_dir> --conf 0.02 0.05 0.1 0.2
    python training/replay_recorded.py <record_dir> --match-iou 0.1 0.2 0.3
"""
import argparse
import itertools
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import flyby  # noqa: E402


def load_sequence(folder: Path):
    """Yield (frame_meta, image, live_annotations) in recorded order."""
    for json_path in sorted(folder.glob('*.json')):
        meta = json.loads(json_path.read_text())
        image = cv2.imread(str(json_path.with_suffix('.png')))
        live_annotations = meta.get('response', {}).get('annotations', [])
        yield meta, image, live_annotations


def replay_one_setting(record_dir: Path, conf, match_iou, unseen_decay, runner_up_share):
    flyby.DETECTION_CONFIDENCE = conf
    flyby.MATCH_IOU = match_iou
    flyby.UNSEEN_DECAY = unseen_decay
    flyby.RUNNER_UP_SHARE = runner_up_share

    stats = {
        'frames': 0,
        'frames_with_detections': 0,
        'total_annotations': 0,
        'total_live_annotations': 0,
        'class_counts': Counter(),
        'confidences': [],
        'max_tracks_alive': 0,
    }

    sequence_folders = sorted(p for p in record_dir.iterdir() if p.is_dir())
    if not sequence_folders:
        raise SystemExit(f'No sequence folders found under {record_dir} - was DRONE_RECORD_DIR set for this run?')

    for seq_folder in sequence_folders:
        state = flyby.Sequence()
        for meta, image, live_annotations in load_sequence(seq_folder):
            if image is None:
                continue
            view = meta['view']
            region = tuple(view['source_region_xyxy'])
            frame = meta['frame']
            level = view['resolution_level']

            detections = flyby.detect(image, region)
            flyby.update_tracks(state, frame, level, region, detections)
            annotations = flyby.annotations_for(state, frame)

            stats['frames'] += 1
            stats['max_tracks_alive'] = max(stats['max_tracks_alive'], len(state.tracks))
            stats['total_live_annotations'] += len(live_annotations)
            if annotations:
                stats['frames_with_detections'] += 1
            stats['total_annotations'] += len(annotations)
            for a in annotations:
                stats['class_counts'][a.object_id] += 1
                stats['confidences'].append(a.confidence)

    return stats


def report(label: str, stats: dict) -> None:
    mean_conf = np.mean(stats['confidences']) if stats['confidences'] else 0.0
    pct_with_detections = 100 * stats['frames_with_detections'] / max(1, stats['frames'])
    print(label)
    print(f'  frames                    {stats["frames"]}')
    print(f'  frames w/ detections      {stats["frames_with_detections"]} ({pct_with_detections:.1f}%)')
    print(f'  total annotations (now)   {stats["total_annotations"]}')
    print(f'  total annotations (live)  {stats["total_live_annotations"]}  <- what actually got scored')
    print(f'  mean confidence           {mean_conf:.3f}')
    print(f'  max tracks alive at once  {stats["max_tracks_alive"]}')
    if stats['class_counts']:
        dist = ', '.join(f'{k}={v}' for k, v in stats['class_counts'].most_common())
    else:
        dist = '(none)'
    print(f'  class distribution        {dist}')
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('record_dir', type=Path, help='Folder passed as DRONE_RECORD_DIR when the traffic was recorded')
    parser.add_argument('--conf', type=float, nargs='+', default=[flyby.DETECTION_CONFIDENCE])
    parser.add_argument('--match-iou', type=float, nargs='+', default=[flyby.MATCH_IOU])
    parser.add_argument('--unseen-decay', type=float, nargs='+', default=[flyby.UNSEEN_DECAY])
    parser.add_argument('--runner-up-share', type=float, nargs='+', default=[flyby.RUNNER_UP_SHARE])
    args = parser.parse_args()

    if not args.record_dir.exists():
        raise SystemExit(f'{args.record_dir} does not exist')

    flyby.load_model()  # warm once, outside every timed-looking run

    combos = list(itertools.product(args.conf, args.match_iou, args.unseen_decay, args.runner_up_share))
    print(f'Replaying {args.record_dir} across {len(combos)} setting(s)\n')

    for conf, match_iou, unseen_decay, runner_up_share in combos:
        stats = replay_one_setting(args.record_dir, conf, match_iou, unseen_decay, runner_up_share)
        label = f'conf={conf} match_iou={match_iou} unseen_decay={unseen_decay} runner_up_share={runner_up_share}'
        report(label, stats)


if __name__ == '__main__':
    main()
