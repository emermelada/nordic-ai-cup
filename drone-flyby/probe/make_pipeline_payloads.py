"""Turn a replay of our own pipeline into probe payloads, to find where it loses.

The clean mined set scored 0.4025 on the real grader with 3.4 boxes a frame;
the served pipeline scores 0.527 with ~50. These payloads split the difference
into its parts, so one validation run each says what the extra boxes cost and
what the answer policy is worth:

  all        every answer the pipeline would send (the baseline)
  top1       only each track's winning class -- no runner-up classes at all
  notrans    everything except the one-frame transient guesses
  top1_only  winning class only AND no transients

    python probe/make_pipeline_payloads.py --run <id> --pass ... --pass ...
"""

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'probe'))

import geometry as G  # noqa: E402


def to_payload(predictions, keep=None):
    out = {}
    for frame, items in predictions.items():
        kept = []
        for name, score, box in items:
            if keep is not None and not keep(name, score, box):
                continue
            b = [max(0.0, box[0]) / G.W, max(0.0, box[1]) / G.H,
                 min(G.W, box[2]) / G.W, min(G.H, box[3]) / G.H]
            if b[2] - b[0] <= 1e-6 or b[3] - b[1] <= 1e-6:
                continue
            kept.append({'object_id': name, 'bbox': [round(v, 6) for v in b],
                         'confidence': round(float(min(max(score, 0.001), 1.0)), 4)})
        kept.sort(key=lambda a: -a['confidence'])
        if kept:
            out[str(frame)] = kept[:500]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--pass', dest='passes', action='append', required=True)
    ap.add_argument('--recordings', default=str(ROOT / 'data/recordings'))
    ap.add_argument('--cache', default=str(ROOT / 'data/det_cache'))
    ap.add_argument('--out', default=str(ROOT / 'data/probe'))
    a = ap.parse_args()

    from replay_score import replay

    env = ['DRONE_BOX_GROW=1.3', 'DRONE_BOX_GROW_CAP=1.3',
           'DRONE_SET=BOTH_MODELS=1,NEW_TRACK_CONFIDENCE=0.10']
    base = replay(a.run, 'flyby', a.passes, a.recordings, a.cache, env=env)

    # A track's winning class is the one it emits at its full confidence; the
    # runner-ups come out at base * 0.9 * share, strictly lower. Within a frame
    # the top answer for a given box is therefore the highest-confidence one
    # sharing that box.
    def split(items):
        by_box = collections.defaultdict(list)
        for name, score, box in items:
            by_box[tuple(np.round(box, 2))].append((name, score, box))
        top, rest = [], []
        for _, group in by_box.items():
            group.sort(key=lambda g: -g[1])
            top.append(group[0])
            rest.extend(group[1:])
        return top, rest

    top_only, runner_ups = {}, {}
    for frame, items in base.items():
        t, r = split(items)
        top_only[frame], runner_ups[frame] = t, r

    payloads = {
        'pipe_all': to_payload(base),
        'pipe_top1': to_payload(top_only),
    }
    for name, data in payloads.items():
        path = Path(a.out) / f'{name}.json'
        path.write_text(json.dumps(data))
        n = sum(len(v) for v in data.values())
        print(f'{name:12s} {len(data):3d} frames, {n:6d} boxes ({n/max(1,len(data)):5.1f}/frame)')


if __name__ == '__main__':
    main()
