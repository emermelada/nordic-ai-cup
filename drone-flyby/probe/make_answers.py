"""Turn a truth-style object file into an answers file the probe server replays.

    python probe/make_answers.py training/scene_objects.json out.json --grow 1.3
    python probe/make_answers.py training/scene_objects.json out.json --classes tank --grow 1.3

Object files are {"objects": [{"id", "class", "observations": [{"frame", "box"}]}]}
with boxes in source pixels. Every observation becomes one answer for its frame,
grown about its centre, clipped to the frame and normalized. Confidence is per
object (all its frames share it), ordered by how many frames it was observed in,
so the grader's ranking is deterministic and not a tie-break accident.
"""

import argparse
import json
from pathlib import Path

W, H = 3840, 2160


def grown(box, gw, gh):
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = (x2 - x1) * gw / 2, (y2 - y1) * gh / 2
    return cx - w, cy - h, cx + w, cy + h


def to_global(box):
    x1, y1, x2, y2 = box
    x1, x2 = max(0.0, min(W, x1)) / W, max(0.0, min(W, x2)) / W
    y1, y2 = max(0.0, min(H, y1)) / H, max(0.0, min(H, y2)) / H
    if x2 - x1 <= 1e-6 or y2 - y1 <= 1e-6:
        return None
    return [round(x1, 6), round(y1, 6), round(x2, 6), round(y2, 6)]


def build(objects, classes=None, grow=1.0, grow_h=None, frames=None):
    gh = grow if grow_h is None else grow_h
    ranked = sorted(objects, key=lambda o: -len(o['observations']))
    answers = {}
    n = len(ranked)
    for rank, obj in enumerate(ranked):
        if classes and obj['class'] not in classes:
            continue
        confidence = round(0.99 - 0.9 * rank / max(1, n), 4)
        for obs in obj['observations']:
            if frames is not None and obs['frame'] not in frames:
                continue
            bbox = to_global(grown(obs['box'], grow, gh))
            if bbox is None:
                continue
            answers.setdefault(str(obs['frame']), []).append(
                {'object_id': obj['class'], 'bbox': bbox, 'confidence': confidence})
    return answers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('truth')
    ap.add_argument('out')
    ap.add_argument('--classes', default='')
    ap.add_argument('--grow', type=float, default=1.0)
    ap.add_argument('--grow-h', type=float, default=None)
    a = ap.parse_args()
    objects = json.loads(Path(a.truth).read_text())['objects']
    classes = set(filter(None, a.classes.split(',')))
    answers = build(objects, classes, a.grow, a.grow_h)
    Path(a.out).write_text(json.dumps(answers))
    print(f'{a.out}: {len(answers)} frames, {sum(map(len, answers.values()))} boxes')


if __name__ == '__main__':
    main()
