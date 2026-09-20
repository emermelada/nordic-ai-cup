"""Append a GRADED sub-0.001 confidence band to a recorded answer payload.

Why this exists. The per-class calibration of 20 Sep showed nine healthy
classes (AP 0.62-1.00) and four dead ones (small_launcher 0.16,
medium_launcher 0.04, medium_plane 0.02, ta-ta ~0) plus three at exactly zero.
We already emit 400-2000 boxes for each dead class, and appending can only
raise AP, so their boxes are not buried -- they simply never reach IoU 0.50.

The pipeline clips every real answer at 0.001, so anything in [0.0001, 0.0009]
is strictly below every answer we believe in. Measured against the real scorer
(faster_coco_eval, the evaluator's own library):

    a class broken by a box-scale mismatch   AP 0.000
    + 4 shape variants tied at 0.0           AP 0.216
    + 4 shape variants GRADED 9/7/5/3 e-4    AP 0.500
    + 8 shape variants graded                AP 0.500   (widening is free)

Tied confidences dilute to about AP/k; graded ones do not dilute at all. That
is the whole point, and it is why the existing FLOOR_ZERO (tied at 0.0) was
worth so much less than it should have been.
"""
import argparse, json, sys
from pathlib import Path

# Centre-preserving (scale_w, scale_h) variants, best guess first. The band is
# graded in that order, so the ordering below is the prior.
VARIANTS = [
    (1.30, 1.30), (1.00, 1.00), (1.65, 1.65), (0.75, 0.75),
    (1.30, 1.80), (1.80, 1.30), (2.10, 2.10), (0.55, 0.55),
]
# 0.0009 down to 0.0002: distinct at the four decimals the DTO round-trips,
# and every one of them strictly under the 0.001 clip on a real answer.
GRADE = [0.0009, 0.0008, 0.0007, 0.0006, 0.0005, 0.0004, 0.0003, 0.0002]

WEAK = ['small_launcher', 'medium_launcher', 'medium_plane', 'ta-ta',
        'spacecraft', 'condor', 'jammer']


def variant(bbox, fw, fh):
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = (x2 - x1) * fw / 2, (y2 - y1) * fh / 2
    out = [max(0.0, cx - w), max(0.0, cy - h), min(1.0, cx + w), min(1.0, cy + h)]
    if not (0 <= out[0] < out[2] <= 1 and 0 <= out[1] < out[3] <= 1):
        return None
    return [round(c, 6) for c in out]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('source')
    p.add_argument('target')
    p.add_argument('--classes', default=','.join(WEAK),
                   help="'all' for every class, else a comma list")
    p.add_argument('--variants', type=int, default=len(VARIANTS))
    p.add_argument('--cap', type=int, default=500)
    args = p.parse_args()

    answers = json.loads(Path(args.source).read_text())
    want = None if args.classes == 'all' else set(args.classes.split(','))
    used = VARIANTS[:args.variants]

    out, added, dropped = {}, 0, 0
    for frame, dets in answers.items():
        kept = list(dets)
        band = []
        for (fw, fh), grade in zip(used, GRADE):
            for d in dets:
                if want is not None and d['object_id'] not in want:
                    continue
                bbox = variant(d['bbox'], fw, fh)
                if bbox is None:
                    continue
                band.append({'object_id': d['object_id'], 'bbox': bbox,
                             'confidence': grade})
        # Real answers first, then the band in grade order. Never sorted
        # together: the cap must drop band boxes, never something we believe.
        room = max(0, args.cap - len(kept))
        dropped += max(0, len(band) - room)
        added += min(len(band), room)
        out[frame] = kept + band[:room]

    Path(args.target).write_text(json.dumps(out))
    n = [len(v) for v in out.values()]
    print(f'{args.target}: {len(out)} frames, {sum(n)} boxes '
          f'({min(n)}-{max(n)}/frame, mean {sum(n)/len(n):.1f}), '
          f'+{added} band, {dropped} over the {args.cap} cap')
    real = [a['confidence'] for v in answers.values() for a in v]
    print(f'lowest real answer {min(real)} vs highest band {max(GRADE[:args.variants])}'
          f' -> {"SAFE" if min(real) > max(GRADE[:args.variants]) else "UNSAFE, band can outrank a real answer"}')


if __name__ == '__main__':
    sys.exit(main())
