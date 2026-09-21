"""Official box vs tight mask box, per class, on the Helsinki cut-outs.

    python training/measure_box_convention.py        # -> training/box_convention.json

The evaluator's boxes are the projected 3D box of an object: rotor span,
wingtips and height included. `extract_patches.py` stores each cut-out at its
OFFICIAL box with the mask in the alpha channel, so the ratio between the patch
and the tight box around its alpha is the convention's effect, per class.

make_dataset.py labels pasted cut-outs with the tight mask box, so our models
learn tight boxes and are then scored against loose ones. Confirmed on
validation 19 Sep: growing reported boxes by these factors (capped at 1.3) took
a real run from 0.3048 to 0.4618.

A caution when reading the output: a class whose GrabCut mask fell back to the
filled ellipse has a tight box equal to the whole patch and so reports ~1.0.
That is a mask failure, not evidence the object is already loose. `fill` below
is the share of the patch the mask covers; near 1.0 means distrust the ratio.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
patches = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / 'data' / 'patches'
out = {}
for folder in sorted(p for p in patches.iterdir() if p.is_dir()):
    rw, rh, fill = [], [], []
    for path in folder.glob('*.png'):
        patch = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if patch is None or patch.ndim < 3 or patch.shape[2] < 4:
            continue
        alpha = patch[:, :, 3] > 127
        ys, xs = np.nonzero(alpha)
        if len(xs) == 0:
            continue
        rw.append(patch.shape[1] / (xs.max() - xs.min() + 1))
        rh.append(patch.shape[0] / (ys.max() - ys.min() + 1))
        fill.append(alpha.mean())
    if rw:
        w, h = float(np.median(rw)), float(np.median(rh))
        out[folder.name] = {'w': round(w, 3), 'h': round(h, 3),
                            'iso': round((w * h) ** 0.5, 3),
                            'fill': round(float(np.median(fill)), 3), 'n': len(rw)}
(HERE / 'box_convention.json').write_text(json.dumps(out, indent=1))
print(f"{'class':16s} {'n':>3s} {'w':>6s} {'h':>6s} {'iso':>6s} {'fill':>6s}")
for name, v in sorted(out.items(), key=lambda kv: -kv[1]['iso']):
    flag = '  <- mask filled, ratio unreliable' if v['fill'] > 0.9 else ''
    print(f"{name:16s} {v['n']:3d} {v['w']:6.2f} {v['h']:6.2f} {v['iso']:6.3f} {v['fill']:6.2f}{flag}")
