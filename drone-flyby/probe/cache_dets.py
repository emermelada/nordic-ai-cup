"""Cache raw detections for one recorded run, one entry per (model, imgsz) pass.

    python probe/cache_dets.py --run 2c09583d... --pass models/drone-yolo11n-v4.pt:960 ...

Writes data/det_cache/<run>__<stem>_<imgsz>.pkl mapping the recording's file
stem to (xyxy in VIEW pixels, per-class scores), which is what flyby.detect
would have produced for that view. Passes are cached separately so a new
combination costs nothing.
"""

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--pass', dest='passes', action='append', required=True,
                    help='path/to/weights.pt:imgsz')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--conf', type=float, default=0.01)
    ap.add_argument('--recordings', default=str(ROOT / 'data' / 'recordings'))
    ap.add_argument('--out', default=str(ROOT / 'data' / 'det_cache'))
    a = ap.parse_args()

    import cv2
    run_dir = Path(a.recordings) / a.run
    pngs = sorted(run_dir.glob('*.png'))
    if not pngs:
        raise SystemExit(f'no frames in {run_dir}')
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for spec in a.passes:
        path, _, size = spec.partition(':')
        size = int(size or 960)
        weights = Path(path)
        target = out_dir / f'{a.run}__{weights.stem}_{size}.pkl'
        if target.exists():
            print(f'{target.name}: already cached')
            continue
        os.environ['DRONE_MODEL'] = str(weights)
        os.environ['DRONE_MODEL_ALT'] = ''
        os.environ['DRONE_IMGSZ'] = str(size)
        os.environ['DRONE_DEVICE'] = a.device
        os.environ['DRONE_DET_CONF'] = str(a.conf)
        for module in ('flyby',):
            sys.modules.pop(module, None)
        import flyby
        flyby.load_model()
        cache, started = {}, time.time()
        for i, png in enumerate(pngs):
            image = cv2.imread(str(png))
            xyxy, probs = flyby.raw_detections(image)
            cache[png.stem] = (xyxy.astype('float32'), probs.astype('float32'))
            if i % 25 == 0:
                print(f'  {weights.stem}@{size}: {i}/{len(pngs)} '
                      f'({(time.time() - started) / max(1, i):.2f}s/frame)', flush=True)
        target.write_bytes(pickle.dumps(cache))
        print(f'{target.name}: {len(cache)} frames in {time.time() - started:.0f}s')


if __name__ == '__main__':
    main()
