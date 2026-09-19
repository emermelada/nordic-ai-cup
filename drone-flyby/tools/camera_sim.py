"""Simulate a whole flight against the REAL stack, to A/B a camera policy.

    python tools/camera_sim.py --camera full
    python tools/camera_sim.py --camera quad0

Closed loop: the view flyby asks for is the view it gets next frame, rendered
out of the rebuilt 4K scene, through flyby.predict itself -- so the camera
policy, the tracker and the four inference passes are all the served code.

Why it is trustworthy, and where it is not:

* The scene is mosaicked from recorded Level-1 views, upsampled back to 4K. An
  L1 view rendered from it is very nearly the original; an L0 view has been
  through one extra resample, so **L0 is if anything slightly pessimistic here**.
* Only frames with complete coverage are scored, so no policy is charged for a
  hole in the mosaic.
* Scoring is recall on the confirmed objects, never precision -- see
  recall_replay.py for why.
"""
import argparse, base64, collections, json, os, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))

SERVED = [('models/drone-yolo11n-v4.pt', 960), ('models/drone-yolo11s-v6.pt', 1280),
          ('models/drone-yolo11m-v8.pt', 1280), ('models/drone-yolo11m-v8.pt', 2560)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--camera', default='full')
    ap.add_argument('--pattern', help='Explicit sweep, e.g. "0:1920,1080 1:960,540". '
                                      'Injected into flyby.SWEEP; flyby.py is not modified.')
    ap.add_argument('--frames', type=int, default=249)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--dump', help='Write the answers to this JSON, for AP scoring elsewhere.')
    args = ap.parse_args()

    os.environ['DRONE_CAMERA'] = args.camera
    os.environ['DRONE_MODEL'] = str(ROOT / SERVED[0][0])
    os.environ['DRONE_MODEL_ALT'] = ','.join(str(ROOT / p) for p, _ in SERVED[1:])
    os.environ['DRONE_IMGSZ'] = ','.join(str(s) for _, s in SERVED)
    os.environ['DRONE_DEVICE'] = args.device
    os.environ.setdefault('DRONE_BOX_GROW', '1.3')
    os.environ.setdefault('DRONE_BOX_GROW_CAP', '1.3')

    import cv2, numpy as np, flyby
    from dtos import (ALLOWED_RESOLUTION_LEVELS, MAXIMUM_CENTER_DELTA_PIXELS,
                      DroneFlybyPredictRequestDto)
    from utils import center_bounds_for_level, describe_camera_rejection, source_region_for_view
    import importlib.util
    spec = importlib.util.spec_from_file_location('rr', HERE / 'recall_replay.py')
    rr = importlib.util.module_from_spec(spec); spec.loader.exec_module(rr)

    if args.pattern:
        sweep = []
        for step in args.pattern.split():
            lv, _, xy = step.partition(':')
            x, y = xy.split(',')
            sweep.append((int(lv), int(x), int(y)))
        flyby.SWEEP = sweep
        print(f'pattern: {len(sweep)} steps, {sum(1 for p in sweep if p[0] == 0)} at level 0')
    flyby.BOTH_MODELS = 1
    flyby.NEW_TRACK_CONFIDENCE = 0.10
    flyby.load_model()
    if len(flyby._models) != 4:
        raise SystemExit(f'{len(flyby._models)} models loaded, wanted 4')

    coverage = json.loads((ROOT / 'data' / 'scene' / 'coverage.json').read_text())
    scored = {int(f) for f, c in coverage.items() if c['any'] >= 0.999}

    level, centre = 0, (1920, 1080)          # every run opens on the full frame
    answers = collections.defaultdict(list)
    refused = 0
    for frame in range(1, args.frames + 1):
        path = ROOT / 'data' / 'scene' / f'frame_{frame:06d}.png'
        if not path.exists():
            continue
        image = cv2.imread(str(path))
        region = source_region_for_view(level, *centre)
        crop = image[region[1]:region[3], region[0]:region[2]]
        view = cv2.resize(crop, (960, 540), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode('.png', view)
        bounds = {}
        request = {
            'sequence_id': 'sim', 'frame_index': frame - 1,
            'request_id': f'sim:{frame}', 'frame': frame, 'frame_interval_ms': 333,
            'response_timeout_ms': 3333, 'original_width': 3840, 'original_height': 2160,
            'view': {'resolution_level': level, 'center_x': centre[0], 'center_y': centre[1],
                     'view_id': f'sim:{frame}', 'image': base64.b64encode(buf).decode(),
                     'image_media_type': 'image/png', 'width': 960, 'height': 540,
                     'source_region_xyxy': list(region)},
            'camera_constraints': {
                'maximum_center_delta': MAXIMUM_CENTER_DELTA_PIXELS[level],
                'allowed_resolution_levels': list(ALLOWED_RESOLUTION_LEVELS[level]),
                'center_bounds': [
                    dict(zip(('minimum_center_x', 'maximum_center_x',
                              'minimum_center_y', 'maximum_center_y'),
                             center_bounds_for_level(lv)),
                         resolution_level=lv, width=960, height=540)
                    for lv in (0, 1, 2)],
                'full_view_reset_exempt_from_delta': True},
            'camera_command_feedback': None,
        }
        response = flyby.predict(DroneFlybyPredictRequestDto(**request))
        for a in response.annotations:
            box = np.array(a.bbox, float) * [3840, 2160, 3840, 2160]
            answers[frame].append((a.object_id, box, a.confidence))
        want = response.requested_view
        if want is not None:
            why = describe_camera_rejection(level, centre, want.resolution_level,
                                            (want.center_x, want.center_y))
            if why is None:
                level, centre = want.resolution_level, (want.center_x, want.center_y)
            else:
                refused += 1
        if frame % 25 == 0:
            print(f'  frame {frame}', flush=True)

    if args.dump:
        Path(args.dump).write_text(json.dumps(
            {str(f): [[n, [float(v) for v in b], float(c)] for n, b, c in v]
             for f, v in answers.items()}))
        print(f'wrote {args.dump}')
    print(f'\ncamera={args.camera}  frames={len(answers)}  refused camera commands={refused}')
    print(f'scoring only the {len(scored)} fully covered frames')
    rr.score({f: v for f, v in answers.items() if f in scored}, 1, True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
