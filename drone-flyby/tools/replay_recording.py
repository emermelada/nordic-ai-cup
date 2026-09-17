"""Replay a recorded validation run through flyby.py, offline.

    python tools/replay_recording.py data/recordings/<sequence_id> --model ~/models/drone-yolo11n-v2.pt
    python tools/replay_recording.py ... --sheet out.jpg     # draw what was reported

The recording has no ground truth, so this cannot score. It shows what a model
and a set of thresholds would have answered on the real validation ground:
how many objects end up remembered, how many are reported per frame, and, with
--sheet, the boxes drawn onto a few of the recorded views.

The camera cannot be replayed (each recorded view is where the camera really
was), so the policy's requested views are ignored.
"""

import argparse
import base64
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('recording', type=Path)
    parser.add_argument('--model', type=Path, help='Sets DRONE_MODEL.')
    parser.add_argument('--det-conf', type=float, help='Sets DRONE_DET_CONF.')
    parser.add_argument('--track-conf', type=float, help='Sets DRONE_TRACK_CONF.')
    parser.add_argument('--sheet', type=Path, help='Write a contact sheet of annotated views here.')
    parser.add_argument('--sheet-views', type=int, default=6)
    parser.add_argument('--min-draw-conf', type=float, default=0.2)
    args = parser.parse_args()

    # flyby reads its settings at import time.
    if args.model:
        os.environ['DRONE_MODEL'] = str(args.model.expanduser())
    if args.det_conf is not None:
        os.environ['DRONE_DET_CONF'] = str(args.det_conf)
    if args.track_conf is not None:
        os.environ['DRONE_TRACK_CONF'] = str(args.track_conf)

    import flyby
    from dtos import DroneFlybyPredictRequestDto

    metas = sorted(args.recording.glob('*.json'))
    if not metas:
        raise SystemExit(f'No recorded frames in {args.recording}')
    flyby.load_model()

    counts, per_class, drawn = [], {}, []
    sheet_every = max(1, len(metas) // args.sheet_views)
    for number, meta_path in enumerate(metas):
        meta = json.loads(meta_path.read_text())
        meta.pop('response', None)
        png = meta_path.with_suffix('.png').read_bytes()
        meta['view']['image'] = base64.b64encode(png).decode()
        request = DroneFlybyPredictRequestDto(**meta)
        response = flyby.predict(request)
        state = flyby._sequences[request.sequence_id]
        counts.append((len(state.tracks), len(response.annotations)))
        for annotation in response.annotations:
            if annotation.confidence >= args.min_draw_conf:
                per_class[annotation.object_id] = per_class.get(annotation.object_id, 0) + 1

        if args.sheet and number % sheet_every == 0 and len(drawn) < args.sheet_views:
            image = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
            rx1, ry1, rx2, ry2 = request.view.source_region_xyxy
            scale = image.shape[1] / (rx2 - rx1)
            for annotation in response.annotations:
                if annotation.confidence < args.min_draw_conf:
                    continue
                x1, y1, x2, y2 = annotation.bbox
                p1 = (int((x1 * 3840 - rx1) * scale), int((y1 * 2160 - ry1) * scale))
                p2 = (int((x2 * 3840 - rx1) * scale), int((y2 * 2160 - ry1) * scale))
                cv2.rectangle(image, p1, p2, (0, 0, 255), 1)
                cv2.putText(image, f'{annotation.object_id} {annotation.confidence:.2f}',
                            (p1[0], p1[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
            cv2.putText(image, f'{meta_path.stem} L{request.view.resolution_level} tracks={len(state.tracks)}',
                        (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            drawn.append(image)

    tracks = np.array([c[0] for c in counts])
    reported = np.array([c[1] for c in counts])
    print(f'frames {len(counts)}')
    print(f'remembered objects: mean {tracks.mean():.1f}  max {tracks.max()}  last {tracks[-1]}')
    print(f'reported per frame: mean {reported.mean():.1f}  max {reported.max()}')
    print(f'reported with confidence >= {args.min_draw_conf}, by class:')
    for name, count in sorted(per_class.items(), key=lambda item: -item[1]):
        print(f'  {name:16s} {count}')

    if args.sheet and drawn:
        while len(drawn) % 2:
            drawn.append(np.zeros_like(drawn[0]))
        sheet = np.vstack([np.hstack(drawn[i:i + 2]) for i in range(0, len(drawn), 2)])
        cv2.imwrite(str(args.sheet), sheet)
        print(f'sheet: {args.sheet}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
