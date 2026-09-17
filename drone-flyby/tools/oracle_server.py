"""Serve flyby.py with a perfect detector, to test memory and camera alone.

    python tools/oracle_server.py            # then: python local_evaluator.py

The fake detector returns the true Helsinki boxes that lie inside the current
view and are at least MIN_VIEW_PIXELS big there, so the score it reaches is
the ceiling of the tracking and camera logic for a detector that good.
"""
import sys
from pathlib import Path

import numpy as np
import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import flyby  # noqa: E402
from api import app  # noqa: E402
from utils import load_annotations  # noqa: E402

MIN_VIEW_PIXELS = {0: 8, 1: 6, 2: 4}
_current = {}


def fake_detect(image, region):
    frame = _current['frame']
    level = _current['level']
    rx1, ry1, rx2, ry2 = region
    scale = image.shape[1] / (rx2 - rx1)
    found = []
    for a in load_annotations(frame):
        x1, y1, x2, y2 = a['bbox']
        if x1 < rx1 or y1 < ry1 or x2 > rx2 or y2 > ry2:
            continue
        if min(x2 - x1, y2 - y1) * scale < MIN_VIEW_PIXELS[level]:
            continue
        found.append((a['object_id'], 0.9, np.array(a['bbox'], float)))
    return found


real_predict = flyby.predict


def predict(request):
    _current['frame'] = request.frame
    _current['level'] = request.view.resolution_level
    return real_predict(request)


flyby.detect = fake_detect
flyby.load_model = lambda: None
import api  # noqa: E402
api.predict = predict

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=9053)
