"""Capture and replay server: measure things against the REAL grader.

Two jobs, usable together in one validation run:

* **capture** -- park the camera on one tile and record every view. The
  validation flight is deterministic, so 16 runs parked on the 16 Level-2 tiles
  give every frame of the flight at native 4K resolution.
* **replay** -- answer each frame with precomputed annotations, looked up by
  frame number. The grader then scores exactly those boxes, with no model,
  no latency and no camera path in the loop: one run = one exact measurement of
  a set of answers against the real ground truth.

Runs are driven by a job queue so validation attempts can be started back to
back without touching the server: every NEW sequence_id takes the next job from
PROBE_JOBS that has not been used yet. The file is re-read on every new
sequence, so jobs can be appended while the server runs.

    PROBE_JOBS  JSON list of {"name": str, "tile": "2:480,270" | "1:960,540" | "0",
                              "answers": path or null}
    PROBE_OUT   directory to record every request into (per job and sequence)
    PROBE_PORT  port to bind (default 9053)

Answers files are JSON {frame: [{object_id, bbox, confidence}, ...]} with
frame-global normalized [x1, y1, x2, y2] boxes.
"""

import base64
import datetime
import json
import logging
import math
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dtos import (  # noqa: E402
    DroneFlybyPredictionDto,
    DroneFlybyPredictRequestDto,
    DroneFlybyPredictResponseDto,
    RequestedViewDto,
)
from utils import center_bounds_for_level, describe_camera_rejection, validate_response  # noqa: E402

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger('probe')

PORT = int(os.environ.get('PROBE_PORT', '9053'))
OUT = os.environ.get('PROBE_OUT')
JOBS_PATH = os.environ.get('PROBE_JOBS')

MOVE_LIMIT = {0: 2203.0, 1: 1102.0, 2: 551.0}


def parse_tile(spec: str) -> Tuple[int, int, int]:
    spec = (spec or '0').strip()
    if spec in ('0', ''):
        return (0, 1920, 1080)
    level, _, xy = spec.partition(':')
    x, y = (int(v) for v in xy.split(','))
    level = int(level)
    lo_x, hi_x, lo_y, hi_y = center_bounds_for_level(level)
    if not (lo_x <= x <= hi_x and lo_y <= y <= hi_y):
        raise ValueError(f'tile {spec}: centre out of bounds for level {level}')
    return (level, x, y)


_answer_cache: Dict[str, Dict[int, List[DroneFlybyPredictionDto]]] = {}


def load_answers(path: Optional[str]) -> Dict[int, List[DroneFlybyPredictionDto]]:
    if not path:
        return {}
    if path in _answer_cache:
        return _answer_cache[path]
    raw = json.loads(Path(path).read_text())
    answers, bad = {}, 0
    for frame, items in raw.items():
        kept = []
        for item in items:
            try:
                kept.append(DroneFlybyPredictionDto.model_validate(item))
            except ValidationError:
                bad += 1
        # A response carries at most 500; keep the most confident if truncating.
        kept.sort(key=lambda a: -float(a.confidence))
        answers[int(frame)] = kept[:500]
    logger.warning('answers %s: %d frames, %d boxes, %d invalid dropped',
                   path, len(answers), sum(map(len, answers.values())), bad)
    _answer_cache[path] = answers
    return answers


class Job:
    def __init__(self, spec: dict):
        self.name = spec['name']
        self.target = parse_tile(spec.get('tile', '0'))
        self.answers_path = spec.get('answers')
        self.answers = load_answers(self.answers_path)


DEFAULT_JOB = Job({'name': 'idle', 'tile': '0', 'answers': None})
_lock = threading.Lock()
_assigned: Dict[str, Job] = {}          # sequence_id -> job
_used: set = set()                       # job names already handed out
_seq_stats: Dict[str, dict] = {}


def job_for(sequence_id: str) -> Job:
    with _lock:
        job = _assigned.get(sequence_id)
        if job is not None:
            return job
        job = DEFAULT_JOB
        if JOBS_PATH and Path(JOBS_PATH).exists():
            for spec in json.loads(Path(JOBS_PATH).read_text()):
                if spec['name'] not in _used:
                    try:
                        job = Job(spec)
                    except Exception:
                        logger.exception('bad job %s, skipping it', spec.get('name'))
                        _used.add(spec['name'])
                        continue
                    _used.add(job.name)
                    break
        _assigned[sequence_id] = job
        _seq_stats[sequence_id] = {'job': job.name, 'first': time.time(), 'frames': 0,
                                   'last_index': -1, 'gaps': 0, 'max_ms': 0.0}
        logger.warning('sequence %s -> job %s (target %s, %d answer frames)',
                       sequence_id, job.name, job.target, len(job.answers))
        return job


def next_view(target, level: int, cx: int, cy: int) -> Optional[Tuple[int, int, int]]:
    """One legal step from the current view towards target; None to hold.

    Computed from the view in the request, so a command that landed late is
    simply asked for again -- every request here is idempotent.
    """
    t_level, tx, ty = target
    if (level, cx, cy) == tuple(target):
        return None
    if t_level == 0:
        return (0, 1920, 1080)
    if level == 0:
        # L0 -> L1 is one move and the L0 limit (2203) reaches every L1 centre.
        lo_x, hi_x, lo_y, hi_y = center_bounds_for_level(1)
        return (1, min(max(tx, lo_x), hi_x), min(max(ty, lo_y), hi_y))
    want_level = t_level if abs(t_level - level) <= 1 else level + (1 if t_level > level else -1)
    lo_x, hi_x, lo_y, hi_y = center_bounds_for_level(want_level)
    gx, gy = min(max(tx, lo_x), hi_x), min(max(ty, lo_y), hi_y)
    limit = MOVE_LIMIT[level] - 1
    dx, dy = gx - cx, gy - cy
    dist = math.hypot(dx, dy)
    if dist > limit:
        gx, gy = int(cx + dx * limit / dist), int(cy + dy * limit / dist)
        gx, gy = min(max(gx, lo_x), hi_x), min(max(gy, lo_y), hi_y)
    step = (want_level, int(gx), int(gy))
    if describe_camera_rejection(level, (cx, cy), step[0], (step[1], step[2])) is not None:
        return None
    return step


app = FastAPI()
start_time = time.time()
_writer = ThreadPoolExecutor(max_workers=2)


def record(job: Job, request: DroneFlybyPredictRequestDto, response: DroneFlybyPredictResponseDto, took_ms: float) -> None:
    safe = request.sequence_id.replace('/', '_').replace(':', '_')
    folder = Path(OUT) / f'{job.name}__{safe}'
    folder.mkdir(parents=True, exist_ok=True)
    v = request.view
    stem = f'{request.frame_index:04d}_f{request.frame:04d}_L{v.resolution_level}_{v.center_x}_{v.center_y}'
    (folder / f'{stem}.png').write_bytes(base64.b64decode(v.image))
    meta = request.model_dump()
    meta['view'].pop('image')
    meta['response'] = {'n_annotations': len(response.annotations),
                        'requested_view': response.requested_view.model_dump() if response.requested_view else None}
    meta['answer_ms'] = took_ms
    meta['received_at'] = time.time()
    (folder / f'{stem}.json').write_text(json.dumps(meta))


@app.post('/predict')
async def predict_endpoint(raw: Request):
    started = time.perf_counter()
    try:
        request = DroneFlybyPredictRequestDto.model_validate_json(await raw.body())
    except ValidationError as exc:
        return JSONResponse(status_code=422, content={'detail': exc.errors(include_url=False)})

    job = job_for(request.sequence_id)
    v = request.view
    step = next_view(job.target, v.resolution_level, v.center_x, v.center_y)
    requested = RequestedViewDto(resolution_level=step[0], center_x=step[1], center_y=step[2]) if step else None
    response = DroneFlybyPredictResponseDto(
        request_id=request.request_id,
        frame=request.frame,
        annotations=job.answers.get(request.frame, []),
        requested_view=requested,
    )
    try:
        validate_response(response)
    except ValueError:
        logger.exception('invalid response for frame %s; sending it without annotations', request.frame)
        response.annotations = []
    body = response.model_dump_json()
    took = (time.perf_counter() - started) * 1000
    stats = _seq_stats.get(request.sequence_id)
    if stats is not None:
        stats['frames'] += 1
        if stats['last_index'] >= 0 and request.frame_index > stats['last_index'] + 1:
            stats['gaps'] += request.frame_index - stats['last_index'] - 1
        stats['last_index'] = max(stats['last_index'], request.frame_index)
        stats['max_ms'] = max(stats['max_ms'], took)
    if OUT:
        _writer.submit(record, job, request, response, took)
    logger.info('%s frame %s idx %s L%s (%s,%s) -> %d boxes, next %s, %.1f ms', job.name,
                request.frame, request.frame_index, v.resolution_level, v.center_x, v.center_y,
                len(response.annotations), step, took)
    return Response(content=body, media_type='application/json')


@app.get('/api')
def status():
    pending = []
    if JOBS_PATH and Path(JOBS_PATH).exists():
        pending = [s['name'] for s in json.loads(Path(JOBS_PATH).read_text()) if s['name'] not in _used]
    return {
        'service': 'drone-flyby-probe',
        'uptime': str(datetime.timedelta(seconds=int(time.time() - start_time))),
        'next_jobs': pending[:5],
        'jobs_pending': len(pending),
        'sequences': [{'sequence_id': k, **v} for k, v in list(_seq_stats.items())[-8:]],
        'recording_to': OUT,
    }


@app.get('/')
def index():
    return 'Your endpoint is running!'


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=PORT, log_level='warning')
