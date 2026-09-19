"""The endpoint the evaluation service calls.

The detector, object memory and camera policy all live in ``flyby.py``; this
module is transport only. (``example.py`` is the untouched original template.)

The URL you submit is used exactly as you give it, path included, so if you
keep the ``/predict`` route below then submit ``http://<your-host>:9053/predict``
rather than just the host.
"""

import base64
import datetime
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from dtos import DroneFlybyPredictRequestDto, DroneFlybyPredictResponseDto
import flyby
from flyby import load_model, predict
from utils import validate_response

HOST = '0.0.0.0'
PORT = 9053

# With a timestamp: serve.log has to be lined up against tunnel.log to tell a
# network stall from a slow model, and the default format has no time at all.
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()
start_time = time.time()

# Set DRONE_RECORD_DIR to keep every request (image + metadata) and response.
# The rules allow recording the validation sequence; it is our only look at it.
RECORD_DIR = os.environ.get('DRONE_RECORD_DIR')
_recorder = ThreadPoolExecutor(max_workers=1)


def record(request: DroneFlybyPredictRequestDto, response: DroneFlybyPredictResponseDto) -> None:
    folder = Path(RECORD_DIR) / request.sequence_id.replace('/', '_')
    folder.mkdir(parents=True, exist_ok=True)
    stem = f'{request.frame_index:04d}_f{request.frame:04d}'
    (folder / f'{stem}.png').write_bytes(base64.b64decode(request.view.image))
    meta = request.model_dump()
    meta['view'].pop('image')
    meta['response'] = response.model_dump()
    (folder / f'{stem}.json').write_text(json.dumps(meta))


@app.on_event('startup')
def warm_up():
    # Load and warm the model before the first frame arrives. Refuse to start
    # without it: a service that answers 200 with no detections passes every
    # health check and scores zero for a whole attempt.
    if load_model() is None:
        raise RuntimeError(f'No model at {flyby.MODEL_PATH} - check DRONE_MODEL and the mount')
    # Same reasoning for the pair: asking for two models and silently getting one
    # is a working-looking service that quietly serves a different configuration
    # than the one measured. Refuse rather than degrade.
    if flyby.ALT_MODEL_PATHS and len(flyby._models) < 1 + len(flyby.ALT_MODEL_PATHS):
        raise RuntimeError(
            f'Asked for {1 + len(flyby.ALT_MODEL_PATHS)} models, loaded {len(flyby._models)} '
            f'({", ".join(str(p) for p in flyby.ALT_MODEL_PATHS)}) - check DRONE_MODEL_ALT and the mount')


@app.post('/predict', response_model=DroneFlybyPredictResponseDto)
async def predict_endpoint(raw: Request):
    """Answer one frame.

    The body is parsed straight from bytes and the answer serialised directly:
    FastAPI's own request/response validation costs several milliseconds on a
    1.5 MB request, and every millisecond is frames we do not skip.
    """
    try:
        request = DroneFlybyPredictRequestDto.model_validate_json(await raw.body())
    except ValidationError as exc:
        return JSONResponse(status_code=422, content={'detail': exc.errors(include_url=False)})
    response = await run_in_threadpool(answer, request)
    return Response(content=response.model_dump_json(), media_type='application/json')


def answer(request: DroneFlybyPredictRequestDto) -> DroneFlybyPredictResponseDto:
    started = time.perf_counter()
    response = predict(request)

    # Check the evaluator's rules, but never fail the request over them: an
    # exception here returns 500 and loses the frame entirely, which is worse
    # than whatever the check found.
    try:
        validate_response(response)
    except ValueError:
        logger.exception('Invalid response for frame %s, salvaging it', request.frame)
        # Drop only what is actually wrong. Emptying the whole frame throws away
        # every good detection over one bad box, and the frame is scored either
        # way, so a single malformed annotation used to cost a whole frame.
        kept = []
        for annotation in response.annotations:
            probe = DroneFlybyPredictResponseDto(
                request_id=response.request_id, frame=response.frame,
                annotations=[annotation], requested_view=None)
            try:
                validate_response(probe)
            except ValueError:
                continue
            kept.append(annotation)
        response.annotations = kept[:500]
        try:
            validate_response(response)
        except ValueError:
            # Then it was the camera command, not the boxes.
            logger.exception('Camera command invalid on frame %s, dropping it', request.frame)
            response.requested_view = None

    if RECORD_DIR:
        # In the background: the frame clock does not wait for the disk.
        _recorder.submit(record, request, response)

    # The per-frame cost, so a bad run can be blamed on the model or the link
    # from this log alone. Budget is 333 ms; the pair answers in ~25 ms on the M4.
    logger.info(
        'frame %s (index %s) L%s at (%s, %s): returned %s detections in %.0f ms',
        request.frame,
        request.frame_index,
        request.view.resolution_level,
        request.view.center_x,
        request.view.center_y,
        len(response.annotations),
        (time.perf_counter() - started) * 1000,
    )
    return response


@app.get('/api')
def hello():
    return {
        'service': 'drone-flyby-usecase',
        'uptime': '{}'.format(datetime.timedelta(seconds=time.time() - start_time)),
        'model': str(flyby.MODEL_PATH),
        'model_alt': str(flyby.ALT_MODEL_PATH) if flyby.ALT_MODEL_PATH else None,
        'models': [str(flyby.MODEL_PATH)] + [str(p) for p in flyby.ALT_MODEL_PATHS],
        'models_requested': 1 + len(flyby.ALT_MODEL_PATHS),
        # 2 means the pair is really alternating; 1 means one model is serving
        # every frame, whatever DRONE_MODEL_ALT was set to.
        'models_loaded': len(flyby._models),
        # One size per loaded model, in order. The served pair runs v4 at 960
        # and v6 at 1280, so a single number here means the sizes did not apply.
        'imgsz': [flyby.size_for(i) for i in range(max(1, len(flyby._models)))],
        'device': flyby.DEVICE,
        # What the answer policy will actually do, so preflight can confirm the
        # served config rather than the one someone meant to serve.
        'box_grow': flyby.BOX_GROW or None,
        'box_grow_cap': flyby.BOX_GROW_CAP,
        'level0_weight': flyby.LEVEL_WEIGHT[0],
        'unseen_decay': flyby.UNSEEN_DECAY,
        'new_track_confidence': flyby.NEW_TRACK_CONFIDENCE,
        # Everything below is here because an arm that runs with the switch off
        # looks exactly like an arm that ran with it on. DRONE_INSPECT in
        # particular had no way to be confirmed from outside the process at all.
        'inspect': flyby.INSPECT,
        'det_conf': flyby.DETECTION_CONFIDENCE,
        'both_models': bool(flyby.BOTH_MODELS),
        'floor_zero': flyby.FLOOR_ZERO,
        'floor_size_tol': flyby.FLOOR_SIZE_TOL if flyby.FLOOR_ZERO else None,
        'floor_all': flyby.FLOOR_ALL_CLASSES or None,
        'miss_penalty': flyby.MISS_PENALTY,
        'model_loaded': flyby._model is not None,
        'camera': flyby.CAMERA,
        'recording': bool(RECORD_DIR),
    }


@app.get('/')
def index():
    return "Your endpoint is running!"


if __name__ == '__main__':
    uvicorn.run('api:app', host=HOST, port=PORT)
