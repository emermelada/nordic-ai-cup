"""The endpoint the evaluation service calls.

You should not need to change much in here. Put your model in ``example.py``
and leave the transport alone.

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
from flyby import load_model, predict
from utils import validate_response

HOST = '0.0.0.0'
PORT = 9053

logging.basicConfig(level=logging.INFO)
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
    # Load and warm the model before the first frame arrives.
    load_model()


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
    response = predict(request)

    # Fail here, loudly, rather than having the evaluator silently discard the
    # frame. Every rule this checks is a rule the evaluator also enforces.
    validate_response(response)

    if RECORD_DIR:
        # In the background: the frame clock does not wait for the disk.
        _recorder.submit(record, request, response)

    logger.info(
        'frame %s (index %s) L%s at (%s, %s): returned %s detections',
        request.frame,
        request.frame_index,
        request.view.resolution_level,
        request.view.center_x,
        request.view.center_y,
        len(response.annotations),
    )
    return response


@app.get('/api')
def hello():
    return {
        'service': 'drone-flyby-usecase',
        'uptime': '{}'.format(datetime.timedelta(seconds=time.time() - start_time)),
    }


@app.get('/')
def index():
    return "Your endpoint is running!"


if __name__ == '__main__':
    uvicorn.run('api:app', host=HOST, port=PORT)
