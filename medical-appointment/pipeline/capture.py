"""Local request captures, written by the API after sending its response."""

import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import threading
import uuid

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent
CAPTURE_DIR = ROOT / 'runs' / 'request-captures'
MAX_CAPTURE_BYTES = 512 * 1024 * 1024
_lock = threading.Lock()
try:
    SOURCE_HASHES = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in ('api.py', 'example.py', 'pipeline/capture.py', 'pipeline/core.py',
                     'pipeline/evidence.py', 'pipeline/runtime.py', 'pipeline/mlx_backend.py')
    }
except OSError:
    logger.exception('Could not fingerprint the capture build')
    SOURCE_HASHES = {}


def save_capture(request, response, trace):
    if 'request_id' not in trace:
        return
    temporary = None
    try:
        record = {
            'schema_version': 1,
            'captured_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'server_pid': os.getpid(),
            'source_sha256': SOURCE_HASHES,
            'request': request.model_dump(),
            'response': response.model_dump(),
            'trace': trace,
        }
        payload = json.dumps(record, ensure_ascii=False, allow_nan=False).encode('utf-8')
        with _lock:
            CAPTURE_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
            used = sum(path.stat().st_size for path in CAPTURE_DIR.iterdir() if path.is_file())
            if used + len(payload) > MAX_CAPTURE_BYTES:
                logger.warning('Request capture storage limit reached; skipping capture')
                return
            path = CAPTURE_DIR / (uuid.uuid4().hex + '.json')
            pending = path.with_suffix('.tmp')
            descriptor = os.open(pending, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            temporary = pending
            with os.fdopen(descriptor, 'wb') as handle:
                handle.write(payload)
            os.link(temporary, path)
            temporary.unlink()
            temporary = None
            logger.info('Saved request capture %s', path.name)
    except Exception:
        logger.exception('Could not save request capture')
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                logger.warning('Could not remove incomplete request capture %s', temporary.name)
