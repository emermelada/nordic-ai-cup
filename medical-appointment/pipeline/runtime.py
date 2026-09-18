"""A resident inference process with a parent-owned request deadline."""

import json
import logging
import multiprocessing
import socket
import struct
import threading
import time

from pipeline.boundary import adjust_boundaries
from pipeline.core import (
    answer_response,
    floor_response,
    retrieval_response,
    sanitize_response,
    words_from_transcript,
)
from pipeline.evidence import primary_evidence_response, refine_evidence
from utils import decode_audio, validate_response

logger = logging.getLogger(__name__)
MAX_FRAME_BYTES = 32 * 1024 * 1024


def _socket_timeout(channel, deadline, cancel):
    if cancel is not None and cancel.is_set():
        raise RuntimeError('Pipeline is stopping')
    remaining = None if deadline is None else deadline - time.monotonic()
    if remaining is not None and remaining <= 0:
        raise TimeoutError('Inference deadline exceeded')
    if cancel is not None:
        remaining = min(remaining, 0.2) if remaining is not None else 0.2
    channel.settimeout(remaining)


def _send_frame(channel, message, deadline=None, cancel=None):
    payload = json.dumps(message, allow_nan=False).encode('utf-8')
    if len(payload) > MAX_FRAME_BYTES:
        raise ValueError('Worker message is too large')
    remaining = memoryview(struct.pack('!I', len(payload)) + payload)
    while remaining:
        _socket_timeout(channel, deadline, cancel)
        try:
            sent = channel.send(remaining)
        except socket.timeout:
            continue
        if sent == 0:
            raise EOFError('Worker channel closed while sending')
        remaining = remaining[sent:]


def _receive_bytes(channel, size, deadline, cancel):
    result = bytearray()
    while len(result) < size:
        _socket_timeout(channel, deadline, cancel)
        try:
            chunk = channel.recv(min(size - len(result), 64 * 1024))
        except socket.timeout:
            continue
        if not chunk:
            raise EOFError('Worker channel closed while receiving')
        result.extend(chunk)
    return result


def _receive_frame(channel, deadline=None, cancel=None):
    header = _receive_bytes(channel, 4, deadline, cancel)
    size = struct.unpack('!I', header)[0]
    if not 0 < size <= MAX_FRAME_BYTES:
        raise ValueError('Invalid worker frame length')
    message = json.loads(_receive_bytes(channel, size, deadline, cancel))
    if not isinstance(message, dict):
        raise ValueError('Invalid worker message')
    return message


def _make_backend():
    from pipeline.mlx_backend import MLXBackend

    return MLXBackend()


def _worker_main(channel, generation, backend_factory):
    try:
        try:
            backend = backend_factory()
            backend.warmup()
            _send_frame(channel, {'kind': 'ready', 'generation': generation})
        except Exception as exc:
            logger.exception('Inference worker startup failed')
            _send_frame(channel, {
                'kind': 'startup_error', 'generation': generation,
                'error': f'{type(exc).__name__}: {exc}',
            })
            return

        while True:
            message = _receive_frame(channel)
            if message.get('kind') != 'predict' or message.get('generation') != generation:
                raise ValueError('Unexpected worker request')
            identity = {'generation': generation, 'request_id': message['request_id']}
            try:
                transcript = backend.transcribe(decode_audio(message['audio_base64']))
                _send_frame(channel, {
                    **identity, 'kind': 'transcript', 'transcript': transcript,
                })
                words = words_from_transcript(transcript)
                started = time.monotonic()
                raw = backend.complete(words, message['questions']) if words else ''
                second = backend.complete_second(words, message['questions']) if raw else ''
                _send_frame(channel, {
                    **identity, 'kind': 'result', 'raw': raw, 'second': second,
                    'generation_seconds': time.monotonic() - started,
                })
            except Exception as exc:
                logger.exception('Inference request failed')
                _send_frame(channel, {
                    **identity, 'kind': 'error',
                    'error': f'{type(exc).__name__}: {exc}',
                })
    except (EOFError, OSError):
        pass
    finally:
        channel.close()


class Pipeline:
    def __init__(
        self, *, request_timeout=52.0, startup_timeout=120.0,
        cleanup_timeout=2.0, backend_factory=None,
    ):
        if min(request_timeout, startup_timeout, cleanup_timeout) <= 0:
            raise ValueError('Timeouts must be positive')
        self.request_timeout = request_timeout
        self.startup_timeout = startup_timeout
        self.cleanup_timeout = cleanup_timeout
        self._backend_factory = backend_factory or _make_backend
        self._context = multiprocessing.get_context('spawn')
        self._lock = threading.Lock()
        self._closed = threading.Event()
        self._recovering = threading.Event()
        self._recovery_thread = None
        self._process = None
        self._channel = None
        self._ready = False
        self._generation = 0
        self._request_id = 0

    @property
    def ready(self):
        return self._ready and not self._closed.is_set() and not self._recovering.is_set()

    def _start_locked(self):
        if self._closed.is_set():
            raise RuntimeError('Pipeline has been closed')
        if self._process is not None:
            raise RuntimeError('Previous inference process has not been reaped')
        deadline = time.monotonic() + self.startup_timeout
        parent, child = socket.socketpair()
        self._generation += 1
        process = self._context.Process(
            target=_worker_main,
            args=(child, self._generation, self._backend_factory),
            name='medical-inference', daemon=True,
        )
        try:
            process.start()
        except Exception:
            parent.close()
            child.close()
            raise
        child.close()
        self._process, self._channel = process, parent
        message = _receive_frame(parent, deadline, self._closed)
        if message.get('generation') != self._generation or message.get('kind') != 'ready':
            raise RuntimeError(message.get('error', 'Worker did not become ready'))
        self._ready = True
        logger.info('Inference worker %s loaded and warmed', process.pid)

    def start(self):
        if not self._lock.acquire(timeout=self.startup_timeout):
            raise TimeoutError('Pipeline startup lock timed out')
        try:
            if self.ready and self._process.is_alive():
                return
            if not self._retire_locked(time.monotonic() + self.cleanup_timeout):
                raise RuntimeError('Previous inference worker is still alive')
            try:
                self._start_locked()
            except Exception:
                self._retire_locked(time.monotonic() + self.cleanup_timeout)
                raise
        finally:
            self._lock.release()

    def _retire_locked(self, deadline, graceful=False):
        self._ready = False
        if self._channel is not None:
            self._channel.close()
            self._channel = None
        process = self._process
        if process is None:
            return True
        if graceful:
            # An idle worker can release library-owned semaphores after channel EOF.
            process.join(timeout=max(0.0, deadline - time.monotonic()) / 2)
        if process.is_alive():
            process.terminate()
        remaining = max(0.0, deadline - time.monotonic())
        process.join(timeout=remaining / 2)
        if process.is_alive():
            process.kill()
            process.join(timeout=max(0.0, deadline - time.monotonic()))
        if process.is_alive():
            logger.error('Worker %s has not exited; no replacement will be started', process.pid)
            return False
        process.close()
        self._process = None
        return True

    def _schedule_recovery_locked(self):
        if self._closed.is_set() or self._recovering.is_set():
            return
        self._recovering.set()
        self._recovery_thread = threading.Thread(
            target=self._recover, name='medical-recovery', daemon=True,
        )
        self._recovery_thread.start()

    def _recover(self):
        try:
            with self._lock:
                if not self._retire_locked(time.monotonic() + self.cleanup_timeout):
                    return
                if self._closed.is_set():
                    return
                try:
                    self._start_locked()
                except Exception:
                    logger.exception('Inference worker recovery failed')
                    self._retire_locked(time.monotonic() + self.cleanup_timeout)
        finally:
            self._recovering.clear()

    def close(self):
        self._closed.set()
        # Startup IPC checks this event, so shutdown cannot wait through a full warm-up.
        if self._lock.acquire(timeout=self.cleanup_timeout + 1.0):
            try:
                self._retire_locked(time.monotonic() + self.cleanup_timeout, graceful=True)
            finally:
                self._lock.release()
        else:
            logger.error('Pipeline shutdown could not acquire the worker lock')
        thread = self._recovery_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=self.cleanup_timeout)

    def predict(self, request, *, trace=None):
        started = time.monotonic()
        deadline = started + self.request_timeout
        fallback = floor_response(len(request.questions))
        if self._closed.is_set() or self._recovering.is_set():
            return fallback
        if not self._lock.acquire(timeout=max(0.0, deadline - time.monotonic())):
            return fallback
        duration = None
        asr_seconds = generation_seconds = None
        outcome = 'fallback'
        try:
            if self._closed.is_set() or self._recovering.is_set():
                return fallback
            if not self.ready or self._process is None or not self._process.is_alive():
                self._schedule_recovery_locked()
                return fallback
            if len(request.audio_base64) + sum(map(len, request.questions)) > MAX_FRAME_BYTES // 2:
                logger.warning('Input exceeds inference message limit')
                return fallback
            self._request_id += 1
            identity = {'generation': self._generation, 'request_id': self._request_id}
            if trace is not None:
                trace.update(identity, evidence_policy='primary-with-secondary-veto-and-learned-boundaries')
            _send_frame(self._channel, {
                **identity, 'kind': 'predict', 'audio_base64': request.audio_base64,
                'questions': request.questions,
            }, deadline, self._closed)
            words = []
            envelope = None
            while True:
                message = _receive_frame(self._channel, deadline, self._closed)
                if any(message.get(key) != value for key, value in identity.items()):
                    raise ValueError('Stale or mismatched worker result')
                kind = message.get('kind')
                if kind == 'transcript':
                    transcript = message['transcript']
                    if trace is not None:
                        trace['transcript'] = transcript
                    words = words_from_transcript(transcript)
                    if len(words) > 10000:
                        raise ValueError('Transcript exceeds inference limit')
                    duration = transcript['duration']
                    asr_seconds = transcript.get('seconds')
                    envelope = transcript.get('energy_db')
                    fallback = refine_evidence(
                        retrieval_response(words, request.questions, duration), words, envelope,
                    )
                elif kind == 'result':
                    if trace is not None:
                        trace['result'] = message
                    generation_seconds = message.get('generation_seconds')
                    response = answer_response(
                        message['raw'], words, request.questions, duration,
                        fallback=fallback, deadline=deadline, alignment='numeric',
                    )
                    response = refine_evidence(response, words, envelope)
                    if trace is not None:
                        trace['primary'] = response.model_dump()
                        trace['referee'] = fallback.model_dump()
                    primary, alternative = response, None
                    if message.get('second'):
                        secondary_fallback = fallback.model_copy(deep=True)
                        # Missing or invalid secondary answers cannot veto the primary.
                        secondary_fallback.answers = response.answers.copy()
                        alternative = answer_response(
                            message['second'], words, request.questions, duration,
                            fallback=secondary_fallback, deadline=deadline, alignment='numeric',
                        )
                        response = primary_evidence_response(response, alternative)
                    try:
                        if alternative is not None:
                            alternative = refine_evidence(alternative, words, envelope)
                            if trace is not None:
                                trace['secondary'] = alternative.model_dump()
                        response = adjust_boundaries(
                            response, words, request.questions, duration, envelope,
                            primary, alternative, fallback, deadline=deadline, trace=trace,
                        )
                    except Exception as exc:
                        logger.exception('Keeping primary evidence after boundary adjustment failure')
                        if trace is not None:
                            trace['boundary'] = {'status': 'skipped', 'reason': f'{type(exc).__name__}: {exc}'}
                    response = sanitize_response(response, len(request.questions), duration)
                    validate_response(response, len(request.questions))
                    # Both platform sets are exactly half yes; the rate is a label-free sanity check.
                    outcome = f'completed, yes={sum(response.answers)}/{len(response.answers)}'
                    return response
                elif kind == 'error':
                    if trace is not None:
                        trace['error'] = message.get('error')
                    logger.warning('Worker request error: %s', message.get('error'))
                    return fallback
                else:
                    raise ValueError('Unexpected worker response')
        except Exception as exc:
            if trace is not None:
                trace['error'] = f'{type(exc).__name__}: {exc}'
            logger.exception('Falling back after inference failure')
            self._retire_locked(time.monotonic() + self.cleanup_timeout)
            self._schedule_recovery_locked()
            return fallback
        finally:
            self._lock.release()
            if trace is not None:
                trace.update(outcome=outcome, total_seconds=time.monotonic() - started)
            logger.info(
                '%s: %.2fs total, ASR=%s, generation=%s, %s',
                request.audio_filename, time.monotonic() - started,
                asr_seconds, generation_seconds, outcome,
            )
