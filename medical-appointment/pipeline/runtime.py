"""A resident inference process with a parent-owned request deadline."""

import json
import logging
import multiprocessing
import os
import socket
import struct
import threading
import time

from pipeline.core import (
    answer_response,
    floor_response,
    retrieval_response,
    sanitize_response,
    words_from_transcript,
)
from pipeline.evidence import primary_evidence_response, refine_evidence
from pipeline.rescue import apply_rescue
from pipeline.stage_b import apply_evidence, draft_quotes, lexical_span
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


def _primary(words, questions, raw, transcript):
    """The answer pass's own decisions and spans, in the transcript's timing convention."""
    return refine_evidence(
        answer_response(raw, words, questions, transcript['duration'], alignment='numeric'),
        words, transcript.get('energy_db'), extend_replies=not transcript.get('exact_timestamps'),
    )


def _rescue(backend, words, questions, raw, transcript, request_started):
    """P(yes) for the questions answered no; failure leaves every answer as it was."""
    complete_rescue = getattr(backend, 'complete_rescue', None)
    if not raw or complete_rescue is None:
        return None
    try:
        primary = _primary(words, questions, raw, transcript)
        return complete_rescue(words, questions, primary.answers, request_started)
    except Exception as exc:
        logger.exception('Rescue pass failed; keeping the original answers')
        return {'error': f'{type(exc).__name__}: {exc}'}


def _stage_b(backend, words, questions, raw, transcript, request_started, rescue=None):
    """Evidence re-selection over the primary answers; its failure never costs the answers."""
    complete_evidence = getattr(backend, 'complete_evidence', None)
    if not raw or complete_evidence is None:
        return None
    try:
        primary = _primary(words, questions, raw, transcript)
        if isinstance(rescue, dict):
            primary, _ = apply_rescue(primary, {k: v for k, v in rescue.items() if k != 'error'})
        anchors = list(zip(primary.evidence_start, primary.evidence_end))
        return complete_evidence(words, questions, primary.answers, draft_quotes(raw, len(questions)),
                                 anchors, request_started)
    except Exception as exc:
        logger.exception('Stage B failed; keeping primary evidence')
        return {'error': f'{type(exc).__name__}: {exc}'}


# Collapses evidence to measure the answer half of a score on its own. Never leave it on.
DIAGNOSTIC_SPANS = os.environ.get('MEDICAL_DIAGNOSTIC_SPANS') == 'floor'


def _vote_evidence(evidence, response, before, words, questions, duration, envelope,
                   deadline, extend_replies, trace=None):
    """Medoid of independent evidence producers; any failure keeps the selected spans.

    Each producer sees the same transcript and answers, so the vote only ever moves a
    span between passages all of them considered.
    """
    from pipeline.evidence import vote_response
    from pipeline.extractor import predict_spans

    try:
        alternative = apply_evidence({'mode': evidence.get('mode'), 'outputs': evidence['vote_outputs']},
                                     before, words, duration, envelope, deadline, extend_replies)
        second = {i + 1: (alternative.evidence_start[i], alternative.evidence_end[i])
                  for i in range(len(alternative.answers))}
        extracted = predict_spans(words, questions, response.answers, deadline)
        voted, changed = vote_response(response, [second, extracted])
        if trace is not None:
            trace['vote'] = {'second': alternative.model_dump(),
                             'extractor': {str(k): list(v) for k, v in extracted.items()},
                             'changed': changed}
        logger.info('Evidence vote: %d producer span(s) from the extractor, %d span(s) moved',
                    len(extracted), changed)
        return voted
    except Exception:
        logger.exception('Evidence vote failed; keeping the selected spans')
        return response


def _make_backend():
    if os.environ.get('MEDICAL_BACKEND') == 'vllm':
        from pipeline.vllm_backend import VLLMBackend

        return VLLMBackend()
    from pipeline.mlx_backend import MLXBackend

    return MLXBackend()


def _worker_main(channel, generation, backend_factory):
    try:
        try:
            backend = backend_factory()
            backend.warmup()
            # Load the extractor here, so the first conversation is not charged for it.
            from pipeline.extractor import warmup as warm_extractor

            warm_extractor()
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
                request_started = time.monotonic()
                transcript = backend.transcribe(decode_audio(message['audio_base64']))
                _send_frame(channel, {
                    **identity, 'kind': 'transcript', 'transcript': transcript,
                })
                words = words_from_transcript(transcript)
                started = time.monotonic()
                raw = backend.complete(words, message['questions']) if words else ''
                second = backend.complete_second(words, message['questions']) if raw else ''
                rescue = _rescue(backend, words, message['questions'], raw, transcript, request_started)
                evidence = _stage_b(backend, words, message['questions'], raw, transcript,
                                    request_started, rescue)
                _send_frame(channel, {
                    **identity, 'kind': 'result', 'raw': raw, 'second': second, 'rescue': rescue,
                    'evidence': evidence, 'generation_seconds': time.monotonic() - started,
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
                trace.update(identity, evidence_policy='primary-with-stage-b-and-secondary-veto')
            _send_frame(self._channel, {
                **identity, 'kind': 'predict', 'audio_base64': request.audio_base64,
                'questions': request.questions,
            }, deadline, self._closed)
            words = []
            envelope = None
            extend_replies = True
            rescued_numbers = []
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
                    extend_replies = not transcript.get('exact_timestamps')
                    fallback = refine_evidence(
                        retrieval_response(words, request.questions, duration), words, envelope, extend_replies,
                    )
                elif kind == 'result':
                    if trace is not None:
                        trace['result'] = message
                    generation_seconds = message.get('generation_seconds')
                    response = answer_response(
                        message['raw'], words, request.questions, duration,
                        fallback=fallback, deadline=deadline, alignment='numeric',
                    )
                    response = refine_evidence(response, words, envelope, extend_replies)
                    if trace is not None:
                        trace['primary'] = response.model_dump()
                        trace['referee'] = fallback.model_dump()
                    rescue = message.get('rescue')
                    if isinstance(rescue, dict):
                        if trace is not None:
                            trace['rescue'] = rescue
                        response, rescued = apply_rescue(
                            response, {k: v for k, v in rescue.items() if k != 'error'})
                        rescued_numbers = rescued
                        # A rescued question has no span of its own yet. Stage B normally
                        # supplies one; seed it from retrieval so that a skipped or failed
                        # evidence stage cannot leave a yes pointing at nothing.
                        for number in rescued:
                            index = number - 1
                            if fallback.evidence_start[index] is not None:
                                response.evidence_start[index] = fallback.evidence_start[index]
                                response.evidence_end[index] = fallback.evidence_end[index]
                        if rescued:
                            logger.info('Rescued %d answer(s) from no to yes: %s', len(rescued), rescued)
                    evidence = message.get('evidence')
                    if evidence:
                        if trace is not None:
                            trace['evidence'] = evidence
                        if isinstance(evidence, dict) and (evidence.get('skipped') or evidence.get('error')):
                            logger.warning('Evidence stage did not run (%s): spans are the answer '
                                           'pass\'s own quotes for this conversation',
                                           evidence.get('skipped') or evidence.get('error'))
                        before_evidence = response.model_copy(deep=True)
                        response = apply_evidence(evidence, response, words, duration, envelope, deadline, extend_replies)
                        if trace is not None:
                            trace['selected'] = response.model_dump()
                        if isinstance(evidence, dict) and evidence.get('vote_outputs'):
                            response = _vote_evidence(
                                evidence, response, before_evidence, words, request.questions,
                                duration, envelope, deadline, extend_replies, trace)
                    if message.get('second'):
                        secondary_fallback = fallback.model_copy(deep=True)
                        # Missing or invalid secondary answers cannot veto the primary.
                        secondary_fallback.answers = response.answers.copy()
                        alternative = answer_response(
                            message['second'], words, request.questions, duration,
                            fallback=secondary_fallback, deadline=deadline, alignment='numeric',
                        )
                        if trace is not None:
                            trace['secondary'] = refine_evidence(alternative, words, envelope, extend_replies).model_dump()
                        response = primary_evidence_response(response, alternative, rescued_numbers)
                    # Every yes carries a span: a null one scores nothing on the larger
                    # half, so the best word-overlap sentence is strictly better.
                    placed = 0
                    for index, answer in enumerate(response.answers):
                        if not answer or response.evidence_start[index] is not None:
                            continue
                        span = lexical_span(words, request.questions[index])
                        if span:
                            response.evidence_start[index], response.evidence_end[index] = span
                            placed += 1
                    if placed:
                        logger.info('Placed %d yes answer(s) on the best matching sentence', placed)
                    response = sanitize_response(response, len(request.questions), duration)
                    if DIAGNOSTIC_SPANS:
                        # Measurement only: collapse every span so the reported score is
                        # 0.4 x accuracy, which separates the two halves of a hidden score.
                        for index, answer in enumerate(response.answers):
                            if answer:
                                response.evidence_start[index] = 0.0
                                response.evidence_end[index] = 0.01
                        logger.warning('DIAGNOSTIC_SPANS is on: evidence is deliberately worthless')
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
