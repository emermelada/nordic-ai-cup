import base64
import concurrent.futures
import functools
import json
import multiprocessing
import os
import socket
import struct
import subprocess
import sys
import threading
import time
import unittest
from unittest import mock

from dtos import ASRQuestionRequestDto
from pipeline.core import retrieval_response, words_from_transcript
from pipeline.runtime import (
    MAX_FRAME_BYTES,
    Pipeline,
    _receive_frame,
    _send_frame,
)
from utils import validate_response


def transcript():
    words = []
    for index, text in enumerate((
        'The daily dose is 100 mg. You should take it after a meal. '
        'Your heart sounds normal. We will meet again next week. '
        'Please call if you have any new symptoms.'
    ).split()):
        words.append({'word': ' ' + text, 'start': index * 0.3, 'end': (index + 1) * 0.3})
    return {'duration': 20.0, 'seconds': 0.001, 'segments': [{'words': words}]}


class FakeBackend:
    def warmup(self):
        pass

    def transcribe(self, audio):
        if audio == b'hang':
            time.sleep(30)
        if audio == b'crash':
            os._exit(9)
        if not audio or audio == b'error':
            raise ValueError('Invalid audio')
        return transcript()

    def complete(self, words, questions):
        if questions and questions[0] == 'Hang generation?':
            time.sleep(30)
        if questions and questions[0] == 'Slow generation?':
            time.sleep(0.15)
        return json.dumps({'results': [
            {'q': index + 1, 'answer': 'yes' if index % 2 == 0 else 'no',
             'units': [0], 'quote': 'The daily dose is 100 mg.'}
            for index in range(len(questions))
        ]})

    def complete_second(self, words, questions):
        # A different span for the same answers.
        return json.dumps({'results': [
            {'q': index + 1, 'answer': 'yes' if index % 2 == 0 else 'no',
             'units': [0], 'quote': 'You should take it after a meal.'}
            for index in range(len(questions))
        ]})


class SecondaryAnswerBackend(FakeBackend):
    def __init__(self, raw):
        self.raw = raw

    def complete_second(self, words, questions):
        return self.raw


class FailedStartupBackend(FakeBackend):
    def warmup(self):
        raise RuntimeError('Missing local weights')


class HangingStartupBackend(FakeBackend):
    def warmup(self):
        time.sleep(30)


def request(audio=b'ok', questions=None):
    return ASRQuestionRequestDto(
        audio_base64=base64.b64encode(audio).decode(), audio_filename='untrusted.mp3',
        questions=questions or [
            'Should the daily dose be 100 mg?', 'Was a concert discussed?',
            'Should the tablets be taken after a meal?',
        ],
    )


class FrameTests(unittest.TestCase):
    def setUp(self):
        self.sender, self.receiver = socket.socketpair()
        self.addCleanup(self.sender.close)
        self.addCleanup(self.receiver.close)

    def test_round_trip(self):
        value = {'kind': 'result', 'text': 'Milligrammes: µg'}
        _send_frame(self.sender, value, time.monotonic() + 1)
        self.assertEqual(_receive_frame(self.receiver, time.monotonic() + 1), value)

    def test_partial_frame_obeys_deadline(self):
        self.sender.sendall(struct.pack('!I', 100) + b'{')
        started = time.monotonic()
        with self.assertRaises(TimeoutError):
            _receive_frame(self.receiver, started + 0.1)
        self.assertLess(time.monotonic() - started, 1.0)

    def test_blocked_large_send_obeys_deadline(self):
        self.sender.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)
        started = time.monotonic()
        with self.assertRaises(TimeoutError):
            _send_frame(self.sender, {'audio': 'a' * (2 * 1024 * 1024)}, started + 0.15)
        self.assertLess(time.monotonic() - started, 1.0)

    def test_oversized_frame_rejected_before_reading_body(self):
        self.sender.sendall(struct.pack('!I', MAX_FRAME_BYTES + 1))
        with self.assertRaises(ValueError):
            _receive_frame(self.receiver, time.monotonic() + 1)

    def test_truncated_frame_is_eof(self):
        self.sender.sendall(struct.pack('!I', 100) + b'{')
        self.sender.close()
        with self.assertRaises(EOFError):
            _receive_frame(self.receiver, time.monotonic() + 1)

    def test_cancel_interrupts_receive(self):
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(RuntimeError):
            _receive_frame(self.receiver, time.monotonic() + 60, cancel)


class RuntimeTests(unittest.TestCase):
    def make_pipeline(self, **kwargs):
        options = dict(request_timeout=1.5, startup_timeout=10, cleanup_timeout=0.4,
                       backend_factory=FakeBackend)
        options.update(kwargs)
        pipeline = Pipeline(**options)
        self.addCleanup(pipeline.close)
        pipeline.start()
        return pipeline

    def assert_valid(self, response, count):
        validate_response(response, count)
        for answer, start, end in zip(
            response.answers, response.evidence_start, response.evidence_end,
        ):
            if not answer:
                self.assertIsNone(start)
                self.assertIsNone(end)
            elif start is not None:
                self.assertGreaterEqual(start, 0)
                self.assertGreater(end, start)
                self.assertLessEqual(end, 20)

    def wait_for_recovery(self, pipeline, old_pid):
        thread = pipeline._recovery_thread
        self.assertIsNotNone(thread)
        thread.join(timeout=10)
        self.assertFalse(thread.is_alive())
        self.assertTrue(pipeline.ready)
        self.assertNotEqual(pipeline._process.pid, old_pid)
        self.assertNotIn(old_pid, [child.pid for child in multiprocessing.active_children()])

    def test_consecutive_requests_share_worker_and_keep_order(self):
        pipeline = self.make_pipeline()
        pid = pipeline._process.pid
        for count in (3, 10, 2):
            response = pipeline.predict(request(questions=['Dose?'] * count))
            self.assertEqual(response.answers, [i % 2 == 0 for i in range(count)])
            self.assert_valid(response, count)
            self.assertEqual(pipeline._process.pid, pid)

    def test_retained_yes_keeps_primary_evidence_before_boundary_adjustment(self):
        pipeline = self.make_pipeline()
        with mock.patch('pipeline.runtime.adjust_boundaries', side_effect=lambda response, *a, **kw: response):
            actual = pipeline.predict(request(questions=['Should the tablets be taken after a meal?']))
        self.assertEqual(actual.answers, [True])
        self.assertEqual(actual.evidence_start, [0.0])
        self.assertAlmostEqual(actual.evidence_end[0], 1.8)
        self.assert_valid(actual, 1)

    def test_learned_boundaries_stay_near_primary_evidence(self):
        pipeline = self.make_pipeline()
        trace = {}
        actual = pipeline.predict(request(questions=['Should the tablets be taken after a meal?']), trace=trace)
        self.assertEqual(actual.answers, [True])
        self.assertEqual(trace['boundary']['status'], 'completed')
        start, end = actual.evidence_start[0], actual.evidence_end[0]
        primary_start, primary_end = trace['primary']['evidence_start'][0], trace['primary']['evidence_end'][0]
        self.assertLess(max(start, primary_start), min(end, primary_end))
        self.assertLessEqual(abs(start - primary_start), 2.)
        self.assertLessEqual(abs(end - primary_end), 2.)
        self.assert_valid(actual, 1)

    def test_boundary_adjustment_runs_after_veto_with_trace_independent_features(self):
        raw = json.dumps({'results': [{'q': 1, 'answer': 'no'}]})
        pipeline = self.make_pipeline(backend_factory=functools.partial(SecondaryAnswerBackend, raw))
        seen = []

        def adjust(response, words, questions, duration, envelope, primary, secondary, referee, **kwargs):
            self.assertEqual(response.answers, [False, False, True])
            self.assertEqual(primary.answers, [True, False, True])
            seen.append([proposal.model_dump() for proposal in (primary, secondary, referee)])
            result = response.model_copy(deep=True)
            result.evidence_start[2] += .05
            return result

        from pipeline.evidence import refine_evidence
        with mock.patch('pipeline.runtime.adjust_boundaries', side_effect=adjust), \
                mock.patch('pipeline.runtime.refine_evidence', wraps=refine_evidence) as refine:
            trace = {}
            actual = pipeline.predict(request(), trace=trace)
            self.assertEqual(refine.call_count, 3)
            refine.reset_mock()
            self.assertEqual(pipeline.predict(request()), actual)
            self.assertEqual(refine.call_count, 3)
        self.assertEqual(seen[0], seen[1])
        self.assertAlmostEqual(actual.evidence_start[2], trace['primary']['evidence_start'][2] + .05)
        self.assertIsNone(actual.evidence_start[0])
        self.assertEqual(seen[0][1], trace['secondary'])
        self.assert_valid(actual, 3)

    def test_boundary_failure_keeps_completed_response_and_worker(self):
        pipeline = self.make_pipeline()
        pid = pipeline._process.pid
        trace = {}
        with mock.patch('pipeline.runtime.adjust_boundaries', side_effect=RuntimeError('Selector failed')), \
                self.assertLogs('pipeline.runtime', level='ERROR'):
            actual = pipeline.predict(request(), trace=trace)
        self.assertEqual(actual.model_dump(), trace['primary'])
        self.assertEqual(trace['boundary']['status'], 'skipped')
        self.assertTrue(trace['outcome'].startswith('completed'))
        self.assertTrue(pipeline.ready)
        self.assertEqual(pipeline._process.pid, pid)
        self.assertEqual(pipeline.predict(request()).answers, actual.answers)

    def test_skipped_secondary_is_missing_in_boundary_features(self):
        pipeline = self.make_pipeline(backend_factory=functools.partial(SecondaryAnswerBackend, ''))
        with mock.patch('pipeline.runtime.adjust_boundaries', side_effect=lambda response, *a, **kw: response) as adjust:
            actual = pipeline.predict(request())
        self.assertIsNone(adjust.call_args.args[6])
        self.assertEqual(actual.answers, [True, False, True])

    def test_trace_keeps_model_outputs_without_changing_predictions(self):
        pipeline = self.make_pipeline()
        trace = {}
        actual = pipeline.predict(request(), trace=trace)
        saved = json.loads(json.dumps(trace))
        self.assertEqual(trace['transcript'], transcript())
        self.assertEqual(trace['primary']['answers'], actual.answers)
        self.assertEqual(trace['secondary']['answers'], actual.answers)
        self.assertNotEqual(trace['primary']['evidence_start'], trace['secondary']['evidence_start'])
        self.assertIn('raw', trace['result'])
        self.assertIn('second', trace['result'])
        self.assertEqual(trace['result']['request_id'], trace['request_id'])
        self.assertTrue(trace['outcome'].startswith('completed'))
        self.assertGreater(trace['total_seconds'], 0)
        self.assertEqual(pipeline.predict(request()), actual)
        other_trace = {}
        pipeline.predict(request(questions=['Dose?']), trace=other_trace)
        self.assertNotEqual(trace['request_id'], other_trace['request_id'])
        self.assertEqual(trace, saved)

    def test_secondary_veto_requires_a_valid_answer_for_that_question(self):
        raw = json.dumps({'results': [
            {'q': 1, 'answer': 'no'}, {'q': 2, 'answer': 'yes'},
            {'q': 3, 'answer': 'unknown'},
            {'q': 5, 'answer': 'no'}, {'q': 5, 'answer': 'yes'},
        ]})
        pipeline = self.make_pipeline(backend_factory=functools.partial(SecondaryAnswerBackend, raw))
        actual = pipeline.predict(request(questions=['Was a concert discussed?'] * 7))
        self.assertEqual(actual.answers, [False, False, True, False, True, False, True])
        self.assert_valid(actual, 7)
        self.assertEqual(actual.evidence_start, [None, None, 0.0, None, 0.0, None, 0.0])

    def test_unusable_or_skipped_secondary_keeps_primary_answers(self):
        for raw in ('', '{}', 'not JSON'):
            with self.subTest(raw=raw):
                pipeline = self.make_pipeline(backend_factory=functools.partial(SecondaryAnswerBackend, raw))
                actual = pipeline.predict(request(questions=['Was a concert discussed?']))
                self.assertEqual(actual.answers, [True])
                self.assertEqual(actual.evidence_start, [0.0])
                self.assert_valid(actual, 1)
                pipeline.close()

    def test_invalid_audio_returns_floor_and_next_request_works(self):
        pipeline = self.make_pipeline()
        for audio in (b'', b'error'):
            response = pipeline.predict(request(audio))
            self.assertEqual(response.answers, [True] * 3)
            self.assertEqual(response.evidence_start, [None] * 3)
            self.assert_valid(response, 3)
        self.assertEqual(pipeline.predict(request()).answers, [True, False, True])

    def test_timeout_before_transcript_recovers(self):
        pipeline = self.make_pipeline(request_timeout=0.25)
        pid = pipeline._process.pid
        started = time.monotonic()
        response = pipeline.predict(request(b'hang'))
        self.assertLess(time.monotonic() - started, 1.5)
        self.assertEqual(response.answers, [True] * 3)
        self.assertEqual(response.evidence_start, [None] * 3)
        self.wait_for_recovery(pipeline, pid)
        self.assertEqual(pipeline.predict(request()).answers, [True, False, True])

    def test_timeout_after_transcript_keeps_retrieval_response(self):
        pipeline = self.make_pipeline(request_timeout=0.3)
        questions = ['Hang generation?', 'Should the daily dose be 100 mg?',
                     'Was a concert discussed?']
        data = transcript()
        expected = retrieval_response(words_from_transcript(data), questions, data['duration'])
        self.assertNotEqual(expected.answers, [True] * 3)
        pid = pipeline._process.pid
        started = time.monotonic()
        trace = {}
        response = pipeline.predict(request(questions=questions), trace=trace)
        self.assertLess(time.monotonic() - started, 1.5)
        self.assertEqual(response, expected)
        self.assertEqual(trace['transcript'], data)
        self.assertEqual(trace['outcome'], 'fallback')
        self.assertIn('TimeoutError', trace['error'])
        self.assertNotIn('result', trace)
        self.assert_valid(response, 3)
        self.wait_for_recovery(pipeline, pid)

    def test_native_worker_crash_recovers(self):
        pipeline = self.make_pipeline()
        pid = pipeline._process.pid
        response = pipeline.predict(request(b'crash'))
        self.assert_valid(response, 3)
        self.assertEqual(response.evidence_start, [None] * 3)
        self.wait_for_recovery(pipeline, pid)
        self.assertEqual(pipeline.predict(request()).answers, [True, False, True])

    def test_lock_wait_is_bounded_and_does_not_kill_active_worker(self):
        pipeline = self.make_pipeline(request_timeout=0.15)
        pid = pipeline._process.pid
        pipeline._lock.acquire()
        try:
            started = time.monotonic()
            response = pipeline.predict(request())
            self.assertLess(time.monotonic() - started, 0.75)
            self.assertEqual(response.answers, [True] * 3)
        finally:
            pipeline._lock.release()
        self.assertEqual(pipeline._process.pid, pid)
        self.assertTrue(pipeline.ready)

    def test_concurrent_requests_are_serialized(self):
        pipeline = self.make_pipeline()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(pipeline.predict, request(questions=['Slow generation?', 'Q']))
            second = executor.submit(pipeline.predict, request())
            self.assertEqual(first.result(timeout=3).answers, [True, False])
            self.assertEqual(second.result(timeout=3).answers, [True, False, True])

    def test_stale_result_retires_worker(self):
        pipeline = self.make_pipeline()
        pid = pipeline._process.pid
        original_receive = _receive_frame
        calling_thread = threading.current_thread()

        def stale_receive(*args, **kwargs):
            if threading.current_thread() is calling_thread:
                return {'kind': 'result', 'generation': -1, 'request_id': -1, 'raw': '{}'}
            return original_receive(*args, **kwargs)

        with mock.patch('pipeline.runtime._receive_frame', side_effect=stale_receive):
            response = pipeline.predict(request())
        self.assertEqual(response.answers, [True] * 3)
        self.wait_for_recovery(pipeline, pid)

    def test_recovery_requests_return_promptly(self):
        pipeline = self.make_pipeline()
        pipeline._recovering.set()
        self.addCleanup(pipeline._recovering.clear)
        started = time.monotonic()
        response = pipeline.predict(request())
        self.assertLess(time.monotonic() - started, 0.2)
        self.assertEqual(response.evidence_start, [None] * 3)

    def test_startup_failure_is_visible_and_process_reaped(self):
        pipeline = Pipeline(backend_factory=FailedStartupBackend, startup_timeout=10,
                            cleanup_timeout=0.4)
        self.addCleanup(pipeline.close)
        with self.assertRaisesRegex(RuntimeError, 'Missing local weights'):
            pipeline.start()
        self.assertFalse(pipeline.ready)
        self.assertIsNone(pipeline._process)

    def test_startup_hang_is_bounded(self):
        pipeline = Pipeline(backend_factory=HangingStartupBackend, startup_timeout=0.5,
                            cleanup_timeout=0.4)
        self.addCleanup(pipeline.close)
        started = time.monotonic()
        with self.assertRaises(TimeoutError):
            pipeline.start()
        self.assertLess(time.monotonic() - started, 2)
        self.assertIsNone(pipeline._process)

    def test_close_during_startup_interrupts_wait(self):
        pipeline = Pipeline(backend_factory=HangingStartupBackend, startup_timeout=30,
                            cleanup_timeout=0.3)
        self.addCleanup(pipeline.close)
        errors = []

        def start():
            try:
                pipeline.start()
            except RuntimeError as exc:
                errors.append(exc)

        thread = threading.Thread(target=start)
        thread.start()
        time.sleep(0.15)
        started = time.monotonic()
        pipeline.close()
        thread.join(timeout=2)
        self.assertLess(time.monotonic() - started, 2)
        self.assertFalse(thread.is_alive())
        self.assertTrue(errors)
        self.assertIsNone(pipeline._process)

    def test_idle_worker_exits_normally_on_shutdown(self):
        pipeline = self.make_pipeline(cleanup_timeout=1.0)
        process = pipeline._process
        original_close = process.close
        exit_codes = []

        def close():
            exit_codes.append(process.exitcode)
            original_close()

        with mock.patch.object(process, 'close', side_effect=close):
            pipeline.close()
        self.assertEqual(exit_codes, [0])

    def test_closed_pipeline_does_not_restart(self):
        pipeline = self.make_pipeline()
        pipeline.close()
        response = pipeline.predict(request())
        self.assertEqual(response.answers, [True] * 3)
        self.assertIsNone(pipeline._process)

    def test_imports_do_not_load_models_or_launch_workers(self):
        code = (
            'import sys, multiprocessing; import api, example; '
            'assert not multiprocessing.active_children(); '
            'assert "mlx.core" not in sys.modules; '
            'assert "mlx_lm" not in sys.modules; '
            'assert "mlx_whisper" not in sys.modules'
        )
        result = subprocess.run([sys.executable, '-c', code], capture_output=True,
                                text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
