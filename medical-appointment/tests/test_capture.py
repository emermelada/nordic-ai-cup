import asyncio
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import api
import example
from pipeline import capture
from tests.test_core import response
from tests.test_runtime import request, transcript


class CaptureTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name) / 'captures'
        patcher = mock.patch.object(capture, 'CAPTURE_DIR', self.directory)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.request = request()
        self.response = response([True, False, True], [0.0, None, 0.0], [1.8, None, 1.8])
        self.trace = {'request_id': 1, 'transcript': transcript(),
                      'result': {'raw': '{}', 'second': '{}'}}

    def save(self):
        capture.save_capture(self.request, self.response, self.trace)

    def test_round_trip_is_private_unique_and_ignores_filename_for_paths(self):
        self.request.audio_filename = '../../outside.mp3'
        self.save()
        self.save()
        paths = list(self.directory.glob('*.json'))
        self.assertEqual(len(paths), 2)
        self.assertEqual(self.directory.stat().st_mode & 0o777, 0o700)
        for path in paths:
            record = json.loads(path.read_text())
            self.assertEqual(record['request'], self.request.model_dump())
            self.assertEqual(record['response'], self.response.model_dump())
            self.assertEqual(record['trace'], self.trace)
            self.assertEqual(record['source_sha256'], capture.SOURCE_HASHES)
            self.assertEqual(record['server_pid'], os.getpid())
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        self.assertFalse((self.directory.parent / 'outside.mp3').exists())
        self.assertEqual(list(self.directory.glob('*.tmp')), [])

    def test_fingerprints_include_boundary_code_and_weights(self):
        for name in ('pipeline/boundary.py', 'pipeline/boundary_model.json'):
            self.assertEqual(capture.SOURCE_HASHES[name], hashlib.sha256((capture.ROOT / name).read_bytes()).hexdigest())

    def test_requests_not_sent_to_worker_are_not_captured(self):
        capture.save_capture(self.request, self.response, {})
        self.assertFalse(self.directory.exists())

    def test_storage_cap_is_respected_by_concurrent_writers(self):
        self.save()
        size = next(self.directory.glob('*.json')).stat().st_size
        with mock.patch.object(capture, 'MAX_CAPTURE_BYTES', size * 3), \
                self.assertLogs('pipeline.capture', level='WARNING'), \
                concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(lambda _: self.save(), range(7)))
        paths = list(self.directory.iterdir())
        self.assertEqual(len(paths), 3)
        self.assertLessEqual(sum(path.stat().st_size for path in paths), size * 3)

    def test_storage_failure_does_not_escape(self):
        with mock.patch('pipeline.capture.os.open', side_effect=OSError('Disk full')), \
                self.assertLogs('pipeline.capture', level='ERROR'):
            self.save()
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_capture_is_complete_before_json_name_becomes_visible(self):
        link = os.link

        def publish(temporary, path):
            self.assertEqual(list(self.directory.glob('*.json')), [])
            self.assertEqual(json.loads(temporary.read_text())['trace'], self.trace)
            link(temporary, path)

        with mock.patch('pipeline.capture.os.link', side_effect=publish):
            self.save()
        self.assertEqual(len(list(self.directory.glob('*.json'))), 1)

    def test_failed_publish_removes_only_its_own_temporary_file(self):
        self.directory.mkdir()
        for suffix in ('json', 'tmp'):
            with self.subTest(suffix=suffix):
                path = self.directory / f'collision.{suffix}'
                path.write_bytes(b'keep this file')
                with mock.patch('pipeline.capture.uuid.uuid4') as identifier, \
                        self.assertLogs('pipeline.capture', level='ERROR'):
                    identifier.return_value.hex = 'collision'
                    self.save()
                self.assertEqual(path.read_bytes(), b'keep this file')
                path.unlink()
        self.assertEqual(list(self.directory.iterdir()), [])

    def post(self):
        events = []
        body = json.dumps(self.request.model_dump()).encode()

        async def receive():
            return {'type': 'http.request', 'body': body, 'more_body': False}

        async def send(message):
            events.append(message)

        def predict(request, *, trace=None):
            if trace is not None:
                trace.update(self.trace)
            return self.response

        def save_after_response(*args):
            self.assertTrue(any(event['type'] == 'http.response.body'
                                and not event.get('more_body', False) for event in events))
            capture.save_capture(*args)

        scope = {
            'type': 'http', 'asgi': {'version': '3.0'}, 'http_version': '1.1',
            'method': 'POST', 'scheme': 'http', 'path': '/predict', 'root_path': '',
            'query_string': b'', 'headers': [(b'content-type', b'application/json')],
            'server': ('127.0.0.1', 9054), 'client': ('127.0.0.1', 12345),
        }
        with mock.patch.object(example._pipeline, 'predict', side_effect=predict), \
                mock.patch('api.save_capture', side_effect=save_after_response):
            asyncio.run(api.app(scope, receive, send))
        self.assertEqual(events[0]['status'], 200)
        payload = b''.join(event.get('body', b'') for event in events)
        self.assertEqual(json.loads(payload), self.response.model_dump())

    def test_api_writes_capture_after_response_is_sent(self):
        with mock.patch.dict(os.environ, {'MEDICAL_CAPTURE_REQUESTS': '1'}):
            self.post()
        self.assertEqual(len(list(self.directory.glob('*.json'))), 1)

    def test_api_can_disable_capture(self):
        with mock.patch.dict(os.environ, {'MEDICAL_CAPTURE_REQUESTS': '0'}):
            self.post()
        self.assertFalse(self.directory.exists())

    def test_capture_failure_does_not_change_http_response(self):
        with mock.patch.dict(os.environ, {'MEDICAL_CAPTURE_REQUESTS': '1'}), \
                mock.patch('pipeline.capture.os.open', side_effect=OSError('Disk full')), \
                self.assertLogs('pipeline.capture', level='ERROR'):
            self.post()


if __name__ == '__main__':
    unittest.main()
