import os
import sys
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from pipeline import runtime, vllm_backend


class FakeWord(SimpleNamespace):
    pass


class VLLMBackendTests(unittest.TestCase):
    def test_generate_uses_served_model_deterministic_sampling_and_strips_thinking(self):
        backend = vllm_backend.VLLMBackend()
        models = Mock(); models.json.return_value = {'data': [{'id': 'qwen'}]}
        completion = Mock()
        completion.json.return_value = {'choices': [{'message': {'content': '<think>\nx\n</think>\n{"results":[]}'}}]}
        with patch.object(vllm_backend.requests, 'get', return_value=models) as get, \
                patch.object(vllm_backend.requests, 'post', return_value=completion) as post:
            self.assertEqual(backend.generate_messages([{'role': 'user', 'content': 'q'}], 7), '\n{"results":[]}')
            self.assertEqual(backend.generate_messages([{'role': 'user', 'content': 'q'}], 7, 'ignored/model', '/adapter'), '\n{"results":[]}')
        get.assert_called_once()
        self.assertEqual(post.call_count, 2)
        body = post.call_args.kwargs['json']
        self.assertEqual(body['model'], 'qwen')
        self.assertEqual(body['temperature'], 0.0)
        self.assertEqual(body['max_tokens'], 7)
        self.assertEqual(body['chat_template_kwargs'], {'enable_thinking': False})
        self.assertGreaterEqual(backend.last_generation_seconds, 0)

    def test_transcribe_matches_the_mlx_transcript_shape(self):
        backend = vllm_backend.VLLMBackend()
        segment = SimpleNamespace(start=0.5, end=1.2, text=' Hello there.', words=[
            FakeWord(word=' Hello', start=0.5, end=0.8, probability=0.9),
            FakeWord(word=' there.', start=0.85, end=1.2, probability=None),
        ])
        model = Mock(); model.transcribe.return_value = (iter([segment]), None)
        loader = Mock(return_value=model)
        samples = [0.0] * 32000
        with patch.dict(sys.modules, {'faster_whisper': SimpleNamespace(WhisperModel=loader)}), \
                patch.object(vllm_backend, 'decode_audio', return_value=samples), \
                patch.object(vllm_backend, 'energy_envelope', return_value=[-60.0]):
            output = backend.transcribe(b'mp3')
            backend.transcribe(b'mp3')
        loader.assert_called_once_with('large-v3-turbo', device='cuda', compute_type='float16')
        self.assertEqual(model.transcribe.call_args.kwargs, vllm_backend.ASR_SETTINGS)
        self.assertEqual(output['duration'], 2.0)
        self.assertEqual(output['text'], ' Hello there.')
        words = output['segments'][0]['words']
        self.assertEqual(words[0], {'word': ' Hello', 'start': 0.5, 'end': 0.8, 'p': 0.9})
        self.assertIsNone(words[1]['p'])
        self.assertEqual(output['energy_db'], [-60.0])

    def test_answer_pass_can_use_a_second_server(self):
        backend = vllm_backend.VLLMBackend()
        models = Mock(); models.json.return_value = {'data': [{'id': 'm'}]}
        completion = Mock(); completion.json.return_value = {'choices': [{'message': {'content': 'ok'}}]}
        with patch.object(vllm_backend, 'VLLM_ANSWER_URL', 'http://answers:8001/v1'), \
                patch.object(vllm_backend.requests, 'get', return_value=models) as get, \
                patch.object(vllm_backend.requests, 'post', return_value=completion) as post, \
                patch('pipeline.evidence.build_compact_messages', return_value=[{'role': 'user', 'content': 'q'}]):
            self.assertEqual(backend.complete([], ['q']), 'ok')
            self.assertEqual(post.call_args.args[0], 'http://answers:8001/v1/chat/completions')
            backend.complete_evidence([], ['q'], [True], {}, [(0.0, 1.0)], time.monotonic())
            self.assertEqual(post.call_args.args[0], vllm_backend.VLLM_URL + '/chat/completions')
        self.assertEqual(get.call_count, 2)

    def test_api_key_is_sent_as_bearer_when_set(self):
        backend = vllm_backend.VLLMBackend()
        models = Mock(); models.json.return_value = {'data': [{'id': 'm'}]}
        completion = Mock(); completion.json.return_value = {'choices': [{'message': {'content': 'ok'}}]}
        with patch.object(vllm_backend, 'VLLM_API_KEY', 'secret'), \
                patch.object(vllm_backend.requests, 'get', return_value=models) as get, \
                patch.object(vllm_backend.requests, 'post', return_value=completion) as post:
            backend.generate_messages([{'role': 'user', 'content': 'q'}], 3)
        self.assertEqual(get.call_args.kwargs['headers'], {'Authorization': 'Bearer secret'})
        self.assertEqual(post.call_args.kwargs['headers'], {'Authorization': 'Bearer secret'})
        with patch.object(vllm_backend, 'VLLM_API_KEY', ''):
            self.assertEqual(vllm_backend.VLLMBackend._headers(), {})

    def test_warmup_runs_asr_and_one_token(self):
        backend = vllm_backend.VLLMBackend()
        with patch.object(backend, '_transcribe') as asr, patch.object(backend, 'generate_messages') as llm:
            backend.warmup()
        asr.assert_called_once()
        self.assertEqual(llm.call_count, 2)
        self.assertEqual(llm.call_args.args[1], 1)
        self.assertIsNone(backend.last_generation_seconds)

    def test_factory_switch(self):
        with patch.dict(os.environ, {'MEDICAL_BACKEND': 'vllm'}):
            self.assertIsInstance(runtime._make_backend(), vllm_backend.VLLMBackend)
        with patch.dict(os.environ, {'MEDICAL_BACKEND': ''}):
            self.assertEqual(type(runtime._make_backend()).__name__, 'MLXBackend')


if __name__ == '__main__':
    unittest.main()
