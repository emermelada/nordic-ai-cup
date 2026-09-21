"""CUDA serving backend: faster-whisper for ASR, a local vLLM server for every LLM pass.

Same request/response contract as the MLX backend, so the runtime, prompts and evidence
stage are unchanged. Select it with MEDICAL_BACKEND=vllm. The model vLLM serves is
discovered from its /v1/models endpoint; the MLX model constants are ignored except the
stage-B mode, budget and token caps.
"""

import concurrent.futures
import os
import time

import requests

import logging

from pipeline.evidence import energy_envelope
from pipeline.mlx_backend import DEFAULT_PROMPT, SAMPLE_RATE, MLXBackend, decode_audio

VLLM_URL = os.environ.get('VLLM_URL', 'http://127.0.0.1:8000/v1').rstrip('/')
# Optional second vLLM server for the answer pass only, so a different model can write the
# drafts that the evidence model refines (another model's drafts refine better than its own).
VLLM_ANSWER_URL = os.environ.get('VLLM_ANSWER_URL', '').rstrip('/') or VLLM_URL
ANSWER_PASS = 'answer'
# Vast.ai's vLLM template protects the server with a bearer token; leave empty for a bare server.
VLLM_API_KEY = os.environ.get('VLLM_API_KEY', '')
# MEDICAL_VLLM_MODEL: set when the endpoint serves several models (a hosted development
# endpoint); otherwise the single served model is discovered from /v1/models. The name avoids
# Vast's template variable VLLM_MODEL, which names the model the template launches.
VLLM_MODEL = os.environ.get('MEDICAL_VLLM_MODEL', '')
# 'turbo': large-v3-turbo on the GPU; 'base': the annotators' base/int8/CPU words (pipeline.base_asr).
ASR_MODE = os.environ.get('MEDICAL_ASR_MODE', 'turbo')
# Per-question evidence prompts run side by side; the server batches them and its prefix
# cache pays for the shared transcript once.
EVIDENCE_PARALLEL = int(os.environ.get('MEDICAL_EVIDENCE_PARALLEL', '10'))
# 'turbo' transcribes the audio a second time with large-v3-turbo purely so the rescue pass
# reads accurate wording; timestamps and evidence still come from the annotators' `base`
# words. Our phonetic-name rule only covers garbles we have already seen, this covers them all.
RESCUE_READING = os.environ.get('MEDICAL_RESCUE_READING', '')
# Letting the model reason before it answers was unaffordable on the Mac (45 s a call); with
# the questions in flight together on a server it costs seconds. Off by default.
EVIDENCE_THINKING = os.environ.get('MEDICAL_EVIDENCE_THINKING') == '1'
THINKING_MAX_TOKENS = int(os.environ.get('MEDICAL_THINKING_MAX_TOKENS', '900'))
WHISPER_CUDA_MODEL = os.environ.get('WHISPER_CUDA_MODEL', 'large-v3-turbo')
WHISPER_COMPUTE_TYPE = os.environ.get('WHISPER_COMPUTE_TYPE', 'float16')
LLM_REQUEST_TIMEOUT = 45.0
logger = logging.getLogger(__name__)
ASR_SETTINGS = {
    'language': 'en',
    'word_timestamps': True,
    'temperature': 0.0,
    'condition_on_previous_text': False,
    'beam_size': 5,
}


class VLLMBackend(MLXBackend):
    def __init__(self, prompt=DEFAULT_PROMPT):
        super().__init__(prompt)
        self._whisper = None
        self._served = {}

    @staticmethod
    def _headers():
        return {'Authorization': f'Bearer {VLLM_API_KEY}'} if VLLM_API_KEY else {}

    def served_model(self, url: str = VLLM_URL) -> str:
        if VLLM_MODEL:
            return VLLM_MODEL
        if url not in self._served:
            response = requests.get(f'{url}/models', headers=self._headers(), timeout=10)
            response.raise_for_status()
            self._served[url] = response.json()['data'][0]['id']
        return self._served[url]

    def complete(self, words: list[dict], questions: list[str]) -> str:
        return self._generate(words, questions, max_tokens=1200, model_id=ANSWER_PASS)

    def _whisper_model(self):
        if self._whisper is None:
            from faster_whisper import WhisperModel

            self._whisper = WhisperModel(WHISPER_CUDA_MODEL, device='cuda', compute_type=WHISPER_COMPUTE_TYPE)
        return self._whisper

    def _transcribe(self, samples):
        segments, _ = self._whisper_model().transcribe(samples, **ASR_SETTINGS)
        return [
            {
                'start': float(segment.start), 'end': float(segment.end), 'text': segment.text,
                'words': [
                    {'word': word.word, 'start': float(word.start), 'end': float(word.end),
                     'p': float(word.probability) if word.probability is not None else None}
                    for word in (segment.words or [])
                ],
            }
            for segment in segments
        ]

    def transcribe(self, audio_bytes: bytes) -> dict:
        self.rescue_reading = None
        if ASR_MODE == 'base':
            from pipeline import base_asr

            result = base_asr.transcribe(audio_bytes)
            if RESCUE_READING == 'turbo':
                from pipeline.stage_b import render_sentences

                try:
                    segments = self._transcribe(decode_audio(audio_bytes))
                    words = [word for segment in segments for word in segment['words']]
                    self.rescue_reading = render_sentences(words) if words else None
                except Exception:
                    logger.warning('Accurate reading for the rescue pass failed; '
                                   'falling back to the served transcript', exc_info=True)
            return result
        started = time.monotonic()
        samples = decode_audio(audio_bytes)
        segments = self._transcribe(samples)
        return {
            'model': f'faster-whisper/{WHISPER_CUDA_MODEL}',
            'seconds': time.monotonic() - started,
            'duration': len(samples) / SAMPLE_RATE,
            'segments': segments,
            'text': ''.join(segment['text'] for segment in segments),
            'energy_db': energy_envelope(samples),
        }

    def generate_messages(self, messages: list[dict], max_tokens: int = 900,
                          model_id: str | None = None, adapter: str | None = None) -> str:
        url = VLLM_ANSWER_URL if model_id == ANSWER_PASS else VLLM_URL
        thinking = EVIDENCE_THINKING and model_id != ANSWER_PASS
        body = {
            'model': self.served_model(url), 'messages': messages, 'max_tokens': max_tokens,
            'temperature': 0.0, 'chat_template_kwargs': {'enable_thinking': thinking},
        }
        started = time.monotonic()
        response = requests.post(f'{url}/chat/completions', json=body, headers=self._headers(),
                                 timeout=LLM_REQUEST_TIMEOUT)
        response.raise_for_status()
        text = response.json()['choices'][0]['message'].get('content') or ''
        self.last_generation_seconds = time.monotonic() - started
        # A model that reasons anyway must not leave its channel in front of the JSON.
        return text.split('</think>')[-1] if '</think>' in text else text

    def score_yes(self, prompts, request_started):
        """P(yes) for each (key, messages), read from the first yes/no token's logprobs."""
        from pipeline.mlx_backend import EVIDENCE_BUDGET_SECONDS
        from pipeline.verify import probability_of_yes

        prompts = list(prompts)
        remaining = EVIDENCE_BUDGET_SECONDS - (time.monotonic() - request_started)
        if not prompts or remaining <= 0:
            return {}

        def ask(messages):
            body = {
                'model': self.served_model(), 'messages': messages, 'max_tokens': 4,
                'temperature': 0.0, 'logprobs': True, 'top_logprobs': 8,
                'chat_template_kwargs': {'enable_thinking': False},
            }
            response = requests.post(f'{VLLM_URL}/chat/completions', json=body,
                                     headers=self._headers(), timeout=LLM_REQUEST_TIMEOUT)
            response.raise_for_status()
            choice = response.json()['choices'][0]
            return probability_of_yes((choice.get('logprobs') or {}).get('content'))

        scores = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=EVIDENCE_PARALLEL) as pool:
            futures = {pool.submit(ask, messages): key for key, messages in prompts}
            done, pending = concurrent.futures.wait(futures, timeout=remaining)
            for future in pending:
                future.cancel()
            for future in done:
                try:
                    scores[futures[future]] = future.result()
                except Exception:
                    logger.warning('Candidate verification request failed', exc_info=True)
        return scores

    def generate_many(self, prompts, max_tokens, request_started):
        """All prompts at once against the server, keeping the budget as a wall-clock cap."""
        from pipeline.mlx_backend import EVIDENCE_ADAPTER, EVIDENCE_BUDGET_SECONDS, EVIDENCE_MODEL

        if EVIDENCE_THINKING:
            max_tokens = max(max_tokens, THINKING_MAX_TOKENS)

        prompts = list(prompts)
        if not prompts:
            return {}
        remaining = EVIDENCE_BUDGET_SECONDS - (time.monotonic() - request_started)
        if remaining <= 0:
            return {}
        outputs = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=EVIDENCE_PARALLEL) as pool:
            futures = {
                pool.submit(self.generate_messages, messages, max_tokens, EVIDENCE_MODEL, EVIDENCE_ADAPTER): key
                for key, messages in prompts
            }
            done, pending = concurrent.futures.wait(futures, timeout=remaining)
            for future in pending:
                future.cancel()
            for future in done:
                try:
                    outputs[futures[future]] = future.result()
                except Exception:
                    logger.warning('Per-question evidence request failed', exc_info=True)
        return outputs

    def warmup(self) -> None:
        import numpy as np

        if ASR_MODE == 'base':
            from pipeline import base_asr

            segments, _ = base_asr.load_model().transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32),
                                                           language='en', word_timestamps=True)
            list(segments)
        else:
            self._transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32))
        self.generate_messages([{'role': 'user', 'content': 'Warm up.'}], 1)
        self.generate_messages([{'role': 'user', 'content': 'Warm up.'}], 1, ANSWER_PASS)
        self.last_generation_seconds = None
