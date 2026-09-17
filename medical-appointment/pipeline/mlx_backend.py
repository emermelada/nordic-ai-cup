"""Local-cache-only Whisper and Qwen inference; importing this module is cheap."""

import io
import os
import time
from pathlib import Path

# The 8-bit ASR is half the size of the fp16 one and scored the same; mlx-whisper wants
# the weights named weights.safetensors, so it is staged locally (see RUNNING.md).
WHISPER_MODEL = 'models/whisper-large-v3-turbo-8bit'
# A 16 GB answering model leaves no room for idle ASR weights on a 24 GB machine.
RELEASE_ASR_AFTER_TRANSCRIBE = True
QWEN_MODEL = 'mlx-community/Qwen3.5-9B-4bit'
DEFAULT_PROMPT = 'compact'
SAMPLE_RATE = 16000
BACKEND_VERSION = 1
ASR_SETTINGS = {
    'language': 'en',
    'word_timestamps': True,
    'temperature': 0.0,
    'condition_on_previous_text': False,
}


def resolve_snapshot(model_id: str) -> str:
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    local = Path(__file__).resolve().parent.parent / model_id
    if local.is_dir():
        return str(local)
    from huggingface_hub import snapshot_download

    try:
        snapshot = Path(snapshot_download(model_id, local_files_only=True))
    except Exception as exc:
        raise RuntimeError(f'Local model cache unavailable for {model_id}: {exc}') from exc
    if not snapshot.is_dir():
        raise RuntimeError(f'Local model snapshot is missing: {snapshot}')
    return str(snapshot)


def decode_audio(audio_bytes: bytes):
    """Decode in-process so terminating inference cannot orphan an ffmpeg child."""
    import av
    import numpy as np

    if not audio_bytes:
        raise ValueError('Audio is empty')
    chunks = []
    with av.open(io.BytesIO(audio_bytes)) as container:
        if not container.streams.audio:
            raise ValueError('Input contains no audio stream')
        resampler = av.AudioResampler(format='fltp', layout='mono', rate=SAMPLE_RATE)
        for frame in container.decode(audio=0):
            chunks.extend(frame.to_ndarray().reshape(-1) for frame in resampler.resample(frame))
        chunks.extend(frame.to_ndarray().reshape(-1) for frame in resampler.resample(None))
    if not chunks:
        raise ValueError('Input decoded to no audio samples')
    samples = np.concatenate(chunks).astype(np.float32, copy=False)
    if not samples.size or not np.isfinite(samples).all():
        raise ValueError('Input decoded to empty or non-finite audio')
    return samples


def release_asr():
    """Drop cached Whisper weights so the answering model keeps the memory."""
    import mlx.core as mx
    from mlx_whisper.transcribe import ModelHolder

    ModelHolder.model = None
    ModelHolder.model_path = None
    mx.clear_cache()


class MLXBackend:
    def __init__(self, prompt=DEFAULT_PROMPT):
        if prompt not in ('legacy', 'focused', 'compact', 'minimal'):
            raise ValueError(f'Unknown answer prompt: {prompt}')
        self.prompt = prompt
        self._whisper_path = None
        self._llm = None
        self._tokenizer = None
        self._sampler = None
        self.last_generation_seconds = None

    def _transcribe(self, samples):
        if self._whisper_path is None:
            self._whisper_path = resolve_snapshot(WHISPER_MODEL)
        import mlx_whisper

        return mlx_whisper.transcribe(
            samples, path_or_hf_repo=self._whisper_path, **ASR_SETTINGS
        )

    def transcribe(self, audio_bytes: bytes) -> dict:
        from pipeline.evidence import energy_envelope

        started = time.monotonic()
        samples = decode_audio(audio_bytes)
        try:
            result = self._transcribe(samples)
        finally:
            if RELEASE_ASR_AFTER_TRANSCRIBE:
                release_asr()
        segments = [
            {
                'start': float(segment['start']),
                'end': float(segment['end']),
                'text': segment['text'],
                'words': [
                    {
                        'word': word['word'],
                        'start': float(word['start']),
                        'end': float(word['end']),
                        'p': (float(word['probability'])
                              if word.get('probability') is not None else None),
                    }
                    for word in segment.get('words', [])
                ],
            }
            for segment in result['segments']
        ]
        return {
            'model': WHISPER_MODEL,
            'seconds': time.monotonic() - started,
            'duration': len(samples) / SAMPLE_RATE,
            'segments': segments,
            'text': result['text'],
            'energy_db': energy_envelope(samples),
        }

    def _generate(self, words: list[dict], questions: list[str], max_tokens: int) -> str:
        from pipeline.core import build_messages
        from pipeline.evidence import (
            build_compact_messages, build_focused_messages, build_minimal_messages,
        )

        builder = {'legacy': build_messages, 'focused': build_focused_messages,
                   'compact': build_compact_messages, 'minimal': build_minimal_messages}[self.prompt]
        return self.generate_messages(builder(words, questions), max_tokens)

    def generate_messages(self, messages: list[dict], max_tokens: int = 900) -> str:
        if self._llm is None:
            snapshot = resolve_snapshot(QWEN_MODEL)
            from mlx_lm import load
            from mlx_lm.sample_utils import make_sampler

            self._llm, self._tokenizer = load(
                snapshot, tokenizer_config={'local_files_only': True}
            )
            self._sampler = make_sampler(temp=0.0)
        from mlx_lm import generate

        prompt = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
        started = time.monotonic()
        result = generate(
            self._llm, self._tokenizer, prompt=prompt, max_tokens=max_tokens,
            sampler=self._sampler, verbose=False,
        )
        self.last_generation_seconds = time.monotonic() - started
        return result

    def complete(self, words: list[dict], questions: list[str]) -> str:
        return self._generate(words, questions, max_tokens=900)

    def warmup(self) -> None:
        import numpy as np

        self._transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32))
        self._generate([], ['Was fever discussed?'], max_tokens=1)
