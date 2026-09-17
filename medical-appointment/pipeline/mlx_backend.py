"""Local-cache-only Whisper and Qwen inference; importing this module is cheap."""

import io
import os
import time
from pathlib import Path

WHISPER_MODEL = 'mlx-community/whisper-large-v3-turbo'
QWEN_MODEL = 'mlx-community/Qwen3-8B-4bit'
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


class MLXBackend:
    def __init__(self):
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
        started = time.monotonic()
        samples = decode_audio(audio_bytes)
        result = self._transcribe(samples)
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
        }

    def _generate(self, words: list[dict], questions: list[str], max_tokens: int) -> str:
        if self._llm is None:
            snapshot = resolve_snapshot(QWEN_MODEL)
            from mlx_lm import load
            from mlx_lm.sample_utils import make_sampler

            self._llm, self._tokenizer = load(
                snapshot, tokenizer_config={'local_files_only': True}
            )
            self._sampler = make_sampler(temp=0.0)
        from mlx_lm import generate
        from pipeline.core import build_messages

        prompt = self._tokenizer.apply_chat_template(
            build_messages(words, questions),
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
