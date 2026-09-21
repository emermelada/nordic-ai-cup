"""Speech recognition in the annotators' coordinate system.

The gold evidence spans are faster-whisper ``base`` word timestamps computed at int8 on
CPU with ``language='en'`` and ``word_timestamps=True`` and every other setting default;
on the 39 supplied conversations that configuration puts a word boundary on all 390
annotated timestamps down to the float bit pattern (found by the team's other branch,
``check_timestamps.py`` there). A machine whose int8 kernels round differently lands most
boundaries within a few tens of milliseconds instead, which still beats every other ASR.

So this module transcribes exactly that way: the audio goes through a file on disk, the
timestamps are kept as the raw floats, no VAD, no rounding, no onset refinement. Sentences
are split after words ending in ``. ? !`` (titles such as ``Dr.`` excepted), which is the
unit the annotators quoted in.
"""

import os
import re
import tempfile
import threading
import time

MODEL_NAME = 'base'
COMPUTE_TYPE = 'int8'
# The snapshot the 390/390 check ran against.
MODEL_REVISION = 'ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66'
_USABLE_CPUS = (
    len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count()
) or 4
CPU_THREADS = int(os.environ.get('ASR_CPU_THREADS', min(_USABLE_CPUS, 16)))
SAMPLE_RATE = 16000

_SENTENCE_END = re.compile(r'[.?!]["\')\]]*$')
_TITLES = frozenset({'dr.', 'mr.', 'mrs.', 'ms.', 'prof.', 'st.'})

_model = None
_lock = threading.Lock()


def load_model():
    global _model
    with _lock:
        if _model is None:
            from faster_whisper import WhisperModel

            _model = WhisperModel(MODEL_NAME, device='cpu', compute_type=COMPUTE_TYPE,
                                  cpu_threads=CPU_THREADS, revision=MODEL_REVISION)
    return _model


def closes_sentence(word_text: str) -> bool:
    token = word_text.strip()
    return bool(_SENTENCE_END.search(token)) and token.lower() not in _TITLES


def sentence_ranges(words):
    """Inclusive word-index ranges split on end punctuation only, the annotators' unit."""
    ranges, first = [], 0
    for index, word in enumerate(words):
        if closes_sentence(word['word']) or index == len(words) - 1:
            ranges.append((first, index))
            first = index + 1
    return ranges


def transcribe_path(path: str, model=None) -> list[dict]:
    """Raw words from the file on disk, exactly as the annotators' pipeline decoded it."""
    model = model or load_model()
    segments, _ = model.transcribe(path, language='en', word_timestamps=True)
    return [
        {'start': float(segment.start), 'end': float(segment.end), 'text': segment.text,
         'words': [{'word': word.word, 'start': word.start, 'end': word.end,
                    'p': float(word.probability) if word.probability is not None else None}
                   for word in (segment.words or [])]}
        for segment in segments
    ]


def transcribe(audio_bytes: bytes, model=None) -> dict:
    """The pipeline's transcript dict, flagged so no timing refinement touches the words."""
    started = time.monotonic()
    handle = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
    try:
        handle.write(audio_bytes)
        handle.close()
        segments = transcribe_path(handle.name, model)
        from faster_whisper.audio import decode_audio

        duration = len(decode_audio(handle.name, sampling_rate=SAMPLE_RATE)) / SAMPLE_RATE
    finally:
        os.unlink(handle.name)
    return {
        'model': f'faster-whisper/{MODEL_NAME}-{COMPUTE_TYPE}-cpu',
        'seconds': time.monotonic() - started,
        'duration': float(duration),
        'segments': segments,
        'text': ''.join(segment['text'] for segment in segments),
        'exact_timestamps': True,
    }
