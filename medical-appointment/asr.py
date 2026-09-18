"""Speech recognition in the annotators' coordinate system.

The evidence spans in the training data are faster-whisper ``base`` word
timestamps, computed at int8 on CPU with ``language='en'`` and
``word_timestamps=True``, every other setting default. ``check_timestamps.py``
proves it: that configuration lands a word boundary on all 390 annotated
timestamps, down to the float bit pattern.

So this module transcribes exactly that way and nothing else. A bigger model, a
different compute type, a GPU or a VAD pass would all move word boundaries, and
every moved boundary costs temporal IoU on the larger half of the score. The
timestamps are kept as the raw floats faster-whisper produced — rounding them
would already break the exact match.

What comes back is a ``Transcript``: the words, plus the words grouped into
numbered sentences. Sentences are the unit an answering model reads and cites;
words are the unit a span is cut from.
"""

import json
import os
import re
import tempfile
import threading
from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple

MODEL_NAME = 'base'
COMPUTE_TYPE = 'int8'
# The snapshot the 390/390 check ran against. Systran has not touched this repo
# in years, but a pinned revision cannot change under us on the serving box.
MODEL_REVISION = 'ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66'
# Thread count does not change the output (checked at 4 and 12), only the
# speed: the library default was ~1.6x slower than 12 threads on the dev laptop.
# Counted from the CPUs this process may run on, not os.cpu_count(): a rented
# container sees every core of the host (448 on one vast box) but may use 6.
_USABLE_CPUS = (
    len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count()
) or 4
CPU_THREADS = int(os.environ.get('ASR_CPU_THREADS', min(_USABLE_CPUS, 16)))

# A word ending in one of these closes a sentence, unless it is a title that
# only looks like it ("Dr. Thorson").
_SENTENCE_END = re.compile(r'[.?!]["\')\]]*$')
_TITLES = frozenset({'dr.', 'mr.', 'mrs.', 'ms.', 'prof.', 'st.'})

_model = None
_model_lock = threading.Lock()

Span = Tuple[float, float]


@dataclass
class Word:
    start: float
    end: float
    text: str


@dataclass
class Sentence:
    first_word: int
    last_word: int
    start: float
    end: float
    text: str


@dataclass
class Transcript:
    words: List[Word]
    sentences: List[Sentence]

    def span(self, first_word: int, last_word: int) -> Span:
        """The time span from the start of one word to the end of another."""
        return self.words[first_word].start, self.words[last_word].end

    def sentence_span(self, first: int, last: int) -> Span:
        """The time span covering whole sentences ``first..last``."""
        return self.sentences[first].start, self.sentences[last].end

    def render(self) -> str:
        """One numbered line per sentence, the way an answering model reads it."""
        return '\n'.join(
            f'[{index}] {sentence.text}'
            for index, sentence in enumerate(self.sentences)
        )

    def to_json(self) -> str:
        return json.dumps([asdict(word) for word in self.words])

    @classmethod
    def from_json(cls, text: str) -> 'Transcript':
        return from_words([Word(**word) for word in json.loads(text)])


def from_words(words: List[Word]) -> Transcript:
    """Group words into sentences on end punctuation."""
    sentences = []
    first = 0

    for index, word in enumerate(words):
        token = word.text.strip()
        closes = _SENTENCE_END.search(token) and token.lower() not in _TITLES

        if closes or index == len(words) - 1:
            sentences.append(Sentence(
                first_word=first,
                last_word=index,
                start=words[first].start,
                end=word.end,
                text=''.join(w.text for w in words[first:index + 1]).strip(),
            ))
            first = index + 1

    return Transcript(words=words, sentences=sentences)


def load_model():
    """Load (once) and return the Whisper model. Safe to call from any thread."""
    global _model

    with _model_lock:
        if _model is None:
            from faster_whisper import WhisperModel

            _model = WhisperModel(
                MODEL_NAME,
                device='cpu',
                compute_type=COMPUTE_TYPE,
                cpu_threads=CPU_THREADS,
                revision=MODEL_REVISION,
            )

    return _model


def transcribe_file(path: str, model=None) -> Transcript:
    """Transcribe an audio file exactly the way the annotators did."""
    model = model or load_model()
    segments, _ = model.transcribe(path, language='en', word_timestamps=True)

    words = [
        Word(start=word.start, end=word.end, text=word.word)
        for segment in segments
        for word in (segment.words or [])
    ]
    return from_words(words)


def transcribe(audio_bytes: bytes, model=None) -> Transcript:
    """Transcribe MP3 bytes off the wire.

    Goes through a file on disk rather than a buffer so the decode path is the
    one the annotators used. ``delete=False`` plus an explicit close because on
    Windows an open NamedTemporaryFile cannot be reopened by PyAV.
    """
    handle = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
    try:
        handle.write(audio_bytes)
        handle.close()
        return transcribe_file(handle.name, model)
    finally:
        os.unlink(handle.name)


def cached_transcript(path: str, cache_path: str, model=None) -> Transcript:
    """``transcribe_file`` behind a JSON cache, for offline experiments."""
    if os.path.exists(cache_path):
        with open(cache_path, encoding='utf-8') as f:
            return Transcript.from_json(f.read())

    transcript = transcribe_file(path, model)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'w', encoding='utf-8') as f:
        f.write(transcript.to_json())
    return transcript


def first_sentence_at(transcript: Transcript, time: float) -> Optional[int]:
    """Index of the sentence that contains ``time``, if any does."""
    for index, sentence in enumerate(transcript.sentences):
        if sentence.start <= time <= sentence.end:
            return index
    return None
