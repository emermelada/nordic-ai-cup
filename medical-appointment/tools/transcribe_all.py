"""Cache the supplied conversations with local Whisper, without loading the LLM."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import sys
import tempfile
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.mlx_backend import (
    ASR_SETTINGS, BACKEND_VERSION, SAMPLE_RATE, WHISPER_MODEL,
    MLXBackend, resolve_snapshot,
)
from utils import group_questions_by_conversation, load_sample_audio

CACHE_PRODUCER = 'tools.transcribe_all'
CACHE_VERSION = 1


def atomic_write_text(destination: Path, text: str, replace: bool = False) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='w', encoding='utf-8', dir=destination.parent,
            prefix=f'.{destination.name}.', suffix='.tmp', delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(temporary, destination)
        else:
            # Linking publishes a complete file without clobbering another writer.
            os.link(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def atomic_write_json(destination: Path, value, replace: bool = False) -> None:
    atomic_write_text(
        destination, json.dumps(value, indent=2, allow_nan=False) + '\n', replace
    )


def _number(value) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def validate_transcript(transcript: dict, require_duration: bool = False) -> None:
    if not isinstance(transcript, dict) or not isinstance(transcript.get('text'), str):
        raise ValueError('Transcript must have text and timestamped segments')
    if not isinstance(transcript.get('model'), str):
        raise ValueError('Transcript must identify its ASR model')
    if not _number(transcript.get('seconds')) or transcript['seconds'] < 0:
        raise ValueError('Transcript must have finite nonnegative ASR seconds')
    duration = transcript.get('duration')
    if require_duration or duration is not None:
        if not _number(duration) or duration <= 0:
            raise ValueError('Transcript duration must be positive and finite')
    if not isinstance(transcript.get('segments'), list):
        raise ValueError('Transcript segments must be a list')
    for segment in transcript['segments']:
        if not isinstance(segment, dict) or not isinstance(segment.get('words'), list):
            raise ValueError('Each transcript segment must have a words list')
        if not isinstance(segment.get('text'), str):
            raise ValueError('Each transcript segment must have text')
        for item in [segment, *segment['words']]:
            if (not isinstance(item, dict) or not _number(item.get('start'))
                    or not _number(item.get('end')) or item['start'] < 0
                    or item['end'] < item['start']):
                raise ValueError('Transcript contains invalid timestamps')
        if any(not isinstance(word.get('word'), str) for word in segment['words']):
            raise ValueError('Transcript contains invalid word text')


def cache_settings() -> dict:
    return {
        'producer': CACHE_PRODUCER,
        'version': CACHE_VERSION,
        'backend_version': BACKEND_VERSION,
        'model': WHISPER_MODEL,
        'snapshot': Path(resolve_snapshot(WHISPER_MODEL)).name,
        'settings': {**ASR_SETTINGS, 'sample_rate': SAMPLE_RATE, 'decoder': 'pyav'},
        'packages': {
            name: importlib.metadata.version(name)
            for name in ('mlx', 'mlx-whisper', 'numpy', 'av')
        },
    }


def compatible_cache(transcript, metadata: dict) -> bool:
    if not isinstance(transcript, dict) or transcript.get('_cache') != metadata:
        return False
    try:
        validate_transcript(transcript, require_duration=True)
    except ValueError:
        return False
    return transcript['model'] == WHISPER_MODEL


def positive_limit(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError('limit must be positive')
    return number


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=positive_limit)
    args = parser.parse_args(argv)
    conversations = group_questions_by_conversation()[:args.limit]
    settings = cache_settings()
    backend = MLXBackend()
    for index, (filename, _) in enumerate(conversations, 1):
        audio = load_sample_audio(filename)
        metadata = {
            **settings, 'source_sha256': hashlib.sha256(audio).hexdigest(),
            'complete': True,
        }
        destination = args.output / f'{Path(filename).stem}.json'
        previous_bytes = None
        if destination.exists() or destination.is_symlink():
            if destination.is_symlink():
                raise FileExistsError(f'Refusing to replace a symlink: {destination}')
            previous_bytes = destination.read_bytes()
            try:
                previous = json.loads(previous_bytes)
            except (ValueError, UnicodeError) as exc:
                raise FileExistsError(f'Refusing to replace unknown file: {destination}') from exc
            if compatible_cache(previous, metadata):
                print(f'{index}/{len(conversations)} {filename}: compatible cache, skipped', flush=True)
                continue
            if (not isinstance(previous, dict)
                    or not isinstance(previous.get('_cache'), dict)
                    or previous['_cache'].get('producer') != CACHE_PRODUCER):
                raise FileExistsError(
                    f'Refusing to replace legacy or unknown cache: {destination}. '
                    'Choose a different output directory.'
                )
        transcript = backend.transcribe(audio)
        validate_transcript(transcript, require_duration=True)
        transcript.update(file=Path(filename).stem, _cache=metadata)
        if previous_bytes is not None and destination.read_bytes() != previous_bytes:
            raise FileExistsError(f'Cache changed during transcription: {destination}')
        atomic_write_json(destination, transcript, replace=previous_bytes is not None)
        print(
            f'{index}/{len(conversations)} {filename}: {transcript["seconds"]:.2f}s '
            f'ASR including decode, {transcript["duration"]:.2f}s audio', flush=True,
        )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
