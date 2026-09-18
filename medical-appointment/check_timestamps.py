"""Check that this machine reproduces the annotators' word timestamps exactly.

    python check_timestamps.py                    # base, int8, CPU: expect 390/390
    python check_timestamps.py --compute-type float32
    python check_timestamps.py --limit 5          # quick look on five conversations

The evidence spans in ``question_train.csv`` were not drawn by hand. They are
faster-whisper word timestamps from the ``base`` model at int8 on CPU
(``language='en'``, ``word_timestamps=True``, every other setting default):
each span runs from the start of the first word of a quoted passage to the end
of its last word. On the 39 supplied conversations that configuration puts a
word boundary on all 390 annotated timestamps, down to the float bit pattern
(``28.520000000000003`` in sample_4, ``80.46000000000001`` in sample_70).

So if we transcribe the same way and quote the same words, the temporal IoU is
exactly 1. That only holds on a machine whose int8 kernels round the same way,
which is what this script checks: run it on the serving machine before trusting
it, and treat anything below 390/390 as a reason to look, not to ship blind.
Measured so far: 390/390 on an x86-64 CPU (AMD Zen 5, 12 threads; 4 threads
gave identical words on the conversations tried), with faster-whisper 1.2.1,
ctranslate2 4.8.2, av 18.1.0. The same model at float32
only lands 41% of spans exactly, but still scores a mean tIoU of 0.99 on the
right words, so a mismatch costs a little, not everything.
"""

import argparse
import collections
import hashlib
import os
import platform
import sys
import time

from faster_whisper import WhisperModel

from utils import AUDIO_DIRECTORY, gold_evidence, group_questions_by_conversation

# Timestamps are sums of 20 ms steps; anything this close is the same float
# arithmetic, anything further is a different word boundary.
TOLERANCE_SECONDS = 1e-9

# What the machine that reproduces the labels 390/390 (Windows, AMD Zen 5,
# ctranslate2 picking AVX2 + DNNL, no MKL) computes for sample_4. When a box
# misses, --fingerprint shows which layer differs: decoding, features or model.
REFERENCE = {
    'audio': 'e5d385d6663ce4ff',
    'features': 'f572bf91ad736a1f',
    'libavcodec': (62, 28, 102),
    'Let': (28.520000000000003, 29.12),
}


def fingerprint() -> int:
    """Print each layer's result for sample_4 next to the reference."""
    import av
    import ctranslate2
    import faster_whisper
    import numpy as np
    from faster_whisper.audio import decode_audio
    from faster_whisper.feature_extractor import FeatureExtractor

    def show(name, value, reference):
        print(f'  {name:<10} {"same" if value == reference else "DIFFERENT":<9} {value}  (reference {reference})')

    path = str(AUDIO_DIRECTORY / 'conversation_sample_4.mp3')
    audio = decode_audio(path, sampling_rate=16000)
    features = FeatureExtractor()(audio)

    print(f'faster-whisper {faster_whisper.__version__}, ctranslate2 {ctranslate2.__version__}, '
          f'av {av.__version__}, numpy {np.__version__}, {platform.system()} {platform.machine()}')
    show('libavcodec', tuple(av.library_versions.get('libavcodec', ())), REFERENCE['libavcodec'])
    show('audio', hashlib.sha256(audio.tobytes()).hexdigest()[:16], REFERENCE['audio'])
    show('features', hashlib.sha256(np.ascontiguousarray(features).tobytes()).hexdigest()[:16],
         REFERENCE['features'])

    # CT2_VERBOSE makes ctranslate2 log the ISA and GEMM backends it picked.
    os.environ['CT2_VERBOSE'] = '1'
    model = WhisperModel('base', device='cpu', compute_type='int8')
    segments, _ = model.transcribe(path, language='en', word_timestamps=True)
    lets = [(w.start, w.end) for s in segments for w in s.words if w.word.strip() == 'Let']
    show('"Let"', lets[0] if lets else None, REFERENCE['Let'])
    return 0


def word_boundaries(model: WhisperModel, audio_filename: str):
    """Every word start and end the annotators' pipeline would have produced."""
    segments, _ = model.transcribe(
        str(AUDIO_DIRECTORY / audio_filename),
        language='en',
        word_timestamps=True,
    )
    words = [word for segment in segments for word in segment.words]
    return [word.start for word in words], [word.end for word in words]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='base')
    parser.add_argument('--compute-type', default='int8')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=0,
                        help='CPU threads, 0 for the library default.')
    parser.add_argument('--limit', type=int, default=0,
                        help='Only check the first N conversations.')
    parser.add_argument('--fingerprint', action='store_true',
                        help='Compare decoding, features and model output on sample_4 '
                             'with the machine that reproduces the labels.')
    args = parser.parse_args()

    if args.fingerprint:
        return fingerprint()

    model = WhisperModel(
        args.model,
        device=args.device,
        compute_type=args.compute_type,
        cpu_threads=args.threads,
    )

    conversations = group_questions_by_conversation()
    if args.limit:
        conversations = conversations[:args.limit]

    counts = collections.Counter()
    slowest = 0.0

    for audio_filename, rows in conversations:
        started = time.time()
        starts, ends = word_boundaries(model, audio_filename)
        slowest = max(slowest, time.time() - started)

        for row in rows:
            gold = gold_evidence(row)
            if gold is None:
                continue

            start_ok = any(abs(gold[0] - s) < TOLERANCE_SECONDS for s in starts)
            end_ok = any(abs(gold[1] - e) < TOLERANCE_SECONDS for e in ends)

            counts['spans'] += 1
            counts['boundaries'] += 2
            counts['exact'] += start_ok + end_ok
            counts['spans_exact'] += start_ok and end_ok

            if not (start_ok and end_ok):
                print(f'  miss {row["question_id"]}: gold {gold[0]!r}-{gold[1]!r}')

    print(
        f'{args.model}/{args.compute_type}/{args.device}: '
        f'{counts["exact"]}/{counts["boundaries"]} boundaries exact, '
        f'{counts["spans_exact"]}/{counts["spans"]} spans exact, '
        f'slowest conversation {slowest:.1f}s'
    )
    return 0 if counts['exact'] == counts['boundaries'] else 1


if __name__ == '__main__':
    sys.exit(main())
