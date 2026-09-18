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
import sys
import time

from faster_whisper import WhisperModel

from utils import AUDIO_DIRECTORY, gold_evidence, group_questions_by_conversation

# Timestamps are sums of 20 ms steps; anything this close is the same float
# arithmetic, anything further is a different word boundary.
TOLERANCE_SECONDS = 1e-9


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
    args = parser.parse_args()

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
