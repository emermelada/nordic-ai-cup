"""Score answerers against the 39 supplied conversations without the HTTP round trip.

    python eval_offline.py cache                 # transcribe once, the annotators' way (~5 min)
    python eval_offline.py score gold            # gold answers + gold quotes: must print 1.000
    python eval_offline.py score gold-lines      # gold answers, whole sentences: the no-quote ceiling
    python eval_offline.py score gold-noisy      # gold quotes with ASR-style noise: exercises fuzzy matching
    python eval_offline.py score lexical         # the no-model fallback on its own
    python eval_offline.py score llm:answer --dump data/answers.jsonl

Everything after the transcript is the endpoint's own code: ``pipeline`` makes
the yes/no call and cuts the spans, and ``local_evaluator.Statistics`` — the
service's scoring, ported by the organisers — scores them. So a number here
means what a validation attempt would say about the same answers, minus the
ASR time, which ``cache`` reports separately.

Each run scores the same answers under every evidence policy (the quote as cut,
or widened to whole sentences) and every yes-threshold, so both settings come
from measurement. ``--dump`` writes one JSON line per question — the question,
what the model cited, what each policy sent, and the gold quote — for reading
where spans go wrong.

The ``gold*`` answerers read the labels. They are not models; they measure the
ceiling of each way of pointing, and prove the plumbing: ``gold`` below 1.000
means a transcript or the matcher is off, not an answerer.

Few-shot examples taken from the training conversations must not be scored on
the conversations they came from. An answerer module that lists them in
``FEW_SHOT_SOURCES`` (``llm.py`` does) has them left out automatically;
``--skip`` adds more.
"""

import argparse
import collections
import importlib
import json
import random
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import asr
import pipeline
from answering import Answer, load
from evidence import Evidence, sentences_overlapping, words_in_span
from local_evaluator import Statistics
from utils import (
    AUDIO_DIRECTORY,
    DATA_DIRECTORY,
    gold_evidence,
    group_questions_by_conversation,
    temporal_iou,
)

CACHE_DIRECTORY = DATA_DIRECTORY / 'transcripts' / f'{asr.MODEL_NAME}-{asr.COMPUTE_TYPE}'

THRESHOLD_GRID = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


@dataclass
class Record:
    """One question: everything needed to rescore it at any threshold or policy."""

    row: dict
    answer: Answer  # what was used: the answerer's own, or the lexical fallback
    evidence: Dict[str, Evidence]  # one per evidence policy
    fell_back: bool
    transcript: asr.Transcript

    @property
    def p_yes(self) -> float:
        return self.answer.p_yes


def cache_path(audio_filename: str) -> Path:
    return CACHE_DIRECTORY / (Path(audio_filename).stem + '.json')


def cache() -> int:
    """Transcribe every supplied conversation once, and time it."""
    model = asr.load_model()
    slowest = 0.0

    for audio_filename, _ in group_questions_by_conversation():
        target = cache_path(audio_filename)
        if target.exists():
            continue

        started = time.time()
        transcript = asr.cached_transcript(str(AUDIO_DIRECTORY / audio_filename), str(target), model)
        elapsed = time.time() - started
        slowest = max(slowest, elapsed)
        print(f'  {audio_filename}: {len(transcript.words)} words, '
              f'{len(transcript.sentences)} sentences, {elapsed:.1f}s')

    print(f'cached in {CACHE_DIRECTORY} (slowest this run: {slowest:.1f}s)')
    return 0


def load_transcript(audio_filename: str) -> asr.Transcript:
    target = cache_path(audio_filename)
    if not target.exists():
        sys.exit(f'No cached transcript for {audio_filename}. Run: python eval_offline.py cache')
    return asr.Transcript.from_json(target.read_text(encoding='utf-8'))


# --------------------------------------------------------------------------- #
# Oracle answerers: they read the labels, to measure ceilings and the plumbing
# --------------------------------------------------------------------------- #

def _words(transcript: asr.Transcript, span) -> Optional[str]:
    if not span:
        return None
    indices = words_in_span(transcript, span)
    return ''.join(transcript.words[i].text for i in indices).strip() if indices else None


def _noisy(quote: str, rng: random.Random) -> str:
    """Roughly what a model does to a quote: case, punctuation, the odd letter."""
    words = []
    for word in quote.split():
        roll = rng.random()
        if roll < 0.10 and len(word) > 4:
            cut = rng.randrange(1, len(word) - 1)
            word = word[:cut] + word[cut + 1:]
        elif roll < 0.15 and len(word) > 3:
            word = word + 'e'
        words.append(word)
    return ' '.join(words).lower().replace(',', '').replace('.', '')


def gold_answers(mode: str, transcript: asr.Transcript, rows, rng: random.Random) -> List[Answer]:
    answers = []

    for row in rows:
        gold = gold_evidence(row)
        if int(row['label']) != 1 or gold is None:
            answers.append(Answer(p_yes=0.0))
            continue

        lines = sentences_overlapping(transcript, gold)
        quote = _words(transcript, gold)

        if mode == 'gold-lines':
            quote = None
        elif mode == 'gold-noisy' and quote:
            quote = _noisy(quote, rng)

        answers.append(Answer(p_yes=1.0, lines=lines, quote=quote))

    return answers


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

def score(args) -> int:
    oracle = args.answerer in ('gold', 'gold-lines', 'gold-noisy')
    answerer = None if oracle else load(args.answerer)
    rng = random.Random(0)
    skip = set(args.skip.split(',')) if args.skip else set()

    # An answerer that learned from some training conversations (its prompt's
    # worked examples) would be marking its own homework on them.
    if not oracle:
        module = importlib.import_module(args.answerer.partition(':')[0])
        skip |= set(getattr(module, 'FEW_SHOT_SOURCES', ()))
    if skip:
        print(f'Leaving out {len(skip)} conversations: {", ".join(sorted(skip))}')

    records = []
    latencies = []

    conversations = group_questions_by_conversation()
    if args.limit:
        conversations = conversations[:args.limit]

    for audio_filename, rows in conversations:
        if rows[0]['transcript_id'] in skip:
            continue

        transcript = load_transcript(audio_filename)
        questions = [row['question'] for row in rows]

        started = time.monotonic()
        if oracle:
            answers = gold_answers(args.answerer, transcript, rows, rng)
        else:
            answers = pipeline.answer_before(transcript, questions, started + args.deadline, answerer)
        latencies.append(time.monotonic() - started)

        backups = pipeline.lexical(transcript, questions, deadline=float('inf'))

        for row, question, answer, backup in zip(rows, questions, answers, backups):
            used = answer or backup
            evidence = {}
            for policy in pipeline.EVIDENCE_POLICIES:
                found = pipeline.evidence_for(transcript, used, question, policy)
                if found.span is None:
                    found = pipeline.evidence_for(transcript, backup, question, policy)
                evidence[policy] = found
            records.append(Record(row, used, evidence, answer is None, transcript))

    stats = _statistics(records, args.threshold, args.policy, verbose=args.verbose)
    stats.conversations = len(latencies)
    print(stats.report())
    _diagnostics(records, args.threshold, args.policy, latencies)
    _compare_policies(records, args.threshold, args.policy)
    _sweep(records, args.threshold, args.policy)
    if args.dump:
        _dump(records, args.dump)
    return 0


def _statistics(records: List[Record], threshold: float, policy: str, verbose: bool = False) -> Statistics:
    stats = Statistics()

    for record in records:
        row, evidence = record.row, record.evidence[policy]
        said_yes = record.p_yes >= threshold
        span = evidence.span if said_yes else None
        gold = gold_evidence(row)
        iou = stats.record(row['question_type'], int(row['label']), int(said_yes), gold, span)

        if verbose:
            mark = 'ok  ' if int(said_yes) == int(row['label']) else 'WRONG'
            tiou = f'tIoU {iou:.3f} ' if gold is not None else ' ' * 11
            print(f'  {mark} {row["question_type"]:<13} p={record.p_yes:.2f} {tiou}'
                  f'{(evidence.method if said_yes else "-"):<8} {row["question"]}')

    return stats


def _exact_spans(records: List[Record], threshold: float, policy: str) -> int:
    exact = 0
    for record in records:
        gold = gold_evidence(record.row)
        span = record.evidence[policy].span
        if record.p_yes >= threshold and gold and span:
            exact += abs(span[0] - gold[0]) < 1e-6 and abs(span[1] - gold[1]) < 1e-6
    return exact


def _diagnostics(records: List[Record], threshold: float, policy: str, latencies: List[float]) -> None:
    said_yes = [record for record in records if record.p_yes >= threshold]
    methods = collections.Counter(record.evidence[policy].method for record in said_yes)
    positives = sum(1 for record in said_yes if gold_evidence(record.row))
    fell_back = sum(record.fell_back for record in records)

    print(f'\nOffline diagnostics (evidence policy: {policy})')
    print(f'  evidence method        {dict(methods)}')
    print(f'  exact spans            {_exact_spans(records, threshold, policy)}/{positives} positives answered yes')
    print(f'  answerer gave nothing  {fell_back} questions (lexical fallback used)')
    if latencies:
        ordered = sorted(latencies)
        p95 = ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]
        print(f'  answerer time          {statistics.mean(latencies):.1f}s mean, {p95:.1f}s p95, '
              f'{max(latencies):.1f}s max per conversation (ASR not included)')


def _compare_policies(records: List[Record], threshold: float, chosen: str) -> None:
    """Same answers, each way of cutting the span: set EVIDENCE_POLICY from this."""
    print('\nEvidence policy (same answers, different span cutting)')
    for policy in pipeline.EVIDENCE_POLICIES:
        stats = _statistics(records, threshold, policy)
        marker = '  <- current' if policy == chosen else ''
        print(f'  {policy:<10} tIoU {stats.mean_tiou:.3f}  exact {_exact_spans(records, threshold, policy):>3}  '
              f'score {stats.final_score:.3f}{marker}')


def _sweep(records: List[Record], chosen: float, policy: str) -> None:
    """Score at every threshold: pick tau from this, not from intuition."""
    print('\nThreshold sweep (yes when p_yes >= tau)')
    best = None
    for tau in sorted(set(THRESHOLD_GRID + [chosen])):
        stats = _statistics(records, tau, policy)
        marker = '  <- current' if tau == chosen else ''
        print(f'  tau {tau:.2f}  accuracy {stats.accuracy:.3f}  tIoU {stats.mean_tiou:.3f}  '
              f'score {stats.final_score:.3f}{marker}')
        if best is None or stats.final_score > best[1]:
            best = (tau, stats.final_score)
    print(f'  best tau {best[0]:.2f} -> {best[1]:.3f}')


def _dump(records: List[Record], path: str) -> None:
    """One JSON line per question: what was asked, cited and cut, and what gold says."""
    with open(path, 'w', encoding='utf-8') as f:
        for record in records:
            row, gold = record.row, gold_evidence(record.row)
            line = {
                'question_id': row['question_id'],
                'type': row['question_type'],
                'label': int(row['label']),
                'question': row['question'],
                'p_yes': round(record.p_yes, 4),
                'model_lines': record.answer.lines,
                'model_quote': record.answer.quote,
                'gold': gold,
                'gold_text': _words(record.transcript, gold),
            }
            for policy, evidence in record.evidence.items():
                line[f'sent_as_{policy}'] = {
                    'method': evidence.method,
                    'span': evidence.span,
                    'text': _words(record.transcript, evidence.span),
                    'tiou': round(temporal_iou(gold, evidence.span), 4) if gold else None,
                }
            f.write(json.dumps(line) + '\n')
    print(f'\nWrote {len(records)} questions to {path}')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('cache', help='Transcribe the supplied conversations once.')

    scoring = commands.add_parser('score', help='Score an answerer on cached transcripts.')
    scoring.add_argument('answerer', help="gold, gold-lines, gold-noisy, lexical or 'module:function'")
    scoring.add_argument('--threshold', type=float, default=pipeline.YES_THRESHOLD)
    scoring.add_argument('--policy', default=pipeline.EVIDENCE_POLICY, choices=pipeline.EVIDENCE_POLICIES,
                         help='Evidence policy the headline numbers use.')
    scoring.add_argument('--deadline', type=float, default=pipeline.ANSWER_DEADLINE_SECONDS,
                         help='Seconds the answerer gets per conversation.')
    scoring.add_argument('--limit', type=int, default=0, help='Only the first N conversations.')
    scoring.add_argument('--skip', default='', help='Comma-separated transcript ids to leave out.')
    scoring.add_argument('--dump', default='', help='Write one JSON line per question here.')
    scoring.add_argument('--verbose', action='store_true')

    args = parser.parse_args()
    if args.command == 'cache':
        return cache()
    if args.answerer == 'lexical':
        args.answerer = 'answering:lexical'
    return score(args)


if __name__ == '__main__':
    sys.exit(main())
