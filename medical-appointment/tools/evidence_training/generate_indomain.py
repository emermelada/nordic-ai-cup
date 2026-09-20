"""Mine extra (question, evidence span) pairs from the real transcripts.

The span is chosen first, on the real ASR word grid, so its timestamps are exact
rather than predicted. A model writes the question that span answers, and two
checks keep only spans that are both sufficient (the question is answerable from
the span alone) and unique (removing it makes the question unanswerable), which is
the annotation convention this task scores against.

Every generated row carries its source conversation, so fold discipline is the
caller's to keep: a row may only ever train a fold whose test set excludes it.
"""

import argparse
import concurrent.futures
import json
import os
import random
import re
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.evidence_training.data import render_words, tiou, validate_records, write_json

STYLE = """Did the patient attend for an annual asthma follow-up?
Is the heart examination without abnormal findings?
Was the patient listened to with a stethoscope?
Were the lungs found to be normal on auscultation?
Will the current treatment continue unchanged?"""

WRITE_SYSTEM = (
    'You write one yes/no question about a medical consultation, in the clinical style of '
    'these examples:\n' + STYLE + '\n\n'
    'The question must be answered YES by the quoted passage alone, must ask about what the '
    'passage actually states, and must not quote its wording. Reply with the question only.'
)
SUFFICIENT_SYSTEM = (
    'Answer yes or no. Decide only from the passage given. Reply with one word.'
)


def sentences_of(words):
    """Word ranges that end where speech punctuation ends, as the annotators read them."""
    spans, start = [], 0
    for index, word in enumerate(words):
        if re.search(r'[.?!]"?$', word['word'].strip()):
            if index >= start:
                spans.append((start, index))
            start = index + 1
    if start < len(words):
        spans.append((start, len(words) - 1))
    return spans


def candidates(words, minimum=4, maximum=20):
    """Sentences, and adjacent pairs, whose length matches the gold span distribution."""
    spans = sentences_of(words)
    found = []
    for i, (a, b) in enumerate(spans):
        if minimum <= b - a + 1 <= maximum:
            found.append((a, b))
        if i + 1 < len(spans):
            c, d = spans[i + 1]
            if minimum <= d - a + 1 <= maximum:
                found.append((a, d))
    return found


def quote_of(words, span):
    return ' '.join(w['word'].strip() for w in words[span[0]:span[1] + 1])


class Model:
    def __init__(self, url, name, timeout=60):
        self.url, self.name, self.timeout = url.rstrip('/'), name, timeout

    def ask(self, system, user, max_tokens=60, temperature=0.0):
        payload = {'model': self.name, 'max_tokens': max_tokens, 'temperature': temperature,
                   'messages': [{'role': 'system', 'content': system},
                                {'role': 'user', 'content': user}]}
        response = requests.post(f'{self.url}/chat/completions', json=payload, timeout=self.timeout)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content'].strip()


def build_row(model, record, span, index):
    """One verified example, or None when a check rejects it."""
    words = record['words']
    quote = quote_of(words, span)
    question = model.ask(WRITE_SYSTEM, f'PASSAGE\n{quote}\n\nQUESTION:')
    question = question.strip().strip('"').split('\n')[0].strip()
    if not question.endswith('?') or len(question.split()) < 4:
        return None, 'unusable_question'
    # Sufficient: the passage alone has to answer it yes.
    if not model.ask(SUFFICIENT_SYSTEM, f'PASSAGE\n{quote}\n\nQUESTION: {question}\nANSWER:',
                     max_tokens=4).lower().startswith('yes'):
        return None, 'passage_does_not_answer_it'
    # Unique: with the passage removed, the transcript must no longer answer it yes,
    # or the annotators' chosen occurrence would be ambiguous.
    remainder = ' '.join(w['word'].strip() for i, w in enumerate(words)
                         if not span[0] <= i <= span[1])
    if model.ask(SUFFICIENT_SYSTEM, f'PASSAGE\n{remainder}\n\nQUESTION: {question}\nANSWER:',
                 max_tokens=4).lower().startswith('yes'):
        return None, 'answerable_elsewhere_in_the_transcript'
    context, ranges = render_words(words)
    a, b = span
    return {'id': f'{record["conversation"]}_gen_q{index:03d}',
            'conversation': record['conversation'], 'question': question, 'label': True,
            'context': context, 'word_chars': ranges, 'words': words,
            'gold_words': [a, b], 'gold_chars': [ranges[a][0], ranges[b][1]],
            'gold': [words[a]['start'], words[b]['end']], 'mapping_tiou': 1.0,
            'baseline_answer': True, 'baseline_span': [words[a]['start'], words[b]['end']],
            'generated': True}, None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--url', default=os.environ.get('VLLM_URL', 'http://127.0.0.1:18000/v1'))
    parser.add_argument('--model', default='qwen')
    parser.add_argument('--per-conversation', type=int, default=40)
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('--seed', type=int, default=17)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('Output exists; choose a fresh path')
    bundle = json.loads(args.data.read_text())
    first = {}
    for record in bundle['records']:
        first.setdefault(record['conversation'], record)
    model = Model(args.url, args.model)
    jobs = []
    rng = random.Random(args.seed)
    for conversation, record in sorted(first.items()):
        spans = candidates(record['words'])
        # Never regenerate a span the annotators already labelled in this conversation.
        taken = {tuple(r['gold_words']) for r in bundle['records']
                 if r['conversation'] == conversation and r['label']}
        spans = [s for s in spans if s not in taken]
        rng.shuffle(spans)
        for index, span in enumerate(spans[:args.per_conversation]):
            jobs.append((record, span, index))
    print(json.dumps({'conversations': len(first), 'candidate_spans': len(jobs)}), flush=True)
    rows, rejected = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(build_row, model, record, span, index): (record['conversation'], span)
                   for record, span, index in jobs}
        for done, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            conversation, span = futures[future]
            try:
                row, reason = future.result()
            except Exception as error:
                row, reason = None, f'{type(error).__name__}'
            if row:
                rows.append(row)
            else:
                rejected.append({'conversation': conversation, 'span': list(span), 'reason': reason})
            if done % 100 == 0:
                print(json.dumps({'checked': done, 'kept': len(rows)}), flush=True)
    rows.sort(key=lambda r: r['id'])
    validate_records(rows + [r for r in bundle['records']])
    from collections import Counter
    summary = {'generated': len(rows), 'rejected': len(rejected),
               'rejection_counts': dict(Counter(r['reason'] for r in rejected)),
               'by_conversation': dict(Counter(r['conversation'] for r in rows)),
               'source_data_sha256': __import__('hashlib').sha256(args.data.read_bytes()).hexdigest(),
               'scope': 'Extra training rows only; fold discipline is enforced by the trainer'}
    write_json(args.output, {'records': rows, 'summary': summary, 'rejected': rejected})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
