"""Bounded offline Stage-B ablations; no inference or serving changes."""

import hashlib
import json
import os

from rank_bm25 import BM25Okapi

from pipeline.core import _norm
from pipeline.span_examples import select_examples
from pipeline.stage_b import (
    REFINE_SYSTEM_V3,
    _content,
    build_perq_messages,
    render_sentences,
    sentence_ranges,
    sentence_text,
)
from utils import gold_evidence

VARIANTS = ('control', 'no-draft', 'retrieved')
FOLD_COUNT = 3
NEIGHBOR_SENTENCES = 2
MAX_DEMO_WORDS = 160
DRAFT_INSTRUCTION = (
    'For each question you are given a draft quote. Keep it when it already follows the '
    'conventions; otherwise trim it, extend it, or replace it with the right mention. '
)
BARE_QUOTE_INSTRUCTION = (
    'Reply with that one passage, copied exactly from the transcript, and nothing else: '
    'no quotation marks, no line number, no explanation.'
)
GREETING_QUOTES = frozenset({
    ('good',), ('hello',), ('hi',), ('hey',), ('morning',),
    ('good', 'morning'), ('good', 'afternoon'), ('good', 'evening'),
})


def _v3_system():
    return (REFINE_SYSTEM_V3.rsplit('Respond with minified JSON only', 1)[0].rstrip()
            + '\n' + BARE_QUOTE_INSTRUCTION)


def _demonstration(item, row, sentences):
    span = gold_evidence(row)
    if span is None or span[1] <= span[0]:
        return None, {'reason': 'invalid_gold_span'}
    words = item['words']
    support = [i for i, word in enumerate(words)
               if word['end'] > span[0] + 1e-6 and word['start'] < span[1] - 1e-6]
    if not support:
        return None, {'reason': 'no_overlapping_words'}
    first, last = support[0], support[-1]
    quote = sentence_text(words, (first, last))
    if tuple(_norm(quote)) in GREETING_QUOTES:
        return None, {'reason': 'greeting_only_quote'}
    if not _content(row['question']):
        return None, {'reason': 'empty_question_tokens'}
    start = next(k for k, (a, b) in enumerate(sentences) if a <= first <= b)
    end = next(k for k, (a, b) in enumerate(sentences) if a <= last <= b)
    if sentences[end][1] - sentences[start][0] + 1 > MAX_DEMO_WORDS:
        return None, {'reason': 'support_sentences_exceed_word_limit'}
    lo, hi = start, end
    for distance in range(1, NEIGHBOR_SENTENCES + 1):
        if start - distance >= 0:
            candidate = start - distance
            if sentences[hi][1] - sentences[candidate][0] + 1 <= MAX_DEMO_WORDS:
                lo = candidate
        if end + distance < len(sentences):
            candidate = end + distance
            if sentences[candidate][1] - sentences[lo][0] + 1 <= MAX_DEMO_WORDS:
                hi = candidate
    excerpt = render_sentences(words, sentences[lo:hi + 1])
    demo = {'conversation': item['id'], 'question_id': row['question_id'],
            'question': row['question'], 'excerpt': excerpt, 'quote': quote}
    provenance = {'reason': 'usable', 'gold_span': list(span), 'quote_word_range': [first, last],
                  'quote_span': [words[first]['start'], words[last]['end']],
                  'support_sentence_range': [start, end], 'excerpt_sentence_range': [lo, hi],
                  'excerpt_word_range': [sentences[lo][0], sentences[hi][1]]}
    return demo, provenance


class EvidenceVariants:
    def __init__(self, items, variant='control'):
        if variant not in VARIANTS:
            raise ValueError(f'unknown evidence variant: {variant!r}')
        self.variant = variant
        self._items = sorted(items, key=lambda item: item['id'])
        ids = [item['id'] for item in self._items]
        if len(ids) != len(set(ids)):
            raise ValueError('conversation IDs must be unique')
        self.folds = {conversation: index % FOLD_COUNT for index, conversation in enumerate(ids)}
        self._pools = {}
        canonical = [dict(item, rows=sorted(item['rows'], key=lambda row: row['question_id']))
                     for item in self._items]
        self.manifest = {
            'version': 1,
            'variant': variant,
            'config': {
                'fold_count': FOLD_COUNT,
                'fold_allocation': 'sorted_conversation_id_round_robin',
                'base_prompt': os.environ.get('MEDICAL_EVIDENCE_PROMPT', 'v3') if variant == 'control' else 'v3',
                'asr_mode': os.environ.get('MEDICAL_ASR_MODE', ''),
                'demonstrations': 2 if variant == 'retrieved' else 0,
                'retrieval': 'BM25Okapi on _content(question) only',
                'bm25': {'k1': 1.5, 'b': 0.75, 'epsilon': 0.25},
                'tie_break': ['conversation', 'question_id'],
                'exclude': 'entire scored conversation fold',
                'neighbor_sentences_each_side': NEIGHBOR_SENTENCES,
                'max_demo_words': MAX_DEMO_WORDS,
            },
            'folds': dict(self.folds),
            'data_sha256': hashlib.sha256(json.dumps(
                canonical, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
            ).encode()).hexdigest(),
            'curation_provenance': {
                'scope': 'donors only; every input row remains available for scoring',
                'quote_projection': 'contiguous donor words with positive temporal overlap (1e-6 tolerance)',
                'greeting_only_quotes': [list(tokens) for tokens in sorted(GREETING_QUOTES)],
                'prepared_folds': {},
            },
        }

    def _pool(self, fold):
        if fold in self._pools:
            return self._pools[fold]
        demos, decisions = [], []
        # Do not inspect labels or build snippets until the whole target fold is excluded.
        for item in self._items:
            if self.folds[item['id']] == fold:
                continue
            sentences = sentence_ranges(item['words'])
            for row in sorted(item['rows'], key=lambda row: row['question_id']):
                if row.get('label') not in ('1', 1):
                    demo, provenance = None, {'reason': 'not_positive'}
                else:
                    demo, provenance = _demonstration(item, row, sentences)
                decisions.append({'conversation': item['id'], 'question_id': row['question_id'], **provenance})
                if demo is not None:
                    demos.append(demo)
        prepared = self.manifest['curation_provenance']['prepared_folds']
        prepared[str(fold)] = decisions
        self.manifest['curation_provenance']['prepared_folds'] = dict(sorted(prepared.items()))
        count = len({demo['conversation'] for demo in demos})
        if count < 2:
            raise ValueError(f'retrieved requires two usable donor conversations outside scored fold {fold}; found {count}')
        ranker = BM25Okapi([_content(demo['question']) for demo in demos], k1=1.5, b=0.75, epsilon=0.25)
        self._pools[fold] = demos, ranker
        return demos, ranker

    def messages(self, item, question, draft):
        transcript = render_sentences(item['words'])
        if self.variant == 'control':
            return build_perq_messages(transcript, question, draft), []
        if self.variant == 'no-draft':
            messages = build_perq_messages(transcript, question, '')
            messages[0]['content'] = _v3_system().replace(DRAFT_INSTRUCTION, '', 1)
            return messages, []
        demos, ranker = self._pool(self.folds[item['id']])
        chosen = select_examples(demos, ranker, question)
        messages = build_perq_messages(transcript, question, draft, examples=chosen)
        return messages, [demo['question_id'] for demo in chosen]
