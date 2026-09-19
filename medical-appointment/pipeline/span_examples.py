"""Frozen public-training demonstrations; known question sets exclude their entire fold."""

import json
from functools import lru_cache
from pathlib import Path

from rank_bm25 import BM25Okapi


@lru_cache(maxsize=1)
def _bank():
    data = json.loads(Path(__file__).with_suffix('.json').read_text())
    question_folds = {tuple(sorted(item['questions'])): data['folds'][item['conversation']]
                      for item in data['question_sets']}
    return data, question_folds


@lru_cache(maxsize=4)
def _pool(fold):
    from pipeline.stage_b import _content

    data, _ = _bank()
    demos = [demo for demo in data['examples'] if data['folds'][demo['conversation']] != fold]
    demos.sort(key=lambda demo: (demo['conversation'], demo['question_id']))
    if len({demo['conversation'] for demo in demos}) < 2:
        raise ValueError('retrieved requires two usable donor conversations')
    ranker = BM25Okapi([_content(demo['question']) for demo in demos], k1=1.5, b=0.75, epsilon=0.25)
    return demos, ranker


def select_examples(demos, ranker, question):
    from pipeline.stage_b import _content

    scores = ranker.get_scores(_content(question))
    ranked = sorted(range(len(demos)), key=lambda i: (
        -float(scores[i]), demos[i]['conversation'], demos[i]['question_id'],
    ))
    chosen, conversations = [], set()
    for index in ranked:
        demo = demos[index]
        if demo['conversation'] in conversations:
            continue
        chosen.append(demo)
        conversations.add(demo['conversation'])
        if len(chosen) == 2:
            break
    return chosen


def examples_for(questions, question):
    """Match all questions, not ASR; unseen sets use the full public-training bank, not OOF."""
    _, question_folds = _bank()
    fold = question_folds.get(tuple(sorted(questions)))
    return select_examples(*_pool(fold), question)


def retrieved_system(demos):
    from pipeline.stage_b import CONVENTIONS, REFINE_SYSTEM_V3

    examples = 'Examples from other consultations (excerpt, then question -> quote):\n' + '\n\n'.join(
        f'{demo["excerpt"]}\n{demo["question"]} -> {json.dumps(demo["quote"], ensure_ascii=False)}'
        for demo in demos
    )
    fixed_examples = '\nExample transcript:\n' + CONVENTIONS.split('\nExample transcript:\n', 1)[1]
    system = (REFINE_SYSTEM_V3.rsplit('Respond with minified JSON only', 1)[0].rstrip()
              + '\nReply with that one passage, copied exactly from the transcript, and nothing else: '
              'no quotation marks, no line number, no explanation.')
    return system.replace(fixed_examples, '\n' + examples + '\n', 1)
