"""Choose the evidence span by how likely the question is, given the passage.

    python tools/gpu_eval/likelihood_eval.py --anchor results/stageb-on-own-named-perq-v3.json

Every earlier selector asked the model to *judge* a candidate ("does this establish it?") or
to *compare* candidates, and every one lost to plain generation: verification 0.653,
probability re-ranking 0.404, multiple choice 0.503, reasoning 0.659, ensembling 0.7185,
against 0.718. That is one failure repeated six times, not six failures.

This asks neither. The questions were written *from* the gold passage, so the annotation is a
draw from P(question | passage). For each candidate we therefore score the log-probability of
this exact question conditioned on that passage, with vLLM's prompt logprobs. It is a
likelihood, not an opinion, so there is no position bias and nothing to be talked out of.

The question is identical across a question's candidates, so the summed log-probabilities are
directly comparable without length normalisation. It also penalises all three error shapes at
once: an over-long passage dilutes the question, a too-short one drops what makes it
answerable, and a wrong mention words it differently.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent

import requests

from pipeline.base_asr import sentence_ranges
from pipeline.verify import candidate_spans, passage_text
from utils import temporal_iou

RESULTS = HERE / 'results'
VLLM = os.environ.get('VLLM_URL', 'http://127.0.0.1:18000/v1').rstrip('/')
BASE = VLLM.rsplit('/v1', 1)[0]
MODEL = os.environ.get('MEDICAL_VLLM_MODEL', 'qwen')
PARALLEL = int(os.environ.get('MEDICAL_LIKELIHOOD_PARALLEL', '16'))

PREFIX = 'Passage from the transcript of a doctor-patient consultation:\n"'
MIDDLE = '"\n\nA yes/no question that this passage answers:\n'


def token_count(text):
    response = requests.post(f'{BASE}/tokenize', json={'model': MODEL, 'prompt': text}, timeout=30)
    response.raise_for_status()
    return response.json()['count']


def question_logprob(passage, question, tail_tokens):
    """Total log-probability of the question's tokens, conditioned on the passage."""
    body = {'model': MODEL, 'prompt': PREFIX + passage + MIDDLE + question,
            'max_tokens': 1, 'temperature': 0.0, 'prompt_logprobs': 0}
    response = requests.post(f'{VLLM}/completions', json=body, timeout=120)
    response.raise_for_status()
    entries = response.json()['choices'][0].get('prompt_logprobs') or []
    tail = [e for e in entries[-tail_tokens:] if isinstance(e, dict)]
    total = 0.0
    for entry in tail:
        total += max(item['logprob'] for item in entry.values())
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--anchor', required=True)
    parser.add_argument('--inputs', default=os.environ.get('EVAL_INPUTS', 'inputs_base.json'))
    parser.add_argument('--tag', default='likelihood')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()

    import concurrent.futures

    items = {it['id']: it for it in json.loads((HERE / args.inputs).read_text())}
    # The anchor file carries decisions and spans; the question text lives in the inputs.
    questions = {r['question_id']: r['question'] for it in items.values() for r in it['rows']}
    rows = [r for r in json.loads(Path(args.anchor).read_text())['rows'] if r['label']]
    if args.limit:
        keep = sorted({r['conversation'] for r in rows})[:args.limit]
        rows = [r for r in rows if r['conversation'] in keep]
    cache_path = RESULTS / f'{args.tag}-scores.json'
    cache = json.loads(cache_path.read_text()) if cache_path.is_file() else {}

    anchor_total = chosen_total = oracle_total = 0.0
    started = time.monotonic()
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=PARALLEL)
    for index, row in enumerate(rows, 1):
        item = items[row['conversation']]
        words = item['words']
        gold = tuple(row['gold'])
        anchor = tuple(row['candidate']) if row['candidate'] else None
        anchor_total += temporal_iou(gold, anchor)
        spans = candidate_spans(words, anchor, sentence_ranges(words))
        if not spans:
            chosen_total += temporal_iou(gold, anchor)
            oracle_total += temporal_iou(gold, anchor)
            continue
        key = row['question_id']
        if key not in cache:
            tail = token_count(questions[row['question_id']])
            texts = [passage_text(words, s) for s in spans]
            question = questions[row['question_id']]
            scores = list(pool.map(lambda t: question_logprob(t, question, tail), texts))
            cache[key] = {'spans': [list(s) for s in spans], 'scores': scores}
            if index % 20 == 0:
                print(f'{index}/{len(rows)} scored, {time.monotonic() - started:.0f}s', flush=True)
        record = cache[key]
        best = max(range(len(record['scores'])), key=lambda i: record['scores'][i])
        first, last = record['spans'][best]
        chosen_total += temporal_iou(gold, (words[first]['start'], words[last]['end']))
        oracle_total += max(temporal_iou(gold, (words[a]['start'], words[b]['end']))
                            for a, b in record['spans'])
    RESULTS.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache))

    n = len(rows)
    print(f'\npositives scored: {n}   ({time.monotonic() - started:.0f}s)')
    print(f'  anchor (the deployed spans)        {anchor_total / n:.4f} tIoU')
    print(f'  chosen by question likelihood      {chosen_total / n:.4f} tIoU')
    print(f'  oracle over the same candidates    {oracle_total / n:.4f} tIoU')
    for name, value in (('anchor', anchor_total), ('likelihood', chosen_total)):
        print(f'  raw at 1.000 accuracy, {name:11s} {0.4 + 0.6 * value / n:.4f}')


if __name__ == '__main__':
    main()
