"""Score the candidate verifier over the cached transcripts, then sweep its threshold offline.

    python tools/gpu_eval/verify_eval.py --anchor results/stageb-on-own-base27b-perq-v3.json

Every candidate's P(yes) is cached, so the sweep and any change to the selection rule cost
no GPU time at all: rerun with the same tag and only the arithmetic repeats.
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

from pipeline.base_asr import sentence_ranges
from pipeline.verify import build_verify_messages, candidate_spans, passage_text, select_span
from utils import temporal_iou

RESULTS = HERE / 'results'


def make_backend():
    if os.environ.get('MEDICAL_BACKEND') == 'vllm':
        from pipeline.vllm_backend import VLLMBackend
        return VLLMBackend()
    from pipeline.mlx_backend import MLXBackend
    return MLXBackend()


def score(rows, spans_by_question):
    ious = [temporal_iou(tuple(r['gold']), spans_by_question.get(r['question_id']))
            for r in rows if r['label']]
    mean = sum(ious) / len(ious)
    return {'mean_tiou': round(mean, 4), 'raw': round(0.4 * 0.9949 + 0.6 * mean, 4),
            'zero_overlap': sum(i == 0 for i in ious), 'ge0.9': sum(i >= 0.9 for i in ious)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--anchor', required=True, help='stage-B result whose spans are the anchors')
    parser.add_argument('--inputs', default=os.environ.get('EVAL_INPUTS', 'inputs_base.json'))
    parser.add_argument('--tag', default='verify')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()

    items = json.loads((HERE / args.inputs).read_text())[:args.limit]
    anchor_rows = json.loads(Path(args.anchor).read_text())['rows']
    anchors = {r['question_id']: r for r in anchor_rows}
    cache_path = RESULTS / f'{args.tag}-scores.json'
    cache = json.loads(cache_path.read_text()) if cache_path.is_file() else {}
    backend = make_backend() if any(
        r['question_id'] not in cache for it in items for r in it['rows']
        if anchors.get(r['question_id'], {}).get('answer')
    ) else None

    for item in items:
        words = item['words']
        sentences = sentence_ranges(words)
        pending, meta = [], {}
        for row in item['rows']:
            anchor = anchors.get(row['question_id'])
            if not anchor or not anchor['answer'] or anchor['question_id'] in cache:
                continue
            span = tuple(anchor['candidate']) if anchor['candidate'] else None
            spans = candidate_spans(words, span, sentences)
            meta[row['question_id']] = spans
            for index, candidate in enumerate(spans):
                pending.append(((row['question_id'], index),
                                build_verify_messages(words, candidate, row['question'])))
        if pending:
            started = time.monotonic()
            scores = backend.score_yes(pending, started)
            for question_id, spans in meta.items():
                cache[question_id] = [
                    {'span': list(span), 'words': span[1] - span[0] + 1,
                     'p': scores.get((question_id, index)),
                     'seconds': p_text(words, span)}
                    for index, span in enumerate(spans)
                ]
            print(f'{item["id"]}: {len(pending)} candidates in {time.monotonic() - started:.1f}s', flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache))

    times = {it['id']: it['words'] for it in items}
    conversation_of = {r['question_id']: r['conversation'] for r in anchor_rows}
    rows = [r for it in items for r in it['rows']]
    rows = [{'question_id': r['question_id'], 'label': int(r['label']),
             'gold': (float(r['evidence_start']), float(r['evidence_end'])) if r['evidence_start'] else None}
            for r in rows]
    base = {r['question_id']: tuple(anchors[r['question_id']]['candidate'])
            for r in rows if anchors.get(r['question_id'], {}).get('candidate')}
    print('anchor            ', score(rows, base))
    best = None
    for threshold in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        chosen = dict(base)
        for question_id, candidates in cache.items():
            if question_id not in base:
                continue
            words = times[conversation_of[question_id]]
            pairs = [((c['span'][0], c['span'][1]), c['p']) for c in candidates]
            anchor_words = None
            picked = select_span(pairs, anchor=anchor_words, threshold=threshold)
            if picked is not None:
                chosen[question_id] = (words[picked[0]]['start'], words[picked[1]]['end'])
        result = score(rows, chosen)
        print(f'threshold {threshold:<5}', result)
        best = max(best or result, result, key=lambda r: r['mean_tiou'])
    print('best threshold result', best)


def p_text(words, span):
    return [round(words[span[0]]['start'], 3), round(words[span[1]]['end'], 3)]


if __name__ == '__main__':
    main()
