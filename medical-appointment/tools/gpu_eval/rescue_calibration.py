"""Measure what a second, calibrated hearing of every "no" is worth, and pick its threshold.

    python tools/gpu_eval/rescue_calibration.py --source results/answers-base27b.json

The scored build answers the 39 training conversations perfectly, so a rescue can only add
false positives there and the benefit would be invisible. Point ``--source`` at a weaker
answer pass instead - the ``compact`` prompt misses exactly two positives, Airomir heard as
"Aromere" and molluscum as "molluscs" - and both sides of the trade become measurable on
real data: how many misses come back, and how many true negatives are spoiled.

Every probability is cached, so the threshold sweep afterwards costs no GPU time.
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

from pipeline.rescue import build_rescue_messages
from pipeline.stage_b import lexical_span, render_sentences, sentence_ranges
from utils import temporal_iou

RESULTS = HERE / 'results'
# What a recovered positive earns once the evidence stage places it, from the deployed build.
EVIDENCE_TIOU = 0.72


def make_backend():
    if os.environ.get('MEDICAL_BACKEND') == 'vllm':
        from pipeline.vllm_backend import VLLMBackend
        return VLLMBackend()
    from pipeline.mlx_backend import MLXBackend
    return MLXBackend()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', required=True, help='cached answer-pass result to rescue')
    parser.add_argument('--inputs', default=os.environ.get('EVAL_INPUTS', 'inputs_base.json'))
    parser.add_argument('--transcript-inputs',
                        help='read the re-ask transcript from another bundle, e.g. the accurate '
                             'turbo text, while labels and spans stay in the scored coordinates')
    parser.add_argument('--tag', default='rescue')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()

    items = json.loads((HERE / args.inputs).read_text())[:args.limit]
    source = {r['question_id']: r for r in json.loads(Path(args.source).read_text())['rows']}
    cache_path = RESULTS / f'{args.tag}-scores.json'
    cache = json.loads(cache_path.read_text()) if cache_path.is_file() else {}
    backend = None
    questions = {}

    alternate = {}
    if args.transcript_inputs:
        alternate = {it['id']: it for it in json.loads((HERE / args.transcript_inputs).read_text())}
    for item in items:
        reading = alternate.get(item['id'], item)
        transcript = render_sentences(reading['words'])
        sentences = sentence_ranges(item['words'])
        pending = []
        for row in item['rows']:
            given = source.get(row['question_id'])
            if given is None or given['answer']:
                continue
            gold = ((float(row['evidence_start']), float(row['evidence_end']))
                    if row['evidence_start'] else None)
            floor = lexical_span(item['words'], row['question'], sentences)
            questions[row['question_id']] = {
                'label': int(row['label']), 'gold': gold,
                'floor_tiou': temporal_iou(gold, floor) if gold and floor else 0.0,
            }
            if row['question_id'] not in cache:
                pending.append((row['question_id'],
                                build_rescue_messages(transcript, row['question'])))
        if pending:
            if backend is None:
                backend = make_backend()
            started = time.monotonic()
            cache.update(backend.score_yes(pending, started))
            print(f'{item["id"]}: {len(pending)} re-asked in {time.monotonic() - started:.1f}s', flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache))

    scored = {q: p for q, p in cache.items() if q in questions and p is not None}
    positives = sorted(p for q, p in scored.items() if questions[q]['label'])
    negatives = sorted(p for q, p in scored.items() if not questions[q]['label'])
    total_q = sum(len(i['rows']) for i in items)
    total_pos = sum(1 for i in items for r in i['rows'] if r['label'] == '1')
    correct = sum(1 for r in source.values() if r['answer'] == bool(r['label']))
    print(f'\nre-asked {len(scored)} "no" answers: {len(positives)} were missed positives, '
          f'{len(negatives)} were true negatives')
    if positives:
        print('  P(yes) on the missed positives:', [round(p, 3) for p in positives])
    for name, values in (('true negatives', negatives),):
        if values:
            band = sum(1 for p in values if 0.2 <= p < 0.5)
            print(f'  P(yes) on {name}: median {values[len(values)//2]:.3f}, '
                  f'{sum(1 for p in values if p >= 0.5)} at or above 0.5, {band} in [0.2, 0.5)')

    print(f'\n{"threshold":>9s} {"recovered":>9s} {"false+":>7s} {"accuracy":>9s} '
          f'{"raw (floor span)":>17s} {"raw (placed span)":>18s}')
    for threshold in (0.15, 0.2, 0.24, 0.3, 0.4, 0.5, 0.7, 0.9, 1.01):
        recovered = [q for q, p in scored.items() if p >= threshold and questions[q]['label']]
        spoiled = [q for q, p in scored.items() if p >= threshold and not questions[q]['label']]
        accuracy = (correct + len(recovered) - len(spoiled)) / total_q
        base_tiou = sum(temporal_iou(tuple(r['gold']), tuple(r['candidate']) if r['candidate'] else None)
                        for r in source.values() if r['label']) / total_pos
        floor = base_tiou + sum(questions[q]['floor_tiou'] for q in recovered) / total_pos
        placed = base_tiou + len(recovered) * EVIDENCE_TIOU / total_pos
        print(f'{threshold:9.2f} {len(recovered):9d} {len(spoiled):7d} {accuracy:9.4f} '
              f'{0.4 * accuracy + 0.6 * floor:17.4f} {0.4 * accuracy + 0.6 * placed:18.4f}')
    print('\nA recovered positive is worth about 3.2x what a false positive costs, so the '
          'lowest threshold whose "false+" column stays small is the one to serve.')


if __name__ == '__main__':
    main()
