"""Score the split-transcript pipeline: read `large-v3-turbo` text, return `base` timestamps.

    python tools/gpu_eval/hybrid_eval.py answers          # answers over the accurate text
    python tools/gpu_eval/hybrid_eval.py stageb --source results/answers-hybrid.json

Both stages read `inputs.json` (turbo words) and every span it produces is carried onto
`inputs_base.json` (the annotators' word boundaries) before scoring, so the number is
directly comparable with the base-only runs.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent

from dtos import ASRQuestionResponseDto
from pipeline.core import answer_response, retrieval_response
from pipeline.evidence import refine_evidence
from pipeline.hybrid import carry_span, word_map
from pipeline.stage_b import apply_evidence, build_perq_messages, draft_quotes, render_sentences
from utils import temporal_iou

RESULTS = HERE / 'results'
TURBO = {it['id']: it for it in json.loads((HERE / 'inputs.json').read_text())}
BASE = {it['id']: it for it in json.loads((HERE / 'inputs_base.json').read_text())}


def make_backend():
    if os.environ.get('MEDICAL_BACKEND') == 'vllm':
        from pipeline.vllm_backend import VLLMBackend
        return VLLMBackend()
    from pipeline.mlx_backend import MLXBackend
    return MLXBackend()


def metrics(rows):
    positives = [r for r in rows if r['label']]
    ious = [temporal_iou(tuple(r['gold']), tuple(r['candidate']) if r['candidate'] else None)
            for r in positives]
    accuracy = sum(r['answer'] == bool(r['label']) for r in rows) / len(rows)
    mean = sum(ious) / len(ious)
    return {'questions': len(rows), 'correct': sum(r['answer'] == bool(r['label']) for r in rows),
            'accuracy': round(accuracy, 4), 'mean_tiou': round(mean, 4),
            'raw': round(0.4 * accuracy + 0.6 * mean, 4),
            'zero_overlap': sum(i == 0 for i in ious), 'ge0.9': sum(i >= 0.9 for i in ious)}


def cached(directory, key, produce):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (key + '.json')
    if path.is_file():
        record = json.loads(path.read_text())
        return record['raw'], record['seconds']
    raw, seconds = produce()
    path.write_text(json.dumps({'raw': raw, 'seconds': seconds}))
    return raw, seconds


def run(mode, backend, tag, source=None, limit=None):
    rows, generations = [], []
    given = None
    if source:
        given = {r['question_id']: r for r in json.loads(Path(source).read_text())['rows']}
    for index, (cid, turbo) in enumerate(sorted(TURBO.items())):
        if limit and index >= limit:
            break
        base = BASE[cid]
        mapping = word_map(turbo['words'], base['words'])
        questions = [row['question'] for row in turbo['rows']]

        if mode == 'answers':
            raw, seconds = cached(RESULTS / f'gen-{tag}', cid, lambda: (
                backend.complete(turbo['words'], questions), backend.last_generation_seconds))
            fallback = refine_evidence(retrieval_response(turbo['words'], questions, turbo['duration']),
                                       turbo['words'], turbo['envelope'])
            response = refine_evidence(
                answer_response(raw, turbo['words'], questions, turbo['duration'],
                                fallback=fallback, alignment='numeric'),
                turbo['words'], turbo['envelope'])
        else:
            response = ASRQuestionResponseDto(
                answers=[bool(given[row['question_id']]['answer']) for row in turbo['rows']],
                evidence_start=[None] * len(questions), evidence_end=[None] * len(questions))
            drafts = {}
            for i, row in enumerate(turbo['rows']):
                span = given[row['question_id']].get('turbo_span')
                if response.answers[i] and span:
                    response.evidence_start[i], response.evidence_end[i] = span
                    drafts[i + 1] = ' '.join(
                        w['word'].strip() for w in turbo['words']
                        if w['end'] > span[0] + 1e-6 and w['start'] < span[1] - 1e-6)
            transcript = render_sentences(turbo['words'])
            prompts = [(i + 1, build_perq_messages(transcript, row['question'], drafts.get(i + 1, '')))
                       for i, row in enumerate(turbo['rows']) if response.answers[i]]
            key = cid + '-' + hashlib.sha256(json.dumps(prompts).encode()).hexdigest()[:12]
            raw, seconds = cached(RESULTS / f'gen-{tag}', key, lambda: (
                json.dumps(backend.generate_many(prompts, 120, time.monotonic())),
                backend.last_generation_seconds))
            response = apply_evidence({'mode': 'perq', 'outputs': json.loads(raw)}, response,
                                      turbo['words'], turbo['duration'], turbo['envelope'])
        generations.append(seconds)

        for i, row in enumerate(base['rows']):
            turbo_span = (response.evidence_start[i], response.evidence_end[i]) \
                if response.evidence_start[i] is not None else None
            carried = carry_span(turbo_span, turbo['words'], base['words'], mapping)
            rows.append({'conversation': cid, 'question_id': row['question_id'], 'label': int(row['label']),
                         'answer': response.answers[i],
                         'gold': (float(row['evidence_start']), float(row['evidence_end']))
                         if row['evidence_start'] else None,
                         'turbo_span': list(turbo_span) if turbo_span else None,
                         'candidate': list(carried) if carried else None})
        print(f'{cid}: {seconds:.1f}s', flush=True)

    summary = {'mode': mode, 'candidate': metrics(rows),
               'generation_mean': sum(generations) / len(generations), 'generation_max': max(generations)}
    (RESULTS / f'{tag}.json').write_text(json.dumps({**summary, 'rows': rows}, indent=1))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('mode', choices=['answers', 'stageb'])
    parser.add_argument('--source')
    parser.add_argument('--tag', default='hybrid')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    summary = run(args.mode, make_backend(), args.tag, args.source, args.limit)
    label = 'ANSWERS' if args.mode == 'answers' else 'STAGE B'
    print(f'{label} (turbo text, base timestamps):', json.dumps(summary['candidate']),
          f"gen {summary['generation_mean']:.1f}s mean / {summary['generation_max']:.1f}s max")


if __name__ == '__main__':
    main()
