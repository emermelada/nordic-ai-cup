"""Score cached training transcripts without ASR: answer pass, evidence pass, or both.

    python tools/gpu_eval/eval.py answers --model-id REPO@REVISION
    python tools/gpu_eval/eval.py stageb --source results/answers.json --model-id REPO@REVISION
    python tools/gpu_eval/eval.py stageb --source results/answers.json --model-id REPO@REVISION --replay
    python tools/gpu_eval/eval.py stageb --source results/answers.json --model-id REPO@REVISION --evidence-mode locate

Legacy inputs_base.json requires --asr-mode base (or MEDICAL_ASR_MODE=base).
Replay reads fingerprinted generation caches only; legacy ANSWER results may be
supplied explicitly with --source, but legacy Stage-B generations are not reused.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent

from dtos import ASRQuestionResponseDto
from pipeline.core import _question_id, _results, answer_response, retrieval_response, sanitize_response
from pipeline.evidence import refine_evidence
from pipeline.stage_b import (
    align_bridged, apply_evidence, build_perq_messages, build_refine_messages,
    clean_window_output, draft_quotes, parse_refine_output, render_sentences,
)
from utils import evidence_interval, gold_evidence, temporal_iou, validate_response

ITEMS = json.loads((HERE / os.environ.get('EVAL_INPUTS', 'inputs.json')).read_text())
BASELINE = json.loads((HERE / 'baseline_9b.json').read_text())
RESULTS = HERE / 'results'
CACHE_VERSION = 2
# Conversations the annotators'-convention prompt quotes examples from; scored separately.
EXAMPLE_SOURCES = {'sample_4', 'sample_5', 'sample_17', 'sample_18', 'sample_19', 'sample_20', 'sample_71'}


def report(rows, field='candidate'):
    held = [r for r in rows if r['conversation'] not in EXAMPLE_SOURCES]
    result = {'all': metrics(rows, field), 'without_example_sources': metrics(held, field)}
    folds = sorted({r['fold'] for r in rows if r.get('fold') is not None})
    if folds:
        result['folds'] = {str(fold): metrics([r for r in rows if r.get('fold') == fold], field)
                           for fold in folds}
    return result


def make_backend():
    if os.environ.get('MEDICAL_BACKEND') == 'vllm':
        from pipeline.vllm_backend import VLLMBackend
        return VLLMBackend()
    from pipeline.mlx_backend import MLXBackend
    return MLXBackend()


def metrics(rows, field='candidate'):
    positives = [r for r in rows if r['label']]
    ious = [temporal_iou(tuple(r['gold']), tuple(r[field]) if r[field] else None) for r in positives]
    correct = sum(r['answer'] == bool(r['label']) for r in rows)
    accuracy = correct / len(rows) if rows else 0.0
    mean = sum(ious) / len(ious) if ious else 0.0
    return {'questions': len(rows), 'correct': correct,
            'accuracy': round(accuracy, 4), 'mean_tiou': round(mean, 4), 'raw': round(0.4 * accuracy + 0.6 * mean, 4),
            'zero_overlap': sum(i == 0 for i in ious), 'ge0.9': sum(i >= 0.9 for i in ious)}


def bootstrap_gain(rows, base_field='baseline'):
    by = {}
    for r in rows:
        if r['label']:
            by.setdefault(r['conversation'], []).append(
                (temporal_iou(tuple(r['gold']), tuple(r[base_field]) if r[base_field] else None),
                 temporal_iou(tuple(r['gold']), tuple(r['candidate']) if r['candidate'] else None)))
    convs = sorted(by)
    if not convs:
        return 0.0, 0.0, 0.0
    def gain(sample):
        pairs = [p for c in sample for p in by[c]]
        return sum(c - b for b, c in pairs) / len(pairs)
    rng = random.Random(0)
    boots = sorted(gain([rng.choice(convs) for _ in convs]) for _ in range(2000))
    return round(gain(convs), 4), round(boots[50], 4), round(boots[1949], 4)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


class CacheError(ValueError):
    pass


def cached(cache_dir, key, produce, *, manifest=None, replay=False):
    if replay and manifest is None:
        raise CacheError('Replay requires a fingerprinted cache, not a legacy generation')
    digest = fingerprint(manifest) if manifest is not None else None
    path = Path(cache_dir) / (key + (f'-{digest}' if digest else '') + '.json')
    if path.is_file():
        try:
            rec = json.loads(path.read_text())
            if manifest is not None and (rec.get('version') != CACHE_VERSION
                                        or rec.get('fingerprint') != digest or rec.get('manifest') != manifest):
                raise ValueError('fingerprint mismatch')
            if not isinstance(rec['raw'], str) or not math.isfinite(rec['seconds']) or rec['seconds'] < 0:
                raise ValueError('invalid generation or timing')
            return rec['raw'], rec['seconds']
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            raise CacheError(f'Incompatible generation cache {path}: {exc}') from exc
    if replay:
        raise CacheError(f'Replay cache miss: {path}')
    raw, seconds = produce()
    rec = {'raw': raw, 'seconds': seconds}
    if manifest is not None:
        rec.update(version=CACHE_VERSION, fingerprint=digest, manifest=manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(rec, handle, allow_nan=False)
    return raw, seconds


def timed(produce):
    started = time.monotonic()
    raw = produce()
    return raw, time.monotonic() - started


def generation_settings(stage, mode=None, *, model_id=None, backend=None):
    from pipeline import mlx_backend as mlx

    identity = model_id or os.environ.get('EVAL_MODEL_ID')
    if not identity or not identity.strip():
        raise ValueError('--model-id (or EVAL_MODEL_ID) must identify the full model revision; a served alias is not provenance')
    engine = os.environ.get('MEDICAL_BACKEND', 'mlx')
    max_tokens = 1200 if stage == 'answers' else mlx.EVIDENCE_MAX_TOKENS[mode]
    result = {'backend': engine, 'model_id': identity, 'temperature': 0.0,
              'requested_max_tokens': max_tokens, 'max_tokens': max_tokens}
    if engine == 'vllm':
        from pipeline import vllm_backend as vllm

        if stage == 'answers' and vllm.VLLM_ANSWER_URL != vllm.VLLM_URL:
            raise ValueError('Separate answer endpoints need a separate run with their own VLLM_URL and --model-id')
        result.update(endpoint=vllm.VLLM_ANSWER_URL if stage == 'answers' else vllm.VLLM_URL,
                      served_alias=vllm.VLLM_MODEL or None, timeout=vllm.LLM_REQUEST_TIMEOUT,
                      chat_template_kwargs={'enable_thinking': stage != 'answers' and vllm.EVIDENCE_THINKING})
        if mode in ('perq', 'locate'):
            result['parallel'] = vllm.EVIDENCE_PARALLEL
            if vllm.EVIDENCE_THINKING:
                result['max_tokens'] = max(result['max_tokens'], vllm.THINKING_MAX_TOKENS)
    else:
        result.update(served_alias=mlx.LLM_MODEL if stage == 'answers' else mlx.EVIDENCE_MODEL,
                      adapter=None if stage == 'answers' else mlx.EVIDENCE_ADAPTER,
                      chat_template_kwargs={'enable_thinking': False, 'reasoning_effort': 'low'})
    if stage == 'answers':
        result['answer_prompt'] = getattr(backend, 'prompt', mlx.DEFAULT_PROMPT)
    if mode in ('perq', 'locate'):
        result['budget_seconds'] = mlx.EVIDENCE_BUDGET_SECONDS
    return result


def timing_policy(item, asr_mode=None):
    transcript = item.get('transcript', {})
    mode = asr_mode or os.environ.get('MEDICAL_ASR_MODE') or item.get('asr_mode') or transcript.get('asr_mode')
    exact = item.get('exact_timestamps', transcript.get('exact_timestamps'))
    if exact is not None and not isinstance(exact, bool):
        raise ValueError('exact_timestamps must be boolean')
    if mode not in (None, 'base', 'turbo'):
        raise ValueError(f'Unknown ASR mode: {mode}')
    if exact is not None and mode is not None and exact != (mode == 'base'):
        raise ValueError(f'ASR mode {mode} conflicts with exact_timestamps={exact}')
    exact = (mode == 'base') if exact is None else exact
    mode = mode or ('base' if exact else 'turbo')
    return {'asr_mode': mode, 'exact_timestamps': exact, 'extend_replies': not exact}


@contextmanager
def timing_context(item, asr_mode=None):
    policy = timing_policy(item, asr_mode)
    previous = os.environ.get('MEDICAL_ASR_MODE')
    os.environ['MEDICAL_ASR_MODE'] = policy['asr_mode']
    try:
        yield policy
    finally:
        if previous is None:
            os.environ.pop('MEDICAL_ASR_MODE', None)
        else:
            os.environ['MEDICAL_ASR_MODE'] = previous


def envelope_for(item):
    return item.get('envelope', item.get('energy_db', item.get('transcript', {}).get('energy_db')))


def primary_response(item, raw, policy):
    questions = [r['question'] for r in item['rows']]
    words, duration, envelope = item['words'], item['duration'], envelope_for(item)
    fallback = refine_evidence(retrieval_response(words, questions, duration), words, envelope,
                               extend_replies=policy['extend_replies'])
    response = answer_response(raw, words, questions, duration, fallback=fallback, alignment='numeric')
    return refine_evidence(response, words, envelope, extend_replies=policy['extend_replies'])


def answer_messages(item, prompt):
    from pipeline.core import build_messages
    from pipeline.evidence import (
        build_compact_messages, build_focused_messages, build_minimal_messages, build_named_messages,
    )

    builder = {'legacy': build_messages, 'focused': build_focused_messages,
               'compact': build_compact_messages, 'minimal': build_minimal_messages,
               'named': build_named_messages}[prompt]
    return builder(item['words'], [r['question'] for r in item['rows']])


def request_manifest(item, messages, settings, policy, *, stage, mode=None, variant=None, donors=None):
    # Labels and gold spans belong to scoring, never to a generation cache key.
    return {'version': CACHE_VERSION, 'stage': stage, 'mode': mode, 'variant': variant,
            'generation': settings, 'timing_policy': policy, 'words': item['words'],
            'messages': messages, 'donors': donors or {}}


def load_source(path):
    rows = json.loads(Path(path).read_text())['rows']
    result = {r['question_id']: r for r in rows}
    if len(result) != len(rows):
        raise ValueError(f'Duplicate question IDs in source: {path}')
    return result


def source_raw(item, src):
    if src is None:
        raw = item.get('primary', {}).get('raw')
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError('These inputs carry no primary raw; use --source with an ANSWER result containing raw')
        return raw
    raws = []
    for row in item['rows']:
        saved = src.get(row['question_id'])
        if saved is None or not isinstance(saved.get('raw'), str) or not saved['raw'].strip():
            raise ValueError(f'--source needs original ANSWER raw for {row["question_id"]}; reconstructed spans are not drafts')
        if saved.get('conversation', item['id']) != item['id'] or not isinstance(saved.get('answer'), bool):
            raise ValueError(f'Invalid source answer for {row["question_id"]}')
        raws.append(saved['raw'])
    if len(set(raws)) != 1:
        raise ValueError(f'Inconsistent source raw within {item["id"]}')
    return raws[0]


def evidence_completeness(frame, response, words, duration=None):
    count = len(response.answers)
    expected = {i + 1 for i, yes in enumerate(response.answers) if yes}
    present = set()
    if frame.get('mode') == 'locate':
        from pipeline.locate import word_spans

        spans = word_spans(frame, words, count, duration)
        present = set(spans)
        valid = {qid for qid in expected & present if spans[qid] is not None}
    else:
        if frame.get('mode') == 'perq':
            outputs = frame.get('outputs')
            outputs = outputs if isinstance(outputs, dict) else {}
            quotes = {}
            for key, value in outputs.items():
                qid = _question_id(key, count)
                if qid is not None:
                    present.add(qid)
                    quotes[qid] = clean_window_output(value)
        else:
            quotes = parse_refine_output(frame.get('raw'), count)
            for entry in _results(frame.get('raw')):
                if isinstance(entry, dict):
                    qid = _question_id(entry.get('q'), count)
                    if qid is not None:
                        present.add(qid)
        valid = {qid for qid in expected & present if quotes.get(qid) and align_bridged(words, quotes[qid]) is not None}
    missing, invalid = expected - present, (expected & present) - valid
    return {'expected': sorted(expected), 'valid': sorted(valid), 'missing': sorted(missing),
            'invalid': sorted(invalid), 'unexpected': sorted(present - expected),
            'complete': not missing and not invalid and not frame.get('skipped') and not frame.get('error'),
            'skipped': frame.get('skipped'), 'error': frame.get('error')}


def save_result(tag, summary, rows):
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f'{tag}.json'
    if path.exists():
        path = RESULTS / f'{tag}-{time.time_ns()}.json'
    summary['result_path'] = str(path)
    with path.open('x') as handle:
        json.dump({**summary, 'rows': rows}, handle, indent=1, allow_nan=False)
    print(f'Result: {path}', flush=True)
    return summary


def run_answers(backend, tag, limit=None, *, replay=False, model_id=None, asr_mode=None):
    rows, gens, requests = [], [], []
    settings = generation_settings('answers', model_id=model_id, backend=backend)
    for item in ITEMS[:limit]:
        with timing_context(item, asr_mode) as policy:
            messages = answer_messages(item, settings['answer_prompt'])
            manifest = request_manifest(item, messages, settings, policy, stage='answers')
            raw, secs = cached(RESULTS / f'gen-v{CACHE_VERSION}-{tag}', item['id'],
                               lambda: timed(lambda: backend.complete(item['words'], [r['question'] for r in item['rows']])),
                               manifest=manifest, replay=replay)
            gens.append(secs)
            requests.append({'conversation': item['id'], 'fingerprint': fingerprint(manifest), 'timing_policy': policy})
            response = primary_response(item, raw, policy)
            validate_response(response, len(item['rows']))
            base = ASRQuestionResponseDto(**BASELINE[item['id']]['response'])
            for i, row in enumerate(item['rows']):
                rows.append({'conversation': item['id'], 'question_id': row['question_id'], 'label': int(row['label']),
                             'answer': response.answers[i], 'gold': gold_evidence(row),
                             'baseline': evidence_interval(base.evidence_start[i], base.evidence_end[i])
                             if not policy['exact_timestamps'] else None,
                             'candidate': evidence_interval(response.evidence_start[i], response.evidence_end[i]),
                             'raw': raw})
        print(f'{item["id"]}: {secs:.1f}s', flush=True)
    summary = {'mode': 'answers', 'candidate': metrics(rows), 'report': report(rows),
               'generation_mean': sum(gens) / len(gens) if gens else 0.0, 'generation_max': max(gens, default=0.0),
               'manifest': {'version': CACHE_VERSION, 'replay': replay, 'generation': settings, 'requests': requests,
                            'cache_directory': str(RESULTS / f'gen-v{CACHE_VERSION}-{tag}'),
                            'inputs_sha256': fingerprint(ITEMS)}}
    return save_result(tag, summary, rows)


def generate_located(backend, words, questions, answers, settings):
    from pipeline.locate import locate_evidence

    started = time.monotonic()
    calls = []

    def generate(phase, prompts):
        outputs, seconds = timed(lambda: backend.generate_many(prompts, settings['requested_max_tokens'], started))
        calls.append({'phase': phase, 'messages': prompts, 'seconds': seconds, 'outputs': outputs})
        return outputs

    frame = locate_evidence(words, questions, answers, generate,
                            deadline=started + settings['budget_seconds'])
    frame['calls'] = calls
    return json.dumps(frame), time.monotonic() - started


def run_stageb(backend, tag, source=None, limit=None, mode='refine', *, replay=False, model_id=None,
               asr_mode=None, variant='control'):
    if variant != 'control' and mode != 'perq':
        raise ValueError('--variant no-draft/retrieved is only supported with --evidence-mode perq')
    rows, gens, requests, completeness, used_donors = [], [], [], [], {}
    settings = generation_settings('stageb', mode, model_id=model_id)
    src = load_source(source) if source else None
    items = ITEMS[:limit]
    raws = {item['id']: source_raw(item, src) for item in items}
    variants = None
    if mode in ('perq', 'locate') and items:
        from tools.gpu_eval.evidence_variants import EvidenceVariants
        with timing_context(items[0], asr_mode):
            variants = EvidenceVariants(ITEMS, variant=variant)
    for item in items:
        with timing_context(item, asr_mode) as policy:
            questions = [r['question'] for r in item['rows']]
            raw_answer = raws[item['id']]
            drafts = draft_quotes(raw_answer, len(questions))
            if src is None:
                baseline = ASRQuestionResponseDto(**BASELINE[item['id']]['response'])
            else:
                baseline = primary_response(item, raw_answer, policy)
                fixed_answers = [src[r['question_id']]['answer'] for r in item['rows']]
                if baseline.answers != fixed_answers:
                    raise ValueError(f'Source answers disagree with parsed raw for {item["id"]}; cannot preserve a runtime-faithful baseline')
                baseline = sanitize_response(baseline, len(questions), item['duration'])
            donors = {}
            expected = [i + 1 for i, yes in enumerate(baseline.answers) if yes]
            if mode == 'perq':
                prompts = []
                for i, row in enumerate(item['rows']):
                    if baseline.answers[i]:
                        messages, donor_ids = variants.messages(item, row['question'], drafts.get(i + 1, ''))
                        prompts.append([i + 1, messages])
                        donors[row['question_id']] = donor_ids
                messages = prompts
                produce = lambda: timed(lambda: json.dumps(backend.generate_many(
                    prompts, settings['requested_max_tokens'], time.monotonic())))
            elif mode == 'locate':
                from pipeline.locate import build_locate_messages

                transcript = render_sentences(item['words'])
                messages = [[i + 1, build_locate_messages(transcript, question)]
                            for i, question in enumerate(questions) if baseline.answers[i]]
                produce = lambda: generate_located(backend, item['words'], questions, baseline.answers, settings)
            else:
                messages = build_refine_messages(item['words'], questions, baseline.answers, drafts)
                from pipeline.mlx_backend import EVIDENCE_ADAPTER, EVIDENCE_MODEL
                produce = lambda: timed(lambda: backend.generate_messages(
                    messages, settings['requested_max_tokens'], EVIDENCE_MODEL, EVIDENCE_ADAPTER))
            manifest = request_manifest(item, messages, settings, policy, stage='stageb', mode=mode,
                                        variant=variant, donors=donors)
            if mode == 'locate':
                from pipeline.locate import CONTEXT_SENTENCES, build_bounds_messages

                # Cache the complete chain, including deadline skips; replay must not launch a previously skipped phase.
                manifest['context_sentences'] = CONTEXT_SENTENCES
                manifest['bounds_full_context_templates'] = [
                    [i + 1, build_bounds_messages(item['words'], question, (0, len(item['words']) - 1))]
                    for i, question in enumerate(questions) if baseline.answers[i]
                ]
                manifest['chain_source_sha256'] = {
                    name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                    for name in ('pipeline/locate.py', 'pipeline/stage_b.py', 'pipeline/core.py', 'pipeline/base_asr.py')
                }
            if expected:
                raw, secs = cached(RESULTS / f'gen-v{CACHE_VERSION}-{tag}', item['id'], produce,
                                   manifest=manifest, replay=replay)
            else:
                raw, secs = ('{}' if mode == 'perq' else '{"results":[]}'), 0.0
            try:
                if mode == 'locate':
                    frame = json.loads(raw) if expected else {'mode': 'locate', 'outputs': {}, 'windows': {}}
                    if not isinstance(frame, dict) or frame.get('mode') != 'locate':
                        raise ValueError('Expected a complete locate evidence frame')
                else:
                    frame = {'mode': 'perq', 'outputs': json.loads(raw)} if mode == 'perq' else {'mode': 'refine', 'raw': raw}
            except (ValueError, TypeError) as exc:
                raise CacheError(f'Invalid evidence generation for {item["id"]}') from exc
            complete = evidence_completeness(frame, baseline, item['words'], item['duration'])
            completeness.append({'conversation': item['id'], **complete})
            used_donors.update(donors)
            requests.append({'conversation': item['id'], 'fingerprint': fingerprint(manifest), 'timing_policy': policy,
                             'generated': bool(expected)})
            if mode == 'locate':
                requests[-1]['calls'] = frame.get('calls', [])
                requests[-1]['windows'] = frame.get('windows', {})
                requests[-1]['locations'] = frame.get('locations', {})
            gens.append(secs)
            response = apply_evidence(frame, baseline, item['words'], item['duration'], envelope_for(item),
                                      extend_replies=policy['extend_replies'])
            response = sanitize_response(response, len(questions), item['duration'])
            if response.answers != baseline.answers:
                raise ValueError(f'Stage B changed fixed answers for {item["id"]}')
            validate_response(response, len(questions))
            for i, row in enumerate(item['rows']):
                status = next((name for name in ('valid', 'missing', 'invalid') if i + 1 in complete[name]), 'not_requested')
                rows.append({'conversation': item['id'], 'question_id': row['question_id'], 'label': int(row['label']),
                             'answer': response.answers[i], 'gold': gold_evidence(row),
                             'baseline': evidence_interval(baseline.evidence_start[i], baseline.evidence_end[i]),
                             'candidate': evidence_interval(response.evidence_start[i], response.evidence_end[i]),
                             'raw': raw_answer, 'evidence_status': status,
                             'fold': variants.folds.get(item['id']) if variants else None,
                             'donors': donors.get(row['question_id'], [])})
        print(f'{item["id"]}: {secs:.1f}s; evidence {len(complete["valid"])}/{len(expected)} valid', flush=True)
    complete_report = {'expected_yes': sum(len(c['expected']) for c in completeness),
                       'valid': sum(len(c['valid']) for c in completeness),
                       'missing': [r['question_id'] for r in rows if r['evidence_status'] == 'missing'],
                       'invalid': [r['question_id'] for r in rows if r['evidence_status'] == 'invalid'],
                       'complete': all(c['complete'] for c in completeness), 'batches': completeness}
    summary = {'mode': f'stageb-{mode}', 'variant': variant, 'source': str(source) if source else None,
               'drafts': metrics(rows, 'baseline'), 'candidate': metrics(rows), 'report': report(rows),
               'drafts_report': report(rows, 'baseline'), 'evidence_completeness': complete_report,
               'tiou_gain_ci95': bootstrap_gain(rows),
               'generation_mean': sum(gens) / len(gens) if gens else 0.0, 'generation_max': max(gens, default=0.0),
               'manifest': {'version': CACHE_VERSION, 'replay': replay, 'generation': settings, 'requests': requests,
                            'cache_directory': str(RESULTS / f'gen-v{CACHE_VERSION}-{tag}'),
                            'inputs_sha256': fingerprint(ITEMS),
                            'source_sha256': hashlib.sha256(Path(source).read_bytes()).hexdigest() if source else None,
                            'variants': variants.manifest if variants else None, 'used_donors': used_donors}}
    if mode == 'locate':
        summary['manifest']['folds'] = variants.folds if variants else {}
        summary['manifest']['variants'] = None
    return save_result(tag, summary, rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('mode', choices=['answers', 'stageb', 'all'])
    parser.add_argument('--source', help='saved ANSWER result with fixed answers and original raw draft quotes')
    parser.add_argument('--tag', default=os.environ.get('EVAL_TAG', 'model'))
    parser.add_argument('--limit', type=int)
    parser.add_argument('--evidence-mode', choices=['refine', 'perq', 'locate'], default='refine',
                        help="'refine': batched quotes; 'perq': one quote per yes; 'locate': sentences then word boundaries.")
    parser.add_argument('--variant', choices=['control', 'no-draft', 'retrieved'], default='control')
    parser.add_argument('--asr-mode', choices=['base', 'turbo'], default=os.environ.get('MEDICAL_ASR_MODE'),
                        help='Timing/sentence policy for legacy inputs without ASR metadata; never inferred from envelope')
    parser.add_argument('--model-id', default=os.environ.get('EVAL_MODEL_ID'),
                        help='Explicit full served model revision (not the API alias); required for cache provenance')
    parser.add_argument('--replay', action='store_true', help='Strict cache-only evaluation; never construct or call a backend')
    args = parser.parse_args()
    if not args.model_id:
        parser.error('--model-id is required; served aliases do not identify weights')
    if args.limit is not None and args.limit <= 0:
        parser.error('--limit must be positive')
    if args.variant != 'control' and (args.mode == 'answers' or args.evidence_mode != 'perq'):
        parser.error('--variant no-draft/retrieved requires stageb/all with --evidence-mode perq')
    backend = None if args.replay else make_backend()
    started = time.monotonic()
    source = args.source
    options = {'replay': args.replay, 'model_id': args.model_id, 'asr_mode': args.asr_mode}
    if args.mode in ('answers', 'all'):
        s = run_answers(backend, f'answers-{args.tag}', args.limit, **options)
        source = source or s['result_path']
        print('ANSWERS:', json.dumps(s['candidate']), f"gen {s['generation_mean']:.1f}s mean / {s['generation_max']:.1f}s max")
    if args.mode in ('stageb', 'all'):
        if not source or (args.mode == 'all' and all(item.get('primary', {}).get('raw') for item in ITEMS[:args.limit])):
            s = run_stageb(backend, f'stageb-on-9b-{args.tag}', None, args.limit, args.evidence_mode,
                           variant=args.variant, **options)
            print('STAGE B on the 9B drafts:', json.dumps(s['candidate']), 'tIoU gain', s['tiou_gain_ci95'])
        if source:
            s = run_stageb(backend, f'stageb-on-own-{args.tag}', source, args.limit, args.evidence_mode,
                           variant=args.variant, **options)
            print('STAGE B on own answers:', json.dumps(s['candidate']), 'tIoU gain', s['tiou_gain_ci95'])
            print('  without example-source conversations:', json.dumps(s['report']['without_example_sources']))
        print('Evidence completeness:', json.dumps(s['evidence_completeness']))
    print(f'Wall {time.monotonic() - started:.0f}s; result {s["result_path"]}')


if __name__ == '__main__':
    main()
